import streamlit as st
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
import pandas as pd
import json

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Mini-Gotham Intelligence Platform")

# --- Caching & Data Loading ---
@st.cache_resource
def load_embedding_model(): return SentenceTransformer('all-MiniLM-L6-v2')
@st.cache_data
def load_data():
    try:
        with open('dossiers.json', 'r') as f: dossiers = json.load(f)
        with open('events.json', 'r') as f: events = json.load(f)
        return dossiers, events
    except FileNotFoundError: return [], []
@st.cache_data
def get_graph_data(_driver):
    def work(tx):
        result = tx.run("MATCH (n) RETURN n")
        nodes = []
        for record in result:
            node = record["n"]
            properties = dict(node)
            properties["label"] = list(node.labels)[0]
            nodes.append(properties)

        result = tx.run("MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end")
        edges = [{"from": r["start"].get("id"), "to": r["end"].get("id"), "label": type(r["r"]).__name__} for r in result]
        return nodes, edges
    with _driver.session() as session: return session.execute_read(work)

# --- Connections & Models ---
embedding_model = load_embedding_model()
es = Elasticsearch('http://localhost:9200')
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))
dossiers, events = load_data()

# --- Main App ---
st.title("Intelligence Analysis Platform")

tab1, tab2, tab3 = st.tabs(["Dossier Search", "Timeline Analysis", "Graph Explorer"])

# --- Tab 1: Dossier Search ---
with tab1:
    st.header("Search and Filter Dossiers")

    # Sidebar Filters
    nationalities = sorted(list(set(d['Nationality'] for d in dossiers)))
    statuses = sorted(list(set(d['Status'] for d in dossiers)))
    selected_nationalities = st.sidebar.multiselect("Filter by Nationality", nationalities)
    selected_statuses = st.sidebar.multiselect("Filter by Status", statuses)

    def build_dossier_query(text, filters):
        query = {"bool": {"must": [{"match_all": {}}], "filter": []}}
        if text: query["bool"]["must"] = [{"multi_match": {"query": text, "fields": ["name", "notes"]}}]
        for n in filters.get("nationalities", []): query["bool"]["filter"].append({"term": {"Nationality": n}})
        for s in filters.get("statuses", []): query["bool"]["filter"].append({"term": {"Status": s}})
        return query

    col1, col2 = st.columns(2)
    filters = {"nationalities": selected_nationalities, "statuses": selected_statuses}
    with col1:
        keyword_query = st.text_input("Keyword search in dossiers", key="dossier_keyword")
        es_query = build_dossier_query(keyword_query, filters)
        response = es.search(index="dossiers", query=es_query)
        st.write(f"{response['hits']['total']['value']} hits found.")
        for res in response['hits']['hits']:
            with st.expander(f"**{res['_source'].get('name')}** (Score: {res['_score']:.2f})"):
                st.json({k: v for k, v in res['_source'].items() if k != 'embedding'})

    with col2:
        semantic_query = st.text_input("Semantic search in dossiers", key="dossier_semantic")
        if semantic_query:
            embedding = embedding_model.encode(semantic_query).tolist()
            es_query = build_dossier_query("", filters)
            response = es.search(index="dossiers", knn={"field": "embedding", "query_vector": embedding, "k": 5, "num_candidates": 10}, query=es_query)
            st.write(f"{len(response['hits']['hits'])} hits found.")
            for res in response['hits']['hits']:
                 with st.expander(f"**{res['_source'].get('name')}** (Similarity: {res['_score']:.2f})"):
                    st.json({k: v for k, v in res['_source'].items() if k != 'embedding'})

# --- Tab 2: Timeline Analysis ---
with tab2:
    st.header("Chronological Event Analysis")
    if events:
        df = pd.DataFrame(events)
        df['date'] = pd.to_datetime(df['date'])
        df_sorted = df.sort_values(by='date', ascending=False)

        for _, row in df_sorted.iterrows():
            with st.expander(f"**{row['date'].strftime('%Y-%m-%d')}: {row['location']}**"):
                st.markdown(f"**Event ID:** {row['event_id']}")
                st.markdown(f"**Summary:** {row['summary']}")
                participant_names = [d['name'] for d in dossiers if d['id'] in row['participants']]
                st.markdown(f"**Participants:** {', '.join(participant_names)}")
    else:
        st.warning("No event data found. Run the ingestion script.")

# --- Tab 3: Graph Explorer ---
with tab3:
    st.header("Interactive Knowledge Graph")
    try:
        nodes, edges = get_graph_data(neo_driver)
        if nodes:
            net = Network(height="1000px", width="100%", bgcolor="#0E1117", font_color="white", notebook=True, directed=True)
            styles = {
                "Person": {"shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf007", "color": "#B0C4DE", "size": 50}},
                "ORG": {"shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf1ad", "color": "#FFD700", "size": 50}},
                "GPE": {"shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf57d", "color": "#98FB98", "size": 40}},
                "Event": {"shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf073", "color": "#FF6347", "size": 30}}
            }
            for node in nodes:
                style = styles.get(node['label'], {"shape": "dot", "size": 20})
                tooltip = pd.DataFrame([node]).to_html(index=False).replace('"', "'")
                net.add_node(str(node.get("id") or node.get("event_id")), label=node.get("name") or node.get("location"), title=tooltip, **style)
            for edge in edges:
                net.add_edge(str(edge["from"]), str(edge["to"]), label=edge["label"])
            net.set_options("""{"physics": {"forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08, "avoidOverlap": 0.5}, "minVelocity": 0.75, "solver": "forceAtlas2Based"}}""")
            components.html(net.generate_html(notebook=False), height=1020)
    except Exception as e:
        st.error(f"Could not visualize graph: {e}")
        st.warning("Ensure Neo4j container is running and data is ingested.")
