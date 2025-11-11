import streamlit as st
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer
import pandas as pd
import json

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Mini-Gotham")

# --- Caching & Data Loading ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_graph_data(_driver):
    def work(tx):
        nodes_result = tx.run("MATCH (n) RETURN n")
        rels_result = tx.run("MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end")
        nodes, edges = [], []
        node_ids = set()
        for record in nodes_result:
            node = record["n"]
            properties = dict(node)
            node_id = properties.get("id")
            if node_id not in node_ids:
                properties["label_text"] = properties.get("name", "Unknown")
                properties["node_label"] = list(node.labels)[0]
                nodes.append(properties)
                node_ids.add(node_id)
        for r in rels_result:
            edges.append({"from": r["start"].get("id"), "to": r["end"].get("id"), "label": type(r["r"]).__name__})
        return nodes, edges
    with _driver.session() as session:
        return session.execute_read(work)

@st.cache_data
def get_filter_options():
    try:
        with open('dossiers.json', 'r') as f:
            dossiers = json.load(f)
        nationalities = sorted(list(set(d['Nationality'] for d in dossiers)))
        statuses = sorted(list(set(d['Status'] for d in dossiers)))
        return nationalities, statuses
    except FileNotFoundError:
        return [], []

# --- Connections & Models ---
embedding_model = load_embedding_model()
es = Elasticsearch('http://localhost:9200', headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"})
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neoj', 'neo4jpassword'))
INDEX_NAME = "intelligence_dossiers"

# --- Sidebar Filters ---
st.sidebar.title("Search Filters")
nationalities, statuses = get_filter_options()
selected_nationalities = st.sidebar.multiselect("Nationality", nationalities)
selected_statuses = st.sidebar.multiselect("Status", statuses)

# --- Main UI ---
st.title("Mini-Gotham: Intelligence Dashboard")
st.markdown("Explore the network of connections and search the dossier database.")

# --- Search Section ---
def build_es_query(query_text, filters):
    query = {"bool": {"must": [], "filter": []}}
    if query_text:
        query["bool"]["must"].append({"multi_match": {"query": query_text, "fields": ["name", "notes"]}})
    else:
        query["bool"]["must"].append({"match_all": {}})

    for nat in filters.get("nationalities", []):
        query["bool"]["filter"].append({"term": {"Nationality.keyword": nat}})
    for stat in filters.get("statuses", []):
        query["bool"]["filter"].append({"term": {"Status.keyword": stat}})
    return query

def display_search_results(results):
    for res in results:
        source = res['_source']
        # Clean up the source for display
        if 'embedding' in source:
            del source['embedding']

        with st.expander(f"**{source.get('name', 'N/A')}** (Score: {res['_score']:.2f})"):
            st.json(source, expanded=False)

col1, col2 = st.columns([1, 1])
active_filters = {"nationalities": selected_nationalities, "statuses": selected_statuses}

with col1:
    st.header("Keyword Search")
    keyword_query = st.text_input("Search notes and metadata...", key="keyword_search")
    es_query = build_es_query(keyword_query, active_filters)
    try:
        response = es.search(index=INDEX_NAME, query=es_query)
        st.write(f"{response['hits']['total']['value']} hits found.")
        display_search_results(response['hits']['hits'])
    except Exception as e: st.error(f"Keyword search error: {e}")

with col2:
    st.header("Semantic Search")
    semantic_query = st.text_input("Find conceptually similar notes...", key="semantic_search")
    if semantic_query:
        try:
            query_embedding = embedding_model.encode(semantic_query).tolist()
            response = es.search(index=INDEX_NAME, knn={"field": "embedding", "query_vector": query_embedding, "k": 5, "num_candidates": 10}, query=build_es_query("", active_filters))
            st.write(f"{len(response['hits']['hits'])} hits found.")
            display_search_results(response['hits']['hits'])
        except Exception as e: st.error(f"Semantic search error: {e}")

# --- Graph Section ---
st.header("Professional Knowledge Graph")
try:
    nodes, edges = get_graph_data(neo_driver)
    if nodes:
        net = Network(height="1000px", width="100%", bgcolor="#0E1117", font_color="#D3D3D3", notebook=True, directed=True)
        styles = {
            "Person": {"shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf007", "size": 50, "color": "#B0C4DE"}},
            "ORG": {"shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf1ad", "size": 50, "color": "#FFD700"}},
            "GPE": {"shape": "icon", "icon": {"face": "'Font Awesome 5 Free'", "code": "\uf57d", "size": 40, "color": "#98FB98"}},
            "default": {"shape": "dot", "size": 20, "color": "#808080"}
        }
        for node in nodes:
            style = styles.get(node['node_label'], styles["default"])
            del node['embedding'] # Ensure embedding is not in tooltip
            tooltip = pd.DataFrame([node]).to_html(index=False).replace('"', "'")
            net.add_node(str(node["id"]), label=node["label_text"], title=tooltip, **style)
        for edge in edges:
            net.add_edge(str(edge["from"]), str(edge["to"]), label=edge["label"], color={"inherit": "to", "opacity": 0.5})
        net.set_options("""{"nodes": {"font": {"size": 14, "strokeWidth": 2, "strokeColor": "#000000"}}, "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}, "font": {"size": 10, "align": "top"}, "smooth": {"type": "continuous"}},"physics": {"forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.01, "springLength": 100, "springConstant": 0.08, "avoidOverlap": 0.5}, "minVelocity": 0.75, "solver": "forceAtlas2Based"}}""")
        components.html(net.generate_html(notebook=False), height=1020)
    else:
        st.warning("No data in graph. Run ingest.py to populate.")
except Exception as e:
    st.error(f"Could not visualize graph: {e}")
    st.warning("Ensure Neo4j container is running.")
