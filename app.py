import streamlit as st
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer

# --- Page Config ---
st.set_page_config(layout="wide")

# --- Caching ---
@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model only once."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_graph_data(_driver):
    """Fetches all nodes and relationships from Neo4j."""
    def work(tx):
        nodes_result = tx.run("MATCH (n) RETURN n")
        rels_result = tx.run("MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end")

        nodes, node_ids = [], set()
        for record in nodes_result:
            node = record["n"]
            node_id = node.get("id")
            if node_id not in node_ids:
                nodes.append({
                    "id": node_id,
                    "label": list(node.labels)[0],
                    "title": node.get("name"),
                })
                node_ids.add(node_id)

        edges = []
        for record in rels_result:
            edges.append({
                "from": record["start"].get("id"),
                "to": record["end"].get("id"),
                "label": type(record["r"]).__name__,
            })
        return nodes, edges

    with _driver.session() as session:
        return session.execute_read(work)

# --- Connections & Models ---
embedding_model = load_embedding_model()
es = Elasticsearch('http://localhost:9200')
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))

# --- UI ---
st.title("Mini-Gotham 🕵️")
st.markdown("A tool for exploring connections in your data.")

col1, col2 = st.columns(2)
with col1:
    st.header("Keyword Search")
    keyword_query = st.text_input("Enter keyword query", key="keyword_search")
    if keyword_query:
        try:
            response = es.search(index='docs', query={"multi_match": {"query": keyword_query, "fields": ["text", "meta.*"]}})
            for res in response['hits']['hits']:
                with st.expander(f"**{res['_source']['meta']['name']}** (Score: {res['_score']:.2f})"):
                    st.write(res['_source']['text']); st.json(res['_source']['meta'])
        except Exception as e: st.error(f"Keyword search error: {e}")

with col2:
    st.header("Semantic Search")
    semantic_query = st.text_input("Enter semantic query", key="semantic_search")
    if semantic_query:
        try:
            query_embedding = embedding_model.encode(semantic_query).tolist()
            response = es.search(index='docs', knn={"field": "embedding", "query_vector": query_embedding, "k": 5, "num_candidates": 10})
            for res in response['hits']['hits']:
                with st.expander(f"**{res['_source']['meta']['name']}** (Similarity: {res['_score']:.2f})"):
                    st.write(res['_source']['text']); st.json(res['_source']['meta'])
        except Exception as e: st.error(f"Semantic search error: {e}")

st.header("Knowledge Graph Explorer")
try:
    nodes, edges = get_graph_data(neo_driver)
    if nodes:
        net = Network(height="800px", width="100%", bgcolor="#0E1117", font_color="white", notebook=True, directed=True)

        # Node styling
        node_styles = {"Person": {"color": "#FF69B4", "size": 25}, "ORG": {"color": "#87CEEB", "size": 20}, "GPE": {"color": "#90EE90", "size": 15}}
        default_style = {"color": "#C0C0C0", "size": 10}

        for node in nodes:
            style = node_styles.get(node['label'], default_style)
            net.add_node(node["id"], label=node["title"], title=f"<b>{node['label']}:</b> {node['title']}", color=style['color'], size=style['size'])

        for edge in edges:
            net.add_edge(edge["from"], edge["to"], label=edge["label"])

        # Improved physics and layout
        net.set_options("""
        var options = {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "centralGravity": 0.3,
              "springLength": 95,
              "springConstant": 0.04
            },
            "stabilization": {
              "iterations": 2000
            }
          }
        }
        """)

        net_html = net.generate_html(notebook=False)
        components.html(net_html, height=820)
    else:
        st.warning("No data in the graph. Run the ingestion script (`ingest.py`) to populate it.")
except Exception as e:
    st.error(f"Could not connect to Neo4j or visualize graph: {e}")
    st.warning("Please ensure the Neo4j container is running.")
