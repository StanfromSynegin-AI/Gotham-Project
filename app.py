import streamlit as st
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_model()

# --- Connections ---
es = Elasticsearch('http://localhost:9200')
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))

# --- UI Setup ---
st.set_page_config(layout="wide")
st.title("Mini-Gotham 🕵️")
st.markdown("A tool for exploring connections in your data.")

# --- Search Section ---
col1, col2 = st.columns(2)

with col1:
    st.header("Keyword Search")
    keyword_query = st.text_input("Enter keyword query", key="keyword_search")
    if keyword_query:
        try:
            response = es.search(
                index='docs',
                query={"multi_match": {"query": keyword_query, "fields": ["text", "meta.*"]}}
            )
            results = response['hits']['hits']
            st.write(f"Found {len(results)} results:")
            for res in results:
                with st.expander(f"**{res['_source']['meta']['name']}** (Score: {res['_score']:.2f})"):
                    st.write(res['_source']['text'])
                    st.json(res['_source']['meta'])
        except Exception as e:
            st.error(f"Error with keyword search: {e}")

with col2:
    st.header("Semantic Search")
    semantic_query = st.text_input("Enter semantic query", key="semantic_search")
    if semantic_query:
        try:
            query_embedding = embedding_model.encode(semantic_query).tolist()
            response = es.search(
                index='docs',
                knn={
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": 5,
                    "num_candidates": 10
                }
            )
            results = response['hits']['hits']
            st.write(f"Found {len(results)} similar results:")
            for res in results:
                with st.expander(f"**{res['_source']['meta']['name']}** (Similarity: {res['_score']:.2f})"):
                    st.write(res['_source']['text'])
                    st.json(res['_source']['meta'])
        except Exception as e:
            st.error(f"Error with semantic search: {e}")

# --- Graph Visualization Section ---
st.header("Knowledge Graph Explorer")

@st.cache_data
def get_graph_data():
    """Fetches all nodes and relationships from Neo4j."""
    with neo_driver.session() as session:
        nodes_result = session.run("MATCH (n) RETURN n")
        rels_result = session.run("MATCH ()-[r]->() RETURN r, startNode(r) as start, endNode(r) as end")

        nodes = []
        node_ids = set()
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

try:
    nodes, edges = get_graph_data()
    if nodes:
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=True)
        for node in nodes:
            net.add_node(node["id"], label=node["title"], title=f"{node['label']}: {node['title']}", group=node['label'])
        for edge in edges:
            net.add_edge(edge["from"], edge["to"], label=edge["label"])

        net.show_buttons(filter_=['physics'])
        net_html = net.generate_html(notebook=False)
        components.html(net_html, height=820)
    else:
        st.warning("No data in the graph. Run the ingestion script (`ingest.py`) to populate it.")
except Exception as e:
    st.error(f"Could not connect to Neo4j or visualize graph: {e}")
    st.warning("Please ensure the Neo4j container is running.")
