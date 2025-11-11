import json
import logging
import re
from itertools import permutations
import spacy
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Models ---
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        logging.info("Downloading spaCy model 'en_core_web_sm'...")
        from spacy.cli import download
        download('en_core_web_sm')
        return spacy.load('en_core_web_sm')

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

nlp = load_spacy_model()
embedding_model = load_embedding_model()

# --- Connections ---
es = Elasticsearch('http://localhost:9200')
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))

# --- Elasticsearch Functions ---
def setup_es_indices():
    # Clean slate
    es.indices.delete(index='_all', ignore_unavailable=True)
    logging.info("Cleared old Elasticsearch indices.")

    # Dossier Index
    dossier_index = "dossiers"
    dossier_mappings = {
        "properties": {
            "name": {"type": "text"}, "notes": {"type": "text"},
            "Nationality": {"type": "keyword"}, "Status": {"type": "keyword"},
            "DOB": {"type": "date"}, "embedding": {"type": "dense_vector", "dims": 384}
        }
    }
    es.indices.create(index=dossier_index, mappings=dossier_mappings)
    logging.info(f"Created '{dossier_index}' index.")

    # Event Index
    event_index = "events"
    event_mappings = {
        "properties": {
            "summary": {"type": "text"}, "location": {"type": "keyword"},
            "date": {"type": "date"}, "embedding": {"type": "dense_vector", "dims": 384}
        }
    }
    es.indices.create(index=event_index, mappings=event_mappings)
    logging.info(f"Created '{event_index}' index.")

def index_item(index, item, text_field):
    embedding = embedding_model.encode(item[text_field]).tolist()
    document = {**item, "embedding": embedding}
    es.index(index=index, id=item.get('id') or item.get('event_id'), document=document)

# --- Neo4j Functions ---
def get_existing_entities(tx, label):
    result = tx.run(f"MATCH (e:{label}) RETURN e.id AS id, e.name AS name")
    return {record["name"]: record["id"] for record in result}

def resolve_and_upsert_entity(tx, label, name, existing_entities):
    # Simplified resolution
    ent_id = str(hash(name.lower()))
    tx.run(f"MERGE (e:{label} {{id: $id}}) SET e.name = $name", id=ent_id, name=name)
    return ent_id

def create_relationship(tx, source_label, source_id, target_label, target_id, rel_type, properties=None):
    props_str = f" {{ {', '.join([f'{k}: ${k}' for k in properties])} }}" if properties else ""
    query = (f"MATCH (a:{source_label} {{id: $source_id}}), (b:{target_label} {{id: $target_id}}) "
             f"MERGE (a)-[r:{rel_type}{props_str}]->(b)")
    tx.run(query, source_id=str(source_id), target_id=str(target_id), **(properties or {}))

# --- Main Execution ---
if __name__ == '__main__':
    setup_es_indices()

    with open('dossiers.json', 'r') as f: dossiers = json.load(f)
    with open('events.json', 'r') as f: events = json.load(f)

    with neo_driver.session() as session:
        # Clear old graph data
        session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
        logging.info("Cleared old Neo4j graph data.")

        # Ingest Dossiers (Entities)
        for dossier in dossiers:
            index_item("dossiers", dossier, 'notes')
            dossier_props = {k: v for k, v in dossier.items() if v}
            session.execute_write(lambda tx: tx.run("MERGE (p:Person {id: $id}) SET p += $props", id=str(dossier['id']), props=dossier_props))
        logging.info("Ingested all dossiers.")

        # Ingest Events and Relationships
        for event in events:
            index_item("events", event, 'summary')
            event_id = event['event_id']
            event_props = {k: v for k, v in event.items() if k != 'participants'}
            session.execute_write(lambda tx: tx.run("MERGE (e:Event {id: $id}) SET e += $props", id=event_id, props=event_props))

            # Link participants to the event
            for person_id in event['participants']:
                create_relationship(session, 'Person', str(person_id), 'Event', event_id, 'PARTICIPATED_IN', properties={"date": event["date"]})
        logging.info("Ingested all events and linked participants.")

    logging.info("Data ingestion complete.")
