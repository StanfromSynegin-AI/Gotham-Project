import json
import logging
import re
import time
from itertools import permutations
import spacy
from elasticsearch import Elasticsearch
from neo4j import GraphDatabase, exceptions
from sentence_transformers import SentenceTransformer

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Models ---
nlp = spacy.load('en_core_web_sm')
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Connections ---
es = Elasticsearch(
    'http://localhost:9200',
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"},
    request_timeout=30
)
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))

# --- Bulletproof Wait Function ---
def wait_for_services():
    logging.info("Waiting for services to become available...")
    # Wait for Elasticsearch
    for _ in range(30):
        try:
            if es.ping():
                logging.info("Elasticsearch is ready.")
                break
        except Exception:
            time.sleep(2)
    else:
        raise ConnectionError("Could not connect to Elasticsearch after 60 seconds.")

    # Wait for Neo4j
    for _ in range(30):
        try:
            with neo_driver.session() as session:
                session.run("RETURN 1")
            logging.info("Neo4j is ready.")
            break
        except exceptions.ServiceUnavailable:
            time.sleep(2)
    else:
        raise ConnectionError("Could not connect to Neo4j after 60 seconds.")

# --- Elasticsearch Functions ---
def setup_es_indices():
    es.indices.delete(index='_all', ignore_unavailable=True)
    dossier_index = "dossiers"
    dossier_mappings = {"properties": {"embedding": {"type": "dense_vector", "dims": 384}}}
    es.indices.create(index=dossier_index, mappings=dossier_mappings)

    event_index = "events"
    event_mappings = {"properties": {"embedding": {"type": "dense_vector", "dims": 384}}}
    es.indices.create(index=event_index, mappings=event_mappings)

def index_item(index, item, text_field):
    embedding = embedding_model.encode(item[text_field]).tolist()
    document = {**item, "embedding": embedding}
    es.index(index=index, id=item.get('id') or item.get('event_id'), document=document)

# --- Neo4j Functions ---
def setup_neo4j(session):
    session.run("MATCH (n) DETACH DELETE n")
    for label in ["Person", "Event", "ORG", "GPE"]:
        session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")

def ingest_dossiers(session, dossiers):
    for dossier in dossiers:
        index_item("dossiers", dossier, 'notes')
        props = {k: v for k, v in dossier.items() if v}
        session.run("MERGE (p:Person {id: $id}) SET p += $props", id=str(dossier['id']), props=props)

def ingest_events(session, events):
    for event in events:
        index_item("events", event, 'summary')
        props = {k: v for k, v in event.items() if k != 'participants'}
        session.run("MERGE (e:Event {id: $id}) SET e += $props", id=event['event_id'], props=props)
        for person_id in event['participants']:
            session.run(
                "MATCH (p:Person {id: $p_id}), (e:Event {id: $e_id}) "
                "MERGE (p)-[r:PARTICIPATED_IN]->(e) SET r.date = $date",
                p_id=str(person_id), e_id=event['event_id'], date=event["date"]
            )

# --- Main Execution ---
if __name__ == '__main__':
    wait_for_services()
    setup_es_indices()

    with open('dossiers.json', 'r') as f: dossiers = json.load(f)
    with open('events.json', 'r') as f: events = json.load(f)

    with neo_driver.session() as session:
        setup_neo4j(session)
        ingest_dossiers(session, dossiers)
        ingest_events(session, events)

    logging.info("Data ingestion complete. The system is ready.")
