import pandas as pd
from elasticsearch import Elasticsearch, helpers
import spacy
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz
import logging
import re
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load models
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logging.info("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Elasticsearch and Neo4j connections
es = Elasticsearch('http://localhost:9200')
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))

def create_es_index():
    """Creates the Elasticsearch index with a specific mapping for vector search."""
    index_name = 'docs'
    mapping = {
        "properties": {
            "text": {"type": "text"},
            "meta": {"type": "object"},
            "embedding": {"type": "dense_vector", "dims": 384}
        }
    }
    try:
        es.indices.create(index=index_name, mappings=mapping, ignore=[400])
        logging.info(f"Created or verified Elasticsearch index '{index_name}'.")
    except Exception as e:
        logging.error(f"Could not create or verify Elasticsearch index: {e}")

def index_doc(doc_id, text, metadata):
    """Generates embedding and indexes a document into Elasticsearch."""
    try:
        embedding = embedding_model.encode(text).tolist()
        document = {
            'text': text,
            'meta': metadata,
            'embedding': embedding
        }
        es.index(index='docs', id=doc_id, document=document)
        logging.info(f"Indexed document {doc_id} into Elasticsearch with embedding.")
    except Exception as e:
        logging.error(f"Error indexing document {doc_id}: {e}")

def extract_entities(text):
    """Extracts named entities from text using spaCy."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def get_existing_entities(tx, label):
    """Fetches existing entities of a certain label from Neo4j."""
    result = tx.run(f"MATCH (e:{label}) RETURN e.id AS id, e.name AS name")
    return {record["name"]: record["id"] for record in result}

def resolve_and_upsert_entity(tx, label, name, existing_entities):
    """Resolves an entity against existing ones and creates or merges it."""
    best_match = process.extractOne(name, existing_entities.keys(), scorer=fuzz.token_set_ratio)

    if best_match and best_match[1] > 90:
        ent_id = existing_entities[best_match[0]]
        logging.info(f"Matched '{name}' with existing '{best_match[0]}'.")
        tx.run(f"MATCH (e:{label} {{id: $id}}) SET e.name = $name", id=ent_id, name=best_match[0])
        return ent_id, False
    else:
        ent_id = str(hash(name.lower()))
        tx.run(f"MERGE (e:{label} {{id: $id}}) SET e.name = $name", id=ent_id, name=name)
        logging.info(f"Created new {label} entity '{name}'.")
        return ent_id, True

def create_relationship(tx, source_label, source_id, target_label, target_id, rel_type):
    """Creates a relationship between two nodes in Neo4j."""
    query = f"MATCH (a:{source_label} {{id: $source_id}}), (b:{target_label} {{id: $target_id}}) MERGE (a)-[:{rel_type}]->(b)"
    tx.run(query, source_id=source_id, target_id=target_id)

def extract_and_create_specific_relationships(tx, person_id, text, entities_map):
    """Creates specific relationships based on keywords."""
    for ent_id, (ent_name, ent_label) in entities_map.items():
        if ent_label == 'ORG' and re.search(f'works at {re.escape(ent_name)}|engineer at {re.escape(ent_name)}', text, re.IGNORECASE):
            create_relationship(tx, 'Person', person_id, 'ORG', ent_id, 'WORKS_FOR')
        elif ent_label == 'GPE' and re.search(f'lives in {re.escape(ent_name)}', text, re.IGNORECASE):
            create_relationship(tx, 'Person', person_id, 'GPE', ent_id, 'LIVES_AT')

if __name__ == '__main__':
    create_es_index()

    try:
        df = pd.read_csv('people.csv')
    except FileNotFoundError:
        logging.error("people.csv not found.")
        exit()

    with neo_driver.session() as session:
        existing_orgs = session.execute_read(get_existing_entities, 'ORG')
        existing_gpes = session.execute_read(get_existing_entities, 'GPE')

        for _, row in df.iterrows():
            text = row.get('notes', '')
            doc_id = str(row['id'])

            index_doc(doc_id, text, row.to_dict())
            ents = extract_entities(text)

            session.execute_write(lambda tx: tx.run("MERGE (p:Person {id: $id}) SET p.name = $name", id=doc_id, name=row['name']))

            entities_map = {}
            for ent_text, ent_label in ents:
                ent_id, newly_created = (None, False)
                if ent_label == 'ORG':
                    ent_id, newly_created = session.execute_write(resolve_and_upsert_entity, 'ORG', ent_text, existing_orgs)
                    if newly_created: existing_orgs[ent_text] = ent_id
                elif ent_label == 'GPE':
                    ent_id, newly_created = session.execute_write(resolve_and_upsert_entity, 'GPE', ent_text, existing_gpes)
                    if newly_created: existing_gpes[ent_text] = ent_id
                else:
                    ent_id = str(hash(ent_text.lower()))
                    session.execute_write(lambda tx: tx.run(f"MERGE (e:{ent_label} {{id: $id}}) SET e.name = $name", id=ent_id, name=ent_text))

                if ent_id:
                    entities_map[ent_id] = (ent_text, ent_label)
                    session.execute_write(create_relationship, 'Person', doc_id, ent_label, ent_id, 'MENTIONED')

            session.execute_write(extract_and_create_specific_relationships, doc_id, text, entities_map)

    logging.info("Data ingestion complete.")
