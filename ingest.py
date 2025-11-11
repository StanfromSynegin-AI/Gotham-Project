import pandas as pd
from elasticsearch import Elasticsearch
import spacy
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz
import logging
import re
from sentence_transformers import SentenceTransformer
from itertools import permutations

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

# Connections
es = Elasticsearch('http://localhost:9200', headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"})
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))

def create_es_index():
    index_name = 'docs'
    mapping = {
        "properties": {
            "text": {"type": "text"}, "meta": {"type": "object", "enabled": False},
            "embedding": {"type": "dense_vector", "dims": 384}
        }
    }
    try:
        es.indices.create(index=index_name, mappings=mapping, ignore=[400])
        logging.info(f"Created or verified Elasticsearch index '{index_name}'.")
    except Exception as e:
        logging.error(f"Could not manage Elasticsearch index: {e}")

def index_doc(doc_id, text, metadata):
    try:
        embedding = embedding_model.encode(text).tolist()
        document = {'text': text, 'meta': metadata, 'embedding': embedding}
        es.index(index='docs', id=doc_id, document=document)
    except Exception as e:
        logging.error(f"Error indexing document {doc_id}: {e}")

def extract_entities(text):
    return [(ent.text, ent.label_) for ent in nlp(text).ents]

def get_existing_entities(tx, label):
    result = tx.run(f"MATCH (e:{label}) RETURN e.id AS id, e.name AS name")
    return {record["name"]: record["id"] for record in result}

def resolve_and_upsert_entity(tx, label, name, existing_entities):
    best_match = process.extractOne(name, existing_entities.keys(), scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] > 90:
        ent_id = existing_entities[best_match[0]]
        return ent_id, False
    else:
        ent_id = str(hash(name.lower()))
        tx.run(f"MERGE (e:{label} {{id: $id}}) SET e.name = $name", id=ent_id, name=name)
        return ent_id, True

def create_relationship(tx, source_label, source_id, target_label, target_id, rel_type):
    query = f"MATCH (a:{source_label} {{id: $source_id}}), (b:{target_label} {{id: $target_id}}) MERGE (a)-[:{rel_type}]->(b)"
    tx.run(query, source_id=source_id, target_id=target_id)

def extract_relationships(tx, person_id, text, entities_map):
    # Person-to-Entity and Entity-to-Entity relationship extraction
    all_entities = list(entities_map.items())

    # Person to Entity
    for ent_id, (ent_name, ent_label) in all_entities:
        if re.search(f"works for|works at|consultant' through his firm|CEO of|director at|analyst for|manager for|{re.escape(ent_name)}", text, re.IGNORECASE):
            if ent_label == 'ORG': create_relationship(tx, 'Person', person_id, 'ORG', ent_id, 'WORKS_FOR')
        if re.search(f"based in|lives in|from {re.escape(ent_name)}", text, re.IGNORECASE):
            if ent_label == 'GPE': create_relationship(tx, 'Person', person_id, 'GPE', ent_id, 'LIVES_AT')
        if re.search(f"associate of|colleague of|contact with {re.escape(ent_name)}|meeting with {re.escape(ent_name)}", text, re.IGNORECASE):
             create_relationship(tx, 'Person', person_id, 'Person', ent_id, 'CONTACT_WITH')
        if re.search(f"sister of|brother of", text, re.IGNORECASE):
            create_relationship(tx, 'Person', person_id, 'Person', ent_id, 'FAMILY_OF')

    # Entity to Entity
    for (id1, (name1, label1)), (id2, (name2, label2)) in permutations(all_entities, 2):
        if re.search(f"{re.escape(name1)}.*funded.*{re.escape(name2)}", text, re.IGNORECASE):
            create_relationship(tx, label1, id1, label2, id2, 'FUNDED')
        if re.search(f"{re.escape(name1)}.*partnership with.*{re.escape(name2)}", text, re.IGNORECASE):
            create_relationship(tx, label1, id1, label2, id2, 'PARTNERSHIP_WITH')
        if re.search(f"{re.escape(name1)}.*financial ties to.*{re.escape(name2)}", text, re.IGNORECASE):
            create_relationship(tx, label1, id1, label2, id2, 'FINANCIAL_TIE_TO')

if __name__ == '__main__':
    create_es_index()
    df = pd.read_csv('people.csv').fillna('')

    with neo_driver.session() as session:
        existing_orgs = session.execute_read(get_existing_entities, 'ORG')
        existing_gpes = session.execute_read(get_existing_entities, 'GPE')
        existing_persons = session.execute_read(get_existing_entities, 'Person')

        for _, row in df.iterrows():
            doc_id = str(row['id'])
            notes = row.get('notes', '')

            # Index full dossier for search
            index_doc(doc_id, notes, row.to_dict())

            # Upsert Person node with all dossier properties
            person_props = row.to_dict()
            session.execute_write(lambda tx: tx.run("MERGE (p:Person {id: $id}) SET p += $props", id=doc_id, props=person_props))

            entities_map = {}
            for ent_text, ent_label in extract_entities(notes):
                ent_id, newly_created = (None, False)
                if ent_label == 'ORG':
                    ent_id, newly_created = session.execute_write(resolve_and_upsert_entity, 'ORG', ent_text, existing_orgs)
                    if newly_created: existing_orgs[ent_text] = ent_id
                elif ent_label == 'GPE':
                    ent_id, newly_created = session.execute_write(resolve_and_upsert_entity, 'GPE', ent_text, existing_gpes)
                    if newly_created: existing_gpes[ent_text] = ent_id
                elif ent_label == 'PERSON' and ent_text != row['name']:
                     ent_id, newly_created = session.execute_write(resolve_and_upsert_entity, 'Person', ent_text, existing_persons)
                     if newly_created: existing_persons[ent_text] = ent_id
                else: # Other entity types
                    ent_id = str(hash(ent_text.lower()))
                    session.execute_write(lambda tx: tx.run(f"MERGE (e:{ent_label} {{id: $id}}) SET e.name = $name", id=ent_id, name=ent_text))

                if ent_id:
                    entities_map[ent_id] = (ent_text, ent_label)
                    session.execute_write(create_relationship, 'Person', doc_id, ent_label, ent_id, 'MENTIONED')

            session.execute_write(extract_relationships, doc_id, notes, entities_map)

    logging.info("Data ingestion complete.")
