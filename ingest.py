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
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logging.info("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Connections ---
es = Elasticsearch('http://localhost:9200')
neo_driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'neo4jpassword'))

# --- Elasticsearch Functions ---
def create_advanced_es_index():
    index_name = "intelligence_dossiers"
    # Delete old indices for a clean slate using a more robust method
    es.indices.delete(index="docs", ignore_unavailable=True)
    es.indices.delete(index=index_name, ignore_unavailable=True)
    logging.info("Cleared any old indices for a fresh start.")

    settings = {
        "analysis": {
            "analyzer": {
                "custom_english_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "english_stop", "english_stemmer"]
                }
            },
            "filter": {
                "english_stop": {"type": "stop", "stopwords": "_english_"},
                "english_stemmer": {"type": "stemmer", "language": "english"}
            }
        }
    }
    mappings = {
        "properties": {
            "name": {"type": "text", "analyzer": "custom_english_analyzer", "fields": {"keyword": {"type": "keyword"}}},
            "notes": {"type": "text", "analyzer": "custom_english_analyzer"},
            "Nationality": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "Status": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "DOB": {"type": "date"},
            "embedding": {"type": "dense_vector", "dims": 384}
        }
    }

    es.indices.create(index=index_name, settings=settings, mappings=mappings)
    logging.info(f"Created advanced Elasticsearch index '{index_name}'.")

def index_dossier(dossier):
    try:
        embedding = embedding_model.encode(dossier['notes']).tolist()
        document = {**dossier, "embedding": embedding}
        es.index(index='intelligence_dossiers', id=dossier['id'], document=document)
    except Exception as e:
        logging.error(f"Error indexing dossier {dossier['id']}: {e}")

# --- Neo4j Functions ---
def get_existing_entities(tx, label):
    result = tx.run(f"MATCH (e:{label}) RETURN e.id AS id, e.name AS name")
    return {record["name"]: record["id"] for record in result}

def resolve_and_upsert_entity(tx, label, name, existing_entities):
    best_match = process.extractOne(name, list(existing_entities.keys()), scorer=fuzz.token_set_ratio)
    if best_match and best_match[1] > 90:
        return existing_entities[best_match[0]], False
    ent_id = str(hash(name.lower()))
    tx.run(f"MERGE (e:{label} {{id: $id}}) SET e.name = $name", id=ent_id, name=name)
    return ent_id, True

def create_relationship(tx, source_label, source_id, target_label, target_id, rel_type):
    query = f"MATCH (a:{source_label} {{id: $source_id}}), (b:{target_label} {{id: $target_id}}) MERGE (a)-[:{rel_type}]->(b)"
    tx.run(query, source_id=str(source_id), target_id=str(target_id))

def extract_relationships(tx, person_id, text, entities_map, all_people_ids):
    all_entities = list(entities_map.items())
    for ent_id, (ent_name, ent_label) in all_entities:
        # Check if the entity is another person in the dataset
        is_person_entity = any(p_id for p_id, p_name in all_people_ids.items() if fuzz.ratio(ent_name, p_name) > 90)

        if is_person_entity:
            target_id = next(p_id for p_id, p_name in all_people_ids.items() if fuzz.ratio(ent_name, p_name) > 90)
            if re.search(f"associate of|colleague of|contact with|meeting with {re.escape(ent_name)}", text, re.IGNORECASE):
                create_relationship(tx, 'Person', person_id, 'Person', target_id, 'CONTACT_WITH')
            if re.search(f"sister of|brother of", text, re.IGNORECASE):
                create_relationship(tx, 'Person', person_id, 'Person', target_id, 'FAMILY_OF')
        elif ent_label == 'ORG':
            if re.search(f"works for|works at|firm|CEO of|director at|analyst for|manager for|{re.escape(ent_name)}", text, re.IGNORECASE):
                create_relationship(tx, 'Person', person_id, 'ORG', ent_id, 'WORKS_FOR')
        elif ent_label == 'GPE':
            if re.search(f"based in|lives in|from {re.escape(ent_name)}", text, re.IGNORECASE):
                create_relationship(tx, 'Person', person_id, 'GPE', ent_id, 'LIVES_AT')

    for (id1, (name1, label1)), (id2, (name2, label2)) in permutations(all_entities, 2):
        if re.search(f"{re.escape(name1)}.*funded.*{re.escape(name2)}", text, re.IGNORECASE):
            create_relationship(tx, label1, id1, label2, id2, 'FUNDED')
        if re.search(f"{re.escape(name1)}.*partnership with.*{re.escape(name2)}", text, re.IGNORECASE):
            create_relationship(tx, label1, id1, label2, id2, 'PARTNERSHIP_WITH')
        if re.search(f"{re.escape(name1)}.*financial ties to.*{re.escape(name2)}", text, re.IGNORECASE):
            create_relationship(tx, label1, id1, label2, id2, 'FINANCIAL_TIE_TO')

# --- Main Execution ---
if __name__ == '__main__':
    create_advanced_es_index()

    with open('dossiers.json', 'r') as f:
        dossiers = json.load(f)

    all_people_ids = {str(d['id']): d['name'] for d in dossiers}

    with neo_driver.session() as session:
        existing_orgs = session.execute_read(get_existing_entities, 'ORG')
        existing_gpes = session.execute_read(get_existing_entities, 'GPE')

        for dossier in dossiers:
            doc_id = str(dossier['id'])
            notes = dossier.get('notes', '')

            index_dossier(dossier)
            session.execute_write(lambda tx: tx.run("MERGE (p:Person {id: $id}) SET p += $props", id=doc_id, props=dossier))

            entities_map = {}
            for ent_text, ent_label in [(e.text, e.label_) for e in nlp(notes).ents]:
                ent_id, newly_created = (None, False)
                if ent_label == 'ORG':
                    ent_id, newly_created = session.execute_write(resolve_and_upsert_entity, 'ORG', ent_text, existing_orgs)
                    if newly_created: existing_orgs[ent_text] = ent_id
                elif ent_label == 'GPE':
                    ent_id, newly_created = session.execute_write(resolve_and_upsert_entity, 'GPE', ent_text, existing_gpes)
                    if newly_created: existing_gpes[ent_text] = ent_id

                if ent_id:
                    entities_map[ent_id] = (ent_text, ent_label)
                    session.execute_write(create_relationship, 'Person', doc_id, ent_label, ent_id, 'MENTIONED')

            session.execute_write(extract_relationships, doc_id, notes, entities_map, all_people_ids)

    logging.info("Data ingestion complete.")
