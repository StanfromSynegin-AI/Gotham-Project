# Mini-Gotham

This project is a personal-scale recreation of a data ingestion, analysis, and visualization platform. It allows you to ingest structured data, build a knowledge graph with resolved entities, and explore it through a web interface that supports both keyword and semantic search.

## Features

-   **Dockerized Services:** Elasticsearch, Neo4j, and other services are managed with Docker Compose.
-   **Robust Data Ingestion:** The ingestion script (`ingest.py`) processes CSV data, extracts entities with spaCy, and performs entity resolution using fuzzy string matching to avoid duplicates.
-   **Rich Knowledge Graph:** Creates a graph of entities (people, organizations, locations) and their relationships (e.g., `WORKS_FOR`, `LIVES_AT`).
-   **Semantic Search:** Generates vector embeddings for text data, enabling search based on meaning, not just keywords.
-   **Interactive UI:** A Streamlit application (`app.py`) provides a user-friendly interface for searching and visualizing the knowledge graph.

## Setup and Usage

### Prerequisites

-   Docker and Docker Compose installed on your machine.
-   Python 3.7+ and `pip`.

### 1. Install Dependencies

Install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The spaCy English model will be downloaded automatically by the ingestion script if it's not already installed.

### 2. Start the Services

Launch the Elasticsearch and Neo4j containers:

```bash
docker compose up -d
```

*Note: If `docker compose` doesn't work, you may need to use the older `docker-compose` command.*

### 3. Ingest the Data

Run the ingestion script. This will process the `people.csv` file, generate embeddings, and load the data into Elasticsearch and Neo4j.

```bash
python ingest.py
```

### 4. Launch the Application

Start the Streamlit web application:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to use the application. You'll find separate inputs for keyword and semantic search, as well as the interactive knowledge graph.

### 5. Shutting Down

To stop the Docker containers, run:

```bash
docker compose down
```
