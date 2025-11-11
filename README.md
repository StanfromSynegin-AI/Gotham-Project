# Mini-Gotham: Intelligence Analysis Platform

This project is a professional-grade, personal-scale intelligence analysis platform. It ingests dossier and event data to build an interconnected knowledge graph and provides a multi-faceted web interface for advanced exploration, temporal analysis, and link discovery.

## Core Features

-   **Multi-Faceted UI:** A tab-based Streamlit dashboard for different analytical tasks:
    -   **Dossier Search:** A powerful search engine for entity records with keyword, semantic, and faceted filtering (by Nationality, Status).
    -   **Timeline Analysis:** A chronological view of all intelligence events, allowing an analyst to track sequences of activity.
    -   **Graph Explorer:** An interactive, professional-grade knowledge graph that visualizes entities, events, and their complex relationships.
-   **Event-Centric Data Model:** Ingests both `dossiers.json` (for entities) and `events.json` (for time-based activities), creating a dynamic analytical environment.
-   **Advanced Search:** Utilizes dedicated Elasticsearch indices with custom language analyzers for highly accurate and intelligent search results.
-   **Rich, Interconnected Graph:** Builds a detailed Neo4j graph with `Person`, `ORG`, `GPE`, and `Event` nodes, linked by meaningful, timestamped relationships.
-   **Professional Aesthetic:** Features a sleek dark theme, a clean layout, and sophisticated graph visualizations using custom icons.

## Foolproof Setup and Execution

The project includes an automated script to ensure a clean, error-free setup.

### Prerequisites

-   **Docker** and **Docker Compose**
-   **Python** (3.7+)

### Automated Setup (Recommended)

This is the easiest and most reliable way to run the project.

**For Linux and macOS:**

Open your terminal, navigate to the project folder, and run the setup script:

```bash
./reset_and_run.sh
```

**For Windows (Command Prompt or PowerShell):**

There is no script for Windows, but you can run the following commands one by one in your terminal. This achieves the same result.

```powershell
# 1. Stop and delete old containers and their data
docker compose down -v

# 2. Start fresh containers
docker compose up -d

# 3. Install Python libraries
pip install -r requirements.txt

# 4. Run the data ingestion
python ingest.py

# 5. Launch the app
streamlit run app.py
```

After running the script or the commands, the application will be available at **`http://localhost:8501`**.
