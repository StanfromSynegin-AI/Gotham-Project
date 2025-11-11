# Mini-Gotham: Intelligence Dashboard

This project is a professional-grade, personal-scale intelligence analysis platform. It ingests dossier-style data, builds an interconnected knowledge graph, and provides a sophisticated web interface for exploration, featuring advanced search and filtering capabilities.

## Features

-   **Professional Dark Theme UI:** A sleek, dark-themed Streamlit application designed for intelligence analysis.
-   **Robust JSON Data Format:** Uses a `dossiers.json` file for complex, reliable data handling, eliminating CSV parsing issues.
-   **Advanced Search Index:** Features a custom Elasticsearch analyzer for "perfect index searching," providing more accurate and intelligent results by understanding word variations.
-   **Palantir-like Filtering:** A sidebar in the UI allows for faceted search, enabling you to filter results by `Nationality` and `Status` to slice and dice the data.
-   **Rich Knowledge Graph:** Builds a detailed Neo4j graph with nuanced relationships (e.g., `CONTACT_WITH`, `FINANCIAL_TIE_TO`) extracted from the data.
-   **Sleek Graph Visualization:** A professionally designed network graph with custom icons, a muted color palette, and a clean, stable layout.

## Foolproof Setup and Execution

This project includes an automated script to ensure a clean, error-free setup.

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

There is no script for Windows, but you can run the following commands one by one in your terminal in the project folder. This achieves the same result.

```powershell
# Stop and delete old containers and their data
docker compose down -v

# Start fresh containers
docker compose up -d

# Install Python libraries
pip install -r requirements.txt

# Run the data ingestion
python ingest.py

# Launch the app
streamlit run app.py
```

After running the script or the commands, the application will be available at **`http://localhost:8501`**.

---

### Manual Steps (for reference)

If you prefer to run the steps manually, here they are:

1.  **Reset Environment:** `docker compose down -v`
2.  **Start Services:** `docker compose up -d`
3.  **Install Dependencies:** `pip install -r requirements.txt`
4.  **Ingest Data:** `python ingest.py`
5.  **Run App:** `streamlit run app.py`
