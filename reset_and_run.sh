#!/bin/bash

# This script automates the entire process of resetting and running the Mini-Gotham project.
# It ensures a clean environment to prevent errors.

echo "--- Stopping and removing old Docker containers and volumes..."
docker compose down -v

echo "--- Starting fresh Docker containers..."
docker compose up -d

echo "--- Installing/updating Python dependencies..."
pip install -r requirements.txt

echo "--- Running the data ingestion script..."
python ingest.py

echo "--- Launching the Streamlit application..."
streamlit run app.py

echo "---"
echo "Setup complete! The application should be running and accessible at http://localhost:8501"
