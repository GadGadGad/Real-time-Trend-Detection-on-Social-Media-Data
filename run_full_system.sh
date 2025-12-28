#!/bin/bash

# Configuration
PROJECT_ROOT="/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
AIRFLOW_BIN="/home/gad/miniforge3/envs/SE363Final/bin/airflow"
MODE=${1:-demo} # Default to demo if no argument provided

echo "🚀 INITIALIZING UNIFIED DEMO SYSTEM"
echo "---------------------------------------------------"

# 1. Infrastructure Check (Docker)
echo "🐳 Checking Infrastructure (Postgres & Kafka)..."
if ! docker ps | grep -q "postgres_demo" || ! docker ps | grep -q "kafka"; then
    echo "⚠️  Infrastructure services are down. Starting them now..."
    cd "$PROJECT_ROOT/streaming"
    docker-compose up -d
    echo "⏳ Waiting for services to stabilize (10s)..."
    sleep 10
else
    echo "✅ Infrastructure services are running."
fi

# 2. Environment Setup
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"

# 3. Start Dashboard in Background
echo "📊 Launching Streamlit Dashboard..."
cd "$PROJECT_ROOT"
streamlit run streaming/dashboard.py &
STREAMLIT_PID=$!

# Function to clean up on exit
cleanup() {
    echo -e "\n🛑 Shutting down demo..."
    kill $STREAMLIT_PID 2>/dev/null
    echo "✅ Done."
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

echo "⏳ Waiting for Dashboard to initialize..."
sleep 5

# 4. Trigger Airflow Pipeline
echo "🧬 Triggering Airflow Pipeline (Mode: $MODE)..."
echo "---------------------------------------------------"
if [ -f "$AIRFLOW_BIN" ]; then
    "$AIRFLOW_BIN" dags test unified_pipeline 2025-01-01 -c "{\"mode\": \"$MODE\"}"
else
    # Fallback to system airflow
    airflow dags test unified_pipeline 2025-01-01 -c "{\"mode\": \"$MODE\"}"
fi

echo "---------------------------------------------------"
echo "✅ Pipeline sequence finished."
echo "💡 The dashboard is still running. Press Ctrl+C to stop it."

# Keep script alive for the background process
wait $STREAMLIT_PID
