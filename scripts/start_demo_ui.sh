#!/bin/bash

# Configuration
PROJECT_ROOT="/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
ENV_BIN="/home/gad/miniforge3/envs/SE363Final/bin"
AIRFLOW_BIN="$ENV_BIN/airflow"
STREAMLIT_BIN="$ENV_BIN/streamlit"

# CRITICAL: Airflow Standalone spawns subprocesses using 'airflow', so it MUST be in the PATH
export PATH="$ENV_BIN:$PATH"

export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export AIRFLOW_HOME="$PROJECT_ROOT/airflow_home" 

echo "-----------------------------------------------------"
echo "🚀 Starting Full Stack Demo (Airflow + Streamlit)..."
echo "-----------------------------------------------------"

# 1. Infrastructure Check (Docker)
echo "🐳 Checking Infrastructure (Postgres & Kafka)..."
if ! docker ps | grep -q "postgres_demo" || ! docker ps | grep -q "kafka"; then
    echo "⚠️  Infrastructure services are down. Starting them now..."
    cd "$PROJECT_ROOT/streaming"
    docker-compose up -d
    echo "   -> Waiting for services to stabilize (10s)..."
    sleep 10
else
    echo "✅ Infrastructure services are running."
fi

# 2. Start Airflow Standalone (Background)
echo "🕒 Starting Airflow Standalone..."
cd "$PROJECT_ROOT"
"$AIRFLOW_BIN" standalone > logs/airflow_standalone.log 2>&1 &
AIRFLOW_PID=$!
echo "   -> Airflow Standalone running (PID: $AIRFLOW_PID)"

# 3. Wait a moment for Airflow to warm up
echo "⏳ Waiting 15s for Airflow to initialize..."
sleep 15

# 4. Start Streamlit Dashboard (Foreground)
echo "📊 Starting Streamlit Dashboard (Port 8501)..."
echo "   -> Access Dashboard at: http://localhost:8501"
echo "   -> Access Airflow at:   http://localhost:8080"
echo "   (Check logs/airflow_standalone.log for credentials!)"
echo "-----------------------------------------------------"
echo "PRESS CTRL+C TO STOP ALL SERVICES"

# Trapping Ctrl+C to kill all background processes
cleanup() {
    echo -e "\n🛑 Shutting down all services..."
    kill $AIRFLOW_PID 2>/dev/null
    echo "✅ Done."
    exit 0
}
trap cleanup INT

"$STREAMLIT_BIN" run streaming/dashboard.py 
