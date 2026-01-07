#!/bin/bash

# Real-time Event Detection - Cyber Intelligence Pro Launcher (Full Stack)
# Architecture: Kafka -> Spark -> Postgres -> FastAPI -> Next.js
# Orchestration: Airflow

PROJECT_ROOT="$(cd "$(dirname "$0")"; pwd)"
ENV_BIN="/home/gad/miniforge3/envs/SE363Final/bin"
AIRFLOW_BIN="$ENV_BIN/airflow"
PYTHON_BIN="$ENV_BIN/python"
export PATH="$ENV_BIN:$PATH"
MODE=${1:-demo}

echo "🚀 INITIALIZING CYBER INTELLIGENCE PRO SYSTEM (Airflow Orchestrated)"
echo "-------------------------------------------------------------------"

# 1. Infrastructure (Docker)
echo "🐳 Checking Infrastructure (Postgres & Kafka)..."
cd "$PROJECT_ROOT/streaming"
docker-compose up -d
cd "$PROJECT_ROOT"

# 2. Environment Setup
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export AIRFLOW_HOME="$HOME/airflow" # Default airflow home
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export SPARK_HOME="$("$PYTHON_BIN" -c "import pyspark; import os; print(os.path.dirname(pyspark.__file__))")"

# 3. Stop potential lingering processes
echo "🧹 Cleaning up background legacy processes..."
pkill -f "api/main.py" || true
pkill -f "next-server" || true
pkill -f "airflow" || true
pkill -f "producer_simple.py" || true
pkill -f "intelligence_worker.py" || true
pkill -f "spark_consumer.py" || true
sleep 1

# 4. Start Serving Layer (FastAPI + Next.js)
echo "⚡ Starting FastAPI Backend..."
nohup python api/main.py > api_server.log 2>&1 &
API_PID=$!

echo "🎨 Starting Next.js Dashboard..."
cd dashboard-ui
nohup npm run dev > frontend.log 2>&1 &
FRONT_PID=$!
cd ..

# 5. Start Orchestration Layer (Airflow)
echo "🌪️  Starting Airflow Standalone..."
"$AIRFLOW_BIN" standalone > airflow_standalone.log 2>&1 &
AIRFLOW_PID=$!

# 6. Start Robustness Layer (Simulated Producer & Workers)
echo "🛡️  Launching Real-time Injection & Analysis Workers..."
nohup python -u streaming/producer_simple.py > producer.log 2>&1 &
PRODUCER_PID=$!
nohup python -u streaming/intelligence_worker.py --max-cycles 100 > worker.log 2>&1 &
WORKER_PID=$!

echo "-------------------------------------------------------------------"
echo "✅ PRO SYSTEM IS ONLINE!"
echo "👉 Dashboard (Premium): http://localhost:3000"
echo "👉 Orchestrator (Airflow): http://localhost:8080"
echo "👉 API Backend (FastAPI): http://localhost:8000/docs"
echo "-------------------------------------------------------------------"
echo "💡 TIP: Use Airflow UI to trigger 'unified_pipeline' for Spark processing."
echo "Press Ctrl+C to shut down all services."

# Cleanup logic
cleanup() {
    echo -e "\n🛑 Shutting down Cyber Pro components..."
    kill $API_PID $FRONT_PID $AIRFLOW_PID $PRODUCER_PID $WORKER_PID 2>/dev/null
    pkill -f "airflow standalone"
    pkill -f "airflow scheduler"
    pkill -f "airflow triggerer"
    pkill -f "airflow webserver"
    pkill -f "next-server"
    echo "✅ All processes terminated."
    exit 0
}

trap cleanup SIGINT

# Keep script running
while true; do sleep 1; done
