#!/bin/bash

# Configuration
PROJECT_ROOT="/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
AIRFLOW_BIN="/home/gad/miniforge3/envs/SE363Final/bin/airflow"
PYTHON_BIN="/home/gad/miniforge3/envs/SE363Final/bin/python"
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
# Add Airflow bin to PATH so 'airflow standalone' subprocesses can find it
export PATH="$(dirname "$AIRFLOW_BIN"):$PATH"
# Force SPARK_HOME to the standard miniforge location where pyspark is installed
export SPARK_HOME="/home/gad/miniforge3/lib/python3.12/site-packages/pyspark"

# 2.5 Initialize Database (Prevent Dashboard Crash)
echo "🌱 Seeding Database Trends..."
python scripts/seed_trends.py

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

# 4. Trigger Pipeline (Parallel Execution for Streaming)
echo "🧬 Triggering Demo Pipeline (Mode: $MODE)..."
echo "---------------------------------------------------"

# 4. Trigger Airflow Pipeline (Full Stack)
echo "🧬 Triggering Demo Pipeline (Mode: $MODE)..."
echo "---------------------------------------------------"

# Start Airflow Components (Standalone Mode for Demo)
echo "🌪️  Starting Airflow Standalone..."
"$AIRFLOW_BIN" standalone > airflow_standalone.log 2>&1 &
STANDALONE_PID=$!

echo "⏳ Waiting 15s for Airflow to Initialize..."
sleep 15

# Extract Credentials
echo "🔑 Airflow Credentials:"
grep "User" airflow_standalone.log || echo "User: admin (check log)"
grep "Password" airflow_standalone.log || echo "Password: (check airflow_standalone.log)"

echo "🛡️  Starting Redundant Background Services (Robustness Layer)..."
# Start Producer manually (Infinite Loop enabled)
"$PYTHON_BIN" -u streaming/producer_simple.py > producer.log 2>&1 &
PRODUCER_PID=$!

# Start Worker manually
"$PYTHON_BIN" -u streaming/intelligence_worker.py --max-cycles 100 > worker.log 2>&1 &
WORKER_PID=$!

# Start Spark Consumer (Eco/Demo Mode)
echo "⚡ Starting Spark Structured Streaming Consumer (Lite Mode)..."
# Tuning for Stability: local[1] to limit threads, 512m-1g memory to prevent OOM
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.postgresql:postgresql:42.7.1 \
    --master local[1] \
    --driver-memory 1g \
    --conf spark.executor.memory=1g \
    --conf spark.sql.shuffle.partitions=4 \
    --conf spark.default.parallelism=2 \
    streaming/spark_consumer.py > consumer.log 2>&1 &
CONSUMER_PID=$!

# Trigger the DAG
if [ -f "$AIRFLOW_BIN" ]; then
    "$AIRFLOW_BIN" dags trigger unified_pipeline -c "{\"mode\": \"$MODE\"}"
else
    airflow dags trigger unified_pipeline -c "{\"mode\": \"$MODE\"}"
fi

echo "---------------------------------------------------"
echo "✅ Pipeline triggered in Airflow!"
echo "👉 Open http://localhost:8080 to view the DAG Graph."
echo "👉 Dashboard is at http://localhost:8501"
echo "💡 Services are running. Press Ctrl+C to stop Dashboard."

cleanup() {
    echo -e "\n🛑 Shutting down..."
    kill $STREAMLIT_PID 2>/dev/null
    kill $STANDALONE_PID 2>/dev/null
    kill $PRODUCER_PID 2>/dev/null
    kill $CONSUMER_PID 2>/dev/null
    kill $WORKER_PID 2>/dev/null
    # Kill any lingering airflow processes
    pkill -f "airflow standalone"
    pkill -f "airflow scheduler"
    pkill -f "airflow triggerer"
    pkill -f "airflow webserver"
    pkill -f "airflow api-server"
    pkill -f "airflow dag-processor"
    echo "✅ Done."
    exit 0
}
trap cleanup SIGINT

# Keep script alive for the background process
wait $STREAMLIT_PID
