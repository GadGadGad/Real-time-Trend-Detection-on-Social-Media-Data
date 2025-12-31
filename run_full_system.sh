#!/bin/bash

# Auto-detect Project Root
PROJECT_ROOT="$(cd "$(dirname "$0")"; pwd)"
# Auto-detect Binaries (Assuming active conda env or standard PATH)
AIRFLOW_BIN="$(which airflow)"
PYTHON_BIN="$(which python)"

if [[ -z "$AIRFLOW_BIN" || -z "$PYTHON_BIN" ]]; then
    echo "❌ Error: Could not find 'airflow' or 'python' in PATH."
    echo "👉 Please activate your conda environment before running this script."
    exit 1
fi
MODE=${1:-demo} # Default to demo if no argument provided

echo "🚀 INITIALIZING UNIFIED DEMO SYSTEM"
echo "---------------------------------------------------"

# 0. Clean up any existing background processes
echo "🧹 Cleaning up existing background processes..."
pkill -f "producer_simple.py" || true
pkill -f "intelligence_worker.py" || true
pkill -f "spark_consumer.py" || true
pkill -f "airflow" || true
pkill -f "streamlit" || true
sleep 2

# 1. Infrastructure Check (Docker)
echo "🐳 Checking Infrastructure (Postgres & Kafka)..."
cd "$PROJECT_ROOT/streaming"
docker-compose up -d

echo "⏳ Waiting for services to be healthy..."
wait_for_container() {
    local container_name=$1
    echo -n "   - Waiting for $container_name... "
    while [ "$(docker inspect --format='{{json .State.Health.Status}}' $container_name 2>/dev/null)" != "\"healthy\"" ]; do
        echo -n "."
        sleep 2
    done
    echo " ✅"
}

wait_for_container "postgres_demo"
wait_for_container "kafka"
cd "$PROJECT_ROOT"

# 2. Environment Setup
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
# Add Airflow bin to PATH so 'airflow standalone' subprocesses can find it
export PATH="$(dirname "$AIRFLOW_BIN"):$PATH"
# Detect SPARK_HOME from the active pyspark installation
export SPARK_HOME="$("$PYTHON_BIN" -c "import pyspark; import os; print(os.path.dirname(pyspark.__file__))")"

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


echo "---------------------------------------------------"
echo "✅ System initialized! Airflow is ready."
echo "👉 Open http://localhost:8080 to trigger the 'unified_pipeline' DAG manually."
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
