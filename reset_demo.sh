#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")"; pwd)"
PYTHON_BIN="$(which python)"
AIRFLOW_BIN="$(which airflow)"

echo "🔄 RESETTING DEMO SYSTEM STATE..."
echo "---------------------------------------------------"

# 1. Kill all processes
echo "🛑 Stopping all processes..."
pkill -f run_full_system.sh 2>/dev/null || true
pkill -f streamlit 2>/dev/null || true
pkill -f airflow 2>/dev/null || true
pkill -f python 2>/dev/null || true
pkill -f java 2>/dev/null || true
sleep 2

# 2. Reset Infrastructure (Docker)
echo "🐳 Restarting Infrastructure (Resetting DB Data)..."
cd "$PROJECT_ROOT/streaming"
docker-compose down
docker-compose up -d
echo "⏳ Waiting for DB to be ready..."
sleep 15

# 3. Clear Files
echo "🧹 Clearing Logs and Spark Checkpoints..."
cd "$PROJECT_ROOT"
rm -rf streaming/checkpoints
rm -rf *.log
rm -rf ~/airflow/*.log
rm -rf ~/airflow/logs/*

# 4. Reset Airflow DB (Metadata)
echo "🌪️ Resetting Airflow Database..."
if [[ -n "$AIRFLOW_BIN" ]]; then
    # Delete the sqlite db if exists for a hard reset
    rm -f ~/airflow/airflow.db 
    # Airflow standalone will re-init it on next run
fi

# 5. Seed Database (Recreates tables)
echo "🌱 Initializing and Seeding Database..."
"$PYTHON_BIN" scripts/seed_trends.py

echo "---------------------------------------------------"
echo "✅ SYSTEM RESET COMPLETE!"
echo "👉 You can now run ./run_full_system.sh to start fresh."
