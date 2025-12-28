#!/bin/bash

# Configuration
PROJECT_ROOT="/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
AIRFLOW_BIN="/home/gad/miniforge3/envs/SE363Final/bin/airflow"

# Check if Airflow exists
if [ ! -f "$AIRFLOW_BIN" ]; then
    echo "❌ Airflow binary not found at $AIRFLOW_BIN"
    echo "Please check your conda environment."
    exit 1
fi

echo "🚀 Running Airflow DAG Test for 'unified_pipeline'..."
echo "-----------------------------------------------------"

# Set Critical Environment Variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"

# Run the test
# We use 'dags test' which runs locally without a scheduler
"$AIRFLOW_BIN" dags test unified_pipeline 2025-01-01 -c '{"mode": "demo"}'

echo "-----------------------------------------------------"
echo "✅ Test Complete. Check logs above for 'Branching Decision' and task execution."
