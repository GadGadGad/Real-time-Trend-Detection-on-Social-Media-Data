# Airflow Setup Guide

## Quick Start

### 1. Install Airflow
```bash
pip install apache-airflow==2.8.0
```

### 2. Initialize Airflow
```bash
# Set home directory
export AIRFLOW_HOME=~/airflow

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### 3. Copy DAG file
```bash
mkdir -p ~/airflow/dags
cp dags/trend_detection_dag.py ~/airflow/dags/
```

### 4. Set GEMINI API Key
```bash
# Option 1: Environment variable
export GEMINI_API_KEY="your-api-key"

# Option 2: Airflow Variable (via UI or CLI)
airflow variables set GEMINI_API_KEY "your-api-key"
```

### 5. Start Airflow
```bash
# Terminal 1: Start webserver
airflow webserver --port 8080

# Terminal 2: Start scheduler
airflow scheduler
```

### 6. Access Airflow UI
Open http://localhost:8080 and login with admin/admin

---

## DAG Structure

```
trend_detection_pipeline
├── load_data          # Load posts/trends from CSV
├── run_pipeline       # Run find_matches_hybrid
├── generate_report    # Create summary JSON
└── cleanup_old_cache  # Remove old cache files
```

## Schedule

Default: **Daily at midnight** (`0 0 * * *`)

To change, edit `schedule_interval` in the DAG file:
- `@hourly` - Every hour
- `@daily` - Every day
- `0 */6 * * *` - Every 6 hours
- `None` - Manual trigger only

## Testing Locally

```bash
cd "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
python dags/trend_detection_dag.py
```
