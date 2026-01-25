#!/bin/bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")"; pwd)"
MODE=${1:-demo}

echo "🚀 INITIALIZING CYBER INTELLIGENCE PRO SYSTEM (Airflow Orchestrated)"
echo "-------------------------------------------------------------------"

# --- Load .env (project root) safely (no variable expansion) ---
# This avoids bash exiting when .env contains undefined vars (important when using strict mode).
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "🔧 Loading environment from .env"
  while IFS='=' read -r key value; do
    # skip comments/empty lines
    [[ -z "${key}" ]] && continue
    [[ "${key}" =~ ^[[:space:]]*# ]] && continue
    key="$(echo "$key" | xargs)"
    value="$(echo "${value:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    # strip surrounding quotes if present
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    export "$key=$value"
  done < "$PROJECT_ROOT/.env"
fi

# --- Resolve python/airflow from current environment ---
PYTHON_BIN="$(command -v python || true)"
AIRFLOW_BIN="$(command -v airflow || true)"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "❌ python not found in PATH. Activate your conda env first (conda activate <env>)."
  exit 1
fi

if [[ -z "${AIRFLOW_BIN}" ]]; then
  echo "❌ airflow not found in current environment. Install it in your env."
  exit 1
fi

# --- Pick docker compose command (prefer v2 plugin) ---
DOCKER_COMPOSE_CMD=""
if docker compose version >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE_CMD="docker-compose"
else
  echo "❌ Neither 'docker compose' nor 'docker-compose' found."
  exit 1
fi

# 1. Infrastructure (Docker)
echo "🐳 Checking Infrastructure (Postgres & Kafka)..."
cd "$PROJECT_ROOT/streaming"
$DOCKER_COMPOSE_CMD up -d
cd "$PROJECT_ROOT"

# 2. Environment Setup
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export AIRFLOW_HOME="$HOME/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"

# Optional: only set SPARK_HOME if pyspark exists
if "$PYTHON_BIN" -c "import pyspark" >/dev/null 2>&1; then
  export SPARK_HOME="$("$PYTHON_BIN" -c "import pyspark, os; print(os.path.dirname(pyspark.__file__))")"
else
  echo "⚠️  pyspark not found in env; SPARK_HOME not set."
fi

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
nohup "$PYTHON_BIN" api/main.py > api_server.log 2>&1 &
API_PID=$!

echo "🎨 Starting Next.js Dashboard..."

# Install deps once (idempotent)
bash -lc "cd \"$PROJECT_ROOT/dashboard-ui\" && npm install" > "$PROJECT_ROOT/dashboard-ui/frontend_install.log" 2>&1 || true

# Run Next.js with Node 20 via nvm if available; otherwise fallback to system node/npm
# NOTE: this works even if your interactive shell is fish.
NEXT_LAUNCH_CMD='
cd "'"$PROJECT_ROOT"'/dashboard-ui" || exit 1

# Try nvm Node 20
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
  . "$NVM_DIR/nvm.sh"
  if command -v nvm >/dev/null 2>&1; then
    nvm use 20 >/dev/null 2>&1 || nvm install 20 >/dev/null 2>&1
  fi
fi

# Check node version requirement (Next >= 20.9.0)
NODEV=$(node -v 2>/dev/null || echo "v0.0.0")
MAJOR=$(echo "$NODEV" | sed "s/v//" | cut -d. -f1)
MINOR=$(echo "$NODEV" | sed "s/v//" | cut -d. -f2)
PATCH=$(echo "$NODEV" | sed "s/v//" | cut -d. -f3)

if [ "$MAJOR" -lt 20 ] || { [ "$MAJOR" -eq 20 ] && [ "$MINOR" -lt 9 ]; }; then
  echo "❌ Node $NODEV detected. Next.js requires >= 20.9.0."
  echo "   Fix: install Node 20 (via nvm) then re-run ./run_pro_system.sh"
  exit 1
fi

npm run dev
'

nohup bash -lc "$NEXT_LAUNCH_CMD" > "$PROJECT_ROOT/dashboard-ui/frontend.log" 2>&1 &
FRONT_PID=$!
cd "$PROJECT_ROOT"

# 5. Start Orchestration Layer (Airflow)
echo "🌪️  Starting Airflow Standalone..."
"$AIRFLOW_BIN" standalone > airflow_standalone.log 2>&1 &
AIRFLOW_PID=$!

# 6. Start Robustness Layer (Simulated Producer & Workers)
echo "🛡️  Launching Real-time Injection & Analysis Workers..."
nohup "$PYTHON_BIN" -u streaming/producer_simple.py > producer.log 2>&1 &
PRODUCER_PID=$!
nohup "$PYTHON_BIN" -u streaming/intelligence_worker.py --max-cycles 100 > worker.log 2>&1 &
WORKER_PID=$!

echo "-------------------------------------------------------------------"
echo "✅ PRO SYSTEM IS ONLINE!"
echo "👉 Dashboard (Premium): http://localhost:3000"
echo "👉 Orchestrator (Airflow): http://localhost:8080"
echo "👉 API Backend (FastAPI): http://localhost:8000/docs"
echo "-------------------------------------------------------------------"
echo "💡 TIP: Use Airflow UI to trigger 'unified_pipeline' for Spark processing."
echo "Press Ctrl+C to shut down all services."

cleanup() {
  echo -e "\n🛑 Shutting down Cyber Pro components..."
  kill $API_PID $FRONT_PID $AIRFLOW_PID $PRODUCER_PID $WORKER_PID 2>/dev/null || true
  pkill -f "airflow standalone" || true
  pkill -f "airflow scheduler" || true
  pkill -f "airflow triggerer" || true
  pkill -f "airflow webserver" || true
  pkill -f "next-server" || true
  echo "✅ All processes terminated."
  exit 0
}
trap cleanup SIGINT

while true; do sleep 1; done
