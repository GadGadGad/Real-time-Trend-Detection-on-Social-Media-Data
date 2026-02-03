# Real-time Event Detection on Social Media Data

A production-grade system for detecting and analyzing social trends by fusing **Real-time Social Media Streams (Kafka)** and **Mainstream News (Crawlers)**, enhanced by **LLM Intelligence (Gemini)**.

## Architecture

![Architecture](assets/full_pipeline.png)

## Key Features

- **Hybrid Data Fusion:** Combines high-velocity social signals with authoritative news sources.
- **Distributed Processing:** Uses **PySpark Structured Streaming** for scalable event clustering.
- **AI-Powered Insights:**
  - **Automated Summary:** Generates concise event descriptions in Vietnamese.
  - **Strategic Advice:** Actionable recommendations for Authorities and Businesses.
  - **Taxonomy Classification:** Categorizes events into standard Vietnamese social monitoring topics (T1-T7).
- **Dual-Mode Dashboard:**
  - **Demo:** Lightweight **Streamlit** UI for rapid prototyping.
  - **Pro:** Modern **Next.js (React)** + **FastAPI** architecture for production.
- **Unified Launch Script:** Start the complete stack (Database, Kafka, Spark, Worker, Dashboard) with a single command.

## Tech Stack

- **Streaming:** Apache Kafka, PySpark
- **Orchestration:** Apache Airflow
- **Database:** PostgreSQL (with SQLAlchemy)
- **AI/LLM:** Google Gemini Pro 1.5, Sentence-Transformers
- **Backend:** FastAPI (Pro Mode)
- **Frontend:** Streamlit (Demo) / Next.js + React (Pro)

## Quick Start

### 1. Prerequisites

- Docker & Docker Compose
- Conda or Python 3.10+
- Gemini API Key

### 2. Setup

```bash
# Clone the repository
git clone [repository-url]
cd [repository-name]

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Launch System

The system includes a unified bootstrap script that handles infrastructure checks, database seeding, and background services.

```bash
chmod +x run_full_system.sh
./run_full_system.sh demo  # Launches Streamlit Demo
# OR
./run_pro_system.sh        # Launches Next.js Pro System
```

*Note: The `demo` mode uses local simulations to showcase system capabilities without requiring live crawler credentials.*

## Project Structure

```text
├── airflow_home/      # Airflow configuration and local DB
├── dags/              # Orchestration logic (Unified Pipeline)
├── scripts/           # Training, Seeding, and Utility scripts
├── src/               # Core business logic (NLP, AI, DB models)
├── streaming/         # Real-time components (Dashboard, Producer, Worker)
└── run_full_system.sh # Main entry point for the demo system
```

## Monitoring

- **Dashboard:** [http://localhost:8501](http://localhost:8501)
- **Airflow UI:** [http://localhost:8080](http://localhost:8080)
- **Kafka / Postgres:** Managed via Docker Compose in `streaming/docker-compose.yml`

---
If you have any questions or need help regarding the project, please contact [me](mailto:[22521027@gm.uit.edu.vn]).
