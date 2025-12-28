# 🚀 Hybrid Real-time Trend Detection System

A production-grade system detecting social trends by fusing **Social Media (Kafka Stream)** and **Mainstream News (Crawlers)**, powered by **LLM Refinement (Gemini)**.

![Architecture](docs/flow.mmd)

## 🔥 Key Features

- **Unified Ingestion Layer:** Hybrid support for **Live Crawling** (Real-time) and **Historical Replay** (Demo Mode).
- **SAHC Clustering:** Specialized Soft-Alignment Hierarchical Clustering for noisy short texts.
- **LLM Intelligence:** 
  - **Reasoning:** Explains *why* something is trending.
  - **Strategic Advice:** Generates actionable insights for State ("Quản lý") and Business ("Thương mại").
  - **Noise Filtering:** Semantic guardrails to remove spam/generic topics.
- **Real-time Dashboard:** Streamlit UI with pulse animations and live updates.

## 🛠 Tech Stack

- **Ingestion:** Apache Kafka, VnExpress Crawler
- **Orchestration:** Apache Airflow (`unified_pipeline_dag.py`)
- **Processing:** Spark-like Micro-batching (`kafka_consumer.py`)
- **AI/ML:** 
  - **Embedding:** `sentence-transformers` (Fine-tuned `visobert`)
  - **LLM:** Gemini Pro 1.5 (via Google GenAI SDK)
- **Serving:** PostgreSQL + Streamlit

## 🚀 Quick Start (Demo)

### 1. Setup
```bash
pip install -r requirements.txt
# Set GEMINI_API_KEY and POSTGRES_URL in .env
```

### 2. Run Components
**Terminal 1: Consumer & Intelligence**
```bash
# Starts the background worker for Clustering & LLM
python streaming/kafka_consumer.py &
python streaming/intelligence_worker.py
```

**Terminal 2: Dashboard**
```bash
streamlit run streaming/dashboard.py
```

**Terminal 3: Trigger Data**
```bash
# Run one-off ingestion batch (Hybrid Mode)
python dags/unified_pipeline_dag.py
```

## 📂 Project Structure
```
├── dags/                  # Airflow DAGs (Unified, Live, Demo)
├── streaming/            # Dashboard & Intelligence Worker
├── streaming/             # Kafka Producers & Consumers
├── slides/                # LaTeX Presentation
├── scripts/               # Training & Evaluation Scripts
└── src/                   # Core Logic (LLM, NLP utils)
```
