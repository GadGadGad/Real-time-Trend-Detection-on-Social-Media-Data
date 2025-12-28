# Kafka + Spark + Airflow Pipeline

## Local Docker Stack for Trend Detection

### Quick Start

```bash
cd streaming

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View Spark UI
open http://localhost:8081

# Stop all
docker-compose down
```

### Architecture

```
CSV Files → [Kafka Producer] → Kafka → [Spark Consumer] → MongoDB → Dashboard
                                              │
                                    [find_matches_hybrid]
```

### Services

| Service | Port | URL |
|---------|------|-----|
| Kafka | 9092 | localhost:9092 |
| Spark UI | 8081 | http://localhost:8081 |
| MongoDB | 27017 | localhost:27017 |
| Airflow | 8080 | http://localhost:8080 |

---

## Demo Mode (Quick)

Nếu đã chạy xong pipeline trên Kaggle, dùng demo mode để demo nhanh:

1. Export results từ Kaggle → `output/kaggle_results.csv`
2. Chạy DAG `demo_mode_quick` trên Airflow
3. Kết quả sẽ được load mà không cần chạy ML pipeline

```bash
# Test locally
python dags/demo_mode_dag.py
```

---

## Full Pipeline Mode

Chạy full pipeline với Kafka + Spark:

```bash
# 1. Start Docker services
cd streaming && docker-compose up -d

# 2. Produce data to Kafka
pip install kafka-python
python kafka_producer.py

# 3. Start Spark consumer
spark-submit spark_consumer.py
```

---

## Fallback

Original streaming setup is still available in `streaming/` folder.

```bash
cd streaming
python consumer.py &
python intelligence_worker.py &
streamlit run dashboard.py
```
