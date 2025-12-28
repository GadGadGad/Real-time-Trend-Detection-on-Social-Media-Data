#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

# DYNAMICALLY FIND SPARK HOME
export SPARK_HOME=$(python -c "import pyspark; print(pyspark.__path__[0])")
echo "🔧 Fixed SPARK_HOME: $SPARK_HOME"

echo "🚀 Launching Spark Structured Streaming Job..."
echo "📦 Packages: Spark-SQL-Kafka, PostgreSQL"

# Use the spark-submit inside the pyspark directory to be sure
$SPARK_HOME/bin/spark-submit \
  --master local[2] \
  --driver-memory 2g \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.postgresql:postgresql:42.6.0 \
  src/spark/streaming_job.py > spark_job.log 2>&1 &

SPARK_PID=$!
echo "✅ Spark Job Submitted! PID: $SPARK_PID"
echo "📄 Logs are being written to spark_job.log"
