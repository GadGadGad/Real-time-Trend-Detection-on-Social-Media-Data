#!/bin/bash

echo "🔄 Resetting Demo Environment..."

# 1. Recreate Kafka Topic (Clears all old messages)
echo "🗑️  Deleting Kafka Topic (posts-stream)..."
docker exec kafka kafka-topics --delete --topic posts-stream --bootstrap-server localhost:29092

# Wait for deletion to complete
echo "⏳ Waiting for topic deletion..."
while docker exec kafka kafka-topics --list --bootstrap-server localhost:29092 | grep -q "posts-stream"; do
    sleep 1
done

echo "✨ Recreating Kafka Topic (posts-stream)..."
docker exec kafka kafka-topics --create --topic posts-stream --bootstrap-server localhost:29092 --partitions 1 --replication-factor 1

# 2. Clear Spark Checkpoints (if any exist locally)
# In this demo setup, we mostly rely on memory, but good to be safe.
if [ -d "checkpoints" ]; then
    echo "🧹 Cleaning Spark checkpoints..."
    rm -rf checkpoints
fi

# 3. Optional: Seed Trends
read -p "🌱 Do you want to seed initial trends? (y/n) " seed_ans
if [[ $seed_ans != "n" ]]; then
    echo "🚀 Running Seed Trends from predefined JSON..."
    python seed_trends.py --json "data/trend_refine_6e87b7f5d9f9833994e38d408d1e1153.json"
fi

echo "✅ Environment Reset Complete!"
echo "👉 Now run: 'python consumer.py' (It will auto-reset the DB)"
