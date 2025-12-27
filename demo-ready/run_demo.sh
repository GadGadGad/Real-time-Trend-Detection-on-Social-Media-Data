#!/bin/bash

echo "🚀 Setting up Demo Environment..."

# 1. Start Kafka
echo "🐳 Starting Kafka (Docker)..."
docker-compose up -d
echo "⏳ Waiting for Kafka to be ready (10s)..."
sleep 10

# 2. Check Dependencies
echo "📦 Checking Python dependencies..."
pip install -r requirements.txt

# 3. Instructions
echo ""
echo "✅ Environment Ready!"
echo "---------------------------------------------------"
echo "👉 Step 1: Run the Consumer (Spark Job) in a new terminal:"
echo "   cd demo-ready && python consumer.py"
echo ""
echo "👉 Step 2: Run the Producer (Data Replay) in another terminal:"
echo "   cd demo-ready && python producer.py"
echo "---------------------------------------------------"
echo "Make sure you put some .csv files in 'demo-ready/data' folder first!"
