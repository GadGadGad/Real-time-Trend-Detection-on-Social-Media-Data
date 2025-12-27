"""
Spark Streaming Consumer: Process posts from Kafka using ML pipeline.

This script runs a Spark Structured Streaming job that:
1. Consumes posts from Kafka
2. Runs trend detection using find_matches_hybrid
3. Writes results to MongoDB

Usage:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
                 spark_consumer.py
"""

import os
import sys
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, from_json, window, collect_list, udf
    from pyspark.sql.types import StringType, StructType, StructField, ArrayType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️ PySpark not available in this environment")


# === CONFIGURATION ===
KAFKA_BOOTSTRAP_SERVERS = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
POSTS_TOPIC = 'posts'
TRENDS_TOPIC = 'trends'

# Processing config
BATCH_INTERVAL = '5 minutes'  # Micro-batch window
MIN_POSTS_PER_BATCH = 10


def create_spark_session():
    """Create Spark session with Kafka and MongoDB packages."""
    return SparkSession.builder \
        .appName("TrendDetection") \
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0") \
        .config("spark.mongodb.output.uri", f"{MONGODB_URI}/trends.results") \
        .getOrCreate()


def get_post_schema():
    """Define schema for post messages."""
    return StructType([
        StructField("content", StringType(), True),
        StructField("source", StringType(), True),
        StructField("time", StringType(), True),
        StructField("likes", StringType(), True),
        StructField("comments", StringType(), True),
        StructField("_ingested_at", StringType(), True),
    ])


def process_batch(batch_df, batch_id):
    """Process a micro-batch of posts using find_matches_hybrid."""
    from src.pipeline.main_pipeline import find_matches_hybrid, load_trends
    
    # Convert Spark DataFrame to Python list
    posts = [row.asDict() for row in batch_df.collect()]
    
    if len(posts) < MIN_POSTS_PER_BATCH:
        print(f"Batch {batch_id}: Only {len(posts)} posts, skipping (min: {MIN_POSTS_PER_BATCH})")
        return
    
    print(f"🔄 Processing batch {batch_id} with {len(posts)} posts...")
    
    # Load trends (could also consume from Kafka in real-time)
    trend_files = [
        os.path.join(PROJECT_ROOT, "crawlers/new_data/trendings/trending_VN_7d_*.csv")
    ]
    import glob
    actual_files = []
    for pattern in trend_files:
        actual_files.extend(glob.glob(pattern))
    
    if not actual_files:
        print(f"❌ No trend files found")
        return
    
    trends = load_trends(actual_files[:1])  # Use latest
    
    # Run ML pipeline
    try:
        matches, components = find_matches_hybrid(
            posts=posts,
            trends=trends,
            taxonomy_method='auto',
            sentiment_method='auto',
            use_llm=bool(GEMINI_API_KEY),
            gemini_api_key=GEMINI_API_KEY,
            min_cluster_size=3,
            return_components=True,
        )
        
        print(f"✅ Batch {batch_id}: Generated {len(matches)} results")
        
        # Save to MongoDB
        save_to_mongodb(matches, batch_id)
        
    except Exception as e:
        print(f"❌ Batch {batch_id} failed: {e}")


def save_to_mongodb(results, batch_id):
    """Save results to MongoDB."""
    try:
        from pymongo import MongoClient
        
        client = MongoClient(MONGODB_URI)
        db = client['trends']
        collection = db['results']
        
        for r in results:
            r['_batch_id'] = batch_id
            r['_processed_at'] = datetime.now().isoformat()
        
        if results:
            collection.insert_many(results)
            print(f"  💾 Saved {len(results)} results to MongoDB")
        
        client.close()
    except Exception as e:
        print(f"  ⚠️ MongoDB save failed: {e}")


def run_streaming():
    """Main streaming loop."""
    if not SPARK_AVAILABLE:
        print("❌ PySpark required. This script should run via spark-submit.")
        return 1
    
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"🚀 Starting Spark Streaming from Kafka ({KAFKA_BOOTSTRAP_SERVERS})...")
    
    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", POSTS_TOPIC) \
        .option("startingOffsets", "latest") \
        .load()
    
    # Parse JSON
    schema = get_post_schema()
    posts_df = df.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")
    
    # Process in windowed batches
    query = posts_df.writeStream \
        .foreachBatch(process_batch) \
        .trigger(processingTime=BATCH_INTERVAL) \
        .start()
    
    print(f"📡 Streaming started. Processing every {BATCH_INTERVAL}...")
    query.awaitTermination()


def run_batch_mode():
    """Fallback: Run in batch mode without Spark Streaming."""
    print("🔄 Running in batch mode (no Spark)...")
    
    from src.pipeline.main_pipeline import find_matches_hybrid, load_json, load_trends
    import glob
    
    # Load posts
    post_files = glob.glob(os.path.join(PROJECT_ROOT, "crawlers/**/*.json"), recursive=True)
    posts = []
    for f in post_files[:5]:  # Limit for testing
        posts.extend(load_json(f))
    
    # Load trends
    trend_files = glob.glob(os.path.join(PROJECT_ROOT, "crawlers/new_data/trendings/*.csv"))
    trends = load_trends(trend_files[:1])
    
    print(f"📊 Loaded {len(posts)} posts, {len(trends)} trends")
    
    # Run pipeline
    matches, _ = find_matches_hybrid(
        posts=posts[:1000],
        trends=trends,
        taxonomy_method='auto',
        sentiment_method='auto',
        use_llm=False,
        min_cluster_size=3,
        return_components=True,
    )
    
    print(f"✅ Generated {len(matches)} results")
    save_to_mongodb(matches, 0)


if __name__ == '__main__':
    if SPARK_AVAILABLE:
        run_streaming()
    else:
        run_batch_mode()
