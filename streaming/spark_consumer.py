"""
Spark Structured Streaming Consumer for Trend Detection
=========================================================
PySpark implementation for scalable real-time trend detection.
Replaces Python consumer with distributed processing.

Usage:
    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 spark_consumer.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_json, struct, udf, pandas_udf, 
    current_timestamp, lit, array, explode, size
)
from pyspark.sql.types import (
    StructType, StructField, StringType, FloatType, 
    ArrayType, IntegerType, TimestampType
)

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# --- CONFIG ---
KAFKA_BOOTSTRAP = "localhost:29092"
KAFKA_TOPIC = "posts_stream_v1"
POSTGRES_URL = "jdbc:postgresql://localhost:5432/trend_db"
POSTGRES_PROPS = {"user": "user", "password": "password", "driver": "org.postgresql.Driver"}
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
SIMILARITY_THRESHOLD = 0.65
MIN_CLUSTER_SIZE = 3
BATCH_INTERVAL = "10 seconds"
CHECKPOINT_DIR = "/tmp/spark-trend-checkpoints"

# --- SCHEMA ---
POST_SCHEMA = StructType([
    StructField("content", StringType(), True),
    StructField("source", StringType(), True),
    StructField("time", StringType(), True),
    StructField("url", StringType(), True),
    StructField("title", StringType(), True),
    StructField("final_topic", StringType(), True),
    StructField("topic_type", StringType(), True),
    StructField("category", StringType(), True),
    StructField("sentiment", StringType(), True),
])

# --- EMBEDDING MODEL (Broadcast) ---
_embedding_model = None

def get_embedding_model():
    """Lazy load embedding model on executor"""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    return _embedding_model

# --- PANDAS UDF FOR EMBEDDING ---
@pandas_udf(ArrayType(FloatType()))
def embed_texts(contents: pd.Series) -> pd.Series:
    """
    Vectorized embedding generation using Pandas UDF.
    Runs on executor, loads model once per partition.
    """
    model = get_embedding_model()
    # Filter empty content
    texts = contents.fillna("").tolist()
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)
    return pd.Series([e.tolist() for e in embeddings])


def load_trends_from_db(spark):
    """Load existing trends with embeddings from PostgreSQL"""
    try:
        trends_df = spark.read.jdbc(
            url=POSTGRES_URL,
            table="detected_trends",
            properties=POSTGRES_PROPS
        ).select("id", "trend_name", "embedding", "keywords")
        
        # Parse embeddings from JSON string to array
        parse_embedding = udf(
            lambda x: json.loads(x) if x else None, 
            ArrayType(FloatType())
        )
        trends_df = trends_df.withColumn("embedding_array", parse_embedding(col("embedding")))
        return trends_df.filter(col("embedding_array").isNotNull())
    except Exception as e:
        print(f"⚠️ Could not load trends from DB: {e}")
        return None


def cosine_similarity_udf(embedding1, embedding2):
    """Calculate cosine similarity between two embedding vectors"""
    if embedding1 is None or embedding2 is None:
        return 0.0
    a = np.array(embedding1)
    b = np.array(embedding2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

cosine_sim = udf(cosine_similarity_udf, FloatType())


def process_micro_batch(batch_df, batch_id, spark):
    """
    Process each micro-batch:
    1. Generate embeddings
    2. Match to existing trends
    3. Discover new trends via HDBSCAN
    4. Write to PostgreSQL
    """
    if batch_df.isEmpty():
        print(f"📦 Batch {batch_id}: Empty, skipping")
        return
    
    count = batch_df.count()
    print(f"📦 Batch {batch_id}: Processing {count} posts...")
    
    # Phase 1: Use pre-assigned final_topic if available
    matched_count = 0
    unmatched_df = batch_df.filter(
        (col("final_topic").isNull()) | 
        (col("final_topic").isin("Unknown", "Unassigned", "N/A", ""))
    )
    
    pre_assigned_df = batch_df.filter(
        (col("final_topic").isNotNull()) & 
        (~col("final_topic").isin("Unknown", "Unassigned", "N/A", ""))
    )
    
    if pre_assigned_df.count() > 0:
        # Write pre-assigned directly (update existing trends)
        pre_assigned_count = pre_assigned_df.count()
        print(f"   ✅ Pre-assigned: {pre_assigned_count} posts")
        matched_count += pre_assigned_count
        
        # Collect and update DB
        for row in pre_assigned_df.collect():
            update_trend_with_post(row.final_topic, row, spark)
    
    # Phase 2: Embedding matching for unmatched
    if unmatched_df.count() >= MIN_CLUSTER_SIZE:
        print(f"   🔄 Embedding matching for {unmatched_df.count()} unmatched posts...")
        
        # Add embeddings
        with_emb = unmatched_df.withColumn("embedding", embed_texts(col("content")))
        
        # Load trends
        trends_df = load_trends_from_db(spark)
        
        if trends_df is not None and trends_df.count() > 0:
            # Cross join for similarity (expensive but correct)
            # For large scale: use LSH approximate matching
            trends_collected = trends_df.collect()
            
            # Collect posts for matching (small batch is OK)
            posts_with_emb = with_emb.collect()
            
            new_matches = 0
            still_unmatched = []
            
            for post in posts_with_emb:
                best_sim = 0
                best_trend = None
                
                for trend in trends_collected:
                    sim = cosine_similarity_udf(post.embedding, trend.embedding_array)
                    if sim > best_sim:
                        best_sim = sim
                        best_trend = trend
                
                if best_sim >= SIMILARITY_THRESHOLD and best_trend:
                    update_trend_with_post(best_trend.trend_name, post, spark, best_sim)
                    new_matches += 1
                else:
                    still_unmatched.append(post)
            
            print(f"   ✅ Matched via embedding: {new_matches}")
            matched_count += new_matches
            
            # Phase 3: HDBSCAN for remaining unmatched
            if len(still_unmatched) >= MIN_CLUSTER_SIZE:
                discover_new_trends(still_unmatched, spark)
        else:
            # No trends in DB, all go to discovery
            still_unmatched = with_emb.collect()
            if len(still_unmatched) >= MIN_CLUSTER_SIZE:
                discover_new_trends(still_unmatched, spark)
    
    print(f"📊 Batch {batch_id} complete: {matched_count} matched")


def update_trend_with_post(trend_name, post, spark, similarity=0.95):
    """Update existing trend with new post"""
    # This is simplified - in production use JDBC write with upsert
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine("postgresql://user:password@localhost:5432/trend_db")
        
        with engine.begin() as conn:
            # Get current trend
            result = conn.execute(text(
                "SELECT id, post_count, representative_posts FROM detected_trends WHERE trend_name = :name"
            ), {"name": trend_name}).fetchone()
            
            if result:
                trend_id, post_count, reps_json = result
                reps = json.loads(reps_json) if reps_json else []
                
                # Add new post
                reps.append({
                    "content": post.content[:300] if post.content else "",
                    "source": post.source or "",
                    "time": post.time or datetime.now().isoformat(),
                    "similarity": float(similarity)
                })
                reps = reps[-100:]  # Keep last 100
                
                conn.execute(text("""
                    UPDATE detected_trends 
                    SET post_count = :count, 
                        representative_posts = :reps,
                        last_updated = :now
                    WHERE id = :id
                """), {
                    "count": post_count + 1,
                    "reps": json.dumps(reps, ensure_ascii=False),
                    "now": datetime.now(),
                    "id": trend_id
                })
    except Exception as e:
        print(f"⚠️ Error updating trend: {e}")


def discover_new_trends(unmatched_posts, spark):
    """Use HDBSCAN to discover new trends from unmatched posts"""
    try:
        import hdbscan
        
        if len(unmatched_posts) < MIN_CLUSTER_SIZE:
            return
        
        print(f"   🔍 HDBSCAN clustering {len(unmatched_posts)} unmatched posts...")
        
        # Extract embeddings
        embeddings = np.array([p.embedding for p in unmatched_posts])
        contents = [p.content for p in unmatched_posts]
        sources = [p.source for p in unmatched_posts]
        times = [p.time for p in unmatched_posts]
        
        # Cluster
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=MIN_CLUSTER_SIZE,
            metric='euclidean',
            cluster_selection_epsilon=0.5
        )
        labels = clusterer.fit_predict(embeddings)
        
        unique_labels = set(labels) - {-1}
        new_trends = 0
        
        from sqlalchemy import create_engine, text
        engine = create_engine("postgresql://user:password@localhost:5432/trend_db")
        
        for label in unique_labels:
            member_idx = [i for i, l in enumerate(labels) if l == label]
            if len(member_idx) < MIN_CLUSTER_SIZE:
                continue
            
            # Get representative post
            cluster_embs = embeddings[member_idx]
            centroid = np.mean(cluster_embs, axis=0)
            
            # Find closest to centroid
            dists = np.dot(cluster_embs, centroid)
            best_idx = member_idx[np.argmax(dists)]
            
            trend_name = "New: " + (contents[best_idx][:40] + "..." if len(contents[best_idx]) > 40 else contents[best_idx])
            
            # Build representative posts
            rep_posts = []
            for i in member_idx[:50]:
                rep_posts.append({
                    "content": contents[i][:300],
                    "source": sources[i],
                    "time": times[i],
                    "similarity": float(dists[list(member_idx).index(i)] if i in member_idx else 0.8)
                })
            
            # Insert new trend
            with engine.begin() as conn:
                exists = conn.execute(text(
                    "SELECT 1 FROM detected_trends WHERE trend_name = :name"
                ), {"name": trend_name}).fetchone()
                
                if not exists:
                    conn.execute(text("""
                        INSERT INTO detected_trends (
                            trend_name, trend_score, volume, post_count,
                            representative_posts, topic_type, created_at, last_updated,
                            embedding
                        ) VALUES (
                            :name, 25.0, :vol, :count,
                            :reps, 'Discovery', :now, :now,
                            :emb
                        )
                    """), {
                        "name": trend_name,
                        "vol": len(member_idx),
                        "count": len(member_idx),
                        "reps": json.dumps(rep_posts, ensure_ascii=False),
                        "now": datetime.now(),
                        "emb": json.dumps(centroid.tolist())
                    })
                    new_trends += 1
        
        print(f"   🌟 Discovered {new_trends} new trends")
        
    except ImportError:
        print("⚠️ HDBSCAN not available, skipping discovery")
    except Exception as e:
        print(f"⚠️ Error in HDBSCAN: {e}")


def main():
    """Main entry point for Spark Structured Streaming consumer"""
    print("🚀 Starting Spark Structured Streaming Consumer...")
    
    # Create Spark Session
    spark = SparkSession.builder \
        .appName("TrendDetectionStreaming") \
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.postgresql:postgresql:42.7.1") \
        .config("spark.sql.streaming.checkpointLocation", CHECKPOINT_DIR) \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print(f"✅ Spark Session created: {spark.sparkContext.applicationId}")
    print(f"📡 Connecting to Kafka: {KAFKA_BOOTSTRAP}")
    
    # Read from Kafka (Throttled for stability in Lite Mode)
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 500) \
        .load()
    
    # Parse JSON
    posts_df = kafka_df \
        .selectExpr("CAST(value AS STRING) as json_str") \
        .select(from_json(col("json_str"), POST_SCHEMA).alias("data")) \
        .select("data.*") \
        .filter(col("content").isNotNull())
    
    print(f"✅ Kafka stream configured, topic: {KAFKA_TOPIC}")
    print(f"⏱️  Batch interval: {BATCH_INTERVAL}")
    
    # Process with foreachBatch for complex logic
    query = posts_df.writeStream \
        .foreachBatch(lambda df, id: process_micro_batch(df, id, spark)) \
        .trigger(processingTime=BATCH_INTERVAL) \
        .start()
    
    print("🎯 Streaming query started. Press Ctrl+C to stop.")
    
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        query.stop()
        spark.stop()
        print("✅ Spark consumer stopped.")


if __name__ == "__main__":
    main()
