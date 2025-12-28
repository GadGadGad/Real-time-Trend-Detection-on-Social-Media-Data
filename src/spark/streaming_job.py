import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import from_json, col, pandas_udf
    from pyspark.sql.types import StructType, StringType, IntegerType, StructField, ArrayType, FloatType
    from typing import Iterator
except ImportError:
    print("PySpark not found. Please install pyspark.")
    sys.exit(1)

# Add project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# Import Core Logic
try:
    # We import db_engine to reuse the connection pool
    from streaming.kafka_consumer import load_active_trends, get_embedding_model, init_db, db_engine
    from sqlalchemy import text
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Import Classifiers
    from src.core.extraction.taxonomy_classifier import TaxonomyClassifier
    from src.core.analysis.sentiment import batch_analyze_sentiment, get_analyzer
except ImportError as e:
    print(f"Error importing core modules: {e}")
    sys.exit(1)

# Global Model Cache for Executors
_model = None
_tax_clf = None
_sent_loaded = False

def get_model_broadcast():
    global _model
    if _model is None:
        print("🔧 Loading Embedding Model on Executor...")
        _model = get_embedding_model()
    return _model

def get_tax_clf_broadcast():
    global _tax_clf
    if _tax_clf is None:
        print("🔧 Loading Taxonomy Model on Executor...")
        _tax_clf = TaxonomyClassifier()
    return _tax_clf

def ensure_sentiment_loaded():
    global _sent_loaded
    if not _sent_loaded:
        print("🔧 Loading Sentiment Model on Executor...")
        get_analyzer()
        _sent_loaded = True

# Helpers defined locally
def update_trend_volume(trend_name, inc):
    try:
        with db_engine.begin() as conn:
            conn.execute(text("""
                UPDATE detected_trends 
                SET post_count = post_count + :inc, 
                    volume = volume + :inc, 
                    last_updated = :now 
                WHERE trend_name = :name
            """), {"inc": inc, "now": datetime.now(), "name": trend_name})
    except Exception as e:
        print(f"Error updating volume: {e}")

def create_new_trend(topic, posts, embedding, category='Unclassified', sentiment='Neutral'):
    try:
        emb_json = json.dumps(embedding.tolist()) if embedding is not None else None
        vol = len(posts)
        reps = json.dumps([p.get('content', '')[:200] for p in posts][:3], ensure_ascii=False)
        
        with db_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO detected_trends 
                (trend_name, trend_score, volume, post_count, interactions, representative_posts, 
                 topic_type, summary, category, sentiment, created_at, last_updated, embedding, google_vol)
                VALUES 
                (:name, :score, :vol, :pc, :inter, :reps, 
                 'Discovery', 'Pending Analysis (AI Verified)', :cat, :sent, :now, :now, :emb, 0)
            """), {
                "name": topic, "score": 10.0 + vol, "vol": vol, "pc": vol, "inter": 0,
                "reps": reps, "now": datetime.now(), "emb": emb_json,
                "cat": category, "sent": sentiment
            })
            print(f"   ✨ Created New Trend: {topic} [{category}/{sentiment}]")
    except Exception as e:
        print(f"Error creating trend {topic}: {e}")

# -----------------------------------------------------------------------------
# Pandas UDFs
# -----------------------------------------------------------------------------
@pandas_udf(ArrayType(FloatType()))
def compute_embeddings_udf(content_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    model = get_model_broadcast()
    for series in content_iter:
        embeddings = model.encode(series.tolist(), show_progress_bar=False)
        yield pd.Series(embeddings.tolist())

@pandas_udf(StringType())
def classify_taxonomy_udf(content_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    clf = get_tax_clf_broadcast()
    for series in content_iter:
        # batch_classify returns [(label, source), ...]
        results = clf.batch_classify(series.tolist())
        yield pd.Series([r[0] for r in results])

@pandas_udf(StringType())
def analyze_sentiment_udf(content_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    ensure_sentiment_loaded()
    for series in content_iter:
        # batch_analyze_sentiment returns ['Positive', ...]
        results = batch_analyze_sentiment(series.tolist())
        yield pd.Series(results)

def start_spark_job():
    print("🚀 Starting Spark Structured Streaming Job (With Pandas UDF & Local Models)...")
    
    spark = SparkSession.builder \
        .appName("SocialTrendStreaming") \
        .config("spark.sql.shuffle.partitions", "2") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
    
    print("🔌 Initializing DB...")
    init_db()
    
    schema = StructType([
        StructField("content", StringType()),
        StructField("source", StringType()),
        StructField("time", StringType()),
        StructField("created_time", StringType()),
        StructField("summary", StringType()),
        StructField("url", StringType()),
        StructField("interaction", IntegerType())
    ])
    
    print("📡 Connecting to Kafka...")
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:29092") \
        .option("subscribe", "posts_stream_v1") \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()
        
    parsed_df = df.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    
    print("⚡ Attaching AI UDFs (Embedding, Taxonomy, Sentiment)...")
    
    # Chain UDFs
    # Note: Chaining python UDFs in Spark is handled by pipelining them in Python worker if possible.
    df_enriched = parsed_df \
        .withColumn("embedding", compute_embeddings_udf(col("content"))) \
        .withColumn("category", classify_taxonomy_udf(col("content"))) \
        .withColumn("sentiment", analyze_sentiment_udf(col("content")))
    
    def batch_processor_func(batch_df, batch_id):
        if batch_df.isEmpty(): return
        
        count = batch_df.count()
        print(f"⚡ Processing Batch {batch_id} with {count} records...")
        
        posts = batch_df.collect()
        active_trends = load_active_trends()
        
        matches_count = 0
        new_trends_count = 0
        new_posts = []
        
        for row in posts:
            p = row.asDict()
            if not p['embedding']: continue
            p_emb = np.array(p['embedding']).reshape(1, -1)
            
            matched = False
            if active_trends:
                trend_embs = np.array([t['embedding'] for t in active_trends])
                if len(trend_embs) > 0:
                     sims = cosine_similarity(p_emb, trend_embs)[0]
                     best_idx = np.argmax(sims)
                     if sims[best_idx] >= 0.65:
                         best_trend = active_trends[best_idx]
                         update_trend_volume(best_trend['trend_name'], 1) 
                         matched = True
                         matches_count += 1
            
            if not matched:
                params = {
                     'content': p['content'],
                     'source': p['source'],
                     'time': p.get('time') or p.get('created_time'),
                     'embedding': p['embedding'],
                     'category': p['category'], # From UDF
                     'sentiment': p['sentiment'] # From UDF
                }
                new_posts.append(params)

        if new_posts:
            # Naive Grouping
            for np_post in new_posts:
                trend_name = f"New: {np_post['content'][:30]}... ({np_post['source']})"
                
                # Pass enriched data
                create_new_trend(
                    trend_name, 
                    [np_post], 
                    np.array(np_post['embedding']),
                    category=np_post['category'],
                    sentiment=np_post['sentiment']
                )
                new_trends_count += 1
                
        print(f"   ✅ Batch {batch_id} Done: {matches_count} Matches | {new_trends_count} New Trends")

    query = df_enriched.writeStream \
        .foreachBatch(batch_processor_func) \
        .trigger(processingTime='5 seconds') \
        .start()
        
    print("▶️ Streaming Query Started. Waiting for data...")
    query.awaitTermination()

if __name__ == "__main__":
    start_spark_job()
