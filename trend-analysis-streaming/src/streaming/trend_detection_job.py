import sys
import os
os.environ['PYSPARK_PYTHON'] = '/home/minsun/miniconda3/envs/CS406/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/minsun/miniconda3/envs/CS406/bin/python'
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from pyspark.sql.functions import from_json, col, pandas_udf
from pyspark.sql.types import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.streaming.spark_utils import get_spark_session
from src.core.scoring.trend_scoring import calculate_unified_score
from src.pipeline.pipeline_stages import run_sahc_clustering, calculate_match_scores
from src.pipeline.main_pipeline import clean_text, strip_news_source_noise

KAFKA_BOOTSTRAP = "localhost:9092"
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TRENDS_PATH = "data/cache/trends.json"

db_engine = create_engine(POSTGRES_URL)
embedder = None
trends_cache = None

def get_embedder_model():
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(MODEL_NAME)
    return embedder

def load_active_trends():
    global trends_cache
    if trends_cache is None:
        if os.path.exists(TRENDS_PATH):
            with open(TRENDS_PATH, 'r', encoding='utf-8') as f:
                trends_cache = json.load(f)
        else:
            trends_cache = {}
    return trends_cache

@pandas_udf(ArrayType(FloatType()))
def compute_embeddings_udf(contents: pd.Series) -> pd.Series:
    from sentence_transformers import SentenceTransformer
    if not hasattr(compute_embeddings_udf, "model"):
        compute_embeddings_udf.model = SentenceTransformer(MODEL_NAME)
    
    embeddings = compute_embeddings_udf.model.encode(contents.tolist(), show_progress_bar=False)
    return pd.Series(embeddings.tolist())

def process_micro_batch(df_batch, batch_id):
    pdf = df_batch.toPandas()
    if pdf.empty: return

    posts = []
    for idx, row in pdf.iterrows():
        item = json.loads(row['value'])
        raw_content = item.get('content', '')
        
        cleaned = clean_text(raw_content)
        cleaned = strip_news_source_noise(cleaned)
        
        item['content'] = cleaned
        if 'embedding' in row and row['embedding']:
            item['embedding'] = np.array(row['embedding'])
            posts.append(item)

    if not posts: return

    post_embeddings = np.array([p['embedding'] for p in posts])
    for p in posts: 
        if 'embedding' in p: del p['embedding']

    labels = run_sahc_clustering(
        posts, 
        post_embeddings,
        min_cluster_size=2,
        epsilon=0.15,
        method='hdbscan'
    )
    
    trends = load_active_trends()
    trend_keys = list(trends.keys())
    model = get_embedder_model()
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_embeddings = model.encode(trend_queries) if trend_queries else np.array([])

    unique_labels = set(labels)
    rows_to_insert = []

    for label in unique_labels:
        if label == -1: continue
        
        indices = [i for i, x in enumerate(labels) if x == label]
        cluster_posts = [posts[i] for i in indices]
        cluster_centroid = np.mean(post_embeddings[indices], axis=0)
        
        best_post = max(cluster_posts, key=lambda x: len(x.get('content','')))
        cluster_query = best_post.get('title') or best_post.get('content')[:100]

        assigned_trend, topic_type, match_score = calculate_match_scores(
            cluster_query=cluster_query,
            cluster_label=label,
            trend_embeddings=trend_embeddings,
            trend_keys=trend_keys,
            trend_queries=trend_queries,
            embedder=model,
            reranker=None,
            rerank=False,
            threshold=0.35,
            cluster_centroid=cluster_centroid
        )
        
        trend_data = trends.get(assigned_trend, {'volume': 0})
        unified_score, components = calculate_unified_score(trend_data, cluster_posts)
        
        rep_posts_data = [{"source": str(p.get('source')), "content": str(p.get('content'))[:200]} for p in cluster_posts[:3]]

        rows_to_insert.append({
            "batch_id": str(batch_id),
            "cluster_label": int(label),
            "trend_name": assigned_trend if topic_type == "Trending" else f"New: {cluster_query[:50]}...",
            "topic_type": topic_type,
            "category": "T7",
            "trend_score": float(unified_score),
            "score_g": components.get('G', 0),
            "score_f": components.get('F', 0),
            "score_n": components.get('N', 0),
            "post_count": len(cluster_posts),
            "representative_posts": json.dumps(rep_posts_data, ensure_ascii=False),
            "created_at": datetime.now()
        })

    if rows_to_insert:
        df_result = pd.DataFrame(rows_to_insert)
        df_result.to_sql('detected_trends', con=db_engine, if_exists='append', index=False)

if __name__ == "__main__":
    spark = get_spark_session()
    
    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
        .option("subscribe", "raw_data") \
        .option("startingOffsets", "latest") \
        .load()
    
    df_text = df_stream.selectExpr("CAST(value AS STRING) as value")
    
    json_schema = StructType([
        StructField("content", StringType(), True),
        StructField("source", StringType(), True)
    ])
    
    df_parsed = df_text.withColumn("data", from_json(col("value"), json_schema)).select("value", "data.content")
    df_embedded = df_parsed.withColumn("embedding", compute_embeddings_udf(col("content")))
    
    query = df_embedded.writeStream \
        .foreachBatch(process_micro_batch) \
        .trigger(processingTime='10 seconds') \
        .start()
        
    query.awaitTermination()