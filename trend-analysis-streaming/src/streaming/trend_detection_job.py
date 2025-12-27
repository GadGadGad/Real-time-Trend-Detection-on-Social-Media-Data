import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from pyspark.sql.functions import from_json, col, pandas_udf
from pyspark.sql.types import *

# Cấu hình môi trường (Điều chỉnh theo máy của bạn)
os.environ['PYSPARK_PYTHON'] = '/home/minsun/miniconda3/envs/CS406/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/minsun/miniconda3/envs/CS406/bin/python'

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.streaming.spark_utils import get_spark_session
# from src.pipeline.pipeline_stages import run_sahc_clustering, calculate_match_scores
# from src.pipeline.main_pipeline import clean_text, strip_news_source_noise

# AI Analysis Modules
from src.core.analysis.sentiment import analyze_sentiment
from src.core.extraction.taxonomy_classifier import TaxonomyClassifier
from src.core.extraction.ner_extractor import get_unique_entities
from src.core.analysis.summarizer import Summarizer


# --- CONFIGURATION ---
KAFKA_BOOTSTRAP = "localhost:9092"
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TRENDS_PATH = "data/cache/trends.json"

db_engine = create_engine(POSTGRES_URL)
embedder = None
bm25_index = None # Index cho Hybrid Search
trends_cache = None
tax_classifier = None
summarizer_service = None

def get_embedder_model():
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(MODEL_NAME)
    return embedder

def get_ai_tools(model):
    global tax_classifier, summarizer_service
    if tax_classifier is None:
        tax_classifier = TaxonomyClassifier(embedding_model=model)
    if summarizer_service is None:
        summarizer_service = Summarizer(model_name='gemini')
    return tax_classifier, summarizer_service

def load_trends_and_build_bm25():
    """Tải Trends và xây dựng Index BM25 cho Hybrid Search theo logic mới"""
    global trends_cache, bm25_index
    if trends_cache is None:
        if os.path.exists(TRENDS_PATH):
            with open(TRENDS_PATH, 'r', encoding='utf-8') as f:
                trends_cache = json.load(f)
            
            # Xây dựng BM25 Index (giống logic find_matches_hybrid của tác giả)
            try:
                from rank_bm25 import BM25Okapi
                trend_keys = list(trends_cache.keys())
                # Tạo query làm sạch từ keywords của trend
                trend_queries = [" ".join(trends_cache[t]['keywords']) for t in trend_keys]
                tokenized_trends = [doc.lower().split() for doc in trend_queries]
                bm25_index = BM25Okapi(tokenized_trends)
                print(f"✅ BM25 Index built for {len(trend_keys)} trends.")
            except ImportError:
                print("⚠️ rank_bm25 not installed. Hybrid Search will fallback to Dense only.")
        else:
            trends_cache = {}
    return trends_cache, bm25_index

@pandas_udf(ArrayType(FloatType()))
def compute_embeddings_udf(contents: pd.Series) -> pd.Series:
    from sentence_transformers import SentenceTransformer
    if not hasattr(compute_embeddings_udf, "model"):
        compute_embeddings_udf.model = SentenceTransformer(MODEL_NAME)
        print(f"🧮 Đang tính Embedding cho {len(contents)} bài viết... (CACHE MISS)")
    embeddings = compute_embeddings_udf.model.encode(contents.tolist(), show_progress_bar=False)
    return pd.Series(embeddings.tolist())

def process_micro_batch(df_batch, batch_id):
    from src.core.scoring.trend_scoring import calculate_unified_score
    from src.pipeline.pipeline_stages import run_sahc_clustering, calculate_match_scores
    from src.pipeline.main_pipeline import clean_text, strip_news_source_noise
    pdf = df_batch.toPandas()
    if pdf.empty: return

    # 1. TIỀN XỬ LÝ (Sử dụng Logic Denoising mới của tác giả)
    posts = []
    for idx, row in pdf.iterrows():
        try:
            item = json.loads(row['value'])
            raw_content = item.get('content', '')
            
            # ÁP DỤNG: Làm sạch và loại bỏ nhiễu nguồn báo (VTV, VNExpress...)
            cleaned = strip_news_source_noise(clean_text(raw_content))
            
            item['content'] = cleaned
            if 'embedding' in row and row['embedding']:
                item['embedding'] = np.array(row['embedding'])
                posts.append(item)
        except: continue

    if not posts: return

    post_embeddings = np.array([p['embedding'] for p in posts])
    for p in posts: 
        if 'embedding' in p: del p['embedding']

    # 2. CLUSTERING (Sử dụng SAHC với tham số tối ưu mới)
    labels = run_sahc_clustering(
        posts, post_embeddings,
        min_cluster_size=3,
        epsilon=0.05, # Epsilon nhỏ hơn theo logic mới của tác giả để tách cụm mịn hơn
        method='hdbscan',
        selection_method='eom' # Dùng EOM (Excess of Mass) như find_matches_hybrid
    )
    
    # 3. MATCHING & AI ANALYSIS
    trends, bm25 = load_trends_and_build_bm25()
    trend_keys = list(trends.keys())
    model = get_embedder_model()
    tax_clf, sum_svc = get_ai_tools(model)
    
    # Chuẩn bị trend embeddings cho Dense matching
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_embeddings = model.encode(trend_queries) if trend_queries else np.array([])

    unique_labels = set(labels)
    rows_to_insert = []

    for label in unique_labels:
        if label == -1: continue
        
        indices = [i for i, x in enumerate(labels) if x == label]
        cluster_posts = [posts[i] for i in indices]
        cluster_centroid = np.mean(post_embeddings[indices], axis=0)
        
        # Chọn bài viết tiêu biểu
        best_post = max(cluster_posts, key=lambda x: len(x.get('content','')))
        rep_text = best_post.get('content', '')
        cluster_query = best_post.get('title') or rep_text[:100]

        # --- MATCHING HYBRID (Tích hợp BM25 + Semantic Guard) ---
        assigned_trend, topic_type, best_match_score = calculate_match_scores(
            cluster_query=cluster_query,
            cluster_label=label,
            trend_embeddings=trend_embeddings,
            trend_keys=trend_keys,
            trend_queries=trend_queries,
            embedder=model,
            reranker=None,
            rerank=False,
            threshold=0.5, # Ngưỡng cao hơn theo tác giả
            bm25_index=bm25, # <--- MỚI: Truyền BM25 Index vào
            cluster_centroid=cluster_centroid,
            semantic_floor=0.35 # <--- MỚI: Rào chắn ngữ nghĩa
        )
        
        # --- AI ANALYSIS (Sentiment, NER, Summarizer) ---
        sentiment = analyze_sentiment(rep_text)
        entities = list(get_unique_entities(rep_text))
        cat_code, _ = tax_clf.classify(rep_text)
        
        summary_text = ""
        if len(cluster_posts) >= 3:
            try:
                summary_text = sum_svc.summarize_batch([rep_text])[0]
            except: summary_text = ""

        # Scoring
        trend_data = trends.get(assigned_trend, {'volume': 0})
        unified_score, components = calculate_unified_score(trend_data, cluster_posts)
        
        rows_to_insert.append({
            "batch_id": str(batch_id),
            "cluster_label": int(label),
            "trend_name": assigned_trend if topic_type == "Trending" else f"New: {cluster_query[:50]}...",
            "topic_type": topic_type,
            "category": cat_code or "T7",
            "trend_score": float(unified_score),
            "score_g": components.get('G', 0),
            "score_f": components.get('F', 0),
            "score_n": components.get('N', 0),
            "summary": summary_text,
            "sentiment": sentiment,
            "top_entities": json.dumps(entities, ensure_ascii=False),
            "post_count": len(cluster_posts),
            "representative_posts": json.dumps([{"source": p.get('source'), "content": p.get('content')[:200]} for p in cluster_posts[:3]], ensure_ascii=False),
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
        .load()
    
    df_text = df_stream.selectExpr("CAST(value AS STRING) as value")
    
    json_schema = StructType([
        StructField("content", StringType(), True),
        StructField("source", StringType(), True),
        StructField("published_at", StringType(), True)
    ])
    
    df_parsed = df_text.withColumn("data", from_json(col("value"), json_schema)).select("value", "data.content")
    # Sử dụng UDF để tạo embedding ngay trên luồng
    df_embedded = df_parsed.withColumn("embedding", compute_embeddings_udf(col("content")))
    
    query = df_embedded.writeStream \
        .foreachBatch(process_micro_batch) \
        .trigger(processingTime='15 seconds') \
        .start()
        
    query.awaitTermination()