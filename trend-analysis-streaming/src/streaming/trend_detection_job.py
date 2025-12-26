import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from pyspark.sql.functions import from_json, col, pandas_udf
from pyspark.sql.types import *

# --- THÊM ĐƯỜNG DẪN PROJECT VÀO SYS.PATH ---
# Để Spark tìm thấy các module trong src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.streaming.spark_utils import get_spark_session
from src.core.scoring.trend_scoring import calculate_unified_score
from src.utils.text_processing.vectorizers import get_embeddings

# Import Logic "Xịn" từ file bạn mới upload
from src.pipeline.pipeline_stages import run_sahc_clustering, calculate_match_scores
from src.pipeline.main_pipeline import clean_text

# --- CẤU HÌNH ---
KAFKA_BOOTSTRAP = "localhost:9092"
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Kết nối DB
db_engine = create_engine(POSTGRES_URL)

# Biến Global
embedder = None
trends_cache = {} # Lưu trữ trends để matching

def get_embedder_model():
    """Load model 1 lần duy nhất trên Driver"""
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        print("⏳ Loading Embedding Model...")
        embedder = SentenceTransformer(MODEL_NAME)
    return embedder

def load_active_trends():
    """
    Load danh sách Google Trends để khớp nối. 
    Trong thực tế, nên đọc từ DB hoặc file JSON shared.
    Ở đây demo tôi tạo giả lập hoặc đọc từ file nếu có.
    """
    global trends_cache
    if not trends_cache:
        # TODO: Bạn có thể code thêm đoạn đọc file refined_trends.json tại đây
        # Để demo chạy được ngay, tôi để danh sách rỗng hoặc sample
        trends_cache = {
            "Bão Yagi": {"keywords": ["bão yagi", "siêu bão", "bão số 3"], "volume": 500000},
            "iPhone 16": {"keywords": ["iphone 16", "apple", "ios 18"], "volume": 200000},
            "Giá vàng": {"keywords": ["giá vàng", "sjc", "vàng nhẫn"], "volume": 100000}
        }
    return trends_cache

# --- SPARK UDF: EMBEDDING (Chạy song song trên Executor) ---
@pandas_udf(ArrayType(FloatType()))
def compute_embeddings_udf(contents: pd.Series) -> pd.Series:
    # Mỗi executor tự load model riêng
    from sentence_transformers import SentenceTransformer
    # Cache model trong biến local của worker
    if not hasattr(compute_embeddings_udf, "model"):
        compute_embeddings_udf.model = SentenceTransformer(MODEL_NAME)
    
    embeddings = compute_embeddings_udf.model.encode(contents.tolist(), show_progress_bar=False)
    return pd.Series(embeddings.tolist())

# --- XỬ LÝ MICRO-BATCH (Logic Chính) ---
def process_micro_batch(df_batch, batch_id):
    pdf = df_batch.toPandas()
    if pdf.empty: return

    print(f"🚀 Batch {batch_id}: Processing {len(pdf)} posts...")
    
    # 1. Chuẩn bị dữ liệu (Chuyển đổi từ Spark Row sang List Dict)
    posts = []
    valid_indices = []
    
    for idx, row in pdf.iterrows():
        try:
            item = json.loads(row['value'])
            # Làm sạch text ngay tại đây dùng hàm từ main_pipeline
            item['content'] = clean_text(item.get('content', ''))
            
            # Nếu đã có embedding từ UDF (cột 'embedding'), dùng luôn
            if 'embedding' in row and row['embedding']:
                item['embedding'] = np.array(row['embedding'])
                posts.append(item)
                valid_indices.append(idx)
        except: continue

    if not posts: return

    # 2. Chuẩn bị Embedding Matrix cho Clustering
    # (Lấy từ kết quả UDF để không phải tính lại)
    post_embeddings = np.array([p['embedding'] for p in posts])
    
    # Xóa embedding khỏi dict posts để đỡ tốn RAM khi xử lý tiếp
    for p in posts: 
        if 'embedding' in p: del p['embedding']

    # 3. CHẠY SAHC CLUSTERING (Logic mới từ pipeline_stages.py)
    # Lưu ý: Streaming data ít hơn Batch, nên giảm min_cluster_size
    labels = run_sahc_clustering(
        posts, 
        post_embeddings,
        min_cluster_size=2,  # Giảm xuống 2 cho demo streaming nhanh nhạy
        epsilon=0.15,        # Tăng nhẹ epsilon để dễ gom nhóm hơn
        method='hdbscan'
    )
    
    # 4. KHỚP NỐI TRENDS (Matching Logic mới)
    trends = load_active_trends()
    trend_keys = list(trends.keys())
    
    # Tạo embedding cho Trends (để so sánh Vector)
    model = get_embedder_model()
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_embeddings = model.encode(trend_queries) if trend_queries else np.array([])

    unique_labels = set(labels)
    rows_to_insert = []

    for label in unique_labels:
        if label == -1: continue # Bỏ qua nhiễu
        
        # Lấy bài viết thuộc cụm
        indices = [i for i, x in enumerate(labels) if x == label]
        cluster_posts = [posts[i] for i in indices]
        
        # Tính Centroid của cụm
        cluster_centroid = np.mean(post_embeddings[indices], axis=0)
        
        # Đặt tên tạm cho cụm (Lấy đoạn text dài nhất hoặc title)
        best_content = max(cluster_posts, key=lambda x: len(x.get('content','')))
        cluster_query = best_content.get('title') or best_content.get('content')[:100]

        # Gọi hàm MATCHING XỊN từ pipeline_stages.py
        assigned_trend, topic_type, match_score = calculate_match_scores(
            cluster_query=cluster_query,
            cluster_label=label,
            trend_embeddings=trend_embeddings,
            trend_keys=trend_keys,
            trend_queries=trend_queries,
            embedder=model,
            reranker=None, # Tạm tắt Reranker cho nhanh (Streaming cần tốc độ)
            rerank=False,
            threshold=0.35, # Ngưỡng nhạy
            cluster_centroid=cluster_centroid
        )
        
        # Tính điểm Trend
        trend_data = trends.get(assigned_trend, {'volume': 0})
        unified_score, components = calculate_unified_score(trend_data, cluster_posts)
        
        # Chuẩn bị ghi DB
        # Tạo JSON mẫu tin đại diện an toàn
        rep_posts_data = []
        for p in cluster_posts[:3]:
            rep_posts_data.append({
                "source": str(p.get('source', 'Unknown')),
                "content": str(p.get('content', ''))[:200]
            })

        trend_record = {
            "batch_id": str(batch_id),
            "cluster_label": int(label),
            
            # Nếu khớp trend -> Dùng tên trend. Nếu không -> Dùng tên cụm (Discovery)
            "trend_name": assigned_trend if topic_type == "Trending" else f"New: {cluster_query[:50]}...",
            "topic_type": topic_type,
            "category": "T7", # Tạm thời default, muốn xịn thì gọi TaxonomyClassifier
            
            "trend_score": float(unified_score),
            "score_g": components.get('G', 0),
            "score_f": components.get('F', 0),
            "score_n": components.get('N', 0),
            
            "post_count": len(cluster_posts),
            "representative_posts": json.dumps(rep_posts_data, ensure_ascii=False),
            "created_at": datetime.now()
        }
        rows_to_insert.append(trend_record)

    # 5. Ghi vào PostgreSQL
    if rows_to_insert:
        df_result = pd.DataFrame(rows_to_insert)
        try:
            df_result.to_sql('detected_trends', con=db_engine, if_exists='append', index=False)
            print(f"✅ Batch {batch_id}: Saved {len(rows_to_insert)} trends. (Top: {df_result.iloc[0]['trend_name']})")
        except Exception as e:
            print(f"❌ DB Error: {e}")

# --- MAIN FLOW ---
if __name__ == "__main__":
    spark = get_spark_session()
    
    # 1. Đọc Kafka
    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
        .option("subscribe", "raw_data") \
        .option("startingOffsets", "latest") \
        .load()
    
    # 2. Parse Value (Giả sử JSON thuần trong cột value)
    # Lấy value as String
    df_text = df_stream.selectExpr("CAST(value AS STRING) as value")
    
    # 3. Tính Embeddings song song (UDF)
    # Cần trích xuất content từ chuỗi JSON trước khi embed
    # (Để đơn giản cho Spark SQL, ta dùng Pandas UDF ở bước sau hoặc parse luôn ở đây)
    # Tuy nhiên, để tối ưu, ta đẩy việc parse vào micro-batch Pandas cho linh hoạt.
    
    # Ở đây ta chỉ gom data, việc tính embedding nên làm trong process_batch 
    # HOẶC dùng UDF nếu muốn tận dụng cluster.
    # Cách tốt nhất cho demo single-node: Làm hết trong process_micro_batch (như code trên).
    # Nhưng nếu muốn đúng chuẩn Spark:
    
    # Parse JSON để lấy content
    json_schema = StructType([
        StructField("content", StringType(), True),
        StructField("source", StringType(), True)
    ])
    df_parsed = df_text.withColumn("data", from_json(col("value"), json_schema)).select("value", "data.content")
    
    # Chạy Embedding UDF
    df_embedded = df_parsed.withColumn("embedding", compute_embeddings_udf(col("content")))
    
    # 4. Trigger
    query = df_embedded.writeStream \
        .foreachBatch(process_micro_batch) \
        .trigger(processingTime='10 seconds') \
        .start()
        
    print(f"📡 Trend Detection Streaming Job Started... (SAHC + Hybrid Match Enabled)")
    query.awaitTermination()