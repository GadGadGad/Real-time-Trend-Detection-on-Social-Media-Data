import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from pyspark.sql.functions import from_json, col, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, MapType

# Import logic cũ
# (Thêm đường dẫn gốc để Python tìm thấy module src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.streaming.spark_utils import get_spark_session
from src.utils.text_processing.utils import normalize_url
from src.utils.text_processing.vectorizers import get_embeddings
from src.core.analysis.clustering import cluster_data, extract_cluster_labels
from src.pipeline.trend_scoring import calculate_unified_score

# --- CẤU HÌNH ---
KAFKA_BOOTSTRAP = "localhost:9092"
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Kết nối DB để ghi kết quả
db_engine = create_engine(POSTGRES_URL)

# Load trước Model Embedding (Global variable để tránh load lại mỗi batch)
embedder = None

def get_embedder_model():
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        print("⏳ Loading Embedding Model (First time)...")
        embedder = SentenceTransformer(MODEL_NAME)
    return embedder

# --- XỬ LÝ MICRO-BATCH ---
def process_micro_batch(df_batch, batch_id):
    """
    Hàm này được gọi mỗi khi Spark gom đủ dữ liệu (hoặc hết timeout).
    df_batch: Spark DataFrame chứa dữ liệu của batch hiện tại.
    """
    # 1. Chuyển sang Pandas để xử lý logic phức tạp
    pdf = df_batch.toPandas()
    
    if pdf.empty:
        print(f"💤 Batch {batch_id} empty.")
        return

    print(f"🚀 Processing Batch {batch_id} with {len(pdf)} records...")
    
    # Chuẩn bị dữ liệu cho Core Logic
    # (Map columns từ Kafka JSON sang format của pipeline cũ)
    posts = []
    for _, row in pdf.iterrows():
        try:
            # Parse JSON string từ Kafka value
            item = json.loads(row['value'])
            posts.append(item)
        except: continue

    if not posts: return

    # --- TÁI SỬ DỤNG LOGIC CŨ (MAIN PIPELINE) ---
    
    # A. Text Processing & Embedding
    contents = [p.get('content', '') for p in posts]
    
    # Dùng model đã load global
    model = get_embedder_model()
    embeddings = model.encode(contents, show_progress_bar=False)
    
    # B. Clustering (SAHC / HDBSCAN)
    # Gọi hàm từ src/core/analysis/clustering.py
    # Lưu ý: Streaming data thường ít, nên giảm min_cluster_size
    labels = cluster_data(
        embeddings, 
        min_cluster_size=2,  # Streaming batch nhỏ nên giảm threshold
        epsilon=0.05,
        method='hdbscan'
    )
    
    # C. Scoring & Saving
    # Gom bài viết theo cluster để tính điểm
    clusters = {}
    unique_labels = set(labels)
    
    rows_to_insert = []
    
    for label in unique_labels:
        if label == -1: continue # Bỏ qua nhiễu
        
        # Lấy các bài trong cụm này
        indices = [i for i, x in enumerate(labels) if x == label]
        cluster_posts = [posts[i] for i in indices]
        
        # Đặt tên cluster (đơn giản hóa cho streaming)
        # Nếu muốn xịn hơn thì gọi LLM ở đây (nhưng sẽ chậm)
        # Tạm thời lấy title bài đầu tiên hoặc extract keyword
        cluster_name = cluster_posts[0].get('content', '')[:50].replace('\n', ' ') + "..."
        
        # Tính điểm Trend (src/pipeline/trend_scoring.py)
        # Giả lập trend_data rỗng vì ta đang streaming khám phá (Discovery)
        trend_dummy = {'volume': 0} 
        score, components = calculate_unified_score(trend_dummy, cluster_posts)
        
        # Chuẩn bị row để insert DB
        trend_record = {
            "batch_id": str(batch_id),
            "cluster_label": int(label),
            "trend_name": cluster_name,
            "topic_type": "Discovery" if score < 50 else "Trending", # Ngưỡng tạm
            "category": "T7", # Tạm thời, cần classify sau
            "trend_score": float(score),
            "score_g": components.get('G', 0),
            "score_f": components.get('F', 0),
            "score_n": components.get('N', 0),
            "post_count": len(cluster_posts),
            # Lưu 1 bài mẫu để hiển thị
            "representative_posts": json.dumps([{
                "source": p.get('source'),
                "content": p.get('content')[:100]
            } for p in cluster_posts[:2]]),
            "created_at": datetime.now()
        }
        rows_to_insert.append(trend_record)
        
    # D. Write to PostgreSQL
    if rows_to_insert:
        df_result = pd.DataFrame(rows_to_insert)
        try:
            df_result.to_sql('detected_trends', con=db_engine, if_exists='append', index=False)
            print(f"✅ Batch {batch_id}: Saved {len(rows_to_insert)} trends to DB.")
        except Exception as e:
            print(f"❌ Error writing to DB: {e}")
            
    # E. (Optional) Write raw logs for debugging
    # Có thể ghi raw posts vào bảng raw_logs nếu cần

# --- MAIN STREAMING FLOW ---
if __name__ == "__main__":
    spark = get_spark_session()
    
    # 1. Đọc từ Kafka
    df_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
        .option("subscribe", "raw_data") \
        .option("startingOffsets", "latest") \
        .load()
    
    # 2. Cast Value sang String (JSON)
    df_string = df_stream.selectExpr("CAST(value AS STRING)")
    
    # 3. Trigger Processing
    # Trigger 10 seconds: Gom dữ liệu mỗi 10s xử lý 1 lần
    query = df_string.writeStream \
        .foreachBatch(process_micro_batch) \
        .trigger(processingTime='10 seconds') \
        .start()
        
    print(f"📡 Trend Detection Streaming Job Started... Listening on {KAFKA_BOOTSTRAP}")
    query.awaitTermination()