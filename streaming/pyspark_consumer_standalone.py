import sys
import os
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, pandas_udf, when
from pyspark.sql.types import *
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity
import math

# --- CONFIG ---
KAFKA_BOOTSTRAP = "localhost:29092"
KAFKA_TOPIC = "posts-stream"
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
# RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
THRESHOLD = 0.65
RERANK_THRESHOLD = -2.5 # Logit threshold from notebook
MIN_CLUSTER_SIZE = 3
CLUSTER_EPSILON = 0.5 # Adjusted for normalized embeddings
COHERENCE_THRESHOLD = 0.75 # Must match matching THRESHOLD
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"

def calculate_realtime_score(g_vol, interactions, post_count):
    """Aligns with src.pipeline.trend_scoring heuristics"""
    MAX_VOL = 5010000
    g_score = (math.log10(g_vol + 1) / math.log10(MAX_VOL + 1)) * 100 if g_vol > 0 else 0
    g_score = min(100, g_score)

    MAX_INTERACTIONS = 20000
    f_score = (math.log10(interactions + 1) / math.log10(MAX_INTERACTIONS + 1)) * 100
    f_score = min(100, f_score)

    MAX_ARTICLES = 10
    n_score = (math.log10(post_count + 1) / math.log10(MAX_ARTICLES + 1)) * 100 if post_count > 0 else 0
    n_score = min(100, n_score)

    active_sources = 0
    if g_score > 10: active_sources += 1
    if f_score > 10: active_sources += 1
    if n_score > 10: active_sources += 1
    
    synergy_mult = 1.0
    if active_sources == 3: synergy_mult = 1.2
    elif active_sources == 2: synergy_mult = 1.1

    base_score = (0.4 * g_score) + (0.35 * f_score) + (0.25 * n_score)
    final_score = min(100, base_score * synergy_mult)
    return round(final_score, 1), round(g_score, 1), round(f_score, 1), round(n_score, 1)

# --- DB SETUP ---
db_engine = create_engine(POSTGRES_URL)

def init_db():
    """Create detected_trends table with full dashboard schema"""
    with db_engine.connect() as conn:
        # [CRITICAL FIX] Commented out DROP TABLE so it doesn't nuke seeded data on restart
        # conn.execute(text("DROP TABLE IF EXISTS detected_trends")) 
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS detected_trends (
                id SERIAL PRIMARY KEY,
                trend_name TEXT,
                trend_score FLOAT,
                volume INT,
                representative_posts TEXT, -- JSON string of list of dicts
                topic_type TEXT,
                post_count INT,
                score_n FLOAT,
                score_f FLOAT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                summary TEXT,
                category TEXT,
                sentiment TEXT,
                advice_state TEXT,
                advice_business TEXT,
                keywords TEXT, -- JSON string of list of keywords
                embedding TEXT, -- JSON string of vector for matching
                google_vol INT DEFAULT 0,
                interactions INT DEFAULT 0
            );
        """))
        conn.commit()
    print("✅ Database initialized with Dashboard Schema (detected_trends).")

# --- SPARK INIT ---
# [FIX] Unset bad global SPARK_HOME if it points to invalid path
if 'SPARK_HOME' in os.environ and not os.path.exists(os.environ['SPARK_HOME']):
    print(f"⚠️ Unsetting invalid SPARK_HOME: {os.environ['SPARK_HOME']}")
    del os.environ['SPARK_HOME']

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1,org.postgresql:postgresql:42.6.0 pyspark-shell'


def get_spark():
    return SparkSession.builder \
        .appName("StatefulTrendDetection") \
        .getOrCreate()

# --- LAZY MODEL LOADING ---
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"   📥 Executor: Loading Embedding Model '{MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        print("   ✅ Executor: Model Loaded.")
    return _embedding_model

# --- GLOBAL RERANKER (DISABLED for Speed) ---
reranker = None
# try:
#     from sentence_transformers import CrossEncoder
#     print("⏳ Loading Reranker Model (This may take a moment)...")
#     reranker = CrossEncoder(RERANKER_MODEL_NAME, trust_remote_code=True)
#     print("✅ Reranker Loaded.")
# except Exception as e:
#     print(f"⚠️ Reranker Not Loaded: {e}")
#     reranker = None

def load_active_trends():
    """Load all trends from DB into memory (List of dicts)"""
    trends = []
    try:
        with db_engine.connect() as conn:
            # Query detected_trends instead of active_trends
            result = conn.execute(text("SELECT id, trend_name, embedding, post_count, trend_score, keywords, google_vol, interactions FROM detected_trends"))
            for row in result:
                # Handle null keywords safely
                kws = []
                if row[5]:
                    try: kws = json.loads(row[5])
                    except: pass
                    
                # Normalize embedding from DB
                emb = np.array(json.loads(row[2]))
                norm = np.linalg.norm(emb)
                if norm > 0: emb = emb / norm

                trends.append({
                    "id": row[0],
                    "name": row[1],
                    "embedding": emb,
                    "post_count": row[3],
                    "trend_score": row[4],
                    "keywords": kws,
                    "google_vol": row[6] or 0,
                    "interactions": row[7] or 0
                })
    except Exception as e:
        print(f"⚠️ Error loading trends: {e}")
    return trends

def process_stateful_batch(df, batch_id):
    pdf = df.toPandas()
    if pdf.empty: return
    
    print(f"\n⚡ Batch {batch_id}: Processing {len(pdf)} posts...")
    
    # 1. Load State (Snapshot Isolation)
    active_trends = load_active_trends()
    print(f"   📊 Active Anchors: {len(active_trends)} trends loaded from database.")
    trend_embeddings = np.array([t['embedding'] for t in active_trends]) if active_trends else np.empty((0, 768))
    
    # 2. Extract Batch Embeddings (Lazy Computation)
    # Check for missing embeddings
    if 'embedding' not in pdf.columns:
        pdf['embedding'] = None
    
    # Identify missing
    missing_mask = pdf['embedding'].isnull() | pdf['embedding'].apply(lambda x: len(x) == 0 if x is not None else True)
    
    if missing_mask.any():
        missing_count = missing_mask.sum()
        print(f"   🧬 Executor: Computing embeddings for {missing_count} posts (Cache Miss)...")
        
        # Load model lazily
        model = get_embedding_model()
        
        # Compute only for missing
        texts_to_embed = pdf.loc[missing_mask, 'content'].tolist()
        # Assign back and normalize
        new_embeddings = model.encode(texts_to_embed, show_progress_bar=False)
        # Unit-normalize
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        new_embeddings = new_embeddings / np.where(norms > 0, norms, 1.0)
        
        pdf.loc[missing_mask, 'embedding'] = pd.Series(list(new_embeddings), index=pdf.loc[missing_mask].index)
        
    post_embeddings = np.array(pdf['embedding'].tolist())
    
    contents = pdf['content'].tolist()
    sources = pdf['source'].tolist()
    times = pdf['time'].tolist()
    
    print(f"   🧠 Processing {len(contents)} posts (Latest: '{contents[0][:30]}...')")
    
    # 3. Micro-batch Logic
    matches = 0
    new_trends = 0
    
    unmatched_indices = []
    
    # A. Attempt to Match Existing Trends
    if len(active_trends) > 0:
        sims = cosine_similarity(post_embeddings, trend_embeddings) # Shape: (N_posts, M_trends)
        
        # --- RRF FUSION (Notebook Logic) ---
        # 1. Dense Scores (Vector) -> float
        dense_scores = sims # (N_posts, M_trends)
        
        # 2. Sparse Scores (Keyword) -> binary 1.0 or 0.0
        # We need a matrix of the same shape
        sparse_scores = np.zeros_like(dense_scores)
        
        trend_names = [t['name'].replace("New: ", "").lower().strip() for t in active_trends]
        trend_keywords_list = [t.get('keywords', []) for t in active_trends]
        curr_contents_lower = [c.lower() for c in contents]
        
        for j, t_name in enumerate(trend_names):
            kws = trend_keywords_list[j] or []
            check_terms = [t_name] + [k.lower() for k in kws if len(k) > 2]
            
            for i, c_text in enumerate(curr_contents_lower):
                if any(term in c_text for term in check_terms):
                    sparse_scores[i][j] = 1.0
                    
        # 3. Compute RRF Scores
        # RRF = 1/(k + rank_dense) + 1/(k + rank_sparse)
        # Note: Rank 1 is best. scipy argsort returns indices of sorted.
        # We need ranks for each row (post) across columns (trends).
        
        rrf_k = 60
        final_scores = np.zeros_like(dense_scores)
        
        for i in range(len(pdf)):
            # Get ranks for this post against all trends
            # argsort twice gives the rank (0-based, 0 is lowest value)
            # We want 0 to be highest value (Rank 1)
            
            # Dense Rank (Higher is better)
            d_row = dense_scores[i]
            d_ranks = len(d_row) - np.argsort(np.argsort(d_row)) # 1-based rank (1 is best)
            
            # Sparse Rank (Higher is better)
            s_row = sparse_scores[i]
            s_ranks = len(s_row) - np.argsort(np.argsort(s_row))
            
            # Fused Score
            final_scores[i] = (1.0 / (rrf_k + d_ranks)) + (1.0 / (rrf_k + s_ranks))

        # --- RERANKER BATCHING ---
        rerank_candidates = []
        candidate_metadata = [] # stores (post_idx, trend_idx, trend_name, trend_id)

        for i in range(len(pdf)):
            best_idx = np.argmax(final_scores[i])
            dense_sim_of_winner = sims[i][best_idx]
            
            if dense_sim_of_winner >= THRESHOLD:  # Use official threshold (0.65)
                trend_name = active_trends[best_idx]['name']
                trend_id = active_trends[best_idx]['id']
                
                # Construct Smart Query
                trend_kws = active_trends[best_idx].get('keywords', []) or []
                unique_signals = [trend_name]
                for kw in trend_kws:
                    if len(unique_signals) >= 6: break
                    if not any(s.lower() in kw.lower() or kw.lower() in s.lower() for s in unique_signals):
                        unique_signals.append(kw)
                smart_trend_query = " ".join(unique_signals)
                
                rerank_candidates.append((contents[i], smart_trend_query))
                candidate_metadata.append({
                    "post_idx": i,
                    "trend_idx": best_idx,
                    "trend_name": trend_name,
                    "trend_id": trend_id,
                    "similarity": float(dense_sim_of_winner)  # Store similarity score
                })
            else:
                unmatched_indices.append(i)

        # Bulk Rerank
        verified_matches = []
        if rerank_candidates and reranker:
            try:
                # Batch inference is MUCH faster
                rr_scores = reranker.predict(rerank_candidates, batch_size=len(rerank_candidates))
                for idx, score in enumerate(rr_scores):
                    if score >= RERANK_THRESHOLD:
                        verified_matches.append(candidate_metadata[idx])
                    else:
                        unmatched_indices.append(candidate_metadata[idx]['post_idx'])
            except Exception as e:
                print(f"Rerank Error: {e}")
                verified_matches = candidate_metadata
        else:
            verified_matches = candidate_metadata

        # --- DB UPDATE BATCH ---
        # [DEBUG] Print unmatched count before DB operations
        print(f"   📊 Matching Complete: {len(verified_matches)} matched, {len(unmatched_indices)} unmatched (Threshold: {THRESHOLD})")
        with db_engine.begin() as conn:
            for match in verified_matches:
                t_idx = match['trend_idx']
                trend_id = match['trend_id']
                post_idx = match['post_idx']
                
                t_state = active_trends[t_idx]
                
                # Increment Stats
                new_post_count = (t_state.get('post_count', 0) or 0) + 1
                
                # Heuristic: Add interactions for the new post
                added_inter = 10
                new_interactions = (t_state.get('interactions', 0) or 0) + added_inter
                
                # Recalculate Score using official heuristics
                new_score, g_s, f_s, n_s = calculate_realtime_score(
                    t_state.get('google_vol', 0) or 0,
                    new_interactions,
                    new_post_count
                )
                
                # Update Memory State for subsequent posts in same batch
                t_state['post_count'] = new_post_count
                t_state['interactions'] = new_interactions
                t_state['trend_score'] = new_score
                
                # Handle Representative Posts
                current_reps_raw = t_state.get('representative_posts', '[]')
                try:
                    current_reps = json.loads(current_reps_raw) if isinstance(current_reps_raw, str) else current_reps_raw
                except: current_reps = []
                
                new_rep = {
                    "content": contents[post_idx], 
                    "source": sources[post_idx], 
                    "time": times[post_idx],
                    "similarity": match.get('similarity', 0.0)  # Include similarity score
                }
                updated_reps = [new_rep] + (current_reps or [])[:999]
                
                # Update memory state for reps so subsequent matches in same batch don't overwrite
                t_state['representative_posts'] = updated_reps
                
                conn.execute(text("""
                    UPDATE detected_trends 
                    SET volume = :vol,
                        post_count = :count,
                        interactions = :inter,
                        trend_score = :score,
                        score_n = :sn,
                        score_f = :sf,
                        representative_posts = :reps,
                        last_updated = :now 
                    WHERE id = :id
                """), {
                    "vol": (t_state.get('google_vol', 0) or 0) + new_post_count,
                    "count": new_post_count,
                    "inter": new_interactions,
                    "score": new_score,
                    "sn": n_s,
                    "sf": f_s,
                    "reps": json.dumps(updated_reps, ensure_ascii=False),
                    "now": datetime.now(),
                    "id": trend_id
                })
                matches += 1
    else:
        unmatched_indices = list(range(len(pdf)))
        
    # B. Discovery (Clustering) on Unmatched
    if unmatched_indices and len(unmatched_indices) >= MIN_CLUSTER_SIZE:
        unmatched_embs = post_embeddings[unmatched_indices]
        print(f"   🧩 Re-Clustering: Running HDBSCAN on {len(unmatched_embs)} unmatched vector points...")
        try:
            import hdbscan
            # Relaxed clustering for demo visibility
            clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean', cluster_selection_epsilon=CLUSTER_EPSILON)
            labels = clusterer.fit_predict(unmatched_embs)
            unique_labels = set(labels) - {-1}
            
            with db_engine.begin() as conn:
                for label in unique_labels:
                    # Gather cluster members
                    member_indices = [unmatched_indices[j] for j, l in enumerate(labels) if l == label]
                    if not member_indices: continue
                    
                    # Content & Source grouping
                    cluster_contents = [contents[m] for m in member_indices]
                    cluster_sources = [sources[m] for m in member_indices]
                    
                    # Create New Trend (Simplified Naming)
                    # 1. Compute Initial Centroid
                    cluster_embs_array = np.array([post_embeddings[m] for m in member_indices])
                    centroid = np.mean(cluster_embs_array, axis=0).reshape(1, -1)
                    c_norm = np.linalg.norm(centroid)
                    if c_norm > 0: centroid = centroid / c_norm

                    # 2. Individual Member Filter: Remove outliers from this cluster
                    dists = cosine_similarity(cluster_embs_array, centroid).flatten() # Shape (N,)
                    valid_mask = dists >= THRESHOLD
                    
                    # Filter indices
                    filtered_member_indices = [member_indices[i] for i, valid in enumerate(valid_mask) if valid]
                    
                    if len(filtered_member_indices) < MIN_CLUSTER_SIZE:
                        print(f"   ⚠️ Discarding cluster: only {len(filtered_member_indices)} members above threshold {THRESHOLD}")
                        continue
                    
                    # Recalculate Centroid for refined cluster
                    cluster_embs_array = np.array([post_embeddings[m] for m in filtered_member_indices])
                    centroid = np.mean(cluster_embs_array, axis=0).reshape(1, -1)
                    c_norm = np.linalg.norm(centroid)
                    if c_norm > 0: centroid = centroid / c_norm
                    
                    # Final Coherence Check
                    dists = cosine_similarity(cluster_embs_array, centroid).flatten()
                    mean_coherence = np.mean(dists)
                    
                    if mean_coherence < COHERENCE_THRESHOLD:
                        print(f"   ⚠️ Discarding incoherent refined cluster (Coherence: {mean_coherence:.2f} < {COHERENCE_THRESHOLD})")
                        continue

                    best_rep_idx = np.argmax(dists)
                    sample_content = [contents[m] for m in filtered_member_indices][best_rep_idx]
                    
                    new_trend_name = "New: " + (sample_content[:40] + "..." if len(sample_content)>40 else sample_content)
                    
                    # Smart Source Type Detection
                    is_news = any(s.lower() in ['vnexpress', 'tuoitre', 'thanhnien', 'news'] for s in cluster_sources)
                    topic_type = 'News' if is_news else 'Social'
                    
                    score_n = 0.0
                    score_f = 0.0
                    
                    # Representative Posts JSON with Similarity to Centroid
                    rep_posts = []
                    # Compute similarities for ALL members up to 1000
                    all_member_embs = post_embeddings[filtered_member_indices[:1000]]
                    all_member_sims = cosine_similarity(all_member_embs, centroid).flatten()
                    
                    for k in range(len(filtered_member_indices[:1000])):
                         idx = filtered_member_indices[k]
                         rep_posts.append({
                             "content": contents[idx],
                             "source": sources[idx],
                             "time": times[idx],
                             "similarity": float(all_member_sims[k])
                         })
                    
                    # INSERT FULL SCHEMA
                    # Deduplication check
                    exists = conn.execute(text("SELECT id FROM detected_trends WHERE trend_name = :name"), {"name": new_trend_name}).fetchone()
                    if exists: continue
                    
                    # Calculate proper initial score
                    init_inter = len(filtered_member_indices) * 10
                    proper_score, g_s, f_s, n_s = calculate_realtime_score(0, init_inter, len(filtered_member_indices))
                    
                    # INSERT FULL SCHEMA
                    conn.execute(text("""
                        INSERT INTO detected_trends (
                            trend_name, trend_score, volume, post_count, 
                            representative_posts, topic_type, 
                            score_n, score_f, created_at, last_updated, embedding,
                            summary, category, sentiment, google_vol, interactions, keywords
                        )
                        VALUES (:name, :score, :vol, :vol, :rep, :type, :sn, :sf, :now, :now, :emb, :summ, :cat, :sent, 0, :inter, :kws)
                    """), {
                        "name": new_trend_name,
                        "score": proper_score,
                        "vol": len(filtered_member_indices),
                        "rep": json.dumps(rep_posts, ensure_ascii=False),
                        "type": topic_type,
                        "sn": n_s,
                        "sf": f_s,
                        "now": datetime.now(),
                        "emb": json.dumps(centroid.flatten().tolist()),
                        "summ": "Waiting for analysis...",
                        "cat": "Unclassified",
                        "sent": "Neutral",
                        "inter": init_inter,
                        "kws": json.dumps([], ensure_ascii=False)
                    })
                    new_trends += 1
                    print(f"   🌟 DISCOVERED NEW TREND: {new_trend_name} (Size: {len(filtered_member_indices)}, Coherence: {mean_coherence:.2f})")
                    
        except ImportError:
            print("   ⚠️ HDBSCAN missing.")
            
    print(f"   ✅ Summary: Matched {matches} existing | Discovered {new_trends} new | Noise {len(unmatched_indices)}")

if __name__ == "__main__":
    # Ensure DB is ready
    time.sleep(5) 
    init_db()
    
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    
    print("🚀 Starting Stateful Consumer (Dashboard Ready)...")
    
    df_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
        .option("subscribe", KAFKA_TOPIC) \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 100) \
        .load()
        
    json_schema = StructType([
        StructField("content", StringType(), True),
        StructField("source", StringType(), True),
        StructField("time", StringType(), True),
        StructField("embedding", ArrayType(FloatType()), True)  # Pre-computed embedding
    ])
    
    df_parsed = df_raw.select(from_json(col("value").cast("string"), json_schema).alias("data")).select("data.*")
    
    df_embedded = df_parsed # No more UDF transformation
    
    query = df_embedded.writeStream \
        .foreachBatch(process_stateful_batch) \
        .option("checkpointLocation", "checkpoints") \
        .trigger(processingTime='5 seconds') \
        .start()
        
    query.awaitTermination()
