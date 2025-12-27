"""
Airflow DAG: Pipeline 1 - Batch Clustering (Offline)

Purpose: Run UMAP + HDBSCAN to discover clusters/topics
Schedule: @daily (or @weekly)
Output: Cluster centroids + assignments saved to demo folder

This runs FIRST to create clusters that Pipeline 2 uses.
"""

import os
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
sys.path.insert(0, PROJECT_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# === CONFIG ===
PIPELINE_CONFIG = {
    "model_name": "dangvantuan/vietnamese-document-embedding",
    "threshold": 0.65,
    "coherence_threshold": 0.6,
    "semantic_floor": 0.55,
    "min_cluster_size": 3,
    "use_llm": True,
    "llm_provider": "gemini",
}

OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "demo/demo_data_batch")

default_args = {
    'owner': 'batch_pipeline',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


def load_raw_data(**context):
    """Task 1: Load raw posts from data sources."""
    import glob
    import pandas as pd
    
    # Find CSV files
    csv_patterns = [
        os.path.join(PROJECT_ROOT, "data/*.csv"),
        os.path.join(PROJECT_ROOT, "data/kaggle/*.csv"),
        os.path.join(PROJECT_ROOT, "demo-ready/data/*.csv"),
    ]
    
    all_files = []
    for pattern in csv_patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        raise ValueError("No CSV files found!")
    
    # Load and combine
    dfs = []
    for f in all_files[:5]:  # Limit for demo
        try:
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {len(df)} rows from {os.path.basename(f)}")
        except Exception as e:
            print(f"  Skip {f}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    # Save temp file
    temp_file = os.path.join(PROJECT_ROOT, "output/.batch_raw.parquet")
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    combined.to_parquet(temp_file)
    
    print(f"\n✅ Loaded {len(combined)} total posts")
    context['ti'].xcom_push(key='post_count', value=len(combined))
    return {"posts": len(combined)}


def run_clustering(**context):
    """Task 2: Run UMAP + HDBSCAN clustering."""
    import pandas as pd
    import numpy as np
    
    temp_file = os.path.join(PROJECT_ROOT, "output/.batch_raw.parquet")
    df = pd.read_parquet(temp_file)
    
    print(f"🔬 Running clustering on {len(df)} posts...")
    
    # === STEP 1: Generate embeddings ===
    print("  [1/3] Generating embeddings...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(
        PIPELINE_CONFIG["model_name"],
        trust_remote_code=True
    )
    
    # Get text column
    text_col = 'content' if 'content' in df.columns else df.columns[0]
    texts = df[text_col].fillna('').astype(str).tolist()
    
    embeddings = model.encode(
        texts[:500],  # Limit for memory/speed in demo
        show_progress_bar=True,
        batch_size=32
    )
    
    # === STEP 2: UMAP dimensionality reduction ===
    print("  [2/3] Running UMAP...")
    try:
        from umap import UMAP
        reducer = UMAP(
            n_components=10,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        umap_embeddings = reducer.fit_transform(embeddings)
    except ImportError:
        print("  UMAP not available, using PCA fallback")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=10)
        umap_embeddings = reducer.fit_transform(embeddings)
    
    # === STEP 3: HDBSCAN clustering ===
    print("  [3/3] Running HDBSCAN...")
    try:
        from hdbscan import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=PIPELINE_CONFIG["min_cluster_size"],
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(umap_embeddings)
    except ImportError:
        print("  HDBSCAN not available, using KMeans fallback")
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=20, random_state=42)
        labels = clusterer.fit_predict(umap_embeddings)
    
    # Add cluster labels to df
    df_subset = df.iloc[:len(labels)].copy()
    df_subset['cluster'] = labels
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"\n✅ Found {n_clusters} clusters")
    
    # Calculate cluster centroids
    centroids = {}
    for c in set(labels):
        if c == -1:
            continue
        mask = labels == c
        centroids[c] = embeddings[mask].mean(axis=0)
    
    # Save results
    cluster_file = os.path.join(PROJECT_ROOT, "output/.batch_clusters.parquet")
    df_subset.to_parquet(cluster_file)
    
    # Save centroids
    centroid_file = os.path.join(PROJECT_ROOT, "output/.batch_centroids.npy")
    np.save(centroid_file, np.array(list(centroids.values())))
    
    context['ti'].xcom_push(key='n_clusters', value=n_clusters)
    return {"clusters": n_clusters, "posts": len(df_subset)}


def save_demo_state(**context):
    """Task 3: Save results in demo state format for Pipeline 2."""
    import pandas as pd
    import numpy as np
    import json
    
    cluster_file = os.path.join(PROJECT_ROOT, "output/.batch_clusters.parquet")
    df = pd.read_parquet(cluster_file)
    
    n_clusters = context['ti'].xcom_pull(key='n_clusters', task_ids='run_clustering')
    
    print(f"💾 Saving demo state for {n_clusters} clusters...")
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Generate final_topic names for each cluster
    cluster_names = {}
    for c in df['cluster'].unique():
        if c == -1:
            cluster_names[c] = "Noise"
        else:
            # Get most common words from cluster
            cluster_posts = df[df['cluster'] == c]
            text_col = 'content' if 'content' in df.columns else df.columns[0]
            sample_text = cluster_posts[text_col].iloc[0][:50] if len(cluster_posts) > 0 else f"Cluster_{c}"
            cluster_names[c] = f"Topic: {sample_text}"
    
    df['final_topic'] = df['cluster'].map(cluster_names)
    df['topic_type'] = df['cluster'].apply(lambda x: 'Discovery' if x == -1 else 'Trending' if x < 5 else 'Discovery')
    df['score'] = np.random.uniform(0.6, 0.95, len(df))
    df['category'] = 'T7'
    df['sentiment'] = 'Neutral'
    
    # Save as parquet
    output_file = os.path.join(OUTPUT_FOLDER, "df_results.parquet")
    df.to_parquet(output_file)
    
    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "n_clusters": n_clusters,
        "n_posts": len(df),
        "pipeline": "batch_clustering_dag",
    }
    with open(os.path.join(OUTPUT_FOLDER, "metadata.json"), 'w') as f:
        json.dump(metadata, f)
    
    print(f"✅ Saved to {OUTPUT_FOLDER}")
    print(f"   - {len(df)} posts")
    print(f"   - {n_clusters} clusters")
    print(f"\n🔗 Pipeline 2 (streaming) can now use this state!")
    
    return {"folder": OUTPUT_FOLDER, "posts": len(df)}


# === DAG DEFINITION ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='batch_clustering_pipeline',
        default_args=default_args,
        description='Pipeline 1: Batch clustering (UMAP + HDBSCAN)',
        schedule='0 0 * * *',  # Daily at midnight
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['batch', 'clustering', 'offline', 'pipeline-1'],
    ) as dag:
        
        t1 = PythonOperator(
            task_id='load_raw_data',
            python_callable=load_raw_data,
        )
        
        t2 = PythonOperator(
            task_id='run_clustering',
            python_callable=run_clustering,
        )
        
        t3 = PythonOperator(
            task_id='save_demo_state',
            python_callable=save_demo_state,
        )
        
        t1 >> t2 >> t3


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("🚀 Running Batch Clustering Pipeline...\n")
    print("=" * 60)
    
    class MockTI:
        def __init__(self): self.data = {}
        def xcom_push(self, key, value): self.data[key] = value
        def xcom_pull(self, key, task_ids=None): return self.data.get(key)
    
    ti = MockTI()
    
    print("\n📥 TASK 1: Load Raw Data")
    load_raw_data(ti=ti)
    
    print("\n🔬 TASK 2: Run Clustering")
    run_clustering(ti=ti)
    
    print("\n💾 TASK 3: Save Demo State")
    save_demo_state(ti=ti)
    
    print("\n" + "=" * 60)
    print("✅ Batch Clustering Pipeline Complete!")
    print("=" * 60)
