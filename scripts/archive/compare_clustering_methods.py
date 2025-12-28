import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
import glob
import json
import re
import time
from src.core.analysis.clustering import cluster_data, extract_cluster_labels

console = Console()

# --- Config ---
MODEL_NAME = 'keepitreal/vietnamese-sbert'
MIN_CLUSTER_SIZE = 5
SAMPLE_SIZE = 1000 # Quick verification

# --- Load Data (Reused) ---
def load_data():
    console.print("[dim]Loading data...[/dim]")
    posts = []
    # Facebook
    fb_files = glob.glob("crawlers/facebook/*.json")
    for f in fb_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                if isinstance(data, list): posts.extend(data)
        except: pass
    # News
    news_files = glob.glob("crawlers/news/**/*.csv", recursive=True)
    for f in news_files:
        try:
            df = pd.read_csv(f)
            posts.extend(df.to_dict('records'))
        except: pass
    return posts[:SAMPLE_SIZE]

def clean_content(text):
    if not isinstance(text, str): return ""
    noise_patterns = [r"May be an image of.*?\n", r"No photo description available.*?\n", r"\+?\d+ others", r"Theanh28.*?\n", r"\d+K likes", r"\d+ comments"]
    cleaned = text
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def compare():
    # 1. Prep Data
    posts = load_data()
    texts = []
    for p in posts:
        src = p.get('source', '')
        if src == 'Facebook' or (isinstance(src, str) and src.startswith('Face:')):
            text = clean_content(p.get('content', ''))
        else:
            text = str(p.get('title', ''))
            if len(text) < 10: text = str(p.get('content', ''))
        texts.append(text[:800])
    
    console.print(f"[bold green]Loaded {len(texts)} documents.[/bold green]")

    # 2. Generate Embeddings (ONCE)
    console.print(f"[bold]Generating Embeddings ({MODEL_NAME})...[/bold]")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 3. Compare Methods
    methods = ['kmeans', 'hdbscan', 'bertopic', 'top2vec']
    results = []

    table = Table(title="Clustering Method Comparison (Metric: Cosine)")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Clusters", justify="right")
    table.add_column("Noise", justify="right")
    table.add_column("Silh (Cos)", justify="right")
    table.add_column("Cohesion", justify="right", style="magenta")
    table.add_column("DB Index", justify="right")
    table.add_column("CH Score", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Dominant Topic Label", style="italic")

    methods = ['kmeans', 'hdbscan', 'bertopic', 'top2vec', 'sahc']
    
    # Import SAHC
    from src.pipeline.pipeline_stages import run_sahc_clustering

    for method in methods:
        console.print(f"\n[bold yellow]>>> Running {method.upper()}...[/bold yellow]")
        start = time.time()
        
        try:
            if method == 'sahc':
                labels = run_sahc_clustering(
                    posts=posts, 
                    post_embeddings=embeddings, 
                    min_cluster_size=MIN_CLUSTER_SIZE, 
                    method='hdbscan', # Internal clusterer
                    n_clusters=15, 
                    post_contents=texts, 
                    epsilon=0.15
                )
            else:
                labels = cluster_data(
                    embeddings, 
                    min_cluster_size=MIN_CLUSTER_SIZE, 
                    method=method, 
                    n_clusters=15 if method == 'kmeans' else None,
                    texts=texts,
                    embedding_model=model,
                    min_cohesion=0.4,
                    coherence_threshold=0.60
                )
            
            elapsed = time.time() - start
            
            # Metrics
            unique = [l for l in set(labels) if l != -1]
            n_clusters = len(unique)
            n_noise = list(labels).count(-1)
            
            # Scores
            sil, db, ch, cohesion = -1.0, -1.0, -1.0, -1.0
            if n_clusters > 0:
                 mask = labels != -1
                 if mask.sum() > n_clusters:
                     if n_clusters > 1:
                        sil = silhouette_score(embeddings[mask], labels[mask], metric='cosine')
                        db = davies_bouldin_score(embeddings[mask], labels[mask])
                        ch = calinski_harabasz_score(embeddings[mask], labels[mask])
                     
                     # Calculate overall cohesion (weighted avg by cluster size)
                     cluster_cohesions = []
                     for label in unique:
                         c_mask = (labels == label)
                         c_embs = embeddings[c_mask]
                         centroid = c_embs.mean(axis=0).reshape(1, -1)
                         sims = cosine_similarity(c_embs, centroid)
                         cluster_cohesions.append(sims.mean())
                     cohesion = np.mean(cluster_cohesions) if cluster_cohesions else 0.0
            
            # Get Labels
            topic_names = extract_cluster_labels(texts, labels, model=model, method="semantic")
            top_label = topic_names.get(0, "N/A") if 0 in topic_names else "N/A"
            if top_label == "N/A" and topic_names:
                top_label = topic_names[list(topic_names.keys())[0]]

            table.add_row(
                method.upper(), 
                str(n_clusters), 
                str(n_noise), 
                f"{sil:.3f}", 
                f"{cohesion:.3f}",
                f"{db:.3f}", 
                f"{ch:.1f}", 
                f"{elapsed:.2f}",
                top_label
            )
        except Exception as e:
            console.print(f"[red]Error running {method}: {e}[/red]")
            table.add_row(method.upper(), "ERR", "-", "-", "-", "-", "-", "-", str(e)[:30])
            

    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    compare()
