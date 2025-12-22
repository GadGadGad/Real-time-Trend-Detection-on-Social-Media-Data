import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import hdbscan
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from rich.console import Console
from rich.table import Table
import glob
import json
import re
import time
from src.core.analysis.clustering import cluster_data

console = Console()

# --- Config ---
MODEL_NAME = 'keepitreal/vietnamese-sbert'
MIN_CLUSTER_SIZE = 5
SAMPLE_SIZE = 1000 # Keep it fast for comparison

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
    table.add_column("Silhouette (Cos)", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Top Topic (Sample)", style="italic")

    for method in methods:
        console.print(f"\n[bold yellow]>>> Running {method.upper()}...[/bold yellow]")
        start = time.time()
        
        try:
            labels = cluster_data(
                embeddings, 
                min_cluster_size=MIN_CLUSTER_SIZE, 
                method=method, 
                n_clusters=15 if method == 'kmeans' else None,
                texts=texts,
                embedding_model=model # For BERTopic
            )
            elapsed = time.time() - start
            
            # Metrics
            unique = set(labels)
            if -1 in unique: unique.remove(-1)
            n_clusters = len(unique)
            n_noise = list(labels).count(-1)
            
            # Silhouette (only valid if > 1 cluster and < n_samples)
            sil = -1.0
            if n_clusters > 1 and n_clusters < len(texts):
                 # For silhouette, we treat noise (-1) as a distinct cluster or ignore it?
                 # Standard practice: ignore noise points for valid score, or treat as own cluster.
                 # Let's filter out noise for a "fair" purity check of formed clusters
                 mask = labels != -1
                 if mask.sum() > n_clusters: # Ensure enough points
                     sil = silhouette_score(embeddings[mask], labels[mask], metric='cosine')
            
            # Sample Topic
            # Just grab text from Cluster 0
            sample = "N/A"
            if 0 in labels:
                c0_texts = [texts[i] for i, l in enumerate(labels) if l == 0]
                sample = c0_texts[0][:50].replace('\n', ' ') + "..." if c0_texts else "Empty"

            table.add_row(
                method.upper(), 
                str(n_clusters), 
                str(n_noise), 
                f"{sil:.3f}", 
                f"{elapsed:.2f}",
                sample
            )
            
        except Exception as e:
            console.print(f"[red]Failed {method}: {e}[/red]")
            table.add_row(method.upper(), "ERR", "-", "-", "-", str(e)[:50])

    console.print("\n")
    console.print(table)

if __name__ == "__main__":
    compare()
