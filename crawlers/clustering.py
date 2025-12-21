import numpy as np
from rich.console import Console
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

console = Console()

def cluster_data(embeddings, min_cluster_size=5):
    """
    Cluster embeddings using UMAP + HDBSCAN.
    
    Args:
        embeddings: numpy array of sentence embeddings
        min_cluster_size: minimum matches to form a cluster
        
    Returns:
        labels: cluster labels for each item (-1 is noise)
    """
    console.print("[bold cyan]🔮 Running UMAP dimensionality reduction...[/bold cyan]")
    # Reduce to 5 dimensions for HDBSCAN (improves performance over raw 768 dims)
    umap_embeddings = umap.UMAP(
        n_neighbors=15, 
        n_components=5, 
        metric='cosine'
    ).fit_transform(embeddings)
    
    console.print(f"[bold cyan]🧩 Running HDBSCAN clustering (min_size={min_cluster_size})...[/bold cyan]")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean', 
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(umap_embeddings)
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = list(labels).count(-1)
    
    console.print(f"[green]✅ Found {num_clusters} clusters (with {num_noise} noise points).[/green]")
    return labels

def extract_cluster_labels(texts, labels):
    """
    Extract top keywords for each cluster using TF-IDF.
    """
    cluster_texts = {}
    for text, label in zip(texts, labels):
        if label == -1: continue
        if label not in cluster_texts:
            cluster_texts[label] = []
        cluster_texts[label].append(text)
        
    cluster_names = {}
    
    for label, texts_in_cluster in cluster_texts.items():
        # Simple TF-IDF to find top words
        try:
            tfidf = TfidfVectorizer(stop_words=['và', 'của', 'là', 'có', 'trong', 'đã', 'ngày', 'theo'], max_features=3)
            tfidf.fit(texts_in_cluster)
            keywords = list(tfidf.vocabulary_.keys())
            cluster_names[label] = ", ".join(keywords)
        except:
            cluster_names[label] = f"Cluster {label}"
            
    return cluster_names
