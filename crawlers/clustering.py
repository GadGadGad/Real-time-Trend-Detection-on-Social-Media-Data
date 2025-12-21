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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_cluster_labels(texts, labels, model=None, method="semantic"):
    """
    Extract labels for clusters.
    Args:
        method: 'semantic' (KeyBERT-style) or 'tfidf' (Frequency-based)
    """
    cluster_texts = {}
    for text, label in zip(texts, labels):
        if label == -1: continue
        if label not in cluster_texts:
            cluster_texts[label] = []
        cluster_texts[label].append(text)
        
    cluster_names = {}
    
    console.print(f"[cyan]🏷️ Generating labels for {len(cluster_texts)} clusters (method={method})...[/cyan]")
    
    for label, texts_in_cluster in cluster_texts.items():
        try:
            # 1. Combine texts to represent the cluster doc
            full_text = " ".join(texts_in_cluster)
            
            # 2. Extract candidate n-grams (phrases)
            vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words=None) 
            vectorizer.fit([full_text])
            candidates = list(vectorizer.vocabulary_.keys())
            
            if not candidates:
                cluster_names[label] = f"Cluster {label}"
                continue
                
            # METHOD 1: Simple Frequency / TF-IDF (Fast)
            if method == "tfidf" or model is None:
                 # Just take top freq words (CountVectorizer already counts them)
                 # Or use TF-IDF if we wanted to penalize common corpus words, 
                 # but here we only have the cluster text.
                 # Let's count occurrence in full_text
                 top_3 = sorted(candidates, key=lambda x: full_text.count(x), reverse=True)[:3]
                 cluster_names[label] = ", ".join(top_3)
                 continue

            # METHOD 2: Semantic (KeyBERT-style)
            # Embed Cluster Centroid
            doc_embedding = model.encode([full_text])
            
            # Embed candidates (Limit to top 20 by frequency for speed)
            top_candidates = sorted(candidates, key=lambda x: full_text.count(x), reverse=True)[:20]
            candidate_embeddings = model.encode(top_candidates)
            
            # Cosine Similarity
            distances = cosine_similarity(doc_embedding, candidate_embeddings)
            
            # Get top 1 keyword
            best_idx = distances.argmax()
            best_keyword = top_candidates[best_idx]
            
            cluster_names[label] = best_keyword.title()
            
        except Exception as e:
            # console.print(f"[red]Labeling error: {e}[/red]")
            cluster_names[label] = f"Cluster {label}"
            
    return cluster_names
