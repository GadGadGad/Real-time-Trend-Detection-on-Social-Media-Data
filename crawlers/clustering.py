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

def extract_cluster_labels(texts, labels, model=None, method="semantic", anchors=None):
    """
    Extract labels for clusters.
    Args:
        method: 'semantic' (KeyBERT-style) or 'tfidf' (Frequency-based)
        anchors: Optional list of important words to prioritize.
    """
    # Group texts by cluster
    cluster_texts = {}
    for text, label in zip(texts, labels):
        if label == -1: continue
        if label not in cluster_texts:
            cluster_texts[label] = []
        cluster_texts[label].append(text)
        
    cluster_names = {}
    
    if not cluster_texts:
        return {}

    console.print(f"[cyan]🏷️ Generating labels for {len(cluster_texts)} clusters (method={method})...[/cyan]")
    
    # Pre-process all clusters into single docs for c-TF-IDF
    unique_labels = sorted(cluster_texts.keys())
    cluster_docs = [" ".join(cluster_texts[l]) for l in unique_labels]
    
    # Calculate c-TF-IDF (TF-IDF where document = cluster)
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), 
                                 max_features=1000, 
                                 stop_words=None)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_docs)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return {l: f"Cluster {l}" for l in unique_labels}

    anchor_set = set(anchors) if anchors else set()

    for i, label in enumerate(unique_labels):
        try:
            # Get top TF-IDF keywords for this cluster
            row = tfidf_matrix[i].toarray().flatten()
            
            # Applying Anchor Guidance Boost
            if anchor_set:
                for idx, feat in enumerate(feature_names):
                    if feat in anchor_set:
                        # Give a major boost to anchor words if they appear in cluster
                        row[idx] *= 5.0 

            top_indices = row.argsort()[-20:][::-1] 
            candidates = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            
            if not candidates:
                 cluster_names[label] = f"Cluster {label}"
                 continue

            full_text = cluster_docs[i]
            
            # METHOD 1: TF-IDF Only
            if method == "tfidf" or model is None:
                 cluster_names[label] = ", ".join(candidates[:3])
                 continue
            
            # METHOD 2: Semantic (Hybrid: c-TF-IDF candidates + Embedding Rerank)
            doc_embedding = model.encode([full_text[:1000]])
            candidate_embeddings = model.encode(candidates)
            
            # Cosine Similarity
            similarities = cosine_similarity(doc_embedding, candidate_embeddings)[0]
            
            # Final Boost for anchors in semantic match too
            for idx, cand in enumerate(candidates):
                if cand in anchor_set:
                    similarities[idx] *= 1.5

            best_idx = similarities.argmax()
            best_keyword = candidates[best_idx]
            
            cluster_names[label] = best_keyword.title()
            
        except Exception as e:
            # console.print(f"[red]Labeling error cluster {label}: {e}[/red]")
            cluster_names[label] = f"Cluster {label}"
            
    return cluster_names
