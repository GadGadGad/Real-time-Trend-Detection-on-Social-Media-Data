import numpy as np
from rich.console import Console
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

console = Console()

# Expanded Vietnamese Stopwords to reduce "garbage bin" clustering
VIETNAMESE_STOPWORDS = [
    "ông", "bà", "anh", "chị", "em", "này", "đó", "kia", "là", "và", 
    "một", "hai", "ba", "như", "trên", "dưới", "trong", "ngoài", "với", 
    "cho", "để", "lại", "vừa", "cũng", "đang", "sẽ", "đã", "có", "không",
    "cái", "con", "chiếc", "bài", "vụ", "trận", "đấu", "tại", "vào", 
    "được", "bị", "nêu", "cho_biết", "cho_hay", "về", "của", "từ",
    "thành_phố", "tphcm", "hà_nội", "việt_nam", "ngày", "năm", "tháng"
]

def cluster_data(embeddings, min_cluster_size=5, epsilon=0.15, method='hdbscan', n_clusters=15, 
                 texts=None, embedding_model=None, min_cohesion=None):
    """
    Cluster embeddings using UMAP + HDBSCAN, K-Means, or BERTopic.
    """
    labels = None

    if method == 'kmeans':
        from sklearn.cluster import KMeans
        console.print(f"[bold cyan]🧩 Running K-Means clustering (k={n_clusters})...[/bold cyan]")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)
        console.print(f"[green]✅ Created {n_clusters} clusters (no noise with K-Means).[/green]")
    
    elif method == 'bertopic':
        try:
            from bertopic import BERTopic
            from bertopic.vectorizers import ClassTfidfTransformer
        except ImportError:
            console.print("[red]❌ BERTopic not installed. Run: pip install bertopic[/red]")
            console.print("[yellow]Falling back to K-Means...[/yellow]")
            return cluster_data(embeddings, method='kmeans', n_clusters=n_clusters or 15)
        
        if texts is None:
            console.print("[red]❌ BERTopic requires texts parameter[/red]")
            return cluster_data(embeddings, method='kmeans', n_clusters=n_clusters or 15)
        
        console.print(f"[bold cyan]🧩 Running BERTopic (min_topic_size={min_cluster_size}, n_clusters={n_clusters})...[/bold cyan]")
        
        # 1. Custom Vectorizer to filter generic words
        vectorizer_model = CountVectorizer(stop_words=VIETNAMESE_STOPWORDS)
        
        # 2. Stronger UMAP for better separation (default 5 is too small)
        umap_model = umap.UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            vectorizer_model=vectorizer_model,
            language="multilingual",
            min_topic_size=min_cluster_size,
            nr_topics=n_clusters if n_clusters else "auto", # Forced reduction can cause garbage bins
            verbose=True,
            calculate_probabilities=True
        )
        
        topics, probs = topic_model.fit_transform(texts, embeddings)
        
        num_topics = len(set(topics)) - (1 if -1 in topics else 0)
        num_noise = list(topics).count(-1)
        console.print(f"[green]✅ Found {num_topics} topics (with {num_noise} outliers).[/green]")
        
        # Store topic model for later use
        cluster_data._bertopic_model = topic_model
        labels = np.array(topics)

    elif method == 'top2vec':
        try:
            from top2vec import Top2Vec
        except ImportError:
            console.print("[red]❌ Top2Vec not installed.[/red]")
            return cluster_data(embeddings, method='kmeans', n_clusters=n_clusters or 15)
        
        if texts is None:
            console.print("[red]❌ Top2Vec requires 'texts' parameter[/red]")
            return cluster_data(embeddings, method='kmeans', n_clusters=n_clusters or 15)

        console.print(f"[bold cyan]🧩 Running Top2Vec...[/bold cyan]")
        try:
            # Using speed='fast-learn' for quicker testing
            model = Top2Vec(documents=texts, embedding_model='paraphrase-multilingual-MiniLM-L12-v2', speed='learn', workers=4, min_count=2)
            labels = model.doc_top
            cluster_data._top2vec_model = model
        except Exception as e:
            console.print(f"[red]Error initializing Top2Vec: {e}[/red]")
            return cluster_data(embeddings, method='kmeans', n_clusters=n_clusters or 15)
    
    else: # Default: HDBSCAN
        console.print("[bold cyan]🔮 Running UMAP dimensionality reduction (10D)...[/bold cyan]")
        umap_embeddings = umap.UMAP(
            n_neighbors=30, 
            n_components=10, 
            metric='cosine',
            random_state=42
        ).fit_transform(embeddings)
        
        console.print(f"[bold cyan]🧩 Running HDBSCAN clustering (min_size={min_cluster_size}, eps={epsilon})...[/bold cyan]")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean', 
            cluster_selection_method='eom',
            cluster_selection_epsilon=epsilon 
        )
        labels = clusterer.fit_predict(umap_embeddings)
        
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = list(labels).count(-1)
        console.print(f"[green]✅ Found {num_clusters} clusters (with {num_noise} noise points).[/green]")

    # --- POST-PROCESSING: Cohesion Filtering ---
    if labels is not None and min_cohesion is not None and min_cohesion > 0:
        labels = refine_clusters_by_cohesion(embeddings, labels, threshold=min_cohesion)

    return labels

def refine_clusters_by_cohesion(embeddings, labels, threshold=0.5):
    """
    Remove clusters whose average internal similarity (cohesion) is below a threshold.
    """
    new_labels = labels.copy()
    unique_labels = sorted(list(set(labels)))
    if -1 in unique_labels: unique_labels.remove(-1)
    
    removed_count = 0
    for label in unique_labels:
        mask = (labels == label)
        cluster_embs = embeddings[mask]
        
        # Calculate centroid
        centroid = cluster_embs.mean(axis=0).reshape(1, -1)
        
        # Calculate average similarity to centroid
        sims = cosine_similarity(cluster_embs, centroid)
        avg_sim = sims.mean()
        
        # Individual topic logging for debugging
        # console.print(f"[dim]  - Topic {label}: Cohesion {avg_sim:.3f}[/dim]")
        
        if avg_sim < threshold:
            new_labels[mask] = -1
            removed_count += 1
            # console.print(f"[dim]    🗑️ REMOVED (under {threshold})[/dim]")
            
    if removed_count > 0:
        console.print(f"[yellow]🧹 Cohesion Filter: Removed {removed_count} clusters due to low cohesion (<{threshold}).[/yellow]")
        
    return new_labels

def extract_cluster_labels(texts, labels, model=None, method="semantic", anchors=None):
    """
    Extract labels for clusters with generic word filtering.
    """
    # Group texts by cluster
    cluster_texts = {}
    for text, label in zip(texts, labels):
        if label == -1: continue
        if label not in cluster_texts:
            cluster_texts[label] = []
        cluster_texts[label].append(text)
        
    cluster_names = {}
    if not cluster_texts: return {}

    unique_labels = sorted(cluster_texts.keys())
    cluster_docs = [" ".join(cluster_texts[l]) for l in unique_labels]
    
    # Use custom stopwords here too
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=2000, stop_words=None)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_docs)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return {l: f"Cluster {l}" for l in unique_labels}

    anchor_set = set(anchors) if anchors else set()

    for i, label in enumerate(unique_labels):
        try:
            row = tfidf_matrix[i].toarray().flatten()
            if anchor_set:
                for idx, feat in enumerate(feature_names):
                    if feat in anchor_set: row[idx] *= 5.0 

            # Stronger penalty for generic stopwords in labels
            for idx, feat in enumerate(feature_names):
                if any(sw == feat.lower() for sw in VIETNAMESE_STOPWORDS):
                    row[idx] *= 0.01
                tokens = feat.split()
                if len(tokens) == 1: row[idx] *= 0.3
                if len(feat) < 4: row[idx] *= 0.1

            top_indices = row.argsort()[-20:][::-1] 
            candidates = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            
            if not candidates:
                 cluster_names[label] = f"Cluster {label}"
                 continue

            if method == "tfidf" or model is None:
                 cluster_names[label] = ", ".join(candidates[:2]).title()
                 continue
            
            # Semantic Reranking
            full_text = cluster_docs[i]
            candidate_embeddings = model.encode(candidates)
            doc_embedding = model.encode([full_text[:800]])
            similarities = cosine_similarity(doc_embedding, candidate_embeddings)[0]
            
            final_scores = []
            for idx, cand in enumerate(candidates):
                sim = similarities[idx]
                len_bonus = 1.1 if len(cand.split()) > 1 else 1.0
                anchor_bonus = 1.5 if cand in anchor_set else 1.0
                final_scores.append(sim * len_bonus * anchor_bonus)

            best_idx = np.argmax(final_scores)
            cluster_names[label] = candidates[best_idx].title()
        except:
            cluster_names[label] = f"Cluster {label}"
            
    return cluster_names
