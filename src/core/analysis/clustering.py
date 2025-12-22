import numpy as np
from rich.console import Console
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

console = Console()

def cluster_data(embeddings, min_cluster_size=5, epsilon=0.15, method='hdbscan', n_clusters=15, 
                 texts=None, embedding_model=None):
    """
    Cluster embeddings using UMAP + HDBSCAN, K-Means, or BERTopic.
    
    Args:
        embeddings: numpy array of embeddings
        min_cluster_size: minimum points to form a cluster (HDBSCAN/BERTopic)
        epsilon: cluster_selection_epsilon - higher values reduce noise (HDBSCAN only)
        method: 'hdbscan', 'kmeans', or 'bertopic'
        n_clusters: number of clusters (K-Means only, or max topics for BERTopic)
        texts: original texts (required for BERTopic)
        embedding_model: SentenceTransformer model (for BERTopic)
    
    Use K-Means if your data has even density (k-distance CV < 0.5).
    Use HDBSCAN if your data has uneven density (CV > 0.5).
    Use BERTopic for topic modeling with automatic topic extraction.
    """
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        console.print(f"[bold cyan]🧩 Running K-Means clustering (k={n_clusters})...[/bold cyan]")
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings)
        console.print(f"[green]✅ Created {n_clusters} clusters (no noise with K-Means).[/green]")
        return labels
    
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
        
        console.print(f"[bold cyan]🧩 Running BERTopic clustering (min_topic_size={min_cluster_size})...[/bold cyan]")
        
        # Configure BERTopic for Vietnamese
        # Use pre-computed embeddings to avoid re-encoding
        topic_model = BERTopic(
            embedding_model=embedding_model,  # Can be None if using precomputed
            language="multilingual",
            min_topic_size=min_cluster_size,
            nr_topics=n_clusters if n_clusters else "auto",
            verbose=True,
            calculate_probabilities=True
        )
        
        # Fit with pre-computed embeddings
        topics, probs = topic_model.fit_transform(texts, embeddings)
        
        num_topics = len(set(topics)) - (1 if -1 in topics else 0)
        num_noise = list(topics).count(-1)
        
        console.print(f"[green]✅ Found {num_topics} topics (with {num_noise} outliers).[/green]")
        
        # Show top topics
        topic_info = topic_model.get_topic_info()
        console.print("[dim]Top 5 topics:[/dim]")
        for _, row in topic_info.head(6).iterrows():
            if row['Topic'] != -1:
                console.print(f"  Topic {row['Topic']}: {row['Name'][:60]}... ({row['Count']} docs)")
        
        # Store topic model for later use (e.g., visualization)
        cluster_data._bertopic_model = topic_model
        
        return np.array(topics)

    elif method == 'top2vec':
        try:
            from top2vec import Top2Vec
        except ImportError:
            console.print("[red]❌ Top2Vec not installed. Run: pip install top2vec[sentence_transformers][/red]")
            return cluster_data(embeddings, method='kmeans', n_clusters=n_clusters or 15)
        
        if texts is None:
            console.print("[red]❌ Top2Vec requires 'texts' parameter[/red]")
            return cluster_data(embeddings, method='kmeans', n_clusters=n_clusters or 15)

        console.print(f"[bold cyan]🧩 Running Top2Vec (embedding_model='sentence-transformers')...[/bold cyan]")
        
        # Use a standard sentence-transformer model compatible with Top2Vec
        # If embedding_model is a SentenceTransformer object, we can't pass it directly usually, 
        # Top2Vec expects a string name or specific handling. 
        # We'll default to a multilingual one or 'keepitreal/vietnamese-sbert' if it works, 
        # but Top2Vec often prefers universal-sentence-encoder or distiluse-base-multilingual.
        # Let's try passing the MODEL_NAME if we can, but since this function doesn't receive MODEL_NAME string, 
        # and embedding_model arg is an object... we might just let Top2Vec use its default or 'distiluse-base-multilingual-cased'.
        # For best vietnamese support without headache, let's use 'keepitreal/vietnamese-sbert' string if we can,
        # but safest is 'distiluse-base-multilingual-cased' which satisfies 'sentence-transformers' backend.
        
        # NOTE: Top2Vec takes 'embedding_model' as string.
        try:
            model = Top2Vec(documents=texts, embedding_model='distiluse-base-multilingual-cased', speed='learn', workers=4, min_count=2)
        except Exception as e:
            console.print(f"[red]Error initializing Top2Vec: {e}[/red]")
            return cluster_data(embeddings, method='kmeans', n_clusters=n_clusters or 15)

        # Get topic sizes and labels
        # model.doc_top is the topic index for each doc
        return model.doc_top
    
    # HDBSCAN path
    console.print("[bold cyan]🔮 Running UMAP dimensionality reduction...[/bold cyan]")
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
    return labels

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    
    if not cluster_texts:
        return {}

    unique_labels = sorted(cluster_texts.keys())
    cluster_docs = [" ".join(cluster_texts[l]) for l in unique_labels]
    
    # Improved: Bias toward Bigrams/Trigrams to avoid "New Topic: Trận", "New Topic: Đấu"
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), 
                                 max_features=2000, 
                                 stop_words=None)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cluster_docs)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return {l: f"Cluster {l}" for l in unique_labels}

    anchor_set = set(anchors) if anchors else set()

    for i, label in enumerate(unique_labels):
        try:
            row = tfidf_matrix[i].toarray().flatten()
            
            # Boost anchors
            if anchor_set:
                for idx, feat in enumerate(feature_names):
                    if feat in anchor_set:
                        row[idx] *= 5.0 

            # Penalty for single words (length < 2 tokens or very short chars)
            # This pushes the logic toward descriptive phrases
            for idx, feat in enumerate(feature_names):
                tokens = feat.split()
                if len(tokens) == 1:
                    row[idx] *= 0.3 # Penalize unigrams
                if len(feat) < 4:
                    row[idx] *= 0.1 # Penalize tiny words like "Đấu", "Vụ", "Trận"

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
            
            # Final logic: combined score of TF-IDF (precision) and Similarity (semantics)
            final_scores = []
            for idx, cand in enumerate(candidates):
                sim = similarities[idx]
                # Length bonus
                len_bonus = 1.1 if len(cand.split()) > 1 else 1.0
                anchor_bonus = 1.5 if cand in anchor_set else 1.0
                final_scores.append(sim * len_bonus * anchor_bonus)

            best_idx = np.argmax(final_scores)
            cluster_names[label] = candidates[best_idx].title()
            
        except:
            cluster_names[label] = f"Cluster {label}"
            
    return cluster_names
