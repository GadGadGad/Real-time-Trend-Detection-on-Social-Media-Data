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
    # "thành_phố", "tphcm", "hà_nội", "việt_nam", "ngày", "năm", "tháng"
]

def cluster_data(embeddings, min_cluster_size=3, epsilon=0.05, method='hdbscan', n_clusters=15, 
                 texts=None, embedding_model=None, min_cohesion=None, max_cluster_size=100, selection_method='eom'):
    """
    Cluster embeddings using UMAP + HDBSCAN, K-Means, or BERTopic.
    Includes Recursive Sub-Clustering for "Mega Clusters".
    
    selection_method: 'eom' (Excess of Mass - larger clusters) or 'leaf' (Fine-grained clusters).
                      'leaf' is better for separating distinct topics.
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
        try:
            umap_embeddings = umap.UMAP(
                n_neighbors=min(30, len(embeddings)-1), 
                n_components=10, 
                metric='cosine',
                random_state=42
            ).fit_transform(embeddings)
        except Exception:
            # Fallback for very small data
            umap_embeddings = embeddings
        
        console.print(f"[bold cyan]🧩 Running HDBSCAN (min_size={min_cluster_size}, eps={epsilon:.3f}, method={selection_method})...[/bold cyan]")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean', 
            cluster_selection_method=selection_method,
            cluster_selection_epsilon=epsilon 
        )
        labels = clusterer.fit_predict(umap_embeddings)
        
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        num_noise = list(labels).count(-1)
        console.print(f"[green]✅ Found {num_clusters} clusters (with {num_noise} noise points).[/green]")

    # --- RECURSIVE SUB-CLUSTERING ---
    # Only applicable if we have valid labels and a max size constraint
    if max_cluster_size and labels is not None:
        unique_labels = list(set(labels))
        if -1 in unique_labels: unique_labels.remove(-1)
        
        next_label_id = max(unique_labels) + 1 if unique_labels else 0
        recursion_occured = False
        
        for label in unique_labels:
            mask = (labels == label)
            cluster_size = np.sum(mask)
            
            if cluster_size > max_cluster_size:
                console.print(f"[yellow]⚡ Recursive Split: Cluster {label} has {cluster_size} items (>{max_cluster_size}). Re-clustering...[/yellow]")
                
                # Extract subset
                sub_embs = embeddings[mask]
                sub_texts = [texts[i] for i in range(len(texts)) if mask[i]] if texts else None
                
                # Recursive call with stricter parameters
                # Decay epsilon to force splitting
                new_epsilon = max(0.01, epsilon * 0.7)
                
                sub_labels = cluster_data(
                    sub_embs, 
                    min_cluster_size=min_cluster_size, 
                    epsilon=new_epsilon, 
                    method=method, 
                    n_clusters=n_clusters,
                    texts=sub_texts, 
                    embedding_model=embedding_model,
                    min_cohesion=None, # Don't filter cohesion recursively, wait for final pass
                    max_cluster_size=max_cluster_size # Keep enforcing limit
                )
                
                # Remap sub-labels to global space
                sub_unique = set(sub_labels)
                remap_dict = {}
                for sl in sub_unique:
                    if sl == -1:
                        remap_dict[sl] = -1 # Keep noise as noise
                    else:
                        remap_dict[sl] = next_label_id
                        next_label_id += 1
                
                # Apply new labels to original array
                final_sub_labels = np.array([remap_dict[l] for l in sub_labels])
                labels[mask] = final_sub_labels
                recursion_occured = True
        
        if recursion_occured:
            final_count = len(set(labels)) - (1 if -1 in labels else 0)
            console.print(f"[bold green]✨ Recursion Complete. Final cluster count: {final_count}[/bold green]")

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

def filter_cluster_outliers(embeddings, labels, cluster_titles, embedding_model, threshold=0.3, texts=None):
    """
    Refines cluster membership by verifying semantic similarity between each post 
    and its cluster's Refined Title (the Dominant Topic chosen by LLM).
    
    If a post is too dissimilar to the title (similarity < threshold), 
    it is reassigned to the Noise cluster (-1).
    
    Args:
        embeddings: Numpy array of post embeddings shape (N, D)
        labels: Array of cluster labels shape (N,)
        cluster_titles: Dict mapping {cluster_id: "Refined Title String"}
        embedding_model: Model with .encode(text) -> np.array
        threshold: Cosine similarity threshold (default 0.3). Lower for broad topics.
        texts: Optional list of texts for debug logging (must correspond to embeddings indices)
        
    Returns:
        new_labels: Refined label array
        stats: Dictionary with 'outliers_removed' count
    """
    if len(embeddings) != len(labels):
        raise ValueError(f"Embeddings length ({len(embeddings)}) != Labels length ({len(labels)})")

    new_labels = np.array(labels).copy()
    stats = {'outliers_removed': 0, 'clusters_processed': 0}
    
    # Pre-compute title embeddings
    title_embeddings = {}
    valid_clusters = [c for c in cluster_titles.keys() if c != -1]
    
    if not valid_clusters:
        return new_labels, stats

    console.print(f"[cyan]🔄 Semantic Filtering: Validating posts against {len(valid_clusters)} cluster titles...[/cyan]")
    
    # Handle cases where some titles might be None or empty
    safe_titles = []
    safe_clusters = []
    
    for c in valid_clusters:
        # cluster_titles values might be dicts (if refiner returns raw result) or strings
        # Adjust based on expected input. Usually this function expects strings.
        val = cluster_titles[c]
        t_str = val['refined_title'] if isinstance(val, dict) and 'refined_title' in val else val
        
        if t_str and isinstance(t_str, str):
            safe_titles.append(t_str)
            safe_clusters.append(c)

    if not safe_titles:
        return new_labels, stats

    # Batch encode titles
    try:
        t_embs = embedding_model.encode(safe_titles)
        if not isinstance(t_embs, np.ndarray):
            t_embs = np.array(t_embs)
    except AttributeError:
        t_embs = np.array([embedding_model.encode(t) for t in safe_titles])
        
    for i, cid in enumerate(safe_clusters):
        title_embeddings[cid] = t_embs[i]

    # Iterate through valid clusters
    for cluster_id in safe_clusters:
        if cluster_id not in title_embeddings:
            continue
            
        # Get indices of posts in this cluster
        indices = np.where(labels == cluster_id)[0]
        if len(indices) == 0:
            continue
            
        stats['clusters_processed'] += 1
        
        # Get post embeddings for this cluster
        cluster_post_embs = embeddings[indices]
        
        # Calculate similarity: (N_posts, D) vs (1, D)
        title_vec = title_embeddings[cluster_id].reshape(1, -1)
        sims = cosine_similarity(cluster_post_embs, title_vec).flatten()
        
        # Identify outliers
        outlier_mask = sims < threshold
        outlier_indices = indices[outlier_mask]
        
        if len(outlier_indices) > 0:
            new_labels[outlier_indices] = -1 # Reassign to noise
            stats['outliers_removed'] += len(outlier_indices)
            
            if texts and len(outlier_indices) > 0:
                first_outlier_idx = outlier_indices[0]
                # Check if texts is indexable
                try:
                    # console.print(f"[dim]      Moved outlier to noise: {texts[first_outlier_idx][:50]}...[/dim]")
                    pass
                except: pass

    if stats['outliers_removed'] > 0:
        console.print(f"[green]✅ Semantic Filtering Complete. Removed {stats['outliers_removed']} outliers.[/green]")
    else:
        console.print(f"[dim]Semantic Filtering: No outliers found.[/dim]")
        
    return new_labels, stats

def recluster_noise(embeddings, labels, min_cluster_size=3, epsilon=0.1):
    """
    Attempts to re-cluster items labeled as noise (-1) to identify missed micro-clusters.
    Useful after "Semantic Filtering" to recover valid topics from rejected outliers.
    
    Args:
        embeddings: Full embeddings array
        labels: Current labels array
        min_cluster_size: Stricter than global default (e.g., 3)
        epsilon: Stricter than global default (e.g., 0.1)
        
    Returns:
        updated_labels: Labels array with noise points potentially assigned to new clusters
    """
    import hdbscan
    import umap
    
    updated_labels = np.array(labels).copy()
    noise_mask = (updated_labels == -1)
    noise_indices = np.where(noise_mask)[0]
    
    if len(noise_indices) < min_cluster_size:
        return updated_labels 
        
    console.print(f"[cyan]♻️  Re-clustering {len(noise_indices)} noise/rejected items...[/cyan]")
    
    try:
        noise_embeddings = embeddings[noise_indices]
        
        # Lightweight UMAP for noise subset
        if len(noise_embeddings) > 15:
             noise_umap = umap.UMAP(n_neighbors=min(10, len(noise_embeddings)-1), 
                                  n_components=5, metric='cosine', random_state=42).fit_transform(noise_embeddings)
        else:
             noise_umap = noise_embeddings
             
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1, 
            metric='euclidean',
            cluster_selection_epsilon=epsilon
        )
        sub_labels = clusterer.fit_predict(noise_umap)
        
        unique_sub = set(sub_labels)
        if -1 in unique_sub: unique_sub.remove(-1)
        
        if not unique_sub:
            console.print("[dim]   No new clusters found in noise.[/dim]")
            return updated_labels
            
        # Determine next available cluster ID
        current_max = max(updated_labels) if len(updated_labels) > 0 else 0
        next_id = current_max + 1
        
        count = 0
        for l in unique_sub:
            # Mask for this new cluster in the noise subset
            l_mask = (sub_labels == l)
            
            # Find corresponding indices in original array
            target_indices = noise_indices[l_mask]
            
            # Assign new global ID
            global_id = next_id + count
            updated_labels[target_indices] = global_id
            
            count += 1
            
        console.print(f"[green]✅ Recovered {count} new clusters from noise![/green]")
        
    except Exception as e:
        console.print(f"[red]Re-clustering noise failed: {e}[/red]")
        
    return updated_labels

def diagnose_clustering(posts, labels, embeddings, console=None):
    import numpy as np
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity
    
    if console is None:
        from rich.console import Console
        console = Console()

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0
    
    console.print(f"\n[bold]🕵️ Clustering Diagnosis[/bold]")
    console.print(f"   • Clusters: {n_clusters}")
    console.print(f"   • Noise: {n_noise} ({noise_ratio:.1%})")
    
    if len(set(labels)) > 1:
        try:
            sil = silhouette_score(embeddings, labels)
            console.print(f"   • Silhouette Score: {sil:.3f} (Target: >0.1)")
        except: pass

    # 0. Global Similarity Check
    if len(embeddings) > 0:
        sample_size = min(1000, len(embeddings))
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        subset = embeddings[indices]
        sims = cosine_similarity(subset)
        avg_sim = np.mean(sims)
        console.print(f"   • Avg Global Similarity: {avg_sim:.3f} (If >0.8, everything is the same!)")

    # 1. Analyze Noise
    if n_noise > 0:
        console.print(f"\n[bold red]🔊 Noise Samples (-1)[/bold red]")
        noise_idx = [i for i, l in enumerate(labels) if l == -1]
        for i in noise_idx[:5]:
            content = posts[i].get('content', '') if isinstance(posts[i], dict) else str(posts[i])
            console.print(f"   - {content[:100]}...")

    # 2. Analyze Top Clusters
    console.print(f"\n[bold cyan]📦 Top 5 Clusters Analysis[/bold cyan]")
    counts = {l: list(labels).count(l) for l in unique_labels if l != -1}
    top_clusters = sorted(counts.items(), key=lambda x: -x[1])[:5]
    
    for label, size in top_clusters:
        indices = [i for i, l in enumerate(labels) if l == label]
        c_embeddings = embeddings[indices]
        c_posts = [posts[i].get('content', '') if isinstance(posts[i], dict) else str(posts[i]) for i in indices]
        
        # Cohesion
        if len(c_embeddings) > 0:
            center = np.mean(c_embeddings, axis=0).reshape(1, -1)
            sims = cosine_similarity(c_embeddings, center)
            cohesion = np.mean(sims)
        else:
            cohesion = 0
        
        console.print(f"\n   [bold]Cluster {label}[/bold] (n={size}, Cohesion={cohesion:.2f})")
        for p in c_posts[:3]:
             console.print(f"      - {p[:100]}...")
