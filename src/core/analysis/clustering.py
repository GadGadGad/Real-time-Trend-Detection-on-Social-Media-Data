import numpy as np
from rich.console import Console
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

console = Console()

from src.utils.text_processing.stopwords import get_stopwords, SAFE_STOPWORDS, JOURNALISTIC_STOPWORDS, RISKY_STOPWORDS

# Legacy support for variable names if they are imported elsewhere
VIETNAMESE_STOPWORDS = list(SAFE_STOPWORDS | JOURNALISTIC_STOPWORDS)

def cluster_data(embeddings, min_cluster_size=3, epsilon=0.05, method='hdbscan', n_clusters=15, 
                 texts=None, embedding_model=None, min_cohesion=None, max_cluster_size=100, 
                 selection_method='leaf', recluster_garbage=False, min_pairwise_sim=0.35,
                 min_quality_cohesion=0.5, min_member_similarity=0.45, 
                 trust_remote_code=False, custom_stopwords=None, recluster_large=True, coherence_threshold=0.70):
    """
    Cluster embeddings using UMAP + HDBSCAN, K-Means, or BERTopic.
    Includes Recursive Sub-Clustering for "Mega Clusters".
    
    Args:
        selection_method: 'eom' (Excess of Mass - larger clusters) or 'leaf' (Fine-grained clusters).
                          'leaf' is better for separating distinct topics.
        recluster_garbage: If True, filter clusters with low pairwise similarity and attempt to 
                           re-cluster them into meaningful micro-clusters.
        min_pairwise_sim: Minimum average pairwise similarity threshold for quality clusters.
                          Clusters below this are considered 'garbage' (default: 0.35).
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
        
        # [ROBUSTNESS] Check length alignment
        if len(texts) != embeddings.shape[0]:
            console.print(f"[red]❌ Shape Mismatch: len(texts)={len(texts)} but embeddings.shape={embeddings.shape}[/red]")
            console.print("[yellow]Check if you filtered posts but kept old embeddings.[/yellow]")
            # Attempt to align if mismatch is small or keep the smallest to avoid crash (risky but better for playground)
            min_len = min(len(texts), embeddings.shape[0])
            texts = list(texts)[:min_len]
            embeddings = embeddings[:min_len]
        
        console.print(f"[bold cyan]🧩 Running BERTopic (min_topic_size={min_cluster_size}, n_clusters={n_clusters})...[/bold cyan]")
        
        # 1. Custom Vectorizer to filter generic words
        all_stopwords = list(VIETNAMESE_STOPWORDS)
        if custom_stopwords:
            all_stopwords.extend(custom_stopwords)
        vectorizer_model = CountVectorizer(stop_words=all_stopwords)
        
        # 2. Stronger UMAP for better separation (default 5 is too small)
        umap_model = umap.UMAP(n_neighbors=15, n_components=10, min_dist=0.0, metric='cosine', random_state=42)
        
        # Prepare embedding model for BERTopic
        if isinstance(embedding_model, str):
             from sentence_transformers import SentenceTransformer
             embedding_model = SentenceTransformer(embedding_model, trust_remote_code=trust_remote_code)

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
            cluster_selection_method=selection_method, # 'leaf' is default for quality
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
            
            # QUALITY-BASED RECURSIVE SPLITTING
            # Split if too big OR if too "messy" (low cohesion)
            should_split = False
            if cluster_size > max_cluster_size:
                should_split = True
            elif cluster_size >= min_cluster_size * 2:
                # Check for "similarity outliers" even if size is okay
                avg_cohesion, member_sims = get_cluster_cohesion(embeddings[mask], return_all=True)
                
                # Split if average is too low OR if we have significant outliers
                # (e.g., if more than 20% of members are below the threshold)
                outlier_ratio = np.sum(member_sims < min_member_similarity) / cluster_size
                
                if avg_cohesion < min_quality_cohesion or outlier_ratio > 0.2:
                    should_split = True
                    reason = "low cohesion" if avg_cohesion < min_quality_cohesion else f"high outliers ({outlier_ratio:.1%})"
                    console.print(f"[yellow]⚖️ Quality Split: Cluster {label} (size {cluster_size}) has {reason}. Splitting...[/yellow]")

            if should_split:
                if not recursion_occured:
                    console.print(f"[yellow]⚡ Running Recursive Quality Split...[/yellow]")
                
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
                    min_cohesion=None, 
                    max_cluster_size=max_cluster_size,
                    min_quality_cohesion=min_quality_cohesion,
                    min_member_similarity=min_member_similarity, # Pass down
                    selection_method=selection_method,
                    custom_stopwords=custom_stopwords
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

    # --- POST-PROCESSING: Member Similarity Pruning (Phase 8) ---
    if labels is not None and min_member_similarity > 0:
        labels = refine_clusters_by_member_similarity(
            embeddings, labels, 
            threshold=min_member_similarity,
            min_cluster_size=min_cluster_size
        )

    # --- POST-PROCESSING: Cohesion Filtering (Cluster-level) ---
    if labels is not None and min_cohesion is not None and min_cohesion > 0:
        labels = refine_clusters_by_cohesion(embeddings, labels, threshold=min_cohesion)

    # --- POST-PROCESSING: Re-cluster Garbage Clusters ---
    if labels is not None and recluster_garbage:
        labels = recluster_garbage_clusters(
            embeddings, labels, 
            min_pairwise_sim=min_pairwise_sim,
            min_cluster_size=min_cluster_size
        )

    # --- POST-PROCESSING: Re-cluster Large/Mixed Clusters ---
    if labels is not None and recluster_large:
        labels = recluster_large_clusters(
            embeddings, labels, 
            texts=texts,
            min_cluster_size=min_cluster_size,
            coherence_threshold=coherence_threshold  # Split if cohesion < 0.60
        )

    return labels

def get_cluster_cohesion(cluster_embs, return_all=False):
    """
    Calculate semantic similarity of cluster members to its centroid.
    
    Args:
        cluster_embs: Embedding matrix for the cluster
        return_all: If True, return (avg_sim, all_member_sims). If False, return avg_sim.
    """
    if len(cluster_embs) <= 1: 
        return (1.0, np.array([1.0])) if return_all else 1.0
        
    centroid = cluster_embs.mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(cluster_embs, centroid).flatten()
    
    if return_all:
        return float(sims.mean()), sims
    return float(sims.mean())

def refine_clusters_by_member_similarity(embeddings, labels, threshold=0.45, min_cluster_size=3):
    """
    Quality Pruning (Phase 8): Remove individual posts from a cluster if they 
    are below a specific similarity to the centroid.
    
    If pruning leaves a cluster with < min_cluster_size members, the whole cluster is dissolved.
    """
    new_labels = labels.copy()
    unique_labels = sorted(list(set(labels)))
    if -1 in unique_labels: unique_labels.remove(-1)
    
    pruned_posts = 0
    dissolved_clusters = 0
    
    for label in unique_labels:
        mask = (labels == label)
        indices = np.where(mask)[0]
        cluster_embs = embeddings[mask]
        
        # Calculate per-post similarity to centroid
        _, member_sims = get_cluster_cohesion(cluster_embs, return_all=True)
        
        # Identify outliers
        outliers_mask = (member_sims < threshold)
        outlier_count = np.sum(outliers_mask)
        
        if outlier_count > 0:
            # Check remaining size
            remaining_size = len(member_sims) - outlier_count
            
            if remaining_size < min_cluster_size:
                # Dissolve entire cluster
                new_labels[mask] = -1
                dissolved_clusters += 1
            else:
                # Prune only outliers
                outlier_indices = indices[outliers_mask]
                new_labels[outlier_indices] = -1
                pruned_posts += outlier_count
                
    if pruned_posts > 0 or dissolved_clusters > 0:
        console.print(f"[yellow]⚖️ Similarity Pruning: Pruned {pruned_posts} outlier posts and dissolved {dissolved_clusters} weak clusters (Threshold: {threshold}).[/yellow]")
        
    return new_labels

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

def recluster_garbage_clusters(embeddings, labels, min_pairwise_sim=0.35, min_cluster_size=3):
    """
    Filter clusters with low pairwise similarity and attempt to re-cluster them.
    
    Args:
        embeddings: Original embedding matrix
        labels: Current cluster labels
        min_pairwise_sim: Clusters below this threshold are considered garbage
        min_cluster_size: Min size for re-clustering
        
    Returns:
        Updated labels with garbage clusters either re-clustered or marked as noise
    """
    from collections import defaultdict
    
    new_labels = labels.copy()
    unique_labels = sorted(list(set(labels)))
    if -1 in unique_labels: unique_labels.remove(-1)
    
    # Step 1: Identify garbage clusters based on pairwise similarity
    garbage_cluster_ids = []
    garbage_indices = []
    
    for label in unique_labels:
        mask = (labels == label)
        cluster_embs = embeddings[mask]
        
        if len(cluster_embs) > 1:
            pairwise = cosine_similarity(cluster_embs)
            avg_sim = np.mean(pairwise[np.triu_indices(len(cluster_embs), k=1)])
        else:
            avg_sim = 1.0
        
        if avg_sim < min_pairwise_sim:
            garbage_cluster_ids.append(label)
            indices = np.where(mask)[0]
            garbage_indices.extend(indices.tolist())
    
    if not garbage_cluster_ids:
        console.print("[green]✅ No garbage clusters found (all clusters have good pairwise similarity).[/green]")
        return new_labels
    
    console.print(f"[yellow]♻️ Garbage Re-clustering: Found {len(garbage_cluster_ids)} low-quality clusters ({len(garbage_indices)} posts).[/yellow]")
    
    # Step 2: Mark garbage clusters as noise first
    for gid in garbage_cluster_ids:
        new_labels[labels == gid] = -1
    
    # Step 3: Attempt to re-cluster garbage embeddings
    if len(garbage_indices) >= min_cluster_size * 2:
        garbage_embs = embeddings[garbage_indices]
        
        try:
            # Re-cluster with tighter parameters
            reclustered = hdbscan.HDBSCAN(
                min_cluster_size=max(3, min_cluster_size - 1),
                min_samples=2,
                metric='euclidean',
                cluster_selection_method='leaf'  # Finer-grained for micro-clusters
            ).fit_predict(garbage_embs)
            
            # Assign new labels (offset from existing max)
            max_label = max(set(new_labels)) if set(new_labels) - {-1} else 0
            next_label = max_label + 1
            
            recovered_count = 0
            label_remap = {}
            
            for i, (orig_idx, new_cluster) in enumerate(zip(garbage_indices, reclustered)):
                if new_cluster != -1:
                    if new_cluster not in label_remap:
                        label_remap[new_cluster] = next_label
                        next_label += 1
                    new_labels[orig_idx] = label_remap[new_cluster]
                    recovered_count += 1
            
            num_new_clusters = len(label_remap)
            console.print(f"[green]✅ Recovered {recovered_count}/{len(garbage_indices)} posts into {num_new_clusters} micro-clusters.[/green]")
            
        except Exception as e:
            console.print(f"[red]⚠️ Re-clustering failed: {e}. Keeping garbage as noise.[/red]")
    else:
        console.print(f"[dim]   Too few garbage posts ({len(garbage_indices)}) for re-clustering. Marked as noise.[/dim]")
    
    return new_labels

def recluster_large_clusters(embeddings, labels, texts=None, 
                              size_percentile=90, min_subclusters=2,
                              min_cluster_size=3, coherence_threshold=0.65):
    """
    Identify and re-cluster large "mega-clusters" that likely contain mixed topics.
    
    The Problem: Large clusters often contain semantically distinct sub-topics
    (e.g., "Sudan conflict" + "Thailand-Cambodia conflict" grouped as "international conflict").
    
    The Fix: For clusters in the top size_percentile, re-run HDBSCAN with stricter settings
    to split them into distinct sub-topics.
    
    Args:
        embeddings: Full embedding matrix
        labels: Current cluster labels
        texts: Optional list of texts (for logging/debugging)
        size_percentile: Only re-cluster clusters larger than this percentile (default: 90th = top 10%)
        min_subclusters: Minimum number of sub-clusters required to accept the split
        min_cluster_size: Minimum sub-cluster size
        coherence_threshold: If cluster coherence is above this, don't split (it's already good)
        
    Returns:
        Updated labels with large clusters potentially split
    """
    new_labels = labels.copy()
    unique_labels = sorted([l for l in set(labels) if l != -1])
    
    if not unique_labels:
        return new_labels
    
    # Calculate cluster sizes
    cluster_sizes = {l: np.sum(labels == l) for l in unique_labels}
    size_values = list(cluster_sizes.values())
    
    if len(size_values) < 3:
        return new_labels  # Not enough clusters to determine outliers
    
    # Find the size threshold (top percentile)
    threshold_size = np.percentile(size_values, size_percentile)
    large_clusters = [l for l, s in cluster_sizes.items() if s >= threshold_size and s >= min_cluster_size * 3]
    
    if not large_clusters:
        return new_labels
    
    console.print(f"[cyan]🔬 Re-Clustering {len(large_clusters)} large clusters (size >= {threshold_size:.0f})...[/cyan]")
    
    # Get next available label ID
    next_label_id = max(unique_labels) + 1
    total_splits = 0
    
    for cluster_id in large_clusters:
        mask = (labels == cluster_id)
        indices = np.where(mask)[0]
        cluster_embs = embeddings[mask]
        cluster_size = len(indices)
        
        # Check current coherence - if already tight, don't split
        current_coherence = get_cluster_cohesion(cluster_embs)
        if current_coherence >= coherence_threshold:
            console.print(f"   [dim]Cluster {cluster_id} (n={cluster_size}): Coherence {current_coherence:.3f} >= {coherence_threshold} - Skipping[/dim]")
            continue
        
        # Re-cluster with stricter settings
        try:
            # Use tighter HDBSCAN params for sub-clustering
            sub_eps = 0.02  # Stricter epsilon
            sub_min_samples = max(2, min_cluster_size // 2)
            
            sub_clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=sub_min_samples,
                metric='euclidean',
                cluster_selection_epsilon=sub_eps,
                cluster_selection_method='leaf'  # Fine-grained
            )
            sub_labels = sub_clusterer.fit_predict(cluster_embs)
            
            # Count sub-clusters (excluding noise)
            sub_unique = set(sub_labels) - {-1}
            num_subs = len(sub_unique)
            
            if num_subs >= min_subclusters:
                # Accept the split
                console.print(f"   [green]✂️ Cluster {cluster_id} (n={cluster_size}, coh={current_coherence:.2f}) -> {num_subs} sub-clusters[/green]")
                
                # Remap sub-labels to global space
                for orig_sub in sub_unique:
                    new_labels[indices[sub_labels == orig_sub]] = next_label_id
                    next_label_id += 1
                
                # Mark noise from sub-clustering
                noise_mask = (sub_labels == -1)
                if np.any(noise_mask):
                    new_labels[indices[noise_mask]] = -1
                    
                total_splits += 1
            else:
                console.print(f"   [dim]Cluster {cluster_id} (n={cluster_size}): Only {num_subs} sub-clusters found - Keeping original[/dim]")
                
        except Exception as e:
            console.print(f"   [yellow]⚠️ Sub-clustering failed for cluster {cluster_id}: {e}[/yellow]")
    
    if total_splits > 0:
        new_count = len(set(new_labels) - {-1})
        console.print(f"[bold green]✨ Large Cluster Re-Clustering: Split {total_splits} mega-clusters. New total: {new_count} clusters.[/bold green]")
    else:
        console.print(f"[dim]   No mega-clusters required splitting.[/dim]")
    
    return new_labels


def extract_cluster_labels(texts, labels, model=None, method="semantic", anchors=None, custom_stopwords=None):
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
                    if feat in anchor_set: row[idx] *= 1.5 

            # Stronger penalty for generic stopwords in labels
            all_stopwords = set(VIETNAMESE_STOPWORDS)
            if custom_stopwords:
                all_stopwords.update(custom_stopwords)
            
            for idx, feat in enumerate(feature_names):
                if any(sw == feat.lower() for sw in all_stopwords):
                    row[idx] *= 0.001 # Aggressive stopword filtering
                
                tokens = feat.split()
                # Heavy penalty for generic single words
                if len(tokens) == 1: 
                    row[idx] *= 0.05
                elif len(tokens) == 2:
                    row[idx] *= 1.2 # Bonus for 2-word phrases
                elif len(tokens) == 3:
                    row[idx] *= 1.5 # Higher bonus for 3-word phrases
                
                if len(feat) < 4: 
                    row[idx] *= 0.05

            top_indices = row.argsort()[-30:][::-1] 
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
