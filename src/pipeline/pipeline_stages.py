import os
import json
import csv
import datetime
import hashlib
import numpy as np
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity
from src.core.analysis.summarizer import Summarizer
from src.core.analysis.clustering import cluster_data

console = Console()

def run_summarization_stage(post_contents, use_llm=False, summarize_all=False, model_name='vit5-large'):
    """
    Phase 0: Summarization.
    Summarizes long posts using the Summarizer model if enabled.
    Handles caching and logging to CSV.
    
    Args:
        post_contents: list of strings (post contents)
        use_llm: enable summarization
        summarize_all: if True, summarize all posts; else only > 2500 chars
        model_name: 'vit5-large', 'vit5-base', 'bartpho' or full HuggingFace path
        
    Returns: list of strings (summarized contents where applicable)
    """
    if not use_llm:
        return post_contents

    # If summarizing always, threshold is 0. Else, 2500 chars.
    len_threshold = 0 if summarize_all else 2500
    
    # Load Cache
    summary_cache_file = "summary_cache.json"
    summary_cache = {}
    if os.path.exists(summary_cache_file):
        try:
            with open(summary_cache_file, 'r', encoding='utf-8') as f:
                summary_cache = json.load(f)
            console.print(f"[dim]📦 Loaded {len(summary_cache)} cached summaries[/dim]")
        except: pass

    post_contents_enriched = list(post_contents) # Copy
    def get_hash(t): return hashlib.md5(t.encode()).hexdigest()

    # Identify what actually needs summarization (not in cache)
    long_indices_all = [i for i, t in enumerate(post_contents_enriched) if len(t) > len_threshold]
    long_indices_to_process = []
    
    # Apply cache first
    for idx in long_indices_all:
        txt = post_contents_enriched[idx]
        h = get_hash(txt)
        if h in summary_cache:
            post_contents_enriched[idx] = f"SUMMARY: {summary_cache[h]}"
        else:
            long_indices_to_process.append(idx)
    
    # Process missing entries
    if long_indices_to_process:
        mode_desc = "ALL" if summarize_all else "LONG"
        console.print(f"[cyan]📝 Phase 0: Summarizing {len(long_indices_to_process)} articles ({mode_desc}) - {len(long_indices_all)-len(long_indices_to_process)} cached...[/cyan]")
        
        summ = Summarizer(model_name=model_name)
        summ.load_model()
        
        long_texts = [post_contents_enriched[i] for i in long_indices_to_process]
        summaries = summ.summarize_batch(long_texts)
        summ.unload_model() # Free GPU immediately
        
        # Save to CSV log & Update Cache
        log_file = "summarized_posts_log.csv"
        file_exists = os.path.isfile(log_file)
        
        new_cache_entries = 0
        
        try:
            with open(log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Original Length', 'Summary Length', 'Summary', 'Original Start'])
                
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                for idx, summary in zip(long_indices_to_process, summaries):
                    original = post_contents_enriched[idx]
                    # Cache Key
                    h = get_hash(original)
                    summary_cache[h] = summary
                    new_cache_entries += 1
                    
                    # Update content in place
                    post_contents_enriched[idx] = f"SUMMARY: {summary}"
                    
                    # Log
                    writer.writerow([
                        now, 
                        len(original), 
                        len(summary), 
                        summary, 
                        original[:200].replace('\n', ' ') + "..."
                    ])
        except Exception as e:
            console.print(f"[red]Failed to save summary log: {e}[/red]")

        # Save Cache to Disk
        if new_cache_entries > 0:
            try:
                with open(summary_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(summary_cache, f, ensure_ascii=False)
                console.print(f"   💾 [green]Cached {new_cache_entries} new summaries to {summary_cache_file}[/green]")
            except: pass

        console.print(f"   ✅ [green]Summarized {len(long_indices_to_process)} posts with {model_name}.[/green]")
    elif long_indices_all:
         console.print(f"   ✅ [green]All {len(long_indices_all)} target posts were found in cache![/green]")
         
    return post_contents_enriched

def run_sahc_clustering(posts, post_embeddings, min_cluster_size=5, method='hdbscan', n_clusters=15, post_contents=None, epsilon=0.15, trust_remote_code=False):
    """
    Phase 1-3: SAHC Clustering
    1. Cluster News (High Quality)
    2. Attach Social to News
    3. Cluster Remaining Social
    
    Args:
        method: 'hdbscan', 'kmeans', or 'bertopic'
        n_clusters: number of clusters (K-Means/BERTopic)
        post_contents: list of post text (required for BERTopic)
        epsilon: cluster_selection_epsilon (lower = more clusters)
    """
    # --- SAHC PHASE 1: NEWS-FIRST CLUSTERING ---
    news_indices = [i for i, p in enumerate(posts) if 'Face' not in p.get('source', '')]
    social_indices = [i for i, p in enumerate(posts) if 'Face' in p.get('source', '')]
    
    console.print(f"🧩 [cyan]SAHC Phase 1: Clustering {len(news_indices)} News articles ({method}, eps={epsilon})...[/cyan]")
    news_embs = post_embeddings[news_indices]
    news_texts = [post_contents[i] for i in news_indices] if post_contents else None
    
    if len(news_embs) >= min_cluster_size:
        news_labels = cluster_data(news_embs, min_cluster_size=min_cluster_size, method=method, 
                                   n_clusters=n_clusters, texts=news_texts, epsilon=epsilon,
                                   selection_method='leaf', min_quality_cohesion=0.55,
                                   trust_remote_code=trust_remote_code)
    else:
        news_labels = np.array([-1] * len(news_indices))

    # Initialize final labels
    final_labels = np.array([-1] * len(posts))
    for idx, nl in zip(news_indices, news_labels):
        final_labels[idx] = nl

    # --- SAHC PHASE 2: SOCIAL ATTACHMENT ---
    unique_news_clusters = sorted([l for l in set(news_labels) if l != -1])
    unattached_social_indices = []
    
    if unique_news_clusters and social_indices:
        console.print(f"🔗 [cyan]SAHC Phase 2: Attaching Social posts to News clusters...[/cyan]")
        # Calculate centroids for News clusters
        centroids = {}
        for l in unique_news_clusters:
            cluster_news_indices = [ni for i, ni in enumerate(news_indices) if news_labels[i] == l]
            centroids[l] = np.mean(post_embeddings[cluster_news_indices], axis=0)
        
        centroid_matrix = np.array([centroids[l] for l in unique_news_clusters])
        social_embs = post_embeddings[social_indices]
        
        # Calculate similarity to centroids
        sims = cosine_similarity(social_embs, centroid_matrix)
        
        # Attachment threshold (strict)
        ATTACH_THRESHOLD = 0.65 
        
        for i, s_idx in enumerate(social_indices):
            best_c_idx = np.argmax(sims[i])
            if sims[i][best_c_idx] >= ATTACH_THRESHOLD:
                final_labels[s_idx] = unique_news_clusters[best_c_idx]
            else:
                unattached_social_indices.append(s_idx)
    else:
        unattached_social_indices = social_indices

    # --- SAHC PHASE 3: SOCIAL DISCOVERY (Clustering the leftovers) ---
    if len(unattached_social_indices) >= min_cluster_size:
        # Phase 3: Use TIGHTER epsilon for discovery (heuristic: epsilon / 2 or just passed epsilon?)
        # Let's use passed epsilon for now to give control.
        console.print(f"🔭 [cyan]SAHC Phase 3: Researching Discovery trends in {len(unattached_social_indices)} social posts (eps={epsilon})...[/cyan]")
        leftover_embs = post_embeddings[unattached_social_indices]
        leftover_texts = [post_contents[i] for i in unattached_social_indices] if post_contents else None
        social_discovery_labels = cluster_data(leftover_embs, min_cluster_size=min_cluster_size, 
                                               method=method, n_clusters=n_clusters, texts=leftover_texts, epsilon=epsilon,
                                               trust_remote_code=trust_remote_code)
        
        # Shift social labels to avoid collision with news clusters
        max_news_label = max(unique_news_clusters) if unique_news_clusters else -1
        for i, s_idx in enumerate(unattached_social_indices):
            if social_discovery_labels[i] != -1:
                final_labels[s_idx] = social_discovery_labels[i] + max_news_label + 1
                
    return final_labels

def calculate_match_scores(cluster_query, cluster_label, trend_embeddings, trend_keys, trend_queries, embedder, reranker, rerank, threshold, bm25_index=None):
    """Helper to match a cluster to existing trends. Supports Hybrid Search (Dense + BM25)."""
    assigned_trend, topic_type, best_match_score = "Discovery", "Discovery", 0.0

    if len(trend_embeddings) > 0:
        # Dense Score
        cluster_emb = embedder.encode(cluster_query)
        dense_sims = cosine_similarity([cluster_emb], trend_embeddings)[0]
        
        final_scores = dense_sims
        
        # Sparse Score (BM25) Fusion
        if bm25_index:
             try:
                tokenized_query = cluster_query.lower().split()
                sparse_scores = np.array(bm25_index.get_scores(tokenized_query))
                
                # Normalize BM25 (Min-Max to 0-1 range) to allow fair fusion
                if sparse_scores.max() > 0:
                    sparse_scores = sparse_scores / sparse_scores.max()
                
                # Fusion: 0.5 Dense + 0.5 Sparse
                # Adjustable alpha could be passed in future
                final_scores = 0.5 * dense_sims + 0.5 * sparse_scores
                
                # console.print(f"[dim]Hybrid Fusion debug: Dense={dense_sims.max():.3f}, Sparse={sparse_scores.max():.3f}[/dim]")
             except Exception as e:
                 # Fallback to dense if BM25 fails
                 pass

        # Select Top Candidates based on FINAL (Hybrid) score
        top_idx = np.argsort(final_scores)[-3:][::-1]
        
        if rerank and reranker:
            # Pair (query, candidate)
            pairs = [(cluster_query, trend_queries[k]) for k in top_idx]
            rerank_scores = reranker.predict(pairs)
            best_s = np.argmax(rerank_scores)
            # Reranker score is usually logit, -2 is a heuristic threshold (verify this!)
            if rerank_scores[best_s] > -2: 
                # Return the hybrid score as "match score" for consistency, or reranker score?
                # Usually we return similarity 0-1. Let's return the hybrid score of the winner.
                # But reranker picks the WINNER, so we trust it.
                best_match_score = float(final_scores[top_idx[best_s]])
                assigned_trend = trend_keys[top_idx[best_s]]
                topic_type = "Trending"
        elif final_scores[top_idx[0]] > threshold:
            best_match_score = float(final_scores[top_idx[0]])
            assigned_trend = trend_keys[top_idx[0]]
            topic_type = "Trending"
            
    return assigned_trend, topic_type, best_match_score
