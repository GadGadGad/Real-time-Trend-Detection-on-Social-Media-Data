import numpy as np
import json
import os
from dateutil import parser
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity

from src.core.analysis.clustering import cluster_data, extract_cluster_labels
from src.core.llm.llm_refiner import LLMRefiner
from src.pipeline.main_pipeline import clean_text, strip_news_source_noise

console = Console()

def run_summarization_stage(post_contents, use_llm, summarize_all, model_name='vit5-large', summary_cache_file="summary_cache.json"):
    """
    Phase 0: Summarization for long posts (Vietnamese).
    Reduces noise for embedding model.
    """
    long_indices_all = [i for i, text in enumerate(post_contents) if len(text) > 500]
    
    if not (summarize_all or long_indices_all):
        return post_contents
        
    post_contents_enriched = post_contents.copy()
    
    # Load Cache
    summary_cache = {}
    if os.path.exists(summary_cache_file):
        try:
            with open(summary_cache_file, 'r', encoding='utf-8') as f:
                summary_cache = json.load(f)
        except: pass

    # Identify what needs processing
    to_process = []
    long_indices_to_process = []
    for i in (range(len(post_contents)) if summarize_all else long_indices_all):
        text = post_contents[i]
        if text in summary_cache:
            post_contents_enriched[i] = summary_cache[text]
        else:
            to_process.append(text)
            long_indices_to_process.append(i)

    if long_indices_to_process:
        console.print(f"   ✂️ [cyan]Summarizing {len(long_indices_to_process)} long/target posts...[/cyan]")
        from src.core.analysis.summarizer import Summarizer
        summarizer = Summarizer(model_name=model_name)
        summaries = summarizer.summarize_batch(to_process)
        
        new_cache_entries = 0
        for i, idx in enumerate(long_indices_to_process):
            summary = summaries[i]
            post_contents_enriched[idx] = summary
            summary_cache[to_process[i]] = summary
            new_cache_entries += 1
            
        # Save Cache
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

def run_sahc_clustering(posts, post_embeddings, min_cluster_size=5, method='hdbscan', n_clusters=15, 
                        post_contents=None, epsilon=0.15, trust_remote_code=False, 
                        custom_stopwords=None, min_member_similarity=0.45, selection_method='leaf',
                        recluster_large=True):
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
        custom_stopwords: optional list of stopwords to merge with defaults
        min_member_similarity: Minimum cosine similarity for cluster membership
        selection_method: HDBSCAN selection method ('leaf' or 'eom')
        recluster_large: If True, re-cluster large mixed clusters to split distinct sub-topics
    """
    # --- SAHC PHASE 1: NEWS-FIRST CLUSTERING ---
    news_indices = [i for i, p in enumerate(posts) if 'Face' not in p.get('source', '')]
    social_indices = [i for i, p in enumerate(posts) if 'Face' in p.get('source', '')]
    
    news_labels = np.full(len(news_indices), -1)
    if len(news_indices) >= min_cluster_size:
        console.print(f"🧩 [cyan]SAHC Phase 1: Clustering {len(news_indices)} News articles ({method}, eps={epsilon})...[/cyan]")
        news_labels = cluster_data(
            post_embeddings[news_indices], 
            min_cluster_size=min_cluster_size, 
            epsilon=epsilon, 
            method=method, 
            n_clusters=n_clusters, 
            texts=[post_contents[i] for i in news_indices] if post_contents else None,
            trust_remote_code=trust_remote_code,
            custom_stopwords=custom_stopwords,
            min_member_similarity=min_member_similarity,
            selection_method=selection_method,
            recluster_large=recluster_large
        )
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
        social_discovery_labels = cluster_data(
            leftover_embs, 
            min_cluster_size=min_cluster_size, 
            method=method, 
            n_clusters=n_clusters, 
            texts=leftover_texts, 
            epsilon=epsilon,
            min_member_similarity=min_member_similarity, # Pass down
            trust_remote_code=trust_remote_code,
            custom_stopwords=custom_stopwords,
            selection_method=selection_method,
            recluster_large=recluster_large
        )
        
        # Shift social labels to avoid collision with news clusters
        max_news_label = max(unique_news_clusters) if unique_news_clusters else -1
        for i, s_idx in enumerate(unattached_social_indices):
            if social_discovery_labels[i] != -1:
                final_labels[s_idx] = social_discovery_labels[i] + max_news_label + 1
                
    return final_labels

def calculate_match_scores(cluster_query, cluster_label, trend_embeddings, trend_keys, trend_queries, embedder, reranker, rerank, threshold, 
                           bm25_index=None, cluster_centroid=None, 
                           use_rrf=False, rrf_k=60, use_prf=False, prf_depth=3,
                           weights={'dense': 0.6, 'sparse': 0.4}):
    """
    Helper to match a cluster to existing trends. 
    Implements ADVANCED IR:
    1. Dense (Centroid): Semantic similarity.
    2. Pseudo Relevance Feedback (PRF): Expands cluster query based on initial top trends.
    3. Sparse (BM25): Keyword similarity.
    4. Fusion (Linear or RRF): Combines signals.
    """
    assigned_trend, topic_type, best_match_score = "Discovery", "Discovery", 0.0

    if len(trend_embeddings) > 0:
        # --- 1. DENSE MATCHING (Centroid-based) ---
        if cluster_centroid is not None:
            c_vec = cluster_centroid.reshape(1, -1)
            dense_sims = cosine_similarity(c_vec, trend_embeddings)[0]
        else:
            cluster_emb = embedder.encode(cluster_query).reshape(1, -1)
            dense_sims = cosine_similarity(cluster_emb, trend_embeddings)[0]
        
        # --- 2. SPARSE MATCHING ---
        sparse_scores = np.zeros(len(trend_keys))
        if bm25_index:
             try:
                # 2a. Initial Query
                query_str = cluster_query.lower()
                
                # 2b. Optional Pseudo Relevance Feedback (PRF)
                if use_prf:
                    # Initial quick dense search to find "relevant" documents (trends)
                    top_dense_ids = np.argsort(dense_sims)[-prf_depth:]
                    expansion_keywords = []
                    for tid in top_dense_ids:
                        # Extract keywords from the trend query/label to expand our search
                        expansion_keywords.extend(trend_queries[tid].lower().split())
                    
                    # Expand query with unique feedback terms
                    unique_feedback = list(set(expansion_keywords))[:10] # limit expansion
                    query_str += " " + " ".join(unique_feedback)

                tokenized_query = query_str.split()
                raw_sparse = np.array(bm25_index.get_scores(tokenized_query))
                
                # [FIX] Normalize to 0-1 scale but DON'T force max to 1.0
                # Use a fixed denominator based on typical BM25 scores
                # BM25 scores can vary widely, so we use sigmoid-like normalization
                if raw_sparse.max() > 0:
                    # Soft normalization: scale to roughly 0-1 without forcing max=1
                    sparse_scores = raw_sparse / (raw_sparse.max() + 5.0)  # +5 prevents max=1
             except Exception:
                 pass

        # --- 3. FUSION ---
        if use_rrf:
            # Reciprocal Rank Fusion: 1 / (k + rank)
            # Ranks: Higher sim = lower rank (1st place = 1)
            dense_ranks = len(dense_sims) - np.argsort(np.argsort(dense_sims)) 
            sparse_ranks = len(sparse_scores) - np.argsort(np.argsort(sparse_scores))
            
            # Combine ranks - RRF naturally produces bounded scores
            rrf_scores = (1.0 / (rrf_k + dense_ranks)) + (1.0 / (rrf_k + sparse_ranks))
            # [FIX] Don't normalize to max=1, keep relative values
            final_scores = rrf_scores
        else:
            # Linear Fusion (Default)
            w_dense = weights.get('dense', 0.6)
            w_sparse = weights.get('sparse', 0.4)
            final_scores = w_dense * dense_sims + w_sparse * sparse_scores

        # Select Top Candidate
        top_idx = np.argsort(final_scores)[-1]
        best_candidate_score = float(final_scores[top_idx])
        
        # [FIX] Also capture the RAW dense similarity for the Semantic Guard
        raw_dense_sim = float(dense_sims[top_idx])
        
        # --- 3. THE SEMANTIC GUARD ---
        # Even if BM25 is high, the semantic floor must be met to avoid "World Cup" mismatches.
        # Trend labels like "World Cup 2026" should have AT LEAST some semantic overlap.
        SEMANTIC_FLOOR = 0.35  # Lowered from 0.4 since we're now using raw dense similarity
        semantic_signal = raw_dense_sim  # Use the raw cosine similarity, not the fused score
        
        is_valid_match = (semantic_signal >= SEMANTIC_FLOOR)
        
        if is_valid_match and best_candidate_score > threshold:
            # High quality match
            assigned_trend = trend_keys[top_idx]
            topic_type = "Trending"
            # [FIX] Return the RAW dense similarity as the match score for interpretability
            best_match_score = raw_dense_sim
            
            # Optional Reranking (Double Check)
            if rerank and reranker:
                try:
                    # We only rerank the winner to verify
                    pair = (cluster_query, trend_queries[top_idx])
                    rerank_score = reranker.predict([pair])[0]
                    # Reranker score is usually logit, -2 is safe
                    if rerank_score < -2.5: 
                         # Reranker rejected the winner!
                         assigned_trend, topic_type, best_match_score = "Discovery", "Discovery", 0.0
                except: pass
        else:
            # Low quality match or under floor: Relegate to Discovery
            assigned_trend = "Discovery"
            topic_type = "Discovery"
            best_match_score = 0.0
            
    return assigned_trend, topic_type, best_match_score
