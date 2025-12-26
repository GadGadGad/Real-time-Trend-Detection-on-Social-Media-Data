import numpy as np
import json
import os
from dateutil import parser
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity

from src.core.analysis.clustering import cluster_data, extract_cluster_labels
from src.core.llm.llm_refiner import LLMRefiner
from src.utils.text_processing.cleaning import clean_text, strip_news_source_noise

console = Console()

def run_summarization_stage(post_contents, use_llm, summarize_all, model_name='vit5-large', summary_cache_file="summary_cache.json"):
    long_indices_all = [i for i, text in enumerate(post_contents) if len(text) > 500]
    
    if not (summarize_all or long_indices_all):
        return post_contents
        
    post_contents_enriched = post_contents.copy()
    summary_cache = {}
    if os.path.exists(summary_cache_file):
        with open(summary_cache_file, 'r', encoding='utf-8') as f:
            summary_cache = json.load(f)

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
        console.print(f"   ✂️ Summarizing {len(long_indices_to_process)} posts...")
        from src.core.analysis.summarizer import Summarizer
        summarizer = Summarizer(model_name=model_name)
        summaries = summarizer.summarize_batch(to_process)
        
        for i, idx in enumerate(long_indices_to_process):
            summary = summaries[i]
            post_contents_enriched[idx] = summary
            summary_cache[to_process[i]] = summary
            
        with open(summary_cache_file, 'w', encoding='utf-8') as f:
            json.dump(summary_cache, f, ensure_ascii=False)
            
    return post_contents_enriched

def run_sahc_clustering(posts, post_embeddings, min_cluster_size=5, method='hdbscan', n_clusters=15, 
                        post_contents=None, epsilon=0.15, trust_remote_code=False, 
                        custom_stopwords=None, min_member_similarity=0.60, selection_method='leaf',
                        recluster_large=True, coherence_threshold=0.60):
    news_indices = [i for i, p in enumerate(posts) if 'Face' not in p.get('source', '')]
    social_indices = [i for i, p in enumerate(posts) if 'Face' in p.get('source', '')]
    
    news_labels = np.full(len(news_indices), -1)
    if len(news_indices) >= min_cluster_size:
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
            recluster_large=recluster_large,
            coherence_threshold=coherence_threshold
        )

    final_labels = np.array([-1] * len(posts))
    for idx, nl in zip(news_indices, news_labels):
        final_labels[idx] = nl

    unique_news_clusters = sorted([l for l in set(news_labels) if l != -1])
    unattached_social_indices = []
    
    if unique_news_clusters and social_indices:
        centroids = {}
        for l in unique_news_clusters:
            cluster_news_indices = [ni for i, ni in enumerate(news_indices) if news_labels[i] == l]
            centroids[l] = np.mean(post_embeddings[cluster_news_indices], axis=0)
        
        centroid_matrix = np.array([centroids[l] for l in unique_news_clusters])
        social_embs = post_embeddings[social_indices]
        sims = cosine_similarity(social_embs, centroid_matrix)
        
        ATTACH_THRESHOLD = 0.65 
        for i, s_idx in enumerate(social_indices):
            best_c_idx = np.argmax(sims[i])
            if sims[i][best_c_idx] >= ATTACH_THRESHOLD:
                final_labels[s_idx] = unique_news_clusters[best_c_idx]
            else:
                unattached_social_indices.append(s_idx)
    else:
        unattached_social_indices = social_indices

    if len(unattached_social_indices) >= min_cluster_size:
        leftover_embs = post_embeddings[unattached_social_indices]
        leftover_texts = [post_contents[i] for i in unattached_social_indices] if post_contents else None
        social_discovery_labels = cluster_data(
            leftover_embs, 
            min_cluster_size=min_cluster_size, 
            method=method, 
            n_clusters=n_clusters, 
            texts=leftover_texts, 
            epsilon=epsilon,
            min_member_similarity=min_member_similarity,
            trust_remote_code=trust_remote_code,
            custom_stopwords=custom_stopwords,
            selection_method=selection_method,
            recluster_large=recluster_large,
            coherence_threshold=coherence_threshold
        )
        
        max_news_label = max(unique_news_clusters) if unique_news_clusters else -1
        for i, s_idx in enumerate(unattached_social_indices):
            if social_discovery_labels[i] != -1:
                final_labels[s_idx] = social_discovery_labels[i] + max_news_label + 1
                
    return final_labels

def calculate_match_scores(cluster_query, cluster_label, trend_embeddings, trend_keys, trend_queries, embedder, reranker, rerank, threshold, 
                           bm25_index=None, cluster_centroid=None, 
                           use_rrf=False, rrf_k=60, use_prf=False, prf_depth=3,
                           weights={'dense': 0.6, 'sparse': 0.4},
                           semantic_floor=0.35):
    assigned_trend, topic_type, best_match_score = "Discovery", "Discovery", 0.0

    if len(trend_embeddings) > 0:
        if cluster_centroid is not None:
            c_vec = cluster_centroid.reshape(1, -1)
            dense_sims = cosine_similarity(c_vec, trend_embeddings)[0]
        else:
            cluster_emb = embedder.encode(cluster_query).reshape(1, -1)
            dense_sims = cosine_similarity(cluster_emb, trend_embeddings)[0]
        
        sparse_scores = np.zeros(len(trend_keys))
        if bm25_index:
            query_str = cluster_query.lower()
            if use_prf:
                top_dense_ids = np.argsort(dense_sims)[-prf_depth:]
                expansion_keywords = []
                for tid in top_dense_ids:
                    expansion_keywords.extend(trend_queries[tid].lower().split())
                query_str += " " + " ".join(list(set(expansion_keywords))[:10])

            tokenized_query = query_str.split()
            raw_sparse = np.array(bm25_index.get_scores(tokenized_query))
            if raw_sparse.max() > 0:
                sparse_scores = raw_sparse / (raw_sparse.max() + 5.0)

        if use_rrf:
            dense_ranks = len(dense_sims) - np.argsort(np.argsort(dense_sims)) 
            sparse_ranks = len(sparse_scores) - np.argsort(np.argsort(sparse_scores))
            rrf_scores = (1.0 / (rrf_k + dense_ranks)) + (1.0 / (rrf_k + sparse_ranks))
            final_scores = rrf_scores / (2.0 / (rrf_k + 1))
        else:
            scaled_sparse = sparse_scores
            if sparse_scores.max() > 0:
                scaled_sparse = sparse_scores / sparse_scores.max()
            final_scores = weights.get('dense', 0.6) * dense_sims + weights.get('sparse', 0.4) * scaled_sparse

        top_idx = np.argsort(final_scores)[-1]
        best_candidate_score = float(final_scores[top_idx])
        raw_dense_sim = float(dense_sims[top_idx])
        
        if raw_dense_sim >= semantic_floor and best_candidate_score >= threshold:
            assigned_trend = trend_keys[top_idx]
            topic_type = "Trending"
            best_match_score = min(best_candidate_score, 1.0)
            
            if rerank and reranker:
                pair = (cluster_query, trend_queries[top_idx])
                rerank_score = reranker.predict([pair])[0]
                if rerank_score < -2.5: 
                     assigned_trend, topic_type, best_match_score = "Discovery", "Discovery", 0.0
        else:
            best_match_score = raw_dense_sim
            
    return assigned_trend, topic_type, best_match_score