# [ENGINE] src/pipeline/pipeline_stages.py
import numpy as np
import json
import os
from dateutil import parser
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity

# Import logic lõi từ các module bạn đã có
from src.core.analysis.clustering import cluster_data
from src.core.llm.llm_refiner import LLMRefiner
from src.pipeline.main_pipeline import clean_text, strip_news_source_noise

console = Console()

def run_summarization_stage(post_contents, use_llm, summarize_all, model_name='vit5-base'):
    """Tóm tắt văn bản dài để tối ưu cho mô hình Embedding."""
    long_indices = [i for i, text in enumerate(post_contents) if len(text) > 500]
    if not (summarize_all or long_indices):
        return post_contents
        
    from src.core.analysis.summarizer import Summarizer
    summarizer = Summarizer(model_name=model_name)
    
    to_process = [post_contents[i] for i in (range(len(post_contents)) if summarize_all else long_indices)]
    if not to_process: return post_contents
    
    summaries = summarizer.summarize_batch(to_process)
    
    result = post_contents.copy()
    mapping_idx = range(len(post_contents)) if summarize_all else long_indices
    for i, idx in enumerate(mapping_idx):
        result[idx] = summaries[i]
    return result

def run_sahc_clustering(posts, post_embeddings, min_cluster_size=5, method='hdbscan', 
                        n_clusters=15, post_contents=None, epsilon=0.15, 
                        min_member_similarity=0.60, selection_method='leaf',
                        recluster_large=True, coherence_threshold=0.60):
    """
    Thuật toán SAHC chuẩn của tác giả:
    Ưu tiên News để tạo 'trọng tâm', sau đó mới hút Social vào.
    """
    # Tách News và Social
    news_idx = [i for i, p in enumerate(posts) if 'Face' not in str(p.get('source', ''))]
    social_idx = [i for i, p in enumerate(posts) if 'Face' in str(p.get('source', ''))]
    
    final_labels = np.full(len(posts), -1)
    
    # Bước 1: Gom cụm News (Nguồn tin cậy cao)
    if len(news_idx) >= min_cluster_size:
        news_labels = cluster_data(
            post_embeddings[news_idx], min_cluster_size=min_cluster_size, 
            epsilon=epsilon, method=method, n_clusters=n_clusters,
            texts=[post_contents[i] for i in news_idx] if post_contents else None,
            min_member_similarity=min_member_similarity, selection_method=selection_method,
            recluster_large=recluster_large, coherence_threshold=coherence_threshold
        )
        for i, idx in enumerate(news_idx):
            final_labels[idx] = news_labels[i]

    # Bước 2: Social Attachment (Gắn Social vào cụm News hiện có)
    unique_news_clusters = sorted([l for l in set(final_labels[news_idx]) if l != -1])
    unattached_social = social_idx.copy()
    
    if unique_news_clusters and social_idx:
        centroids = {l: np.mean(post_embeddings[[ni for i, ni in enumerate(news_idx) if final_labels[ni] == l]], axis=0) 
                     for l in unique_news_clusters}
        centroid_matrix = np.array([centroids[l] for l in unique_news_clusters])
        
        sims = cosine_similarity(post_embeddings[social_idx], centroid_matrix)
        ATTACH_THRESHOLD = 0.65 # Ngưỡng của tác giả
        
        unattached_social = []
        for i, s_idx in enumerate(social_idx):
            best_c_idx = np.argmax(sims[i])
            if sims[i][best_c_idx] >= ATTACH_THRESHOLD:
                final_labels[s_idx] = unique_news_clusters[best_c_idx]
            else:
                unattached_social.append(s_idx)

    # Bước 3: Social Discovery (Gom cụm phần Social còn lại)
    if len(unattached_social) >= min_cluster_size:
        offset = max(unique_news_clusters) + 1 if unique_news_clusters else 0
        disc_labels = cluster_data(
            post_embeddings[unattached_social], min_cluster_size=min_cluster_size, 
            method=method, epsilon=epsilon, min_member_similarity=min_member_similarity
        )
        for i, s_idx in enumerate(unattached_social):
            if disc_labels[i] != -1:
                final_labels[s_idx] = disc_labels[i] + offset
                
    return final_labels

def calculate_match_scores(cluster_query, cluster_label, trend_embeddings, trend_keys, trend_queries, 
                           embedder, reranker, rerank, threshold, bm25_index=None, 
                           cluster_centroid=None, use_rrf=False, rrf_k=60, use_prf=False, 
                           weights={'dense': 0.6, 'sparse': 0.4}, semantic_floor=0.35):
    """
    Matching Hybrid: Kết hợp Vector (Dense) + Từ khóa (BM25) + Rào chắn ngữ nghĩa.
    """
    assigned_trend, topic_type, best_score = "Discovery", "Discovery", 0.0

    if len(trend_embeddings) > 0:
        # 1. Điểm Vector (Dense)
        c_vec = cluster_centroid.reshape(1, -1) if cluster_centroid is not None else embedder.encode(cluster_query).reshape(1, -1)
        dense_sims = cosine_similarity(c_vec, trend_embeddings)[0]
        
        # 2. Điểm Từ khóa (BM25/Sparse)
        sparse_scores = np.zeros(len(trend_keys))
        if bm25_index:
            try:
                raw_sparse = np.array(bm25_index.get_scores(cluster_query.lower().split()))
                if raw_sparse.max() > 0:
                    sparse_scores = raw_sparse / (raw_sparse.max() + 5.0) # Chuẩn hóa mềm
            except: pass

        # 3. Kết hợp điểm (Fusion)
        if use_rrf: # Xếp hạng nghịch đảo (Nếu bạn muốn dùng)
            d_ranks = len(dense_sims) - np.argsort(np.argsort(dense_sims))
            s_ranks = len(sparse_scores) - np.argsort(np.argsort(sparse_scores))
            rrf_scores = (1.0 / (rrf_k + d_ranks)) + (1.0 / (rrf_k + s_ranks))
            final_scores = rrf_scores / (2.0 / (rrf_k + 1))
        else: # Kết hợp tuyến tính (Mặc định)
            final_scores = weights['dense'] * dense_sims + weights['sparse'] * sparse_scores

        top_idx = np.argmax(final_scores)
        
        # 4. Semantic Guard (Rào chắn của tác giả)
        if dense_sims[top_idx] >= semantic_floor and final_scores[top_idx] >= threshold:
            assigned_trend = trend_keys[top_idx]
            topic_type = "Trending"
            best_score = float(final_scores[top_idx])
        else:
            # Nếu không vượt qua rào chắn ngữ nghĩa -> Coi như xu hướng mới (Discovery)
            best_score = float(dense_sims[top_idx]) 

    return assigned_trend, topic_type, best_score