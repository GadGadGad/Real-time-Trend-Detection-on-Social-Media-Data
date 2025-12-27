# [ORCHESTRATOR] src/pipeline/main_pipeline.py
import re
import numpy as np
from src.utils.text_processing.cleaning import clean_text

def strip_news_source_noise(text):
    """Loại bỏ tên báo chí đứng đầu câu (VTV, VNExpress...)"""
    if not text: return ""
    patterns = [
        r'(?i)^Face:\s*[^:]+[-–—:]\s*',
        r'^[\(\[][^\]\)]+[\)\]]\s*[-–—:]\s*',
        r'(?i)^(VNEXPRESS|NLD|THANHNIEN|TUOITRE|VIETNAMNET|VTV|ZING|DANTRI)\s*[:\-–—]\s*',
        r'(?i)^\(REUTERS\)\s*[-–—:]?\s*',
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, '', cleaned, count=1)
    return cleaned.strip()

def filter_obvious_noise(trends):
    """
    Bộ lọc nhiễu 'siêu cấp' của tác giả:
    - Loại bỏ xổ số, thời tiết, giá vàng.
    - Loại bỏ bóng đá CLB quốc tế (MU, Real, Barca...) trừ khi có đội tuyển VN.
    """
    noise_kws = ['xo so', 'kqxs', 'xsmb', 'thoi tiet', 'gia vang', 'tu vi', 'gia ca']
    club_blacklist = ['man city', 'man utd', 'mu', 'real madrid', 'barca', 'chelsea', 'arsenal', 'liverpool', 'juve']
    vn_whitelist = ['viet nam', 'việt nam', 'u23', 'đội tuyển', 'tuyển nữ']

    filtered = {}
    for k, v in trends.items():
        norm_k = k.lower()
        if any(nk in norm_k for nk in noise_kws): continue
        
        # Logic bóng đá: Nếu nhắc đến CLB mà không có 'Việt Nam' thì loại
        if any(c in norm_k for c in club_blacklist):
            if not any(w in norm_k for w in vn_whitelist):
                continue
                
        filtered[k] = v
    return filtered

def find_matches_hybrid(posts, trends, **kwargs):
    """Hàm chạy Offline/Replay kết hợp tất cả các bước (Full Pipeline)"""
    from sentence_transformers import SentenceTransformer
    from src.pipeline.pipeline_stages import run_sahc_clustering, calculate_match_scores
    from src.core.extraction.taxonomy_classifier import TaxonomyClassifier
    from src.core.scoring.trend_scoring import calculate_unified_score

    # Khởi tạo mô hình
    model = SentenceTransformer(kwargs.get('model_name', "paraphrase-multilingual-mpnet-base-v2"))
    
    # 1. Làm sạch & Embedding
    contents = [strip_news_source_noise(clean_text(p.get('content', ''))) for p in posts]
    post_embs = model.encode(contents)
    
    # 2. Gom cụm SAHC
    labels = run_sahc_clustering(posts, post_embs, post_contents=contents, **kwargs)
    
    # 3. Xử lý Trends & BM25
    trends = filter_obvious_noise(trends)
    trend_keys = list(trends.keys())
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_embs = model.encode(trend_queries)
    
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([q.lower().split() for q in trend_queries]) if trend_queries else None

    # 4. Duyệt qua từng cụm để gán nhãn
    results = []
    tax_clf = TaxonomyClassifier(embedding_model=model)
    unique_labels = [l for l in set(labels) if l != -1]
    
    for label in unique_labels:
        idx = [i for i, l in enumerate(labels) if l == label]
        cluster_posts = [posts[i] for i in idx]
        rep_text = contents[idx[0]]
        
        assigned, t_type, match_score = calculate_match_scores(
            rep_text[:100], label, trend_embs, trend_keys, trend_queries, model, 
            reranker=None, rerank=False, threshold=kwargs.get('threshold', 0.5), 
            bm25_index=bm25, cluster_centroid=np.mean(post_embs[idx], axis=0)
        )
        
        u_score, components = calculate_unified_score(trends.get(assigned, {'volume': 0}), cluster_posts)
        cat, _ = tax_clf.classify(rep_text)
        
        for i in idx:
            res = posts[i].copy()
            res.update({
                "final_topic": assigned if t_type == "Trending" else f"New: {rep_text[:50]}...",
                "topic_type": t_type,
                "category": cat,
                "trend_score": u_score,
                "score": match_score
            })
            results.append(res)
            
    return results