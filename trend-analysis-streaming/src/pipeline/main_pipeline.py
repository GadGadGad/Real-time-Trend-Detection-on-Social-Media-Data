import json
import csv
import re
import os
import numpy as np
import argparse
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console

console = Console()
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.analysis.clustering import extract_cluster_labels
from src.core.extraction.ner_extractor import enrich_text_with_entities
from src.core.analysis.sentiment import batch_analyze_sentiment
from src.utils.text_processing.vectorizers import get_embeddings
from src.core.extraction.taxonomy_classifier import TaxonomyClassifier
from src.core.llm.llm_refiner import LLMRefiner
from src.core.scoring.trend_scoring import calculate_unified_score
from src.utils.text_processing.cleaning import clean_text, strip_news_source_noise, normalize_text

def filter_obvious_noise(trends):
    from src.utils.text_processing.stopwords import get_noise_keywords
    noise_kws = get_noise_keywords()
    
    filtered = {}
    for k, v in trends.items():
        norm_k = normalize_text(k)
        if any(nk in norm_k for nk in noise_kws): continue
        if re.match(r'^\d+$', norm_k.replace(' ', '')): continue
        filtered[k] = v
    return filtered

def load_trends(csv_files):
    trends = {}
    for filepath in csv_files:
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                trends.update(json.load(f))
            continue
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) < 5: continue
                name, vol_str, kws_str = row[0].strip(), row[1].strip(), row[4]
                clean_vol = vol_str.upper().replace(',', '').replace('.', '')
                mult = 1000 if 'N' in clean_vol else (1000000 if 'TR' in clean_vol else 1)
                nums = re.findall(r'\d+', clean_vol)
                vol = int(nums[0]) * mult if nums else 0
                kws = [kw.strip() for kw in kws_str.split(',') if kw.strip()]
                if name not in kws: kws.insert(0, name)
                trends[name] = {"keywords": kws, "volume": vol}
    return trends

def find_matches_hybrid(posts, trends, model_name=None, threshold=0.5, rerank=True, use_llm=False, gemini_api_key=None):
    from src.pipeline.pipeline_stages import run_sahc_clustering, calculate_match_scores
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL, device=device)
    
    post_contents = [strip_news_source_noise(clean_text(p.get('content', ''))) for p in posts]
    post_embeddings = get_embeddings(post_contents, method="sentence-transformer", existing_model=embedder, device=device)
    
    labels = run_sahc_clustering(posts, post_embeddings, post_contents=post_contents)
    unique_labels = sorted([l for l in set(labels) if l != -1])
    
    trend_keys = list(trends.keys())
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_embeddings = embedder.encode(trend_queries)
    
    cluster_names = extract_cluster_labels(post_contents, labels, model=embedder)
    taxonomy_clf = TaxonomyClassifier(embedding_model=embedder)
    
    results = []
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        cluster_posts = [posts[i] for i in indices]
        cluster_centroid = np.mean(post_embeddings[indices], axis=0)
        
        assigned_trend, topic_type, match_score = calculate_match_scores(
            cluster_names[label], label, trend_embeddings, trend_keys, trend_queries, 
            embedder, None, False, threshold, cluster_centroid=cluster_centroid
        )
        
        score, comp = calculate_unified_score(trends.get(assigned_trend, {'volume':0}), cluster_posts)
        
        for p in cluster_posts:
            res = p.copy()
            res.update({
                "final_topic": assigned_trend if topic_type == "Trending" else cluster_names[label],
                "topic_type": topic_type,
                "score": match_score,
                "trend_score": score,
                "is_matched": (topic_type == "Trending")
            })
            results.append(res)
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--social", nargs="+")
    parser.add_argument("--trends", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    
    if args.social and args.trends:
        posts = []
        for f in args.social:
            with open(f, 'r', encoding='utf-8') as jf:
                posts.extend(json.load(jf))
        
        trends = load_trends(args.trends)
        results = find_matches_hybrid(posts, trends)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)