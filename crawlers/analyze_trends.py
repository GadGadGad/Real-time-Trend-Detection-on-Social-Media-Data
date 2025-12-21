"""
Multi-Source Trend Analysis Pipeline
Matches social & news posts to Google Trends using semantic similarity.
"""

import json
import csv
import re
import os
import glob
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from rich.console import Console
from rich.table import Table
import argparse
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Import project modules
import sys
import os

# Ensure the parent directory is in path for package imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from crawlers.clustering import cluster_data, extract_cluster_labels
    from crawlers.alias_normalizer import normalize_with_aliases, build_alias_dictionary, batch_normalize_texts
    from crawlers.ner_extractor import enrich_text_with_entities, batch_enrich_texts, HAS_NER
    from crawlers.sentiment import batch_analyze_sentiment
    from crawlers.vectorizers import get_embeddings
    from crawlers.taxonomy_classifier import TaxonomyClassifier
    from crawlers.llm_refiner import LLMRefiner
    from crawlers.trend_scoring import calculate_unified_score
except (ImportError, ModuleNotFoundError):
    # Fallback for local/flat execution
    try:
        from clustering import cluster_data, extract_cluster_labels
        from alias_normalizer import normalize_with_aliases, build_alias_dictionary, batch_normalize_texts
        from ner_extractor import enrich_text_with_entities, batch_enrich_texts, HAS_NER
        from sentiment import batch_analyze_sentiment
        from vectorizers import get_embeddings
        from taxonomy_classifier import TaxonomyClassifier
        from llm_refiner import LLMRefiner
        from trend_scoring import calculate_unified_score
    except Exception as e:
        console.print(f"[yellow]⚠️ Partial imports failed: {e}. Some features may be disabled.[/yellow]")

DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
console = Console()

def clean_text(text):
    if not text: return ""
    patterns = [r'(?i)\b(cre|credit|via|nguồn)\s*[:.-]\s*.*$', r'(?i)\b(cre|credit)\s+by\s*[:.-]?\s*.*$']
    cleaned = text
    for p in patterns: cleaned = re.sub(p, '', cleaned)
    return cleaned.strip()

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            unified = []
            for item in data:
                text = item.get('text') or item.get('content') or ''
                time_str = item.get('time') or item.get('time_label') or ''
                page_name = item.get('pageName') or item.get('page_name') or 'Unknown'
                unified.append({
                    "source": f"Face: {page_name}",
                    "content": clean_text(text),
                    "title": "",
                    "url": item.get('url') or item.get('postUrl') or '',
                    "stats": item.get('stats') or {'likes': item.get('likes', 0), 'comments': item.get('comments', 0), 'shares': item.get('shares', 0)},
                    "time": time_str,
                    "timestamp": item.get('timestamp')
                })
            return unified
    except Exception as e:
        console.print(f"[red]Error loading JSON {filepath}: {e}[/red]")
        return []

def load_trends(csv_files):
    trends = {}
    for filepath in csv_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) 
                for row in reader:
                    if len(row) < 5: continue
                    main_trend = row[0].strip()
                    vol_str = row[1].strip()
                    # Parse volume
                    clean_vol = vol_str.upper().replace(',', '').replace('.', '')
                    multiplier = 1000 if 'N' in clean_vol or 'K' in clean_vol else (1000000 if 'M' in clean_vol or 'TR' in clean_vol else 1)
                    num_parts = re.findall(r'\d+', clean_vol)
                    volume = int(num_parts[0]) * multiplier if num_parts else 0
                    
                    keywords = [k.strip() for k in row[4].split(',') if k.strip()]
                    if main_trend not in keywords: keywords.insert(0, main_trend)
                    trends[main_trend] = {"keywords": keywords, "volume": volume}
        except Exception: pass
    return trends

def extract_dynamic_anchors(posts, trends, top_n=20):
    from sklearn.feature_extraction.text import CountVectorizer
    trend_kws = set()
    for t in trends.values():
        for kw in t.get('keywords', []): trend_kws.add(kw.lower())
    texts = [p.get('content', '').lower() for p in posts]
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
    try:
        X = vectorizer.fit_transform(texts)
        post_words = vectorizer.get_feature_names_out()
        word_counts = X.toarray().sum(axis=0)
        anchors = sorted([(w, c) for w, c in zip(post_words, word_counts) if w in trend_kws], key=lambda x: -x[1])
        return [a[0] for a in anchors[:top_n]]
    except: return []

def apply_guidance_enrichment(text, anchors):
    found = [a for a in anchors if a in text.lower()]
    if found: return f"{' '.join(found * 2)} | {text}"
    return text

def find_matches_hybrid(posts, trends, model_name=None, threshold=0.5, 
                        use_aliases=True, use_ner=False, 
                        embedding_method="sentence-transformer", save_all=False,
                        rerank=True, min_cluster_size=5, labeling_method="semantic",
                        reranker_model_name=None, use_llm=False, gemini_api_key=None,
                        llm_provider="gemini", llm_model_path=None):
    if not posts: return []
    
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL)
    llm_refiner = LLMRefiner(provider=llm_provider, api_key=gemini_api_key, model_path=llm_model_path) if use_llm else None
    taxonomy_clf = TaxonomyClassifier(embedding_model=embedder) if TaxonomyClassifier else None
    reranker = None
    if rerank:
        ce_model = reranker_model_name or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        try: reranker = CrossEncoder(ce_model)
        except: pass

    post_contents = [p.get('content', '')[:500] for p in posts]
    anchors = extract_dynamic_anchors(posts, trends)
    
    if use_aliases:
        build_alias_dictionary({k: v['keywords'] for k, v in trends.items()})
        post_contents_enriched = batch_normalize_texts(post_contents)
    else:
        post_contents_enriched = post_contents

    if anchors:
        post_contents_enriched = [apply_guidance_enrichment(t, anchors) for t in post_contents_enriched]

    post_embeddings = get_embeddings(post_contents_enriched, method=embedding_method, model_name=model_name)
    cluster_labels = cluster_data(post_embeddings, min_cluster_size=min_cluster_size)
    unique_labels = sorted([l for l in set(cluster_labels) if l != -1])
    sentiments = batch_analyze_sentiment(post_contents)

    trend_keys = list(trends.keys())
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_embeddings = embedder.encode(trend_queries) if trend_queries else []
    
    cluster_names = extract_cluster_labels(post_contents, cluster_labels, model=embedder, method=labeling_method, anchors=anchors)
    cluster_mapping = {}

    for label in unique_labels:
        indices = [i for i, l in enumerate(cluster_labels) if l == label]
        cluster_posts = [posts[i] for i in indices]
        cluster_query = cluster_names.get(label, f"Cluster {label}")
        assigned_trend, topic_type, best_match_score = "Discovery", "Discovery", 0.0

        if len(trend_embeddings) > 0:
            cluster_emb = embedder.encode(cluster_query)
            sims = cosine_similarity([cluster_emb], trend_embeddings)[0]
            top_idx = np.argsort(sims)[-3:][::-1]
            if rerank and reranker:
                rerank_scores = reranker.predict([(cluster_query, trend_queries[k]) for k in top_idx])
                best_s = np.argmax(rerank_scores)
                if rerank_scores[best_s] > -2:
                    best_match_score = float(sims[top_idx[best_s]])
                    assigned_trend = trend_keys[top_idx[best_s]]
                    topic_type = "Trending"
            elif sims[top_idx[0]] > threshold:
                best_match_score = float(sims[top_idx[0]])
                assigned_trend = trend_keys[top_idx[0]]
                topic_type = "Trending"

        trend_data = trends.get(assigned_trend, {'volume': 0})
        unified_score, components = calculate_unified_score(trend_data, cluster_posts)
        category, cat_method = taxonomy_clf.classify(cluster_query + " " + assigned_trend) if taxonomy_clf else ("Unclassified", "None")
        
        llm_reasoning = ""
        final_topic_name = assigned_trend if assigned_trend != "Discovery" else f"New: {cluster_query}"
        
        cluster_mapping[label] = {
            "final_topic": final_topic_name, "topic_type": topic_type, "cluster_name": cluster_query,
            "category": category, "category_method": cat_method, "match_score": best_match_score,
            "trend_score": unified_score, "score_components": components, "llm_reasoning": "",
            "posts": cluster_posts # Temporary for LLM
        }

    # --- NEW: BATCH LLM REFINEMENT ---
    if llm_refiner:
        to_refine = []
        for l, m in cluster_mapping.items():
            if m["topic_type"] == "Discovery" or m["trend_score"] > 30:
                to_refine.append({
                    "label": l, "name": m["cluster_name"], "topic_type": m["topic_type"],
                    "category": m["category"], "sample_posts": m["posts"]
                })
        
        if to_refine:
            console.print(f"   🤖 [cyan]Batch Refining {len(to_refine)} clusters with {llm_provider}...[/cyan]")
            batch_results = llm_refiner.refine_batch(to_refine)
            
            for l, res in batch_results.items():
                label_key = int(l) if isinstance(l, (str, int)) else l
                if label_key in cluster_mapping:
                    m = cluster_mapping[label_key]
                    if m["topic_type"] == "Discovery":
                        m["final_topic"] = f"New: {res['refined_title']}"
                    m["category"] = res["category"]
                    m["category_method"] = "LLM"
                    m["llm_reasoning"] = res["reasoning"]

    # Consolidated Mapping
    consolidated_mapping = {}
    topic_groups = {}
    for l, m in cluster_mapping.items():
        t = m["final_topic"]
        if t not in topic_groups: topic_groups[t] = []
        topic_groups[t].append(l)

    for topic, labels in topic_groups.items():
        all_posts = []
        for l in labels:
            idx = [i for i, val in enumerate(cluster_labels) if val == l]
            all_posts.extend([posts[i] for i in idx])
        
        m = cluster_mapping[labels[0]]
        combined_score, combined_comp = calculate_unified_score(trends.get(topic, {'volume': 0}), all_posts)
        consolidated_mapping[topic] = {**m, "trend_score": combined_score, "score_components": combined_comp}

    matches = []
    for i, post in enumerate(posts):
        label = cluster_labels[i]
        if label != -1:
            t_name = cluster_mapping[label]["final_topic"]
            m = consolidated_mapping[t_name]
            matches.append({
                "source": post.get('source'), "time": post.get('time'), "post_content": post.get('content'),
                "final_topic": m["final_topic"], "topic_type": m["topic_type"], "category": m["category"],
                "score": m["match_score"], "trend_score": m["trend_score"], "llm_reasoning": m["llm_reasoning"],
                "sentiment": sentiments[i], "is_matched": (m["topic_type"] == "Trending"), "trend": m["final_topic"]
            })
        elif save_all:
            matches.append({"final_topic": "Unassigned", "topic_type": "Noise", "sentiment": sentiments[i], "is_matched": False})

    return matches

def find_matches(posts, trends, model_name=None, threshold=0.35, 
                 use_aliases=True, embedding_method="sentence-transformer", save_all=False):
    """Baseline semantic matching without clustering."""
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL)
    trend_keys = list(trends.keys())
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    
    post_contents = [p.get('content', '')[:500] for p in posts]
    post_embeddings = get_embeddings(post_contents, method=embedding_method, model_name=model_name)
    trend_embeddings = embedder.encode(trend_queries)
    
    sims = cosine_similarity(post_embeddings, trend_embeddings)
    matches = []
    for i, post in enumerate(posts):
        best_idx = np.argmax(sims[i])
        best_score = sims[i][best_idx]
        topic = trend_keys[best_idx] if best_score >= threshold else "Unassigned"
        if topic != "Unassigned" or save_all:
            matches.append({
                "source": post.get('source'), "time": post.get('time'), "post_content": post.get('content'),
                "trend": topic, "score": float(best_score), "is_matched": (topic != "Unassigned"),
                "final_topic": topic
            })
    return matches

# Aliases for notebook compatibility
def load_social_data(files):
    all_data = []
    for f in files: all_data.extend(load_json(f))
    return all_data

def load_news_data(files):
    all_data = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    all_data.append({
                        "source": os.path.basename(os.path.dirname(f)).upper(),
                        "content": f"{row.get('title', '')}\n{row.get('content', '')}",
                        "title": row.get('title', ''), "url": row.get('url', ''),
                        "stats": {'likes': 0, 'comments': 0, 'shares': 0},
                        "time": row.get('published_at', '')
                    })
        except: pass
    return all_data

load_google_trends = load_trends

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Source Trend Analysis")
    parser.add_argument("--social", type=str, nargs="+", help="Social JSON files")
    parser.add_argument("--news", type=str, nargs="+", help="News CSV files")
    parser.add_argument("--trends", type=str, nargs="+", help="Google Trends CSV files")
    parser.add_argument("--llm", action="store_true", help="Enable LLM refinement")
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "kaggle", "local"], help="LLM Provider")
    parser.add_argument("--llm-model-path", type=str, help="Local path or HF ID for local LLM")
    
    args = parser.parse_args()
    
    if args.social and args.trends:
        social_posts = []
        for f in args.social: social_posts.extend(load_json(f))
        trends = load_trends(args.trends)
        
        results = find_matches_hybrid(
            social_posts, trends, 
            use_llm=args.llm, 
            llm_provider=args.llm_provider, 
            llm_model_path=args.llm_model_path
        )
        print(f"Analyzed {len(social_posts)} posts. Found {len(set(r['final_topic'] for r in results))} trends.")
