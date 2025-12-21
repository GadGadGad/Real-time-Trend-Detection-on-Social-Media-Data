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
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
console = Console()

# --- PROJECT IMPORTS ---
import sys
import os

# Ensure the parent directory is in path for package imports
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from crawlers.clustering import cluster_data, extract_cluster_labels
    from crawlers.alias_normalizer import normalize_with_aliases, build_alias_dictionary, batch_normalize_texts
    from crawlers.ner_extractor import enrich_text_with_entities, batch_enrich_texts, HAS_NER
    from crawlers.sentiment import batch_analyze_sentiment, clear_sentiment_analyzer
    from crawlers.vectorizers import get_embeddings
    from crawlers.taxonomy_classifier import TaxonomyClassifier
    from crawlers.llm_refiner import LLMRefiner
    from crawlers.trend_scoring import calculate_unified_score
except (ImportError, ModuleNotFoundError):
    # Fallback for flat directory structure (some notebook environments)
    try:
        import clustering
        import alias_normalizer
        import ner_extractor
        import sentiment
        import vectorizers
        import taxonomy_classifier
        import llm_refiner
        import trend_scoring
        
        cluster_data = clustering.cluster_data
        extract_cluster_labels = clustering.extract_cluster_labels
        normalize_with_aliases = alias_normalizer.normalize_with_aliases
        build_alias_dictionary = alias_normalizer.build_alias_dictionary
        batch_normalize_texts = alias_normalizer.batch_normalize_texts
        enrich_text_with_entities = ner_extractor.enrich_text_with_entities
        batch_enrich_texts = ner_extractor.batch_enrich_texts
        HAS_NER = ner_extractor.HAS_NER
        batch_analyze_sentiment = sentiment.batch_analyze_sentiment
        clear_sentiment_analyzer = sentiment.clear_sentiment_analyzer
        get_embeddings = vectorizers.get_embeddings
        TaxonomyClassifier = taxonomy_classifier.TaxonomyClassifier
        LLMRefiner = llm_refiner.LLMRefiner
        calculate_unified_score = trend_scoring.calculate_unified_score
    except Exception as e:
        console.print(f"[yellow]⚠️ Partial imports failed: {e}. System might be unstable.[/yellow]")

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
                        llm_provider="gemini", llm_model_path=None,
                        llm_custom_instruction=None, use_cache=True,
                        debug_llm=False, summarize_all=False):
    if not posts: return []
    
    # In Sequential mode, we use CUDA for everything, but one at a time.
    import torch
    embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    console.print(f"🚀 [cyan]Phase 1: High-Speed Embeddings & Sentiment on {embedding_device}...[/cyan]")
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL, device=embedding_device)
    
    taxonomy_clf = TaxonomyClassifier(embedding_model=embedder) if TaxonomyClassifier else None
    reranker = None
    if rerank:
        ce_model = reranker_model_name or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        try: reranker = CrossEncoder(ce_model, device=embedding_device)
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

    # --- PHASE 0: SUMMARIZATION ---
    if use_llm: # Only run if advanced features enabled
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

        import hashlib
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
            from crawlers.summarizer import Summarizer
            mode_desc = "ALL" if summarize_all else "LONG"
            console.print(f"[cyan]📝 Phase 0: Summarizing {len(long_indices_to_process)} articles ({mode_desc}) - {len(long_indices_all)-len(long_indices_to_process)} cached...[/cyan]")
            
            summ = Summarizer()
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
                    
                    import datetime
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

            console.print(f"   ✅ [green]Summarized {len(long_indices_to_process)} posts with ViT5.[/green]")
        elif long_indices_all:
             console.print(f"   ✅ [green]All {len(long_indices_all)} target posts were found in cache![/green]")

    post_embeddings = get_embeddings(
        post_contents_enriched, 
        method=embedding_method, 
        model_name=model_name,
        existing_model=embedder,
        device=embedding_device,
        cache_dir="embeddings_cache" if use_cache else None
    )
    cluster_labels = cluster_data(post_embeddings, min_cluster_size=min_cluster_size)
    unique_labels = sorted([l for l in set(cluster_labels) if l != -1])
    sentiments = batch_analyze_sentiment(post_contents)

    trend_keys = list(trends.keys())
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_embeddings = get_embeddings(
        trend_queries, 
        method=embedding_method, 
        model_name=model_name,
        existing_model=embedder,
        device=embedding_device,
        cache_dir="embeddings_cache" if use_cache else None
    ) if trend_queries else []
    
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

    # --- PHASE 2: SEQUENTIAL GPU CLEANUP ---
    if use_llm and embedding_device == 'cuda' and llm_provider != 'gemini':
        console.print("[yellow]🧹 Phase 2: Unloading Phase 1 models to free VRAM for LLM...[/yellow]")
        if 'embedder' in locals(): del embedder
        if 'reranker' in locals(): del reranker
        if 'taxonomy_clf' in locals(): del taxonomy_clf
        clear_sentiment_analyzer()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # --- PHASE 3: BATCH LLM REFINEMENT ---
    console.print(f"🚀 [cyan]Phase 3: LLM Refinementpass using {llm_provider}...[/cyan]")
    llm_refiner = LLMRefiner(provider=llm_provider, api_key=gemini_api_key, model_path=llm_model_path, debug=debug_llm) if use_llm else None
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
            batch_results = llm_refiner.refine_batch(to_refine, custom_instruction=llm_custom_instruction)
            
            for l, res in batch_results.items():
                label_key = int(l) if isinstance(l, (str, int)) else l
                if label_key in cluster_mapping:
                    m = cluster_mapping[label_key]
                    if m["topic_type"] == "Discovery":
                        m["final_topic"] = f"New: {res['refined_title']}"
                    m["category"] = res["category"]
                    m["event_type"] = res.get("event_type", "Specific") # Default to specific if missing
                    m["category_method"] = "LLM"
                    m["llm_reasoning"] = res["reasoning"]

                    # FILTER: Downgrade "Generic" events unless they are massive viral hits
                    if m["event_type"] == "Generic":
                        if m["trend_score"] < 80:
                            m["topic_type"] = "Noise"
                            m["category"] = "Generic/Routine"
                            m["final_topic"] = f"[Generic] {res['refined_title']}"
                        else:
                            # Keep it but mark it
                            m["final_topic"] = f"Viral: {res['refined_title']}"
            
            success_count = len(batch_results)
            console.print(f"   ✅ [bold green]LLM Pass Complete: Successfully refined {success_count}/{len(to_refine)} clusters.[/bold green]")

        # --- PHASE 4: SEMANTIC DEDUPLICATION ---
        console.print("🔗 [cyan]Phase 4: Semantic Topic Deduplication...[/cyan]")
        all_topics = [m["final_topic"] for m in cluster_mapping.values() if m["topic_type"] != "Discovery"]
        if all_topics:
            canonical_map = llm_refiner.deduplicate_topics(all_topics)
            dedup_count = 0
            for label, m in cluster_mapping.items():
                orig = m["final_topic"]
                if orig in canonical_map and canonical_map[orig] != orig:
                    m["final_topic"] = canonical_map[orig]
                    dedup_count += 1
            
            if dedup_count > 0:
                console.print(f"   ✨ [green]Merged {dedup_count} clusters into canonical topics.[/green]")

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
    parser.add_argument("--summarize-all", action="store_true", help="Summarize ALL posts before clustering")
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "kaggle", "local"], help="LLM Provider")
    parser.add_argument("--llm-model-path", type=str, help="Local path or HF ID for local LLM")
    parser.add_argument("--llm-instruction", type=str, help="Custom instructions for LLM refinement")
    
    args = parser.parse_args()
    
    if args.social and args.trends:
        social_posts = []
        for f in args.social: social_posts.extend(load_json(f))
        trends = load_trends(args.trends)
        
        results = find_matches_hybrid(
            social_posts, trends, 
            use_llm=args.llm, 
            llm_provider=args.llm_provider, 
            llm_model_path=args.llm_model_path,
            llm_custom_instruction=args.llm_instruction,
            summarize_all=args.summarize_all
        )
        print(f"Analyzed {len(social_posts)} posts. Found {len(set(r['final_topic'] for r in results))} trends.")
