"""
Multi-Source Trend Analysis Pipeline
Matches social & news posts to Google Trends using semantic similarity.

Key Features:
- Alias-based text normalization (using Google Trends keywords)
- Multilingual sentence embeddings
- Direct trend assignment (no HDBSCAN clustering needed)
"""

import json
import csv
import os
import glob
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from rich.console import Console
import argparse
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

try:
    from crawlers.vectorizers import get_embeddings
except ImportError:
    from vectorizers import get_embeddings

try:
    from crawlers.alias_normalizer import build_alias_dictionary, batch_normalize_texts
except ImportError:
    from alias_normalizer import build_alias_dictionary, batch_normalize_texts

# Optional NER support
try:
    from crawlers.ner_extractor import batch_enrich_texts, HAS_NER
except ImportError:
    try:
        from ner_extractor import batch_enrich_texts, HAS_NER
    except ImportError:
        HAS_NER = False
        def batch_enrich_texts(texts, weight_factor=2):
            return texts

# Import new modules
try:
    from crawlers.clustering import cluster_data, extract_cluster_labels
    from crawlers.sentiment import batch_analyze_sentiment
except ImportError:
    from clustering import cluster_data, extract_cluster_labels
    from sentiment import batch_analyze_sentiment

# Best model for Vietnamese semantic similarity
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"

console = Console()


def load_json(filepath):
    """
    Load Facebook data from JSON (supports Apify format).
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            unified = []
            for item in data:
                # Handle Apify schema
                text = item.get('text') or item.get('content') or ''
                
                # Timestamp handling
                time_str = item.get('time') or item.get('time_label') or ''
                timestamp = item.get('timestamp')
                
                # Source handling
                page_name = item.get('pageName') or item.get('page_name') or 'Unknown'
                source = f"Face: {page_name}"
                
                # Stats handling (Apify flat structure vs old nested)
                if 'stats' in item:
                    stats = item['stats']
                else:
                    stats = {
                        'likes': item.get('likes', 0),
                        'comments': item.get('comments', 0),
                        'shares': item.get('shares', 0)
                    }

                unified.append({
                    "source": source,
                    "content": text,
                    "title": "",
                    "url": item.get('url') or item.get('postUrl') or '',
                    "stats": stats,
                    "time": time_str,
                    "timestamp": timestamp
                })
            return unified
    except Exception as e:
        console.print(f"[red]Error loading JSON {filepath}: {e}[/red]")
        return []


def load_social_data(files):
    """Wrapper to load multiple social JSON files."""
    all_data = []
    for f in files:
        all_data.extend(load_json(f))
    return all_data

def load_news_articles(data_dir):
    """Load news articles from CSV files in subdirectories."""
    unified = []
    pattern = os.path.join(data_dir, "**", "articles.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    console.print(f"[dim]Found {len(csv_files)} news CSV files.[/dim]")
    
    for filepath in csv_files:
        try:
            source_name = os.path.basename(os.path.dirname(filepath)).upper()
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    content = row.get('content', '')
                    title = row.get('title', '')
                    full_text = f"{title}\n{content}"
                    
                    unified.append({
                        "source": source_name,
                        "content": full_text,
                        "title": title,
                        "url": row.get('url', ''),
                        "stats": {'likes': 0, 'comments': 0, 'shares': 0},
                        "time": row.get('published_at', '')
                    })
        except Exception as e:
            console.print(f"[red]Error loading News CSV {filepath}: {e}[/red]")
            
    return unified

def load_news_data(files):
    """Wrapper to load multiple news CSV files (Notebook compatible)."""
    unified = []
    for filepath in files:
        try:
            source_name = os.path.basename(os.path.dirname(filepath)).upper()
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    content = row.get('content', '')
                    title = row.get('title', '')
                    full_text = f"{title}\n{content}"
                    
                    unified.append({
                        "source": source_name,
                        "content": full_text,
                        "title": title,
                        "url": row.get('url', ''),
                        "stats": {'likes': 0, 'comments': 0, 'shares': 0},
                        "time": row.get('published_at', '')
                    })
        except Exception as e:
            console.print(f"[red]Error loading News CSV {filepath}: {e}[/red]")
    return unified




def load_trends(csv_files):
    """Load trends from multiple CSV files."""
    trends = {}
    
    for filepath in csv_files:
        if not os.path.exists(filepath):
            console.print(f"[yellow]Warning: CSV file not found: {filepath}[/yellow]")
            continue
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                for row in reader:
                    if len(row) < 5: 
                        continue
                    
                    main_trend = row[0].strip()
                    related_keywords = row[4].split(',')
                    clean_keywords = [k.strip() for k in related_keywords if k.strip()]
                    
                    if main_trend not in clean_keywords:
                        clean_keywords.insert(0, main_trend)
                        
                    trends[main_trend] = clean_keywords
                    
        except Exception as e:
            console.print(f"[red]Error loading CSV {filepath}: {e}[/red]")
            
    return trends

# Alias for notebook
load_google_trends = load_trends


def find_matches(posts, trends, model_name=None, threshold=0.35, 
                 min_interactions=10, use_aliases=True, use_ner=False, 
                 embedding_method="sentence-transformer", save_all=False,
                 max_days=None):
    """
    Find matches using Semantic Similarity with text enrichment.
    
    Args:
        posts: List of post dictionaries
        trends: Dictionary of trends with keywords
        model_name: HuggingFace model for embeddings (sentence-transformer only)
        threshold: Cosine similarity threshold
        min_interactions: Minimum engagement for FB posts
        use_aliases: Whether to use alias normalization (default)
        use_ner: Whether to use NER enrichment (alternative to aliases)
        embedding_method: 'sentence-transformer', 'tfidf', 'bow', or 'glove'
        save_all: Save unmatched posts too
    """
    start_time = datetime.now()
    
    # 0. Date Filtering
    if max_days is not None:
        console.print(f"[bold yellow]🕒 Filtering posts older than {max_days} days...[/bold yellow]")
        cutoff_date = start_time - timedelta(days=max_days)
        filtered_posts = []
        for p in posts:
            # Parse time (handles ISO and timestamps)
            t_str = str(p.get('time', ''))
            p_time = None
            try:
                # Try ISO format first
                # Handle "2025-12-15T09:27:49.000Z"
                p_time = parser.parse(t_str).replace(tzinfo=None) # naive filtering
            except:
                # Try epoch if distinct
                pass
                
            # If parsing failed or empty, decided keep or drop? 
            # Let's assume keep if unknown, or drop? Drop is safer for "deprecated".
            # For simplicity, if we can't parse, we ignore this filter (keep it) except if it's clearly old.
            # But Apify data usually has valid ISO time.
            
            if p_time:
                if p_time >= cutoff_date:
                    filtered_posts.append(p)
            else:
                # Try timestamp if available
                ts = p.get('timestamp')
                if ts:
                    try:
                        p_time = datetime.fromtimestamp(int(ts)) # local time
                        if p_time >= cutoff_date:
                            filtered_posts.append(p)
                    except:
                        filtered_posts.append(p) # Keep if unknown
                else:
                    filtered_posts.append(p) # Keep if unknown
                    
        console.print(f"[dim]Dropped {len(posts) - len(filtered_posts)} old posts. remaining: {len(filtered_posts)}[/dim]")
        posts = filtered_posts
    
    matches = []
    
    # Build alias dictionary from trends (for alias mode)
    if use_aliases and not use_ner:
        build_alias_dictionary(trends)
    
    # Prepare texts
    trend_keys = list(trends.keys())
    trend_texts = [f"{t} " + " ".join(trends[t][:5]) for t in trend_keys]
    post_contents = [p.get('content', '')[:500] for p in posts]
    
    # Apply text enrichment
    if use_ner and HAS_NER:
        console.print("[bold magenta]🏷️ Enriching texts with NER (underthesea)...[/bold magenta]")
        trend_texts = batch_enrich_texts(trend_texts, weight_factor=2)
        post_contents = batch_enrich_texts(post_contents, weight_factor=2)
        console.print("[green]✅ NER enrichment complete![/green]")
    elif use_ner and not HAS_NER:
        console.print("[yellow]⚠️ NER requested but underthesea not installed. Falling back to aliases.[/yellow]")
        build_alias_dictionary(trends)
        trend_texts = batch_normalize_texts(trend_texts, show_progress=False)
        post_contents = batch_normalize_texts(post_contents, show_progress=True)
    elif use_aliases:
        console.print("[bold magenta]🔄 Normalizing texts with trend aliases...[/bold magenta]")
        trend_texts = batch_normalize_texts(trend_texts, show_progress=False)
        post_contents = batch_normalize_texts(post_contents, show_progress=True)
        console.print("[green]✅ Normalization complete![/green]")

    # Encode
    all_texts = trend_texts + post_contents
    
    if embedding_method == "sentence-transformer":
        console.print(f"[bold cyan]🧠 Encoding with Sentence Transformer: {model_name}...[/bold cyan]")
        model = SentenceTransformer(model_name)
        trend_embeddings = model.encode(trend_texts, show_progress_bar=True)
        post_embeddings = model.encode(post_contents, show_progress_bar=True)
    else:
        console.print(f"[bold cyan]📊 Encoding with {embedding_method.upper()}...[/bold cyan]")
        # For TF-IDF/BoW, we need to fit on all texts together
        all_embeddings = get_embeddings(all_texts, method=embedding_method)
        trend_embeddings = all_embeddings[:len(trend_texts)]
        post_embeddings = all_embeddings[len(trend_texts):]

    # --- NEW: Unsupervised Clustering ---
    console.print("[bold cyan]🧩 Running Unsupervised Clustering (Discovery)...[/bold cyan]")
    if len(post_embeddings) > 10: # Only cluster if enough data
        # Use HDBSCAN from clustering module
        cluster_labels = cluster_data(post_embeddings, min_cluster_size=5)
        # Name clusters
        cluster_names = extract_cluster_labels(post_contents, cluster_labels)
    else:
        cluster_labels = [-1] * len(post_embeddings)
        cluster_names = {}

    # --- NEW: Sentiment Analysis ---
    console.print("[bold cyan]😊 Running Sentiment Analysis...[/bold cyan]")
    sentiments = batch_analyze_sentiment(post_contents)

    console.print("[bold cyan]📐 Calculating Cosine Similarity...[/bold cyan]")
    similarity_matrix = cosine_similarity(post_embeddings, trend_embeddings)

    count_matched = 0
    count_unmatched = 0

    for i, post in enumerate(posts):
        stats = post.get('stats', {'likes': 0, 'comments': 0, 'shares': 0})
        total_interactions = stats.get('likes', 0) + stats.get('comments', 0) + stats.get('shares', 0)
        
        is_facebook = 'Face' in post.get('source', '')
        
        if not save_all and is_facebook and total_interactions < min_interactions:
            continue
            
        sim_scores = similarity_matrix[i]
        best_trend_idx = np.argmax(sim_scores)
        best_score = sim_scores[best_trend_idx]
        
        match_data = {
            "post_content": post.get('content', ''),
            "source": post.get('source', 'Unknown'),
            "time": post.get('time', 'Unknown'),
            "stats": stats,
            "processed_content": post_contents[i], # Visualizing normalization/NER
            "entities": str(post_contents[i]) if use_ner else "" # Rudimentary entity capture (post_contents IS enriched text)
        }

        # Adaptive Threshold: If no match found but score is decent, take it?
        # NO, just use the lower threshold globally.
        # But let's log best potential match for debugging
        
        if best_score > threshold:
            trend_name = trend_keys[best_trend_idx]
            match_data.update({
                "trend": trend_name,
                "keyword": "semantic-match", 
                "score": float(best_score),
                "is_matched": True,
                "cluster_id": int(cluster_labels[i]),
                "cluster_name": cluster_names.get(cluster_labels[i], "Unclustered"),
                "sentiment": sentiments[i]
            })
            matches.append(match_data)
            count_matched += 1
        elif save_all:
            match_data.update({
                "trend": "Unassigned",
                "keyword": "none", 
                "score": float(best_score),
                "closest_trend": trend_keys[best_trend_idx],
                "is_matched": False,
                "cluster_id": int(cluster_labels[i]),
                "cluster_name": cluster_names.get(cluster_labels[i], "Unclustered"),
                "sentiment": sentiments[i]
            })
            matches.append(match_data)
            count_unmatched += 1

    # Stats
    scores = [m['score'] for m in matches if m.get('is_matched')]
    if scores:
        console.print(f"[bold yellow]📊 Score Stats: Min={min(scores):.2f}, Max={max(scores):.2f}, Avg={sum(scores)/len(scores):.2f}[/bold yellow]")
    
    console.print(f"[dim]Stats: Matched={count_matched}, Unmatched={count_unmatched}[/dim]")

    return matches


def find_matches_hybrid(posts, trends, model_name=None, threshold=0.5, 
                        use_aliases=True, use_ner=False, 
                        embedding_method="sentence-transformer", save_all=False,
                        rerank=True):
    """
    Cluster-First approach with Cross-Encoder Reranking.
    1. Cluster all data (Unsupervised).
    2. Convert clusters to text vectors.
    3. Bi-Encoder: Top-10 Retrieval per cluster.
    4. Cross-Encoder: Rerank Top-10 to find precise match > 0.5.
    """
    if not posts:
        return []

    # --- 0. Initialize Models ---
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL)
    
    reranker = None
    if rerank:
        console.print("[bold yellow]⚡ Loading Cross-Encoder for Precision Reranking...[/bold yellow]")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

    trend_keys = list(trends.keys())
    trend_texts = [f"{t} " + " ".join(trends[t][:5]) for t in trend_keys]
    post_contents = [p.get('content', '')[:500] for p in posts]

    if use_ner and HAS_NER:
        console.print("[bold magenta]🏷️ Enriching texts with NER...[/bold magenta]")
        trend_texts = batch_enrich_texts(trend_texts, weight_factor=2)
        post_contents = batch_enrich_texts(post_contents, weight_factor=2)
    elif use_aliases:
        console.print("[bold magenta]🔄 Normalizing texts with aliases...[/bold magenta]")
        trend_texts = batch_normalize_texts(trend_texts, show_progress=False)
        post_contents = batch_normalize_texts(post_contents, show_progress=True)

    # Encode All
    console.print(f"[bold cyan]🧠 Encoding {len(post_contents)} posts for Hybrid Analysis...[/bold cyan]")
    # For hybrid, we always use sentence-transformer for embeddings for consistency with clustering
    # and reranking, so we simplify this block.
    trend_embeddings = embedder.encode(trend_texts, show_progress_bar=True)
    post_embeddings = embedder.encode(post_contents, show_progress_bar=True)

    # 2. Cluster Everything (HDBSCAN)
    console.print("[bold cyan]🧩 Running HDBSCAN on ALL posts...[/bold cyan]")
    cluster_labels = cluster_data(post_embeddings, min_cluster_size=5)
    cluster_names = extract_cluster_labels(post_contents, cluster_labels) # Top TF-IDF keywords
    
    # 3. Match Clusters to Trends (Bi-Encoder Retrieval + Reranking) ---
    # cluster_names is a dict {cluster_id: "keywords"}
    # trend_keys is a list of trend names
    
    # Filter out noise cluster (-1) for matching
    unique_labels = [label for label in cluster_names.keys() if label != -1]
    
    cluster_queries = [cluster_names[label] for label in unique_labels]
    trend_corpus = trend_keys # Use the actual trend names as the corpus
    
    console.print(f"[bold cyan]🎯 Matching {len(cluster_queries)} clusters to {len(trend_corpus)} trends...[/bold cyan]")
    
    # Encode clusters (Query) and Trends (Corpus)
    cluster_embeddings = embedder.encode(cluster_queries)
    trend_embeddings = embedder.encode(trend_corpus)
    
    # Bi-Encoder Retrieval (Top-K)
    TOP_K = 10
    bi_scores = cosine_similarity(cluster_embeddings, trend_embeddings)
    
    # Map cluster_label to assigned trend
    cluster_mapping = {} 
    
    for i, label in enumerate(unique_labels):
        cluster_query = cluster_queries[i]
        # Get Top-K candidates from Bi-Encoder
        top_k_indices = np.argsort(bi_scores[i])[::-1][:TOP_K]
        
        best_trend = None
        best_score = 0.0
        
        if rerank and reranker:
            # Prepare pairs for Cross-Encoder: [[query, candidate_1], [query, candidate_2], ...]
            candidates = [trend_corpus[idx] for idx in top_k_indices]
            pairs = [[cluster_query, cand] for cand in candidates]
            
            # Predict (Returns logits or 0-1 scores depending on model, ms-marco leads to unbounded logits usually, but MiniLM-L-6-v2 output 0-1?)
            # Actually `cross-encoder/ms-marco-MiniLM-L-6-v2` is trained for CE. predict() returns scores.
            ce_scores = reranker.predict(pairs)
            
            # Find max
            best_idx_in_k = np.argmax(ce_scores)
            best_score_raw = ce_scores[best_idx_in_k] # This is logit usually 
            
            # Sigmoid normalization for heuristic threshold (approx)
            # Logit > 0 is roughly relevance > 0.5. 
            # Let's apply simple sigmoid for normalized score display
            # score_norm = 1 / (1 + np.exp(-best_score))
            
            # Actually, let's stick to using the provided threshold logic, but CrossEncoder scores are different distribution.
            # MS Marco models: score < 0 is usually non-relevant. Score > 0 is relevant.
            # Let's use a mapping: if score > -1 (lenient) -> Match.
            
            # For simplicity in this demo, let's trust the rank. If the rank 1 is high enough.
            # Let's use a dedicated threshold for CE.
            # CE_THRESHOLD = 0.5 # For probability output models. 
            # Check if model outputs logits or prob. (MiniLM-L-6-v2 is logits).
            # We will use sigmoid to normalize to 0-1 for compatibility with the system.
            norm_score = 1 / (1 + np.exp(-best_score_raw))
            
            if norm_score > threshold:
                best_trend = candidates[best_idx_in_k]
                best_score = norm_score
                
        else:
            # Fallback to Bi-Encoder (Cosine)
            best_idx = top_k_indices[0] 
            best_score = bi_scores[i][best_idx]
            if best_score > threshold:
                best_trend = trend_corpus[best_idx]

        final_topic = best_trend if best_trend else cluster_query
        topic_type = "Trending" if best_trend else "Discovery"
        
        cluster_mapping[label] = {
            "topic": final_topic,
            "type": topic_type,
            "score": float(best_score)
        }
        if topic_type == "Trending":
            console.print(f"   ✅ Cluster {label} ('{cluster_query}') -> [green]{final_topic}[/green] ({best_score:.2f})")
        else:
            # Discovery!
            # Find the closest trend from the bi-encoder for logging purposes
            closest_bi_trend_idx = np.argmax(bi_scores[i])
            closest_bi_trend_name = trend_corpus[closest_bi_trend_idx]
            closest_bi_score = bi_scores[i][closest_bi_trend_idx]
            console.print(f"   ✨ Cluster {label} -> [cyan]New: {cluster_query}[/cyan] (Closest: {closest_bi_trend_name} {closest_bi_score:.2f})")

    # 4. Construct Results
    matches = []
    sentiments = batch_analyze_sentiment(post_contents) # Get sentiments
    
    for i, post in enumerate(posts):
        c_id = cluster_labels[i]
        
        # Default Info
        final_topic = "Unassigned"
        topic_type = "Noise"
        score = 0
        
        if c_id != -1:
            # It's in a cluster
            mapped = cluster_mapping.get(c_id)
            if mapped:
                final_topic = mapped["topic"]
                topic_type = mapped["type"]
                score = mapped["score"]
        else:
            # Noise point - Optional: Try Individual Match?
            # For Hybrid strict, we leave as Noise or try to match individually if very high
            pass
            
        stats = post.get('stats', {'likes': 0, 'comments': 0, 'shares': 0})
        
        if not save_all and topic_type == "Noise": 
            continue # Skip noise if not saving all
            
        matches.append({
            "post_content": post.get('content', ''),
            "source": post.get('source', 'Unknown'),
            "time": post.get('time', 'Unknown'),
            "stats": stats,
            "processed_content": post_contents[i],
            "entities": "", 
            
            # Hybrid Fields
            "final_topic": final_topic,
            "topic_type": topic_type,
            "score": float(score),
            
            # Backward Compat
            "trend": final_topic if topic_type == "Trending" else "Unassigned", 
            "is_matched": True if topic_type == "Trending" else False,
            "cluster_id": int(c_id),
            "cluster_name": cluster_names.get(c_id, "Unclustered"),
            "sentiment": sentiments[i]
        })
        
    console.print(f"[bold green]✅ Hybrid Analysis Complete: {len(matches)} posts processed.[/bold green]")
    return matches


def analyze_trend_coverage(matches, trends, min_posts=3):
    """
    Analyze which trends have actual coverage in data.
    
    Args:
        matches: List of matched items
        trends: Original trends dictionary
        min_posts: Minimum posts for a valid trend
        
    Returns:
        Dictionary of valid trends with their post counts
    """
    trend_coverage = Counter([m['trend'] for m in matches if m.get('is_matched')])
    
    console.print("\n[bold cyan]📊 Trend Coverage Analysis[/bold cyan]")
    console.print(f"   Total Google Trends: {len(trends)}")
    console.print(f"   Trends with matches: {len(trend_coverage)}")
    console.print(f"   Trends with NO data: {len(trends) - len(trend_coverage)}")
    
    valid_trends = {t: c for t, c in trend_coverage.items() if c >= min_posts}
    console.print(f"\n[green]✅ Valid trends (>= {min_posts} posts): {len(valid_trends)}[/green]")
    
    # Show top trends
    console.print("\n[bold]🔥 TOP 20 REAL TRENDS:[/bold]")
    for i, (trend, count) in enumerate(sorted(valid_trends.items(), key=lambda x: -x[1])[:20]):
        console.print(f"   {i+1}. {trend}: {count} posts")
    
    return valid_trends


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Source Trend Analysis")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"HuggingFace model for embeddings (default: {DEFAULT_MODEL})")
    parser.add_argument("--threshold", type=float, default=0.55, 
                        help="Similarity threshold for matching (default: 0.55)")
    parser.add_argument("--output", default="results/results.json", type=str, 
                        help="Path to save matched results (JSON)")
    parser.add_argument("--input", type=str, 
                        help="Path to load existing results (skip matching)")
    parser.add_argument("--no-aliases", action="store_true", default=False,
                        help="Disable alias normalization")
    parser.add_argument("--use-ner", action="store_true", default=False,
                        help="Use NER enrichment instead of aliases (requires underthesea)")
    parser.add_argument("--embedding", type=str, default="sentence-transformer",
                        choices=["sentence-transformer", "tfidf", "bow", "glove"],
                        help="Embedding method (default: sentence-transformer)")
    parser.add_argument("--save-all", action="store_true", help="Save all posts including noise/unmatched.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of posts to process (for testing).")
    parser.add_argument("--min-posts", type=int, default=3,
                        help="Minimum posts for a valid trend (default: 3)")
    parser.add_argument("--max-days", type=int, default=None,
                        help="Filter posts older than N days (default: None/All)")
    parser.add_argument("--method", type=str, default="semantic", choices=["semantic", "hybrid"],
                        help="Analysis method: 'semantic' (default) or 'hybrid' (Cluster-First)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    matches = []
    trends = {} # This will store the combined trends dictionary
    all_posts = [] # This will store all posts (FB + News)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if args.input:
        if os.path.exists(args.input):
            console.print(f"[bold green]📂 Loading matches from: {args.input}[/bold green]")
            with open(args.input, 'r', encoding='utf-8') as f:
                matches = json.load(f)
            console.print(f"[bold green]✅ Loaded {len(matches)} matches.[/bold green]")
        else:
            console.print(f"[bold red]❌ Input file not found: {args.input}[/bold red]")
            return
    else:
        # --- 1. Load Trend Data (Google Trends) ---
        trend_files = glob.glob(os.path.join("crawlers", "trendings", "*.csv"))
        if not trend_files:
            # Fallback to current directory
            trend_files = glob.glob("trending_*.csv")
        
        all_trends = {}
        if trend_files:
            console.print(f"[green]Found {len(trend_files)} trend CSV files.[/green]")
            all_trends = load_trends(trend_files)
        else:
            console.print("[red]No trend CSV files found![/red]")

        # --- 2. Load Social Data (Facebook) ---
        fb_files = glob.glob(os.path.join("crawlers", "facebook", "*.json"))
        if fb_files:
            console.print(f"[green]Found {len(fb_files)} Facebook JSON files.[/green]")
            for f in fb_files:
                fb_data = load_json(f)
                all_posts.extend(fb_data)
        else:
            console.print("[yellow]No Facebook JSON files found in crawlers/facebook/.[/yellow]")

        # --- 3. Load News Data ---
        news_files = glob.glob(os.path.join("crawlers", "news", "**", "*.csv"), recursive=True)
        # Also try default data dir
        if os.path.exists(os.path.join(project_root, "data")):
             news_files.extend(glob.glob(os.path.join(project_root, "data", "**", "*.csv"), recursive=True))

        if news_files:
             console.print(f"[green]Found {len(news_files)} news CSV files.[/green]")
             for f in news_files:
                 # load_news_articles expects a directory, but let's change logic or just pass file
                 # Wait, load_news_articles takes a directory. Let's fix loop to use load_news_csv if I implemented it, 
                 # or reuse load_news_articles logic.
                 # The existing load_news_articles logic iterates globs itself.
                 pass
             
             # Simpler: just use the function I saw earlier:
             news_data = load_news_articles(os.path.join(project_root, "data")) # Fallback
             # But wait, user's news data might be elsewhere. 
             # Let's inspect the files found.
             # Actually, let's just stick to the text seen in file view: load_news_articles takes data_dir.
             
             # Re-implement explicit file loading for clarity:
             for f in news_files:
                 # Simple loader inline or call function?
                 # Let's use the one in the file: load_news_articles matches "data_dir/**/*.csv".
                 pass
        
        # Let's rely on the previous implementation of load_news_articles but fix the path
        # Actually I will just replace the block with robust loading.
        
        news_data = [] 
        # Search in project root/data
        if os.path.exists(os.path.join(project_root, "data")):
             news_data.extend(load_news_articles(os.path.join(project_root, "data")))
        
        # Search in crawlers/news
        if os.path.exists(os.path.join("crawlers", "news")):
             news_data.extend(load_news_articles(os.path.join("crawlers", "news")))

        all_posts.extend(news_data)

        console.print(f"[bold]Loaded Total Data:[/bold] {len(all_posts)} posts.")
        
        if args.limit and args.limit > 0:
            all_posts = all_posts[:args.limit]
            console.print(f"[yellow]⚠️ Limiting analysis to first {len(all_posts)} posts.[/yellow]")

        console.print(f"[bold cyan]🤖 Using Model: {args.model}[/bold cyan]")
        console.print(f"[bold]Loaded Trends:[/bold] {len(all_trends)} trends.")
        
        if not all_posts or not all_trends:
            console.print("[red]No data to analyze![/red]")
            return


        
        if args.method == "hybrid":
            console.print("[bold cyan]🚀 Running Hybrid Analysis (Cluster -> Match)...[/bold cyan]")
            matches = find_matches_hybrid(
                all_posts,
                all_trends,
                model_name=args.model,
                threshold=args.threshold,
                use_aliases=not args.no_aliases,
                use_ner=args.use_ner,
                embedding_method=args.embedding,
                save_all=args.save_all
            )
        else:
            matches = find_matches(
                all_posts, 
                all_trends, 
                model_name=args.model, 
                threshold=args.threshold, 
                use_aliases=not args.no_aliases,
                use_ner=args.use_ner,
                embedding_method=args.embedding,
                save_all=args.save_all,
                max_days=args.max_days
            )
        
        if args.output:
            # Ensure dir exists
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            console.print(f"[bold green]💾 Saving results to: {args.output}[/bold green]")
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(matches, f, ensure_ascii=False, indent=2)

    if not matches:
        console.print("[yellow]No matches found between trends and data.[/yellow]")
        return
    
    # Analyze trend coverage
    valid_trends = analyze_trend_coverage(matches, trends, min_posts=args.min_posts)
    
    console.print(f"\n[bold green]✅ Analysis complete![/bold green]")
    console.print(f"[dim]Run evaluate_trends.py for visualization:[/dim]")
    console.print(f"[cyan]  python crawlers/evaluate_trends.py --input {args.output}[/cyan]")


if __name__ == "__main__":
    main()
