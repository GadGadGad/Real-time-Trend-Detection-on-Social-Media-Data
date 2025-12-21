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
from rich.table import Table
import argparse
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from crawlers.clustering import cluster_data, extract_cluster_labels
from crawlers.alias_normalizer import normalize_with_aliases, build_alias_dictionary, batch_normalize_texts
from crawlers.ner_extractor import enrich_text_with_entities, batch_enrich_texts, HAS_NER
from crawlers.sentiment import batch_analyze_sentiment
from crawlers.vectorizers import get_embeddings
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


import re

def clean_text(text):
    """
    Remove common social media noise like credits.
    """
    if not text:
        return ""
        
    # Remove "Cre:", "Credit:", "Nguồn:", "Via:" followed by text
    # Case insensitive, handles various separators
    patterns = [
        r'(?i)\b(cre|credit|via|nguồn)\s*[:.-]\s*.*$', # Matches "Cre: abc" to end of line/string
        r'(?i)\b(cre|credit)\s+by\s*[:.-]?\s*.*$'
    ]
    
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, '', cleaned)
    
    return cleaned.strip()

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
                    "content": clean_text(text), # Apply cleaning
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




def parse_trend_volume(vol_str):
    """
    Parse Google Trends volume string (e.g., '100 N+', '20 N+') to integer.
    'N+' roughly translates to '000+'.
    """
    if not vol_str:
        return 0
        
    try:
        # Remove non-numeric except 'N' or 'K' or 'M' if they exist, but typical VN format is "100 N+"
        clean_str = vol_str.upper().replace(',', '').replace('.', '')
        
        multiplier = 1
        if 'N' in clean_str: # "Ngàn" ~ Thousand
            multiplier = 1000
        elif 'TR' in clean_str or 'M' in clean_str: # "Triệu" ~ Million
            multiplier = 1000000
        elif 'K' in clean_str:
            multiplier = 1000
            
        # Extract numbers
        num_part = re.findall(r'\d+', clean_str)
        if num_part:
            return int(num_part[0]) * multiplier
        return 0
    except:
        return 0

def load_trends(csv_files):
    """Load trends from multiple CSV files. Returns dict with keywords and volume."""
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
                    volume_str = row[1].strip() # Assuming 2nd column is volume "100 N+"
                    related_keywords = row[4].split(',')
                    clean_keywords = [k.strip() for k in related_keywords if k.strip()]
                    
                    if main_trend not in clean_keywords:
                        clean_keywords.insert(0, main_trend)
                    
                    volume = parse_trend_volume(volume_str)
                        
                    trends[main_trend] = {
                        "keywords": clean_keywords,
                        "volume": volume
                    }
                    
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
    """
    start_time = datetime.now()
    matches = []
    
    # 0. Date Filtering
    if max_days is not None:
        console.print(f"[bold yellow]🕒 Filtering posts older than {max_days} days...[/bold yellow]")
        cutoff_date = start_time - timedelta(days=max_days)
        filtered_posts = []
        for p in posts:
            t_str = str(p.get('time', ''))
            p_time = None
            try:
                p_time = parser.parse(t_str).replace(tzinfo=None)
            except:
                ts = p.get('timestamp')
                if ts:
                    try:
                        p_time = datetime.fromtimestamp(int(ts))
                    except:
                        pass
            
            if p_time and p_time < cutoff_date:
                continue
            filtered_posts.append(p)
        posts = filtered_posts

    # Build alias dictionary from trends (using keywords from new structure)
    if use_aliases and not use_ner:
        # Re-build alias dict using the new structure if needed, 
        # but build_alias_dictionary expects {trend: [keywords]}.
        # We need to adapt it or pass a simplified dict.
        simple_trends = {k: v['keywords'] for k, v in trends.items()}
        build_alias_dictionary(simple_trends)
    
    # Prepare texts
    trend_keys = list(trends.keys())
    # Access keywords from new structure
    trend_texts = [f"{t} " + " ".join(trends[t]['keywords'][:5]) for t in trend_keys]
    
    post_contents = [p.get('content', '')[:500] for p in posts]
    
    # Apply text enrichment
    if use_ner and HAS_NER:
        console.print("[bold magenta]🏷️ Enriching texts with NER (underthesea)...[/bold magenta]")
        trend_texts = batch_enrich_texts(trend_texts, weight_factor=2)
        post_contents = batch_enrich_texts(post_contents, weight_factor=2)
    elif use_aliases and not use_ner:
        console.print("[bold magenta]🏷️ Enriching texts with Alias Normalization...[/bold magenta]")
        # normalize_with_aliases uses the global dict built above
        post_contents = batch_normalize_texts(post_contents)

    console.print(f"[cyan]🔄 Generating embeddings for {len(posts)} posts and {len(trends)} trends...[/cyan]")
    
    # Generate embeddings
    post_embeddings = get_embeddings(post_contents, method=embedding_method, model_name=model_name)
    trend_embeddings = get_embeddings(trend_texts, method=embedding_method, model_name=model_name)
    
    # Calculate Similarity
    console.print("[bold cyan]➗ Computing Cosine Similarity...[/bold cyan]")
    similarity_matrix = cosine_similarity(post_embeddings, trend_embeddings)
    
    matched_indices = set()
    
    for i, post in enumerate(posts):
        best_trend_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_trend_idx]
        
        # Check interactions
        stats = post.get('stats', {})
        total_interactions = stats.get('likes', 0) + stats.get('comments', 0) + stats.get('shares', 0)
        
        is_matched = False
        topic = "Unassigned"
        
        if best_score >= threshold and total_interactions >= min_interactions:
            topic = trend_keys[best_trend_idx]
            is_matched = True
            matched_indices.add(i)
        
        if is_matched or save_all:
            matches.append({
                "source": post.get('source'),
                "time": post.get('time'),
                "post_content": post.get('content'),
                "trend": topic,
                "score": float(best_score),
                "is_matched": is_matched,
                "interactions": total_interactions
            })
            
    return matches


def calculate_unified_score(trend_data, posts_in_cluster, w_g=0.3, w_f=0.5, w_n=0.2):
    """
    Calculate Unified Trend Score (T) = w_G*G + w_F*F + w_N*N.
    Normalize components to 0-100 scale roughly.
    """
    # G-Score: Google Search Volume
    g_raw = trend_data.get('volume', 0)
    # Log scale for G because it can be 100k+
    # 100k -> 5, 10k -> 4. Map to 0-100?
    # Let's say 200k+ is max (100). 
    g_score = min(100, (np.log10(g_raw + 1) / 6.0) * 100) if g_raw > 0 else 0
    
    # F-Score: Facebook Interactions
    f_raw = sum([
        p.get('stats', {}).get('likes', 0) + 
        p.get('stats', {}).get('comments', 0) + 
        p.get('stats', {}).get('shares', 0) 
        for p in posts_in_cluster if 'Face' in p.get('source', '')
    ])
    # Cap at 10k interactions?
    f_score = min(100, (f_raw / 5000) * 100)
    
    # N-Score: News Count
    n_raw = sum([1 for p in posts_in_cluster if 'Face' not in p.get('source', '')])
    # Cap at 20 articles
    n_score = min(100, (n_raw / 10) * 100)
    
    final_score = (w_g * g_score) + (w_f * f_score) + (w_n * n_score)
    return round(final_score, 2), {"G": round(g_score,1), "F": round(f_score,1), "N": round(n_score,1)}


def find_matches_hybrid(posts, trends, model_name=None, threshold=0.5, 
                        use_aliases=True, use_ner=False, 
                         embedding_method="sentence-transformer", save_all=False,
                        rerank=True, min_cluster_size=5, labeling_method="semantic"):
    """
    Cluster-First approach with Cross-Encoder Reranking + Scoring + Sentiment.
    """
    if not posts:
        return []

    # --- 0. Initialize Models ---
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL)
    
    reranker = None
    if rerank:
        console.print("[bold yellow]⚡ Loading Cross-Encoder for Precision Reranking...[/bold yellow]")

    if use_ner and HAS_NER:
        console.print("[bold magenta]🏷️ Enriching texts with NER...[/bold magenta]")
        trend_texts = batch_enrich_texts(trend_texts, weight_factor=2)
        post_contents = batch_enrich_texts(post_contents, weight_factor=2)
    elif use_aliases:
        console.print("[bold magenta]🔄 Normalizing texts with aliases...[/bold magenta]")
        trend_texts = batch_normalize_texts(trend_texts, show_progress=False)
        post_contents = batch_normalize_texts(post_contents, show_progress=True)

    # Encode All
    console.print(f"[bold cyan]🧠 Encoding {len(post_contents)} posts for Hybrid Analysis (method={embedding_method})...[/bold cyan]")
    
    # 1. Embeddings for CLUSTERING (Variable: TF-IDF, BoW, or SentenceTransformer)
    clustering_embeddings = get_embeddings(post_contents, method=embedding_method, model_name=model_name or DEFAULT_MODEL)
    
    # 2. Embeddings for MATCHING (Always SentenceTransformer for Semantic Match)
    # We need these later for matching clusters to trends
    if embedding_method == "sentence-transformer":
        post_embeddings = clustering_embeddings # Reuse if same
    else:
        # If clustering with TF-IDF/BoW, we still need dense matching embeddings later?
        # Actually, Step 3 matches Cluster Keywords -> Trends.
        # But for 'Discovery', we might want dense embeddings?
        # Let's keep it simple: Matching uses string-based Bi-Encoder query.
        pass

    # 2. Cluster Everything (HDBSCAN)
    console.print(f"[bold cyan]🧩 Running HDBSCAN on ALL posts (min_size={min_cluster_size})...[/bold cyan]")
    cluster_labels = cluster_data(clustering_embeddings, min_cluster_size=min_cluster_size)
    
    # Use Semantic Labeling (KeyBERT-style) or TF-IDF
    cluster_names = extract_cluster_labels(post_contents, cluster_labels, model=embedder, method=labeling_method) 
    
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
