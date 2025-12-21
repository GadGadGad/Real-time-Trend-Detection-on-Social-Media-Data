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


# Import Taxonomy Classifier
try:
    from crawlers.taxonomy_classifier import TaxonomyClassifier
except ImportError:
    try:
        from taxonomy_classifier import TaxonomyClassifier
    except ImportError:
        TaxonomyClassifier = None
        console.print("[yellow]⚠️ TaxonomyClassifier module not found. Skipping categorization.[/yellow]")


def extract_dynamic_anchors(posts, trends, top_n=20):
    """
    Automatically identify 'Anchor' words that exist in both Google Trends 
    and are frequent in the current post batch.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    
    # 1. Collect all trend keywords
    trend_kws = set()
    for t_data in trends.values():
        for kw in t_data.get('keywords', []):
            trend_kws.add(kw.lower())
            
    if not trend_kws:
        return []
        
    # 2. Find frequent keywords in posts (Unigrams & Bigrams)
    texts = [p.get('content', '').lower() for p in posts]
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
    try:
        X = vectorizer.fit_transform(texts)
        post_words = vectorizer.get_feature_names_out()
        word_counts = X.toarray().sum(axis=0)
        
        # 3. Intersect
        anchors = []
        for word, count in zip(post_words, word_counts):
            if word in trend_kws:
                anchors.append((word, count))
                
        # Sort by frequency
        anchors = sorted(anchors, key=lambda x: x[1], reverse=True)
        final_anchors = [a[0] for a in anchors[:top_n]]
        
        if final_anchors:
            console.print(f"[bold magenta]⚓ Found {len(final_anchors)} Dynamic Anchors:[/bold magenta] {', '.join(final_anchors[:5])}...")
            
        return final_anchors
    except:
        return []

def apply_guidance_enrichment(text, anchors):
    """Prepend anchors found in text to boost their embedding signal."""
    found = [a for a in anchors if a in text.lower()]
    if found:
        # Prepend with a special tag to trick the embedder into focusing
        # Doubling the words increases their 'attention' weight in most models
        prefix = " ".join(found * 2) 
        return f"{prefix} | {text}"
    return text


def find_matches_hybrid(posts, trends, model_name=None, threshold=0.5, 
                        use_aliases=True, use_ner=False, 
                         embedding_method="sentence-transformer", save_all=False,
                        rerank=True, min_cluster_size=5, labeling_method="semantic",
                        reranker_model_name=None):
    """
    Cluster-First approach with Cross-Encoder Reranking + Scoring + Sentiment + Taxonomy.
    """
    if not posts:
        return []

    # --- 0. Initialize Models ---
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL)
    
    # Initialize Taxonomy Classifier
    taxonomy_clf = None
    if TaxonomyClassifier:
        taxonomy_clf = TaxonomyClassifier(embedding_model=embedder)
    
    reranker = None
    if rerank:
        ce_model = reranker_model_name or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        console.print(f"[bold yellow]⚡ Loading Cross-Encoder ({ce_model}) for Precision Reranking...[/bold yellow]")
        try:
            reranker = CrossEncoder(ce_model) 
        except Exception as e:
             console.print(f"[red]Failed to load CrossEncoder: {e}. Skipping rerank.[/red]")
             reranker = None

    # --- 1. Preprocessing & Enrichment ---
    console.print("[bold cyan]🧹 Preprocessing & Enriching Texts...[/bold cyan]")
    post_contents = [p.get('content', '')[:500] for p in posts]
    
    # NEW: Extract Dynamic Anchors for Guidance
    anchors = extract_dynamic_anchors(posts, trends)
    
    if use_ner and HAS_NER:
        post_contents_enriched = batch_enrich_texts(post_contents, weight_factor=2)
    elif use_aliases:
        # Build dictionary from new trend structure
        simple_trends = {k: v['keywords'] for k, v in trends.items()}
        build_alias_dictionary(simple_trends)
        post_contents_enriched = batch_normalize_texts(post_contents)
    else:
        post_contents_enriched = post_contents

    # NEW: Apply Guidance Enrichment (Bias clustering toward anchors)
    if anchors:
        post_contents_enriched = [apply_guidance_enrichment(t, anchors) for t in post_contents_enriched]

    # --- 2. Generate Embeddings (for Clustering) ---
    console.print(f"[bold cyan]🧠 Generating Embeddings ({embedding_method})...[/bold cyan]")
    post_embeddings = get_embeddings(post_contents_enriched, method=embedding_method, model_name=model_name)
    
    # --- 3. Clustering (Discovery) ---
    console.print(f"[bold cyan]🧩 Running Hybrid Clustering (Min Size={min_cluster_size})...[/bold cyan]")
    cluster_labels = cluster_data(post_embeddings, min_cluster_size=min_cluster_size)
    
    unique_labels = sorted([l for l in set(cluster_labels) if l != -1])
    console.print(f"[green]Found {len(unique_labels)} clusters.[/green]")

    # --- 4. Sentiment Analysis (Batch) ---
    console.print("[bold cyan]😊 Analyzing Sentiment...[/bold cyan]")
    sentiments = batch_analyze_sentiment(post_contents) # Analyze original text

    # --- 5. Match Clusters to Trends ---
    # Prepare Trend Embeddings (Keywords)
    trend_keys = list(trends.keys())
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    
    if trend_queries:
        console.print(f"[dim]Encoding {len(trend_queries)} trends for matching...[/dim]")
        trend_embeddings = embedder.encode(trend_queries)
    else:
        trend_embeddings = []

    # NEW: Pass anchors to labeling
    cluster_names = extract_cluster_labels(post_contents, cluster_labels, model=embedder, method=labeling_method, anchors=anchors)

    # Map cluster_label to assigned trend and score
    cluster_mapping = {}
    
    for label in unique_labels:
        indices = [i for i, l in enumerate(cluster_labels) if l == label]
        cluster_posts = [posts[i] for i in indices]
        cluster_query = cluster_names.get(label, f"Cluster {label}")
        
        assigned_trend = "Discovery" 
        topic_type = "Discovery"
        best_match_score = 0.0
        
        # Matching against Google Trends
        if len(trend_embeddings) > 0:
            cluster_emb = embedder.encode(cluster_query)
            sims = cosine_similarity([cluster_emb], trend_embeddings)[0]
            top_k_idx = np.argsort(sims)[-3:][::-1] # Top 3 candidates
            
            if rerank and reranker:
                candidates = [trend_keys[k] for k in top_k_idx]
                rerank_pairs = [(cluster_query, trend_queries[k]) for k in top_k_idx]
                rerank_scores = reranker.predict(rerank_pairs)
                best_sub_idx = np.argmax(rerank_scores)
                
                # Use a combined score for logic (CE score > -2 is usually a match)
                if rerank_scores[best_sub_idx] > -2:
                    best_match_score = float(sims[top_k_idx[best_sub_idx]]) # Keep bi-encoder score for display
                    assigned_trend = candidates[best_sub_idx]
                    topic_type = "Trending"
            else:
                if sims[top_k_idx[0]] > threshold:
                    best_match_score = float(sims[top_k_idx[0]])
                    assigned_trend = trend_keys[top_k_idx[0]]
                    topic_type = "Trending"
        
        # Trend Scoring
        trend_data = trends.get(assigned_trend, {'volume': 0})
        unified_score, components = calculate_unified_score(trend_data, cluster_posts)
        
        # Taxonomy Classification
        if taxonomy_clf:
            category, cat_method = taxonomy_clf.classify(cluster_query + " " + assigned_trend)
        else:
            category, cat_method = "Unclassified", "None"

        cluster_mapping[label] = {
            "final_topic": assigned_trend if assigned_trend != "Discovery" else f"New: {cluster_query}",
            "topic_type": topic_type,
            "cluster_name": cluster_query,
            "category": category,
            "category_method": cat_method,
            "match_score": best_match_score,
            "trend_score": unified_score,
            "score_components": components
        }
    # Merge clusters that map to the same high-confidence trend
    consolidated_mapping = {}
    topic_to_labels = {}
    
    for label, m in cluster_mapping.items():
        topic = m["final_topic"]
        if topic not in topic_to_labels:
            topic_to_labels[topic] = []
        topic_to_labels[topic].append(label)
        
    for topic, labels in topic_to_labels.items():
        # Combine indices and posts for score recalculation
        all_indices = []
        all_posts = []
        for l in labels:
            idx = [i for i, val in enumerate(cluster_labels) if val == l]
            all_indices.extend(idx)
            all_posts.extend([posts[i] for i in idx])
        
        # Pick the representation from the "main" cluster
        main_label = labels[0]
        m = cluster_mapping[main_label]
        
        # RE-CALCULATE SCORE on consolidated data
        trend_obj = trends.get(m["final_topic"], {'volume': 0})
        combined_score, combined_components = calculate_unified_score(trend_obj, all_posts)
        
        consolidated_mapping[topic] = {
            "final_topic": topic,
            "topic_type": m["topic_type"],
            "cluster_name": m["cluster_name"], # Keep first or merge? Let's keep first
            "category": m["category"],
            "category_method": m["category_method"],
            "match_score": m["match_score"],
            "trend_score": combined_score,
            "score_components": combined_components
        }
        
    for topic, cm in consolidated_mapping.items():
        if cm["topic_type"] == "Trending":
             console.print(f"   ✅ Consolidated Trend: [green]{topic}[/green] (Score: {cm['trend_score']})")
        else:
             # For discovery, we only merge if they are extremely close? 
             # Actually, for discovery topics, they are already separate clusters.
             # But if they share the same Discovery name (rare), they merge too.
             pass

    # --- 6. Construct Results per Post ---
    matches = []
    for i, post in enumerate(posts):
        label = cluster_labels[i]
        stats = post.get('stats', {'likes': 0, 'comments': 0, 'shares': 0})
        
        if label != -1:
            # Look up by topic name now
            topic_name = cluster_mapping[label]["final_topic"]
            m = consolidated_mapping[topic_name]
            
            match_data = {
                "source": post.get('source'),
                "time": post.get('time'),
                "post_content": post.get('content'),
                "processed_content": post_contents_enriched[i],
                "final_topic": m["final_topic"],
                "topic_type": m["topic_type"],
                "cluster_name": m["cluster_name"],
                "category": m["category"],
                "category_method": m["category_method"],
                "score": m["match_score"], 
                "trend_score": m["trend_score"],
                "score_components": m["score_components"],
                "sentiment": sentiments[i],
                "stats": stats,
                "is_matched": (m["topic_type"] == "Trending"),
                "trend": m["final_topic"] if m["topic_type"] == "Trending" else "Unassigned",
                "cluster_id": int(label)
            }
            matches.append(match_data)
        elif save_all:
             matches.append({
                "source": post.get('source'),
                "time": post.get('time'),
                "post_content": post.get('content'),
                "processed_content": post_contents_enriched[i],
                "final_topic": "Unassigned",
                "topic_type": "Noise",
                "cluster_name": "Noise",
                "category": "Noise",
                "category_method": "None",
                "score": 0.0,
                "trend_score": 0,
                "score_components": {},
                "sentiment": sentiments[i],
                "stats": stats,
                "is_matched": False,
                "trend": "Unassigned",
                "cluster_id": -1
            })

    console.print(f"[bold green]✅ Hybrid Analysis Complete: {len(matches)} posts saved.[/bold green]")
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
