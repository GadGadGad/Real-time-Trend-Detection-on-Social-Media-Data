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
from rich.console import Console
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

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

# Best model for Vietnamese semantic similarity
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"

console = Console()


def load_json(filepath):
    """Load Facebook data from JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            unified = []
            for item in data:
                unified.append({
                    "source": f"Face: {item.get('page_name', 'Unknown')}",
                    "content": item.get('content', ''),
                    "title": "",
                    "url": "",
                    "stats": item.get('stats', {'likes': 0, 'comments': 0, 'shares': 0}),
                    "time": item.get('time_label', '')
                })
            return unified
    except Exception as e:
        console.print(f"[red]Error loading JSON {filepath}: {e}[/red]")
        return []


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


def find_matches(posts, trends, model_name=None, threshold=0.55, 
                 min_interactions=10, use_aliases=True, use_ner=False, save_all=False):
    """
    Find matches using Semantic Similarity with text enrichment.
    
    Args:
        posts: List of post dictionaries
        trends: Dictionary of trends with keywords
        model_name: HuggingFace model for embeddings
        threshold: Cosine similarity threshold
        min_interactions: Minimum engagement for FB posts
        use_aliases: Whether to use alias normalization (default)
        use_ner: Whether to use NER enrichment (alternative to aliases)
        save_all: Save unmatched posts too
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
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
    console.print(f"[bold cyan]🧠 Encoding {len(trend_texts)} Trends with {model_name}...[/bold cyan]")
    model = SentenceTransformer(model_name)
    trend_embeddings = model.encode(trend_texts, show_progress_bar=True)

    console.print(f"[bold cyan]🧠 Encoding {len(posts)} Posts...[/bold cyan]")
    post_embeddings = model.encode(post_contents, show_progress_bar=True)

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
            "stats": stats
        }

        if best_score > threshold:
            trend_name = trend_keys[best_trend_idx]
            match_data.update({
                "trend": trend_name,
                "keyword": "semantic-match", 
                "score": float(best_score),
                "is_matched": True
            })
            matches.append(match_data)
            count_matched += 1
        elif save_all:
            match_data.update({
                "trend": "Unassigned",
                "keyword": "none", 
                "score": float(best_score),
                "closest_trend": trend_keys[best_trend_idx],
                "is_matched": False
            })
            matches.append(match_data)
            count_unmatched += 1

    # Stats
    scores = [m['score'] for m in matches if m.get('is_matched')]
    if scores:
        console.print(f"[bold yellow]📊 Score Stats: Min={min(scores):.2f}, Max={max(scores):.2f}, Avg={sum(scores)/len(scores):.2f}[/bold yellow]")
    
    console.print(f"[dim]Stats: Matched={count_matched}, Unmatched={count_unmatched}[/dim]")

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
    parser.add_argument("--output", default="results.json", type=str, 
                        help="Path to save matched results (JSON)")
    parser.add_argument("--input", type=str, 
                        help="Path to load existing results (skip matching)")
    parser.add_argument("--no-aliases", action="store_true", default=False,
                        help="Disable alias normalization")
    parser.add_argument("--use-ner", action="store_true", default=False,
                        help="Use NER enrichment instead of aliases (requires underthesea)")
    parser.add_argument("--save-all", action="store_true", 
                        help="Save all posts including unmatched")
    parser.add_argument("--min-posts", type=int, default=3,
                        help="Minimum posts for a valid trend (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    matches = []
    trends = {}
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    fb_file = os.path.join(script_dir, "facebook", "fb_data.json")
    news_data_dir = os.path.join(project_root, "data")
    
    # Find all trend CSV files
    csv_pattern = os.path.join(script_dir, "trending_VN_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        console.print("[yellow]No trend CSV files found. Using default paths.[/yellow]")
        csv_files = [
            os.path.join(script_dir, "trending_VN_7d_20251208-1451.csv"),
        ]

    if args.input:
        if os.path.exists(args.input):
            console.print(f"[bold green]📂 Loading matches from: {args.input}[/bold green]")
            with open(args.input, 'r', encoding='utf-8') as f:
                matches = json.load(f)
            console.print(f"[bold green]✅ Loaded {len(matches)} matches.[/bold green]")
        else:
            console.print(f"[bold red]❌ Input file not found: {args.input}[/bold red]")
            return
        
        trends = load_trends(csv_files)

    else:
        # Load all data
        fb_data = load_json(fb_file)
        news_data = load_news_articles(news_data_dir)
        all_data = fb_data + news_data
        
        console.print(f"[bold]Loaded Total Data:[/bold] {len(fb_data)} FB posts + {len(news_data)} News articles = {len(all_data)} items")
        console.print(f"[bold cyan]🤖 Using Model: {args.model}[/bold cyan]")
        
        trends = load_trends(csv_files)
        console.print(f"[bold]Loaded Trends:[/bold] {len(trends)} trends from {len(csv_files)} CSV files")
        
        if not all_data or not trends:
            console.print("[red]No data to analyze![/red]")
            return

        matches = find_matches(
            all_data, 
            trends, 
            model_name=args.model, 
            threshold=args.threshold, 
            use_aliases=not args.no_aliases,
            use_ner=args.use_ner,
            save_all=args.save_all
        )
        
        if args.output:
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
