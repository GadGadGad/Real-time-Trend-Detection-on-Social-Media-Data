import json
import csv
import os
import glob
import numpy as np
from rich.console import Console
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

console = Console()

# ... (load_json, load_news_articles, load_trends function definitions remain the same) ...

console = Console()

def load_json(filepath):
    """Load Facebook data from JSON."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Normalize to unified structure
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
            # Extract source from parent directory name
            source_name = os.path.basename(os.path.dirname(filepath)).upper()
            
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    content = row.get('content', '')
                    title = row.get('title', '')
                    
                    # Combine title and content for better matching context
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
                header = next(reader) # Skip header
                
                for row in reader:
                    if len(row) < 5: continue
                    
                    main_trend = row[0].strip()
                    related_keywords = row[4].split(',')
                    
                    # Clean up related keywords
                    clean_keywords = [k.strip() for k in related_keywords if k.strip()]
                    
                    # Add main trend to keyword list if not present
                    if main_trend not in clean_keywords:
                        clean_keywords.insert(0, main_trend)
                        
                    trends[main_trend] = clean_keywords
                    
        except Exception as e:
            console.print(f"[red]Error loading CSV {filepath}: {e}[/red]")
            
    return trends

def find_matches(posts, trends, threshold=0.58, min_interactions=10, model_name='all-MiniLM-L6-v2', save_all=False): 
    """
    Find matches using Semantic Similarity (Embeddings).
    Filters:
    - Cosine Similarity > threshold (default 0.58 for stricter matching)
    - Total Interactions (Likes+Comments+Shares) > min_interactions (default 10)
    """
    matches = []
    
    # 1. Prepare Trend Strings (Main Trend + Keywords joined)
    trend_keys = list(trends.keys())
    trend_texts = []
    for t in trend_keys:
        # Join main trend and top 3 keywords to form a descriptive text
        keywords = trends[t]
        context = f"{t} " + " ".join(keywords[:5]) 
        trend_texts.append(context)

    # 2. Encode Trends
    console.print(f"[bold cyan]🧠 Encoding {len(trend_texts)} Trends with {model_name}...[/bold cyan]")
    model = SentenceTransformer(model_name)
    trend_embeddings = model.encode(trend_texts, show_progress_bar=True)

    # 3. Encode Posts
    console.print(f"[bold cyan]🧠 Encoding {len(posts)} Posts...[/bold cyan]")
    post_contents = [p.get('content', '')[:500] for p in posts] # Truncate for speed/noise reduction
    post_embeddings = model.encode(post_contents, show_progress_bar=True)


    console.print("[bold cyan]📐 Calculating Cosine Similarity...[/bold cyan]")
    similarity_matrix = cosine_similarity(post_embeddings, trend_embeddings)

    # For each post, find the Trend with the MAX similarity.
    
    count_matched = 0
    count_unmatched = 0

    for i, post in enumerate(posts):
        stats = post.get('stats', {'likes': 0, 'comments': 0, 'shares': 0})
        total_interactions = stats.get('likes', 0) + stats.get('comments', 0) + stats.get('shares', 0)
        
        # Filter by Engagement (only for Facebook posts, identified by Source starting with Face or having stats)
        # News articles usually have 0 stats in this dataset, so we might skip this filter for them 
        # or assume if stats are missing/zero it's a News article (checked by source name).
        is_facebook = 'Face' in post.get('source', '')
        
        # If save_all is False, apply strict filters early
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
            # Add Unmatched items if save_all is True
            # Optional: Relax interaction filter for saved unmatched items? 
            # For now, keeping everything if save_all is True allows clustering to decide.
            match_data.update({
                "trend": "Unassigned",
                "keyword": "none", 
                "score": float(best_score), # Store best score anyway to see "closeness"
                "closest_trend": trend_keys[best_trend_idx],
                "is_matched": False
            })
            matches.append(match_data)
            count_unmatched += 1

    # Statistics for Score Distribution
    scores = [m['score'] for m in matches if m.get('is_matched')]
    if scores:
        avg_score = sum(scores)/len(scores)
        console.print(f"[bold yellow]📊 Sem Sim Stats: Min={min(scores):.2f}, Max={max(scores):.2f}, Avg={avg_score:.2f}[/bold yellow]")
    
    console.print(f"[dim]Stats: Matched={count_matched}, Unmatched (Saved)={count_unmatched}[/dim]")

    return matches


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Search Trends vs Social Data")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                        help="HuggingFace model name for embeddings (e.g. 'paraphrase-multilingual-mpnet-base-v2')")
    parser.add_argument("--threshold", type=float, default=0.58, 
                        help="Similarity threshold for matching (default: 0.58)")
    parser.add_argument("--output", default="results.json", type=str, help="Path to save matched results (JSON)")
    parser.add_argument("--input", type=str, help="Path to load matched results from (JSON), skipping matching process")
    parser.add_argument("--save-all", action="store_true", help="Save all posts (including unmatched) for unsupervised clustering")
    return parser.parse_args()

def main():
    args = parse_args()
    
    matches = []
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Loop back to project root
    
    fb_file = os.path.join(script_dir, "facebook", "fb_data.json")
    news_data_dir = os.path.join(project_root, "data")
    
    csv_files = [
        os.path.join(script_dir, "trending_VN_7d_20251208-1451.csv"),
        os.path.join(script_dir, "trending_VN_7d_20251208-1452.csv")
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
            
        # Ensure we have trends dict for coloring if possible, or infer from matches
        # For simplicity, we'll reload trends from source to get the full list for stats
        trends = load_trends(csv_files) 

    else:
        # Load all data
        fb_data = load_json(fb_file)
        news_data = load_news_articles(news_data_dir)
        all_data = fb_data + news_data
        
        console.print(f"[bold]Loaded Total Data:[/bold] {len(fb_data)} FB posts + {len(news_data)} News articles = {len(all_data)} items")
        console.print(f"[bold cyan]🤖 Using Model: {args.model}[/bold cyan]")
        
        trends = load_trends(csv_files)
        
        if not all_data or not trends:
            console.print("[red]No data to analyze![/red]")
            return

        # Pass model name to find_matches
        matches = find_matches(all_data, trends, model_name=args.model, threshold=args.threshold, save_all=args.save_all)
        
        if args.output:
            console.print(f"[bold green]💾 Saving results to: {args.output}[/bold green]")
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(matches, f, ensure_ascii=False, indent=2)

    if not matches:
        console.print("[yellow]No matches found between trends and data.[/yellow]")
        return
    
    console.print(f"\n[bold green]✅ Analysis complete! Found {len(matches)} matches.[/bold green]")
    console.print(f"[dim]Run evaluate_trends.py for visualization and metrics:[/dim]")
    console.print(f"[cyan]  python crawlers/evaluate_trends.py --input {args.output} --filter-routine --show-routine[/cyan]")

if __name__ == "__main__":
    main()
