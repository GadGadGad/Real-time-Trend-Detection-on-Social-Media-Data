"""
Evaluate Trend Analysis Results
Visualizes trends, computes scores, and generates reports.

This version uses direct trend assignment from semantic matching,
NOT HDBSCAN clustering (which doesn't work well for this data).
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sentence_transformers import SentenceTransformer
import re
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import os

try:
    from crawlers.trend_scoring import ScoreCalculator
except ImportError:
    try:
        from trend_scoring import ScoreCalculator
    except ImportError:
        ScoreCalculator = None

try:
    from crawlers.vectorizers import get_embeddings
except ImportError:
    from vectorizers import get_embeddings

# Best model for Vietnamese semantic similarity
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"

console = Console()


# ============================================================================
# ROUTINE TREND FILTERING
# ============================================================================

DEFAULT_BLACKLIST_PATTERNS = [
    r'(không khí lạnh|thời tiết|dự báo thời tiết|mưa lớn|bão số|nhiệt độ|nắng nóng)',
    r'(giá xăng|giá vàng|tỷ giá|giá dầu|giá điện)',
    r'(kết quả xổ số|lịch thi đấu|nhận định trận)',
    r'(tình hình giao thông|kẹt xe|ùn tắc)',
]


def is_routine_by_blacklist(trend_name: str, patterns: list) -> bool:
    """Check if trend matches any blacklist pattern."""
    trend_lower = trend_name.lower()
    for pattern in patterns:
        if re.search(pattern, trend_lower):
            return True
    return False


def filter_routine_trends(trends_data: dict, patterns: list = None) -> tuple:
    """Filter out routine/recurring trends."""
    if patterns is None:
        patterns = DEFAULT_BLACKLIST_PATTERNS
    
    event_trends = {}
    routine_trends = {}
    
    for trend_name, data in trends_data.items():
        if is_routine_by_blacklist(trend_name, patterns):
            routine_trends[trend_name] = data
        else:
            event_trends[trend_name] = data
    
    return event_trends, routine_trends


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trend Analysis Results")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to matched results JSON")
    parser.add_argument("--trends-file", type=str, 
                        help="Path to Google Trends CSV for G-Score")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, 
                        help=f"Model for embeddings (default: {DEFAULT_MODEL})")
    parser.add_argument("--min-posts", type=int, default=3,
                        help="Minimum posts for valid trend (default: 3)")
    parser.add_argument("--filter-routine", action="store_true",
                        help="Filter out routine trends (weather, prices, etc.)")
    parser.add_argument("--show-routine", action="store_true",
                        help="Show filtered routine trends")
    parser.add_argument("--use-hdbscan", action="store_true",
                        help="Use HDBSCAN clustering instead of direct trend assignment (experimental)")
    parser.add_argument("--hdbscan-min-size", type=int, default=15,
                        help="HDBSCAN min_cluster_size (default: 15)")
    parser.add_argument("--embedding", type=str, default="sentence-transformer",
                        choices=["sentence-transformer", "tfidf", "bow", "glove"],
                        help="Embedding method (default: sentence-transformer)")
    return parser.parse_args()


def build_trend_clusters(matches: list, min_posts: int = 3) -> dict:
    """
    Build clusters directly from trend assignments (no HDBSCAN needed).
    
    This works better than HDBSCAN because:
    1. We already have labels from Google Trends
    2. Data doesn't have natural density-based clusters
    3. Avoids hyperparameter tuning issues
    """
    clusters = defaultdict(lambda: {
        "items": [],
        "sources": set(),
        "stats": {"likes": 0, "comments": 0, "shares": 0}
    })
    
    for m in matches:
        if not m.get('is_matched', False):
            continue
            
        trend = m.get('trend', 'Unassigned')
        clusters[trend]["items"].append(m)
        clusters[trend]["sources"].add(m.get('source', '').split(':')[0])
        
        stats = m.get('stats', {})
        clusters[trend]["stats"]["likes"] += stats.get("likes", 0)
        clusters[trend]["stats"]["comments"] += stats.get("comments", 0)
        clusters[trend]["stats"]["shares"] += stats.get("shares", 0)
    
    # Filter by min_posts
    valid_clusters = {
        k: v for k, v in clusters.items() 
        if len(v["items"]) >= min_posts
    }
    
    console.print(f"\n[bold cyan]📊 Trend Coverage:[/bold cyan]")
    console.print(f"   Trends with matches: {len(clusters)}")
    console.print(f"   Valid trends (>= {min_posts} posts): {len(valid_clusters)}")
    
    return valid_clusters


def build_hdbscan_clusters(matches: list, model_name: str, min_cluster_size: int = 15) -> dict:
    """
    Build clusters using HDBSCAN (experimental).
    
    Note: This method doesn't work well for trend data because:
    1. Data has 652+ small topics with no density peaks
    2. High-dimensional embeddings cause sparse space
    3. Direct trend assignment from semantic matching is more effective
    
    Use --use-hdbscan flag to try this method anyway.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        try:
            import hdbscan
            HDBSCAN = hdbscan.HDBSCAN
        except ImportError:
            console.print("[red]❌ HDBSCAN not available. Install with: pip install hdbscan[/red]")
            return {}
    
    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False
        
    texts = [m.get('post_content', '')[:500] for m in matches]
    if len(texts) < min_cluster_size * 2:
        console.print("[yellow]Not enough data for HDBSCAN clustering[/yellow]")
        return {}
    
    console.print(f"\n[cyan]🧠 HDBSCAN: Generating embeddings for {len(texts)} texts...[/cyan]")
    console.print(f"\n[cyan]🧠 HDBSCAN: Generating embeddings for {len(texts)} texts...[/cyan]")
    # Use generic get_embeddings
    # But wait, this function signature doesn't pass 'method'. 
    # Let's hardcode to tfidf or pass it in. For now, let's use TF-IDF if model_name is None/default?
    # Actually, let's assume we want speed if we are here.
    # But arguments must be consistent.
    # Refactor: passing embedding as argument to build_hdbscan_clusters would be better, but for now let's use TF-IDF as default fallback
    
    # Ideally, use the same method as find_matches.
    # Assuming user wants same embedding method.
    # Let's just use get_embeddings directly.
    embeddings = get_embeddings(texts, method="tfidf") # Default to fast TF-IDF for HDBSCAN experimental
    
    # Optional UMAP reduction
    if HAS_UMAP:
        console.print("[cyan]📉 UMAP: Reducing dimensions...[/cyan]")
        reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, 
                           metric='cosine', random_state=42)
        reduced = reducer.fit_transform(embeddings)
    else:
        reduced = embeddings
    
    console.print(f"[cyan]🧩 HDBSCAN: Clustering with min_cluster_size={min_cluster_size}...[/cyan]")
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(reduced)
    
    # Build clusters from labels
    clusters = defaultdict(lambda: {
        "items": [],
        "sources": set(),
        "stats": {"likes": 0, "comments": 0, "shares": 0}
    })
    
    from collections import Counter
    label_counts = Counter(labels)
    noise_count = label_counts.get(-1, 0)
    
    console.print(f"\n[bold]HDBSCAN Results:[/bold]")
    console.print(f"   Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
    console.print(f"   Noise points: {noise_count} ({noise_count/len(labels)*100:.1f}%)")
    
    for i, m in enumerate(matches):
        label = labels[i]
        if label == -1:
            continue  # Skip noise
        
        cluster_name = f"HDBSCAN_Cluster_{label}"
        clusters[cluster_name]["items"].append(m)
        clusters[cluster_name]["sources"].add(m.get('source', '').split(':')[0])
        
        stats = m.get('stats', {})
        clusters[cluster_name]["stats"]["likes"] += stats.get("likes", 0)
        clusters[cluster_name]["stats"]["comments"] += stats.get("comments", 0)
        clusters[cluster_name]["stats"]["shares"] += stats.get("shares", 0)
    
    return dict(clusters)


def compute_scores(trend_name: str, data: dict, calculator=None) -> dict:
    """Compute G, F, N scores for a trend."""
    items = data["items"]
    
    # G Score (from calculator if available)
    g_score = 0
    if calculator:
        vol = calculator.google_trends_volume.get(trend_name.lower(), 0)
        if vol > 0:
            g_score = (math.log10(vol + 1) / math.log10(1000000 + 1)) * 100
    
    # F Score (Facebook engagement)
    total_interactions = data["stats"]["likes"] + data["stats"]["comments"] * 2 + data["stats"]["shares"] * 3
    f_score = (math.log10(total_interactions + 1) / math.log10(20000 + 1)) * 100
    
    # N Score (News count)
    news_count = len([i for i in items if 'Face' not in i.get('source', '')])
    n_score = (news_count / 50) * 100
    
    # Composite
    composite = 0.4 * min(g_score, 100) + 0.35 * min(f_score, 100) + 0.25 * min(n_score, 100)
    
    # Classification
    HIGH = 40
    if g_score > HIGH and f_score > HIGH and n_score > HIGH:
        classification = "Strong Multi-source"
    elif g_score > HIGH and f_score > HIGH:
        classification = "Search & Social"
    elif f_score > HIGH and n_score > HIGH:
        classification = "Social & News"
    elif f_score > HIGH:
        classification = "Social-Driven"
    elif n_score > HIGH:
        classification = "News-Driven"
    elif g_score > HIGH:
        classification = "Search-Driven"
    else:
        classification = "Emerging"
    
    return {
        "G": round(g_score, 1),
        "F": round(f_score, 1),
        "N": round(n_score, 1),
        "Composite": round(composite, 1),
        "Class": classification,
        "Interactions": total_interactions,
        "NewsCount": news_count
    }


def display_results(clusters: dict, title: str = "TREND RANKINGS"):
    """Display trend results in a table."""
    # Sort by item count then engagement
    sorted_clusters = sorted(
        clusters.items(),
        key=lambda x: (len(x[1]["items"]), x[1]["stats"]["likes"]),
        reverse=True
    )
    
    table = Table(title=f"📊 {title} ({len(sorted_clusters)} trends)")
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Trend", style="bold cyan", max_width=35)
    table.add_column("Class", style="magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("G/F/N", justify="right", style="dim")
    table.add_column("Posts", justify="right")
    table.add_column("Likes", justify="right")
    
    for i, (trend, data) in enumerate(sorted_clusters[:50]):
        scores = data.get('scores', {})
        gfn = f"{scores.get('G', 0):.0f}/{scores.get('F', 0):.0f}/{scores.get('N', 0):.0f}"
        
        table.add_row(
            str(i + 1),
            trend[:35],
            scores.get('Class', 'N/A'),
            str(scores.get('Composite', 0)),
            gfn,
            str(len(data['items'])),
            str(data['stats']['likes'])
        )
    
    console.print(table)
    return sorted_clusters


def plot_trends(sorted_clusters: list, output_file: str = "results/trend_analysis.png"):
    """Plot top trends bar chart."""
    if not sorted_clusters:
        return
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    top = sorted_clusters[:20]
    labels = [c[0][:25] + "..." if len(c[0]) > 25 else c[0] for c in top][::-1]
    counts = [len(c[1]["items"]) for c in top][::-1]
    
    plt.figure(figsize=(12, 10))
    bars = plt.barh(labels, counts, color='steelblue')
    plt.xlabel('Number of Posts')
    plt.title('Top 20 Real Trends (with data coverage)')
    
    for bar in bars:
        plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 str(int(bar.get_width())), va='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    console.print(f"[green]📊 Chart saved: {output_file}[/green]")


def plot_tsne(matches: list, clusters: dict, embedding_method: str = "tfidf", model_name: str = None, output_file: str = "results/trend_tsne.png"):
    """Visualize trends with t-SNE, colored by trend assignment."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    valid_trends = set(clusters.keys())
    filtered = [m for m in matches if m.get('trend') in valid_trends]
    
    if len(filtered) < 10:
        console.print("[yellow]Not enough data for t-SNE[/yellow]")
        return
    
    texts = [m['post_content'][:500] for m in filtered]
    trend_labels = [m['trend'] for m in filtered]
    
    console.print(f"\n[cyan]🧠 Generating embeddings for {len(texts)} posts...[/cyan]")
    console.print(f"\n[cyan]🧠 Generating embeddings for {len(texts)} posts ({embedding_method})...[/cyan]")
    embeddings = get_embeddings(texts, method=embedding_method, model_name=model_name)
    
    console.print("[cyan]📉 Running t-SNE...[/cyan]")
    perplexity = min(30, len(texts) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(embeddings)
    
    # Color by top 10 trends
    sorted_trends = sorted(clusters.items(), key=lambda x: len(x[1]["items"]), reverse=True)
    top_10 = [t[0] for t in sorted_trends[:10]]
    colors = [top_10.index(t) if t in top_10 else -1 for t in trend_labels]
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='tab10', alpha=0.7, s=40)
    plt.colorbar(scatter, label='Trend Index')
    plt.title(f'Trend Visualization ({len(valid_trends)} trends)')
    
    # Add labels
    for i, trend in enumerate(top_10):
        mask = [c == i for c in colors]
        if any(mask):
            centroid = coords[[j for j, m in enumerate(mask) if m]].mean(axis=0)
            plt.annotate(trend[:20], centroid, fontsize=9, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    console.print(f"[green]📊 t-SNE saved: {output_file}[/green]")


def evaluate_clustering_metrics(matches: list, clusters: dict, embedding_method: str = "tfidf", model_name: str = None):
    """Evaluate clustering quality metrics."""
    valid_trends = set(clusters.keys())
    filtered = [m for m in matches if m.get('trend') in valid_trends]
    
    if len(filtered) < 20 or len(valid_trends) < 2:
        console.print("[yellow]Not enough data for metrics[/yellow]")
        return
    
    texts = [m['post_content'][:500] for m in filtered[:1000]]  # Limit for speed
    labels = [m['trend'] for m in filtered[:1000]]
    
    unique_labels = list(set(labels))
    if len(unique_labels) < 2:
        return
    
    label_map = {l: i for i, l in enumerate(unique_labels)}
    numeric_labels = [label_map[l] for l in labels]
    
    console.print(f"\n[cyan]📏 Computing metrics...[/cyan]")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)
    
    sil = silhouette_score(embeddings, numeric_labels)
    ch = calinski_harabasz_score(embeddings, numeric_labels)
    db = davies_bouldin_score(embeddings, numeric_labels)
    
    console.print("\n[bold]📈 Clustering Metrics (based on trend assignments):[/bold]")
    console.print(f"   • Silhouette Score: {sil:.4f} (-1 to 1, higher=better)")
    console.print(f"   • Calinski-Harabasz: {ch:.2f} (higher=better)")
    console.print(f"   • Davies-Bouldin: {db:.4f} (lower=better)")


def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        console.print(f"[red]❌ File not found: {args.input}[/red]")
        return
    
    console.print(f"[bold green]📂 Loading: {args.input}[/bold green]")
    with open(args.input, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    console.print(f"[green]✅ Loaded {len(matches)} items[/green]")
    
    # Choose clustering method
    if args.use_hdbscan:
        console.print("\n[yellow]⚠️ Using HDBSCAN clustering (experimental)[/yellow]")
        console.print("[dim]Note: Direct trend assignment usually works better for this data[/dim]")
        clusters = build_hdbscan_clusters(matches, args.model, args.hdbscan_min_size)
    else:
        # Build clusters directly from trend assignments (recommended)
        clusters = build_trend_clusters(matches, min_posts=args.min_posts)
    
    if not clusters:
        console.print("[yellow]No valid trends/clusters found[/yellow]")
        return
    
    # Load score calculator if available
    calculator = None
    if args.trends_file and ScoreCalculator:
        calculator = ScoreCalculator(args.trends_file)
    
    # Compute scores
    for trend_name, data in clusters.items():
        data['scores'] = compute_scores(trend_name, data, calculator)
    
    # Filter routine trends
    if args.filter_routine:
        event_trends, routine_trends = filter_routine_trends(clusters)
        console.print(f"\n[yellow]🔄 Filtered {len(routine_trends)} routine trends[/yellow]")
        
        sorted_clusters = display_results(event_trends, "EVENT TRENDS")
        
        if args.show_routine and routine_trends:
            display_results(routine_trends, "ROUTINE TRENDS (Filtered)")
    else:
        sorted_clusters = display_results(clusters, "ALL TRENDS")
    
    # Plots
    plot_trends(sorted_clusters)
    try:
        plot_tsne(matches, clusters, embedding_method=args.embedding, model_name=args.model)
    except Exception as e:
        console.print(f"[red]Error plotting t-SNE: {e}[/red]")
    
    # Metrics
    evaluate_clustering_metrics(matches, clusters, embedding_method=args.embedding, model_name=args.model)
    
    console.print("\n[bold green]✅ Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
