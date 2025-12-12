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
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import os

try:
    from crawlers.trend_scoring import ScoreCalculator
except ImportError:
    from trend_scoring import ScoreCalculator

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    from sklearn.cluster import DBSCAN
    HAS_HDBSCAN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False



console = Console()

# ============================================================================
# ROUTINE TREND FILTERING
# ============================================================================

# Default blacklist patterns for recurring/routine topics
DEFAULT_BLACKLIST_PATTERNS = [
    # Thời tiết / Weather
    r'(không khí lạnh|thời tiết|dự báo thời tiết|mưa lớn|bão số|nhiệt độ|nắng nóng|gió mùa|rét đậm|rét hại|áp thấp nhiệt đới)',
    # Giá cả / Prices (updated daily)
    r'(giá xăng|giá vàng|tỷ giá|giá dầu|giá điện|giá gas)',
    # Lịch trình / Kết quả thể thao định kỳ
    r'(kết quả xổ số|lịch thi đấu|nhận định trận|tỉ số)',
    # Giao thông hàng ngày
    r'(tình hình giao thông|kẹt xe|ùn tắc)',
    # Thêm nữa đi huhu, nhiều quá
]

def is_routine_by_blacklist(trend_name: str, patterns: list) -> bool:
    """
    Check if a trend matches any blacklist pattern (keyword-based filtering).
    
    Args:
        trend_name: Name of the trend to check
        patterns: List of regex patterns to match against
    
    Returns:
        True if the trend matches any blacklist pattern
    """
    trend_lower = trend_name.lower()
    for pattern in patterns:
        if re.search(pattern, trend_lower):
            return True
    return False

def detect_routine_by_frequency(matches: list, threshold_days: int = 3) -> set:
    """
    Detect routine trends based on temporal frequency.
    A trend appearing on X or more consecutive days is considered routine.
    
    Args:
        matches: List of match dictionaries with 'trend' and 'time' keys
        threshold_days: Number of consecutive days for a trend to be routine
    
    Returns:
        Set of trend names that are considered routine
    """
    routine_trends = set()
    
    trend_dates = defaultdict(set)
    
    for match in matches:
        trend = match.get('trend', '')
        time_str = match.get('time', '')
        
        if not time_str:
            continue
            
        # Try to parse date from various formats
        date = None
        for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y-%m-%dT%H:%M:%S', '%d-%m-%Y']:
            try:
                date = datetime.strptime(time_str[:10], fmt).date()
                break
            except (ValueError, IndexError):
                continue
        
        if date:
            trend_dates[trend].add(date)
    
    # Check for consecutive days
    for trend, dates in trend_dates.items():
        if len(dates) < threshold_days:
            continue
            
        sorted_dates = sorted(dates)
        consecutive_count = 1
        max_consecutive = 1
        
        for i in range(1, len(sorted_dates)):
            if (sorted_dates[i] - sorted_dates[i-1]).days == 1:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 1
        
        if max_consecutive >= threshold_days:
            routine_trends.add(trend)
    
    return routine_trends

def filter_routine_trends(matches: list, sorted_clusters: list, 
                          use_blacklist: bool = True,
                          use_frequency: bool = False,
                          blacklist_patterns: list = None,
                          frequency_threshold: int = 3) -> tuple:
    """
    Filter out routine/recurring trends from the analysis.
    
    Args:
        matches: List of all matches
        sorted_clusters: List of (trend_name, cluster_data) tuples
        use_blacklist: Enable keyword-based filtering
        use_frequency: Enable temporal frequency filtering
        blacklist_patterns: Custom blacklist patterns (uses default if None)
        frequency_threshold: Days threshold for frequency filtering
    
    Returns:
        Tuple of (filtered_clusters, routine_clusters, routine_matches)
    """
    if blacklist_patterns is None:
        blacklist_patterns = DEFAULT_BLACKLIST_PATTERNS
    
    routine_trend_names = set()
    
    if use_blacklist:
        for trend_name, _ in sorted_clusters:
            if is_routine_by_blacklist(trend_name, blacklist_patterns):
                routine_trend_names.add(trend_name)
    
    if use_frequency:
        freq_routine = detect_routine_by_frequency(matches, frequency_threshold)
        routine_trend_names.update(freq_routine)
    
    filtered_clusters = []
    routine_clusters = []
    
    for trend_name, data in sorted_clusters:
        if trend_name in routine_trend_names:
            routine_clusters.append((trend_name, data))
        else:
            filtered_clusters.append((trend_name, data))
    
    # Separate matches
    routine_matches = [m for m in matches if m.get('trend', '') in routine_trend_names]
    filtered_matches = [m for m in matches if m.get('trend', '') not in routine_trend_names]
    
    return filtered_clusters, routine_clusters, filtered_matches, routine_matches

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trend Analysis Results")
    parser.add_argument("--input", type=str, required=True, 
                        help="Path to matched results JSON file (e.g., results.json)")
    parser.add_argument("--trends-file", type=str, 
                        help="Path to Google Trends CSV file for G-Score calculation")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                        help="HuggingFace model name for embeddings")
    
    # Filtering options
    filter_group = parser.add_argument_group('Routine Trend Filtering')
    filter_group.add_argument("--filter-routine", action="store_true", default=False,
                              help="Enable filtering of routine/recurring trends (weather, prices, etc.)")
    filter_group.add_argument("--filter-blacklist", action="store_true", default=True,
                              help="Use keyword blacklist filtering (default: enabled when --filter-routine is set)")
    filter_group.add_argument("--filter-frequency", action="store_true", default=False,
                              help="Use temporal frequency filtering (trends appearing on consecutive days)")
    filter_group.add_argument("--frequency-threshold", type=int, default=3, metavar="DAYS",
                              help="Number of consecutive days for a trend to be considered routine (default: 3)")
    filter_group.add_argument("--blacklist-patterns", type=str, nargs='+', metavar="PATTERN",
                              help="Additional regex patterns to add to blacklist (e.g., 'covid' 'vaccine')")
    filter_group.add_argument("--show-routine", action="store_true", default=False,
                              help="Show filtered routine trends in a separate table")
    
    return parser.parse_args()

def evaluate_clusters(matches, sorted_clusters, model_name='all-MiniLM-L6-v2'):
    """
    Calculate unsupervised clustering metrics.
    Note: We need embeddings for this. Re-calculating them might be slow,
    but it's the only way to measure 'compactness' in vector space.
    """
    if len(matches) < 2:
        console.print("[yellow]Not enough matches to evaluate clustering.[/yellow]")
        return

    texts = [m['post_content'][:500] for m in matches]
    labels = [m['trend'] for m in matches]
    
    unique_labels = list(set(labels))
    if len(unique_labels) < 2:
         console.print("[yellow]Only 1 cluster found. Cannot calculate separation metrics.[/yellow]")
         return
         
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_map[l] for l in labels]


    

    
    valid_trends = set([c[0] for c in sorted_clusters])
    
    filtered_texts = []
    filtered_labels = []
    
    for m in matches:
        if m.get('trend') in valid_trends:
            filtered_texts.append(m['post_content'][:500])
            filtered_labels.append(m['trend'])
            
    if len(filtered_labels) < 2:
        console.print("[yellow]Not enough clustered items to evaluate metrics.[/yellow]")
        return
        
    unique_filtered = list(set(filtered_labels))
    if len(unique_filtered) < 2:
         console.print("[yellow]Only 1 cluster found (after noise filtering). Cannot calculate separation metrics.[/yellow]")
         return

    label_map = {label: i for i, label in enumerate(unique_filtered)}
    numeric_labels = [label_map[l] for l in filtered_labels]
    texts = filtered_texts



    console.print(f"\n[bold cyan]📏 Calculating Evaluation Metrics (Model: {model_name})...[/bold cyan]")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=False)

    # 3. Calculate Metrics
    sil = silhouette_score(embeddings, numeric_labels)
    ch = calinski_harabasz_score(embeddings, numeric_labels)
    db = davies_bouldin_score(embeddings, numeric_labels)

    # 4. Display
    console.print("\n[bold]📈 Clustering Performance Metrics:[/bold]")
    console.print(f"   • [cyan]Silhouette Score[/cyan]: [bold white]{sil:.4f}[/bold white] (Higher is better, range -1 to 1)")
    console.print(f"     [dim](Measures how similar an object is to its own cluster vs other clusters)[/dim]")
    
    console.print(f"   • [cyan]Calinski-Harabasz Index[/cyan]: [bold white]{ch:.2f}[/bold white] (Higher is better)")
    console.print(f"     [dim](Ratio of between-cluster dispersion to within-cluster dispersion)[/dim]")
    
    console.print(f"   • [cyan]Davies-Bouldin Index[/cyan]: [bold white]{db:.4f}[/bold white] (Lower is better)")
    console.print(f"     [dim](Average similarity between each cluster and its most similar one)[/dim]")

def plot_clusters(sorted_clusters, output_file="trend_analysis.png"):
    """Plot the top clusters."""
    if not sorted_clusters: return

    # Limit to top 20 for readability
    top_clusters = sorted_clusters[:20]
    labels = [c[0][:20] + "..." if len(c[0]) > 20 else c[0] for c in top_clusters]
    counts = [len(c[1]["items"]) for c in top_clusters]
    # Reverse for horizontal bar chart (top at top)
    labels = labels[::-1]
    counts = counts[::-1]

    plt.figure(figsize=(10, 8))
    bars = plt.barh(labels, counts, color='skyblue')
    plt.xlabel('Number of Matched Items (Coverage)')
    plt.ylabel('Trend Topic')
    plt.title('Top 20 Trending Topics in Crawled Data')
    plt.tight_layout()
    
    # Add counts to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.3, bar.get_y() + bar.get_height()/2, 
                 str(int(width)), ha='left', va='center')

    plt.savefig(output_file)
    console.print(f"[bold green]Chart saved to: {output_file}[/bold green]")

def plot_tsne(matches, sorted_clusters, output_file="trend_tsne.png", model_name='all-MiniLM-L6-v2'):
    """
    Generate embeddings for matched content and plot t-SNE.
    Points are colored by their assigned Trend Cluster (Top 10).
    """
    if not matches: return

    texts = [m['post_content'][:500] for m in matches] # Truncate for speed
    labels = []
    
    # Map matched items back to their main trend
    top_10_trends = [c[0] for c in sorted_clusters[:10]]
    trend_color_map = {t: i for i, t in enumerate(top_10_trends)}
    
    color_indices = []

    for m in matches:
        trend = m['trend']
        if trend in trend_color_map:
            color_indices.append(trend_color_map[trend])
            labels.append(trend)
        else:
            color_indices.append(-1) # Other
            labels.append("Other")


    console.print(f"\n[bold cyan]🧠 Generating embeddings ({model_name})...[/bold cyan]")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)

    console.print("[bold cyan]📉 Running t-SNE (Refined Separation)...[/bold cyan]")
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=25, 
        init='pca', 
        max_iter=3000, 
        early_exaggeration=20,
        learning_rate='auto'
    )
    params = tsne.fit_transform(embeddings)

    plt.figure(figsize=(14, 12)) 
    
    # Scatter plot
    # Separate "Others" (gray) from "Top Trends" (colored)
    
    params_others = np.array([params[i] for i, c in enumerate(color_indices) if c == -1])
    if len(params_others) > 0:
        plt.scatter(params_others[:, 0], params_others[:, 1], c='lightgray', label='Other', alpha=0.5, s=20)

    # Plot top trends with distinct colors
    cmap = plt.get_cmap('tab10')
    
    for trend_name, idx in trend_color_map.items():
        indices = [i for i, c in enumerate(color_indices) if c == idx]
        if not indices: continue
        
        points = np.array([params[i] for i in indices])
        plt.scatter(points[:, 0], points[:, 1], color=cmap(idx), label=trend_name, alpha=0.8, s=30)
        
        # Add centroid label
        centroid = np.mean(points, axis=0)
        plt.text(centroid[0], centroid[1], trend_name, fontsize=9, weight='bold', 
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.title(f'Semantic Landscape of Facebook & News Trends (t-SNE) - {model_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_file)
    console.print(f"[bold green]t-SNE Chart saved to: {output_file}[/bold green]")

def process_and_visualize(matches, args):
    """
    Process matches and visualize trends with optional routine filtering.
    
    Args:
        matches: List of matched items
        args: Parsed command line arguments
    """
    model_name = args.model
    
    # --- UNSUPERVISED CLUSTERING (DISCOVERY) ---
    # Re-computing embeddings is safer.
    
    all_texts = [m['post_content'][:500] for m in matches]
    
    console.print(f"[bold cyan]🧠 Generating Embeddings for Clustering ({len(all_texts)} items)...[/bold cyan]")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(all_texts, show_progress_bar=True)
    
    if HAS_UMAP:
        console.print(f"[bold cyan]📉 Reducing dimensions with UMAP (to 10 components)...[/bold cyan]")
        reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        cluster_data = reduced_embeddings
    else:
        console.print("[yellow]UMAP not found. Using raw high-dimensional embeddings (performance may suffer).[/yellow]")
        cluster_data = embeddings

    console.print(f"[bold cyan]🧩 Running Unsupervised Clustering ({'HDBSCAN' if HAS_HDBSCAN else 'DBSCAN'})...[/bold cyan]")
    
    if HAS_HDBSCAN:
        # Tuned parameters: bigger cluster size to avoid micro-fragmentation
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=15, 
            min_samples=5, 
            metric='euclidean', 
            cluster_selection_method='eom'
        )
        cluster_labels = clusterer.fit_predict(cluster_data)
    else:
        # DBSCAN needs careful epsilon tuning for cosine distance (using 1-cos)
        # eps=0.3 (Sim 0.7) was too loose, caused giant blobs. 
        # Reducing to 0.15 (Sim ~0.85) for stricter clustering.
        clusterer = DBSCAN(eps=0.15, min_samples=3, metric='cosine')
        cluster_labels = clusterer.fit_predict(embeddings)
        
    # New structure: map cluster_id -> items
    raw_clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        if label == -1: continue # Noise
        raw_clusters[label].append(matches[i])
        
    console.print(f"[green]Found {len(raw_clusters)} natural clusters.[/green]")
    
    final_clusters = {} # trend_name -> data
    
    for label, items in raw_clusters.items():
        total_items = len(items)
        original_trends = [m.get('trend', 'Unassigned') for m in items if m.get('is_matched', False)]
        
        cluster_name = f"Cluster {label}"
        
        # Heuristic: Only adopt a Google Trend name if it explains a significant chunk of the cluster
        # e.g., > 20% of the TOTAL cluster, or > 70% of the MATCHED items (if matches are sparse)
        # The previous bug was: one 'Weather' post in a 4000-item blob labeled the whole blob 'Weather'.
        
        assigned_name = None
        if original_trends:
            most_common = Counter(original_trends).most_common(1)
            primary_trend, count = most_common[0]
            
            if count > total_items * 0.2: 
                assigned_name = primary_trend
            
            # But we must be careful not to let 1 item dictate 4000.
            # So combine with minimum support count (e.g. at least 5 posts match this trend)
            elif count > 5 and count > len(original_trends) * 0.8:
                assigned_name = primary_trend

        if assigned_name:
            cluster_name = assigned_name
        else:

            text_blob = " ".join([m['post_content'] for m in items]).lower()
            words = re.findall(r'\w+', text_blob)
            # Simple stopword filter (Vietnamese)
            stopwords = {'và', 'của', 'là', 'có', 'trong', 'đã', 'cho', 'với', 'không', 'các', 'những', 'được', 'tại', 'về', 'người', 'cũng', 'này', 'năm', 'khi', 'ra', 'đến', 'từ', 'rằng', 'thì', 'unassigned', 'face', 'theanh28'}
            common_words = [w for w in words if w not in stopwords and len(w) > 2]
            if common_words:
                top_3 = [w[0] for w in Counter(common_words).most_common(3)]
                cluster_name = f"Emerging: {' '.join(top_3)}"
            else:
                cluster_name = f"Unknown Cluster {label}"

        if cluster_name not in final_clusters:
            final_clusters[cluster_name] = {
                "items": [],
                "sources": set(),
                "stats": {"likes": 0, "comments": 0, "shares": 0},
                "keywords": set()
            }
        
        # Merge items
        final_clusters[cluster_name]["items"].extend(items)
        for item in items:
             final_clusters[cluster_name]["sources"].add(item.get('source', '').split(':')[0])
             final_clusters[cluster_name]["keywords"].add("cluster-discovered")
             stats = item.get('stats', {"likes": 0, "comments": 0, "shares": 0})
             final_clusters[cluster_name]["stats"]["likes"] += stats.get("likes", 0)
             final_clusters[cluster_name]["stats"]["comments"] += stats.get("comments", 0)
             final_clusters[cluster_name]["stats"]["shares"] += stats.get("shares", 0)

    # Use the new clusters
    clusters = final_clusters

    # Sort clusters by Item Count (Coverage) then Engagement
    sorted_clusters = sorted(
        clusters.items(), 
        key=lambda x: (len(x[1]["items"]), x[1]["stats"]["likes"]), 
        reverse=True
    )

    # --- SCORING CALCULATION ---
    console.print(f"\n[bold cyan]🧮 Calculating Multi-source Scores...[/bold cyan]")
    calculator = ScoreCalculator(args.trends_file)
    
    for trend_name, data in sorted_clusters:
        sources_list = [m.get('source', '') for m in data['items']]
        scores = calculator.compute_scores(trend_name, data['items'], sources_list)
        data['scores'] = scores

    # --- ROUTINE TREND FILTERING ---
    active_clusters = sorted_clusters
    active_matches = matches
    routine_clusters = []
    routine_matches = []
    
    if args.filter_routine:
        # Prepare custom blacklist patterns
        blacklist_patterns = DEFAULT_BLACKLIST_PATTERNS.copy()
        if args.blacklist_patterns:
            blacklist_patterns.extend(args.blacklist_patterns)
        
        console.print(f"\n[bold yellow]🔄 Filtering routine trends...[/bold yellow]")
        console.print(f"   • Blacklist filtering: {'✅ Enabled' if args.filter_blacklist else '❌ Disabled'}")
        console.print(f"   • Frequency filtering: {'✅ Enabled' if args.filter_frequency else '❌ Disabled'}")
        if args.filter_frequency:
            console.print(f"   • Frequency threshold: {args.frequency_threshold} consecutive days")
        
        active_clusters, routine_clusters, active_matches, routine_matches = filter_routine_trends(
            matches=matches,
            sorted_clusters=sorted_clusters,
            use_blacklist=args.filter_blacklist,
            use_frequency=args.filter_frequency,
            blacklist_patterns=blacklist_patterns,
            frequency_threshold=args.frequency_threshold
        )
        
        console.print(f"\n[bold green]✅ Filtering complete:[/bold green]")
        console.print(f"   • Event Trends: {len(active_clusters)} ({len(active_matches)} items)")
        console.print(f"   • Routine Trends: {len(routine_clusters)} ({len(routine_matches)} items)")

    # --- DISPLAY MAIN TRENDS TABLE ---
    if args.filter_routine:
        table_title = f"🎯 EVENT TRENDS (Filtered) - {len(active_clusters)} Trends"
    else:
        table_title = f"ALL MATCHED TRENDS (Unfiltered) - {len(sorted_clusters)} Trends"
    
    console.print(f"\n[bold green]📊 CLUSTERING RESULTS[/bold green]\n")
    
    table = Table(title=table_title)
    table.add_column("Rank", style="dim", width=5)
    table.add_column("Trend Name", style="bold cyan")
    table.add_column("Class", style="magenta")
    table.add_column("Score", justify="right", style="green")
    table.add_column("G/F/N", justify="right", style="dim")
    table.add_column("Items", justify="right")
    table.add_column("Likes", justify="right")
    
    for i, (trend, data) in enumerate(active_clusters[:50]): # Show top 50
        stats = data['stats']
        scores = data.get('scores', {})
        
        # Format "G:80 F:20 N:10"
        gfn_str = f"{scores.get('G',0):.0f}/{scores.get('F',0):.0f}/{scores.get('N',0):.0f}"
        
        table.add_row(
            str(i+1),
            trend.upper(),
            scores.get('Class', 'N/A'),
            str(scores.get('Composite', 0)),
            gfn_str,
            str(len(data['items'])),
            str(stats['likes'])
        )
    console.print(table)
    
    # --- DISPLAY ROUTINE TRENDS TABLE (if enabled) ---
    if args.filter_routine and args.show_routine and routine_clusters:
        console.print(f"\n[bold yellow]🔄 ROUTINE TRENDS (Filtered Out) - {len(routine_clusters)} Trends[/bold yellow]\n")
        
        routine_table = Table(title="Routine/Recurring Topics")
        routine_table.add_column("Rank", style="dim", width=5)
        routine_table.add_column("Trend Name", style="yellow")
        routine_table.add_column("Items", justify="right")
        routine_table.add_column("Likes", justify="right")
        routine_table.add_column("Reason", style="dim italic")
        
        for i, (trend, data) in enumerate(routine_clusters[:20]): # Show top 20 routine
            stats = data['stats']
            # Determine reason for filtering
            reasons = []
            if is_routine_by_blacklist(trend, DEFAULT_BLACKLIST_PATTERNS):
                reasons.append("Blacklist")
            # Note: Can't easily check frequency reason here without re-running
            reason_str = ", ".join(reasons) if reasons else "Frequency"
            
            routine_table.add_row(
                str(i+1),
                trend.upper(),
                str(len(data['items'])),
                str(stats['likes']),
                reason_str
            )
        console.print(routine_table)
    
    # --- DETAILED VIEW ---
    console.print("\n[bold]🔍 TOP 5 TRENDS DEEP DIVE:[/bold]")
    for i, (trend, data) in enumerate(active_clusters[:5]):
        all_text = " ".join([m['post_content'] for m in data['items']]).lower()
        words = re.findall(r'\w+', all_text)
        stopwords = {'và', 'của', 'là', 'có', 'trong', 'đã', 'cho', 'với', 'không', 'các', 'những', 'này', 'khi', 'được', 'tại', 'về', 'người', 'như', 'nhưng', 'từ', 'ra', 'đến', 'để', 'vì', 'sẽ', 'lên', 'theo', 'năm', 'ngày', 'tháng', 'việt', 'nam'}
        filtered_words = [w for w in words if len(w) > 2 and w not in stopwords]
        common = Counter(filtered_words).most_common(5)
        common_str = ", ".join([f"{w}({c})" for w, c in common])
        
        console.print(f"\n[bold blue]#{i+1} {trend.upper()}[/bold blue] ({len(data['items'])} items)")
        console.print(f"[dim i]Common words: {common_str}[/dim i]")
        
        detail_table = Table(show_header=True, header_style="bold magenta")
        detail_table.add_column("Source", width=20)
        detail_table.add_column("Snippet", width=60)
        detail_table.add_column("Score", justify="right")
        detail_table.add_column("Time")
        
        for item in data['items'][:5]:
            snippet = item['post_content'][:100].replace('\n', ' ') + "..."
            detail_table.add_row(
                item['source'],
                snippet,
                f"{item.get('score', 0):.2f}",
                item.get('time', '')
            )
        console.print(detail_table)

    # --- METRICS & PLOTS ---
    if active_clusters:
        evaluate_clusters(active_matches, active_clusters, model_name=model_name)
        plot_clusters(active_clusters, output_file="trend_analysis.png")
        plot_tsne(active_matches, active_clusters, model_name=model_name)

def main():
    args = parse_args()
    
    if os.path.exists(args.input):
        console.print(f"[bold green]📂 Loading matches from: {args.input}[/bold green]")
        with open(args.input, 'r', encoding='utf-8') as f:
            matches = json.load(f)
        console.print(f"[bold green]✅ Loaded {len(matches)} matches.[/bold green]")
        
        process_and_visualize(matches, args)
        
    else:
        console.print(f"[bold red]❌ Input file not found: {args.input}[/bold red]")

if __name__ == "__main__":
    main()
