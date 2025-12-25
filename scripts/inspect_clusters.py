
import pandas as pd
import random
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def audit_cluster_reasoning(df, n_clusters=3, sample_posts=3):
    """
    Selection of clusters to audit LLM reasoning accuracy.
    Checks if reasoning synthesis reflects multiple posts or just one.
    """
    # Filter out Noise
    df_valid = df[df['topic_type'] != 'Noise'].copy()
    
    # Select clusters (mix of Trending and Discovery)
    topics = df_valid['final_topic'].unique()
    selected_topics = random.sample(list(topics), min(n_clusters, len(topics)))
    
    for topic in selected_topics:
        cluster_posts = df_valid[df_valid['final_topic'] == topic]
        reasoning = cluster_posts.iloc[0].get('llm_reasoning', 'No reasoning found')
        category = cluster_posts.iloc[0].get('category', 'N/A')
        
        # 1. Header
        console.print(Panel(f"[bold cyan]Topic:[/bold cyan] {topic}\n[bold magenta]Category:[/bold magenta] {category}\n[bold yellow]Reasoning:[/bold yellow] {reasoning}", 
                          title="Cluster Audit", expand=False))
        
        # 2. Posts Table
        table = Table(title=f"Sample Posts for: {topic[:50]}...", show_lines=True)
        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Source", style="green")
        table.add_column("Content (Snippet)", style="white")
        table.add_column("LLM Input?", justify="center")

        # Sort posts by original index to replicate pipeline order (LLM usually sees first 3)
        posts_sorted = cluster_posts.sort_index()
        
        for i, (idx, row) in enumerate(posts_sorted.iterrows()):
            if i >= 10: break # Don't show too many
            
            # Label if it was likely an LLM input (Phase 3 refinement usually takes first 3)
            is_llm_input = "✅ YES" if i < 3 else "No (Control)"
            content_snippet = str(row['content'])[:200].replace('\n', ' ') + "..."
            
            table.add_row(str(i+1), str(row.get('source', 'Unknown')), content_snippet, is_llm_input)
            
        console.print(table)
        console.print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    # Example usage if reading from a file
    import sys
    import json
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            audit_cluster_reasoning(df)
        except Exception as e:
            console.print(f"[red]Error loading results: {e}[/red]")
    else:
        console.print("[yellow]Please provide path to results.json or call audit_cluster_reasoning(df) in your notebook.[/yellow]")
