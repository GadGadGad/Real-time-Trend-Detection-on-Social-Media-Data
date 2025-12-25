
import pandas as pd
import random
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity

console = Console()

def audit_potential_misses(df_results, trend_embeddings, trend_keys, embedder, floor=0.25):
    """
    Checks 'Discovery' clusters to see if they were ALMOST matches for a trend.
    This helps identify if the 0.35 Semantic Guard is too strict.
    """
    discoveries = df_results[df_results['topic_type'] == 'Discovery']
    if discoveries.empty: 
        console.print("[green]No Discoveries to audit![/green]")
        return
        
    console.print(f"[cyan]📡 Auditing {len(discoveries['final_topic'].unique())} Discoveries for 'Missed Trends'...[/cyan]")
    
    table = Table(title="Potential 'Missed' Trend Matches", show_lines=True)
    table.add_column("Discovery Topic", style="cyan")
    table.add_column("Near-Match Trend", style="yellow")
    table.add_column("Similarity", justify="center")
    table.add_column("Status", justify="center")

    found_borderline = False
    for topic in discoveries['final_topic'].unique():
        # Represent the discovery by its LLM-refined name
        refined_name = topic.replace("New: ", "")
        d_emb = embedder.encode([refined_name])
        
        sims = cosine_similarity(d_emb, trend_embeddings)[0]
        top_idx = np.argmax(sims)
        top_sim = sims[top_idx]
        
        # If it was close (above floor but below our 0.35/0.40 guard)
        if top_sim >= floor:
            found_borderline = True
            status = "🚨 Missed?" if top_sim >= 0.3 else "Omitted"
            table.add_row(refined_name[:40], trend_keys[top_idx][:40], f"{top_sim:.3f}", status)

    if not found_borderline:
        console.print("[green]✅ No hidden trend matches found. Your thresholds are well-tuned![/green]")
    else:
        console.print(table)
        console.print(f"\n[bold yellow]Insight:[/bold yellow] If similarity is > 0.35, the system rejected it to keep Precision high.")

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
        first_row = cluster_posts.iloc[0]
        
        reasoning = first_row.get('llm_reasoning', 'No reasoning found')
        category = first_row.get('category', 'N/A')
        summary = first_row.get('summary', 'No summary provided')
        
        # Calculate Topic Sentiment (Prefer LLM field 'topic_sentiment' if matched, 
        # but keep distribution logic for context)
        topic_sentiment = first_row.get('topic_sentiment', 'Neutral')
        s_color = "green" if topic_sentiment == "Positive" else "red" if topic_sentiment == "Negative" else "yellow"

        # Get Intelligence (5W1H)
        intel = first_row.get('intelligence', {})
        if not isinstance(intel, dict): intel = {}
        
        intel_txt = ""
        if intel and any(v and v != "N/A" for v in intel.values()):
            intel_txt = (
                f"\n[bold white]--- 5W1H Intelligence ---[/bold white]\n"
                f"[dim]Who:[/dim]   {intel.get('who', '-')}\n"
                f"[dim]What:[/dim]  {intel.get('what', '-')}\n"
                f"[dim]Where:[/dim] {intel.get('where', '-')}\n"
                f"[dim]When:[/dim]  {intel.get('when', '-')}\n"
                f"[dim]Why:[/dim]   {intel.get('why', '-')}\n"
            )

        # 1. Header
        console.print(Panel(
            f"[bold cyan]Topic:[/bold cyan] {topic}\n"
            f"[bold magenta]Category:[/bold magenta] {category}\n"
            f"[bold {s_color}]Final Sentiment:[/bold {s_color}] {topic_sentiment}\n"
            f"[bold green]Summary:[/bold green] {summary}\n"
            f"{intel_txt}"
            f"[bold yellow]Reasoning:[/bold yellow] {reasoning}", 
            title="Cluster Audit", expand=False
        ))
        
        # 2. Posts Table
        table = Table(title=f"Sample Posts for: {topic[:50]}...", show_lines=True)
        table.add_column("Rank", justify="center", style="dim")
        table.add_column("Source", style="green")
        table.add_column("Content (Snippet)", style="white")
        table.add_column("Sentiment", justify="center")
        table.add_column("LLM Input?", justify="center")

        # Sort posts by original index to replicate pipeline order (LLM usually sees first 3)
        posts_sorted = cluster_posts.sort_index()
        
        for i, (idx, row) in enumerate(posts_sorted.iterrows()):
            if i >= 10: break # Don't show too many
            
            # Label if it was likely an LLM input (Phase 3 refinement usually takes first 3)
            is_llm_input = "✅ YES" if i < 3 else "No (Control)"
            content_snippet = str(row['content'])[:200].replace('\n', ' ') + "..."
            
            # Sentiment color mapping
            s = str(row.get('sentiment', 'Neutral'))
            s_style = "green" if s == "Positive" else "red" if s == "Negative" else "yellow"
            
            table.add_row(
                str(i+1), 
                str(row.get('source', 'Unknown')), 
                content_snippet, 
                f"[{s_style}]{s}[/{s_style}]", 
                is_llm_input
            )
            
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
