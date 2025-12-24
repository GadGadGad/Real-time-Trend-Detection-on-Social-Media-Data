import pandas as pd
import numpy as np
from src.core.analysis.clustering import cluster_data
from src.utils.text_processing.vectorizers import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table

console = Console()

def debug_bertopic_topic0(posts_df, sample_size=2000):
    """
    Diagnostic script to solve the 'Massive Topic 0' issue in BERTopic.
    """
    console.print(f"🔍 [bold yellow]Diagnosing BERTopic Cluster Quality (Sample: {sample_size})...[/bold yellow]")
    
    # 1. Prepare Data
    pdf = posts_df.sample(min(len(posts_df), sample_size)).copy()
    texts = pdf['Processed_Text_To_Embed'].tolist()
    embeddings = get_embeddings(texts)
    
    # 2. Run BERTopic WITHOUT and WITH Cohesion Filter
    console.print("\n--- Running BERTopic (No Cohesion Filter) ---")
    labels_raw = cluster_data(embeddings, texts=texts, method='bertopic', min_cluster_size=10, min_cohesion=None)
    
    console.print("\n--- Running BERTopic (With Cohesion Filter @ 0.45) ---")
    labels_filtered = cluster_data(embeddings, texts=texts, method='bertopic', min_cluster_size=10, min_cohesion=0.45)
    
    # 3. Analyze Topic 0 specifically
    def analyze_label(lbls, target_label=0):
        mask = (lbls == target_label)
        count = np.sum(mask)
        if count == 0:
            return None
        
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0).reshape(1, -1)
        sims = cosine_similarity(cluster_embs, centroid).flatten()
        
        # Distance to other clusters
        other_mask = (lbls != target_label) & (lbls != -1)
        if np.any(other_mask):
            other_embs = embeddings[other_mask]
            other_centroid = other_embs.mean(axis=0).reshape(1, -1)
            dist_to_others = cosine_similarity(centroid, other_centroid)[0][0]
        else:
            dist_to_others = 0
            
        return {
            "count": count,
            "avg_sim": sims.mean(),
            "min_sim": sims.min(),
            "max_sim": sims.max(),
            "dist_to_others": dist_to_others,
            "texts": [texts[i] for i, m in enumerate(mask) if m][:5]
        }

    stats_raw = analyze_label(labels_raw)
    stats_fil = analyze_label(labels_filtered)
    
    table = Table(title="BERTopic Topic 0 Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Raw (Unfiltered)", style="magenta")
    table.add_column("Filtered (0.45)", style="green")
    
    if stats_raw:
        table.add_row("Doc Count", str(stats_raw['count']), str(stats_fil['count']) if stats_fil else "REMOVED")
        table.add_row("Internal Cohesion (Avg Sim)", f"{stats_raw['avg_sim']:.3f}", f"{stats_fil['avg_sim']:.3f}" if stats_fil else "N/A")
        table.add_row("Dist to Other Clusters", f"{stats_raw['dist_to_others']:.3f}", "N/A")
    
    console.print(table)
    
    if stats_raw and stats_raw['avg_sim'] < 0.45:
        console.print("[bold red]❌ CONCLUSION: Topic 0 is a 'Garbage Bin'.[/bold red]")
        console.print(f"Lower average similarity ({stats_raw['avg_sim']:.3f}) than threshold (0.45) means these {stats_raw['count']} docs are too diverse.")
        console.print("BERTopic failed to split them because they likely share generic keywords (e.g., 'ông', 'hòa', 'ukraine').")
    
    console.print("\n[bold cyan]Sample docs from Topic 0:[/bold cyan]")
    if stats_raw:
        for i, t in enumerate(stats_raw['texts']):
            console.print(f"  {i+1}. {t[:150]}...")

if __name__ == "__main__":
    # Assuming 'posts' exists in global scope if run in notebook, 
    # or load dummy data for script testing.
    try:
        import joblib
        # Try to load existing data if available
        # This is a placeholder for the actual interactive test
        pass
    except:
        pass
