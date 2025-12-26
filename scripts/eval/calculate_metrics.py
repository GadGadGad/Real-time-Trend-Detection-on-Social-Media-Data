import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
sys.path.append(PROJECT_ROOT)

from rich.console import Console
from rich.table import Table
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, f1_score
from src.pipeline.main_pipeline import find_matches_hybrid

console = Console()

# Config
LABEL_FILE = os.path.join(PROJECT_ROOT, "scripts/eval/grouth_truth_annotation.csv")
CONTENT_FILE = os.path.join(PROJECT_ROOT, "scripts/eval/ground_truth_annotation_sample.csv")
MODEL_NAME = 'keepitreal/vietnamese-sbert'

def purity_score(y_true, y_pred):
    # compute contingency matrix (distribution of true labels in each cluster)
    from sklearn.metrics.cluster import contingency_matrix
    contingency = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

def main():
    if not os.path.exists(LABEL_FILE):
        console.print(f"[red]Error: Label file not found at {LABEL_FILE}[/red]")
        return
    if not os.path.exists(CONTENT_FILE):
        console.print(f"[red]Error: Content file not found at {CONTENT_FILE}[/red]")
        console.print("Please run [bold]sample_for_annotation.py[/bold] first.")
        return

    console.print(f"[dim]Loading labels from {LABEL_FILE}...[/dim]")
    try:
        # Use utf-8-sig to handle potentially bom-ed CSVs
        df_labels = pd.read_csv(LABEL_FILE, encoding='utf-8-sig')
        # Debug
        console.print(f"DEBUG: Label columns: {list(df_labels.columns)}")
        
        # Robust handling: Strip whitespace from column names
        df_labels.columns = [c.strip() for c in df_labels.columns]
        
        if 'GROUND_TRUTH_EVENT_ID' in df_labels.columns:
            pass 
        elif 'grouth_truth' in df_labels.columns:
             df_labels.rename(columns={'grouth_truth': 'GROUND_TRUTH_EVENT_ID'}, inplace=True)
        elif 'ground_truth' in df_labels.columns:
             df_labels.rename(columns={'ground_truth': 'GROUND_TRUTH_EVENT_ID'}, inplace=True)
        else:
             if len(df_labels.columns) >= 2:
                 console.print(f"[yellow]Warning: defaulting 2nd col '{df_labels.columns[1]}' as label[/yellow]")
                 df_labels.rename(columns={df_labels.columns[1]: 'GROUND_TRUTH_EVENT_ID'}, inplace=True)

        # Ensure ID is string/stripped
        df_labels['id'] = df_labels['id'].astype(str).str.strip()
        
    except Exception as e:
        console.print(f"[red]Error reading Label CSV: {e}[/red]")
        return

    console.print(f"[dim]Loading content from {CONTENT_FILE}...[/dim]")
    try:
        df_content = pd.read_csv(CONTENT_FILE, encoding='utf-8-sig')
        df_content.columns = [c.strip() for c in df_content.columns]
        # Ensure ID is string/stripped
        df_content['id'] = df_content['id'].astype(str).str.strip()
    except Exception as e:
        console.print(f"[red]Error reading Content CSV: {e}[/red]")
        return

    # Merge
    console.print("[dim]Merging labels and content...[/dim]")
    try:
        # Merge on ID
        df_labeled = pd.merge(df_content, df_labels[['id', 'GROUND_TRUTH_EVENT_ID']], on='id', how='inner')
        console.print(f"DEBUG: Merged columns: {list(df_labeled.columns)}")
        console.print(f"DEBUG: Merged rows: {len(df_labeled)}")
        
        # Handle merge suffixes if both had the column
        if 'GROUND_TRUTH_EVENT_ID_y' in df_labeled.columns:
            df_labeled.rename(columns={'GROUND_TRUTH_EVENT_ID_y': 'GROUND_TRUTH_EVENT_ID'}, inplace=True)
        if 'GROUND_TRUTH_EVENT_ID_x' in df_labeled.columns:
            df_labeled.drop(columns=['GROUND_TRUTH_EVENT_ID_x'], inplace=True)
            
    except KeyError as k:
        console.print(f"[red]Merge failed (KeyError): {k}. Columns available: Content={list(df_content.columns)}, Label={list(df_labels.columns)}[/red]")
        return

    # Filter rows with labels
    if 'GROUND_TRUTH_EVENT_ID' not in df_labeled.columns:
        console.print("[red]Column 'GROUND_TRUTH_EVENT_ID' missing after merge.[/red]")
        return

    # Treat empty strings or NaNs as unassigned and skip them
    df_labeled = df_labeled.dropna(subset=['GROUND_TRUTH_EVENT_ID']).copy()
    
    # Convert labels to int
    try:
        df_labeled['GROUND_TRUTH_EVENT_ID'] = df_labeled['GROUND_TRUTH_EVENT_ID'].astype(int)
    except:
        console.print("[red]Error: GROUND_TRUTH_EVENT_ID column must contain integers.[/red]")
        return
        
    if len(df_labeled) < 10:
        console.print("[yellow]Warning: Very few labeled samples found. Metrics might be unstable.[/yellow]")

    # Prepare posts object for pipeline
    posts = []
    for _, row in df_labeled.iterrows():
        posts.append({
            'source': row.get('source', 'Unknown'),
            'content': row.get('full_content', ''),
            'time': row.get('published_date', ''),
            'original_id': row.get('original_id', '')
        })

    true_labels = df_labeled['GROUND_TRUTH_EVENT_ID'].values
    
    console.print(f"[bold green]Loaded {len(posts)} annotated samples.[/bold green]")
    console.print(f"True Event Count: {len(set(true_labels) - {-1})}")
    console.print(f"Noise Count: {np.sum(true_labels == -1)}")

    # 2. Run Pipeline (System Output)
    console.print("[bold]Running Hybrid Pipeline (Evaluation Mode)...[/bold]")
    # Using parameters close to production but without LLM for speed/cost if desired
    # For accurate evaluation, we should ideally use the same params.
    # We pass empty trends because we only want to evaluate the clustering component here,
    # unless the user wants to evaluate trend matching too. 
    # But ARI/NMI is about clustering quality.
    
    # We set return_components=True to get access to raw cluster labels
    try:
        matches, components = find_matches_hybrid(
            posts=posts, 
            trends={}, # Empty trends to force clustering-only discovery mode
            use_llm=False, # Disable LLM for basic clustering eval to save cost, or True if critical
            min_cluster_size=2, # Adjusted for small sample
            return_components=True,
            threshold=0.5,
            semantic_floor=0.35, # Default params matching snippet
            coherence_threshold=0.85 
        )
        
        # Extraction
        if 'cluster_labels' in components:
            pred_labels = components['cluster_labels']
            # Ensure length match (pipeline might drop or reorder?)
            # find_matches_hybrid reorders posts! We need to handle this.
            # actually find_matches_hybrid sorts posts by time/source.
            # We must re-align valid labels to our ground truth.
            
            # Re-alignment Logic
            # The pipeline sorted the posts. We need to find where each original post went.
            # We can map by 'content' (assuming unique enough) or 'original_id'
            
            # Map original ID to True Label
            id_to_true_label = {str(p['original_id']): l for p, l in zip(posts, true_labels)}
            
            # Re-construct aligned arrays
            aligned_true = []
            aligned_pred = []
            
            # Iterate through the PIPELINE SORTED posts to build aligned lists
            # The components['cluster_labels'] corresponds to the sorted posts internal to the function
            # But wait, find_matches_hybrid doesn't return the sorted post list directly readily...
            # Actually, `cluster_mapping` has the posts. But raw `cluster_labels` corresponds to the internal sorted list.
            
            # Let's assume simpler approach: modifying find_matches to not re-sort OR 
            # we rely on content matching.
            
            # BETTER APPROACH: Use `cluster_data` directly but with the EXACT preprocessing from pipeline.
            # However, user wants to use `find_matches_hybrid`.
            # Let's stick to `find_matches_hybrid` and rely on its sorted order.
            
            # ...Wait, `find_matches_hybrid` does `posts.sort(...)`. 
            # We can replicate that sort here on our `posts` and `true_labels` before calling it?
            # Yes, that's safest.
            
            # Sort local data exactly as pipeline does:
            # key=lambda x: (str(x.get('time', '')), str(x.get('source', '')), str(x.get('content', ''))[:50])
            
            # Zip posts and labels together to sort them in sync
            combined = list(zip(posts, true_labels))
            combined.sort(key=lambda x: (str(x[0].get('time', '')), str(x[0].get('source', '')), str(x[0].get('content', ''))[:50]))
            
            posts_sorted, true_labels_sorted = zip(*combined)
            posts_sorted = list(posts_sorted)
            true_labels_sorted = np.array(true_labels_sorted)
            
            # Now call pipeline with pre-sorted posts (it will sort again but order won't change)
            matches, components = find_matches_hybrid(
                posts=posts_sorted, 
                trends={}, 
                use_llm=False,
                min_cluster_size=2, # Small sample
                return_components=True
            )
            pred_labels = components['cluster_labels']
            
            if len(pred_labels) != len(true_labels_sorted):
                console.print(f"[red]Mismatch: Pred {len(pred_labels)} vs True {len(true_labels_sorted)}[/red]")
                # Truncate to min to allow debug
                min_len = min(len(pred_labels), len(true_labels_sorted))
                pred_labels = pred_labels[:min_len]
                true_labels_sorted = true_labels_sorted[:min_len]

    except Exception as e:
        console.print(f"[red]Pipeline Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return

    # 3. Calculate Metrics
    ari = adjusted_rand_score(true_labels_sorted, pred_labels)
    nmi = normalized_mutual_info_score(true_labels_sorted, pred_labels)
    purity = purity_score(true_labels_sorted, pred_labels)
    
    # Metric table
    table = Table(title="Evaluation Results (Ground Truth vs. Hybrid Pipeline)")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Description")
    
    table.add_row("Adjusted Rand Index (ARI)", f"{ari:.4f}", "Similarity of grouping (1.0 = perfect match, 0.0 = random)")
    table.add_row("Normalized Mutual Info (NMI)", f"{nmi:.4f}", "Information shared between clusterings")
    table.add_row("Purity", f"{purity:.4f}", "% of clusters containing only a single true event class")
    
    console.print("\n")
    console.print(table)
    
    # Detailed Analysis
    console.print("\n[bold]Detailed Error Analysis:[/bold]")
    
    # Check for Noise Confusion
    true_noise_mask = (true_labels_sorted == -1)
    pred_noise_mask = (pred_labels == -1)
    
    noise_precision = 0
    if np.sum(pred_noise_mask) > 0:
        noise_precision = np.sum(true_noise_mask & pred_noise_mask) / np.sum(pred_noise_mask)
        
    noise_recall = 0
    if np.sum(true_noise_mask) > 0:
        noise_recall = np.sum(true_noise_mask & pred_noise_mask) / np.sum(true_noise_mask)
        
    console.print(f"Noise Detection Precision: [blue]{noise_precision:.2%}[/blue] (True Noise / Predicted Noise)")
    console.print(f"Noise Detection Recall:    [blue]{noise_recall:.2%}[/blue] (Predicted Noise / True Noise)")

if __name__ == "__main__":
    main()
