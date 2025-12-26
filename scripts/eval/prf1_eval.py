"""
P/R/F1 Metrics for Event Detection/Clustering.
Computes BCubed, Pairwise, and standard metrics against ground truth.

Usage:
    python scripts/eval/prf1_eval.py \
        --state demo/demo_finetuned_reranker \
        --ground-truth scripts/eval/ground_truth_sample.csv
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from rich.console import Console
from rich.table import Table
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

console = Console()

def load_ground_truth(gt_path, annotation_path=None):
    """Load ground truth labels."""
    gt_df = pd.read_csv(gt_path)
    gt_df.columns = [c.strip().lower().replace('grouth_truth', 'ground_truth') for c in gt_df.columns]
    
    if annotation_path and os.path.exists(annotation_path):
        ann_df = pd.read_csv(annotation_path)
        gt_df = gt_df.merge(ann_df, on='id', how='left')
    
    return gt_df

def load_predictions(state_path):
    """Load predictions from demo state."""
    results_path = os.path.join(state_path, 'results.parquet')
    if os.path.exists(results_path):
        return pd.read_parquet(results_path)
    return None

def bcubed_precision_recall_f1(true_labels, pred_labels):
    """
    Compute BCubed Precision, Recall, F1.
    BCubed evaluates clustering by looking at pairs within predicted clusters.
    """
    n = len(true_labels)
    if n == 0:
        return 0, 0, 0
    
    # Build cluster mappings
    true_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)
    
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        true_clusters[t].add(i)
        pred_clusters[p].add(i)
    
    # Compute BCubed
    precision_sum = 0
    recall_sum = 0
    
    for i in range(n):
        t_label = true_labels[i]
        p_label = pred_labels[i]
        
        true_cluster = true_clusters[t_label]
        pred_cluster = pred_clusters[p_label]
        
        # Items in same predicted cluster that are also in same true cluster
        correct_in_pred = len(true_cluster & pred_cluster)
        
        # Precision: what fraction of my predicted cluster is correct
        precision_sum += correct_in_pred / len(pred_cluster)
        
        # Recall: what fraction of my true cluster was retrieved
        recall_sum += correct_in_pred / len(true_cluster)
    
    precision = precision_sum / n
    recall = recall_sum / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def pairwise_precision_recall_f1(true_labels, pred_labels):
    """
    Compute Pairwise Precision, Recall, F1.
    Evaluates all pairs of items.
    """
    n = len(true_labels)
    if n < 2:
        return 0, 0, 0
    
    tp = fp = fn = tn = 0
    
    for i, j in combinations(range(n), 2):
        same_true = true_labels[i] == true_labels[j]
        same_pred = pred_labels[i] == pred_labels[j]
        
        if same_true and same_pred:
            tp += 1
        elif not same_true and same_pred:
            fp += 1
        elif same_true and not same_pred:
            fn += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def cluster_purity(true_labels, pred_labels):
    """Compute cluster purity - fraction of items in majority class per cluster."""
    clusters = defaultdict(list)
    for t, p in zip(true_labels, pred_labels):
        clusters[p].append(t)
    
    total_correct = 0
    for cluster_items in clusters.values():
        most_common = max(set(cluster_items), key=cluster_items.count)
        total_correct += cluster_items.count(most_common)
    
    return total_correct / len(true_labels) if true_labels else 0

def match_predictions_to_gt(gt_df, pred_df):
    """Match predictions to ground truth by content similarity or ID."""
    # Try matching by original_index or id
    if 'original_index' in pred_df.columns:
        merged = gt_df.merge(pred_df, left_on='id', right_on='original_index', how='inner')
    elif 'id' in pred_df.columns:
        merged = gt_df.merge(pred_df, on='id', how='inner')
    else:
        # Fallback: match by row order (assumes same order)
        min_len = min(len(gt_df), len(pred_df))
        gt_df = gt_df.head(min_len).copy()
        pred_df = pred_df.head(min_len).copy()
        merged = pd.concat([gt_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    
    return merged

def main():
    parser = argparse.ArgumentParser(description="P/R/F1 Evaluation for Event Detection")
    parser.add_argument('--state', required=True, help='Path to demo state directory')
    parser.add_argument('--ground-truth', required=True, help='Path to ground truth CSV')
    parser.add_argument('--annotation', help='Path to annotation CSV with content')
    args = parser.parse_args()
    
    console.print(f"[bold cyan]📊 P/R/F1 Evaluation for Event Detection[/bold cyan]")
    console.print(f"   State: {args.state}")
    console.print(f"   Ground Truth: {args.ground_truth}\n")
    
    # Load data
    gt_df = load_ground_truth(args.ground_truth, args.annotation)
    pred_df = load_predictions(args.state)
    
    if pred_df is None:
        console.print("[red]Error: Could not load predictions.[/red]")
        return
    
    console.print(f"[dim]Ground Truth: {len(gt_df)} samples[/dim]")
    console.print(f"[dim]Predictions: {len(pred_df)} samples[/dim]")
    
    # Match and prepare labels
    merged = match_predictions_to_gt(gt_df, pred_df)
    console.print(f"[dim]Matched: {len(merged)} samples[/dim]\n")
    
    if len(merged) == 0:
        console.print("[red]Error: No matching samples found.[/red]")
        return
    
    # Get labels
    true_labels = merged['ground_truth'].fillna(-1).astype(int).tolist()
    
    # Predicted cluster - use final_topic or cluster_id
    if 'cluster_id' in merged.columns:
        pred_labels = merged['cluster_id'].fillna(-1).astype(int).tolist()
    elif 'final_topic' in merged.columns:
        # Convert topic names to numeric labels
        topics = merged['final_topic'].fillna('Noise')
        topic_to_id = {t: i for i, t in enumerate(topics.unique())}
        pred_labels = [topic_to_id[t] for t in topics]
    else:
        console.print("[red]Error: No cluster/topic column found.[/red]")
        return
    
    # Compute all metrics
    results = {}
    
    # 1. BCubed
    console.print("[dim]Computing BCubed metrics...[/dim]")
    bc_p, bc_r, bc_f1 = bcubed_precision_recall_f1(true_labels, pred_labels)
    results['bcubed'] = {'precision': bc_p, 'recall': bc_r, 'f1': bc_f1}
    
    # 2. Pairwise
    console.print("[dim]Computing Pairwise metrics...[/dim]")
    pw_p, pw_r, pw_f1 = pairwise_precision_recall_f1(true_labels, pred_labels)
    results['pairwise'] = {'precision': pw_p, 'recall': pw_r, 'f1': pw_f1}
    
    # 3. Standard clustering metrics
    console.print("[dim]Computing clustering metrics...[/dim]")
    results['clustering'] = {
        'ari': adjusted_rand_score(true_labels, pred_labels),
        'nmi': normalized_mutual_info_score(true_labels, pred_labels),
        'v_measure': v_measure_score(true_labels, pred_labels),
        'purity': cluster_purity(true_labels, pred_labels)
    }
    
    # Display results
    console.print("\n")
    
    # Table 1: BCubed
    t1 = Table(title="BCubed Metrics")
    t1.add_column("Metric", style="bold")
    t1.add_column("Value", justify="right")
    t1.add_row("Precision", f"{bc_p:.4f}")
    t1.add_row("Recall", f"{bc_r:.4f}")
    t1.add_row("[bold]F1[/bold]", f"[bold]{bc_f1:.4f}[/bold]")
    console.print(t1)
    console.print()
    
    # Table 2: Pairwise
    t2 = Table(title="Pairwise Metrics")
    t2.add_column("Metric", style="bold")
    t2.add_column("Value", justify="right")
    t2.add_row("Precision", f"{pw_p:.4f}")
    t2.add_row("Recall", f"{pw_r:.4f}")
    t2.add_row("[bold]F1[/bold]", f"[bold]{pw_f1:.4f}[/bold]")
    console.print(t2)
    console.print()
    
    # Table 3: Other metrics
    t3 = Table(title="Clustering Quality Metrics")
    t3.add_column("Metric", style="bold")
    t3.add_column("Value", justify="right")
    for k, v in results['clustering'].items():
        t3.add_row(k.upper(), f"{v:.4f}")
    console.print(t3)
    
    # Save results
    output_path = os.path.join(args.state, 'prf1_eval.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")

if __name__ == "__main__":
    main()
