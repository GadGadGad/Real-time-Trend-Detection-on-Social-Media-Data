from sentence_transformers import CrossEncoder
import json
import os
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Configuration
BASE_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
TUNED_MODEL = 'models/reranker-vietnamese-v1'
TEST_FILE = 'data/reranker_train.jsonl'

def run_eval(model_name_or_path):
    # If it's a local path (starts with ./ or / or models/) it must exist
    is_local = model_name_or_path.startswith("/") or \
               model_name_or_path.startswith("./") or \
               model_name_or_path.startswith("models/") or \
               os.path.exists(model_name_or_path)
               
    if is_local and not os.path.exists(model_name_or_path):
        console.print(f"[yellow]Local model path {model_name_or_path} not found. Skipping...[/yellow]")
        return None  # Path not found
        
    try:
        model = CrossEncoder(model_name_or_path)
    except Exception as e:
        console.print(f"[red]Error loading {model_name_or_path}: {e}[/red]")
        return None

    samples = []
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    if not samples:
        return None

    correct = 0
    pos_scores = []
    neg_scores = []

    for ex in samples:
        # CrossEncoder.predict for num_labels=1 returns raw logits
        score = model.predict([ex['text']])[0]
        
        # Threshold for logits is 0.0 (equivalent to 0.5 for sigmoid)
        is_pos = score > 0.0
        
        if ex['label'] == 1.0:
            pos_scores.append(score)
            if is_pos: correct += 1
        else:
            neg_scores.append(score)
            if not is_pos: correct += 1

    return {
        "accuracy": (correct / len(samples)) * 100,
        "avg_pos": np.mean(pos_scores) if pos_scores else 0,
        "avg_neg": np.mean(neg_scores) if neg_scores else 0,
        "separation": np.mean(pos_scores) - np.mean(neg_scores) if pos_scores and neg_scores else 0
    }

def main():
    if not os.path.exists(TEST_FILE):
        console.print(f"[red]Error: {TEST_FILE} not found.[/red]")
        return

    console.print("[cyan]Evaluating Base Model...[/cyan]")
    base_results = run_eval(BASE_MODEL)
    
    console.print("[cyan]Evaluating Tuned Model...[/cyan]")
    tuned_results = run_eval(TUNED_MODEL)

    table = Table(title="Reranker Comparison (Base vs Tuned)")
    table.add_column("Metric", style="bold")
    table.add_column("Base Model", justify="right")
    table.add_column("Tuned Model", justify="right", style="green")
    table.add_column("Improvement", justify="right")

    metrics = [
        ("Accuracy (%)", "accuracy"),
        ("Avg Pos Score", "avg_pos"),
        ("Avg Neg Score", "avg_neg"),
        ("Score Separation", "separation")
    ]

    for label, key in metrics:
        b_val = base_results[key] if base_results else 0
        t_val = tuned_results[key] if tuned_results else 0
        diff = t_val - b_val if key != "accuracy" else t_val - b_val
        
        table.add_row(
            label,
            f"{b_val:.4f}",
            f"{t_val:.4f}",
            f"{'+' if diff > 0 else ''}{diff:.4f}"
        )

    console.print(table)

if __name__ == "__main__":
    main()
