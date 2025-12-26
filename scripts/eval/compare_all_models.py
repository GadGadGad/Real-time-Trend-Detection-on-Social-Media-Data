"""
Compare Base vs Fine-tuned models for all classifiers.
Run this on Kaggle after training to see improvements.

Usage:
    python scripts/eval/compare_all_models.py --all
    python scripts/eval/compare_all_models.py --sentiment --reranker
    python scripts/eval/compare_all_models.py --taxonomy --max-samples 500
"""
import argparse
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder
import json
import torch
import os
from collections import Counter
from rich.console import Console
from rich.table import Table

console = Console()

# Default Paths
DEFAULT_PATHS = {
    'sentiment_base': 'wonrax/phobert-base-vietnamese-sentiment',
    'sentiment_tuned': 'models/sentiment-classifier-vietnamese-v1',
    'taxonomy_tuned': 'models/taxonomy-classifier-vietnamese-v1',
    'reranker_base': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'reranker_tuned': 'models/reranker-vietnamese-v1',
    'sentiment_data': 'data/sentiment_train.jsonl',
    'taxonomy_data': 'data/taxonomy_train.jsonl',
    'reranker_data': 'data/reranker_train.jsonl',
}

def load_test_samples(file_path, max_samples=200):
    samples = []
    if not os.path.exists(file_path):
        console.print(f"[yellow]File not found: {file_path}[/yellow]")
        return samples
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples: break
            samples.append(json.loads(line))
    return samples

def eval_classifier(model_name, test_file, label_map, max_samples=200):
    """Evaluate a classifier and return accuracy."""
    if not os.path.exists(model_name) and not model_name.startswith(('vinai/', 'wonrax/', 'uitnlp/', 'distilbert', 'models/')):
        console.print(f"[yellow]Model not found: {model_name}[/yellow]")
        return None
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text-classification", model=model_name, device=device, trust_remote_code=True)
    except Exception as e:
        console.print(f"[yellow]Skip {model_name}: {e}[/yellow]")
        return None
    
    samples = load_test_samples(test_file, max_samples)
    if not samples:
        return None
    correct = 0
    
    for s in samples:
        try:
            result = pipe(s['text'][:512])[0]
            pred_label = result['label']
            
            # Map labels
            if pred_label in label_map:
                pred = label_map[pred_label]
            else:
                pred = pred_label
                
            if pred == s['label']:
                correct += 1
        except:
            pass
    
    return correct / len(samples) * 100 if samples else 0

def eval_reranker(model_path, test_file, max_samples=200):
    """Evaluate a reranker and return accuracy."""
    if not os.path.exists(model_path) and 'cross-encoder' not in model_path:
        console.print(f"[yellow]Model not found: {model_path}[/yellow]")
        return None
    try:
        model = CrossEncoder(model_path)
    except Exception as e:
        console.print(f"[yellow]Skip {model_path}: {e}[/yellow]")
        return None
    
    samples = load_test_samples(test_file, max_samples)
    if not samples:
        return None
    correct = 0
    
    for s in samples:
        score = model.predict([s['text']])[0]
        pred = 1 if score > 0 else 0
        if pred == int(s['label']):
            correct += 1
    
    return correct / len(samples) * 100 if samples else 0

def main():
    parser = argparse.ArgumentParser(description="Compare Base vs Fine-tuned models")
    parser.add_argument('--all', action='store_true', help='Evaluate all models')
    parser.add_argument('--sentiment', action='store_true', help='Evaluate Sentiment classifier')
    parser.add_argument('--taxonomy', action='store_true', help='Evaluate Taxonomy classifier')
    parser.add_argument('--reranker', action='store_true', help='Evaluate Reranker')
    parser.add_argument('--max-samples', type=int, default=200, help='Max samples to evaluate (default: 200)')
    parser.add_argument('--sentiment-base', type=str, default=DEFAULT_PATHS['sentiment_base'])
    parser.add_argument('--sentiment-tuned', type=str, default=DEFAULT_PATHS['sentiment_tuned'])
    parser.add_argument('--taxonomy-tuned', type=str, default=DEFAULT_PATHS['taxonomy_tuned'])
    parser.add_argument('--reranker-base', type=str, default=DEFAULT_PATHS['reranker_base'])
    parser.add_argument('--reranker-tuned', type=str, default=DEFAULT_PATHS['reranker_tuned'])
    args = parser.parse_args()
    
    # If no specific model selected, evaluate all
    if not (args.sentiment or args.taxonomy or args.reranker):
        args.all = True
    
    console.print("[bold cyan]🔍 Comparing Base vs Fine-tuned Models...[/bold cyan]\n")
    
    table = Table(title="Model Comparison: Base vs Fine-tuned")
    table.add_column("Model", style="bold")
    table.add_column("Base Accuracy", justify="right")
    table.add_column("Tuned Accuracy", justify="right", style="green")
    table.add_column("Improvement", justify="right")

    # Sentiment
    if args.all or args.sentiment:
        console.print("[dim]Evaluating Sentiment...[/dim]")
        sent_label_map = {'POS': 'Positive', 'NEG': 'Negative', 'NEU': 'Neutral',
                          'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
        base_sent = eval_classifier(args.sentiment_base, DEFAULT_PATHS['sentiment_data'], 
                                     sent_label_map, args.max_samples)
        tuned_sent = eval_classifier(args.sentiment_tuned, DEFAULT_PATHS['sentiment_data'], 
                                      {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'},
                                      args.max_samples)
        if base_sent is not None and tuned_sent is not None:
            diff = tuned_sent - base_sent
            table.add_row("Sentiment", f"{base_sent:.1f}%", f"{tuned_sent:.1f}%", 
                         f"{'+' if diff >= 0 else ''}{diff:.1f}%")
        elif tuned_sent is not None:
            table.add_row("Sentiment", "N/A", f"{tuned_sent:.1f}%", "Tuned only")

    # Taxonomy
    if args.all or args.taxonomy:
        console.print("[dim]Evaluating Taxonomy...[/dim]")
        tax_map = {f'LABEL_{i}': f'T{i+1}' for i in range(7)}
        tuned_tax = eval_classifier(args.taxonomy_tuned, DEFAULT_PATHS['taxonomy_data'], 
                                     tax_map, args.max_samples)
        if tuned_tax is not None:
            table.add_row("Taxonomy", "N/A (new)", f"{tuned_tax:.1f}%", "New model")

    # Reranker
    if args.all or args.reranker:
        console.print("[dim]Evaluating Reranker...[/dim]")
        base_rr = eval_reranker(args.reranker_base, DEFAULT_PATHS['reranker_data'], args.max_samples)
        tuned_rr = eval_reranker(args.reranker_tuned, DEFAULT_PATHS['reranker_data'], args.max_samples)
        if base_rr is not None and tuned_rr is not None:
            diff = tuned_rr - base_rr
            table.add_row("Reranker", f"{base_rr:.1f}%", f"{tuned_rr:.1f}%", 
                         f"{'+' if diff >= 0 else ''}{diff:.1f}%")
        elif tuned_rr is not None:
            table.add_row("Reranker", "N/A", f"{tuned_rr:.1f}%", "Tuned only")

    console.print(table)

if __name__ == "__main__":
    main()

