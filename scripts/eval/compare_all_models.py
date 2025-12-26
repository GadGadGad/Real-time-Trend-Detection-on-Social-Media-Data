"""
Compare Base vs Fine-tuned models for all classifiers.
Run this on Kaggle after training to see improvements.
"""
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder
import json
import torch
from collections import Counter
from rich.console import Console
from rich.table import Table

console = Console()

# Paths
SENTIMENT_BASE = 'wonrax/phobert-base-vietnamese-sentiment'
SENTIMENT_TUNED = 'models/sentiment-classifier-vietnamese-v1'
TAXONOMY_BASE = 'distilbert-base-multilingual-cased'  # Original
TAXONOMY_TUNED = 'models/taxonomy-classifier-vietnamese-v1'
RERANKER_BASE = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
RERANKER_TUNED = 'models/reranker-vietnamese-v1'

SENTIMENT_FILE = 'data/sentiment_train.jsonl'
TAXONOMY_FILE = 'data/taxonomy_train.jsonl'
RERANKER_FILE = 'data/reranker_train.jsonl'

def load_test_samples(file_path, max_samples=200):
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_samples: break
            samples.append(json.loads(line))
    return samples

def eval_classifier(model_name, test_file, label_map, max_samples=200):
    """Evaluate a classifier and return accuracy."""
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("text-classification", model=model_name, device=device)
    except Exception as e:
        console.print(f"[yellow]Skip {model_name}: {e}[/yellow]")
        return None
    
    samples = load_test_samples(test_file, max_samples)
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
    try:
        model = CrossEncoder(model_path)
    except Exception as e:
        console.print(f"[yellow]Skip {model_path}: {e}[/yellow]")
        return None
    
    samples = load_test_samples(test_file, max_samples)
    correct = 0
    
    for s in samples:
        score = model.predict([s['text']])[0]
        pred = 1 if score > 0 else 0
        if pred == int(s['label']):
            correct += 1
    
    return correct / len(samples) * 100 if samples else 0

def main():
    console.print("[bold cyan]🔍 Comparing Base vs Fine-tuned Models...[/bold cyan]\n")
    
    table = Table(title="Model Comparison: Base vs Fine-tuned")
    table.add_column("Model", style="bold")
    table.add_column("Base Accuracy", justify="right")
    table.add_column("Tuned Accuracy", justify="right", style="green")
    table.add_column("Improvement", justify="right")

    # Sentiment
    console.print("[dim]Evaluating Sentiment...[/dim]")
    sent_label_map = {'POS': 'Positive', 'NEG': 'Negative', 'NEU': 'Neutral',
                      'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}
    base_sent = eval_classifier(SENTIMENT_BASE, SENTIMENT_FILE, sent_label_map)
    tuned_sent = eval_classifier(SENTIMENT_TUNED, SENTIMENT_FILE, 
                                  {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'})
    if base_sent and tuned_sent:
        table.add_row("Sentiment", f"{base_sent:.1f}%", f"{tuned_sent:.1f}%", f"+{tuned_sent-base_sent:.1f}%")

    # Taxonomy - Base doesn't have labels, skip comparison
    console.print("[dim]Evaluating Taxonomy...[/dim]")
    tax_map = {f'LABEL_{i}': f'T{i+1}' for i in range(7)}
    tuned_tax = eval_classifier(TAXONOMY_TUNED, TAXONOMY_FILE, tax_map)
    if tuned_tax:
        table.add_row("Taxonomy", "N/A (no base)", f"{tuned_tax:.1f}%", "New model")

    # Reranker
    console.print("[dim]Evaluating Reranker...[/dim]")
    base_rr = eval_reranker(RERANKER_BASE, RERANKER_FILE)
    tuned_rr = eval_reranker(RERANKER_TUNED, RERANKER_FILE)
    if base_rr and tuned_rr:
        table.add_row("Reranker", f"{base_rr:.1f}%", f"{tuned_rr:.1f}%", f"+{tuned_rr-base_rr:.1f}%")

    console.print(table)

if __name__ == "__main__":
    main()
