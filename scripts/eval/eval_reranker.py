from sentence_transformers import CrossEncoder
import json
import os
import numpy as np

# Configuration
model_path = 'models/reranker-vietnamese-v1'
test_file = 'data/reranker_train.jsonl' # Using same for demo, ideally split it

def evaluate():
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found.")
        return

    # Load the model
    model = CrossEncoder(model_path)

    # Load test samples
    samples = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))

    if not samples:
        print("No samples to evaluate.")
        return

    print(f"Loaded {len(samples)} samples for evaluation.")

    # Predict and Calculate Metrics
    # For a cross-encoder, we usually evaluate how well it distinguishing positive from negative
    correct = 0
    total = len(samples)
    
    # We can also measure average score for Pos vs Neg
    pos_scores = []
    neg_scores = []

    for ex in samples:
        # CrossEncoder.predict takes pairs [A, B]
        score = model.predict([ex['text']])[0]
        
        if ex['label'] == 1.0:
            pos_scores.append(score)
            if score >= 0.5: correct += 1
        else:
            neg_scores.append(score)
            if score < 0.5: correct += 1

    accuracy = (correct / total) * 100
    avg_pos = np.mean(pos_scores) if pos_scores else 0
    avg_neg = np.mean(neg_scores) if neg_scores else 0

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Avg Positive Score: {avg_pos:.4f}")
    print(f"Avg Negative Score: {avg_neg:.4f}")
    print(f"Score Separation: {avg_pos - avg_neg:.4f}")

if __name__ == "__main__":
    evaluate()
