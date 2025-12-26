"""
Train a sentiment classifier (Positive/Negative/Neutral).
Uses data from: data/sentiment_train.jsonl
Outputs: models/sentiment-classifier-vietnamese-v1
"""
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Configuration
TRAIN_FILE = 'data/sentiment_train.jsonl'
OUTPUT_DIR = 'models/sentiment-classifier-vietnamese-v1'
MODEL_NAME = 'uitnlp/visobert'  # Same as taxonomy for consistency

LABEL2ID = {"Positive": 0, "Negative": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

BATCH_SIZE = 32
EPOCHS = 5
MAX_LENGTH = 256

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=MAX_LENGTH)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='binary')
    return {"accuracy": acc, "f1": f1}

def load_data():
    texts, labels = [], []
    with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item['label'] in LABEL2ID:
                texts.append(item['text'])
                labels.append(LABEL2ID[item['label']])
    return texts, labels

def main():
    print(f"Loading data from {TRAIN_FILE}...")
    texts, labels = load_data()
    print(f"Loaded {len(texts)} samples.")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Starting training...")
    trainer.train()

    # Save only final model (no checkpoints)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Clean up checkpoints
    import shutil
    for item in os.listdir(OUTPUT_DIR):
        if item.startswith('checkpoint-'):
            shutil.rmtree(os.path.join(OUTPUT_DIR, item))
    
    print(f"Model saved to {OUTPUT_DIR}")

    results = trainer.evaluate()
    print(f"Final: Accuracy={results['eval_accuracy']:.4f}, F1={results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()
