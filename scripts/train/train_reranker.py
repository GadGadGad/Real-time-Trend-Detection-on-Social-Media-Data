from sentence_transformers import CrossEncoder, InputExample, LoggingHandler
from torch.utils.data import DataLoader
import logging
import json
import os
import math

# Configuration
model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
train_file = 'data/reranker_train.jsonl'
output_path = 'models/reranker-vietnamese-v1'
batch_size = 64
epochs = 3

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def train():
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Run prepare_reranker_data.py first.")
        return

    # 1. Initialize the Cross-Encoder
    model = CrossEncoder(model_name, num_labels=1)

    # 2. Load training data
    train_samples = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            train_samples.append(InputExample(texts=data['text'], label=data['label']))

    print(f"Loaded {len(train_samples)} samples for training.")

    # 3. Create DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    # 4. Configure warm-up steps (10% of training data)
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)
    logging.info(f"Warmup steps: {warmup_steps}")

    # 5. Fine-tune
    model.fit(
        train_dataloader=train_dataloader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True
    )

    # 6. Explicitly save final model
    os.makedirs(output_path, exist_ok=True)
    model.save(output_path)
    print(f"Training complete. Model saved to {output_path}")

if __name__ == "__main__":
    train()
