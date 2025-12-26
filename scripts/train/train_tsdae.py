import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import logging
# Configuration
model_name = 'dangvantuan/vietnamese-document-embedding'
train_file = 'data/train_tsdae.txt'
output_path = 'models/tuned-embeddings-vietnamese-v1'
batch_size = 4  # reduced to fit GPU memory
epochs = 1
# Disable external logging (W&B) – use empty list for report_to
report_to = []

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def train():
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Run extract_train_data.py first.")
        return

    # 1. Load the model
    # We use a base transformer model for TSDAE
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # 2. Load the training sentences
    with open(train_file, 'r', encoding='utf-8') as f:
        train_sentences = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(train_sentences)} sentences for training.")

    # 3. Create the dataset
    train_dataset = DenoisingAutoEncoderDataset(train_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 4. Define the loss
    # TSDAE loss function
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 5. Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True,
        checkpoint_path=os.path.join(output_path, 'checkpoints'),
        checkpoint_save_steps=1000
    )

    # 6. Save the model
    model.save(output_path)
    print(f"Training complete. Model saved to {output_path}")

if __name__ == "__main__":
    train()
