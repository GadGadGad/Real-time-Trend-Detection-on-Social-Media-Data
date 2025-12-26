import argparse
import os
import pandas as pd
from sentence_transformers import CrossEncoder, LoggingHandler
import logging

# ------------------------------------------------------------
# Reranker training script
# ------------------------------------------------------------
# This script fine‑tunes a cross‑encoder model to score
# (post, trend) pairs.  The training data should be a CSV
# with three columns: `post`, `trend`, `label` where `label`
# is 1 for a relevant pair and 0 for a non‑relevant pair.
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a reranker (cross‑encoder) for post‑to‑trend ranking")
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/reranker_train.csv",
        help="Path to CSV file containing training pairs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        help="Base cross‑encoder model name or path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/reranker-vietnamese-v1",
        help="Directory where the fine‑tuned model will be saved",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    return parser.parse_args()


def load_data(train_file):
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    df = pd.read_csv(train_file)
    # Expect columns: post, trend, label
    required = {"post", "trend", "label"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns {required}")
    # Convert to list of (text1, text2, label)
    samples = df[["post", "trend", "label"]].values.tolist()
    return samples


def main():
    args = parse_args()

    # Logging configuration (mirrors TSDAE script)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )

    logging.info("Loading training data …")
    train_samples = load_data(args.train_file)
    logging.info(f"Loaded {len(train_samples)} (post, trend, label) samples")

    # Initialise cross‑encoder
    logging.info(f"Loading base model: {args.model_name}")
    model = CrossEncoder(
        args.model_name,
        num_labels=1,  # regression score (higher = more relevant)
        max_length=256,
    )

    # Train the model
    logging.info("Starting fine‑tuning …")
    model.fit(
        train_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        show_progress_bar=True,
    )

    # Save the fine‑tuned model
    os.makedirs(args.output_path, exist_ok=True)
    model.save(args.output_path)
    logging.info(f"Reranker model saved to {args.output_path}")


if __name__ == "__main__":
    main()
