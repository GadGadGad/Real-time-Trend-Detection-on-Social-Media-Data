import pandas as pd
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer
from rich.console import Console

# Configuration
INPUT_PARQUET = '../demo/demo_data/results.parquet'
OUTPUT_CACHE = 'embeddings_cache.pkl'
MODEL_NAME = 'dangvantuan/vietnamese-document-embedding'

console = Console()

def main():
    if not os.path.exists(INPUT_PARQUET):
        console.print(f"[red]❌ Input file not found: {INPUT_PARfQUET}[/red]")
        return

    # 1. Load Data
    console.print(f"[cyan]📂 Loading posts from {INPUT_PARQUET}...[/cyan]")
    df = pd.read_parquet(INPUT_PARQUET)
    posts = df.to_dict(orient='records')
    console.print(f"   ✅ Loaded {len(posts)} posts.")

    # 2. Initialize Model
    console.print(f"[cyan]📥 Loading embedding model: {MODEL_NAME}...[/cyan]")
    embedder = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    console.print("   ✅ Model loaded.")

    # 3. Compute Embeddings
    console.print(f"[cyan]🧬 Computing embeddings for {len(posts)} posts...[/cyan]")
    texts = [p.get('content', '') for p in posts]
    
    # Batch encode for efficiency
    embeddings = embedder.encode(
        texts, 
        batch_size=64, 
        show_progress_bar=True, 
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    # Ensure float32
    embeddings = embeddings.astype(np.float32)
    console.print(f"   ✅ Computed embeddings shape: {embeddings.shape}")

    # 4. Save to Cache
    console.print(f"[cyan]💾 Saving to {OUTPUT_CACHE}...[/cyan]")
    with open(OUTPUT_CACHE, 'wb') as f:
        pickle.dump((posts, embeddings), f)
    
    console.print(f"[green]🎉 Success! Data imported and embeddings compatible with system.[/green]")
    console.print(f"[dim]You can now run 'python producer.py' to stream this data.[/dim]")

if __name__ == "__main__":
    main()
