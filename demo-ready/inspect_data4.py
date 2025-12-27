import pandas as pd
import numpy as np
import os

DATA_DIR = 'demo/data_4'
PARQUET_FILE = os.path.join(DATA_DIR, 'results.parquet')
EMB_FILE = os.path.join(DATA_DIR, 'post_embeddings.npy')

def inspect():
    if not os.path.exists(PARQUET_FILE):
        print(f"Parquet not found: {PARQUET_FILE}")
        return

    print(f"--- Inspecting {PARQUET_FILE} ---")
    df = pd.read_parquet(PARQUET_FILE)
    print(df.columns.tolist())
    print(f"Rows: {len(df)}")
    print(df.head(2).to_dict('records'))
    
    if os.path.exists(EMB_FILE):
        print(f"--- Inspecting {EMB_FILE} ---")
        embs = np.load(EMB_FILE)
        print(f"Shape: {embs.shape}")
        if len(embs) == len(df):
            print("✅ Embeddings count matches Parquet rows.")
        else:
            print(f"❌ Mismatch! Parquet: {len(df)}, Embs: {len(embs)}")

if __name__ == "__main__":
    inspect()
