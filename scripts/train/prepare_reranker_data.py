import pandas as pd
import json
import random
import os
from sentence_transformers import InputExample
from tqdm import tqdm

# Configuration
LABEL_FILE = "scripts/eval/ground_truth_sample.csv"
CONTENT_FILE = "scripts/eval/ground_truth_annotation_sample.csv"
OUTPUT_FILE = "data/reranker_train.jsonl"
NEGATIVES_PER_POSITIVE = 3

def prepare_data():
    if not os.path.exists(LABEL_FILE) or not os.path.exists(CONTENT_FILE):
        print("Error: Missing label or content files.")
        return

    # 1. Load data
    df_labels = pd.read_csv(LABEL_FILE)
    df_content = pd.read_csv(CONTENT_FILE)
    
    # Merge on 'id'
    # Ensure column names are clean (remove BOM or spaces if any)
    df_labels.columns = [c.strip().replace('\ufeff', '') for c in df_labels.columns]
    df_content.columns = [c.strip().replace('\ufeff', '') for c in df_content.columns]
    
    # Map 'grouth_truth' to 'event_id'
    df_labels = df_labels.rename(columns={'grouth_truth': 'event_id'})
    
    merged = pd.merge(df_labels, df_content[['id', 'full_content']], on='id')
    
    # Filter out noise (-1)
    train_df = merged[merged['event_id'] != -1].copy()
    print(f"Loaded {len(train_df)} labeled posts after filtering noise.")

    # 2. Group by event_id
    groups = train_df.groupby('event_id')
    event_to_posts = {eid: grp['full_content'].tolist() for eid, grp in groups}
    all_event_ids = list(event_to_posts.keys())

    examples = []
    
    # 3. Generate Positive Pairs
    for eid, posts in event_to_posts.items():
        if len(posts) < 2:
            continue
        
        # Every pair within the group is a positive
        for i in range(len(posts)):
            for j in range(i + 1, len(posts)):
                # Label 1 for match
                examples.append({"text": [posts[i], posts[j]], "label": 1.0})

    pos_count = len(examples)
    print(f"Generated {pos_count} positive pairs.")

    # 4. Generate Negative Pairs (Random Negatives)
    # We want roughly NEGATIVES_PER_POSITIVE times more negatives
    for eid, posts in event_to_posts.items():
        for p in posts:
            # Pick other event IDs
            other_eids = [other for other in all_event_ids if other != eid]
            if not other_eids:
                continue
                
            samples_needed = NEGATIVES_PER_POSITIVE
            for _ in range(samples_needed):
                random_eid = random.choice(other_eids)
                random_post = random.choice(event_to_posts[random_eid])
                examples.append({"text": [p, random_post], "label": 0.0})

    print(f"Total samples (Pos + Neg): {len(examples)}")

    # 5. Save to JSONL
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
            
    print(f"Saved training data to {OUTPUT_FILE}")

if __name__ == "__main__":
    prepare_data()
