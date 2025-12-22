import json
import os

NB_PATH = "notebooks/analysis-playground.ipynb"

def main():
    print(f"Reading {NB_PATH}...")
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    updated = False
    for cell in nb['cells']:
        src = "".join(cell.get('source', []))
        if "cluster_labels = run_sahc_clustering(posts, post_embeddings, min_cluster_size=5)" in src:
            print("✅ Found Clustering Cell. Updating min_cluster_size to 3...")
            new_src = [line.replace("min_cluster_size=5", "min_cluster_size=3") for line in cell['source']]
            cell['source'] = new_src
            updated = True
            
    if updated:
        with open(NB_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("✅ Notebook updated successfully.")
    else:
        print("⚠️ Could not find specific line to update.")

if __name__ == "__main__":
    main()
