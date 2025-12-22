import json
import os

nb_path = 'notebooks/analysis-playground_v2.ipynb'
if not os.path.exists(nb_path):
    print(f"Error: {nb_path} does not exist.")
    exit(1)

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Define injections
injections = [
    {
        "name": "Import Metrics",
        "search": "import os", # Ideally top cell
        "target_code": "from src.evaluation.metrics import evaluate_embeddings, evaluate_clustering, evaluate_refinement\n",
        "position": "after", 
        "once": True,
        "done": False
    },
    {
        "name": "Embeddings Eval",
        "search": "post_embeddings = get_embeddings(",
        "target_code": """
    # [EVAL] Embeddings
    print("\\n📊 Evaluating Embeddings...")
    emb_stats = evaluate_embeddings(post_embeddings)
    print(f"   Dims: {emb_stats.get('dim')}, Variance: {emb_stats.get('variance_mean'):.4f}")
""",
        "position": "after_block", # heuristic: find matching closing parenthesis/line
        "done": False
    },
    {
        "name": "Clustering Eval",
        "search": "cluster_labels = ",
        "target_code": """
    # [EVAL] Clustering
    print("\\n📊 Evaluating Clustering...")
    clust_stats = evaluate_clustering(post_embeddings, cluster_labels)
    if 'error' not in clust_stats:
        print(f"   Clusters: {clust_stats['n_clusters']} (Noise: {clust_stats['n_noise']})")
        print(f"   Silhouette: {clust_stats.get('silhouette', 0):.4f} | CHI: {clust_stats.get('calinski_harabasz', 0):.1f}")
    else:
        print(f"   Clustering Eval Error: {clust_stats['error']}")
""",
        "position": "after_line",
        "done": False
    },
    {
        "name": "Refinement Eval",
        "search": "match_results = llm.refine_batch(",
        "alt_search": "refined_results = llm.refine_batch(",
        "target_code": """
        # [EVAL] Refinement
        print("\\n📊 Evaluating Refinement...")
        # Need to reconstruct inputs for eval if possible, or just analyze outputs
        # refine_batch returns a dict.
        ref_stats = evaluate_refinement([], refined_results) # Empty input samples list for now
        print(f"   Refined {ref_stats['total_clusters']} clusters.")
        print(f"   Categories: {dict(ref_stats['categories'])}")
""",
        "position": "after_line",
        "done": False
    }
]

# Apply injections
for cell in nb['cells']:
    if cell['cell_type'] != 'code': continue
    
    if isinstance(cell['source'], str):
        cell['source'] = [cell['source']]
    
    source_lines = cell['source']
    if not source_lines: continue
    
    full_source = "".join(source_lines)
    
    # Check Imports
    inj = injections[0]
    if not inj['done'] and ("import " in full_source or "from " in full_source):
        # We assume the first cell with imports is good enough
        # Check if already imported
        if "from src.evaluation.metrics" not in full_source:
             # Prepend
             cell['source'] = [inj['target_code']] + source_lines
             inj['done'] = True
             print("Injected Imports")
    
    # Check Embeddings
    inj = injections[1]
    if not inj['done'] and inj['search'] in full_source:
        # We need to insert AFTER the block.
        # Simple heuristic: insert at end of cell if it contains get_embeddings? 
        # Or find the line.
        # Let's verify 'post_embeddings' is available.
        if "post_embeddings" not in full_source: 
             # Maybe finding the wrong get_embeddings (e.g. trend_embeddings)
             pass
        else:
             # Just append to end of cell for safety
             cell['source'].append(inj['target_code'])
             inj['done'] = True
             print("Injected Embedding Eval")

    # Check Clustering
    inj = injections[2]
    if not inj['done'] and inj['search'] in full_source:
         # Append to end of cell
         cell['source'].append(inj['target_code'])
         inj['done'] = True
         print("Injected Clustering Eval")

    # Check Refinement
    inj = injections[3]
    if not inj['done']:
        if inj['search'] in full_source or (inj.get('alt_search') and inj['alt_search'] in full_source):
            # Append to end of cell
            cell['source'].append(inj['target_code'])
            inj['done'] = True
            print("Injected Refinement Eval")

# Save
with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=4)

print("Finished Notebook Injection.")
