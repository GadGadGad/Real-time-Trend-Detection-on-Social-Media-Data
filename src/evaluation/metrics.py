import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import Counter

def evaluate_embeddings(embeddings):
    """
    Evaluate technical properties of embeddings.
    """
    if embeddings is None or len(embeddings) == 0:
        return {"status": "Empty"}
    
    emb_array = np.array(embeddings)
    norms = np.linalg.norm(emb_array, axis=1)
    
    return {
        "n_samples": len(embeddings),
        "dim": emb_array.shape[1],
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "variance_mean": float(np.mean(np.var(emb_array, axis=0)))
    }

def evaluate_clustering(embeddings, labels):
    """
    Evaluate clustering quality using internal validity indices.
    """
    # Filter out noise (-1) for valid assessment
    mask = [l != -1 for l in labels]
    valid_embs = np.array(embeddings)[mask]
    valid_labels = np.array(labels)[mask]
    
    if len(set(valid_labels)) < 2 or len(valid_labels) < len(set(valid_labels)) + 1:
        return {"status": "Insufficient clusters for evaluation"}
        
    try:
        results = {
            "n_clusters": len(set(valid_labels)),
            "n_noise": len(labels) - len(valid_labels),
            "silhouette": float(silhouette_score(valid_embs, valid_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(valid_embs, valid_labels)),
            "davies_bouldin": float(davies_bouldin_score(valid_embs, valid_labels))
        }
    except Exception as e:
        results = {"error": str(e)}
        
    return results

def evaluate_refinement(original_samples, refined_results):
    """
    Evaluate the impact of LLM refinement.
    """
    # refined_results is a dict of {cluster_id: {refined_title, category...}}
    stats = {
        "total_clusters": len(refined_results),
        "categories": Counter(),
        "event_types": Counter(),
        "avg_title_length": 0
    }
    
    total_len = 0
    for res in refined_results.values():
        stats["categories"][res.get("category", "Unknown")] += 1
        stats["event_types"][res.get("event_type", "Unknown")] += 1
        total_len += len(res.get("refined_title", ""))
        
    if stats["total_clusters"] > 0:
        stats["avg_title_length"] = total_len / stats["total_clusters"]
        
    return stats
