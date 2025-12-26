"""
Pseudo-Streaming Demo State Management.
Save and load pipeline state for simulating real-time event detection.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from rich.console import Console
from datetime import datetime

console = Console()

def save_demo_state(
    save_dir: str,
    df_results: pd.DataFrame,
    trends: dict,
    trend_embeddings: np.ndarray,
    post_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_mapping: dict,
    model_name: str = None,
    metadata: dict = None
):
    """
    Save all necessary state for pseudo-streaming demo.
    
    Args:
        save_dir: Directory to save state files
        df_results: Final results DataFrame
        trends: Trend dictionary
        trend_embeddings: Numpy array of trend embeddings
        post_embeddings: Numpy array of post embeddings  
        cluster_labels: Numpy array of cluster labels
        cluster_mapping: Dictionary mapping cluster IDs to metadata
        model_name: Name of the embedding model used
        metadata: Optional extra metadata (e.g., config settings)
    """
    os.makedirs(save_dir, exist_ok=True)
    console.print(f"[cyan]💾 Saving demo state to {save_dir}...[/cyan]")
    
    # 1. Results DataFrame
    results_path = os.path.join(save_dir, 'results.parquet')
    df_results.to_parquet(results_path, index=False)
    console.print(f"   ✅ Results: {len(df_results)} rows -> results.parquet")
    
    # 2. Trends
    trends_path = os.path.join(save_dir, 'trends.json')
    with open(trends_path, 'w', encoding='utf-8') as f:
        json.dump(trends, f, ensure_ascii=False, indent=2)
    console.print(f"   ✅ Trends: {len(trends)} trends -> trends.json")
    
    # 3. Trend Embeddings
    trend_emb_path = os.path.join(save_dir, 'trend_embeddings.npy')
    np.save(trend_emb_path, trend_embeddings)
    console.print(f"   ✅ Trend Embeddings: {trend_embeddings.shape} -> trend_embeddings.npy")
    
    # 4. Post Embeddings
    post_emb_path = os.path.join(save_dir, 'post_embeddings.npy')
    np.save(post_emb_path, post_embeddings)
    console.print(f"   ✅ Post Embeddings: {post_embeddings.shape} -> post_embeddings.npy")
    
    # 5. Cluster Labels
    labels_path = os.path.join(save_dir, 'cluster_labels.npy')
    np.save(labels_path, cluster_labels)
    console.print(f"   ✅ Cluster Labels: {len(cluster_labels)} labels -> cluster_labels.npy")
    
    # 6. Cluster Mapping (convert keys to int for JSON)
    mapping_path = os.path.join(save_dir, 'cluster_mapping.json')
    serializable_mapping = {}
    for k, v in cluster_mapping.items():
        key = int(k) if isinstance(k, (np.integer, int)) else str(k)
        # Handle non-serializable values
        val = {}
        for vk, vv in v.items():
            if isinstance(vv, (np.ndarray, np.floating, np.integer)):
                val[vk] = float(vv) if isinstance(vv, (np.floating, np.integer)) else vv.tolist()
            elif isinstance(vv, list) and vv and isinstance(vv[0], dict):
                # Skip posts list (too large)
                val[vk] = f"[{len(vv)} posts]"
            else:
                val[vk] = vv
        serializable_mapping[key] = val
        
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_mapping, f, ensure_ascii=False, indent=2, default=str)
    console.print(f"   ✅ Cluster Mapping: {len(cluster_mapping)} clusters -> cluster_mapping.json")
    
    # 7. Cluster Centroids (for attaching new posts)
    centroids = {}
    unique_labels = set(cluster_labels)
    for label in unique_labels:
        if label != -1:
            mask = cluster_labels == label
            centroids[int(label)] = post_embeddings[mask].mean(axis=0)
    
    centroids_path = os.path.join(save_dir, 'centroids.pkl')
    with open(centroids_path, 'wb') as f:
        pickle.dump(centroids, f)
    console.print(f"   ✅ Centroids: {len(centroids)} clusters -> centroids.pkl")
    
    # 8. Metadata
    meta = {
        'model_name': model_name,
        'saved_at': datetime.now().isoformat(),
        'num_results': len(df_results),
        'num_trends': len(trends),
        'num_clusters': len(centroids),
        'embedding_dim': int(post_embeddings.shape[1]) if len(post_embeddings.shape) > 1 else 0,
        **(metadata or {})
    }
    meta_path = os.path.join(save_dir, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    console.print(f"   ✅ Metadata -> metadata.json")
    
    console.print(f"[bold green]✨ Demo state saved successfully![/bold green]")
    return save_dir


def load_demo_state(save_dir: str):
    """
    Load demo state for pseudo-streaming.
    
    Args:
        save_dir: Directory containing saved state files
        
    Returns:
        dict with keys: df_results, trends, trend_embeddings, post_embeddings,
                       cluster_labels, cluster_mapping, centroids, metadata
    """
    console.print(f"[cyan]📂 Loading demo state from {save_dir}...[/cyan]")
    
    state = {}
    
    # 1. Results
    results_path = os.path.join(save_dir, 'results.parquet')
    if os.path.exists(results_path):
        state['df_results'] = pd.read_parquet(results_path)
        console.print(f"   ✅ Results: {len(state['df_results'])} rows")
    
    # 2. Trends
    trends_path = os.path.join(save_dir, 'trends.json')
    if os.path.exists(trends_path):
        with open(trends_path, 'r', encoding='utf-8') as f:
            loaded_trends = json.load(f)
            
        # [FIX] Handle case where trends is a list (backward compatibility / user error)
        if isinstance(loaded_trends, list):
            console.print(f"   ⚠️ Trends loaded as list ({len(loaded_trends)} items). Converting to dict...")
            # Assume list of strings or dicts with 'name'/'query'
            state['trends'] = {}
            for item in loaded_trends:
                if isinstance(item, str):
                    state['trends'][item] = {'volume': 0}
                elif isinstance(item, dict):
                    # Try to find a key
                    key = item.get('name') or item.get('query') or str(item)
                    state['trends'][key] = item
                else:
                    state['trends'][str(item)] = {'original': item}
        else:
            state['trends'] = loaded_trends

        console.print(f"   ✅ Trends: {len(state['trends'])} trends")
    
    # 3. Trend Embeddings
    trend_emb_path = os.path.join(save_dir, 'trend_embeddings.npy')
    if os.path.exists(trend_emb_path):
        state['trend_embeddings'] = np.load(trend_emb_path)
        console.print(f"   ✅ Trend Embeddings: {state['trend_embeddings'].shape}")
    
    # 4. Post Embeddings
    post_emb_path = os.path.join(save_dir, 'post_embeddings.npy')
    if os.path.exists(post_emb_path):
        state['post_embeddings'] = np.load(post_emb_path)
        console.print(f"   ✅ Post Embeddings: {state['post_embeddings'].shape}")
    
    # 5. Cluster Labels
    labels_path = os.path.join(save_dir, 'cluster_labels.npy')
    if os.path.exists(labels_path):
        state['cluster_labels'] = np.load(labels_path)
        console.print(f"   ✅ Cluster Labels: {len(state['cluster_labels'])} labels")
    
    # 6. Cluster Mapping
    mapping_path = os.path.join(save_dir, 'cluster_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            state['cluster_mapping'] = json.load(f)
        console.print(f"   ✅ Cluster Mapping: {len(state['cluster_mapping'])} clusters")
    
    # 7. Centroids
    centroids_path = os.path.join(save_dir, 'centroids.pkl')
    if os.path.exists(centroids_path):
        with open(centroids_path, 'rb') as f:
            state['centroids'] = pickle.load(f)
        console.print(f"   ✅ Centroids: {len(state['centroids'])} clusters")
    
    # 8. Metadata
    meta_path = os.path.join(save_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            state['metadata'] = json.load(f)
        console.print(f"   ✅ Metadata: saved at {state['metadata'].get('saved_at', 'unknown')}")
    
    console.print(f"[bold green]✨ Demo state loaded successfully![/bold green]")
    return state


def attach_new_post(
    new_post: dict,
    centroids: dict,
    trend_embeddings: np.ndarray,
    trend_keys: list,
    embedder,
    threshold: float = 0.5,
    attach_threshold: float = 0.6,
    cluster_mapping: dict = None
):
    """
    Process a single new post for pseudo-streaming demo.
    
    Args:
        new_post: Dict with 'content', 'source', 'time', etc.
        centroids: Dict mapping cluster_id -> centroid embedding
        trend_embeddings: Numpy array of trend embeddings
        trend_keys: List of trend names (same order as trend_embeddings)
        embedder: SentenceTransformer model
        threshold: Minimum similarity for trend matching
        attach_threshold: Minimum similarity for cluster attachment
        cluster_mapping: Optional dict with cluster metadata (for intelligence retrieval)
        
    Returns:
        dict with: cluster_id, final_topic, score, topic_type, summary, advice...
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 1. Embed the new post
    content = new_post.get('content', '')[:500]
    post_emb = embedder.encode([content])
    
    # 2. Find nearest cluster centroid
    centroid_ids = list(centroids.keys())
    centroid_matrix = np.array([centroids[cid] for cid in centroid_ids])
    
    cluster_sims = cosine_similarity(post_emb, centroid_matrix)[0]
    best_cluster_idx = np.argmax(cluster_sims)
    best_cluster_sim = cluster_sims[best_cluster_idx]
    
    if best_cluster_sim >= attach_threshold:
        assigned_cluster = centroid_ids[best_cluster_idx]
    else:
        assigned_cluster = -1  # New/unattached
    
    # 3. Match to trends
    trend_sims = cosine_similarity(post_emb, trend_embeddings)[0]
    best_trend_idx = np.argmax(trend_sims)
    best_trend_score = trend_sims[best_trend_idx]
    
    if best_trend_score >= threshold:
        final_topic = trend_keys[best_trend_idx]
        topic_type = "Trending"
    else:
        final_topic = "Discovery"
        topic_type = "Discovery"
    
    res = {
        'cluster_id': assigned_cluster,
        'cluster_similarity': float(best_cluster_sim),
        'final_topic': final_topic,
        'score': float(best_trend_score),
        'topic_type': topic_type,
        'content': content,
        'source': new_post.get('source', 'Unknown'),
        'time': new_post.get('time', None)
    }

    # Enrich with cluster intelligence if mapping is provided
    if cluster_mapping and str(assigned_cluster) in cluster_mapping:
        m = cluster_mapping[str(assigned_cluster)]
        res.update({
            'summary': m.get('summary', ''),
            'category': m.get('category', 'Unclassified'),
            'topic_sentiment': m.get('sentiment', 'Neutral'),
            'intelligence': m.get('intelligence', {}),
            'advice_state': m.get('advice_state') or m.get('intelligence', {}).get('advice_state', ''),
            'advice_business': m.get('advice_business') or m.get('intelligence', {}).get('advice_business', '')
        })
    
    return res

