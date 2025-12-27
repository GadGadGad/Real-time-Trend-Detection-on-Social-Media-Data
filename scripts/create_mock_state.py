import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
SAVE_DIR = os.path.join(PROJECT_ROOT, "demo_states/demo_data_batch")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"🚀 Creating mock demo state in {SAVE_DIR}...")

# 1. Define dummy trends
trend_names = [
    "Dịch bệnh bùng phát", 
    "Sự kiện Apple", 
    "Thời tiết chuyển lạnh", 
    "Giá vàng tăng cao",
    "Bóng đá Việt Nam"
]

# 2. Create results DataFrame
data = []
for i, trend in enumerate(trend_names):
    for j in range(5): # 5 posts per trend
        data.append({
            "content": f"Bài viết số {j} về chủ đề {trend}. Nội dung tóm tắt để test.",
            "source": "Facebook" if j % 2 == 0 else "VnExpress",
            "time": datetime.now().isoformat(),
            "final_topic": trend,
            "cluster": i,
            "score": 0.85
        })

df_results = pd.DataFrame(data)
df_results.to_parquet(os.path.join(SAVE_DIR, "results.parquet"), index=False)

# 3. Create dummy embeddings (dim=768)
dim = 768
n_trends = len(trend_names)
n_posts = len(df_results)

trend_embeddings = np.random.randn(n_trends, dim).astype('float32')
post_embeddings = np.random.randn(n_posts, dim).astype('float32')
cluster_labels = np.array([i // 5 for i in range(n_posts)])

np.save(os.path.join(SAVE_DIR, "trend_embeddings.npy"), trend_embeddings)
np.save(os.path.join(SAVE_DIR, "post_embeddings.npy"), post_embeddings)
np.save(os.path.join(SAVE_DIR, "cluster_labels.npy"), cluster_labels)

# 4. Create trends.json
trends_dict = {name: {"volume": 5, "score": 80.0} for name in trend_names}
with open(os.path.join(SAVE_DIR, "trends.json"), 'w', encoding='utf-8') as f:
    json.dump(trends_dict, f, ensure_ascii=False, indent=2)

# 5. Create cluster_mapping.json
mapping = {
    str(i): {
        "name": trend,
        "volume": 5,
        "summary": f"Tóm tắt về {trend}",
        "category": "Xã hội",
        "sentiment": "Neutral"
    } for i, trend in enumerate(trend_names)
}
with open(os.path.join(SAVE_DIR, "cluster_mapping.json"), 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

# 6. Create centroids.pkl
centroids = {i: trend_embeddings[i] for i in range(n_trends)}
with open(os.path.join(SAVE_DIR, "centroids.pkl"), 'wb') as f:
    pickle.dump(centroids, f)

# 7. Metadata
metadata = {
    "model_name": "dangvantuan/vietnamese-document-embedding",
    "saved_at": datetime.now().isoformat(),
    "num_results": n_posts,
    "num_trends": n_trends,
    "num_clusters": n_trends,
    "embedding_dim": dim
}
with open(os.path.join(SAVE_DIR, "metadata.json"), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("✅ Mock demo state created successfully!")
