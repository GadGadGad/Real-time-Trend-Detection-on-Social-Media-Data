import numpy as np
import sys
import os

# Ensure src is in path
sys.path.append(os.getcwd())

from src.core.analysis.clustering import extract_cluster_labels

# Test data designed to force a specific word to high TF-IDF
texts = [
    "GARBAGE sạt lở Lào Cai",
    "GARBAGE sạt lở Lào Cai",
    "GARBAGE sạt lở Lào Cai",
    "PRESIDENT giá vàng tăng",
    "PRESIDENT giá vàng tăng",
    "PRESIDENT giá vàng tăng"
]

labels = np.array([0, 0, 0, 1, 1, 1])

print("--- Testing WITHOUT custom stopwords ---")
labels_out = extract_cluster_labels(texts, labels, custom_stopwords=None)
print(f"Cluster 0 Label: {labels_out[0]}")
print(f"Cluster 1 Label: {labels_out[1]}")

print("\n--- Testing WITH custom stopwords ['garbage', 'president'] ---")
labels_out_custom = extract_cluster_labels(texts, labels, custom_stopwords=["garbage", "president"])
print(f"Cluster 0 Label: {labels_out_custom[0]}")
print(f"Cluster 1 Label: {labels_out_custom[1]}")

# Verification logic (case insensitive check because .title() is applied)
if "Garbage" in labels_out[0] and "Garbage" not in labels_out_custom[0]:
    print("\n✅ Filtered 'GARBAGE'.")
else:
    print(f"\n❌ FAILED for 'GARBAGE'. Got {labels_out[0]} -> {labels_out_custom[0]}")
    sys.exit(1)

if "President" in labels_out[1] and "President" not in labels_out_custom[1]:
    print("✅ Filtered 'PRESIDENT'.")
else:
    print(f"❌ FAILED for 'PRESIDENT'. Got {labels_out[1]} -> {labels_out_custom[1]}")
    sys.exit(1)

print("\n✨ FINAL STATUS: CUSTOM STOPWORDS ARE WORKING!")
