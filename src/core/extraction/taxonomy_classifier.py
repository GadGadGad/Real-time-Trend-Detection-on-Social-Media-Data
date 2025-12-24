import re
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

console = Console()

from src.core.extraction.taxonomy_keywords import get_flattened_keywords_by_group

# --- 1. KEYWORD DEFINITIONS (Replaced by taxonomy_keywords.py) ---
KEYWORDS_A = get_flattened_keywords_by_group("A_CRITICAL")
KEYWORDS_B = get_flattened_keywords_by_group("B_SOCIAL")
KEYWORDS_C = get_flattened_keywords_by_group("C_MARKET")

CATEGORY_MAP = {
    "Group A": KEYWORDS_A,
    "Group B": KEYWORDS_B,
    "Group C": KEYWORDS_C
}

CATEGORY_DESCRIPTIONS = {
    "Group A": "Tai nạn, thảm họa thiên nhiên, tội phạm, dịch bệnh, khẩn cấp.",
    "Group B": "Chính trị, xã hội, tranh cãi, quan điểm công chúng, drama.",
    "Group C": "Giải trí, tiêu dùng, mua sắm, du lịch, văn hóa, thị trường."
}

def classify_by_keywords(text):
    """
    Classify based on keyword presence.
    Returns: Category Name or None
    """
    text_lower = text.lower()
    
    # Check A (High Priority)
    for kw in KEYWORDS_A:
        if kw in text_lower:
            return "A"
            
    # Check B
    for kw in KEYWORDS_B:
        if kw in text_lower:
            return "B"
            
    # Check C
    for kw in KEYWORDS_C:
        if kw in text_lower:
            return "C"
            
    return None

class TaxonomyClassifier:
    def __init__(self, embedding_model=None):
        self.embedder = embedding_model
        self.category_embeddings = None
        self.categories = list(CATEGORY_DESCRIPTIONS.keys())
        
        if self.embedder:
            self._precompute_embeddings()
            
    def _precompute_embeddings(self):
        """Pre-compute embeddings for category descriptions"""
        console.print("[dim]🧠 Pre-computing Taxonomy Embeddings...[/dim]")
        descriptions = [CATEGORY_DESCRIPTIONS[c] for c in self.categories]
        self.category_embeddings = self.embedder.encode(descriptions)
        
    def classify(self, text, threshold=0.25):
        """
        Hybrid Classification:
        1. Check Keywords (Fast, High Precision).
        2. If None, use Semantic Similarity (High Recall).
        """
        # 1. Keyword Check
        kw_result = classify_by_keywords(text)
        if kw_result:
            return kw_result, "Keyword"
            
        # 2. Semantic Check (if model available)
        if self.embedder and self.category_embeddings is not None:
            text_emb = self.embedder.encode([text])
            sims = cosine_similarity(text_emb, self.category_embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            
            if best_score > threshold:
                cat_name = self.categories[best_idx]
                if cat_name == "Group A": return "A", "Semantic"
                if cat_name == "Group B": return "B", "Semantic"
                if cat_name == "Group C": return "C", "Semantic"
                
        return "Unclassified", "None"
