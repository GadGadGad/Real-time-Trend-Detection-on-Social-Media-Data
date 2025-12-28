import re
import os
import torch
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

console = Console()

from src.core.extraction.taxonomy_keywords import get_flattened_keywords_by_group

# --- 1. KEYWORD DEFINITIONS (T1-T7) ---
KEYWORDS_T1 = get_flattened_keywords_by_group("T1_CRISIS")
KEYWORDS_T2 = get_flattened_keywords_by_group("T2_GOVERNANCE")
KEYWORDS_T3 = get_flattened_keywords_by_group("T3_REPUTATION")
KEYWORDS_T4 = get_flattened_keywords_by_group("T4_MARKET")
KEYWORDS_T5 = get_flattened_keywords_by_group("T5_CULTURE")
KEYWORDS_T6 = get_flattened_keywords_by_group("T6_OPERATIONAL")
KEYWORDS_T7 = get_flattened_keywords_by_group("T7_ROUTINE")

CATEGORY_MAP = {
    "T1": KEYWORDS_T1,
    "T2": KEYWORDS_T2,
    "T3": KEYWORDS_T3,
    "T4": KEYWORDS_T4,
    "T5": KEYWORDS_T5,
    "T6": KEYWORDS_T6,
    "T7": KEYWORDS_T7
}

CATEGORY_DESCRIPTIONS = {
    "T1": "Crisis & Public Risk: Tai nạn, thảm họa thiên nhiên, tội phạm, dịch bệnh, khẩn cấp.",
    "T2": "Policy & Governance: Nghị định, thông tư, chính sách mới, phát biểu của chính phủ.",
    "T3": "Reputation & Trust: Scandal, lùm xùm, tẩy chay, tranh cãi công luận.",
    "T4": "Market Opportunity: Xu hướng tiêu dùng, sản phẩm mới, kinh tế, công nghệ AI.",
    "T5": "Cultural Trend: Meme, người nổi tiếng, giải trí, viral mạng xã hội.",
    "T6": "Operational Pain: Kẹt xe, mất điện, quá tải dịch vụ công, tăng giá xăng.",
    "T7": "Routine Signals: Thời tiết hàng ngày, xổ số, kết quả thể thao định kỳ."
}

# --- Model Configuration ---
CUSTOM_MODEL_PATH = None
_transformer_classifier = None
_using_transformer = False

def _find_custom_model():
    """Look for trained taxonomy classifier in common locations."""
    possible_paths = [
        "demo/models/models/taxonomy-classifier-vietnamese-v1", # Local Demo Path
        "models/taxonomy-classifier-vietnamese-v1",
        "../models/taxonomy-classifier-vietnamese-v1",
        "/kaggle/input/taxonomy-classifier/taxonomy-classifier-vietnamese-v1",
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            if os.path.exists(os.path.join(path, "config.json")):
                return path
    return None

def classify_by_keywords(text):
    """
    Classify based on keyword presence in priority order.
    Returns: Category Name (T1..T7) or None
    """
    text_lower = text.lower()
    
    # Priority Order: T1 > T2 > T3 > T6 > T4 > T5 > T7
    priority = ["T1", "T2", "T3", "T6", "T4", "T5", "T7"]
    
    for cat in priority:
        for kw in CATEGORY_MAP[cat]:
            # Use regex for short keywords (<= 3 chars) to ensure word boundaries
            # Vietnamese doesn't use word boundaries like spaces for everything, 
            # but for "AI", "VTV", etc., it works.
            if len(kw) <= 3:
                pattern = rf"\b{re.escape(kw.lower())}\b"
                if re.search(pattern, text_lower):
                    return cat
            else:
                if kw in text_lower:
                    return cat
            
    return None


class TransformerTaxonomyClassifier:
    """Transformer-based taxonomy classifier using trained VISOBert model."""
    
    ID2LABEL = {0: "T1", 1: "T2", 2: "T3", 3: "T4", 4: "T5", 5: "T6", 6: "T7"}
    
    def __init__(self, model_path=None):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        self.model_path = model_path or _find_custom_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.enabled = False
        
        if self.model_path and os.path.exists(self.model_path):
            try:
                console.print(f"[bold green]✅ Loading TRAINED Taxonomy Classifier from {self.model_path}...[/bold green]")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
                self.model.eval()
                self.enabled = True
            except Exception as e:
                console.print(f"[red]Failed to load taxonomy model: {e}[/red]")
                self.enabled = False
    
    def classify(self, text):
        """Classify a single text into T1-T7."""
        if not self.enabled:
            return "Unclassified", "None"
        
        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, truncation=True, padding=True, 
                    max_length=256, return_tensors="pt"
                ).to(self.device)
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                return self.ID2LABEL[pred], "Transformer"
        except Exception as e:
            console.print(f"[dim red]Taxonomy classify error: {e}[/dim red]")
            return "Unclassified", "Error"
    
    def batch_classify(self, texts, batch_size=64):
        """Classify multiple texts efficiently."""
        if not self.enabled:
            return [("Unclassified", "None") for _ in texts]
        
        results = []
        try:
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    inputs = self.tokenizer(
                        batch, truncation=True, padding=True,
                        max_length=256, return_tensors="pt"
                    ).to(self.device)
                    outputs = self.model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                    results.extend([(self.ID2LABEL[p], "Transformer") for p in preds])
            return results
        except Exception as e:
            console.print(f"[red]Batch taxonomy error: {e}[/red]")
            return [("Unclassified", "Error") for _ in texts]
    
    def clear(self):
        """Free GPU memory."""
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        self.enabled = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TaxonomyClassifier:
    """
    Hybrid Taxonomy Classifier with priority:
    1. Transformer model (if available) - highest accuracy
    2. Keyword matching - fast, high precision
    3. Semantic similarity - high recall fallback
    """
    
    def __init__(self, embedding_model=None, use_transformer=True):
        self.embedder = embedding_model
        self.category_embeddings = None
        self.categories = list(CATEGORY_DESCRIPTIONS.keys())
        self.transformer_clf = None
        
        # Try to load transformer model first
        if use_transformer:
            model_path = _find_custom_model()
            if model_path:
                self.transformer_clf = TransformerTaxonomyClassifier(model_path)
                if not self.transformer_clf.enabled:
                    self.transformer_clf = None
        
        # If no transformer, prepare embedding-based fallback
        if self.embedder and not self.transformer_clf:
            self._precompute_embeddings()
            
    def _precompute_embeddings(self):
        """Pre-compute embeddings for category descriptions"""
        console.print("[dim]🧠 Pre-computing 7-Group Taxonomy Embeddings (fallback)...[/dim]")
        descriptions = [CATEGORY_DESCRIPTIONS[c] for c in self.categories]
        self.category_embeddings = self.embedder.encode(descriptions)
        
    def classify(self, text, threshold=0.25):
        """
        Hybrid Classification:
        1. Use Transformer if available (Best accuracy)
        2. Check Keywords (Fast, High Precision)
        3. If None, use Semantic Similarity (High Recall)
        """
        # 1. Transformer Check (Prioritized)
        if self.transformer_clf and self.transformer_clf.enabled:
            return self.transformer_clf.classify(text)
        
        # 2. Keyword Check
        cat_code = classify_by_keywords(text)
        if cat_code:
            return cat_code, "Keyword"
            
        # 3. Semantic Check (if model available)
        if self.embedder and self.category_embeddings is not None:
            text_emb = self.embedder.encode([text])
            sims = cosine_similarity(text_emb, self.category_embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            
            if best_score > threshold:
                return self.categories[best_idx], "Semantic"
                
        return "Unclassified", "None"
    
    def batch_classify(self, texts, threshold=0.25):
        """Batch classification for efficiency."""
        if self.transformer_clf and self.transformer_clf.enabled:
            return self.transformer_clf.batch_classify(texts)
        
        # Fallback to individual classification
        return [self.classify(text, threshold) for text in texts]
    
    def is_using_transformer(self):
        """Check if using transformer model."""
        return self.transformer_clf is not None and self.transformer_clf.enabled
    
    def clear(self):
        """Free resources."""
        if self.transformer_clf:
            self.transformer_clf.clear()

