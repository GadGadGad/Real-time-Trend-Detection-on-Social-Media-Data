import pandas as pd
import numpy as np
import re
import os
from typing import List, Set, Dict
from collections import Counter
from src.utils.config.locations import get_known_locations
from src.utils.text_processing.alias_normalizer import normalize_with_aliases
from src.core.extraction.taxonomy_keywords import get_all_event_keywords

class KeywordExtractor:
    def __init__(self, segmentation_method: str = "underthesea"):
        """
        segmentation_method: 
            - "underthesea" (fast, CRF) 
            - "transformer" (accurate, Underthesea Deep)
            - "phonlp" (very accurate, VinAI Multi-task Transformer with VnCoreNLP segmenter)
        """
        self.known_locations = get_known_locations()
        self.taxonomy_keywords = get_all_event_keywords()
        self.segmentation_method = segmentation_method
        self.phonlp_model = None
        self.vncorenlp_model = None
        self.vncorenlp_path = os.path.join(os.path.expanduser("~"), ".cache", "vncorenlp_models")
        # Common Vietnamese stopwords (minimal set for extraction)
        self.stopwords = {
            'và', 'của', 'là', 'có', 'trong', 'đã', 'ngày', 'theo', 'với', 
            'cho', 'người', 'những', 'tại', 'về', 'các', 'được', 'ra', 'khi',
            'mới', 'này', 'cho', 'nhiều',
            # Technical artifacts/noise
            'volume', 'keywords', 'time', 'automatic', 'ssi', 
            'doanh_nghiệp', 'việc_làm', # Sometimes noise if just list columns
        }

    # ... (skipping unchanged methods) ...



    # Global cache for the singleton model
    _SHARED_VNCORENLP_MODEL = None

    def _load_vncorenlp(self):
        # 1. Return instance-level if already set
        if self.vncorenlp_model:
            return self.vncorenlp_model

        # 2. Check Class-level singleton
        if KeywordExtractor._SHARED_VNCORENLP_MODEL:
            self.vncorenlp_model = KeywordExtractor._SHARED_VNCORENLP_MODEL
            return self.vncorenlp_model

        # 3. Initialize if global is empty
        try:
            import py_vncorenlp
            print(f"[VnCoreNLP] Checking for models at: {self.vncorenlp_path}")
            
            # Ensure model is downloaded
            models_dir = os.path.join(self.vncorenlp_path, 'models')
            if not os.path.exists(models_dir):
                print("[VnCoreNLP] Models not found. Attempting download...")
                print("[VnCoreNLP] ⚠️  Note: This requires internet access and may fail on Kaggle.")
                py_vncorenlp.download_model(save_dir=self.vncorenlp_path)
                print("[VnCoreNLP] ✅ Download complete!")
            else:
                print(f"[VnCoreNLP] ✅ Models found at {models_dir}")
            
            # Load segmenter
            print("[VnCoreNLP] Loading word segmentation model...")
            # Use try-block specifically for wrapping the loader which triggers JVM
            KeywordExtractor._SHARED_VNCORENLP_MODEL = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=self.vncorenlp_path)
            self.vncorenlp_model = KeywordExtractor._SHARED_VNCORENLP_MODEL
            print("[VnCoreNLP] ✅ Model loaded successfully (Singleton)!")
            
        except Exception as e:
            # Check specifically for JVM error to provide helpful context
            if "VM is already running" in str(e):
                 print("[VnCoreNLP] ⚠️ VM already running. Attempting to recover existing controller if possible, or fallback.")
                 # In many cases we can't recover the JAVA object if we lost the reference. Fallback is safer.
            
            print(f"[VnCoreNLP] ❌ Failed to load: {e}")
            print("[VnCoreNLP] 🔄 Falling back to underthesea (CRF) segmentation")
            # Set flag to use fallback for this session
            KeywordExtractor._SHARED_VNCORENLP_MODEL = "FALLBACK"
            self.vncorenlp_model = "FALLBACK"
            
        return self.vncorenlp_model

    def _load_phonlp(self):
        if self.phonlp_model is None:
            import phonlp
            # Load PhoNLP from VinAI (auto-downloads to ~/.cache/phonlp if not present)
            self.phonlp_model = phonlp.load(save_dir=os.path.join(os.path.expanduser("~"), ".cache", "phonlp"))
        return self.phonlp_model

    def extract_keywords(self, text: str, max_keywords: int = 15) -> str:
        """
        Extract high-signal keywords from text.
        Returns a space-separated string of keywords.
        """
        if not text:
            return ""

        # 1. Alias Normalization (Phase 0)
        # This prepends canonical terms if informal ones are found
        text = normalize_with_aliases(text)
        
        # 2. Extract Locations (High Signal)
        found_locations = []
        text_lower = text.lower()
        for loc in self.known_locations:
            if len(loc) > 3 and loc.lower() in text_lower:
                found_locations.append(loc)

        # 3. Extract Taxonomy Keywords (High Signal)
        found_taxonomy = []
        for kw in self.taxonomy_keywords:
            if len(kw) > 3 and kw.lower() in text_lower:
                found_taxonomy.append(kw)

        # 4. Extract Date/Time (High Signal)
        found_temporal = []
        # Support: day/month, day/month/year, hh:mm, "ngày dd/mm", "tháng mm"
        temporal_patterns = [
            r'\d{1,2}/\d{1,2}(?:/\d{2,4})?', # 7/12, 07/12, 07/12/2025
            r'\bngay\s+\d{1,2}/\d{1,2}\b',    # ngay 7/12
            r'\bthang\s+\d{1,2}\b',           # thang 12
            r'\d{1,2}:\d{2}'                 # 10:37, 22:00
        ]
        text_no_accents = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]', ' ', text_lower)
        for pattern in temporal_patterns:
            matches = re.findall(pattern, text_lower)
            found_temporal.extend(matches)

        # 4. Clean and Tokenize with Word Segmentation
        try:
            if self.segmentation_method == "phonlp":
                # Use VnCoreNLP for word segmentation as recommended for PhoNLP-level tasks
                segmenter = self._load_vncorenlp()
                # Returns list of segmented sentences: ["Ông Nguyễn_Khắc_Chúc ...", "..."]
                segmented_sentences = segmenter.word_segment(text_lower)
                text_segmented = " ".join(segmented_sentences)
            else:
                import underthesea
                if self.segmentation_method == "transformer":
                    # Use deep learning model for better accuracy (requires more resources)
                    text_segmented = underthesea.word_tokenize(text_lower, format="text", model="deep")
                else:
                    # Default CRF-based fast segmentation
                    text_segmented = underthesea.word_tokenize(text_lower, format="text")
            
            # format="text" or VnCoreNLP output replaces spaces with underscores in compound words
            clean_text = re.sub(r'[^\w\s]', ' ', text_segmented)
            clean_text = re.sub(r'\d+', ' ', clean_text)
            words = clean_text.split()
        except Exception as e:
            # Fallback to simple split if error
            clean_text = re.sub(r'[^\w\s]', ' ', text_lower)
            clean_text = re.sub(r'\d+', ' ', clean_text)
            words = clean_text.split()

        # 5. Frequency Analysis
        filtered_words = [w for w in words if len(w) > 2 and w.replace('_', '') not in self.stopwords]
        word_counts = Counter(filtered_words)
        
        # Get most common topical words
        top_words = [w for w, c in word_counts.most_common(max_keywords)]

        # 6. Combine and Weight
        # Locations get triple weight, Taxonomy keywords get double
        keywords = found_locations * 2 + found_taxonomy * 2 + found_temporal + top_words
        
        # Deduplicate while preserving order (Locations first)
        seen = set()
        final_keywords = []
        for kw in keywords:
            kw_low = kw.lower()
            if kw_low not in seen:
                final_keywords.append(kw)
                seen.add(kw_low)
        
        return " ".join(final_keywords[:max_keywords])

    def batch_extract(self, texts: List[str]) -> List[str]:
        """Process a list of texts into keyword blobs."""
        from rich.progress import track
        if not texts: return []
        return [self.extract_keywords(t) for t in track(texts, description="[cyan]Extracting keywords...[/cyan]")]

if __name__ == "__main__":
    extractor = KeywordExtractor()
    sample = "Cơn bão số 3 đang gây mưa lớn tại thủ đô Hà Nội. Người dân Sài Gòn đang theo dõi tình hình bão Yagi."
    print(f"Original: {sample}")
    print(f"Keywords: {extractor.extract_keywords(sample)}")
