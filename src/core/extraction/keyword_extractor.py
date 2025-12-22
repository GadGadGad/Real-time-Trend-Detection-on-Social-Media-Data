import pandas as pd
import numpy as np
import re
import os
from typing import List, Set, Dict
from collections import Counter
import json
from src.utils.config.locations import get_known_locations
from src.utils.text_processing.alias_normalizer import normalize_with_aliases
from src.core.extraction.taxonomy_keywords import get_all_event_keywords

class KeywordExtractor:
    def __init__(self, segmentation_method: str = "underthesea", use_llm=False, llm_refiner=None):
        """
        segmentation_method: 
            - "underthesea" (fast, CRF) 
            - "transformer" (accurate, Underthesea Deep)
            - "phonlp" (very accurate, VinAI Multi-task Transformer with VnCoreNLP segmenter)
            - "bert" (Vietnamese PhoBERT + NER, requires tpha4308/keyword-extraction-viet)
        use_llm: If True, uses the provided llm_refiner for extraction (slower but semantic).
        llm_refiner: Instance of LLMRefiner.
        """
        self.known_locations = get_known_locations()
        self.taxonomy_keywords = get_all_event_keywords()
        self.segmentation_method = segmentation_method
        self.use_llm = use_llm
        self.llm = llm_refiner
        
        # Cache Init
        self.cache_path = "data/cache/keyword_llm_cache.json"
        self.cache = self._load_cache()
        
        self.phonlp_model = None
        self.vncorenlp_model = None
        self.vncorenlp_path = os.path.join(os.path.expanduser("~"), ".cache", "vncorenlp_models")
        
        # BERT pipeline (lazy loaded)
        self.bert_pipeline = None
        self.bert_repo_path = os.path.join(os.path.expanduser("~"), ".cache", "keyword-extraction-viet")
        
        # Common Vietnamese stopwords (minimal set for extraction)
        self.stopwords = {
            'và', 'của', 'là', 'có', 'trong', 'đã', 'ngày', 'theo', 'với', 
            'cho', 'người', 'những', 'tại', 'về', 'các', 'được', 'ra', 'khi',
            'mới', 'này', 'cho', 'nhiều',
            # Technical artifacts/noise
            'volume', 'keywords', 'time', 'automatic', 'ssi', 
            'doanh_nghiệp', 'việc_làm', # Sometimes noise if just list columns
        }

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return {}
        return {}

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _extract_with_llm(self, text):
        if text in self.cache:
            return self.cache[text]
            
        if not self.llm:
            return ""
            
        prompt = f"""
        Extract the Main Keywords from this Vietnamese text.
        Focus on:
        - Entities (People, Organizations, Locations)
        - Main Event / Topic
        - Important Dates/Times

        Text: "{text}"

        Output: JSON list of strings only. Example: ["Hà Nội", "bão Yagi", "ngập lụt"]
        """
        
        try:
            resp = self.llm._generate(prompt)
            # Use LLMRefiner's helper if available, or manual extract
            if hasattr(self.llm, '_extract_json'):
                kws = self.llm._extract_json(resp, is_list=True)
            else:
                # Basic fallback
                import re
                kws = re.findall(r'"([^"]+)"', resp)
            
            if kws:
                result = " ".join(kws[:10])
                self.cache[text] = result
                # Auto-save every 50 new items or similar? 
                # For now rely on manual save or pipeline end call.
                return result
        except Exception as e:
            print(f"LLM Extract Error: {e}")
            
        return "" # Fallback to rule-based if LLM fails

    def _init_bert_pipeline(self):
        """Initialize the BERT-based Vietnamese keyword extraction pipeline."""
        if self.bert_pipeline is not None:
            return self.bert_pipeline
        
        import subprocess
        import sys
        import torch
        
        # 1. Clone repo if not exists
        if not os.path.exists(self.bert_repo_path):
            print("[BERT-KW] Cloning keyword-extraction-viet repo...")
            os.makedirs(os.path.dirname(self.bert_repo_path), exist_ok=True)
            subprocess.run(
                ["git", "clone", "https://huggingface.co/tpha4308/keyword-extraction-viet", self.bert_repo_path],
                check=True
            )
            print("[BERT-KW] ✅ Repo cloned.")
        
        # 2. Check for pretrained models, download if needed
        phobert_path = os.path.join(self.bert_repo_path, "pretrained-models", "phobert.pt")
        ner_path = os.path.join(self.bert_repo_path, "pretrained-models", "ner-vietnamese-electra-base.pt")
        
        if not os.path.exists(phobert_path) or not os.path.exists(ner_path):
            print("[BERT-KW] Downloading and saving PhoBERT + NER models...")
            os.makedirs(os.path.join(self.bert_repo_path, "pretrained-models"), exist_ok=True)
            
            from transformers import AutoModel, AutoModelForTokenClassification
            
            phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
            phobert.eval()
            torch.save(phobert, phobert_path)
            print("[BERT-KW] ✅ PhoBERT saved.")
            
            ner_model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
            ner_model.eval()
            torch.save(ner_model, ner_path)
            print("[BERT-KW] ✅ NER model saved.")
        
        # 3. Add repo to path and import pipeline
        if self.bert_repo_path not in sys.path:
            sys.path.insert(0, self.bert_repo_path)
        
        from pipeline import KeywordExtractorPipeline
        
        phobert = torch.load(phobert_path, weights_only=False)
        phobert.eval()
        ner_model = torch.load(ner_path, weights_only=False)
        ner_model.eval()
        
        self.bert_pipeline = KeywordExtractorPipeline(phobert, ner_model)
        print("[BERT-KW] ✅ Pipeline loaded.")
        return self.bert_pipeline
    
    def _extract_with_bert(self, text: str, top_n: int = 10) -> str:
        """Extract keywords using Vietnamese PhoBERT + NER pipeline."""
        try:
            pipeline = self._init_bert_pipeline()
            # The pipeline expects {"title": ..., "text": ...}
            # We'll use the first 100 chars as title proxy if not provided
            title = text[:100] if len(text) > 100 else text
            inp = {"title": title, "text": text}
            kws = pipeline(inputs=inp, min_freq=1, ngram_n=(1, 3), top_n=top_n, diversify_result=False)
            
            if kws:
                return " ".join(kws)
        except Exception as e:
            print(f"[BERT-KW] ❌ Error: {e}. Falling back to rule-based.")
        
        return ""  # Fallback

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

        # 0. LLM Extraction (Optional)
        if self.use_llm:
            llm_res = self._extract_with_llm(text)
            if llm_res: return llm_res

        # 0.5 BERT Extraction (Optional, Vietnamese PhoBERT + NER)
        if self.segmentation_method == "bert":
            bert_res = self._extract_with_bert(text, top_n=max_keywords)
            if bert_res: return bert_res
            # Fall through to rule-based if BERT fails

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
        results = [self.extract_keywords(t) for t in track(texts, description="[cyan]Extracting keywords...[/cyan]")]
        if self.use_llm:
            self.save_cache()
        return results

if __name__ == "__main__":
    extractor = KeywordExtractor()
    sample = "Cơn bão số 3 đang gây mưa lớn tại thủ đô Hà Nội. Người dân Sài Gòn đang theo dõi tình hình bão Yagi."
    print(f"Original: {sample}")
    print(f"Keywords: {extractor.extract_keywords(sample)}")
