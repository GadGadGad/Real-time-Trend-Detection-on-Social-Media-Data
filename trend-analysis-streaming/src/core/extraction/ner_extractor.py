"""
NER (Named Entity Recognition) Extractor for Vietnamese Text
Supports two backends:
1. ELECTRA (NlpHUST/ner-vietnamese-electra-base) - More accurate, requires GPU
2. Underthesea - Lighter, rule-based + CRF

Entity Types:
- PER: Person names (Công Phượng, Quang Hải, etc.)
- LOC: Locations (Hà Nội, Sài Gòn, bão Yagi, etc.)
- ORG: Organizations (VinGroup, VTV, Bộ Y tế, etc.)
- MISC: Miscellaneous entities
"""

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
from rich.console import Console
import torch

try:
    from src.utils.config.locations import get_known_locations
    VIETNAM_LOCS = get_known_locations()
except ImportError:
    VIETNAM_LOCS = []

console = Console()

# --- Backend Detection ---
HAS_UNDERTHESEA = False
HAS_ELECTRA = False
HAS_NER = False

try:
    from underthesea import ner as underthesea_ner
    HAS_UNDERTHESEA = True
    HAS_NER = True
except ImportError:
    pass

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    HAS_ELECTRA = True
    HAS_NER = True
except ImportError:
    pass

if not HAS_NER:
    console.print("[yellow]Warning: No NER backend available. Install underthesea or transformers.[/yellow]")

# --- ELECTRA Model Singleton ---
_electra_model = None
_electra_tokenizer = None
_electra_device = None

ELECTRA_MODEL_NAME = "NlpHUST/ner-vietnamese-electra-base"
ELECTRA_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_electra_ner(device: str = "auto"):
    """Load ELECTRA NER model (singleton) with OOM Protection."""
    global _electra_model, _electra_tokenizer, _electra_device
    
    if _electra_model is not None:
        return _electra_model, _electra_tokenizer, _electra_device
    
    if not HAS_ELECTRA:
        return None, None, None
    
    try:
        # [SMART DEVICE SELECTION]
        if device == "auto":
             # Nếu VRAM > 6GB thì mới dám dùng GPU cho NER, không thì dùng CPU để nhường cho Embedder
            if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 6e9:
                _electra_device = "cuda"
            else:
                _electra_device = "cpu"
        else:
            _electra_device = device
            
        console.print(f"[cyan]🔌 Loading ELECTRA NER model ({ELECTRA_MODEL_NAME}) on {_electra_device}...[/cyan]")
        _electra_tokenizer = AutoTokenizer.from_pretrained(ELECTRA_MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(ELECTRA_MODEL_NAME)
        
        # [OOM PROTECTION]
        try:
            model.to(_electra_device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                console.print("[yellow]⚠️ GPU OOM for NER. Falling back to CPU...[/yellow]")
                _electra_device = "cpu"
                model.to("cpu")
                torch.cuda.empty_cache()
            else:
                raise e
        
        _electra_model = model
        _electra_model.eval()
        console.print(f"[green]✅ ELECTRA NER loaded successfully on {_electra_device}[/green]")
        return _electra_model, _electra_tokenizer, _electra_device
    except Exception as e:
        console.print(f"[yellow]⚠️ Failed to load ELECTRA NER: {e}. Falling back to Underthesea.[/yellow]")
        return None, None, None


def extract_entities_electra(text: str) -> Dict[str, List[str]]:
    """Extract entities using ELECTRA model."""
    model, tokenizer, device = load_electra_ner()
    
    if model is None:
        return {}
    
    entities = defaultdict(list)
    
    try:
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [ELECTRA_LABELS[p] for p in predictions[0].cpu().numpy()]
        
        # Parse BIO tags
        current_entity = ""
        current_type = None
        
        for token, label in zip(tokens, labels):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
                
            # Handle subwords (##)
            if token.startswith("##"):
                if current_entity:
                    current_entity += token[2:]
                continue
            
            if label.startswith("B-"):
                # Save previous
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                # Start new
                current_type = label[2:]
                current_entity = token
                
            elif label.startswith("I-") and current_type:
                current_entity += " " + token
                
            else:  # O tag
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                current_entity = ""
                current_type = None
        
        # Last entity
        if current_entity and current_type:
            entities[current_type].append(current_entity.strip())
            
    except Exception as e:
        console.print(f"[red]ELECTRA NER error: {e}[/red]")
    
    return dict(entities)


def extract_entities_underthesea(text: str) -> Dict[str, List[str]]:
    """Extract entities using Underthesea."""
    if not HAS_UNDERTHESEA:
        return {}
    
    entities = defaultdict(list)
    
    try:
        # underthesea.ner returns list of tuples: (word, pos, chunk, entity)
        ner_result = underthesea_ner(text)
        
        current_entity = ""
        current_type = None
        
        for item in ner_result:
            word = item[0]
            entity_tag = item[3] if len(item) > 3 else "O"
            
            if entity_tag.startswith("B-"):
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                current_type = entity_tag[2:]
                current_entity = word
                
            elif entity_tag.startswith("I-") and current_type:
                current_entity += " " + word
                
            else:
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                current_entity = ""
                current_type = None
        
        if current_entity and current_type:
            entities[current_type].append(current_entity.strip())
            
    except Exception as e:
        console.print(f"[red]Underthesea NER error: {e}[/red]")
    
    return dict(entities)


def extract_entities(text: str, backend: str = "auto") -> Dict[str, List[str]]:
    """
    Extract named entities from Vietnamese text.
    """
    if not text or len(text.strip()) == 0:
        return {}
    
    entities = {}
    
    if backend == "electra":
        entities = extract_entities_electra(text)
    elif backend == "underthesea":
        entities = extract_entities_underthesea(text)
    else:  # auto
        if HAS_ELECTRA:
            entities = extract_entities_electra(text)
        if not entities and HAS_UNDERTHESEA:
            entities = extract_entities_underthesea(text)
    
    # --- DICTIONARY-BASED ENHANCEMENT ---
    # Supplement with our custom locations list
    if VIETNAM_LOCS:
        text_lower = text.lower()
        if "LOC" not in entities:
            entities["LOC"] = []
        for loc in VIETNAM_LOCS:
            if len(loc) > 3 and loc.lower() in text_lower:
                if loc not in entities["LOC"]:
                    entities["LOC"].append(loc)
    
    return entities


def get_unique_entities(text: str) -> Set[str]:
    """Get all unique entities from text as a flat set."""
    entities = extract_entities(text)
    unique = set()
    for entity_list in entities.values():
        unique.update(entity_list)
    return unique


def enrich_text_with_entities(text: str, weight_factor: int = 2) -> str:
    """Enrich text by prepending extracted entities."""
    if not HAS_NER:
        return text
        
    entities = get_unique_entities(text)
    
    if not entities:
        return text
        
    # Create entity prefix
    entity_str = " ".join(entities)
    enriched = (entity_str + " ") * weight_factor + text
    
    return enriched


def batch_extract_entities(texts: List[str], show_progress: bool = True) -> List[Dict[str, List[str]]]:
    """Extract entities from multiple texts."""
    if not HAS_NER:
        return [{} for _ in texts]
    
    results = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        if show_progress and (i + 1) % 100 == 0:
            console.print(f"[dim]NER Progress: {i+1}/{total}[/dim]")
        results.append(extract_entities(text))
        
    return results


def batch_enrich_texts(texts: List[str], weight_factor: int = 2, show_progress: bool = True) -> List[str]:
    """Enrich multiple texts with their entities."""
    if not HAS_NER:
        return texts
    
    results = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        if show_progress and (i + 1) % 100 == 0:
            console.print(f"[dim]NER Enrichment Progress: {i+1}/{total}[/dim]")
        results.append(enrich_text_with_entities(text, weight_factor))
        
    return results


def summarize_entities(texts: List[str]) -> Dict[str, int]:
    """Get entity frequency summary across multiple texts."""
    entity_counts = defaultdict(int)
    
    for text in texts:
        entities = get_unique_entities(text)
        for entity in entities:
            entity_counts[entity] += 1
            
    return dict(sorted(entity_counts.items(), key=lambda x: x[1], reverse=True))


# Demo/Test
if __name__ == "__main__":
    sample_texts = [
        "Thầy Park Hang-seo dẫn đội tuyển Việt Nam đến Qatar thi đấu vòng loại World Cup.",
        "Bão Yagi đổ bộ vào Hà Nội gây thiệt hại nặng nề cho người dân thủ đô.",
        "Vingroup công bố kế hoạch mở rộng tại TP.HCM và Đà Nẵng."
    ]
    
    console.print("[bold cyan]🔍 Testing NER Extraction...[/bold cyan]\n")
    
    for text in sample_texts:
        console.print(f"[bold]Input:[/bold] {text}")
        entities = extract_entities(text)
        console.print(f"[green]Entities:[/green] {entities}")
        enriched = enrich_text_with_entities(text)
        console.print(f"[blue]Enriched:[/blue] {enriched[:100]}...")
        console.print()