"""
NER (Named Entity Recognition) Extractor for Vietnamese Text
Uses underthesea library for Vietnamese NLP processing.

Entity Types:
- PER: Person names (Công Phượng, Quang Hải, etc.)
- LOC: Locations (Hà Nội, Sài Gòn, bão Yagi, etc.)
- ORG: Organizations (VinGroup, VTV, Bộ Y tế, etc.)
- MISC: Miscellaneous entities
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
from rich.console import Console
try:
    from src.utils.config.locations import get_known_locations
    VIETNAM_LOCS = get_known_locations()
except ImportError:
    VIETNAM_LOCS = []

console = Console()

try:
    from underthesea import ner
    HAS_NER = True
except ImportError:
    HAS_NER = False
    console.print("[yellow]Warning: underthesea not installed. NER features disabled.[/yellow]")
    console.print("[dim]Install with: pip install underthesea[/dim]")


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from Vietnamese text.
    
    Args:
        text: Vietnamese text to extract entities from
        
    Returns:
        Dictionary mapping entity types to list of entity values
        Example: {"PER": ["Công Phượng", "Quang Hải"], "LOC": ["Hà Nội"]}
    """
    if not HAS_NER:
        return {}
    
    if not text or len(text.strip()) == 0:
        return {}
    
    entities = defaultdict(list)
    
    try:
        # underthesea.ner returns list of tuples: (word, pos, chunk, entity)
        # Entity format: B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, O
        ner_result = ner(text)
        
        current_entity = ""
        current_type = None
        
        for item in ner_result:
            word = item[0]
            entity_tag = item[3] if len(item) > 3 else "O"
            
            if entity_tag.startswith("B-"):
                # Save previous entity if exists
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                    
                # Start new entity
                current_type = entity_tag[2:]  # Remove "B-" prefix
                current_entity = word
                
            elif entity_tag.startswith("I-") and current_type:
                # Continue current entity
                current_entity += " " + word
                
            else:  # "O" tag - not an entity
                # Save previous entity if exists
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                current_entity = ""
                current_type = None
        
        # Don't forget the last entity
        if current_entity and current_type:
            entities[current_type].append(current_entity.strip())
            
    except Exception as e:
        console.print(f"[red]NER extraction error: {e}[/red]")
        
    # --- DICTIONARY-BASED ENHANCEMENT ---
    # Supplement underthesea with our custom locations list
    text_lower = text.lower()
    for loc in VIETNAM_LOCS:
        if len(loc) > 3 and loc.lower() in text_lower:
            if loc not in entities["LOC"]:
                entities["LOC"].append(loc)
    
    return dict(entities)


def get_unique_entities(text: str) -> Set[str]:
    """
    Get all unique entities from text as a flat set.
    
    Args:
        text: Vietnamese text to process
        
    Returns:
        Set of unique entity strings
    """
    entities = extract_entities(text)
    unique = set()
    for entity_list in entities.values():
        unique.update(entity_list)
    return unique


def enrich_text_with_entities(text: str, weight_factor: int = 2) -> str:
    """
    Enrich text by prepending extracted entities.
    This gives more weight to named entities in embedding-based similarity.
    
    Args:
        text: Original text
        weight_factor: How many times to repeat the entities (default: 2)
        
    Returns:
        Enriched text with entities prepended
        
    Example:
        Input: "Thầy Park dẫn đội tuyển Việt Nam đến Qatar"
        Output: "Park Việt Nam Qatar Park Việt Nam Qatar Thầy Park dẫn đội tuyển Việt Nam đến Qatar"
    """
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
    """
    Extract entities from multiple texts.
    
    Args:
        texts: List of texts to process
        show_progress: Whether to show progress bar
        
    Returns:
        List of entity dictionaries
    """
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
    """
    Enrich multiple texts with their entities.
    
    Args:
        texts: List of texts to enrich
        weight_factor: Entity repetition factor
        show_progress: Whether to show progress
        
    Returns:
        List of enriched texts
    """
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
    """
    Get entity frequency summary across multiple texts.
    Useful for understanding what entities are most common.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        Dictionary mapping entities to their frequency
    """
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
