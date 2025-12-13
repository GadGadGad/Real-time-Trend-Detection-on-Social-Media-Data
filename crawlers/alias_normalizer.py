"""
Alias-based Text Normalizer for Vietnamese Trend Detection.
Replaces NER with more effective alias matching from Google Trends keywords.
"""

from typing import Dict, List, Set
from rich.console import Console

console = Console()

# Global alias dictionary - will be populated from trends data
TREND_ALIASES: Dict[str, List[str]] = {}


def build_alias_dictionary(trends: dict) -> Dict[str, List[str]]:
    """
    Build alias dictionary from Google Trends data.
    Each trend's keywords become aliases for that trend.
    
    Args:
        trends: Dictionary mapping trend names to their keywords
        
    Returns:
        Dictionary mapping canonical trend name to list of aliases
    """
    global TREND_ALIASES
    TREND_ALIASES = {}
    
    for main_trend, keywords in trends.items():
        canonical = main_trend.lower().strip()
        aliases = [k.lower().strip() for k in keywords if k.strip()]
        
        if canonical not in aliases:
            aliases.insert(0, canonical)
            
        TREND_ALIASES[canonical] = aliases
    
    console.print(f"[green]📚 Built {len(TREND_ALIASES)} alias groups from trends data[/green]")
    return TREND_ALIASES


def normalize_with_aliases(text: str, max_additions: int = 10) -> str:
    """
    Normalize text by adding known aliases.
    This improves semantic matching by adding related terms.
    
    Args:
        text: Text to normalize
        max_additions: Maximum number of alias terms to add
        
    Returns:
        Normalized text with aliases prepended
    """
    if not TREND_ALIASES:
        return text
        
    text_lower = text.lower()
    additions = set()
    
    for canonical, aliases in TREND_ALIASES.items():
        # Check if canonical term is in text
        if canonical in text_lower:
            additions.update(aliases[:5])
            additions.add(canonical)
        # Check if any alias is in text
        for alias in aliases[:5]:
            if len(alias) > 3 and alias in text_lower:
                additions.add(canonical)
                additions.update(aliases[:3])
                break
    
    if additions:
        # Limit additions to prevent text explosion
        addition_list = list(additions)[:max_additions]
        return " ".join(addition_list) + " " + text
    
    return text


def batch_normalize_texts(texts: List[str], show_progress: bool = True) -> List[str]:
    """
    Normalize multiple texts with alias matching.
    
    Args:
        texts: List of texts to normalize
        show_progress: Whether to show progress
        
    Returns:
        List of normalized texts
    """
    results = []
    total = len(texts)
    
    for i, text in enumerate(texts):
        if show_progress and (i + 1) % 500 == 0:
            console.print(f"[dim]Normalization progress: {i+1}/{total}[/dim]")
        results.append(normalize_with_aliases(text))
        
    return results


# Demo/Test
if __name__ == "__main__":
    # Sample trends data
    sample_trends = {
        "Bão Yagi": ["bão yagi", "bão số 3", "siêu bão yagi", "cơn bão số 3"],
        "Công Phượng": ["công phượng", "nguyễn công phượng", "cong phuong"],
        "Giá vàng": ["giá vàng", "vàng sjc", "giá vàng hôm nay"],
    }
    
    build_alias_dictionary(sample_trends)
    
    test_texts = [
        "Bão Yagi đổ bộ Hà Nội",
        "Cơn bão số 3 gây thiệt hại",
        "Công Phượng ghi bàn đẹp",
    ]
    
    console.print("\n[bold cyan]🧪 Testing Alias Normalization[/bold cyan]\n")
    for text in test_texts:
        normalized = normalize_with_aliases(text)
        console.print(f"[bold]Input:[/bold] {text}")
        console.print(f"[green]Output:[/green] {normalized}\n")
