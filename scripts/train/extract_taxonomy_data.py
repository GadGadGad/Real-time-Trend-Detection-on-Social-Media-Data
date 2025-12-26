"""
Extract taxonomy training data from existing pipeline results or by running LLM classification.
Outputs: data/taxonomy_train.jsonl with format {"text": "...", "label": "T1"}
"""
import json
import os
import glob
from rich.console import Console

console = Console()

# Configuration
INPUT_PATTERNS = [
    "crawlers/results/results_*.json",
    "crawlers/new_data/facebook/*.json"
]
OUTPUT_FILE = "data/taxonomy_train.jsonl"

# T1-T7 Label descriptions for keyword fallback
CATEGORY_KEYWORDS = {
    "T1": ["tai nạn", "cháy", "lũ", "bão", "động đất", "dịch bệnh", "tử vong", "thiệt mạng", "khẩn cấp"],
    "T2": ["nghị định", "thông tư", "chính sách", "chính phủ", "quốc hội", "thủ tướng", "chủ tịch nước"],
    "T3": ["scandal", "lùm xùm", "tẩy chay", "tranh cãi", "bê bối", "tố cáo", "khiếu nại"],
    "T4": ["sản phẩm", "công nghệ", "AI", "startup", "đầu tư", "thị trường", "kinh doanh"],
    "T5": ["viral", "meme", "ca sĩ", "diễn viên", "phim", "nhạc", "giải trí", "hot girl", "tiktoker"],
    "T6": ["kẹt xe", "mất điện", "quá tải", "xăng", "dầu", "dịch vụ công"],
    "T7": ["thời tiết", "xổ số", "tỉ giá", "giá vàng", "kết quả bóng đá"]
}

def classify_by_keywords(text):
    """Simple keyword-based classification as fallback."""
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return cat
    return None

def extract_from_results():
    """Extract data from pipeline result files that already have LLM labels."""
    samples = []
    
    for pattern in INPUT_PATTERNS:
        files = glob.glob(pattern)
        console.print(f"[dim]Checking {pattern}: {len(files)} files[/dim]")
        
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                if not isinstance(data, list):
                    continue
                    
                for item in data:
                    content = item.get('post_content') or item.get('content') or item.get('text', '')
                    if not content or len(content) < 30:
                        continue
                    
                    # Try to get existing LLM label
                    label = item.get('category')
                    
                    # Fallback to keyword classification
                    if not label or label not in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']:
                        label = classify_by_keywords(content)
                    
                    if label:
                        samples.append({
                            "text": content[:512],  # Limit length for BERT
                            "label": label
                        })
            except Exception as e:
                console.print(f"[yellow]Skip {f}: {e}[/yellow]")
                
    return samples

def main():
    console.print("[cyan]Extracting taxonomy training data...[/cyan]")
    
    samples = extract_from_results()
    
    if not samples:
        console.print("[red]No samples found. Please run the pipeline with --use-llm first.[/red]")
        return
    
    # Deduplicate by text
    seen = set()
    unique_samples = []
    for s in samples:
        key = s['text'][:100]
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)
    
    # Show distribution
    from collections import Counter
    dist = Counter(s['label'] for s in unique_samples)
    console.print(f"[green]Total unique samples: {len(unique_samples)}[/green]")
    console.print(f"[dim]Distribution: {dict(dist)}[/dim]")
    
    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for s in unique_samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    
    console.print(f"[bold green]Saved to {OUTPUT_FILE}[/bold green]")

if __name__ == "__main__":
    main()
