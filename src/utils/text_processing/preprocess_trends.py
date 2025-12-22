"""
Google Trends Preprocessor
Cleans and filters raw Google Trends data before analysis.
Outputs: refined_trends.json
"""

import json
import csv
import argparse
import os
import re
from rich.console import Console

console = Console()

def load_trends_raw(csv_files):
    """Load raw trends from CSV files."""
    trends = {}
    for filepath in csv_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) < 5: continue
                    main_trend = row[0].strip()
                    vol_str = row[1].strip()
                    # Parse volume
                    clean_vol = vol_str.upper().replace(',', '').replace('.', '')
                    multiplier = 1000 if 'N' in clean_vol or 'K' in clean_vol else (1000000 if 'M' in clean_vol or 'TR' in clean_vol else 1)
                    num_parts = re.findall(r'\d+', clean_vol)
                    volume = int(num_parts[0]) * multiplier if num_parts else 0
                    
                    keywords = [k.strip() for k in row[4].split(',') if k.strip()]
                    if main_trend not in keywords: keywords.insert(0, main_trend)
                    trends[main_trend] = {"keywords": keywords, "volume": volume}
        except Exception as e:
            console.print(f"[red]Error loading {filepath}: {e}[/red]")
    return trends

def refine_trends(trends_dict, llm_provider="gemini", api_key=None, model_path=None, debug=False):
    """Use LLM to filter and merge trends."""
    try:
        from crawlers.llm_refiner import LLMRefiner
    except ImportError:
        from llm_refiner import LLMRefiner
    
    refiner = LLMRefiner(provider=llm_provider, api_key=api_key, model_path=model_path, debug=debug)
    
    if not refiner.enabled:
        console.print("[yellow]⚠️ LLM not available. Returning unrefined trends.[/yellow]")
        return trends_dict
    
    refined = refiner.refine_trends(trends_dict)
    
    if refined:
        original_count = len(trends_dict)
        
        # Remove filtered
        for bad_term in refined.get("filtered", []):
            if bad_term in trends_dict:
                del trends_dict[bad_term]
                
        # Merge synonym volumes
        for variant, canonical in refined.get("merged", {}).items():
            if variant in trends_dict and canonical in trends_dict:
                trends_dict[canonical]['volume'] += trends_dict[variant]['volume']
                trends_dict[canonical]['keywords'].extend(trends_dict[variant]['keywords'])
                del trends_dict[variant]
            elif variant in trends_dict:
                # Rename
                trends_dict[canonical] = trends_dict.pop(variant)
        
        console.print(f"✨ [green]Refined: {original_count} -> {len(trends_dict)} trends[/green]")
        console.print(f"   Removed: {len(refined.get('filtered', []))}, Merged: {len(refined.get('merged', {}))}")
    
    return trends_dict

def save_refined_trends(trends_dict, output_path):
    """Save refined trends to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(trends_dict, f, ensure_ascii=False, indent=2)
    console.print(f"💾 [green]Saved refined trends to {output_path}[/green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Google Trends Data")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input CSV files")
    parser.add_argument("--output", type=str, default="refined_trends.json", help="Output JSON file")
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "kaggle", "local"])
    parser.add_argument("--llm-model-path", type=str, help="Local model path or HuggingFace ID")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    console.print(f"📥 Loading trends from {len(args.input)} file(s)...")
    raw_trends = load_trends_raw(args.input)
    console.print(f"   Found {len(raw_trends)} raw trends.")
    
    console.print("🧹 Refining with LLM...")
    refined = refine_trends(raw_trends, llm_provider=args.llm_provider, model_path=args.llm_model_path, debug=args.debug)
    
    save_refined_trends(refined, args.output)
    console.print("✅ Done! Use --trends refined_trends.json in analyze_trends.py")
