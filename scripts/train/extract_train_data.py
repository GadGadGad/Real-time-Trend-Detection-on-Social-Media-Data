import os
import json
import pandas as pd
import glob
from rich.console import Console
from rich.progress import track

console = Console()

def extract_text_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = []
        for item in data:
            # Common fields in Facebook scrapers
            text = item.get('text') or item.get('content') or item.get('post_content')
            if text and isinstance(text, str) and len(text.strip()) > 30:
                texts.append(text.strip().replace('\n', ' '))
        return texts
    except Exception as e:
        console.print(f"[red]Error parsing {file_path}: {e}[/red]")
        return []

def extract_text_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        texts = []
        # Priority fields for news datasets
        possible_fields = ['refine_summary', 'summaried_text', 'summary', 'text', 'content', 'post_content']
        target_field = None
        for field in possible_fields:
            if field in df.columns:
                target_field = field
                break
        
        if target_field:
            for text in df[target_field].dropna():
                if isinstance(text, str) and len(text.strip()) > 30:
                    texts.append(text.strip().replace('\n', ' '))
        return texts
    except Exception as e:
        console.print(f"[red]Error parsing {file_path}: {e}[/red]")
        return []

def main(input_dirs, output_file):
    all_texts = []
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            console.print(f"[yellow]Skipping non-existent directory: {input_dir}[/yellow]")
            continue
            
        console.print(f"[cyan]Scanning {input_dir}...[/cyan]")
        
        # JSON files
        json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
        for f in track(json_files, description="Processing JSONs"):
            all_texts.extend(extract_text_from_json(f))
            
        # CSV files
        csv_files = glob.glob(os.path.join(input_dir, "**/*.csv"), recursive=True)
        for f in track(csv_files, description="Processing CSVs"):
            all_texts.extend(extract_text_from_csv(f))

    # Deduplicate and filter length
    unique_texts = list(set(all_texts))
    console.print(f"[green]Extracted {len(all_texts)} raw lines. Unique lines: {len(unique_texts)}[/green]")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in unique_texts:
            f.write(text + '\n')
            
    console.print(f"[bold green]Saved training corpus to {output_file}[/bold green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs='+', default=["crawlers/new_data"])
    parser.add_argument("--output", default="data/train_tsdae.txt")
    args = parser.parse_args()
    
    main(args.inputs, args.output)
