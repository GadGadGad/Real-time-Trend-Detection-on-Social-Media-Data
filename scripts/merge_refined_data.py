
import pandas as pd
import os
import glob
from rich.console import Console
from rich.table import Table

console = Console()

DATA_DIR = "summarized_data"

def merge_refined_data():
    console.print(f"[bold cyan]🚀 Starting Merge Process in {DATA_DIR}...[/bold cyan]")
    
    refined_files = glob.glob(os.path.join(DATA_DIR, "*_refined.csv"))
    
    if not refined_files:
        console.print("[red]❌ No *_refined.csv files found.[/red]")
        return

    print(f"{'Source':<30} | {'Target':<30} | {'Total':<10} | {'Valid':<10} | {'Merged':<10} | {'Status'}")
    print("-" * 120)

    for refined_path in refined_files:
        filename = os.path.basename(refined_path)
        source_name = filename.replace("_refined.csv", "")
        
        # Determine Target File
        target_path = os.path.join(DATA_DIR, f"{source_name}_summarized_content.csv")
        
        # Pattern 2: all_[source]_summarized.csv (e.g. facebook)
        if not os.path.exists(target_path):
             target_path = os.path.join(DATA_DIR, f"all_{source_name}_summarized.csv")
             
        if not os.path.exists(target_path):
            console.print(f"[yellow]DEBUG: Target not found for {filename}. Checked: {target_path}[/yellow]")
            print(f"{filename:<30} | {'N/A':<30} | {'-':<10} | {'-':<10} | {'-':<10} | Target Not Found")
            continue
            
        try:
            # Read Files
            console.print(f"[dim]Reading {refined_path} and {target_path}...[/dim]")
            
            # Use 'python' engine for more robust parsing of messy CSVs
            # on_bad_lines='skip' will ignore rows with too many fields instead of crashing
            try:
                df_refined = pd.read_csv(refined_path, on_bad_lines='skip', engine='python') 
            except Exception as e:
                # Fallback to default engine if python engine fails for some reason
                 console.print(f"[yellow]⚠️ Python engine failed for {filename}, trying default: {e}[/yellow]")
                 df_refined = pd.read_csv(refined_path, on_bad_lines='skip')

            df_target = pd.read_csv(target_path)
            
            # Normalize columns
            df_refined.columns = [c.strip() for c in df_refined.columns]
            
            total_refined = len(df_refined)
            
            # Clean Refined Data
            if 'refined_summary' not in df_refined.columns:
                 # Try to find it
                 cols = [c for c in df_refined.columns if 'refined' in c]
                 if cols: df_refined.rename(columns={cols[0]: 'refined_summary'}, inplace=True)

            if 'refined_summary' not in df_refined.columns:
                 print(f"{filename:<30} | {os.path.basename(target_path):<30} | {total_refined:<10} | {'-':<10} | {'-':<10} | No 'refined_summary' col")
                 continue

            valid_mask = (
                df_refined['refined_summary'].notna() & 
                (df_refined['refined_summary'] != "") & 
                (df_refined['refined_summary'].astype(str).str.strip() != "") &
                (df_refined['refined_summary'].astype(str).str.lower() != "nan")
            )
            
            df_refined_clean = df_refined[valid_mask].copy()
            valid_count = len(df_refined_clean)
            
            # Merge
            merged_count = 0
            if 'index' in df_refined_clean.columns:
                # Force index to int
                # Force index to int, coercing errors (like 'index' string) to NaN
                df_refined_clean['index'] = pd.to_numeric(df_refined_clean['index'], errors='coerce')
                # Drop invalid indices
                df_refined_clean = df_refined_clean.dropna(subset=['index'])
                df_refined_clean['index'] = df_refined_clean['index'].astype(int)

                summary_map = dict(zip(df_refined_clean['index'], df_refined_clean['refined_summary']))
                
                if 'refined_summary' not in df_target.columns:
                    df_target['refined_summary'] = None
                
                for idx, summary in summary_map.items():
                    try:
                        if 0 <= idx < len(df_target):
                            df_target.at[idx, 'refined_summary'] = summary
                            merged_count += 1
                    except Exception:
                        pass
            else:
                 # Fallback: Merge by row position if lengths match roughly? 
                 # Unsafe. Better to error.
                 # Check if maybe the first column is the index but unnamed?
                 if df_refined_clean.columns[0] == 'Unnamed: 0':
                     # Use that
                     summary_map = dict(zip(df_refined_clean.iloc[:, 0].astype(int), df_refined_clean['refined_summary']))
                     if 'refined_summary' not in df_target.columns: df_target['refined_summary'] = None
                     for idx, summary in summary_map.items():
                         if 0 <= idx < len(df_target):
                             df_target.at[idx, 'refined_summary'] = summary
                             merged_count += 1
                 else:
                     print(f"{filename:<30} | {os.path.basename(target_path):<30} | {total_refined:<10} | {valid_count:<10} | {'0':<10} | Missing 'index' col. Found: {list(df_refined.columns)}")
                     continue

            # Save
            output_filename = f"{source_name}_merged.csv"
            output_path = os.path.join(DATA_DIR, output_filename)
            df_target.to_csv(output_path, index=False)
            
            print(f"{filename:<30} | {os.path.basename(target_path):<30} | {total_refined:<10} | {valid_count:<10} | {merged_count:<10} | Saved {output_filename}")
            
        except Exception as e:
            print(f"{filename:<30} | {os.path.basename(target_path):<30} | {'-':<10} | {'-':<10} | {'-':<10} | Error: {str(e)}")

    console.print("[bold cyan]✨ Merge Process Completed.[/bold cyan]")
    console.print("[bold cyan]✨ Merge Process Completed.[/bold cyan]")

if __name__ == "__main__":
    merge_refined_data()
