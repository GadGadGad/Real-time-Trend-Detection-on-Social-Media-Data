import pandas as pd
import glob
import os
import sys

DATA_DIR = "crawlers/new_data/news"

def check_content_nans():
    print(f"🧐 Checking for NaNs in 'content' column in {DATA_DIR}...\n")
    
    files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    
    total_articles = 0
    total_nans = 0
    
    results = []
    
    for f in files:
        try:
            source = os.path.basename(os.path.dirname(f))
            df = pd.read_csv(f)
            
            n_rows = len(df)
            total_articles += n_rows
            
            # Check content
            if 'content' in df.columns:
                n_na = df['content'].isna().sum()
                # Also check for empty strings or massive whitespace
                n_empty = (df['content'].astype(str).str.strip() == '').sum()
                
                # Treat "nan" string as nan
                n_str_nan = (df['content'].astype(str).str.lower() == 'nan').sum()
                
                real_problems = n_na + n_empty + n_str_nan
                total_nans += real_problems
                
                if real_problems > 0:
                    results.append({
                        "file": os.path.basename(f),
                        "source": source,
                        "total": n_rows,
                        "missing": real_problems,
                        "pct": (real_problems / n_rows) * 100
                    })
            else:
                print(f"⚠️  Column 'content' missing in {f}")
                
        except Exception as e:
            print(f"❌ Error reading {f}: {e}")

    print("\n" + "="*50)
    print(f"SUMMARY: {total_articles} articles checked.")
    print(f"TOTAL MISSING/EMPTY CONTENT: {total_nans} ({total_nans/total_articles*100:.2f}%)")
    print("="*50)
    
    if results:
        res_df = pd.DataFrame(results)
        res_df = res_df.sort_values('missing', ascending=False)
        print("\n📄 Detailed Breakdown (Only files with issues):")
        print(res_df.to_string(index=False))
        
        # Save bad rows to inspect?
        # Maybe later if user asks.
    else:
        print("\n✅ No NaN content found!")

if __name__ == "__main__":
    check_content_nans()
