
import pickle
import glob
import csv
import os

CACHE_FILE = 'streaming/embeddings_cache.pkl'
DATA_DIR = 'streaming/data'

def count_cache():
    if not os.path.exists(CACHE_FILE):
        print(f"Cache file {CACHE_FILE} not found.")
        return 0
    
    try:
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
            
        if isinstance(data, dict):
            count = len(data.get('posts', []))
        elif isinstance(data, tuple) or isinstance(data, list):
             # Likely (posts, embeddings)
             print(f"Cache data is type: {type(data)} with len {len(data)}")
             count = len(data[0]) 
        else:
             print(f"Unknown cache format: {type(data)}")
             return 0
             
        print(f"Cache contains: {count} posts.")
        return count
    except Exception as e:
        print(f"Error reading cache: {e}")
        return 0

def count_raw_data():
    csv_files = glob.glob(os.path.join(DATA_DIR, "**/*.csv"), recursive=True)
    total_rows = 0
    print(f"Found {len(csv_files)} CSV files.")
    for f in csv_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                rows = sum(1 for _ in reader)
                print(f"  - {os.path.basename(f)}: {rows} rows")
                total_rows += rows
        except Exception as e:
            print(f"Error reading {f}: {e}")
    
    print(f"Total raw posts: {total_rows}")
    return total_rows

if __name__ == "__main__":
    c_cache = count_cache()
    c_raw = count_raw_data()
    
    if c_cache < c_raw:
        print("\nMISMATCH: Cache is MISSING data.")
    elif c_cache > c_raw:
        print("\nMISMATCH: Cache has MORE data than current files (Old data?).")
    else:
        print("\nMATCH: Cache is up to date.")
