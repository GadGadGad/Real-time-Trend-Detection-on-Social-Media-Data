
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from streaming.kafka_producer import load_kaggle_data

def test_loading():
    print("Testing load_kaggle_data()...")
    try:
        data = load_kaggle_data()
        
        if isinstance(data, pd.DataFrame):
            print(f"✅ Loaded DataFrame with shape: {data.shape}")
            print("Columns:", data.columns.tolist())
            print("Sample content:")
            print(data[['content', 'source', 'final_topic']].head())
            
            # Verify we have Facebook and News sources
            sources = data['source'].unique()
            print(f"Sources found: {sources}")
            if 'Facebook' in sources and 'News' in sources:
                print("✅ Both Facebook and News sources detected!")
            else:
                print("⚠️ Warning: Missing some sources.")
                
        elif isinstance(data, list):
            print(f"✅ Loaded List with {len(data)} items")
            print("Sample item:", data[0])
        else:
            print(f"❌ Unexpected data type: {type(data)}")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
