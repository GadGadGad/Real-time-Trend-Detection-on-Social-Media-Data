import json

nb_path = 'notebooks/analysis-playground_v2.ipynb'

try:
    with open(nb_path, 'r') as f:
        nb = json.load(f)

    print(f"Scanning {nb_path} for data loading...")
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'facebook_summarized' in source:
                print(f"\n--- Cell {i} ---")
                print(source)
                print("----------------")
except Exception as e:
    print(f"Error: {e}")
