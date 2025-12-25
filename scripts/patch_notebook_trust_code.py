import nbformat
import os

NOTEBOOK_PATH = '/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project/notebooks/analysis-playground-v1.ipynb'

def patch_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    target_snippet = "def find_matches_segmented"
    found = False

    for cell in nb.cells:
        if cell.cell_type == 'code' and target_snippet in cell.source:
            # Replace the problematic lines inside the function
            new_source = cell.source.replace(
                "post_embeddings = get_embeddings(post_contents_seg, model_name=model_name)",
                "post_embeddings = get_embeddings(post_contents_seg, model_name=model_name, trust_remote_code=True)"
            ).replace(
                "embedder = SentenceTransformer(model_name)",
                "embedder = SentenceTransformer(model_name, trust_remote_code=True)"
            )
            cell.source = new_source
            found = True
            break

    if found:
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Successfully patched {NOTEBOOK_PATH}")
    else:
        print("Function 'find_matches_segmented' not found in notebook.")

if __name__ == "__main__":
    patch_notebook()
