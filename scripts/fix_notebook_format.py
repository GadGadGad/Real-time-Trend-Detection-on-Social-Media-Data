import nbformat
import sys

def fix_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Standardizing cells
    for cell in nb.cells:
        if isinstance(cell.source, str):
            # Split by line and keep newlines
            lines = cell.source.splitlines(keepends=True)
            cell.source = lines
        
        # Ensure outputs are valid list
        if cell.cell_type == 'code' and not hasattr(cell, 'outputs'):
            cell.outputs = []
        if cell.cell_type == 'code' and not hasattr(cell, 'execution_count'):
            cell.execution_count = None

    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Fixed format for {path}")

if __name__ == "__main__":
    fix_notebook('notebooks/analysis-playground-v1.ipynb')
