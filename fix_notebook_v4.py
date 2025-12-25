import nbformat

def fix_notebook_plot_df(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Target: The visualization cell usage of plot_df
    # We want to insert the definition before it's used.
    target_code = "print(f'Visualizing {len(plot_df)} clustered posts...')"
    
    # Definition to inject
    new_definition = "plot_df = df_results[~df_results['topic_type'].isin(['Noise', 'Unassigned'])].copy()\n"

    modified = False
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if target_code in cell.source and new_definition not in cell.source:
                # Prepend the definition
                cell.source = new_definition + cell.source
                modified = True
                print("Found match, updating cell...")
                
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Successfully injected plot_df definition in {filepath}")
    else:
        print("Target text for plot_df fix not found or already fixed.")

if __name__ == "__main__":
    fix_notebook_plot_df("notebooks/analysis-playground-v1.ipynb")
