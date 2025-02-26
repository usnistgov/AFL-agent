#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

def add_colab_badge(notebook_path, repo_url, branch):
    """Add a Colab badge to the first markdown cell of a Jupyter notebook."""
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Get the relative path for the Colab URL
    rel_path = os.path.relpath(notebook_path, start=os.getcwd())
    colab_url = f"https://colab.research.google.com/github/{repo_url}/blob/{branch}/{rel_path}"
    
    # Check if the first cell is a markdown cell
    if notebook['cells'] and notebook['cells'][0]['cell_type'] == 'markdown':
        # Check if the badge is already there
        if "colab-badge.svg" not in notebook['cells'][0]['source'][0]:
            # Add the badge to the beginning of the first markdown cell
            badge_markdown = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n\n"
            notebook['cells'][0]['source'][0] = badge_markdown + notebook['cells'][0]['source'][0]
            
            # Write the modified notebook back
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1)
            
            print(f"Added Colab badge to {notebook_path}")
            return True
        else:
            print(f"Colab badge already exists in {notebook_path}")
            return False
    else:
        # If the first cell is not markdown, insert a new markdown cell with the badge
        badge_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})\n"
            ]
        }
        notebook['cells'].insert(0, badge_cell)
        
        # Write the modified notebook back
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"Added Colab badge to {notebook_path}")
        return True

def main():
    # Repository information
    repo_url = "usnistgov/AFL-agent"
    branch = "23-documentation-improvements-v2"
    
    # Find all notebook files in the docs directory
    docs_dir = Path("docs/source")
    notebook_files = list(docs_dir.glob("**/*.ipynb"))
    
    # Exclude checkpoint files
    notebook_files = [nb for nb in notebook_files if ".ipynb_checkpoints" not in str(nb)]
    
    if not notebook_files:
        print("No notebook files found in the docs directory.")
        return
    
    print(f"Found {len(notebook_files)} notebook files.")
    
    # Add badges to all notebooks
    modified_count = 0
    for nb_path in notebook_files:
        if add_colab_badge(nb_path, repo_url, branch):
            modified_count += 1
    
    print(f"Added Colab badges to {modified_count} notebooks.")

if __name__ == "__main__":
    main() 