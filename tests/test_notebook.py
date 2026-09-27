import ast
import json
from pathlib import Path


def test_demo_notebook_structure_and_syntax() -> None:
    notebook_path = Path("notebooks") / "demo.ipynb"
    assert notebook_path.exists(), "notebooks/demo.ipynb must exist"

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Verify standard nbformat structure
    assert nb.get("nbformat") == 4
    assert "cells" in nb
    assert len(nb["cells"]) >= 10

    markdown_cells = [c for c in nb["cells"] if c.get("cell_type") == "markdown"]
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]

    assert len(markdown_cells) >= 4
    assert len(code_cells) >= 6

    # Verify Colab badge and key content
    full_markdown = "\n".join("".join(c.get("source", [])) for c in markdown_cells)
    assert "colab-badge.svg" in full_markdown
    assert "colab.research.google.com" in full_markdown

    # Verify all 4 required sections exist in markdown
    assert "Tokenization" in full_markdown and "Boundary Injection" in full_markdown
    assert "Transition Matrix" in full_markdown or "Transition Probability" in full_markdown
    assert "Temperature" in full_markdown
    assert "Perplexity" in full_markdown and "Coverage" in full_markdown

    # Verify embedded mini-corpus exists
    full_code = "\n".join("".join(c.get("source", [])) for c in code_cells)
    assert "FAIRY_TALES_TRAIN" in full_code
    assert "Once upon a time" in full_code
    assert "Three Little Pigs" in full_code or "three little pigs" in full_code

    # Verify Python syntax of every code cell using ast.parse
    for idx, cell in enumerate(code_cells, 1):
        source_lines = cell.get("source", [])
        code_str = "".join(source_lines)

        # Filter out Colab/IPython magic lines for ast parsing
        clean_lines = []
        for line in code_str.splitlines():
            trimmed = line.strip()
            if trimmed.startswith("!") or trimmed.startswith("%"):
                clean_lines.append(f"# {line}")
            else:
                clean_lines.append(line)
        clean_code = "\n".join(clean_lines)

        try:
            ast.parse(clean_code)
        except SyntaxError as e:
            assert False, f"Code cell {idx} failed Python syntax parsing: {e}"
