"""Render recovered historical source snapshots using the preserved cache.

These exports are reconstructions, not recovered original image bytes.
Only CACHE and OUT are overridden; the recovered source is left intact.
Run from the repository root with .venv/bin/python.
"""
from pathlib import Path
import importlib.util
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent.parent / "cache_rsg4_mechanism"

for directory in sorted(HERE.glob("draft_*")):
    source = directory / "draw_scarf_overview.original.py"
    spec = importlib.util.spec_from_file_location(directory.name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CACHE = CACHE
    module.OUT = directory / "fig_scarf_overview_reconstructed"
    module.main()
    plt.close("all")
