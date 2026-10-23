"""Materialize reproducible DSTA cube delivery tables without changing upstream data."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dsta.cube_data import export_cube_tables

if __name__ == "__main__":
    paths = export_cube_tables(ROOT, ROOT / "outputs/ppt_case/cube")
    print("Saved:", *paths, sep="\n - ")
