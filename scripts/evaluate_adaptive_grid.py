'''
评估现有密度四叉树
'''

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsta.grid_evaluation import evaluate_grid

POINTS = ROOT / "data/processed/world_tracks.parquet"
CELLS = ROOT / "data/processed/adaptive_grid_cells.csv"


def main():
    points = pd.read_parquet(POINTS)
    cells = pd.read_csv(CELLS)

    result = evaluate_grid(points, cells)

    print("\n=== DENSITY QUADTREE EVALUATION ===")

    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
