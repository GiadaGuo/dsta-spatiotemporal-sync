'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-10-22 16:39:31
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-10-22 16:39:40
FilePath: /dsta-spatiotemporal-sync/scripts/build_quality_grid.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsta.grid_features import prepare_features
from dsta.grid_evaluation import evaluate_grid
from dsta.quality_quadtree import GridParams, QualityQuadtree


def main():
    tracks = pd.read_parquet(
        ROOT / "data/processed/world_tracks.parquet"
    )

    matches = pd.read_csv(
        ROOT / "outputs/reports/spatial_matches.csv",
        dtype={
            "vehicle_frame": str,
            "infrastructure_frame": str,
            "vehicle_track_id": str,
            "infrastructure_track_id": str,
        },
    )

    features = prepare_features(tracks, matches)

    params = GridParams()
    model = QualityQuadtree(params)

    cells = model.fit(features)

    result = evaluate_grid(features, cells)

    output = ROOT / "data/processed/quality_grid_cells.csv"
    cells.to_csv(output, index=False)

    print("\n=== QUALITY-AWARE QUADTREE ===")
    print("Normalization scales:", model.scales)
    print("\nCells by resolution:")
    print(cells.groupby("size").size().sort_index())

    print("\nEvaluation:")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
