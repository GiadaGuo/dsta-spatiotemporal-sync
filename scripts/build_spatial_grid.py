'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-10-22 16:08:55
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-10-22 16:09:35
FilePath: /dsta-spatiotemporal-sync/scripts/build_spatial_grid.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/world_tracks.parquet")
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

GRID_SIZES = [1.0, 2.0, 5.0, 10.0]


def main():
    df = pd.read_parquet(INPUT)

    # 统一使用固定空间原点
    # 注意：不能为不同模态分别计算原点
    origin_x = np.floor(df["world_x"].min() / 10) * 10
    origin_y = np.floor(df["world_y"].min() / 10) * 10

    print("Grid origin:", origin_x, origin_y)

    for size in GRID_SIZES:
        label = f"{size:g}m"

        gx = np.floor(
            (df["world_x"] - origin_x) / size
        ).astype("int64")

        gy = np.floor(
            (df["world_y"] - origin_y) / size
        ).astype("int64")

        df[f"grid_x_{label}"] = gx
        df[f"grid_y_{label}"] = gy

        df[f"grid_id_{label}"] = (
            gx.astype(str) + "_" + gy.astype(str)
        )

        print(
            f"{label}: "
            f"{df[f'grid_id_{label}'].nunique()} "
            "occupied cells"
        )

    output = OUTPUT / "gridded_tracks.parquet"
    df.to_parquet(output, index=False)

    print("\nTotal observations:", len(df))
    print("Saved:", output)


if __name__ == "__main__":
    main()
