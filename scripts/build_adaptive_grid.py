'''
自适应四叉树
'''

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

INPUT = Path("data/processed/world_tracks.parquet")
OUTPUT = Path("data/processed")
FIGURES = Path("outputs/figures")

OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MIN_SIZE = 2.0
ROOT_SIZE = 64.0
THRESHOLD = 30
TIME_BUCKET_US = 1_000_000
MAX_DEPTH = 5


def build_tree(df, x0, y0, size, depth=0, path="R"):
    """递归构建四叉树叶子网格。"""
    if df.empty:
        return []

    occupancy = df[
        ["time_bucket", "source", "track_id"]
    ].drop_duplicates().shape[0]

    if (
        occupancy <= THRESHOLD
        or size <= MIN_SIZE
        or depth >= MAX_DEPTH
    ):
        return [{
            "grid_id": path,
            "x0": x0,
            "y0": y0,
            "size": size,
            "depth": depth,
            "occupancy": occupancy,
        }]

    half = size / 2
    leaves = []

    for i in range(2):
        for j in range(2):
            cx = x0 + i * half
            cy = y0 + j * half

            mask = (
                (df["world_x"] >= cx)
                & (df["world_x"] < cx + half)
                & (df["world_y"] >= cy)
                & (df["world_y"] < cy + half)
            )

            child = df.loc[mask]

            leaves.extend(
                build_tree(
                    child, cx, cy, half,
                    depth + 1, path + f"{i}{j}"
                )
            )

    return leaves


def main():
    df = pd.read_parquet(INPUT).copy()

    # 相对时间桶，避免大整数时间戳直接运算
    t0 = int(df["timestamp_us"].min())
    df["time_bucket"] = (
        (df["timestamp_us"] - t0) // TIME_BUCKET_US
    ).astype("int64")

    min_x = np.floor(df["world_x"].min() / ROOT_SIZE) * ROOT_SIZE
    min_y = np.floor(df["world_y"].min() / ROOT_SIZE) * ROOT_SIZE
    max_x = df["world_x"].max()
    max_y = df["world_y"].max()

    leaves = []

    # 使用多个 64m 根网格覆盖整个空间
    for x0 in np.arange(min_x, max_x + ROOT_SIZE, ROOT_SIZE):
        for y0 in np.arange(min_y, max_y + ROOT_SIZE, ROOT_SIZE):
            subset = df[
                (df["world_x"] >= x0)
                & (df["world_x"] < x0 + ROOT_SIZE)
                & (df["world_y"] >= y0)
                & (df["world_y"] < y0 + ROOT_SIZE)
            ]

            if subset.empty:
                continue

            root_id = f"R_{int(x0)}_{int(y0)}"
            leaves.extend(
                build_tree(
                    subset, x0, y0,
                    ROOT_SIZE, path=root_id
                )
            )

    grid = pd.DataFrame(leaves)

    print("\n=== ADAPTIVE GRID SUMMARY ===")
    print("Leaf cells:", len(grid))
    print("\nCells by resolution:")
    print(grid.groupby("size").size().sort_index())

    grid.to_csv(OUTPUT / "adaptive_grid_cells.csv", index=False)

    # 绘制四叉树网格
    fig, ax = plt.subplots(figsize=(11, 9))

    ax.scatter(
        df["world_x"],
        df["world_y"],
        s=2,
        alpha=0.15,
        color="#64748b",
    )

    for row in grid.itertuples():
        ax.add_patch(Rectangle(
            (row.x0, row.y0),
            row.size,
            row.size,
            fill=False,
            edgecolor="#2563eb",
            linewidth=0.65,
            alpha=0.75,
        ))

    ax.set(
        title="Density-Adaptive Quadtree Spatial Grid",
        xlabel="World X (m)",
        ylabel="World Y (m)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.1)
    fig.tight_layout()

    path = FIGURES / "05_adaptive_grid.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
