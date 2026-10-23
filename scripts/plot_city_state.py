
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data/processed/city_state.parquet"
FIGURES = ROOT / "outputs/figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def plot_heatmap(state):
    # 累计各空间网格的观测活跃度
    spatial = (
        state.groupby(["grid_id", "x0", "y0", "size"])
        ["observation_count"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = max(2, int(spatial["observation_count"].max()))
    norm = LogNorm(vmin=1, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    for row in spatial.itertuples():
        color = cmap(norm(row.observation_count))
        ax.add_patch(
            Rectangle(
                (row.x0, row.y0),
                row.size,
                row.size,
                facecolor=color,
                edgecolor="white",
                linewidth=0.3,
            )
        )

    ax.set_xlim(3584, 3968)
    ax.set_ylim(2112, 2496)
    ax.set_aspect("equal")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_title("Traffic Observation Activity by Adaptive Grid")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(
        sm, ax=ax, label="Observation count (log scale)"
    )

    fig.tight_layout()
    path = FIGURES / "08_city_state_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print("Saved:", path)


def plot_timeline(state):
    temporal = (
        state.groupby("window_id")
        .agg(
            observations=("observation_count", "sum"),
            mean_speed=("mean_speed_mps", "mean"),
        )
        .reset_index()
    )

    # 以第一个时间窗口为 0 秒
    x = temporal["window_id"].to_numpy() * 2

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(
        x,
        temporal["observations"],
        color="#2563eb",
        marker="o",
        linewidth=2,
        label="Observations",
    )
    ax1.set_xlabel("Time from start (s)")
    ax1.set_ylabel("Observation count", color="#2563eb")

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        temporal["mean_speed"],
        color="#ea580c",
        marker="s",
        linewidth=2,
        label="Mean estimated speed",
    )
    ax2.set_ylabel("Estimated speed (m/s)", color="#ea580c")

    ax1.set_title("Traffic State Evolution")
    ax1.grid(alpha=0.2)

    fig.tight_layout()
    path = FIGURES / "09_city_state_timeline.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print("Saved:", path)


if __name__ == "__main__":
    state = pd.read_parquet(STATE)
    plot_heatmap(state)
    plot_timeline(state)
