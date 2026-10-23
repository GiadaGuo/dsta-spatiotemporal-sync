
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT = Path("data/processed/frame_pairs.parquet")
OUTPUT = Path("outputs/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

# 使用英文标签，避免系统缺少中文字体
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

df = pd.read_parquet(INPUT).sort_values(
    "veh_image_ts"
).reset_index(drop=True)

x = np.arange(len(df))

# Figure 01: 时间差变化
fig, ax = plt.subplots(figsize=(11, 4.5))

ax.plot(
    x, df["image_delta_ms"],
    color="#2563eb", linewidth=1.5,
    label="Camera"
)
ax.plot(
    x, df["lidar_delta_ms"],
    color="#d97706", linewidth=1.5,
    label="LiDAR"
)

ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set(
    title="Temporal Offsets Across Cooperative Frame Pairs",
    xlabel="Cooperative Frame Pair Index",
    ylabel="Vehicle - Infrastructure (ms)",
)
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT / "01_temporal_offsets.png")
plt.close(fig)

# Figure 02: 时间差分布
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.hist(
    df["image_delta_ms"], bins=20,
    color="#2563eb", alpha=0.65,
    label="Camera"
)
ax.hist(
    df["lidar_delta_ms"], bins=20,
    color="#d97706", alpha=0.65,
    label="LiDAR"
)

ax.set(
    title="Distribution of Temporal Offsets",
    xlabel="Vehicle - Infrastructure (ms)",
    ylabel="Number of Frame Pairs",
)
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT / "02_offset_distribution.png")
plt.close(fig)

# Figure 03: 采样间隔分布
sources = {
    "Vehicle Camera": ("veh_image_ts", "#2563eb"),
    "Infrastructure Camera": ("inf_image_ts", "#7c3aed"),
    "Vehicle LiDAR": ("veh_lidar_ts", "#d97706"),
    "Infrastructure LiDAR": ("inf_lidar_ts", "#0891b2"),
}

fig, ax = plt.subplots(figsize=(10, 4.5))

for label, (col, color) in sources.items():
    # 去重后排序：官方关联表中的时间戳
    ts = np.sort(df[col].dropna().unique())
    intervals = np.diff(ts) / 1000.0

    ax.hist(
        intervals,
        bins=np.arange(0, 251, 10),
        histtype="step",
        linewidth=1.8,
        color=color,
        label=label,
    )

ax.set(
    title="Sampling Intervals in Matched Frames",
    xlabel="Inter-frame Interval (ms)",
    ylabel="Count",
)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(OUTPUT / "03_sampling_intervals.png")
plt.close(fig)

print("Processed pairs:", len(df))
print("Generated figures:")
for path in sorted(OUTPUT.glob("0*.png")):
    print(" -", path)
