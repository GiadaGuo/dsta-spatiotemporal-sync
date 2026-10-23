'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-10-22 15:21:50
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-10-22 15:21:58
FilePath: /dsta-spatiotemporal-sync/scripts/plot_world_tracks.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT = Path("data/processed/world_tracks.parquet")
OUTPUT = Path("outputs/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

fig, ax = plt.subplots(figsize=(10, 8))

colors = {
    "vehicle-side": "#2563eb",
    "infrastructure-side": "#d97706",
}

for source, group in df.groupby("source"):
    color = colors[source]

    # 只画至少出现 5 帧的轨迹
    for track_id, track in group.groupby("track_id"):
        if len(track) < 5:
            continue

        track = track.sort_values("timestamp_us")

        ax.plot(
            track["world_x"],
            track["world_y"],
            color=color,
            linewidth=1.1,
            alpha=0.5,
        )

    # 用代理图例避免重复标签
    ax.plot([], [], color=color, label=source)

ax.set(
    title="Vehicle and Infrastructure Tracks in World Coordinates",
    xlabel="World X (m)",
    ylabel="World Y (m)",
)

ax.set_aspect("equal", adjustable="box")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()
output = OUTPUT / "04_world_tracks.png"
fig.savefig(output, dpi=200)
plt.close(fig)

print("Saved:", output)
