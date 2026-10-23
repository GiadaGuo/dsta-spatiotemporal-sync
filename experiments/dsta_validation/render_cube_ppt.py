"""Render a 300-DPI static DSTA cube figure from the delivered cube tables."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsta-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ppt_case/cube"
BG, PANEL, TEXT, MUTED = "#080F1F", "#101C30", "#F1F5F9", "#A8B8CC"
BLUE, ORANGE, PURPLE = "#38BDF8", "#FB923C", "#C084FC"


def main():
    cube = pd.read_parquet(OUT / "unified_cube_states.parquet")
    grid = pd.read_csv(ROOT / "data/processed/optimized_grid_cells.csv")
    event = pd.read_csv(ROOT / "outputs/reports/detected_events_v2.csv").query("event_id=='EVT2_006'").iloc[0]
    span = 80.0
    xr = [event.centroid_x-span/2, event.centroid_x+span/2]
    yr = [event.centroid_y-span/2, event.centroid_y+span/2]
    view = cube[cube.mean_world_x.between(*xr) & cube.mean_world_y.between(*yr)].copy()
    local_grid = grid[(grid.x0+grid["size"] >= xr[0]) & (grid.x0 <= xr[1]) &
                      (grid.y0+grid["size"] >= yr[0]) & (grid.y0 <= yr[1])]
    event_states = view[view.event_id.eq("EVT2_006")].sort_values("event_time_s")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": TEXT, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "savefig.facecolor": BG})
    fig = plt.figure(figsize=(13.333, 7.5), facecolor=BG)
    ax = fig.add_axes([.04, .08, .68, .79], projection="3d", facecolor=BG)
    info = fig.add_axes([.74, .10, .235, .76], facecolor=PANEL)
    info.set_xticks([]); info.set_yticks([])
    for spine in info.spines.values(): spine.set_color("#334A66")

    norm = Normalize(vmin=float(view.observation_count.min()), vmax=float(view.observation_count.max()))
    colors = cm.turbo(norm(view.observation_count))
    sizes = 10 + 38*np.sqrt(view.observation_count/view.observation_count.max())
    ax.scatter(view.mean_world_x, view.mean_world_y, view.event_time_s, c=colors, s=sizes,
               alpha=.48, edgecolors="none", depthshade=False)
    for row in local_grid.itertuples(index=False):
        corners = [(row.x0,row.y0),(row.x0+row.size,row.y0),(row.x0+row.size,row.y0+row.size),
                   (row.x0,row.y0+row.size),(row.x0,row.y0)]
        ax.plot([p[0] for p in corners], [p[1] for p in corners], zs=0, color=BLUE, alpha=.24, lw=.65)
    ax.plot(event_states.mean_world_x, event_states.mean_world_y, event_states.event_time_s,
            color=PURPLE, lw=3.2, alpha=.95)
    ax.scatter(event_states.mean_world_x, event_states.mean_world_y, event_states.event_time_s,
               marker="D", s=62, color=PURPLE, edgecolors=TEXT, linewidths=.8,
               depthshade=False, label="EVT2_006 member state")
    xx, yy = np.meshgrid(xr, yr)
    ax.plot_surface(xx, yy, np.full_like(xx, 8.0), color=ORANGE, alpha=.09, shade=False)
    ax.set(xlim=xr, ylim=yr, zlim=(0,24), xlabel="World X (m)", ylabel="World Y (m)",
           zlabel="T (s)")
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect((1,1,.72))
    ax.xaxis.pane.set_facecolor(PANEL); ax.yaxis.pane.set_facecolor(PANEL); ax.zaxis.pane.set_facecolor(PANEL)
    ax.xaxis.pane.set_alpha(.52); ax.yaxis.pane.set_alpha(.52); ax.zaxis.pane.set_alpha(.52)
    ax.grid(True, alpha=.2)
    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT)

    sm = cm.ScalarMappable(norm=norm, cmap="turbo")
    cb = fig.colorbar(sm, ax=ax, fraction=.025, pad=.03, shrink=.66)
    cb.set_label("Observation count per grid-window state", color=TEXT)
    cb.ax.tick_params(colors=MUTED)

    info.text(.07,.94,"CUBE DEFINITION",fontsize=13,fontweight="bold",color=TEXT)
    lines = [
        ("X / Y", "DAIR real-world coordinates"),
        ("T", "Relative event time, not height"),
        ("Base", "Quality-aware adaptive grid"),
        ("Voxel", "One grid × one 2 s window"),
        ("Color / size", "Observation count"),
        ("Purple", "EVT2_006 downstream evidence"),
    ]
    y=.84
    for key,value in lines:
        info.text(.07,y,key,fontsize=10,fontweight="bold",color=BLUE)
        info.text(.07,y-.045,value,fontsize=9,color=MUTED)
        y-=.115
    info.text(.07,.17,"DISPLAY SCOPE",fontsize=11,fontweight="bold",color=TEXT)
    info.text(.07,.12,f"80 m local context · {len(view)} states",fontsize=9,color=MUTED)
    info.text(.07,.075,"All 830 states retained in data delivery",fontsize=9,color=MUTED)
    info.text(.07,.03,"No spatial coordinate or sensor source is synthetic",fontsize=8.2,color=ORANGE)

    fig.suptitle("DSTA unified spatiotemporal data cube · EVT2_006 validation context",
                 x=.035,y=.965,ha="left",fontsize=20,fontweight="bold")
    fig.text(.035,.915,"Real grid-window states across 13 time slices  |  orange plane highlights window 4 (T=8 s)",
             fontsize=10,color=MUTED)
    for suffix in ("png","svg"):
        fig.savefig((OUT/"dsta_cube_ppt").with_suffix("."+suffix), dpi=300 if suffix=="png" else None,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Rendered {len(view)} local states from {len(cube)} complete cube states")


if __name__ == "__main__":
    main()
