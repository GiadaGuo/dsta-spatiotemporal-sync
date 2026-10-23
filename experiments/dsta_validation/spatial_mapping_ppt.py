"""PPT-first redraw of the existing DSTA spatial-grid comparison.

This script reuses frozen split data, existing grid generators with the already
selected parameters, and optimized_grid_cells.csv. It does not optimize grids.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsta-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dsta.grid_experiment import ROI, density_cells, fixed_cells, load_split, load_train_features

OUTPUT = ROOT / "outputs/ppt_case/spatial_mapping"
COMPARISON = ROOT / "outputs/reports/grid_comparison.csv"
OPTIMIZED = ROOT / "data/processed/optimized_grid_cells.csv"
EVENTS = ROOT / "outputs/reports/detected_events_v2.csv"
EVENT_ID = "EVT2_006"
BUDGET = 200
COLORS = {"blue": "#38BDF8", "orange": "#FB923C", "purple": "#C084FC",
          "ink": "#EDF4FF", "muted": "#A8B8CC", "grid": "#46617E",
          "bg": "#080F1F", "panel": "#101C30"}


def grid_segments(cells: pd.DataFrame):
    segments = []
    for row in cells.itertuples(index=False):
        x0, y0, size = float(row.x0), float(row.y0), float(row.size)
        segments.extend([[(x0, y0), (x0 + size, y0)],
                         [(x0 + size, y0), (x0 + size, y0 + size)],
                         [(x0 + size, y0 + size), (x0, y0 + size)],
                         [(x0, y0 + size), (x0, y0)]])
    return segments


def draw_grid(ax, cells, points, event, title, subtitle, color):
    ax.set_facecolor(COLORS["panel"])
    ax.scatter(points.world_x, points.world_y, s=3, color=COLORS["muted"],
               alpha=.22, linewidths=0, rasterized=True, label="Held-out test observations")
    ax.add_collection(LineCollection(grid_segments(cells), colors=color,
                                     linewidths=.42, alpha=.75))
    cx, cy = float(event.centroid_x), float(event.centroid_y)
    ax.scatter([cx], [cy], s=85, marker="o", facecolors="none",
               edgecolors=COLORS["orange"], linewidths=2.2, zorder=6)
    ax.scatter([cx], [cy], s=22, marker="x", color=COLORS["orange"],
               linewidths=1.8, zorder=7)
    ax.annotate("EVT2_006 centroid\n(real world coordinate)", (cx, cy),
                xytext=(12, 13), textcoords="offset points", fontsize=7.5,
                color=COLORS["orange"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=COLORS["orange"], lw=.8))
    xmin, ymin, xmax, ymax = ROI
    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
           xlabel="World X (m)", ylabel="World Y (m)")
    ax.set_title(title, pad=29, color=COLORS["ink"], fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.text(.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=COLORS["muted"])
    ax.grid(color="#263C56", alpha=.35, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
                         "axes.titlecolor": COLORS["ink"], "xtick.color": COLORS["muted"],
                         "ytick.color": COLORS["muted"], "axes.edgecolor": COLORS["grid"],
                         "savefig.facecolor": COLORS["bg"]})
    comparison = pd.read_csv(COMPARISON)
    train = load_train_features(ROOT)
    test = load_split(ROOT, "test")
    event = pd.read_csv(EVENTS).query("event_id == @EVENT_ID").iloc[0]

    grids = {
        "Fixed 32 m": fixed_cells(32.0),
        "Density quadtree": density_cells(train, threshold=30),
        "Quality-aware": pd.read_csv(OPTIMIZED),
    }
    expected = {"Fixed 32 m": 144, "Density quadtree": 126, "Quality-aware": 186}
    for name, cells in grids.items():
        if len(cells) != expected[name] or len(cells) > BUDGET:
            raise RuntimeError(f"Unexpected grid definition for {name}: {len(cells)} cells")

    report_names = {"Fixed Grid": "Fixed 32 m", "Density Quadtree": "Density quadtree",
                    "Quality + Optuna": "Quality-aware"}
    summary = comparison.copy()
    summary["display_name"] = summary.method.map(report_names)
    summary["budget_limit_cells"] = BUDGET
    summary["event_id_highlighted"] = EVENT_ID
    summary["event_centroid_x"] = float(event.centroid_x)
    summary["event_centroid_y"] = float(event.centroid_y)
    summary.to_csv(OUTPUT / "spatial_mapping_metrics.csv", index=False)

    fig = plt.figure(figsize=(13.333, 7.5), facecolor=COLORS["bg"])
    gs = fig.add_gridspec(2, 3, height_ratios=[1, .42])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["purple"]]
    methods = ["Fixed 32 m", "Density quadtree", "Quality-aware"]
    error_by_name = dict(zip(summary.display_name, summary.mean_quantization_error_m))
    for ax, name, color in zip(axes, methods, colors):
        draw_grid(ax, grids[name], test, event, name,
                  f"{len(grids[name])} cells / budget ≤ {BUDGET}  ·  test mean error {error_by_name[name]:.2f} m",
                  color)

    bar_ax = fig.add_subplot(gs[1, :])
    values = [error_by_name[name] for name in methods]
    bars = bar_ax.barh(methods[::-1], values[::-1], color=colors[::-1], height=.52)
    for bar, value in zip(bars, values[::-1]):
        bar_ax.text(value + .18, bar.get_y() + bar.get_height()/2, f"{value:.2f} m",
                    va="center", color=COLORS["ink"], fontweight="bold")
    bar_ax.set(title="Held-out test mean spatial quantization error (lower is better)",
               xlabel="Mean distance from observation to assigned cell center (m)", xlim=(0, 15.2))
    bar_ax.grid(axis="x", color="#263C56", alpha=.45)
    bar_ax.spines[["top", "right", "left"]].set_visible(False)
    bar_ax.set_facecolor(COLORS["panel"])
    fig.subplots_adjust(left=.065, right=.985, top=.82, bottom=.10, hspace=.42, wspace=.18)
    fig.suptitle("Unified spatial mapping · same ROI, test split and grid budget",
                 x=.025, y=.965, ha="left", fontsize=20, fontweight="bold", color=COLORS["ink"])
    fig.text(.025, .915,
             "Real DAIR world coordinates  |  validation-selected configurations  |  EVT2_006 marks location only, not an impact area",
             color=COLORS["muted"], fontsize=10)
    fig.text(.025, .025,
             "All methods cover 1,235/1,235 held-out observations (100%). Grid budget ≤ 200 leaf cells; no re-optimization performed.",
             color=COLORS["muted"], fontsize=9)
    for suffix in ("png", "svg"):
        fig.savefig((OUTPUT / "spatial_mapping_ppt").with_suffix("." + suffix),
                    dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)

    config = {
        "event_id": EVENT_ID, "roi": list(map(float, ROI)), "grid_budget_cells": BUDGET,
        "selection_split": "validation", "reported_error_split": "held-out test",
        "test_observations": int(summary.n_observations.iloc[0]),
        "coverage_rate_all_methods": float(summary.coverage_rate.min()),
        "methods": summary[["method", "configuration", "n_cells",
                            "mean_quantization_error_m", "p90_quantization_error_m"]].to_dict("records"),
        "event_marker": {"x": float(event.centroid_x), "y": float(event.centroid_y),
                         "meaning": "event member-state centroid; not a road or impact-area label"},
        "provenance": [str(COMPARISON.relative_to(ROOT)), str(OPTIMIZED.relative_to(ROOT)),
                       "data/processed/splits/train.parquet", "data/processed/splits/test.parquet"],
    }
    (OUTPUT / "spatial_mapping_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary[["display_name", "configuration", "n_cells", "coverage_rate",
                   "mean_quantization_error_m", "p90_quantization_error_m"]].to_string(index=False))
    print("Saved to", OUTPUT)


if __name__ == "__main__":
    main()
