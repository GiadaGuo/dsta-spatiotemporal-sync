
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dsta.grid_evaluation import assign_points_to_cells

TRACKS_PATH = ROOT / "data/processed/world_tracks.parquet"
GRID_PATH = ROOT / "data/processed/optimized_grid_cells.csv"
OUTPUT_PATH = ROOT / "data/processed/spatiotemporal_events.parquet"


def main():
    tracks = pd.read_parquet(TRACKS_PATH)
    cells = pd.read_csv(GRID_PATH)

    print("=== INPUT ===")
    print("Tracks:", len(tracks))
    print("Grid cells:", len(cells))
    print("Track columns:", tracks.columns.tolist())
    print("Grid columns:", cells.columns.tolist())

    # 为避免因接口差异导致静默错误，先检查字段
    required = [
        "source", "sequence_id", "frame_id",
        "track_id", "timestamp_us", "world_x", "world_y"
    ]
    missing = [c for c in required if c not in tracks.columns]
    if missing:
        raise ValueError(f"Missing track columns: {missing}")

    if tracks["timestamp_us"].isna().any():
        raise ValueError("Missing timestamps")

    if tracks[["world_x", "world_y"]].isna().any().any():
        raise ValueError("Missing world coordinates")

    # 使用之前实验中的同一个网格分配函数
    assigned = assign_points_to_cells(tracks, cells)

    print("Assigned columns:", assigned.columns.tolist())

    # 兼容不同版本的网格分配函数：
    # 若返回数组/Series，则将其作为网格索引；
    # 若返回 DataFrame，则直接使用其输出。
    if isinstance(assigned, pd.DataFrame):
        events = assigned.copy()
    else:
        events = tracks.copy()
        events["grid_id"] = assigned

    if len(events) != len(tracks):
        raise ValueError("Grid assignment changed observation count")

    if "grid_id" not in events.columns:
        raise ValueError(
            "Grid assignment did not produce grid_id. "
            "Inspect assign_points_to_cells output columns."
        )

    if events["grid_id"].isna().any():
        raise ValueError("Some observations were not assigned to a grid")

    events = events.reset_index(drop=True)

    # 原始观测的唯一 ID，不依赖跨源目标关联
    events["event_id"] = [
        f"obs_{i:06d}" for i in range(len(events))
    ]

    # 后续用于模拟网络扰动；当前不注入延迟
    events["arrival_timestamp_us"] = events["timestamp_us"]

    events["is_missing"] = False
    events["is_out_of_order"] = False
    events["is_interpolated"] = False

    # 标注数据来源，不误称传感器融合真值
    events["coordinate_provenance"] = (
        "calibrated_lidar_annotation"
    )

    events = events.sort_values(
        ["timestamp_us", "source", "frame_id", "track_id"],
        kind="stable"
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OUTPUT_PATH, index=False)

    print("\n=== OUTPUT ===")
    print("Observations:", len(events))
    print("Unique event IDs:", events["event_id"].nunique())
    print("Grid coverage:", events["grid_id"].notna().mean())
    print("Sources:")
    print(events["source"].value_counts())
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
