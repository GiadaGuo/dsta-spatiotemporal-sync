
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/spatiotemporal_events.parquet"
OUTPUT = ROOT / "data/processed/city_state.parquet"

WINDOW_US = 2_000_000
MAX_TRACK_GAP_US = 500_000
MAX_SPEED_MPS = 45.0


def main():
    events = pd.read_parquet(INPUT).copy()
    events["timestamp_us"] = events["timestamp_us"].astype("int64")

    # 时间窗口以整个数据集的起始时间为基准
    t0 = int(events["timestamp_us"].min())
    events["window_id"] = (
        (events["timestamp_us"] - t0) // WINDOW_US
    ).astype("int64")
    events["window_start_us"] = (
        t0 + events["window_id"] * WINDOW_US
    )

    # 只在同一来源、序列、轨迹内估计运动
    keys = ["source", "sequence_id", "track_id"]
    events = events.sort_values(
        keys + ["timestamp_us"], kind="stable"
    )

    grouped = events.groupby(keys, dropna=False)
    dt_us = grouped["timestamp_us"].diff()
    dx = grouped["world_x"].diff()
    dy = grouped["world_y"].diff()

    valid_motion = (
        dt_us.gt(0)
        & dt_us.le(MAX_TRACK_GAP_US)
    )

    speed = np.sqrt(dx**2 + dy**2) / (dt_us / 1e6)

    # 去除明显异常的速度估计，不将其当作真实速度
    events["speed_mps"] = speed.where(
        valid_motion & speed.between(0, MAX_SPEED_MPS)
    )

    events["is_vehicle_side"] = (
        events["source"] == "vehicle-side"
    ).astype(int)
    events["is_infrastructure_side"] = (
        events["source"] == "infrastructure-side"
    ).astype(int)

    # 每个空间网格、每个时间窗口生成一个状态
    state = (
        events.groupby(
            ["window_id", "window_start_us", "grid_id"],
            dropna=False
        )
        .agg(
            observation_count=("event_id", "size"),
            unique_local_tracks=("track_id", "nunique"),
            vehicle_observations=("is_vehicle_side", "sum"),
            infrastructure_observations=(
                "is_infrastructure_side", "sum"
            ),
            mean_speed_mps=("speed_mps", "mean"),
            speed_std_mps=("speed_mps", "std"),
            valid_speed_count=("speed_mps", "count"),
            mean_world_x=("world_x", "mean"),
            mean_world_y=("world_y", "mean"),
        )
        .reset_index()
    )

    # 补充真实网格面积，避免大小不同的网格直接比较计数
    cells = pd.read_csv(
        ROOT / "data/processed/optimized_grid_cells.csv"
    )
    cells = cells[["grid_id", "x0", "y0", "size"]].copy()
    cells["grid_area_m2"] = cells["size"] ** 2

    state = state.merge(
        cells, on="grid_id", how="left", validate="many_to_one"
    )

    if state["grid_area_m2"].isna().any():
        raise ValueError("Missing grid metadata")

    state["observation_density"] = (
        state["observation_count"] / state["grid_area_m2"]
    )
    state["vehicle_share"] = (
        state["vehicle_observations"] / state["observation_count"]
    )

    # 只有单来源观测时，不把它直接解释为跨源不一致
    state["both_sources_present"] = (
        (state["vehicle_observations"] > 0)
        & (state["infrastructure_observations"] > 0)
    )

    state = state.sort_values(
        ["window_id", "grid_id"]
    ).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(OUTPUT, index=False)

    print("=== CITY STATE ===")
    print("Input observations:", len(events))
    print("State rows:", len(state))
    print("Time windows:", state["window_id"].nunique())
    print("Occupied grids:", state["grid_id"].nunique())
    print("Valid speed observations:",
          events["speed_mps"].notna().sum())
    print("Both-source state ratio:",
          round(state["both_sources_present"].mean(), 4))
    print("Mean speed (m/s):",
          round(state["mean_speed_mps"].mean(), 4))
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
