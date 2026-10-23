"""Build observation-level and grid-window-level DSTA cube views."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def quality_flag_text(frame: pd.DataFrame) -> pd.Series:
    labels = []
    for row in frame[["is_missing", "is_out_of_order", "is_interpolated"]].itertuples(index=False):
        flags = []
        if bool(row.is_missing): flags.append("missing")
        if bool(row.is_out_of_order): flags.append("out_of_order")
        if bool(row.is_interpolated): flags.append("interpolated")
        labels.append("|".join(flags) if flags else "observed_ok")
    return pd.Series(labels, index=frame.index, dtype="string")


def build_cube_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc, report = root / "data/processed", root / "outputs/reports"
    obs = pd.read_parquet(proc / "spatiotemporal_events.parquet").copy()
    states = pd.read_parquet(proc / "city_state.parquet").copy()
    anomaly = pd.read_parquet(proc / "embedding_anomaly_states.parquet")
    members = pd.read_csv(report / "event_members_v2.csv")

    starts = states[["window_id", "window_start_us"]].drop_duplicates().sort_values("window_start_us")
    start_us = starts.window_start_us.to_numpy(dtype="int64")
    start_window = starts.window_id.to_numpy(dtype="int64")
    positions = np.searchsorted(start_us, obs.timestamp_us.to_numpy(dtype="int64"), side="right") - 1
    if (positions < 0).any():
        raise ValueError("Observation precedes the first city-state window")
    obs["window_id"] = start_window[positions]
    t0 = int(start_us.min())
    obs["event_time_us"] = obs.timestamp_us.astype("int64")
    obs["event_time_s"] = (obs.event_time_us - t0) / 1_000_000
    obs["observation_count"] = 1
    obs["quality_flags"] = quality_flag_text(obs)
    obs["provenance"] = obs.coordinate_provenance.astype("string")
    obs["record_granularity"] = "observation"

    quality = (obs.groupby(["window_id", "grid_id"], as_index=False)
        .agg(missing_count=("is_missing", "sum"),
             out_of_order_count=("is_out_of_order", "sum"),
             interpolated_count=("is_interpolated", "sum"),
             provenance=("provenance", lambda x: "|".join(sorted(set(map(str, x)))))))
    flags = []
    for row in quality.itertuples(index=False):
        parts = []
        if row.missing_count: parts.append(f"missing:{row.missing_count}")
        if row.out_of_order_count: parts.append(f"out_of_order:{row.out_of_order_count}")
        if row.interpolated_count: parts.append(f"interpolated:{row.interpolated_count}")
        flags.append("|".join(parts) if parts else "observed_ok")
    quality["quality_flags"] = flags

    anomaly_cols = ["window_id", "grid_id", "raw_is_anomaly", "ae_is_anomaly",
                    "either_anomaly", "both_anomaly", "candidate_kind"]
    cube = states.merge(anomaly[anomaly_cols], on=["window_id", "grid_id"],
                        how="left", validate="one_to_one")
    cube = cube.merge(quality, on=["window_id", "grid_id"], how="left", validate="one_to_one")
    event_members = members[["event_id", "window_id", "grid_id"]].copy()
    event_members["is_event_member"] = True
    cube = cube.merge(event_members, on=["window_id", "grid_id"], how="left", validate="one_to_many")
    # A state can belong to at most one v2 component in the current result.
    if cube.duplicated(["window_id", "grid_id"]).any():
        raise ValueError("A cube state belongs to multiple event components")
    cube["is_event_member"] = cube.is_event_member.fillna(False).astype(bool)
    cube["event_id"] = cube.event_id.fillna("").astype("string")
    cube["event_time_us"] = cube.window_start_us.astype("int64")
    cube["event_time_s"] = (cube.window_start_us - t0) / 1_000_000
    cube["source"] = "vehicle+infrastructure aggregate"
    cube["source_coverage_count"] = ((cube.vehicle_observations > 0).astype(int)
                                     + (cube.infrastructure_observations > 0).astype(int))
    cube["quality_flags"] = cube.quality_flags.fillna("no_observation_quality_record")
    cube["provenance"] = cube.provenance.fillna("city_state_aggregation")
    cube["record_granularity"] = "grid_window_state"
    return obs, cube


def export_cube_tables(root: Path, output: Path) -> tuple[Path, Path]:
    obs, cube = build_cube_tables(root)
    output.mkdir(parents=True, exist_ok=True)
    obs_path = output / "unified_observations.parquet"
    cube_path = output / "unified_cube_states.parquet"
    obs.to_parquet(obs_path, index=False)
    cube.to_parquet(cube_path, index=False)
    return obs_path, cube_path
