
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path("data/raw/V2X-Seq-SPD-Example")
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_frames(side):
    """读取车端或路侧元数据，保留原始时间戳。"""
    records = load_json(ROOT / side / "data_info.json")
    df = pd.DataFrame(records)

    prefix = "veh" if side == "vehicle-side" else "inf"

    df = df[
        [
            "frame_id",
            "sequence_id",
            "image_timestamp",
            "pointcloud_timestamp",
        ]
    ].copy()

    df = df.rename(
        columns={
            "frame_id": f"{prefix}_frame",
            "sequence_id": f"{prefix}_sequence",
            "image_timestamp": f"{prefix}_image_ts",
            "pointcloud_timestamp": f"{prefix}_lidar_ts",
        }
    )

    # 使用 int64 保存原始整数，避免浮点精度损失
    for col in [f"{prefix}_image_ts", f"{prefix}_lidar_ts"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")

    return df


def summarize_intervals(df, col):
    """检查连续帧的采样间隔，单位暂按微秒换算。"""
    ts = np.sort(df[col].to_numpy())
    intervals_ms = np.diff(ts) / 1000.0

    print(f"\n{col}")
    print(f"  median interval: {np.median(intervals_ms):.3f} ms")
    print(f"  min interval:    {np.min(intervals_ms):.3f} ms")
    print(f"  max interval:    {np.max(intervals_ms):.3f} ms")


def main():
    vehicle = load_frames("vehicle-side")
    infra = load_frames("infrastructure-side")

    coop = pd.DataFrame(
        load_json(ROOT / "cooperative/data_info.json")
    )

    coop["delta_x"] = coop["system_error_offset"].apply(
        lambda x: x.get("delta_x")
    )
    coop["delta_y"] = coop["system_error_offset"].apply(
        lambda x: x.get("delta_y")
    )

    coop = coop.drop(columns=["system_error_offset"])

    # 校验关联键唯一性，防止意外多对多 Join
    assert not vehicle["veh_frame"].duplicated().any()
    assert not infra["inf_frame"].duplicated().any()

    pairs = coop.merge(
        vehicle,
        left_on=["vehicle_frame", "vehicle_sequence"],
        right_on=["veh_frame", "veh_sequence"],
        how="left",
        validate="many_to_one",
        indicator="_vehicle_match",
    )

    pairs = pairs.merge(
        infra,
        left_on=["infrastructure_frame", "infrastructure_sequence"],
        right_on=["inf_frame", "inf_sequence"],
        how="left",
        validate="many_to_one",
        indicator="_infra_match",
    )

    unmatched_vehicle = (
        pairs["_vehicle_match"] != "both"
    ).sum()
    unmatched_infra = (
        pairs["_infra_match"] != "both"
    ).sum()

    print("Unmatched vehicle:", unmatched_vehicle)
    print("Unmatched infrastructure:", unmatched_infra)

    assert unmatched_vehicle == 0
    assert unmatched_infra == 0

    pairs = pairs.drop(
        columns=["_vehicle_match", "_infra_match"]
    )

    # 时间戳差：微秒 -> 毫秒
    pairs["image_delta_ms"] = (
        pairs["veh_image_ts"] - pairs["inf_image_ts"]
    ) / 1000.0

    pairs["lidar_delta_ms"] = (
        pairs["veh_lidar_ts"] - pairs["inf_lidar_ts"]
    ) / 1000.0

    pairs = pairs.sort_values("veh_image_ts").reset_index(
        drop=True
    )

    output_path = OUTPUT / "frame_pairs.parquet"
    pairs.to_parquet(output_path, index=False)

    print("\n=== FRAME PAIR SUMMARY ===")
    print("Vehicle frames:", len(vehicle))
    print("Infrastructure frames:", len(infra))
    print("Official pairs:", len(coop))
    print("Joined pairs:", len(pairs))

    summarize_intervals(vehicle, "veh_image_ts")
    summarize_intervals(infra, "inf_image_ts")
    summarize_intervals(vehicle, "veh_lidar_ts")
    summarize_intervals(infra, "inf_lidar_ts")

    print("\n=== TEMPORAL OFFSETS (ms) ===")
    print(
        pairs[["image_delta_ms", "lidar_delta_ms"]]
        .describe()
        .round(3)
    )

    print("\n=== FIRST 5 PAIRS ===")
    print(
        pairs[
            [
                "vehicle_frame",
                "infrastructure_frame",
                "image_delta_ms",
                "lidar_delta_ms",
                "delta_x",
                "delta_y",
            ]
        ].head().to_string(index=False)
    )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
