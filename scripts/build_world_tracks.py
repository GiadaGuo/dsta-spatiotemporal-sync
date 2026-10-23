
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path("data/raw/V2X-Seq-SPD-Example")
OUTPUT = Path("data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def transform_point(point, calibration):
    """应用刚体变换 P' = R @ P + t。"""
    rotation = np.asarray(
        calibration["rotation"], dtype=float
    ).reshape(3, 3)

    translation = np.asarray(
        calibration["translation"], dtype=float
    ).reshape(3)

    return rotation @ point + translation


def vehicle_to_world(point, frame_id):
    base = ROOT / "vehicle-side" / "calib"

    lidar_to_novatel = read_json(
        base / "lidar_to_novatel" / f"{frame_id}.json"
    )["transform"]

    novatel_to_world = read_json(
        base / "novatel_to_world" / f"{frame_id}.json"
    )

    point_novatel = transform_point(
        point, lidar_to_novatel
    )

    return transform_point(
        point_novatel, novatel_to_world
    )


def infrastructure_to_world(point, frame_id):
    calibration = read_json(
        ROOT
        / "infrastructure-side"
        / "calib"
        / "virtuallidar_to_world"
        / f"{frame_id}.json"
    )

    return transform_point(point, calibration)


def extract_side(side):
    base = ROOT / side
    records = read_json(base / "data_info.json")

    rows = []

    for frame in records:
        frame_id = frame["frame_id"]
        timestamp = int(frame["pointcloud_timestamp"])

        label_path = base / frame["label_lidar_std_path"]
        labels = read_json(label_path)

        for obj in labels:
            loc = obj["3d_location"]

            point = np.array(
                [loc["x"], loc["y"], loc["z"]],
                dtype=float
            )

            if side == "vehicle-side":
                world = vehicle_to_world(point, frame_id)
            else:
                world = infrastructure_to_world(
                    point, frame_id
                )

            rows.append({
                "source": side,
                "sequence_id": frame["sequence_id"],
                "frame_id": frame_id,
                "timestamp_us": timestamp,
                "track_id": str(obj["track_id"]),
                "object_type": obj["type"],
                "world_x": world[0],
                "world_y": world[1],
                "world_z": world[2],
            })

    return pd.DataFrame(rows)


def main():
    vehicle = extract_side("vehicle-side")
    infra = extract_side("infrastructure-side")

    tracks = pd.concat(
        [vehicle, infra],
        ignore_index=True
    )

    tracks = tracks.sort_values(
        ["source", "track_id", "timestamp_us"]
    ).reset_index(drop=True)

    output = OUTPUT / "world_tracks.parquet"
    tracks.to_parquet(output, index=False)

    print("\n=== WORLD TRACK SUMMARY ===")
    print("Total observations:", len(tracks))

    print("\nObservations by source:")
    print(tracks.groupby("source").size())

    print("\nUnique tracks by source:")
    print(
        tracks.groupby("source")["track_id"].nunique()
    )

    print("\nWorld coordinate ranges:")
    print(
        tracks[
            ["world_x", "world_y", "world_z"]
        ].describe().round(3)
    )

    print("\nFirst 5 observations:")
    print(tracks.head().to_string(index=False))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
