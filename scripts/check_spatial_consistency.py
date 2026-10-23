
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

DATA = Path("data/processed")
OUTPUT = Path("outputs/reports")
OUTPUT.mkdir(parents=True, exist_ok=True)

MAX_DISTANCE = 5.0  # 米，初步候选匹配阈值


def main():
    tracks = pd.read_parquet(DATA / "world_tracks.parquet")
    pairs = pd.read_parquet(DATA / "frame_pairs.parquet")

    results = []

    for _, pair in pairs.iterrows():
        veh = tracks[
            (tracks["source"] == "vehicle-side")
            & (tracks["frame_id"] == pair["vehicle_frame"])
        ].reset_index(drop=True)

        inf = tracks[
            (tracks["source"] == "infrastructure-side")
            & (tracks["frame_id"] == pair["infrastructure_frame"])
        ].reset_index(drop=True)

        if veh.empty or inf.empty:
            continue

        # 二维空间距离
        vxy = veh[["world_x", "world_y"]].to_numpy()
        ixy = inf[["world_x", "world_y"]].to_numpy()

        distances = np.linalg.norm(
            vxy[:, None, :] - ixy[None, :, :],
            axis=2,
        )

        # 不同类别禁止匹配
        same_type = (
            veh["object_type"].to_numpy()[:, None]
            == inf["object_type"].to_numpy()[None, :]
        )

        cost = np.where(same_type, distances, 1e9)

        # 匈牙利算法：同一帧内一对一匹配
        rows, cols = linear_sum_assignment(cost)

        for r, c in zip(rows, cols):
            if not same_type[r, c]:
                continue

            dist = distances[r, c]
            if dist > MAX_DISTANCE:
                continue

            results.append({
                "vehicle_frame": pair["vehicle_frame"],
                "infrastructure_frame": pair["infrastructure_frame"],
                "vehicle_track_id": veh.iloc[r]["track_id"],
                "infrastructure_track_id": inf.iloc[c]["track_id"],
                "object_type": veh.iloc[r]["object_type"],
                "distance_m": dist,
                "dx_m": (
                    veh.iloc[r]["world_x"]
                    - inf.iloc[c]["world_x"]
                ),
                "dy_m": (
                    veh.iloc[r]["world_y"]
                    - inf.iloc[c]["world_y"]
                ),
                "lidar_delta_ms": pair["lidar_delta_ms"],
            })

    result = pd.DataFrame(results)

    if result.empty:
        print("No candidate matches found.")
        return

    output = OUTPUT / "spatial_matches.csv"
    result.to_csv(output, index=False)

    print("\n=== SPATIAL CONSISTENCY ===")
    print("Candidate matches:", len(result))
    print("Mean distance:", round(result["distance_m"].mean(), 3))
    print("Median distance:", round(result["distance_m"].median(), 3))
    print("P90 distance:", round(result["distance_m"].quantile(0.9), 3))
    print("Mean dx:", round(result["dx_m"].mean(), 3))
    print("Mean dy:", round(result["dy_m"].mean(), 3))
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
