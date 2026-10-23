'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-10-22 16:30:36
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-10-22 16:30:57
FilePath: /dsta-spatiotemporal-sync/src/dsta/grid_features.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import pandas as pd


def prepare_features(
    tracks: pd.DataFrame,
    spatial_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    构造四叉树所需的逐观测特征。

    density: 后续在每个网格中聚合
    motion: 相邻观测速度变化的代理指标
    disagreement: 跨源候选匹配残差
    """
    df = tracks.copy()

    df = df.sort_values(
        ["source", "sequence_id", "track_id", "timestamp_us"]
    ).reset_index(drop=True)

    keys = ["source", "sequence_id", "track_id"]
    grouped = df.groupby(keys, sort=False)

    dt = grouped["timestamp_us"].diff() / 1e6
    dx = grouped["world_x"].diff()
    dy = grouped["world_y"].diff()

    speed = np.sqrt(dx**2 + dy**2) / dt.where(dt > 0)

    # 防止异常短时间间隔导致速度爆炸
    df["speed_mps"] = speed.replace(
        [np.inf, -np.inf], np.nan
    )

    # 加速度绝对值作为运动变化代理
    dv = df.groupby(keys, sort=False)["speed_mps"].diff()
    df["motion_score"] = (
        (dv / dt.where(dt > 0))
        .abs()
        .replace([np.inf, -np.inf], np.nan)
    )

    # 使用官方关联帧下的候选空间匹配残差
    veh = spatial_matches[
        [
            "vehicle_frame",
            "vehicle_track_id",
            "distance_m",
        ]
    ].rename(columns={
        "vehicle_frame": "frame_id",
        "vehicle_track_id": "track_id",
    })
    veh["source"] = "vehicle-side"

    inf = spatial_matches[
        [
            "infrastructure_frame",
            "infrastructure_track_id",
            "distance_m",
        ]
    ].rename(columns={
        "infrastructure_frame": "frame_id",
        "infrastructure_track_id": "track_id",
    })
    inf["source"] = "infrastructure-side"

    residuals = pd.concat([veh, inf], ignore_index=True)

    # 同一观测如果有多条候选匹配，取最小残差
    residuals = (
        residuals.groupby(
            ["source", "frame_id", "track_id"],
            as_index=False,
        )["distance_m"].min()
    )

    df = df.merge(
        residuals,
        on=["source", "frame_id", "track_id"],
        how="left",
        validate="many_to_one",
    )

    df["disagreement_score"] = df["distance_m"]

    # 不在这里用全局均值填补缺失
    # 避免将无跨源匹配的观测误认为一致
    return df
