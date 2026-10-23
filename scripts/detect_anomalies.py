'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-10-23 09:46:51
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-10-23 09:46:59
FilePath: /dsta-spatiotemporal-sync/scripts/detect_anomalies.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/city_state.parquet"
OUTPUT = ROOT / "data/processed/anomaly_states.parquet"

FEATURES = [
    "observation_density",
    "mean_speed_mps",
    "speed_std_mps",
    "vehicle_share",
]

CONTAMINATION = 0.08
RANDOM_STATE = 42


def main():
    state = pd.read_parquet(INPUT).copy()

    # 排除首尾窗口，减少截断窗口引起的异常
    min_window = state["window_id"].min()
    max_window = state["window_id"].max()

    eligible = (
        (state["window_id"] > min_window)
        & (state["window_id"] < max_window)
    )

    # 对于没有有效速度的状态，保留缺失指示
    X = state.loc[eligible, FEATURES].copy()
    X["speed_missing"] = X["mean_speed_mps"].isna().astype(int)

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # 分数越大，越异常
    anomaly_score = -model.score_samples(X_scaled)
    prediction = model.predict(X_scaled)

    state["anomaly_score"] = np.nan
    state["is_anomaly"] = False
    state["eligible_for_detection"] = eligible

    state.loc[eligible, "anomaly_score"] = anomaly_score
    state.loc[eligible, "is_anomaly"] = prediction == -1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(OUTPUT, index=False)

    print("=== ANOMALY DETECTION ===")
    print("Total states:", len(state))
    print("Eligible states:", eligible.sum())
    print("Anomaly states:", state["is_anomaly"].sum())
    print("Anomaly ratio among eligible:",
          round(state.loc[eligible, "is_anomaly"].mean(), 4))
    print("Anomaly grids:",
          state.loc[state["is_anomaly"], "grid_id"].nunique())
    print("Anomaly windows:",
          state.loc[state["is_anomaly"], "window_id"].nunique())

    print("\nTop 10 anomalous states:")
    print(
        state.loc[eligible]
        .sort_values("anomaly_score", ascending=False)
        [[
            "window_id", "grid_id", "observation_count",
            "mean_speed_mps", "anomaly_score"
        ]]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
