
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/processed/anomaly_states.parquet"
OUTPUT = ROOT / "data/processed/city_state_embeddings.parquet"
FIGURE = ROOT / "outputs/figures/12_state_embedding.png"

N_COMPONENTS = 8


def main():
    state = pd.read_parquet(INPUT).copy()
    state = state.sort_values(
        ["grid_id", "window_id"]
    ).reset_index(drop=True)

    # 1. 基础状态特征
    state["log_density"] = np.log1p(
        state["observation_density"].clip(lower=0)
    )

    base_features = [
        "log_density",
        "mean_speed_mps",
        "speed_std_mps",
        "vehicle_share",
    ]

    # 2. 时间上下文：仅使用同一网格的上一窗口
    prev = state.groupby("grid_id", sort=False).shift(1)
    prev_window = prev["window_id"]
    consecutive = (
        state["window_id"] - prev_window == 1
    )

    for col in ["log_density", "mean_speed_mps"]:
        state[f"delta_{col}"] = (
            state[col] - prev[col]
        ).where(consecutive)

    state["has_previous_window"] = consecutive.astype(int)

    # 3. 空间上下文：
    # 使用同时间窗口其他网格的中心点近邻
    # 这是轻量近邻近似，不等同于真实道路拓扑邻接
    state["neighbor_density"] = np.nan
    state["neighbor_speed"] = np.nan

    for _, idx in state.groupby("window_id").groups.items():
        subset = state.loc[idx]
        if len(subset) < 2:
            continue

        coords = subset[["mean_world_x", "mean_world_y"]].to_numpy()
        k = min(5, len(subset))

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(coords)
        neighbors = nn.kneighbors(
            coords, return_distance=False
        )

        density = subset["log_density"].to_numpy()
        speed = subset["mean_speed_mps"].to_numpy()

        neighbor_density = []
        neighbor_speed = []

        for i, indices in enumerate(neighbors):
            others = indices[indices != i]
            if len(others) == 0:
                neighbor_density.append(np.nan)
                neighbor_speed.append(np.nan)
                continue

            neighbor_density.append(
                float(np.nanmean(density[others]))
            )

            valid_speed = speed[others]
            neighbor_speed.append(
                float(np.nanmean(valid_speed))
                if np.isfinite(valid_speed).any()
                else np.nan
            )

        state.loc[idx, "neighbor_density"] = neighbor_density
        state.loc[idx, "neighbor_speed"] = neighbor_speed

    features = base_features + [
        "delta_log_density",
        "delta_mean_speed_mps",
        "has_previous_window",
        "neighbor_density",
        "neighbor_speed",
    ]

    # 4. 缺失指示：保留观测不足的信息
    X = state[features].replace(
        [np.inf, -np.inf], np.nan
    )
    missing = X.isna().astype(int).add_suffix("_missing")

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )
    X_filled = imputer.fit_transform(X)
    X_full = np.column_stack(
        [X_filled, missing.to_numpy()]
    )

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_full)

    # 5. PCA 状态表示
    pca = PCA(
        n_components=N_COMPONENTS,
        random_state=42,
    )
    embedding = pca.fit_transform(X_scaled)

    for i in range(N_COMPONENTS):
        state[f"embedding_{i}"] = embedding[:, i]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    state.to_parquet(OUTPUT, index=False)

    # 6. 可视化：前两个主成分
    FIGURE.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        state["embedding_0"],
        state["embedding_1"],
        c=state["mean_speed_mps"],
        cmap="plasma",
        s=18,
        alpha=0.75,
    )

    fig.colorbar(
        scatter, ax=ax,
        label="Estimated speed (m/s)"
    )
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Spatiotemporal State Embedding (PCA)")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)

    print("=== STATE EMBEDDING ===")
    print("Input states:", len(state))
    print("Feature dimensions:", X_full.shape[1])
    print("Embedding dimensions:", N_COMPONENTS)
    print("Explained variance ratio:",
          np.round(pca.explained_variance_ratio_, 4))
    print("Cumulative explained variance:",
          round(
              float(pca.explained_variance_ratio_.sum()),
              4
          ))
    print("Embedding missing values:",
          int(np.isnan(embedding).sum()))
    print("Saved:", OUTPUT)
    print("Figure:", FIGURE)


if __name__ == "__main__":
    main()
