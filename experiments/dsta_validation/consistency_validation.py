"""Controlled validation of DSTA joint spatiotemporal consistency checks.

The 815 spatial matches are candidate associations produced within official
DAIR frame pairs.  They are not identity ground truth.  This experiment keeps
the observed residuals intact for calibration, then injects labelled faults
only into an independent, later-frame test partition.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsta-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TRACKS_PATH = ROOT / "data/processed/world_tracks.parquet"
PAIRS_PATH = ROOT / "data/processed/frame_pairs.parquet"
MATCHES_PATH = ROOT / "outputs/reports/spatial_matches.csv"
OUT = ROOT / "outputs/ppt_case/dsta_validation/consistency_validation"
SEED = 20261023
CALIBRATION_FRAME_FRACTION = 0.40
QUANTILE = 0.99
STRENGTHS = {
    "low": {"time_ms": 12.0, "space_m": 0.8, "trajectory_m": 0.8},
    "medium": {"time_ms": 35.0, "space_m": 2.0, "trajectory_m": 2.0},
    "high": {"time_ms": 70.0, "space_m": 4.0, "trajectory_m": 4.0},
}
COLORS = {"blue": "#38BDF8", "orange": "#FB923C", "purple": "#C084FC",
          "yellow": "#FACC15", "text": "#F1F5F9", "muted": "#A8B8CC",
          "grid": "#334A66", "bg": "#080F1F", "panel": "#101C30"}
FEATURES = ["time_residual_ms", "space_residual_m", "trajectory_residual_m"]


def pad_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def add_causal_continuity(tracks: pd.DataFrame) -> pd.DataFrame:
    """Prediction error uses only the previous two points of the same track."""
    keys = ["source", "sequence_id", "track_id"]
    t = tracks.sort_values(keys + ["timestamp_us", "frame_id"]).copy()
    g = t.groupby(keys, sort=False)
    x1, y1, ts1 = g.world_x.shift(1), g.world_y.shift(1), g.timestamp_us.shift(1)
    x2, y2, ts2 = g.world_x.shift(2), g.world_y.shift(2), g.timestamp_us.shift(2)
    dt_prev = (ts1 - ts2) / 1e6
    dt_now = (t.timestamp_us - ts1) / 1e6
    valid = dt_prev.gt(0) & dt_now.gt(0) & dt_prev.le(0.35) & dt_now.le(0.35)
    pred_x = x1 + (x1 - x2) * dt_now / dt_prev
    pred_y = y1 + (y1 - y2) * dt_now / dt_prev
    residual = np.hypot(t.world_x - pred_x, t.world_y - pred_y).where(valid)
    t["continuity_prediction_x"] = pred_x.where(valid)
    t["continuity_prediction_y"] = pred_y.where(valid)
    t["continuity_residual_m"] = residual
    return t


def build_real_candidates() -> tuple[pd.DataFrame, dict]:
    tracks = add_causal_continuity(pd.read_parquet(TRACKS_PATH))
    pairs = pd.read_parquet(PAIRS_PATH).copy()
    matches = pd.read_csv(MATCHES_PATH).copy()
    for col in ["vehicle_frame", "infrastructure_frame", "vehicle_track_id", "infrastructure_track_id"]:
        matches[col] = pad_id(matches[col])

    pair_cols = ["vehicle_frame", "infrastructure_frame", "vehicle_sequence",
                 "infrastructure_sequence", "veh_lidar_ts", "inf_lidar_ts", "lidar_delta_ms"]
    candidates = matches.drop(columns=["lidar_delta_ms"]).merge(
        pairs[pair_cols], on=["vehicle_frame", "infrastructure_frame"], how="left", validate="many_to_one")

    obs_cols = ["sequence_id", "frame_id", "track_id", "timestamp_us", "world_x", "world_y",
                "continuity_prediction_x", "continuity_prediction_y", "continuity_residual_m"]
    veh = tracks[tracks.source.eq("vehicle-side")][obs_cols].rename(columns={
        "sequence_id": "vehicle_sequence", "frame_id": "vehicle_frame", "track_id": "vehicle_track_id",
        "timestamp_us": "vehicle_timestamp_us", "world_x": "vehicle_world_x", "world_y": "vehicle_world_y",
        "continuity_prediction_x": "vehicle_prediction_x", "continuity_prediction_y": "vehicle_prediction_y",
        "continuity_residual_m": "vehicle_continuity_residual_m"})
    inf = tracks[tracks.source.eq("infrastructure-side")][obs_cols].rename(columns={
        "sequence_id": "infrastructure_sequence", "frame_id": "infrastructure_frame", "track_id": "infrastructure_track_id",
        "timestamp_us": "infrastructure_timestamp_us", "world_x": "infrastructure_world_x", "world_y": "infrastructure_world_y",
        "continuity_prediction_x": "infrastructure_prediction_x", "continuity_prediction_y": "infrastructure_prediction_y",
        "continuity_residual_m": "infrastructure_continuity_residual_m"})
    candidates = candidates.merge(veh, on=["vehicle_sequence", "vehicle_frame", "vehicle_track_id"], how="left", validate="many_to_one")
    candidates = candidates.merge(inf, on=["infrastructure_sequence", "infrastructure_frame", "infrastructure_track_id"], how="left", validate="many_to_one")
    required = ["vehicle_timestamp_us", "infrastructure_timestamp_us", "vehicle_world_x", "infrastructure_world_x"]
    if candidates[required].isna().any().any():
        raise RuntimeError("Candidate keys did not uniquely recover every source observation")

    candidates["match_id"] = [f"M{i:04d}" for i in range(len(candidates))]
    candidates["time_residual_ms"] = (candidates.vehicle_timestamp_us - candidates.infrastructure_timestamp_us).abs() / 1000
    candidates["space_residual_m"] = np.hypot(candidates.vehicle_world_x - candidates.infrastructure_world_x,
                                                candidates.vehicle_world_y - candidates.infrastructure_world_y)
    candidates["trajectory_residual_m"] = candidates[["vehicle_continuity_residual_m", "infrastructure_continuity_residual_m"]].max(axis=1, skipna=True)
    candidates["trajectory_feature_available"] = candidates[["vehicle_continuity_residual_m", "infrastructure_continuity_residual_m"]].notna().any(axis=1)
    candidates.loc[~candidates.trajectory_feature_available, "trajectory_residual_m"] = np.nan
    space_check = float(np.max(np.abs(candidates.space_residual_m - candidates.distance_m)))
    time_check = float(np.max(np.abs(candidates.time_residual_ms - candidates.lidar_delta_ms.abs())))
    audit = {
        "candidate_matches": len(candidates), "official_frame_pairs_in_candidates": int(candidates.groupby(["vehicle_frame", "infrastructure_frame"]).ngroups),
        "all_candidate_keys_recovered": True, "max_space_reconstruction_difference_m": space_check,
        "max_time_reconstruction_difference_ms": time_check,
        "trajectory_feature_available": int(candidates.trajectory_feature_available.sum()),
        "trajectory_feature_missing": int((~candidates.trajectory_feature_available).sum()),
        "identity_status": "candidate one-to-one associations within official frame pairs; not verified cross-source identity labels",
    }
    return candidates.sort_values(["veh_lidar_ts", "vehicle_frame", "match_id"]).reset_index(drop=True), audit


def split_frames(data: pd.DataFrame) -> pd.DataFrame:
    frame_order = data[["vehicle_frame", "infrastructure_frame", "veh_lidar_ts"]].drop_duplicates().sort_values("veh_lidar_ts")
    cut = max(1, int(np.floor(len(frame_order) * CALIBRATION_FRAME_FRACTION)))
    calibration = set(map(tuple, frame_order.iloc[:cut][["vehicle_frame", "infrastructure_frame"]].to_numpy()))
    out = data.copy()
    out["split"] = ["calibration" if (v, i) in calibration else "test" for v, i in zip(out.vehicle_frame, out.infrastructure_frame)]
    return out


def assign_test_labels(test: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    out = test.copy().reset_index(drop=True)
    classes = np.array(["normal", "time_offset", "spatial_drift", "trajectory_break", "combined"])
    probs = np.array([0.40, 0.15, 0.15, 0.15, 0.15])
    counts = np.floor(probs * len(out)).astype(int); counts[0] += len(out) - counts.sum()
    labels = np.concatenate([np.repeat(c, n) for c, n in zip(classes, counts)])
    rng.shuffle(labels)
    out["anomaly_type"] = labels
    out["synthetic_anomaly_label"] = out.anomaly_type.ne("normal")
    out["injection_angle_rad"] = rng.uniform(-np.pi, np.pi, len(out))
    return out


def inject(test: pd.DataFrame, strength: str) -> pd.DataFrame:
    cfg = STRENGTHS[strength]; out = test.copy()
    out["strength"] = strength
    out["injected_time_shift_ms"] = 0.0
    out["injected_space_shift_m"] = 0.0
    out["injected_trajectory_shift_m"] = 0.0
    time_mask = out.anomaly_type.isin(["time_offset", "combined"])
    space_mask = out.anomaly_type.isin(["spatial_drift", "combined"])
    traj_mask = out.anomaly_type.isin(["trajectory_break", "combined"])
    out.loc[time_mask, "injected_time_shift_ms"] = cfg["time_ms"]
    out.loc[space_mask, "injected_space_shift_m"] = cfg["space_m"]
    out.loc[traj_mask, "injected_trajectory_shift_m"] = cfg["trajectory_m"]

    # Time shift increases the already observed absolute residual; direction is traceable.
    out["evaluated_time_residual_ms"] = out.time_residual_ms + out.injected_time_shift_ms
    # Spatial drift is applied away from the counterpart, avoiding accidental cancellation.
    out["evaluated_space_residual_m"] = out.space_residual_m + out.injected_space_shift_m
    # A common displacement of both current source positions preserves cross-source distance
    # but breaks their source-specific causal trajectory predictions.
    base_traj = out.trajectory_residual_m.copy()
    out["evaluated_trajectory_residual_m"] = base_traj + out.injected_trajectory_shift_m
    out.loc[~out.trajectory_feature_available, "evaluated_trajectory_residual_m"] = np.nan
    return out


def robust_model(calibration: pd.DataFrame) -> dict:
    model = {"features": {}, "quantile": QUANTILE}
    score_parts = []
    for f in FEATURES:
        values = calibration[f].dropna().to_numpy()
        med = float(np.median(values)); mad = float(np.median(np.abs(values - med)))
        scale = max(1.4826 * mad, float(np.std(values)) * 0.10, 1e-6)
        threshold = float(np.quantile(values, QUANTILE))
        model["features"][f] = {"median": med, "scale": scale, "threshold": threshold, "n": len(values)}
        score_parts.append(np.maximum((calibration[f] - med) / scale, 0).rename(f))
    z = pd.concat(score_parts, axis=1)
    calibration_score = np.sqrt((z.pow(2).sum(axis=1)) / z.notna().sum(axis=1).clip(lower=1))
    model["joint_threshold"] = float(np.quantile(calibration_score, QUANTILE))
    return model


def evaluate_samples(samples: pd.DataFrame, model: dict) -> pd.DataFrame:
    out = samples.copy(); zcols = []
    mapping = {"time_residual_ms": "evaluated_time_residual_ms", "space_residual_m": "evaluated_space_residual_m",
               "trajectory_residual_m": "evaluated_trajectory_residual_m"}
    for f, eval_col in mapping.items():
        p = model["features"][f]
        zcol = "z_" + f
        out[zcol] = np.maximum((out[eval_col] - p["median"]) / p["scale"], 0)
        zcols.append(zcol)
    out["available_feature_count"] = out[zcols].notna().sum(axis=1)
    out["consistency_score"] = np.sqrt(out[zcols].pow(2).sum(axis=1) / out.available_feature_count.clip(lower=1))
    out["pred_time_only"] = out.evaluated_time_residual_ms.gt(model["features"]["time_residual_ms"]["threshold"])
    out["pred_space_only"] = out.evaluated_space_residual_m.gt(model["features"]["space_residual_m"]["threshold"])
    out["pred_simple_or"] = out.pred_time_only | out.pred_space_only
    out["pred_joint_multidim"] = out.consistency_score.gt(model["joint_threshold"])
    out["consistency_label"] = out.pred_joint_multidim
    return out


def metric_row(y: np.ndarray, pred: np.ndarray, method: str, strength: str) -> dict:
    tp = int(np.sum(y & pred)); fp = int(np.sum(~y & pred)); fn = int(np.sum(y & ~pred)); tn = int(np.sum(~y & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"strength": strength, "method": method, "n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "normal_false_positive_rate": fp / (fp + tn) if fp + tn else 0.0}


def summarize(samples_by_strength: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []; recalls = []
    methods = {"time_only": "pred_time_only", "space_only": "pred_space_only", "simple_or": "pred_simple_or", "joint_multidim": "pred_joint_multidim"}
    for strength, data in samples_by_strength.items():
        y = data.synthetic_anomaly_label.to_numpy(bool)
        for method, col in methods.items():
            pred = data[col].to_numpy(bool)
            metrics.append(metric_row(y, pred, method, strength))
            for cls, group in data[data.synthetic_anomaly_label].groupby("anomaly_type"):
                recalls.append({"strength": strength, "method": method, "anomaly_type": cls,
                                "n": len(group), "detection_rate": float(group[col].mean())})
    return pd.DataFrame(metrics), pd.DataFrame(recalls)


def make_figure(cal: pd.DataFrame, test: pd.DataFrame, metrics: pd.DataFrame, recalls: pd.DataFrame) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.2, "text.color": COLORS["text"],
        "axes.labelcolor": COLORS["text"], "axes.titlecolor": COLORS["text"], "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"], "axes.edgecolor": COLORS["grid"], "savefig.facecolor": COLORS["bg"]})
    fig, axes = plt.subplots(2, 2, figsize=(13.333, 7.5), facecolor=COLORS["bg"])
    ax1, ax2, ax3, ax4 = axes.ravel()
    ax1.scatter(cal.time_residual_ms, cal.space_residual_m, c=COLORS["blue"], s=15, alpha=.55)
    ax1.set(title="A  Measured residuals · calibration only", xlabel="Observed time residual (ms)", ylabel="Observed spatial residual (m)")
    palette = {"normal": COLORS["blue"], "time_offset": COLORS["orange"], "spatial_drift": COLORS["yellow"],
               "trajectory_break": COLORS["purple"], "combined": "#F472B6"}
    for cls, group in test.groupby("anomaly_type"):
        ax2.scatter(group.evaluated_time_residual_ms, group.evaluated_space_residual_m, s=17, alpha=.62, color=palette[cls], label=cls.replace("_", " "))
    ax2.set(title="B  Independent test · medium synthetic faults", xlabel="Evaluated time residual (ms)", ylabel="Evaluated spatial residual (m)")
    medium = metrics[metrics.strength.eq("medium")].set_index("method").loc[["time_only", "space_only", "simple_or", "joint_multidim"]]
    x = np.arange(len(medium)); width = .23
    for j, (metric, color) in enumerate([("precision", COLORS["blue"]), ("recall", COLORS["orange"]), ("f1", COLORS["purple"])]):
        ax3.bar(x + (j - 1) * width, medium[metric], width=width, color=color, label=metric.upper())
    ax3.set(title="C  Detection metrics · medium strength", ylabel="Score", ylim=(0, 1.08), xticks=x,
            xticklabels=["Time", "Space", "OR", "Joint"])
    matrix = recalls[(recalls.strength.eq("medium"))].pivot(index="method", columns="anomaly_type", values="detection_rate")
    matrix = matrix.reindex(index=["time_only", "space_only", "simple_or", "joint_multidim"], columns=["time_offset", "spatial_drift", "trajectory_break", "combined"])
    im = ax4.imshow(matrix, vmin=0, vmax=1, cmap="plasma", aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax4.text(j, i, f"{matrix.iloc[i,j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    ax4.set(title="D  Per-fault detection matrix", xticks=np.arange(4), yticks=np.arange(4),
            xticklabels=["Time", "Space", "Trajectory", "Combined"], yticklabels=["Time", "Space", "OR", "Joint"])
    fig.colorbar(im, ax=ax4, fraction=.04, pad=.03, label="Detection rate")
    for ax in axes.ravel():
        ax.set_facecolor(COLORS["panel"]); ax.grid(color=COLORS["grid"], alpha=.28, lw=.6); ax.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=7, ncol=2, facecolor=COLORS["panel"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    ax3.legend(fontsize=7, facecolor=COLORS["panel"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.105, hspace=.38, wspace=.24)
    fig.suptitle("DSTA joint spatiotemporal consistency validation", x=.025, y=.965, ha="left", fontsize=20, fontweight="bold")
    fig.text(.025, .916, "815 DAIR candidate matches · measured calibration residuals · controlled faults on held-out frames", color=COLORS["muted"], fontsize=10)
    fig.text(.025, .025, "Candidate associations are not verified identities. Orange/purple/yellow test marks include synthetic perturbations; thresholds use normal calibration only.", color=COLORS["muted"], fontsize=8.7)
    fig.savefig(OUT / "consistency_validation_ppt.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "consistency_validation_ppt.svg", bbox_inches="tight")
    plt.close(fig)


def write_docs(audit: dict, model: dict, split: pd.DataFrame, metrics: pd.DataFrame, recalls: pd.DataFrame) -> None:
    med = metrics[metrics.strength.eq("medium")].set_index("method")
    traj_avail = audit["trajectory_feature_available"]
    audit_md = f"""# 时空一致性样本审查

- `spatial_matches.csv` 共 {audit['candidate_matches']} 条，覆盖 {audit['official_frame_pairs_in_candidates']} 组 DAIR 官方帧对；它们是同类别、5 m 门限内匈牙利匹配得到的**候选关联**，不是已验证的跨源身份真值。
- 815 条均能通过 `帧对 + 各源 sequence_id + 各源 track_id` 唯一回连到 `world_tracks.parquet` 的时间戳和世界坐标。
- 重算空间残差与原 CSV 最大差异 {audit['max_space_reconstruction_difference_m']:.3e} m；重算时间残差与帧对 LiDAR 时间差最大差异 {audit['max_time_reconstruction_difference_ms']:.3e} ms。
- 轨迹连续性只使用同源、同序列、同轨迹的前两个历史点作因果恒速预测。可用 {traj_avail} 条，缺失 {audit['trajectory_feature_missing']} 条；缺失维度不填造数值，在联合分数中按可用维度归一。
- 按官方帧对时间顺序切分：校准 {int((split.split=='calibration').sum())} 条，测试 {int((split.split=='test').sum())} 条。扰动仅注入测试副本，校准集未扰动。
"""
    (OUT / "data_audit.md").write_text(audit_md, encoding="utf-8")
    readme = f"""# DSTA 时空一致性联合校验

## 口径

真实部分是 DAIR 官方帧对、候选匹配的事件时间、世界坐标和同轨迹历史连续性残差。异常标签全部是独立测试副本上的 **synthetic anomaly labels**。候选匹配并非跨源身份真值，因此本实验验证的是受控扰动下的校验能力，不宣称真实身份匹配准确率。

正常校准集只用于计算 99% 单维阈值、稳健中位数/MAD 尺度和联合分数阈值；测试标签未参与调参。联合方法对时间、空间、轨迹残差分别稳健归一化，再按每条记录实际可用的维度计算 RMS 分数。轨迹缺失被显式保留。

## 中等强度结果

| 方法 | Precision | Recall | F1 | FPR |
|---|---:|---:|---:|---:|
| 单时间阈值 | {med.loc['time_only','precision']:.3f} | {med.loc['time_only','recall']:.3f} | {med.loc['time_only','f1']:.3f} | {med.loc['time_only','fpr']:.3f} |
| 单空间阈值 | {med.loc['space_only','precision']:.3f} | {med.loc['space_only','recall']:.3f} | {med.loc['space_only','f1']:.3f} | {med.loc['space_only','fpr']:.3f} |
| 时间/空间 OR | {med.loc['simple_or','precision']:.3f} | {med.loc['simple_or','recall']:.3f} | {med.loc['simple_or','f1']:.3f} | {med.loc['simple_or','fpr']:.3f} |
| 多维联合 | {med.loc['joint_multidim','precision']:.3f} | {med.loc['joint_multidim','recall']:.3f} | {med.loc['joint_multidim','f1']:.3f} | {med.loc['joint_multidim','fpr']:.3f} |

## 文件

- `consistency_validation_ppt.png` / `.svg`：PPT 主图（PNG 300 DPI）。
- `consistency_metrics.csv`：Precision、Recall、F1、FPR 与混淆计数。
- `consistency_class_recall.csv`：各异常类型检出率。
- `consistency_sensitivity.csv`：低/中/高扰动强度敏感性结果。
- `consistency_samples.parquet`：三档扰动的逐点结果。
- `cube_consistency_layer.parquet`：中等强度独立实验层，不覆盖统一真实数据表。
- `consistency_validation_config.json`：固定种子、切分、扰动和阈值配置。
- `data_audit.md`：真实键关联及可用性审查。

## 限制

空间候选在生成时已受 5 m 门限截断，不能代表无条件真实配对分布；正常参考也可能包含误匹配。轨迹残差仅在已有历史点足够且时间间隔合理时可用。所有性能数字只适用于所声明的受控异常强度，不能解释为真实事故检出率。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real, audit = build_real_candidates()
    split = split_frames(real)
    calibration = split[split.split.eq("calibration")].copy()
    test_base = assign_test_labels(split[split.split.eq("test")].copy())
    if set(calibration.match_id) & set(test_base.match_id):
        raise RuntimeError("Calibration/test leakage detected")
    model = robust_model(calibration)
    scenarios = {s: evaluate_samples(inject(test_base, s), model) for s in STRENGTHS}
    metrics, recalls = summarize(scenarios)
    samples = pd.concat(scenarios.values(), ignore_index=True)

    metrics.to_csv(OUT / "consistency_metrics.csv", index=False)
    recalls.to_csv(OUT / "consistency_class_recall.csv", index=False)
    metrics.to_csv(OUT / "consistency_sensitivity.csv", index=False)
    samples.to_parquet(OUT / "consistency_samples.parquet", index=False)
    medium = scenarios["medium"].copy()
    medium["provenance"] = "DAIR_candidate_match_plus_controlled_test_perturbation"
    cube_cols = ["match_id", "vehicle_frame", "infrastructure_frame", "vehicle_sequence", "infrastructure_sequence",
                 "vehicle_track_id", "infrastructure_track_id", "consistency_label", "consistency_score",
                 "anomaly_type", "synthetic_anomaly_label", "trajectory_feature_available", "available_feature_count",
                 "evaluated_time_residual_ms", "evaluated_space_residual_m", "evaluated_trajectory_residual_m",
                 "injected_time_shift_ms", "injected_space_shift_m", "injected_trajectory_shift_m", "provenance"]
    medium[cube_cols].to_parquet(OUT / "cube_consistency_layer.parquet", index=False)
    config = {"seed": SEED, "source_files": [str(p.relative_to(ROOT)) for p in [TRACKS_PATH, PAIRS_PATH, MATCHES_PATH]],
              "candidate_identity_status": audit["identity_status"], "split": {"unit": "official_frame_pair", "chronological": True,
              "calibration_fraction": CALIBRATION_FRAME_FRACTION, "calibration_rows": len(calibration), "test_rows": len(test_base)},
              "calibration": {"normal_only": True, "quantile": QUANTILE, "model": model}, "perturbation_strengths": STRENGTHS,
              "labels": "synthetic anomaly labels on held-out test copies", "test_labels_used_for_tuning": False,
              "trajectory": {"method": "causal constant-velocity prediction", "same_source_sequence_track_only": True,
              "max_history_gap_s": 0.35, "missing_handling": "excluded per row; joint RMS normalized by available feature count"}}
    (OUT / "consistency_validation_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(calibration, medium, metrics, recalls)
    write_docs(audit, model, split, metrics, recalls)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("\nSplit:", split.split.value_counts().to_dict())
    print("\nSynthetic labels:", test_base.anomaly_type.value_counts().to_dict())
    print("\nMetrics:\n", metrics.to_string(index=False))
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
