"""Controlled adaptive time-alignment validation around DSTA event EVT2_006.

Real component: DAIR-V2X-Seq official vehicle/infrastructure LiDAR frame pairs.
Synthetic component: seeded clock jitter and network-arrival delay injection.
No upstream DSTA artifact is modified.
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
INPUT_PAIRS = ROOT / "data/processed/frame_pairs.parquet"
INPUT_STATES = ROOT / "data/processed/city_state.parquet"
OUTPUT = ROOT / "outputs/ppt_case/time_alignment"
EVENT_ID = "EVT2_006"
SEED = 20261023
BASELINE_WINDOW_MS = 50.0
CALIBRATION_PAIRS = 10

COLORS = {"blue": "#38BDF8", "orange": "#FB923C", "purple": "#C084FC",
          "ink": "#E8EEF7", "muted": "#9FB0C5", "grid": "#334A66",
          "bg": "#080F1F", "panel": "#101C30"}


def greedy_match(left_ms: np.ndarray, right_ms: np.ndarray,
                 tolerances_ms: np.ndarray) -> pd.DataFrame:
    """One-to-one nearest-neighbour match in time, without reusing right records."""
    available = set(range(len(right_ms)))
    rows = []
    for left_idx, value in enumerate(left_ms):
        if not available:
            break
        candidates = np.fromiter(sorted(available), dtype=int)
        distances = np.abs(right_ms[candidates] - value)
        best_pos = int(np.argmin(distances))
        right_idx, error = int(candidates[best_pos]), float(distances[best_pos])
        if error <= float(tolerances_ms[left_idx]):
            rows.append({"vehicle_pair_id": left_idx, "matched_infrastructure_pair_id": right_idx,
                         "matching_residual_ms": error,
                         "window_ms": float(tolerances_ms[left_idx])})
            available.remove(right_idx)
    return pd.DataFrame(rows)


def rolling_mad(values: pd.Series, window: int = 11) -> pd.Series:
    return values.rolling(window, min_periods=3, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True).bfill().ffill()


def evaluate(name: str, matches: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    if matches.empty:
        return {"method": name, "official_pairs": len(pairs), "matched_pairs": 0,
                "coverage_rate": 0.0, "correct_pairs": 0, "correct_pair_rate": 0.0,
                "precision_among_matches": 0.0, "median_time_error_ms": np.nan,
                "p90_time_error_ms": np.nan}
    matches = matches.copy()
    matches["is_official_pair"] = matches.vehicle_pair_id.eq(
        matches.matched_infrastructure_pair_id)
    # Remove the physical inter-sensor offset estimated on calibration pairs.
    offset = float(np.median((pairs.veh_event_true_us - pairs.inf_event_true_us).iloc[:CALIBRATION_PAIRS]) / 1000)
    inf_matched = pairs.inf_event_true_us.to_numpy()[matches.matched_infrastructure_pair_id] / 1000
    veh = pairs.veh_event_true_us.to_numpy()[matches.vehicle_pair_id] / 1000
    matches["evaluation_time_error_ms"] = np.abs((veh - inf_matched) - offset)
    correct = int(matches.is_official_pair.sum())
    return {"method": name, "official_pairs": len(pairs), "matched_pairs": len(matches),
            "coverage_rate": len(matches) / len(pairs), "correct_pairs": correct,
            "correct_pair_rate": correct / len(pairs),
            "precision_among_matches": correct / len(matches),
            "median_time_error_ms": float(matches.evaluation_time_error_ms.median()),
            "p90_time_error_ms": float(matches.evaluation_time_error_ms.quantile(.9))}


def make_figure(data: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.facecolor": COLORS["panel"], "figure.facecolor": COLORS["bg"],
                         "text.color": COLORS["ink"], "axes.labelcolor": COLORS["ink"],
                         "xtick.color": COLORS["muted"], "ytick.color": COLORS["muted"],
                         "axes.edgecolor": COLORS["grid"], "savefig.facecolor": COLORS["bg"]})
    fig = plt.figure(figsize=(13.333, 7.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 1])
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    sample = data.iloc[:35]
    t0 = data.veh_event_true_us.iloc[0]
    x = (sample.veh_event_true_us - t0) / 1e6
    ax1.scatter(x, (sample.veh_arrival_us - t0) / 1e6, s=28, color=COLORS["blue"],
                label="Vehicle arrival (synthetic delay)")
    ax1.scatter(x, (sample.inf_arrival_us - t0) / 1e6, s=28, marker="D", color=COLORS["orange"],
                label="Infrastructure arrival (synthetic delay)")
    ax1.plot([x.min(), x.max()], [x.min(), x.max()], ls="--", lw=1, color=COLORS["muted"],
             label="Ideal event-time line")
    ax1.set(title="A  Arrival-time baseline: two sources drift apart",
            xlabel="Official pair time from case start (s)", ylabel="Observed arrival time (s)")
    ax1.legend(fontsize=8, loc="upper left")

    event_x = (sample.veh_event_observed_us - t0) / 1e6
    ax2.scatter(event_x, (sample.inf_event_corrected_us - t0) / 1e6, s=30,
                c=sample.adaptive_window_ms, cmap="plasma", vmin=20, vmax=80)
    limits = [min(event_x.min(), ((sample.inf_event_corrected_us-t0)/1e6).min()),
              max(event_x.max(), ((sample.inf_event_corrected_us-t0)/1e6).max())]
    ax2.plot(limits, limits, ls="--", lw=1.2, color=COLORS["muted"])
    ax2.set(title="B  Event-time alignment after offset calibration",
            xlabel="Vehicle event time (s)", ylabel="Infrastructure event time, corrected (s)")

    idx = np.arange(len(data))
    ax3.plot(idx, data.adaptive_window_ms, color=COLORS["purple"], lw=2.2,
             label="Adaptive matching window")
    ax3.axhline(BASELINE_WINDOW_MS, color=COLORS["orange"], ls="--", lw=1.5,
                label="Fixed baseline window")
    burst = data.synthetic_burst_delay
    if burst.any():
        ax3.fill_between(idx, 0, 85, where=burst, color=COLORS["orange"], alpha=.10,
                         label="Injected delay burst")
    ax3.set(title="C  Dynamic window responds to observed timing volatility",
            xlabel="Official cooperative frame-pair index", ylabel="Matching window (ms)", ylim=(0, 85))
    ax3.legend(fontsize=9, ncol=3, loc="upper right")
    for ax in (ax1, ax2, ax3):
        ax.grid(color=COLORS["grid"], alpha=.35, lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    title = "DSTA adaptive time alignment · EVT2_006 context"
    fig.subplots_adjust(left=.075, right=.985, top=.85, bottom=.125, hspace=.40, wspace=.12)
    fig.suptitle(title, fontsize=20, fontweight="bold", x=.02, y=.975, ha="left")
    fig.text(.02, .925,
             f"Real: {len(data)} official DAIR vehicle–infrastructure LiDAR pairs  |  Synthetic: seeded clock jitter and network delay",
             color=COLORS["muted"], fontsize=10)
    note = "  |  ".join(
        f"{r.method}: coverage {r.coverage_rate:.1%}, correct-pair rate {r.correct_pair_rate:.1%}, P90 error {r.p90_time_error_ms:.1f} ms"
        for r in metrics.itertuples())
    fig.text(.02, .025, note, fontsize=8.5, color=COLORS["ink"])
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix("." + suffix), dpi=300 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_parquet(INPUT_PAIRS).sort_values("veh_lidar_ts").reset_index(drop=True)
    states = pd.read_parquet(INPUT_STATES)
    starts = states[["window_id", "window_start_us"]].drop_duplicates().sort_values("window_id")
    start_us = int(starts.loc[starts.window_id.eq(2), "window_start_us"].iloc[0])
    end_us = int(starts.loc[starts.window_id.eq(6), "window_start_us"].iloc[0] + 2_000_000)
    pairs = pairs[pairs.veh_lidar_ts.between(start_us, end_us, inclusive="left")].copy().reset_index(drop=True)
    if len(pairs) < 30:
        raise RuntimeError("Insufficient official cooperative pairs around EVT2_006")

    rng = np.random.default_rng(SEED)
    n = len(pairs)
    veh_clock_ms = rng.normal(0, 4.0, n)
    inf_clock_ms = rng.normal(0, 7.0, n)
    veh_delay_ms = np.clip(rng.lognormal(np.log(28), .32, n), 5, 100)
    inf_delay_ms = np.clip(rng.lognormal(np.log(48), .42, n), 8, 150)
    burst = np.zeros(n, dtype=bool)
    burst[int(.38*n):int(.62*n)] = True
    inf_delay_ms[burst] += rng.uniform(70, 145, burst.sum())

    out = pairs[["vehicle_frame", "infrastructure_frame", "veh_lidar_ts", "inf_lidar_ts",
                 "lidar_delta_ms"]].copy()
    out.insert(0, "official_pair_id", np.arange(n))
    out.rename(columns={"veh_lidar_ts": "veh_event_true_us", "inf_lidar_ts": "inf_event_true_us"}, inplace=True)
    out["veh_clock_jitter_ms_synthetic"] = veh_clock_ms
    out["inf_clock_jitter_ms_synthetic"] = inf_clock_ms
    out["veh_network_delay_ms_synthetic"] = veh_delay_ms
    out["inf_network_delay_ms_synthetic"] = inf_delay_ms
    out["synthetic_burst_delay"] = burst
    out["veh_event_observed_us"] = out.veh_event_true_us + np.rint(veh_clock_ms * 1000).astype("int64")
    out["inf_event_observed_us"] = out.inf_event_true_us + np.rint(inf_clock_ms * 1000).astype("int64")
    out["veh_arrival_us"] = out.veh_event_observed_us + np.rint(veh_delay_ms * 1000).astype("int64")
    out["inf_arrival_us"] = out.inf_event_observed_us + np.rint(inf_delay_ms * 1000).astype("int64")

    calibration_offset_ms = float(np.median(
        (out.veh_event_observed_us - out.inf_event_observed_us).iloc[:CALIBRATION_PAIRS]) / 1000)
    out["inf_event_corrected_us"] = out.inf_event_observed_us + int(round(calibration_offset_ms * 1000))
    veh_interval = pd.Series(out.veh_event_observed_us).diff().abs().div(1000)
    interval_med = veh_interval.rolling(13, min_periods=3, center=True).median().bfill().ffill()
    timing_volatility = rolling_mad(pd.Series(np.r_[veh_delay_ms, inf_delay_ms]).rolling(
        2, min_periods=1).mean().iloc[:n], 11)
    out["adaptive_window_ms"] = np.clip(.22 * interval_med + 1.5 * timing_volatility, 20, 80)

    baseline = greedy_match(out.veh_arrival_us.to_numpy()/1000,
                            out.inf_arrival_us.to_numpy()/1000,
                            np.full(n, BASELINE_WINDOW_MS))
    improved = greedy_match(out.veh_event_observed_us.to_numpy()/1000,
                            out.inf_event_corrected_us.to_numpy()/1000,
                            out.adaptive_window_ms.to_numpy())
    baseline["method"] = "Fixed arrival-time window"
    improved["method"] = "Adaptive event-time window"
    matches = pd.concat([baseline, improved], ignore_index=True)
    metrics = pd.DataFrame([
        evaluate("Fixed arrival-time window", baseline, out),
        evaluate("Adaptive event-time window", improved, out)])

    out.to_csv(OUTPUT / "time_alignment_samples.csv", index=False)
    matches.to_csv(OUTPUT / "time_alignment_matches.csv", index=False)
    metrics.to_csv(OUTPUT / "time_alignment_metrics.csv", index=False)
    config = {
        "experiment": "adaptive_time_alignment", "event_id": EVENT_ID, "seed": SEED,
        "real_ground_truth": "DAIR-V2X-Seq official cooperative LiDAR frame pairs",
        "case_windows": [2, 6], "case_window_seconds": 2, "official_pairs": n,
        "synthetic_injection": {
            "vehicle_clock_jitter_ms_sigma": 4.0, "infrastructure_clock_jitter_ms_sigma": 7.0,
            "vehicle_delay_lognormal_median_ms": 28, "infrastructure_delay_lognormal_median_ms": 48,
            "infrastructure_burst_fraction": [0.38, 0.62], "burst_extra_delay_ms": [70, 145]},
        "baseline": {"time_basis": "synthetic arrival time", "fixed_window_ms": BASELINE_WINDOW_MS},
        "improved": {"time_basis": "event time", "calibration_pairs": CALIBRATION_PAIRS,
                     "adaptive_window_formula": "clip(0.22*rolling_median_interval + 1.5*rolling_MAD_delay, 20, 80)"},
        "limitations": ["Network delays and clock jitter are synthetic, not measured in DAIR.",
                        "Official frame association is used only as evaluation ground truth and initial offset calibration.",
                        "Results validate alignment robustness under this controlled perturbation, not deployment latency."],
    }
    (OUTPUT / "time_alignment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(out, metrics, OUTPUT / "time_alignment_ppt")
    print(metrics.to_string(index=False))
    print("Saved to", OUTPUT)


if __name__ == "__main__":
    main()
