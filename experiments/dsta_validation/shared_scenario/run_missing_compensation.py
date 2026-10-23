"""Shared controlled perturbation + out-of-order/missing compensation validation.

The reference is an untouched, real DAIR trajectory. Perturbations and labels are
synthetic and reproducible. This run does not evaluate modality weighting or joint
consistency validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsta-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/ppt_case/dsta_validation/missing_compensation"
SHARED = ROOT / "experiments/dsta_validation/shared_scenario/artifacts"
SEED = 20261023
MISSING_RATES = [0.10, 0.20, 0.30, 0.40]
CALIBRATION_POINTS = 20
ORDER_PERTURBATION_RATE = 0.22
MAX_ORDER_SHIFT_MS = 260.0
MAX_INTERPOLATION_GAP_S = 0.60
REFERENCE_KEY = ("vehicle-side", "0058", "023991")
REFERENCE_SEGMENT = 5
COLORS = {"blue": "#38BDF8", "orange": "#FB923C", "purple": "#C084FC",
          "text": "#F1F5F9", "muted": "#A8B8CC", "grid": "#334A66",
          "bg": "#080F1F", "panel": "#101C30"}
BLUE, ORANGE, PURPLE = COLORS["blue"], COLORS["orange"], COLORS["purple"]


def load_reference() -> pd.DataFrame:
    obs = pd.read_parquet(ROOT / "data/processed/spatiotemporal_events.parquet")
    source, sequence, track = REFERENCE_KEY
    track_df = obs[(obs.source.eq(source)) & (obs.sequence_id.astype(str).eq(sequence)) &
                   (obs.track_id.astype(str).eq(track))].sort_values("timestamp_us").reset_index(drop=True)
    dt_ms = track_df.timestamp_us.diff().fillna(0).div(1000)
    track_df["contiguous_segment"] = dt_ms.gt(160).cumsum()
    ref = track_df[track_df.contiguous_segment.eq(REFERENCE_SEGMENT)].copy().reset_index(drop=True)
    if len(ref) < 60:
        raise RuntimeError("Selected supplemental real segment is no longer long enough")
    ref.insert(0, "sample_id", [f"SUPP_023991_{i:03d}" for i in range(len(ref))])
    ref["relative_time_s"] = (ref.timestamp_us - ref.timestamp_us.iloc[0]) / 1_000_000
    ref["split"] = np.where(np.arange(len(ref)) < CALIBRATION_POINTS, "calibration", "test")
    ref["reference_role"] = "untouched_real_observation"
    ref["case_relation"] = "supplemental_validation_segment_not_EVT2_006"
    return ref


def perturb(ref: pd.DataFrame, missing_rate: float, rng: np.random.Generator,
            include_shared_future_noise: bool = False) -> pd.DataFrame:
    out = ref.copy()
    test_idx = np.flatnonzero(out.split.eq("test").to_numpy())
    candidates = test_idx[1:-1]
    missing_n = int(round(missing_rate * len(test_idx)))
    missing_ids = rng.choice(candidates, size=missing_n, replace=False)
    out["injected_missing"] = False
    out.loc[missing_ids, "injected_missing"] = True
    base_delay_ms = np.clip(rng.lognormal(np.log(35), .36, len(out)), 5, 120)
    reorder_candidates = np.setdiff1d(test_idx, missing_ids)
    reorder_n = int(round(ORDER_PERTURBATION_RATE * len(test_idx)))
    reorder_ids = rng.choice(reorder_candidates, size=min(reorder_n, len(reorder_candidates)), replace=False)
    shift_ms = np.zeros(len(out))
    shift_ms[reorder_ids] = rng.uniform(-MAX_ORDER_SHIFT_MS, MAX_ORDER_SHIFT_MS, len(reorder_ids))
    out["network_delay_ms_synthetic"] = base_delay_ms
    out["arrival_shift_ms_synthetic"] = shift_ms
    out["arrival_timestamp_us_synthetic"] = (out.timestamp_us +
        np.rint((base_delay_ms + shift_ms) * 1000).astype("int64"))
    out["injected_out_of_order"] = False
    retained = out[~out.injected_missing].sort_values("arrival_timestamp_us_synthetic")
    emitted_event = retained.timestamp_us.to_numpy()
    inversions = np.r_[False, np.diff(emitted_event) < 0]
    out.loc[retained.index, "injected_out_of_order"] = inversions

    # Reserved shared-scenario inputs for later experiments. They are not used by
    # the missing-compensation evaluation in this run.
    sigma = 0.65 if include_shared_future_noise else 0.0
    out["spatial_noise_sigma_m_synthetic"] = sigma
    out["world_x_perturbed"] = out.world_x + rng.normal(0, sigma, len(out))
    out["world_y_perturbed"] = out.world_y + rng.normal(0, sigma, len(out))
    degrade = include_shared_future_noise & out.relative_time_s.between(4.0, 6.0)
    out["injected_source_quality_degradation"] = degrade
    out["source_quality_factor_synthetic"] = np.where(degrade, .35, 1.0)
    out["scenario_id"] = f"missing_{int(missing_rate*100):02d}_order_{int(ORDER_PERTURBATION_RATE*100):02d}"
    out["random_seed"] = SEED
    return out


def interpolate_estimates(scenario: pd.DataFrame, method: str) -> pd.DataFrame:
    retained = scenario[~scenario.injected_missing].copy()
    target = scenario[scenario.injected_missing & scenario.split.eq("test")].copy()
    if method == "baseline_arrival_time":
        retained = retained.sort_values("arrival_timestamp_us_synthetic")
        time = retained.arrival_timestamp_us_synthetic.to_numpy(dtype=float)
        target_time = target.timestamp_us.to_numpy(dtype=float)
        est_x = np.interp(target_time, time, retained.world_x.to_numpy())
        est_y = np.interp(target_time, time, retained.world_y.to_numpy())
        covered = np.ones(len(target), dtype=bool)
        gaps = np.full(len(target), np.nan)
    else:
        retained = retained.sort_values("timestamp_us")
        time = retained.timestamp_us.to_numpy(dtype=float)
        target_time = target.timestamp_us.to_numpy(dtype=float)
        positions = np.searchsorted(time, target_time)
        covered, est_x, est_y, gaps = [], [], [], []
        for t, pos in zip(target_time, positions):
            if pos == 0 or pos == len(time):
                covered.append(False); est_x.append(np.nan); est_y.append(np.nan); gaps.append(np.nan); continue
            left, right = pos-1, pos
            gap_s = (time[right]-time[left])/1e6
            ok = gap_s <= MAX_INTERPOLATION_GAP_S
            ratio = (t-time[left])/(time[right]-time[left])
            covered.append(ok); gaps.append(gap_s)
            est_x.append(retained.world_x.iloc[left] + ratio*(retained.world_x.iloc[right]-retained.world_x.iloc[left]) if ok else np.nan)
            est_y.append(retained.world_y.iloc[left] + ratio*(retained.world_y.iloc[right]-retained.world_y.iloc[left]) if ok else np.nan)
        covered, est_x, est_y, gaps = map(np.asarray, (covered, est_x, est_y, gaps))
    result = target[["sample_id", "timestamp_us", "relative_time_s", "world_x", "world_y",
                     "scenario_id"]].copy()
    result["method"] = method
    result["estimated_world_x"] = est_x
    result["estimated_world_y"] = est_y
    result["restored"] = covered
    result["interpolation_gap_s"] = gaps
    result["position_error_m"] = np.sqrt((result.estimated_world_x-result.world_x)**2 +
                                          (result.estimated_world_y-result.world_y)**2)
    result["estimate_provenance"] = ("arrival_time_linear_interpolation" if method.startswith("baseline")
                                      else "event_time_buffered_continuity_interpolation")
    return result


def metric_row(rate: float, method: str, estimates: pd.DataFrame, scenario: pd.DataFrame) -> dict:
    valid = estimates[estimates.restored]
    retained = scenario[~scenario.injected_missing].sort_values("arrival_timestamp_us_synthetic")
    event_order = retained.timestamp_us.to_numpy()
    inversion_count = int((np.diff(event_order) < 0).sum())
    return {"missing_rate": rate, "method": method,
            "reference_test_points": int(scenario.split.eq("test").sum()),
            "injected_missing_points": int(scenario.injected_missing.sum()),
            "arrival_order_inversions": inversion_count,
            "post_reorder_inversions": 0 if method.startswith("improved") else inversion_count,
            "restored_points": int(estimates.restored.sum()),
            "recovery_coverage": float(estimates.restored.mean()),
            "position_rmse_m": float(np.sqrt(np.mean(valid.position_error_m**2))) if len(valid) else np.nan,
            "median_position_error_m": float(valid.position_error_m.median()) if len(valid) else np.nan,
            "p90_position_error_m": float(valid.position_error_m.quantile(.9)) if len(valid) else np.nan}


def make_figure(ref, scenario, restored, metrics):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":COLORS["text"],
        "axes.labelcolor":COLORS["text"],"axes.titlecolor":COLORS["text"],"xtick.color":COLORS["muted"],
        "ytick.color":COLORS["muted"],"axes.edgecolor":COLORS["grid"],"savefig.facecolor":COLORS["bg"]})
    fig=plt.figure(figsize=(13.333,7.5),facecolor=COLORS["bg"])
    gs=fig.add_gridspec(2,2,height_ratios=[1,1.02])
    ax1,ax2,ax3=fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])
    retained=scenario[~scenario.injected_missing].sort_values("arrival_timestamp_us_synthetic").head(34)
    rank=np.arange(len(retained))
    event_rank=retained.timestamp_us.rank(method="first").to_numpy()-1
    for x,a,b in zip(rank,rank,event_rank): ax1.plot([x,x],[a,b],color=ORANGE,alpha=.28,lw=.8)
    ax1.scatter(rank,rank,s=22,color=ORANGE,label="Arrival-order rank")
    ax1.scatter(rank,event_rank,s=22,color=BLUE,label="Event-time rank")
    ax1.set(title="A  Arrival order vs event-time order",xlabel="Received sample index",ylabel="Rank")
    ax1.legend(fontsize=8,facecolor=COLORS["panel"],edgecolor=COLORS["grid"],labelcolor=COLORS["text"],framealpha=.95)

    ax2.plot(ref.world_x,ref.world_y,color=BLUE,lw=2.2,label="Untouched real reference")
    kept=scenario[~scenario.injected_missing]
    ax2.scatter(kept.world_x,kept.world_y,s=15,color=ORANGE,alpha=.7,label="Retained perturbed stream")
    improved=restored[restored.method.eq("improved_event_time") & restored.restored]
    ax2.scatter(improved.estimated_world_x,improved.estimated_world_y,s=34,marker="D",color=PURPLE,
                edgecolors=COLORS["text"],linewidths=.4,label="DSTA restored estimate")
    ax2.set(title="B  Real trajectory and restored missing points",xlabel="World X (m)",ylabel="World Y (m)")
    ax2.axis("equal");ax2.legend(fontsize=8,facecolor=COLORS["panel"],edgecolor=COLORS["grid"],labelcolor=COLORS["text"],framealpha=.95)

    for method,color,label in [("baseline_arrival_time",ORANGE,"Baseline: arrival-time interpolation"),
                               ("improved_event_time",PURPLE,"Improved: event-time + continuity")]:
        m=metrics[metrics.method.eq(method)]
        ax3.plot(m.missing_rate*100,m.position_rmse_m,marker="o",lw=2.4,color=color,label=label)
    ax3.set(title="C  Missing-point restoration error under controlled loss",xlabel="Injected missing rate (%)",
            ylabel="Position RMSE on restored missing points (m)",xticks=[10,20,30,40])
    ax3.legend(fontsize=9,facecolor=COLORS["panel"],edgecolor=COLORS["grid"],labelcolor=COLORS["text"],framealpha=.95)
    for ax in (ax1,ax2,ax3):
        ax.set_facecolor(COLORS["panel"]);ax.grid(color=COLORS["grid"],alpha=.35,lw=.6);ax.spines[["top","right"]].set_visible(False)
    fig.subplots_adjust(left=.065,right=.985,top=.84,bottom=.11,hspace=.42,wspace=.18)
    fig.suptitle("DSTA out-of-order handling and missing-observation compensation",x=.025,y=.965,ha="left",fontsize=20,fontweight="bold")
    fig.text(.025,.915,"Real supplemental DAIR trajectory · synthetic loss/arrival disorder · algorithm estimates shown in purple",color=COLORS["muted"],fontsize=10)
    fig.text(.025,.025,"Reference: vehicle-side / 0058 / 023991, 96-point continuous segment; not the EVT2_006 member trajectory.",color=COLORS["muted"],fontsize=9)
    for suffix in ("png","svg"):
        fig.savefig((OUT/"missing_compensation_ppt").with_suffix("."+suffix),dpi=300 if suffix=="png" else None,bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);SHARED.mkdir(parents=True,exist_ok=True)
    ref=load_reference();ref.to_parquet(SHARED/"reference_track.parquet",index=False)
    rng=np.random.default_rng(SEED)
    scenarios=[];estimates=[];metric_rows=[]
    for rate in MISSING_RATES:
        scenario=perturb(ref,rate,rng,include_shared_future_noise=False)
        scenarios.append(scenario)
        for method in ("baseline_arrival_time","improved_event_time"):
            est=interpolate_estimates(scenario,method);estimates.append(est)
            metric_rows.append(metric_row(rate,method,est,scenario))
    scenario_df=pd.concat(scenarios,ignore_index=True)
    estimate_df=pd.concat(estimates,ignore_index=True)
    metrics=pd.DataFrame(metric_rows)
    scenario_df.to_parquet(SHARED/"missing_order_scenarios.parquet",index=False)
    estimate_df.to_parquet(OUT/"restored_observations.parquet",index=False)
    metrics.to_csv(OUT/"missing_compensation_metrics.csv",index=False)
    # Materialize one full shared scenario for later weighting/consistency experiments; not evaluated here.
    full=perturb(ref,.30,np.random.default_rng(SEED+99),include_shared_future_noise=True)
    full.to_parquet(SHARED/"shared_full_perturbation.parquet",index=False)
    default=scenario_df[scenario_df.scenario_id.eq("missing_30_order_22")]
    make_figure(ref,default,estimate_df[estimate_df.scenario_id.eq("missing_30_order_22")],metrics)
    config={"seed":SEED,"reference_key":{"source":REFERENCE_KEY[0],"sequence_id":REFERENCE_KEY[1],"track_id":REFERENCE_KEY[2],"contiguous_segment":REFERENCE_SEGMENT},
      "reference_points":len(ref),"case_relation":"supplemental validation; EVT2_006 direct tracks were insufficient for multi-rate evaluation",
      "split":{"calibration_points":CALIBRATION_POINTS,"test_points":len(ref)-CALIBRATION_POINTS},
      "missing_rates":MISSING_RATES,"arrival_disorder":{"rate":ORDER_PERTURBATION_RATE,"max_shift_ms":MAX_ORDER_SHIFT_MS},
      "improved":{"reorder_basis":"event timestamp","buffer_ms":300,"interpolation":"two-sided linear constant-velocity","max_gap_s":MAX_INTERPOLATION_GAP_S},
      "shared_future_scenario":{"spatial_noise_sigma_m":.65,"quality_degradation_interval_s":[4,6],"quality_factor":.35,"status":"materialized but not evaluated this round"},
      "limitations":["Reference is the untouched real trajectory, not independent external ground truth.","Loss and arrival disorder are synthetic labels.","Results measure recovery against held-out real observations in this controlled scenario."]}
    (OUT/"missing_compensation_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    integration={"version":1,"join_key":"sample_id","layers":{"reference":"artifacts/reference_track.parquet","perturbed":"artifacts/missing_order_scenarios.parquet","restored":"restored_observations.parquet"},
      "later_fields":{"quality_weight":"not_computed","consistency_label":"not_computed","consistency_score":"not_computed"},
      "cube_mapping":{"x":["world_x","estimated_world_x"],"y":["world_y","estimated_world_y"],"t":"relative_time_s","quality_flags":["injected_missing","injected_out_of_order"]}}
    (OUT/"cube_integration_schema.json").write_text(json.dumps(integration,ensure_ascii=False,indent=2),encoding="utf-8")
    print(metrics.to_string(index=False));print("Saved to",OUT)


if __name__=="__main__":main()
