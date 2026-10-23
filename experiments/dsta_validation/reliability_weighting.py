"""Dynamic modality reliability weighting on a real trajectory with two simulated channels."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dsta-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiments/dsta_validation/shared_scenario/artifacts/reference_track.parquet"
OUT = ROOT / "outputs/ppt_case/dsta_validation/reliability_weighting"
SEEDS = list(range(20261023, 20261033))
CALIBRATION_POINTS = 20
DEGRADE_START_S, DEGRADE_END_S = 4.0, 6.0
SIGMA_A, SIGMA_B, SIGMA_B_DEGRADED = 0.35, 0.50, 3.00
BIAS_B_DEGRADED = np.array([2.0, -1.5])
DROPOUT_B_DEGRADED = 0.20
EWMA_ALPHA = 0.82
QUALITY_B_DEGRADED = 0.35
COLORS = {"blue":"#38BDF8","orange":"#FB923C","purple":"#C084FC","yellow":"#FACC15",
          "text":"#F1F5F9","muted":"#A8B8CC","grid":"#334A66","bg":"#080F1F","panel":"#101C30"}


def simulate_channels(ref: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(ref); degraded = ref.relative_time_s.between(DEGRADE_START_S, DEGRADE_END_S).to_numpy()
    sigma_b = np.where(degraded, SIGMA_B_DEGRADED, SIGMA_B)
    a = ref[["world_x","world_y"]].to_numpy() + rng.normal(0, SIGMA_A, (n,2))
    b = ref[["world_x","world_y"]].to_numpy() + rng.normal(0, sigma_b[:,None], (n,2))
    b[degraded] += BIAS_B_DEGRADED
    available_a = np.ones(n,dtype=bool); available_b = np.ones(n,dtype=bool)
    degraded_idx = np.flatnonzero(degraded & ref.split.eq("test").to_numpy())
    drop_n = int(round(DROPOUT_B_DEGRADED*len(degraded_idx)))
    if drop_n: available_b[rng.choice(degraded_idx,drop_n,replace=False)] = False
    b[~available_b] = np.nan
    out = ref[["sample_id","relative_time_s","split","world_x","world_y"]].copy()
    out[["channel_a_x","channel_a_y"]] = a; out[["channel_b_x","channel_b_y"]] = b
    out["available_a"] = available_a;out["available_b"] = available_b
    out["quality_degradation_flag"] = degraded
    out["configured_sigma_a_m"] = SIGMA_A;out["configured_sigma_b_m"] = sigma_b
    out["quality_signal_a"] = 1.0;out["quality_signal_b"] = np.where(degraded,QUALITY_B_DEGRADED,1.0)
    out["seed"] = seed
    return out


def dynamic_fuse(data: pd.DataFrame) -> pd.DataFrame:
    """Causal weighting: only current observations and past fused states are used."""
    result=data.copy();n=len(data)
    fused=np.full((n,2),np.nan);weights=np.zeros((n,2));innov=np.full((n,2),np.nan)
    variance=np.array([SIGMA_A**2,SIGMA_B**2],dtype=float)
    for i,row in enumerate(data.itertuples(index=False)):
        obs=np.array([[row.channel_a_x,row.channel_a_y],[row.channel_b_x,row.channel_b_y]],dtype=float)
        available=np.array([row.available_a,row.available_b]) & np.isfinite(obs).all(axis=1)
        if i>=2 and np.isfinite(fused[i-2:i]).all(): prediction=fused[i-1]+(fused[i-1]-fused[i-2])
        elif i>=1 and np.isfinite(fused[i-1]).all(): prediction=fused[i-1]
        else: prediction=np.nanmean(obs[available],axis=0)
        residual=np.linalg.norm(obs-prediction,axis=1);innov[i]=residual
        if i>0:
            for channel in range(2):
                if available[channel]: variance[channel]=EWMA_ALPHA*variance[channel]+(1-EWMA_ALPHA)*residual[channel]**2
        quality=np.array([row.quality_signal_a,row.quality_signal_b])
        reliability=np.where(available,quality/(variance+1e-4),0.0)
        if reliability.sum()==0: continue
        weights[i]=reliability/reliability.sum();fused[i]=np.nansum(obs*weights[i,:,None],axis=0)
    result[["dynamic_fused_x","dynamic_fused_y"]]=fused
    result["weight_a"]=weights[:,0];result["weight_b"]=weights[:,1]
    result["innovation_a_m"]=innov[:,0];result["innovation_b_m"]=innov[:,1]
    available_count=result.available_a.astype(int)+result.available_b.astype(int)
    result["equal_fused_x"]=(result.channel_a_x.fillna(0)+result.channel_b_x.fillna(0))/available_count
    result["equal_fused_y"]=(result.channel_a_y.fillna(0)+result.channel_b_y.fillna(0))/available_count
    # Non-deployable oracle: known injected variances, for theoretical context only.
    inv_a=np.where(result.available_a,1/SIGMA_A**2,0)
    inv_b=np.where(result.available_b,1/result.configured_sigma_b_m**2,0)
    wa=inv_a/(inv_a+inv_b);wb=1-wa
    result["oracle_fused_x"]=result.channel_a_x.fillna(0)*wa+result.channel_b_x.fillna(0)*wb
    result["oracle_fused_y"]=result.channel_a_y.fillna(0)*wa+result.channel_b_y.fillna(0)*wb
    return result


def errors(data: pd.DataFrame, x: str, y: str) -> np.ndarray:
    return np.sqrt((data[x]-data.world_x)**2+(data[y]-data.world_y)**2).to_numpy()


def metrics_for_seed(data: pd.DataFrame, seed: int) -> list[dict]:
    test=data[data.split.eq("test")].copy();rows=[]
    methods={"channel_a":("channel_a_x","channel_a_y"),"channel_b":("channel_b_x","channel_b_y"),
             "equal_weight":("equal_fused_x","equal_fused_y"),"dynamic_reliability":("dynamic_fused_x","dynamic_fused_y"),
             "oracle_non_deployable":("oracle_fused_x","oracle_fused_y")}
    for method,(x,y) in methods.items():
        err=errors(test,x,y)
        for phase,mask in [("overall",np.ones(len(test),dtype=bool)),("normal",~test.quality_degradation_flag.to_numpy()),
                           ("degraded",test.quality_degradation_flag.to_numpy())]:
            value=err[mask & np.isfinite(err)]
            rows.append({"seed":seed,"method":method,"phase":phase,"n":len(value),
                         "position_rmse_m":float(np.sqrt(np.mean(value**2))),"mean_error_m":float(np.mean(value)),
                         "median_error_m":float(np.median(value)),"p90_error_m":float(np.quantile(value,.9))})
    return rows


def integration_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for r in data.itertuples(index=False):
        for channel in ("a","b"):
            rows.append({"sample_id":r.sample_id,"relative_time_s":r.relative_time_s,
                "source_or_channel":f"simulated_channel_{channel.upper()}",
                "observed_world_x":getattr(r,f"channel_{channel}_x"),"observed_world_y":getattr(r,f"channel_{channel}_y"),
                "quality_weight":getattr(r,f"weight_{channel}"),"fused_world_x":r.dynamic_fused_x,
                "fused_world_y":r.dynamic_fused_y,"quality_degradation_flag":r.quality_degradation_flag,
                "provenance":"controlled_independent_channel_from_real_reference_track","split":r.split,"seed":r.seed})
    return pd.DataFrame(rows)


def make_figure(rep: pd.DataFrame, summary: pd.DataFrame):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"text.color":COLORS["text"],
      "axes.labelcolor":COLORS["text"],"axes.titlecolor":COLORS["text"],"xtick.color":COLORS["muted"],
      "ytick.color":COLORS["muted"],"axes.edgecolor":COLORS["grid"],"savefig.facecolor":COLORS["bg"]})
    fig=plt.figure(figsize=(13.333,7.5),facecolor=COLORS["bg"]);gs=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)];ax1,ax2,ax3,ax4=axes
    ax1.plot(rep.relative_time_s,rep.configured_sigma_a_m,color=COLORS["blue"],lw=2,label="Channel A configured σ")
    ax1.plot(rep.relative_time_s,rep.configured_sigma_b_m,color=COLORS["orange"],lw=2,label="Channel B configured σ")
    ax1.scatter(rep.loc[~rep.available_b,"relative_time_s"],np.zeros((~rep.available_b).sum()),marker="x",color=COLORS["orange"],label="B unavailable")
    ax1.axvspan(DEGRADE_START_S,DEGRADE_END_S,color=COLORS["orange"],alpha=.10);ax1.set(title="A  Controlled channel quality",xlabel="Relative time (s)",ylabel="Injected noise σ (m)")
    ax2.plot(rep.relative_time_s,rep.weight_a,color=COLORS["blue"],lw=2,label="Weight A")
    ax2.plot(rep.relative_time_s,rep.weight_b,color=COLORS["orange"],lw=2,label="Weight B")
    ax2.axvspan(DEGRADE_START_S,DEGRADE_END_S,color=COLORS["orange"],alpha=.10);ax2.set(title="B  Causal dynamic reliability weights",xlabel="Relative time (s)",ylabel="Normalized weight",ylim=(-.03,1.03))
    methods=["channel_a","channel_b","equal_weight","dynamic_reliability"]
    labels=["Channel A","Channel B","Equal","Dynamic"]
    colors=[COLORS["blue"],COLORS["orange"],COLORS["yellow"],COLORS["purple"]]
    normal=summary[(summary.phase.eq("normal"))&summary.method.isin(methods)].set_index("method").loc[methods]
    degraded=summary[(summary.phase.eq("degraded"))&summary.method.isin(methods)].set_index("method").loc[methods]
    x=np.arange(len(methods));w=.36
    ax3.bar(x-w/2,normal.rmse_mean,width=w,yerr=normal.rmse_std,color=colors,alpha=.58,label="Normal")
    ax3.bar(x+w/2,degraded.rmse_mean,width=w,yerr=degraded.rmse_std,color=colors,label="Quality degraded")
    ax3.set(title="C  Test RMSE across 10 fixed seeds",ylabel="Position RMSE (m)",xticks=x,xticklabels=labels)
    zoom=rep[rep.relative_time_s.between(3.7,6.3)]
    ax4.plot(zoom.world_x,zoom.world_y,color=COLORS["text"],lw=2.5,label="Hidden real reference")
    ax4.scatter(zoom.channel_a_x,zoom.channel_a_y,s=15,color=COLORS["blue"],alpha=.65,label="Sim channel A")
    ax4.scatter(zoom.channel_b_x,zoom.channel_b_y,s=15,color=COLORS["orange"],alpha=.65,label="Sim channel B")
    ax4.plot(zoom.dynamic_fused_x,zoom.dynamic_fused_y,color=COLORS["purple"],lw=2.2,label="Dynamic fusion")
    ax4.set(title="D  Representative degraded interval",xlabel="World X (m)",ylabel="World Y (m)")
    for ax in axes:
        ax.set_facecolor(COLORS["panel"]);ax.grid(color=COLORS["grid"],alpha=.35,lw=.6);ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=7.5,facecolor=COLORS["panel"],edgecolor=COLORS["grid"],labelcolor=COLORS["text"],framealpha=.95)
    fig.subplots_adjust(left=.065,right=.985,top=.84,bottom=.10,hspace=.38,wspace=.18)
    fig.suptitle("DSTA dynamic modality reliability weighting",x=.025,y=.965,ha="left",fontsize=20,fontweight="bold")
    fig.text(.025,.915,"Real reference trajectory · two independently simulated channels · reference excluded from weight computation",color=COLORS["muted"],fontsize=10)
    fig.text(.025,.025,"Shaded interval: synthetic Channel B quality degradation. Error bars: mean ± standard deviation across 10 fixed seeds.",color=COLORS["muted"],fontsize=9)
    for suffix in ("png","svg"):
        fig.savefig((OUT/"reliability_weighting_ppt").with_suffix("."+suffix),dpi=300 if suffix=="png" else None,bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);ref=pd.read_parquet(REFERENCE)
    if len(ref)!=96 or (ref.split.eq("calibration").sum()!=CALIBRATION_POINTS):raise RuntimeError("Reference split changed")
    all_points=[];rows=[]
    for seed in SEEDS:
        fused=dynamic_fuse(simulate_channels(ref,seed));all_points.append(fused);rows.extend(metrics_for_seed(fused,seed))
    points=pd.concat(all_points,ignore_index=True);metrics=pd.DataFrame(rows)
    summary=(metrics.groupby(["method","phase"],as_index=False).agg(rmse_mean=("position_rmse_m","mean"),rmse_std=("position_rmse_m","std"),mean_error_mean=("mean_error_m","mean"),median_error_mean=("median_error_m","mean"),p90_error_mean=("p90_error_m","mean"),seeds=("seed","nunique")))
    points.to_parquet(OUT/"all_seed_point_results.parquet",index=False);metrics.to_csv(OUT/"reliability_metrics_by_seed.csv",index=False);summary.to_csv(OUT/"reliability_metrics_summary.csv",index=False)
    representative=points[points.seed.eq(SEEDS[0])].copy();integration_rows(representative).to_parquet(OUT/"cube_reliability_layer.parquet",index=False)
    response=(representative[representative.split.eq("test")].groupby("quality_degradation_flag").agg(mean_weight_a=("weight_a","mean"),mean_weight_b=("weight_b","mean"),channel_b_availability=("available_b","mean")).reset_index())
    response.to_csv(OUT/"weight_response_summary.csv",index=False);make_figure(representative,summary)
    config={"scheme":"B","evaluation":"real trajectory reference with two controlled independent simulated channels","seeds":SEEDS,
      "reference":{"path":str(REFERENCE.relative_to(ROOT)),"points":len(ref),"calibration_points":CALIBRATION_POINTS,"test_points":len(ref)-CALIBRATION_POINTS,"used_by_weight_algorithm":False},
      "channels":{"A":{"sigma_m":SIGMA_A},"B":{"normal_sigma_m":SIGMA_B,"degraded_sigma_m":SIGMA_B_DEGRADED,"degraded_bias_m":BIAS_B_DEGRADED.tolist(),"degraded_dropout_rate":DROPOUT_B_DEGRADED,"quality_signal":QUALITY_B_DEGRADED}},
      "degraded_interval_s":[DEGRADE_START_S,DEGRADE_END_S],"dynamic_weight":{"ewma_alpha":EWMA_ALPHA,"inputs":["past fused states","current channel observations","availability","current quality signal"],"future_data_used":False,"reference_used":False},
      "oracle":{"included":True,"deployable":False,"uses_known_injected_variance":True},
      "limitations":["Channels are simulated, not real cross-source fusion.","Real reference is hidden from weighting but is not an independent survey-grade ground truth.","Conclusions apply only to the controlled noise configuration."]}
    (OUT/"reliability_weighting_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding="utf-8")
    print(summary.to_string(index=False));print("\nWeight response\n",response.to_string(index=False));print("Saved to",OUT)


if __name__=="__main__":main()
