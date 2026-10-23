"""Interactive DSTA unified data cube and frozen validation views."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsta.cube_data import build_cube_tables

BG, PANEL, TEXT = "#080F1F", "#101C30", "#F1F5F9"
BLUE, ORANGE, PURPLE, YELLOW = "#38BDF8", "#FB923C", "#C084FC", "#FACC15"

st.set_page_config(page_title="DSTA · 时空数据立方体", page_icon="🧊", layout="wide")
st.html(f"""<style>
.stApp {{background:{BG}}}
.block-container {{max-width:1780px;padding-top:2.2rem}}
[data-testid="stMetric"] {{background:#15243B;border:1px solid #2B405E;border-radius:12px;padding:.7rem 1rem}}
</style>""")


@st.cache_data(show_spinner=False, max_entries=2)
def load_all_data():
    obs, cube = build_cube_tables(ROOT)
    base = ROOT / "outputs/ppt_case/dsta_validation"
    return {
        "obs": obs, "cube": cube,
        "grid": pd.read_csv(ROOT / "data/processed/optimized_grid_cells.csv"),
        "events": pd.read_csv(ROOT / "outputs/reports/detected_events_v2.csv"),
        "timing": pd.read_csv(ROOT / "outputs/ppt_case/time_alignment/time_alignment_metrics.csv"),
        "spatial": pd.read_csv(ROOT / "outputs/ppt_case/spatial_mapping/spatial_mapping_metrics.csv"),
        "missing_metrics": pd.read_csv(base / "missing_compensation/missing_compensation_metrics.csv"),
        "missing_layer": pd.read_parquet(base / "missing_compensation/restored_observations.parquet"),
        "perturbed": pd.read_parquet(ROOT / "experiments/dsta_validation/shared_scenario/artifacts/missing_order_scenarios.parquet"),
        "reliability_metrics": pd.read_csv(base / "reliability_weighting/reliability_metrics_summary.csv"),
        "reliability_layer": pd.read_parquet(base / "reliability_weighting/cube_reliability_layer.parquet"),
        "consistency_metrics": pd.read_csv(base / "consistency_validation/consistency_metrics.csv"),
        "consistency_layer": pd.read_parquet(base / "consistency_validation/cube_consistency_layer.parquet"),
    }


def grid_trace(grid: pd.DataFrame, z: float = 0.0) -> go.Scatter3d:
    x, y, zz = [], [], []
    for row in grid.itertuples(index=False):
        corners = [(row.x0, row.y0), (row.x0 + row.size, row.y0),
                   (row.x0 + row.size, row.y0 + row.size), (row.x0, row.y0 + row.size), (row.x0, row.y0)]
        for px_, py_ in corners:
            x.append(px_); y.append(py_); zz.append(z)
        x.append(None); y.append(None); zz.append(None)
    return go.Scatter3d(x=x, y=y, z=zz, mode="lines", name="自适应空间网格",
                        line=dict(color="#516B88", width=2), opacity=.42, hoverinfo="skip")


def cube_figure(cube, grid, metric, selected_window, x_range, y_range):
    specs = {"观测数量": ("observation_count", "Turbo", "条"),
             "平均速度": ("mean_speed_mps", "Plasma", "m/s"),
             "来源覆盖": ("source_coverage_count", [[0, "#52657D"], [.5, BLUE], [1, ORANGE]], "个来源"),
             "质量状态": ("interpolated_count", [[0, BLUE], [1, PURPLE]], "插值记录")}
    col, scale, unit = specs[metric]
    view = cube[cube.mean_world_x.between(*x_range) & cube.mean_world_y.between(*y_range)].copy()
    values = pd.to_numeric(view[col], errors="coerce")
    sizes = 4 + 9 * np.sqrt(view.observation_count.clip(lower=1) / max(float(view.observation_count.max()), 1.0))
    custom = np.stack([view.window_id.astype(str), view.grid_id.astype(str), view.observation_count.astype(str),
                       view.mean_speed_mps.round(2).astype(str), view.source_coverage_count.astype(str),
                       view.quality_flags.astype(str)], axis=-1)
    fig = go.Figure([grid_trace(grid)])
    fig.add_trace(go.Scatter3d(x=view.mean_world_x, y=view.mean_world_y, z=view.event_time_s,
        mode="markers", name="网格窗口状态", customdata=custom,
        marker=dict(size=sizes, color=values, colorscale=scale, opacity=.70,
                    colorbar=dict(title=f"{metric}<br>{unit}", thickness=14)),
        hovertemplate=("窗口=%{customdata[0]}<br>grid=%{customdata[1]}<br>观测=%{customdata[2]}"
                       "<br>速度=%{customdata[3]} m/s<br>来源覆盖=%{customdata[4]}<br>质量=%{customdata[5]}"
                       "<br>X=%{x:.2f} m · Y=%{y:.2f} m · T=%{z:.1f} s<extra></extra>")))
    members = view[view.event_id.eq("EVT2_006")]
    if not members.empty:
        fig.add_trace(go.Scatter3d(x=members.mean_world_x, y=members.mean_world_y, z=members.event_time_s,
            mode="markers+lines", name="EVT2_006 下游验证成员",
            marker=dict(size=10, color=PURPLE, symbol="diamond", line=dict(color=TEXT, width=2)),
            line=dict(color=PURPLE, width=5), customdata=np.stack([members.window_id.astype(str)], axis=-1),
            hovertemplate="EVT2_006<br>window=%{customdata[0]}<extra></extra>"))
    z = float(selected_window * 2)
    fig.add_trace(go.Surface(x=[[x_range[0], x_range[1]], [x_range[0], x_range[1]]],
        y=[[y_range[0], y_range[0]], [y_range[1], y_range[1]]], z=[[z, z], [z, z]],
        colorscale=[[0, BLUE], [1, BLUE]], showscale=False, opacity=.10, hoverinfo="skip"))
    fig.update_layout(height=690, margin=dict(l=0, r=0, t=35, b=0), paper_bgcolor=BG,
        font_color=TEXT, legend=dict(bgcolor="rgba(8,15,31,.75)", orientation="h"),
        scene=dict(bgcolor=BG, aspectmode="manual", aspectratio=dict(x=1, y=1, z=.75),
            xaxis=dict(title="World X (m)", backgroundcolor=PANEL, gridcolor="#334A66", range=x_range),
            yaxis=dict(title="World Y (m)", backgroundcolor=PANEL, gridcolor="#334A66", range=y_range),
            zaxis=dict(title="Relative event time T (s)", backgroundcolor=PANEL, gridcolor="#334A66", range=[0, 24])),
        uirevision=f"{metric}-{x_range[0]}-{y_range[0]}")
    return fig, view


def dark_figure(fig, height=500):
    fig.update_layout(height=height, paper_bgcolor=BG, plot_bgcolor=PANEL, font_color=TEXT,
                      margin=dict(l=30, r=20, t=55, b=30), legend=dict(bgcolor="rgba(8,15,31,.75)"))
    fig.update_xaxes(gridcolor="#334A66"); fig.update_yaxes(gridcolor="#334A66")
    return fig


data = load_all_data()
obs, cube, grid, events = data["obs"], data["cube"], data["grid"], data["events"]
event = events[events.event_id.eq("EVT2_006")].iloc[0]

st.title("DSTA · 统一时空数据流与算法验证")
st.caption("真实统一数据、受控扰动实验和算法估计采用独立场景展示；任何实验层都不会覆盖真实统一数据表。")
scene = st.segmented_control("展示场景", ["真实数据立方体", "乱序缺失补偿", "可信度动态加权", "一致性联合校验", "五项指标概览"],
                             default="真实数据立方体", key="scene")

if scene == "真实数据立方体":
    st.info("本场景只读取 8,327 条真实观测和 830 条真实网格窗口状态；X/Y 是世界坐标，T 是相对事件时间而非高度。", icon=":material/database:")
    with st.container(horizontal=True):
        st.metric("观测级记录", f"{len(obs):,}", border=True)
        st.metric("网格窗口状态", f"{len(cube):,}", border=True)
        st.metric("时间切片", f"{cube.window_id.nunique()}", border=True)
        st.metric("自适应网格", f"{len(grid):,}", border=True)
    c1, c2, c3 = st.columns([1, 1, 1.4], vertical_alignment="bottom")
    metric = c1.segmented_control("体素图层", ["观测数量", "平均速度", "来源覆盖", "质量状态"], default="观测数量", key="cube_metric")
    focus = c2.segmented_control("空间视角", ["全局", "EVT2_006 局部"], default="EVT2_006 局部", key="cube_focus")
    selected_window = c3.slider("突出时间切片", 0, 12, 4, key="cube_window")
    if focus == "EVT2_006 局部":
        span = 40.0
        x_range = [float(event.centroid_x) - span / 2, float(event.centroid_x) + span / 2]
        y_range = [float(event.centroid_y) - span / 2, float(event.centroid_y) + span / 2]
        shown_grid = grid[(grid.x0 + grid["size"] >= x_range[0]) & (grid.x0 <= x_range[1]) &
                          (grid.y0 + grid["size"] >= y_range[0]) & (grid.y0 <= y_range[1])]
    else:
        x_range = [float(grid.x0.min()), float((grid.x0 + grid["size"]).max())]
        y_range = [float(grid.y0.min()), float((grid.y0 + grid["size"]).max())]
        shown_grid = grid
    fig, shown = cube_figure(cube, shown_grid, metric or "观测数量", selected_window, x_range, y_range)
    st.plotly_chart(fig, width="stretch", theme=None, key="dsta-cube", alt="真实世界坐标和相对事件时间构成的三维网格状态立方体")
    st.caption(f"显示 {len(shown):,}/{len(cube):,} 个完整网格窗口状态，未做统计抽样。EVT2_006 仅为下游候选事件验证，不是已确认事故。")
    detail = st.segmented_control("数据说明", ["Before / After", "统一数据 Schema", "当前切片数据"], default="Before / After", key="real_detail")
    if detail == "Before / After":
        left, right = st.columns(2)
        with left.container(border=True):
            st.subheader("Before · 异步多源观测")
            st.markdown("两个真实来源为 `vehicle-side` 与 `infrastructure-side`；轨迹 ID 只在来源与序列内有效。")
            st.dataframe(obs[["source", "sequence_id", "frame_id", "timestamp_us", "arrival_timestamp_us", "world_x", "world_y", "quality_flags"]].head(12), hide_index=True)
        with right.container(border=True):
            st.subheader("After · 统一时空结构")
            st.markdown("统一 `window_id`、`grid_id`、世界坐标与 provenance；观测级和网格窗口级严格分表。")
            st.dataframe(data["timing"][["method", "coverage_rate", "correct_pair_rate", "p90_time_error_ms"]], hide_index=True)
    elif detail == "统一数据 Schema":
        st.info("完整字段来源见 outputs/ppt_case/cube/unified_data_schema.md。", icon=":material/schema:")
        st.dataframe(pd.DataFrame({"数据集": ["unified_observations", "unified_cube_states"], "粒度": ["单条来源观测", "时间窗口 × 网格"],
            "主键": ["source + sequence_id + frame_id + track_id", "window_id + grid_id"], "记录数": [len(obs), len(cube)]}), hide_index=True)
    else:
        slice_data = shown[shown.window_id.eq(selected_window)].sort_values("observation_count", ascending=False)
        st.dataframe(slice_data[["window_id", "grid_id", "mean_world_x", "mean_world_y", "observation_count", "mean_speed_mps", "source_coverage_count", "quality_flags", "event_id"]], hide_index=True)

elif scene == "乱序缺失补偿":
    st.warning("受控验证：参考为同一 DAIR 序列的 96 点真实车端轨迹，不是 EVT2_006 成员；缺失率和到达乱序均为 synthetic injection。", icon=":material/science:")
    rate = st.segmented_control("注入缺失率", [10, 20, 30, 40], default=20, format_func=lambda x: f"{x}%", key="missing_rate")
    scenario_id = f"missing_{rate}_order_22"
    pert = data["perturbed"][data["perturbed"].scenario_id.eq(scenario_id)]
    restored = data["missing_layer"][data["missing_layer"].scenario_id.eq(scenario_id)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pert.relative_time_s, y=pert.world_y, mode="lines", name="真实参考轨迹 Y", line=dict(color=TEXT, width=3)))
    visible = pert[~pert.injected_missing]
    fig.add_trace(go.Scatter(x=visible.relative_time_s, y=visible.world_y_perturbed, mode="markers", name="受扰后可用观测", marker=dict(color=ORANGE, size=6)))
    for method, color, label in [("baseline_arrival_time", YELLOW, "Baseline 恢复"), ("improved_event_time", PURPLE, "DSTA 恢复")]:
        part = restored[restored.method.eq(method)].sort_values("relative_time_s")
        fig.add_trace(go.Scatter(x=part.relative_time_s, y=part.estimated_world_y, mode="markers", name=label, marker=dict(color=color, size=9, symbol="diamond")))
    st.plotly_chart(dark_figure(fig.update_layout(title=f"参考、受扰与恢复轨迹 · 缺失 {rate}%", xaxis_title="相对时间 (s)", yaxis_title="World Y (m)")),
                    width="stretch", theme=None, alt="真实参考轨迹与受控缺失后的两种恢复估计")
    rows = data["missing_metrics"][data["missing_metrics"].missing_rate.eq(rate / 100)]
    st.dataframe(rows[["method", "injected_missing_points", "arrival_order_inversions", "post_reorder_inversions", "recovery_coverage", "position_rmse_m"]], hide_index=True)

elif scene == "可信度动态加权":
    st.warning("受控双通道仿真：两条观测通道由一段 96 点真实轨迹独立加噪生成，不是真实车路双源融合；真实参考不参与权重计算。", icon=":material/science:")
    layer = data["reliability_layer"]
    a = layer[layer.source_or_channel.eq("simulated_channel_A")].sort_values("relative_time_s")
    b = layer[layer.source_or_channel.eq("simulated_channel_B")].sort_values("relative_time_s")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a.relative_time_s, y=a.quality_weight, name="仿真通道 A 权重", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=b.relative_time_s, y=b.quality_weight, name="仿真通道 B 权重", line=dict(color=ORANGE, width=3)))
    fig.add_vrect(x0=4, x1=6, fillcolor=ORANGE, opacity=.12, line_width=0, annotation_text="B 质量退化")
    st.plotly_chart(dark_figure(fig.update_layout(title="因果动态可信度权重", xaxis_title="相对时间 (s)", yaxis_title="归一化权重", yaxis_range=[0, 1])),
                    width="stretch", theme=None, alt="受控双通道质量退化期间的动态可信度权重")
    summary = data["reliability_metrics"]
    shown = summary[(summary.phase.isin(["overall", "degraded"])) & summary.method.isin(["channel_a", "channel_b", "equal_weight", "dynamic_reliability"])]
    st.dataframe(shown[["method", "phase", "rmse_mean", "rmse_std", "median_error_mean", "seeds"]], hide_index=True)
    st.caption("10 个固定随机种子下：质量退化阶段等权融合 RMSE 2.150 m，动态加权 0.508 m；结论仅适用于当前受控噪声配置。")

elif scene == "一致性联合校验":
    st.warning("815 条是官方帧对内的候选跨源匹配，不是身份真值；当前页面显示独立测试副本上的 synthetic anomaly labels。", icon=":material/science:")
    layer = data["consistency_layer"]
    colors = {"normal": BLUE, "time_offset": ORANGE, "spatial_drift": YELLOW, "trajectory_break": PURPLE, "combined": "#F472B6"}
    fig = px.scatter(layer, x="evaluated_time_residual_ms", y="evaluated_space_residual_m", color="anomaly_type",
        color_discrete_map=colors, hover_data=["match_id", "consistency_score", "consistency_label", "vehicle_frame", "infrastructure_frame"],
        labels={"evaluated_time_residual_ms": "时间残差 (ms)", "evaluated_space_residual_m": "空间残差 (m)", "anomaly_type": "异常类型"},
        title="独立测试集：实测残差基础上的中等强度受控异常")
    st.plotly_chart(dark_figure(fig), width="stretch", theme=None, alt="候选匹配测试集的时间和空间残差及受控异常类型")
    medium = data["consistency_metrics"][data["consistency_metrics"].strength.eq("medium")]
    st.dataframe(medium[["method", "precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn"]], hide_index=True)
    st.info("必须同时解读：时间/空间 OR 的 Recall 0.776、F1 0.832、FPR 0.130；多维联合的 FPR 降至 0.022，但 Recall 降至 0.529、F1 降至 0.686。联合方法更保守，并非所有指标全面提升。", icon=":material/balance:")

else:
    st.subheader("五项核心算法 · 冻结量化结果")
    st.caption("各卡片来自不同冻结实验，样本和真实性口径不同；仅用于同一技术链的证据概览，不做跨实验横向排名。")
    timing, spatial = data["timing"].set_index("method"), data["spatial"].set_index("display_name")
    missing, reliability, consistency = data["missing_metrics"], data["reliability_metrics"], data["consistency_metrics"]
    cards = [
        ("1 自适应时间对齐", "98.02%", "正确配对率", "101 组真实帧对 + 延迟/抖动仿真", f"P90 {timing.loc['Adaptive event-time window','p90_time_error_ms']:.2f} ms"),
        ("2 统一空间映射", "8.60 m", "平均量化误差", "1,235 条真实冻结测试观测", f"{int(spatial.loc['Quality-aware','n_cells'])}/≤200 网格"),
        ("3 乱序缺失补偿", f"{missing[(missing.missing_rate.eq(.4)) & missing.method.eq('improved_event_time')].iloc[0].position_rmse_m:.3f} m", "40% 缺失 RMSE", "96 点真实补充轨迹 + 受控缺失乱序", "恢复覆盖率 100%"),
        ("4 动态可信度加权", f"{reliability[(reliability.method.eq('dynamic_reliability')) & reliability.phase.eq('degraded')].iloc[0].rmse_mean:.3f} m", "退化阶段 RMSE", "真实参考轨迹上的双通道仿真", "10 个固定随机种子"),
        ("5 联合一致性校验", f"{consistency[(consistency.strength.eq('medium')) & consistency.method.eq('joint_multidim')].iloc[0].fpr:.3f}", "中强度 FPR", "815 条候选匹配 + 测试集异常注入", "Recall 0.529 · F1 0.686"),
    ]
    for start in (0, 3):
        cols = st.columns(3)
        for col, card in zip(cols, cards[start:start + 3]):
            with col.container(border=True, height="stretch"):
                st.markdown(f"### {card[0]}"); st.metric(card[2], card[1]); st.caption(card[3]); st.markdown(f":violet-badge[{card[4]}]")
    st.success("DSTA 已将异步、异构、质量不一致的城市感知观测组织为可追溯、可校验、可交互的统一时空数据流。", icon=":material/check_circle:")

st.divider()
st.caption("技术攻关 1 展示边界：EVT2_006 是下游候选事件验证；AE、事件聚合和风险排序不属于 DSTA 核心算法。相对优先级不是事故概率。")
