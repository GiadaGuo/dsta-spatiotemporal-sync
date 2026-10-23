"""Interactive DSTA unified spatiotemporal data cube (technology track 1)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dsta.cube_data import build_cube_tables

BG, PANEL, TEXT, MUTED = "#080F1F", "#101C30", "#F1F5F9", "#A8B8CC"
BLUE, ORANGE, PURPLE = "#38BDF8", "#FB923C", "#C084FC"

st.set_page_config(page_title="DSTA · 时空数据立方体", page_icon="🧊", layout="wide")
st.html(f"""<style>
.stApp {{background:{BG}}}
.block-container {{max-width:1780px}}
[data-testid="stMetric"] {{background:#15243B;border:1px solid #2B405E;border-radius:12px;padding:.7rem 1rem}}
</style>""")


@st.cache_data(show_spinner=False, max_entries=2)
def load_cube():
    obs, cube = build_cube_tables(ROOT)
    grid = pd.read_csv(ROOT / "data/processed/optimized_grid_cells.csv")
    events = pd.read_csv(ROOT / "outputs/reports/detected_events_v2.csv")
    timing = ROOT / "outputs/ppt_case/time_alignment/time_alignment_metrics.csv"
    timing_metrics = pd.read_csv(timing) if timing.exists() else pd.DataFrame()
    return obs, cube, grid, events, timing_metrics


def grid_trace(grid, z=0.0):
    x, y, zz = [], [], []
    for row in grid.itertuples(index=False):
        corners = [(row.x0, row.y0), (row.x0+row.size, row.y0),
                   (row.x0+row.size, row.y0+row.size), (row.x0, row.y0+row.size),
                   (row.x0, row.y0)]
        for px, py in corners:
            x.append(px); y.append(py); zz.append(z)
        x.append(None); y.append(None); zz.append(None)
    return go.Scatter3d(x=x, y=y, z=zz, mode="lines", name="自适应空间网格",
                        line=dict(color="#516B88", width=2), opacity=.42, hoverinfo="skip")


def cube_figure(cube, grid, metric, selected_window, x_range, y_range):
    specs = {
        "观测数量": ("observation_count", "Turbo", "条"),
        "平均速度": ("mean_speed_mps", "Plasma", "m/s"),
        "来源覆盖": ("source_coverage_count", [[0, "#52657D"], [.5, BLUE], [1, ORANGE]], "个来源"),
        "质量状态": ("interpolated_count", [[0, BLUE], [1, PURPLE]], "插值记录"),
    }
    col, scale, unit = specs[metric]
    view = cube[cube.mean_world_x.between(*x_range) & cube.mean_world_y.between(*y_range)].copy()
    values = pd.to_numeric(view[col], errors="coerce")
    sizes = 4 + 9 * np.sqrt(view.observation_count.clip(lower=1) / view.observation_count.max())
    custom = np.stack([view.window_id.astype(str), view.grid_id.astype(str),
                       view.observation_count.astype(str), view.mean_speed_mps.round(2).astype(str),
                       view.source_coverage_count.astype(str), view.quality_flags.astype(str)], axis=-1)
    fig = go.Figure([grid_trace(grid)])
    fig.add_trace(go.Scatter3d(
        x=view.mean_world_x, y=view.mean_world_y, z=view.event_time_s,
        mode="markers", name="网格窗口状态", customdata=custom,
        marker=dict(size=sizes, color=values, colorscale=scale, opacity=.70,
                    colorbar=dict(title=f"{metric}<br>{unit}", thickness=14)),
        hovertemplate=("窗口=%{customdata[0]}<br>grid=%{customdata[1]}"
                       "<br>观测=%{customdata[2]}<br>速度=%{customdata[3]} m/s"
                       "<br>来源覆盖=%{customdata[4]}<br>质量=%{customdata[5]}"
                       "<br>X=%{x:.2f} m · Y=%{y:.2f} m · T=%{z:.1f} s<extra></extra>")))
    members = view[view.event_id.eq("EVT2_006")]
    if not members.empty:
        fig.add_trace(go.Scatter3d(
            x=members.mean_world_x, y=members.mean_world_y, z=members.event_time_s,
            mode="markers+lines", name="EVT2_006 下游验证成员",
            marker=dict(size=10, color=PURPLE, symbol="diamond", line=dict(color=TEXT, width=2)),
            line=dict(color=PURPLE, width=5),
            hovertemplate="EVT2_006<br>window=%{customdata[0]}<extra></extra>",
            customdata=np.stack([members.window_id.astype(str)], axis=-1)))
    z = float(selected_window * 2)
    fig.add_trace(go.Surface(
        x=[[x_range[0], x_range[1]], [x_range[0], x_range[1]]],
        y=[[y_range[0], y_range[0]], [y_range[1], y_range[1]]], z=[[z, z], [z, z]],
        colorscale=[[0, BLUE], [1, BLUE]], showscale=False, opacity=.10,
        name=f"窗口 {selected_window}", hoverinfo="skip"))
    fig.update_layout(
        height=690, margin=dict(l=0, r=0, t=35, b=0), paper_bgcolor=BG,
        font_color=TEXT, legend=dict(bgcolor="rgba(8,15,31,.75)", orientation="h"),
        scene=dict(bgcolor=BG, aspectmode="manual", aspectratio=dict(x=1, y=1, z=.75),
            xaxis=dict(title="World X (m)", backgroundcolor=PANEL, gridcolor="#334A66", range=x_range),
            yaxis=dict(title="World Y (m)", backgroundcolor=PANEL, gridcolor="#334A66", range=y_range),
            zaxis=dict(title="Relative event time T (s)", backgroundcolor=PANEL, gridcolor="#334A66", range=[0, 24])),
        uirevision=f"{metric}-{x_range[0]}-{y_range[0]}")
    return fig, view


obs, cube, grid, events, timing_metrics = load_cube()
event = events[events.event_id.eq("EVT2_006")].iloc[0]
st.title("DSTA · 统一时空数据立方体")
st.caption("X/Y 为 DAIR 数据集真实世界坐标；T 为相对事件时间，不是地理高度。第一版只展示已存在或可严格聚合的数据字段。")

with st.container(horizontal=True):
    st.metric("观测级记录", f"{len(obs):,}", border=True)
    st.metric("网格窗口状态", f"{len(cube):,}", border=True)
    st.metric("时间切片", f"{cube.window_id.nunique()}", border=True)
    st.metric("自适应网格", f"{len(grid):,}", border=True)

st.subheader("DSTA 技术链与当前证据状态")
status_cols = st.columns(6)
statuses = [
    ("1 时间对齐", "已完成 · 真实帧对 + 仿真扰动", BLUE),
    ("2 空间映射", "已完成 · 真实测试集", BLUE),
    ("3 乱序缺失补偿", "待验证", ORANGE),
    ("4 可信度动态加权", "待验证", ORANGE),
    ("5 一致性联合校验", "待验证", ORANGE),
    ("6 统一数据流", "第一版 Schema 已构建", PURPLE),
]
for col, (name, status, color) in zip(status_cols, statuses):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{name}**")
        st.markdown(f":{('blue' if color == BLUE else 'orange' if color == ORANGE else 'violet')}-badge[{status}]")

st.warning("立方体不是五项算法均已完成的声明：补偿、动态权重与联合一致性仍缺正式验证证据。", icon=":material/science:")

control, control2, control3 = st.columns([1, 1, 1.4], vertical_alignment="bottom")
metric = control.segmented_control("体素图层", ["观测数量", "平均速度", "来源覆盖", "质量状态"], default="观测数量")
focus = control2.segmented_control("空间视角", ["全局", "EVT2_006 局部"], default="EVT2_006 局部")
selected_window = control3.slider("突出时间切片", 0, 12, 4)
if focus == "EVT2_006 局部":
    span = 40.0
    x_range = [float(event.centroid_x)-span/2, float(event.centroid_x)+span/2]
    y_range = [float(event.centroid_y)-span/2, float(event.centroid_y)+span/2]
    shown_grid = grid[(grid.x0 + grid["size"] >= x_range[0]) & (grid.x0 <= x_range[1]) &
                      (grid.y0 + grid["size"] >= y_range[0]) & (grid.y0 <= y_range[1])]
else:
    x_range = [float(grid.x0.min()), float((grid.x0 + grid["size"]).max())]
    y_range = [float(grid.y0.min()), float((grid.y0 + grid["size"]).max())]
    shown_grid = grid

fig, shown = cube_figure(cube, shown_grid, metric or "观测数量", selected_window, x_range, y_range)
st.plotly_chart(fig, width="stretch", theme=None, key="dsta-cube",
                alt="DSTA 世界坐标和相对事件时间构成的三维网格状态立方体")
st.caption(f"当前显示 {len(shown):,}/{len(cube):,} 个完整网格窗口状态；未做统计抽样。标记大小表示观测量，半透明平面表示窗口 {selected_window}。")

view = st.segmented_control("补充视图", ["Before / After", "统一数据 Schema", "当前切片数据"], default="Before / After")
if view == "Before / After":
    left, right = st.columns(2)
    with left.container(border=True):
        st.subheader("Before · 异步多源观测")
        st.markdown("- 两个真实来源：`vehicle-side`、`infrastructure-side`\n- 原始事件时间与到达时间字段并存\n- 轨迹 ID 只在来源和序列内有效\n- 网络延迟实验属于明确标注的仿真扰动")
        st.dataframe(obs[["source", "sequence_id", "frame_id", "timestamp_us", "arrival_timestamp_us",
                          "world_x", "world_y", "quality_flags"]].head(12), hide_index=True, width="stretch")
    with right.container(border=True):
        st.subheader("After · 统一时空结构")
        st.markdown("- 统一 `window_id` 与相对时间轴\n- 统一 `grid_id` 与世界坐标\n- 保留来源覆盖、质量标记与 provenance\n- 观测级和网格窗口级严格分表")
        if not timing_metrics.empty:
            st.dataframe(timing_metrics[["method", "coverage_rate", "correct_pair_rate", "p90_time_error_ms"]], hide_index=True, width="stretch")
elif view == "统一数据 Schema":
    st.info("完整字段来源和粒度说明见 outputs/ppt_case/cube/unified_data_schema.md。", icon=":material/schema:")
    st.dataframe(pd.DataFrame({"数据集": ["unified_observations", "unified_cube_states"],
        "粒度": ["单条来源观测", "单个时间窗口 × 单个网格"],
        "主键": ["source + sequence_id + frame_id + track_id", "window_id + grid_id"],
        "记录数": [len(obs), len(cube)]}), hide_index=True, width="stretch")
else:
    slice_data = shown[shown.window_id.eq(selected_window)].sort_values("observation_count", ascending=False)
    st.dataframe(slice_data[["window_id", "grid_id", "mean_world_x", "mean_world_y", "observation_count",
                             "mean_speed_mps", "source_coverage_count", "quality_flags", "event_id"]],
                 hide_index=True, width="stretch")

st.caption("下游 EVT2_006 仅作为数据立方体可支持事件智能感知的验证场景；AE、事件聚合和风险排序不属于技术攻关 1 核心算法。")
