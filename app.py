"""DSTA Step 8.1: real-data urban traffic situation replay."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

st.set_page_config(page_title="DSTA · 城市交通态势", page_icon="🚦", layout="wide")
ROOT = Path(__file__).resolve().parent
PROC, REPORT = ROOT / "data/processed", ROOT / "outputs/reports"
BG, CARD, TEXT, MUTED = "#080F1F", "#15243B", "#F1F5F9", "#A8B8CC"
BLUE, ORANGE, PURPLE, GRID = "#38BDF8", "#FB923C", "#C084FC", "#52657D"
SOURCE_STYLE = {
    "vehicle-side": (BLUE, "circle", "车端观测"),
    "infrastructure-side": (ORANGE, "diamond", "路侧观测"),
}
st.html(f"""<style>
.stApp {{background:{BG}}}
.block-container {{max-width:1780px}}
[data-testid="stMetric"] {{background:{CARD};border:1px solid #2B405E;border-radius:12px;padding:.85rem 1rem}}
[data-testid="stMetricLabel"],[data-testid="stMetricDelta"] {{color:{MUTED}}}
[data-testid="stMetricValue"] {{color:{TEXT}}}
</style>""")


@st.cache_data(show_spinner=False, max_entries=2)
def load_data():
    paths = {
        "observations": PROC / "spatiotemporal_events.parquet",
        "states": PROC / "city_state.parquet",
        "anomalies": PROC / "embedding_anomaly_states.parquet",
        "grid": PROC / "optimized_grid_cells.csv",
        "events": REPORT / "detected_events_v2.csv",
        "members": REPORT / "event_members_v2.csv",
        "evolution": REPORT / "event_evolution.csv",
        "priorities": REPORT / "event_risk_priorities.csv",
        "traces": REPORT / "event_observation_traces.csv",
        "graph": REPORT / "event_evidence_graph.json",
        "cognition": REPORT / "event_cognition.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {}, missing
    result = {}
    for key, path in paths.items():
        if path.suffix == ".parquet":
            result[key] = pd.read_parquet(path)
        elif path.suffix == ".csv":
            result[key] = pd.read_csv(path)
        else:
            result[key] = json.loads(path.read_text(encoding="utf-8"))
    return result, []


@st.cache_data(show_spinner=False, max_entries=2)
def prepare_data(data):
    result = {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in data.items()}
    states, obs = result["states"], result["observations"]
    starts = states[["window_id", "window_start_us"]].drop_duplicates().sort_values("window_start_us")
    start_values = starts.window_start_us.to_numpy(dtype="int64")
    start_windows = starts.window_id.to_numpy(dtype="int64")
    positions = np.searchsorted(start_values, obs.timestamp_us.to_numpy(dtype="int64"), side="right") - 1
    valid = positions >= 0
    obs["window_id"] = pd.array([pd.NA] * len(obs), dtype="Int64")
    obs.loc[valid, "window_id"] = start_windows[positions[valid]]
    obs.sort_values(["source", "sequence_id", "track_id", "timestamp_us"], inplace=True)
    traces = result["traces"]
    if "event_id_x" in traces and "event_id" not in traces:
        traces.rename(columns={"event_id_x": "event_id", "event_id_y": "observation_id"}, inplace=True)
    return result


def style_figure(fig, height=560):
    fig.update_layout(
        height=height, margin=dict(l=12, r=12, t=36, b=10), paper_bgcolor=BG,
        plot_bgcolor="#0D192A", font=dict(color=TEXT),
        hoverlabel=dict(bgcolor=CARD, font_color=TEXT),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(8,15,31,.78)"),
        xaxis=dict(title="World X (m)", gridcolor="#21334D", zeroline=False),
        yaxis=dict(title="World Y (m)", gridcolor="#21334D", zeroline=False,
                   scaleanchor="x", scaleratio=1),
    )


def add_grid(fig, grid, active, mode):
    indexed = active.set_index("grid_id") if not active.empty else pd.DataFrame()
    value_col = {"观测数量": "observation_count", "平均速度": "mean_speed_mps",
                 "异常分布": "observation_density"}[mode]
    values = pd.to_numeric(active.get(value_col, pd.Series(dtype=float)), errors="coerce")
    lo, hi = ((float(values.min()), float(values.max()))
              if values.notna().any() else (0.0, 1.0))
    hi = max(hi, lo + 1.0)
    scale = [[0, "#1A2D47"], [.45, "#176B8E"], [.75, PURPLE], [1, ORANGE]]
    for row in grid.itertuples(index=False):
        fill, opacity = "rgba(0,0,0,0)", .55
        if not active.empty and row.grid_id in indexed.index:
            state = indexed.loc[row.grid_id]
            state = state.iloc[0] if isinstance(state, pd.DataFrame) else state
            value = pd.to_numeric(state.get(value_col), errors="coerce")
            if pd.notna(value):
                fill = sample_colorscale(scale, [(float(value) - lo) / (hi - lo)])[0]
                opacity = .5
        fig.add_shape(type="rect", x0=row.x0, y0=row.y0, x1=row.x0 + row.size,
                      y1=row.y0 + row.size, line=dict(color=GRID, width=.55),
                      fillcolor=fill, opacity=opacity, layer="below")


def event_range(event, minimum_span=32.0):
    cx, cy = float(event.centroid_x), float(event.centroid_y)
    span = max(float(event.x_max - event.x_min) + 8,
               float(event.y_max - event.y_min) + 8, minimum_span)
    return [cx - span / 2, cx + span / 2], [cy - span / 2, cy + span / 2]


def build_replay_figure(obs, states, grid, event_members, event, current, perspective,
                        layer_mode, show_lines, show_points, show_grid, show_anomalies,
                        event_only):
    fig = go.Figure()
    current_states = states[states.window_id.eq(current)]
    if show_grid:
        add_grid(fig, grid, current_states, layer_mode)
    history = obs[obs.window_id.between(max(0, current - 2), current)]
    if perspective == "事件局部":
        xr, yr = event_range(event)
        history = history[history.world_x.between(*xr) & history.world_y.between(*yr)]
    else:
        xr = [float(obs.world_x.min()), float(obs.world_x.max())]
        yr = [float(obs.world_y.min()), float(obs.world_y.max())]
    if event_only:
        history = history[history.grid_id.astype(str).isin(set(event_members.grid_id.astype(str)))]

    if show_lines:
        for (source, _, _), group in history.groupby(
                ["source", "sequence_id", "track_id"], sort=False):
            if len(group) < 2:
                continue
            color = SOURCE_STYLE.get(str(source), (MUTED, "circle", str(source)))[0]
            fig.add_trace(go.Scattergl(
                x=group.world_x, y=group.world_y, mode="lines", showlegend=False,
                line=dict(color=color, width=1), opacity=.28, hoverinfo="skip"))
    if show_points:
        points = history[history.window_id.eq(current)]
        for source, group in points.groupby("source", sort=False):
            color, symbol, label = SOURCE_STYLE.get(
                str(source), (MUTED, "circle", str(source)))
            custom = np.stack([
                group.sequence_id.astype(str), group.track_id.astype(str),
                group.object_type.astype(str), group.timestamp_us.astype(str)], axis=-1)
            fig.add_trace(go.Scattergl(
                x=group.world_x, y=group.world_y, mode="markers", name=label,
                marker=dict(color=color, symbol=symbol, size=8,
                            line=dict(color=TEXT, width=.5)), customdata=custom,
                hovertemplate=("%{fullData.name}<br>类型=%{customdata[2]}"
                               "<br>sequence=%{customdata[0]}<br>track=%{customdata[1]}"
                               "<br>timestamp_us=%{customdata[3]}"
                               "<br>X=%{x:.2f} m · Y=%{y:.2f} m<extra></extra>")))

    active = event_members[event_members.window_id.eq(current)]
    if show_anomalies and not active.empty:
        for row in active.itertuples(index=False):
            state = current_states[current_states.grid_id.astype(str).eq(str(row.grid_id))]
            if not state.empty:
                cell = state.iloc[0]
                fig.add_shape(
                    type="rect", x0=cell.x0, y0=cell.y0,
                    x1=cell.x0 + cell["size"], y1=cell.y0 + cell["size"],
                    line=dict(color=PURPLE, width=3),
                    fillcolor="rgba(192,132,252,.12)")
        fig.add_trace(go.Scatter(
            x=active.mean_world_x, y=active.mean_world_y, mode="markers",
            name="当前事件异常状态",
            marker=dict(color=PURPLE, size=14, symbol="x", line=dict(width=2)),
            customdata=np.stack([active.grid_id.astype(str),
                                 active.both_anomaly.astype(str)], axis=-1),
            hovertemplate="grid=%{customdata[0]}<br>双分支共同异常=%{customdata[1]}<extra></extra>"))
    style_figure(fig)
    fig.update_xaxes(range=xr)
    fig.update_yaxes(range=yr)
    return fig


def build_evolution(evo, current):
    fig = go.Figure()
    fig.add_bar(x=evo.window_id, y=evo.observation_count,
                name="观测记录数", marker_color=BLUE)
    fig.add_scatter(x=evo.window_id, y=evo.mean_speed_mps,
                    name="平均速度 (m/s)", mode="lines+markers",
                    line=dict(color=ORANGE, width=3), yaxis="y2")
    fig.add_vline(x=current, line_color=PURPLE, line_dash="dash", line_width=2)
    fig.update_layout(
        height=350, margin=dict(l=12, r=12, t=34, b=10),
        paper_bgcolor=BG, plot_bgcolor="#0D192A", font_color=TEXT,
        legend=dict(orientation="h", y=1.02),
        xaxis=dict(title="时间窗口", dtick=1, gridcolor="#21334D"),
        yaxis=dict(title="观测记录数", gridcolor="#21334D"),
        yaxis2=dict(title="平均速度 (m/s)", overlaying="y", side="right"))
    return fig


raw, missing = load_data()
if missing:
    st.error("缺少必要文件，无法建立可追溯回放。", icon=":material/error:")
    st.code("\n".join(missing))
    st.stop()
d = prepare_data(raw)
observations, states, anomalies, grid = (d[k] for k in
    ("observations", "states", "anomalies", "grid"))
events, members, evolution, priorities, traces = (d[k] for k in
    ("events", "members", "evolution", "priorities", "traces"))
traffic = events[events.is_sustained_traffic_candidate.astype(bool)].copy()
options = traffic.event_id.astype(str).tolist()
windows = sorted(states.window_id.astype(int).unique().tolist())
default_event = "EVT2_006" if "EVT2_006" in options else options[0]
st.session_state.setdefault("selected_event", default_event)
default_start = int(events.loc[events.event_id.eq(
    st.session_state.selected_event), "start_window"].iloc[0])
st.session_state.setdefault("current_window", default_start)
st.session_state.setdefault("playing", False)
st.session_state.setdefault("last_tick", time.monotonic())


def event_changed():
    selected = st.session_state.selected_event
    st.session_state.current_window = int(events.loc[
        events.event_id.astype(str).eq(selected), "start_window"].iloc[0])
    st.session_state.playing = False


st.title("DSTA · 城市交通时空态势认知")
st.caption("DAIR-V2X-Seq SPD-Example · 真实车路协同观测回放 · 数据集世界坐标（非地理经纬度）")
with st.container(horizontal=True):
    st.metric("时空观测", f"{len(observations):,}", border=True)
    st.metric("城市网格状态", f"{len(states):,}", border=True)
    st.metric("异常状态并集", f"{int(anomalies.either_anomaly.sum()):,}", border=True)
    st.metric("持续交通候选事件", f"{len(traffic):,}", border=True)

control, toggles = st.columns([1.45, 2.55], vertical_alignment="bottom")
with control:
    st.selectbox("候选事件", options, key="selected_event", on_change=event_changed)
with toggles:
    perspective = st.segmented_control(
        "视角", ["全局城市", "事件局部"], default="事件局部")
    layer_mode = st.segmented_control(
        "网格着色", ["观测数量", "平均速度", "异常分布"], default="观测数量")

event = events[events.event_id.astype(str).eq(st.session_state.selected_event)].iloc[0]
event_members = members[members.event_id.astype(str).eq(
    st.session_state.selected_event)].copy()
event_evo = evolution[evolution.event_id.astype(str).eq(
    st.session_state.selected_event)].copy().sort_values("window_id")
priority = priorities[priorities.event_id.astype(str).eq(
    st.session_state.selected_event)]
priority_score = float(priority.iloc[0].relative_priority_score) if not priority.empty else np.nan

with st.expander("图层与回放设置", expanded=True):
    a, b, c, e, f, g = st.columns(6)
    show_lines = a.toggle("历史短轨迹", value=True)
    show_points = b.toggle("当前观测点", value=True)
    show_grid = c.toggle("自适应网格", value=True)
    show_anomalies = e.toggle("事件异常区", value=True)
    event_only = f.toggle("仅事件网格观测", value=False)
    loop = g.toggle("循环播放", value=False)
    speed = st.select_slider(
        "播放间隔（秒/窗口）", options=[2.0, 1.5, 1.0, 0.5], value=1.0,
        format_func=lambda value: f"{value:g} s")


@st.fragment(run_every=float(speed) if st.session_state.playing else None)
def replay_panel():
    if st.session_state.playing:
        now = time.monotonic()
        if now - st.session_state.last_tick >= float(speed) * .8:
            idx = windows.index(int(st.session_state.current_window))
            if idx == len(windows) - 1:
                if loop:
                    st.session_state.current_window = windows[0]
                else:
                    st.session_state.playing = False
            else:
                st.session_state.current_window = windows[idx + 1]
            st.session_state.last_tick = now

    p, prev, nxt, status = st.columns([1, 1, 1, 5], vertical_alignment="center")
    if p.button("暂停" if st.session_state.playing else "播放",
                icon=":material/pause:" if st.session_state.playing else ":material/play_arrow:",
                width="stretch"):
        st.session_state.playing = not st.session_state.playing
        st.session_state.last_tick = time.monotonic()
        st.rerun()
    if prev.button("上一窗口", icon=":material/skip_previous:",
                   width="stretch", disabled=st.session_state.playing):
        idx = windows.index(int(st.session_state.current_window))
        st.session_state.current_window = windows[max(0, idx - 1)]
        st.rerun()
    if nxt.button("下一窗口", icon=":material/skip_next:",
                  width="stretch", disabled=st.session_state.playing):
        idx = windows.index(int(st.session_state.current_window))
        st.session_state.current_window = windows[min(len(windows) - 1, idx + 1)]
        st.rerun()
    status.caption("播放到窗口 12 后" + ("回到窗口 0。" if loop else "自动停止。"))
    st.slider("统一时间窗口（每窗口约 2 秒）", min(windows), max(windows),
              key="current_window", disabled=st.session_state.playing)
    current = int(st.session_state.current_window)

    left, right = st.columns([3.25, 1], gap="large")
    with left:
        st.subheader("车辆轨迹鸟瞰 · 世界坐标")
        replay = build_replay_figure(
            observations, states, grid, event_members, event, current,
            perspective or "事件局部", layer_mode or "观测数量",
            show_lines, show_points, show_grid, show_anomalies, event_only)
        st.plotly_chart(
            replay, width="stretch", key=f"replay-{current}", theme=None,
            alt="真实世界坐标中的车端和路侧观测、短轨迹、自适应网格与异常状态")
        st.caption("轨迹仅在同一 source × sequence_id × track_id 内连接，显示当前及前 2 个窗口；车路观测未做跨源唯一车辆去重。")
    with right:
        st.subheader(f"{st.session_state.selected_event} · 窗口 {current}")
        st.metric("持续窗口", f"{int(event.start_window)}–{int(event.end_window)}", border=True)
        st.metric("异常状态 / 网格", f"{int(event.state_count)} / {int(event.grid_count)}", border=True)
        st.metric("观测支持（记录）", f"{int(event.observation_sum):,}", border=True)
        st.metric("双分支共同异常", f"{int(event.both_branch_states)}", border=True)
        st.metric("相对优先级（非概率）",
                  f"{priority_score:.2f} / 100" if pd.notna(priority_score) else "—",
                  border=True)
        active = event_members[event_members.window_id.eq(current)]
        if active.empty:
            st.info("当前窗口无该事件的异常成员状态。", icon=":material/info:")
        else:
            st.markdown("当前异常状态")
            st.dataframe(
                active[["grid_id", "raw_is_anomaly", "ae_is_anomaly",
                        "both_anomaly", "observation_count", "mean_speed_mps"]],
                hide_index=True, width="stretch", height=min(220, 38 + 35 * len(active)),
                column_config={"mean_speed_mps": st.column_config.NumberColumn(
                    "平均速度 (m/s)", format="%.2f")})
    st.subheader("事件时序演化")
    st.plotly_chart(
        build_evolution(event_evo, current), width="stretch",
        key=f"evolution-{current}", theme=None,
        alt="候选事件各时间窗口的平均速度和观测记录数")
    st.caption("观测记录数不是独立车辆数；平均速度为事件成员状态按观测量聚合的描述指标。")


replay_panel()
st.subheader("事件证据回溯")
section = st.segmented_control(
    "证据视图", ["原始观测", "异常状态成员", "证据关系"],
    default="原始观测", label_visibility="collapsed")
if section == "原始观测":
    event_traces = traces[traces.event_id.astype(str).eq(
        st.session_state.selected_event)].copy()
    q1, q2 = st.columns(2)
    source_options = sorted(event_traces.source.dropna().astype(str).unique())
    source_filter = q1.multiselect("来源", source_options, default=source_options)
    trace_windows = sorted(event_traces.window_id.dropna().astype(int).unique())
    window_filter = q2.multiselect(
        "时间窗口", trace_windows,
        default=[w for w in trace_windows if w == st.session_state.current_window])
    filtered = event_traces[event_traces.source.astype(str).isin(source_filter)]
    if window_filter:
        filtered = filtered[filtered.window_id.astype(int).isin(window_filter)]
    st.caption(f"匹配 {len(filtered):,} / {len(event_traces):,} 条原始观测回溯记录；同一目标可在多帧中重复出现。")
    cols = [c for c in ["window_id", "source", "sequence_id", "frame_id",
        "track_id", "object_type", "timestamp_us", "world_x", "world_y", "grid_id"]
        if c in filtered]
    st.dataframe(
        filtered[cols], hide_index=True, width="stretch", height=360,
        column_config={
            "world_x": st.column_config.NumberColumn("World X (m)", format="%.3f"),
            "world_y": st.column_config.NumberColumn("World Y (m)", format="%.3f")})
elif section == "异常状态成员":
    st.dataframe(
        event_members.sort_values(["window_id", "grid_id"]),
        hide_index=True, width="stretch", height=360,
        column_config={"mean_speed_mps": st.column_config.NumberColumn(
            "平均速度 (m/s)", format="%.3f")})
else:
    graph = d["graph"]
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    event_node = st.session_state.selected_event
    edges = [edge for edge in graph.get("edges", [])
             if edge["source"].startswith(event_node) or edge["target"].startswith(event_node)]
    window_nodes = sorted(
        [node for node in nodes.values()
         if node["id"].startswith(event_node + ":window:")],
        key=lambda node: int(node["id"].split(":")[-1]))
    st.markdown(f"**候选事件 `{event_node}` → {len(window_nodes)} 个观测窗口 → {len(event_members)} 个异常网格状态**")
    for node in window_nodes:
        window_id = int(node["id"].split(":")[-1])
        rows = event_members[event_members.window_id.eq(window_id)]
        with st.expander(f"窗口 {window_id} · {len(rows)} 个异常状态"):
            for row in rows.itertuples(index=False):
                st.markdown(
                    f"- `{row.grid_id}` · 原始分支={row.raw_is_anomaly}"
                    f" · AE 分支={row.ae_is_anomaly} · 观测={row.observation_count}")
    st.caption(f"证据图中与该事件相关的边：{len(edges)}。{graph.get('disclaimer', '')}")

st.warning(
    "本页展示统计算法生成的交通模式异常候选，不是已确认事故或拥堵；相对优先级不是事故概率。",
    icon=":material/science:")
