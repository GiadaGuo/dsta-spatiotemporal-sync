#!/usr/bin/env python3
"""Regroup existing DSTA anomaly states into conservative, traceable event candidates.

No anomaly model is retrained.  A 20 m distance threshold is a demo heuristic,
not a calibrated traffic-event detection radius.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed/embedding_anomaly_states.parquet'
REPORT = ROOT / 'outputs/reports'
DISTANCE_M = 20.0
MAX_WINDOW_GAP = 1
MIN_TRAFFIC_STATES = 3
MIN_TRAFFIC_WINDOWS = 2
MIN_TRAFFIC_OBSERVATIONS = 5


def regroup(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = ['window_id', 'grid_id', 'observation_count', 'mean_speed_mps',
                'mean_world_x', 'mean_world_y', 'raw_is_anomaly', 'ae_is_anomaly',
                'x0', 'y0', 'size']
    absent = [c for c in required if c not in df.columns]
    if absent:
        raise ValueError(f'Missing columns in embedding_anomaly_states.parquet: {absent}')

    # Original Step 6.2 assigned state_row_id after sorting by window_id/grid_id.
    df = df.sort_values(['window_id', 'grid_id'], kind='stable').reset_index(drop=True)
    df['state_row_id'] = np.arange(len(df), dtype=int)
    df['raw_is_anomaly'] = df['raw_is_anomaly'].fillna(False).astype(bool)
    df['ae_is_anomaly'] = df['ae_is_anomaly'].fillna(False).astype(bool)
    df['either_anomaly'] = df['raw_is_anomaly'] | df['ae_is_anomaly']
    df['both_anomaly'] = df['raw_is_anomaly'] & df['ae_is_anomaly']

    # Recompute labels with the SAME threshold as the original Step 6.2.
    df['candidate_kind'] = np.where(
        (df['observation_count'] < MIN_TRAFFIC_OBSERVATIONS)
        | df['mean_speed_mps'].isna(),
        'observation_quality', 'traffic_pattern'
    )
    candidates = df.loc[df['either_anomaly']].copy().reset_index(drop=True)
    if candidates[['mean_world_x', 'mean_world_y']].isna().any().any():
        raise ValueError('Candidate states have missing mean_world_x/mean_world_y')
    n = len(candidates)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    # Sorted by time. Only connect states of the same candidate type.
    # Centers here are actual mean OBSERVATION positions, not cell centers.
    for i in range(n):
        a = candidates.iloc[i]
        for j in range(i + 1, n):
            b = candidates.iloc[j]
            delta_window = int(b.window_id) - int(a.window_id)
            if delta_window > MAX_WINDOW_GAP:
                break
            if a.candidate_kind != b.candidate_kind:
                continue
            distance = float(np.hypot(
                a.mean_world_x - b.mean_world_x,
                a.mean_world_y - b.mean_world_y
            ))
            if distance <= DISTANCE_M:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    sorted_groups = sorted(
        groups.values(),
        key=lambda ids: (int(candidates.iloc[ids].window_id.min()),
                         -len(ids), int(candidates.iloc[ids].state_row_id.min()))
    )

    event_rows = []
    member_rows = []
    for k, ids in enumerate(sorted_groups, 1):
        sub = candidates.iloc[ids]
        kind = str(sub.candidate_kind.iloc[0])
        windows = sorted(int(v) for v in sub.window_id.unique())
        sustained = (kind == 'traffic_pattern'
                     and len(windows) >= MIN_TRAFFIC_WINDOWS
                     and len(sub) >= MIN_TRAFFIC_STATES)
        event_id = f'EVT2_{k:03d}'
        event_type = ('traffic_pattern_candidate' if sustained
                      else 'traffic_pattern_isolated' if kind == 'traffic_pattern'
                      else 'observation_quality_candidate')
        row = dict(
            event_id=event_id,
            event_type=event_type,
            candidate_kind=kind,
            start_window=windows[0], end_window=windows[-1],
            duration_windows=len(windows),
            state_count=len(sub), grid_count=int(sub.grid_id.nunique()),
            observation_sum=int(sub.observation_count.sum()),
            both_branch_states=int(sub.both_anomaly.sum()),
            raw_branch_states=int(sub.raw_is_anomaly.sum()),
            ae_branch_states=int(sub.ae_is_anomaly.sum()),
            mean_ae_score=(float(sub.ae_anomaly_score.mean())
                           if 'ae_anomaly_score' in sub and sub.ae_anomaly_score.notna().any()
                           else np.nan),
            centroid_x=float(np.average(sub.mean_world_x, weights=sub.observation_count.clip(lower=1))),
            centroid_y=float(np.average(sub.mean_world_y, weights=sub.observation_count.clip(lower=1))),
            observation_x_min=float(sub.mean_world_x.min()),
            observation_x_max=float(sub.mean_world_x.max()),
            observation_y_min=float(sub.mean_world_y.min()),
            observation_y_max=float(sub.mean_world_y.max()),
            x_min=float(sub.x0.min()), y_min=float(sub.y0.min()),
            x_max=float((sub.x0 + sub['size']).max()),
            y_max=float((sub.y0 + sub['size']).max()),
            grid_ids='|'.join(sorted(map(str, sub.grid_id.unique()))),
            is_sustained_traffic_candidate=bool(sustained)
        )
        event_rows.append(row)
        for member in sub.itertuples(index=False):
            member_rows.append(dict(
                event_id=event_id, state_row_id=int(member.state_row_id),
                window_id=int(member.window_id), grid_id=str(member.grid_id),
                raw_is_anomaly=bool(member.raw_is_anomaly),
                ae_is_anomaly=bool(member.ae_is_anomaly),
                both_anomaly=bool(member.both_anomaly),
                candidate_kind=str(member.candidate_kind),
                observation_count=int(member.observation_count),
                mean_speed_mps=float(member.mean_speed_mps) if pd.notna(member.mean_speed_mps) else np.nan,
                mean_world_x=float(member.mean_world_x),
                mean_world_y=float(member.mean_world_y)
            ))

    events = pd.DataFrame(event_rows)
    members = pd.DataFrame(member_rows)
    if len(members) != n or (not members.empty and members.state_row_id.nunique() != n):
        raise AssertionError('Some candidate states were dropped or duplicated')
    if not members.empty and members.groupby('event_id').candidate_kind.nunique().max() != 1:
        raise AssertionError('Cross-kind merging detected')

    counts = events.event_type.value_counts().to_dict() if not events.empty else {}
    summary = dict(
        input_states=len(df), union_anomaly_states=n,
        raw_anomaly_states=int(df.raw_is_anomaly.sum()),
        ae_anomaly_states=int(df.ae_is_anomaly.sum()),
        both_branch_states=int(df.both_anomaly.sum()),
        total_components=len(events),
        sustained_traffic_candidates=int(counts.get('traffic_pattern_candidate', 0)),
        isolated_traffic_components=int(counts.get('traffic_pattern_isolated', 0)),
        observation_quality_components=int(counts.get('observation_quality_candidate', 0)),
        spatial_distance_threshold_m=DISTANCE_M,
        max_window_gap=MAX_WINDOW_GAP,
        min_traffic_states=MIN_TRAFFIC_STATES,
        min_traffic_windows=MIN_TRAFFIC_WINDOWS,
        min_traffic_observations_per_state=MIN_TRAFFIC_OBSERVATIONS,
        note='Exploratory candidate grouping, not verified accidents or causal events.'
    )
    return events, members, summary


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f'Run detect_and_group_events.py first: {INPUT}')
    events, members, summary = regroup(pd.read_parquet(INPUT))
    REPORT.mkdir(parents=True, exist_ok=True)
    events.to_csv(REPORT / 'detected_events_v2.csv', index=False)
    members.to_csv(REPORT / 'event_members_v2.csv', index=False)
    (REPORT / 'event_regroup_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print('=== EVENT REGROUP SUMMARY ===')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\n=== TOP SUSTAINED TRAFFIC CANDIDATES ===')
    columns = ['event_id', 'start_window', 'end_window', 'state_count',
               'grid_count', 'observation_sum', 'both_branch_states', 'event_type']
    selected = events.loc[events.event_type == 'traffic_pattern_candidate']
    print(selected.sort_values(['state_count', 'both_branch_states'], ascending=False)[columns]
          .head(10).to_string(index=False) if len(selected) else '(none; inspect isolated components)')
    print('\n=== LARGEST COMPONENTS (ALL TYPES) ===')
    print(events.sort_values('state_count', ascending=False)[columns].head(10).to_string(index=False))
    print('\nSaved:', REPORT / 'detected_events_v2.csv',
          REPORT / 'event_members_v2.csv', REPORT / 'event_regroup_summary.json')


if __name__ == '__main__':
    main()
