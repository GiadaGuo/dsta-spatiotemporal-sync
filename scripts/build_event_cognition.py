#!/usr/bin/env python3
"""Step 7: evidence-grounded cognition and relative prioritization of DSTA candidate events.

Run from anywhere: python scripts/build_event_cognition.py
Uses observed state and event membership; does not infer accidents or causal relations.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed'
REPORT = ROOT / 'outputs/reports'
FIGURES = ROOT / 'outputs/figures'

def safe_float(v):
    return float(v) if pd.notna(v) and np.isfinite(float(v)) else None

def main():
    events = pd.read_csv(REPORT / 'detected_events_v2.csv')
    members = pd.read_csv(REPORT / 'event_members_v2.csv')
    states = pd.read_parquet(DATA / 'embedding_anomaly_states.parquet')
    needed = ['window_id','grid_id','observation_count','mean_speed_mps','mean_world_x','mean_world_y',
              'vehicle_observations','infrastructure_observations','valid_speed_count','anomaly_score',
              'ae_anomaly_score','raw_is_anomaly','ae_is_anomaly']
    missing = [c for c in needed if c not in states.columns]
    if missing: raise ValueError(f'Missing state columns: {missing}')
    # Key-based lookup: never assume state_row_id matches parquet physical row order.
    if states.duplicated(['window_id','grid_id']).any():
        raise ValueError('State keys (window_id, grid_id) are not unique')
    if members.duplicated(['event_id','window_id','grid_id']).any():
        raise ValueError('Duplicate event membership keys')
    joined = members.merge(states[needed], on=['window_id','grid_id'], how='left', validate='many_to_one',
                           indicator=True, suffixes=('_member',''))
    if not joined['_merge'].eq('both').all():
        raise ValueError(f'Unmatched event members: {joined.loc[joined._merge.ne("both"), ["event_id","window_id","grid_id"]].to_dict("records")}')
    # Check the member snapshot against the actual state values.
    for col in ['observation_count','mean_speed_mps','mean_world_x','mean_world_y']:
        mcol = col + '_member'
        if mcol in joined:
            a = pd.to_numeric(joined[mcol], errors='coerce').to_numpy(float)
            b = pd.to_numeric(joined[col], errors='coerce').to_numpy(float)
            if not np.allclose(a,b,rtol=1e-6,atol=1e-6,equal_nan=True):
                raise ValueError(f'Member/state mismatch in {col}')
    for col in ['raw_is_anomaly','ae_is_anomaly']:
        mcol = col + '_member'
        if mcol in joined and not (joined[mcol].fillna(False).astype(bool).to_numpy() == joined[col].fillna(False).astype(bool).to_numpy()).all():
            raise ValueError(f'Member/state mismatch in {col}')
    if 'state_row_id' in joined:
        # IDs are provenance only: the key join above is authoritative.
        joined['state_row_id_is_provenance_only'] = True
    if set(joined.event_id) != set(events.event_id):
        raise ValueError('Event IDs and member IDs disagree')

    # Relative score components are empirical ranks across *sustained traffic candidates* only.
    candidates = events.loc[events.event_type.eq('traffic_pattern_candidate')].copy()
    if candidates.empty: raise ValueError('No sustained traffic candidates')
    rows, timeline_rows = [], []
    for _, e in candidates.iterrows():
        sub = joined.loc[joined.event_id.eq(e.event_id)].sort_values(['window_id','grid_id']).copy()
        counts = sub.observation_count.clip(lower=1).to_numpy(float)
        valid = sub.mean_speed_mps.notna().to_numpy()
        avg_speed = np.average(sub.loc[valid,'mean_speed_mps'],weights=counts[valid]) if valid.any() else np.nan
        raw_scores = sub.anomaly_score.dropna()
        ae_scores = sub.ae_anomaly_score.dropna()
        by_window = sub.groupby('window_id',sort=True).apply(
            lambda g: pd.Series({
                'state_count':int(len(g)),
                'observation_count':int(g.observation_count.sum()),
                'mean_speed_mps':float(np.average(g.loc[g.mean_speed_mps.notna(),'mean_speed_mps'],
                                        weights=g.loc[g.mean_speed_mps.notna(),'observation_count'].clip(lower=1)))
                                if g.mean_speed_mps.notna().any() else np.nan,
                'both_branch_states':int((g.raw_is_anomaly.astype(bool)&g.ae_is_anomaly.astype(bool)).sum()),
                'vehicle_observations':int(g.vehicle_observations.sum()),
                'infrastructure_observations':int(g.infrastructure_observations.sum()),
            }),include_groups=False).reset_index()
        for rec in by_window.to_dict('records'):
            timeline_rows.append({'event_id':e.event_id,**rec})
        rows.append(dict(event_id=e.event_id, start_window=int(e.start_window),end_window=int(e.end_window),
                         window_span=int(e.end_window-e.start_window+1),state_count=int(len(sub)),
                         grid_count=int(sub.grid_id.nunique()),observation_sum=int(sub.observation_count.sum()),
                         both_branch_states=int((sub.raw_is_anomaly.astype(bool)&sub.ae_is_anomaly.astype(bool)).sum()),
                         both_branch_ratio=float((sub.raw_is_anomaly.astype(bool)&sub.ae_is_anomaly.astype(bool)).mean()),
                         weighted_mean_speed_mps=safe_float(avg_speed),
                         mean_raw_anomaly_score=safe_float(raw_scores.mean()) if len(raw_scores) else None,
                         mean_ae_anomaly_score=safe_float(ae_scores.mean()) if len(ae_scores) else None,
                         vehicle_observations=int(sub.vehicle_observations.sum()),
                         infrastructure_observations=int(sub.infrastructure_observations.sum()),
                         centroid_x=float(np.average(sub.mean_world_x,weights=counts)),
                         centroid_y=float(np.average(sub.mean_world_y,weights=counts)),
                         spatial_x_min=float(sub.mean_world_x.min()),spatial_x_max=float(sub.mean_world_x.max()),
                         spatial_y_min=float(sub.mean_world_y.min()),spatial_y_max=float(sub.mean_world_y.max())))
    scored = pd.DataFrame(rows)
    # A ranking score, not a probability or a calibrated safety-risk score.
    features = {'anomaly_strength':'mean_raw_anomaly_score','duration':'window_span',
                'observation_support':'observation_sum','cross_branch_support':'both_branch_ratio'}
    weights = {'anomaly_strength':.35,'duration':.25,'observation_support':.20,'cross_branch_support':.20}
    for name,col in features.items():
        scored[name+'_percentile'] = scored[col].rank(method='average',pct=True,na_option='bottom')
    scored['relative_priority_score'] = 100*sum(weights[k]*scored[k+'_percentile'] for k in weights)
    scored = scored.sort_values(['relative_priority_score','event_id'],ascending=[False,True])
    scored.to_csv(REPORT/'event_risk_priorities.csv',index=False)
    timeline = pd.DataFrame(timeline_rows).sort_values(['event_id','window_id'])
    timeline.to_csv(REPORT/'event_evolution.csv',index=False)
    joined.drop(columns=['_merge']).to_csv(REPORT/'event_evidence_states.csv',index=False)

    # Build evidence graph as typed nodes/edges. Temporal order != causality.
    nodes,edges=[],[]
    for e in scored.to_dict('records'):
        eid=e['event_id']; nodes.append({'id':eid,'type':'candidate_event','label':eid})
        sub=joined.loc[joined.event_id.eq(eid)].sort_values(['window_id','grid_id'])
        windows=sorted(int(w) for w in sub.window_id.unique())
        for w in windows:
            wid=f'{eid}:window:{w}'
            nodes.append({'id':wid,'type':'time_window','label':f'window {w}'})
            edges.append({'source':eid,'target':wid,'relation':'HAS_OBSERVATION_WINDOW'})
        for a,b in zip(windows,windows[1:]):
            edges.append({'source':f'{eid}:window:{a}','target':f'{eid}:window:{b}',
                          'relation':'PRECEDES_OBSERVED_WINDOW'})
        for r in sub.itertuples(index=False):
            sid=f'{eid}:state:{int(r.window_id)}:{r.grid_id}'
            nodes.append({'id':sid,'type':'grid_state','label':str(r.grid_id)})
            edges.append({'source':f'{eid}:window:{int(r.window_id)}','target':sid,'relation':'CONTAINS_ANOMALOUS_STATE'})
            edges.append({'source':eid,'target':sid,'relation':'SUPPORTED_BY_STATE'})
    graph={'nodes':nodes,'edges':edges,'disclaimer':'Temporal/spatial association only; no causal event inference.'}
    (REPORT/'event_evidence_graph.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2),encoding='utf-8')
    payload={'method':'Relative prioritization among sustained traffic candidates only',
             'weights':weights,'score_is_probability':False,'score_is_calibrated_risk':False,
             'time_window_seconds':2,'events':[]}
    for e in scored.to_dict('records'):
        eid=e['event_id']
        ev_timeline=timeline.loc[timeline.event_id.eq(eid)].replace({np.nan:None}).to_dict('records')
        payload['events'].append({'event_id':eid,'relative_priority_score':round(e['relative_priority_score'],2),
                                  'evidence_summary':{k:(v.item() if isinstance(v,np.generic) else v)
                                      for k,v in e.items() if k not in [z+'_percentile' for z in weights]},
                                  'timeline':ev_timeline,
                                  'interpretation':'Spatiotemporally persistent anomalous traffic-state pattern; not a confirmed incident.'})
    (REPORT/'event_cognition.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

    # Optional trace-back to observation-level data, using matching grid/window keys.
    obs_file=DATA/'spatiotemporal_events.parquet'
    if obs_file.exists():
        obs=pd.read_parquet(obs_file)
        if {'grid_id','timestamp_us'}.issubset(obs.columns) and 'window_start_us' in states.columns:
            starts=states[['window_id','window_start_us']].drop_duplicates()
            if not starts.groupby('window_id').window_start_us.nunique().eq(1).all():
                print('WARNING: window_start_us inconsistent; skipping observation-level export')
            else:
                t0=int(starts.loc[starts.window_id.eq(0),'window_start_us'].min()) if starts.window_id.eq(0).any() else int(starts.window_start_us.min())
                obs=obs.copy()
                obs['window_id']=np.floor((obs.timestamp_us.astype('int64')-t0)/2_000_000).astype(int)
                keys=joined[['event_id','window_id','grid_id']].drop_duplicates()
                matched=keys.merge(obs,on=['window_id','grid_id'],how='inner')
                matched.to_csv(REPORT/'event_observation_traces.csv',index=False)
                print(f'Observation trace rows: {len(matched)} (validate timestamp window alignment if counts differ)')
        else: print('WARNING: observation file lacks required columns; skipped observation trace')
    else: print('NOTE: spatiotemporal_events.parquet not found; state-level evidence still available')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        primary='EVT2_006' if 'EVT2_006' in set(timeline.event_id) else str(scored.iloc[0].event_id)
        t=timeline.loc[timeline.event_id.eq(primary)]
        fig,ax=plt.subplots(figsize=(8,4))
        ax.plot(t.window_id,t.mean_speed_mps,marker='o',color='#2563eb',label='Observation-weighted mean speed')
        ax.set_xlabel('2-second window');ax.set_ylabel('Speed (m/s)');ax.grid(alpha=.2)
        ax2=ax.twinx();ax2.bar(t.window_id,t.observation_count,alpha=.25,color='#f97316',label='Observations')
        ax2.set_ylabel('Observation count')
        fig.suptitle(f'{primary}: observed state evolution (not a confirmed incident)')
        fig.tight_layout();fig.savefig(FIGURES/'17_event_evolution.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,4))
        b=scored.sort_values('relative_priority_score')
        ax.barh(b.event_id,b.relative_priority_score,color='#2563eb')
        ax.set_xlim(0,100);ax.set_xlabel('Relative priority score (not risk probability)')
        ax.set_title('Sustained traffic candidate prioritization')
        fig.tight_layout();fig.savefig(FIGURES/'18_event_priorities.png',dpi=180);plt.close(fig)
    except ImportError:
        print('NOTE: matplotlib missing, figures skipped')
    print('\n=== STEP 7 COGNITION SUMMARY ===')
    print(scored[['event_id','window_span','state_count','observation_sum','both_branch_states','relative_priority_score']].to_string(index=False))
    print('\nOutputs in',REPORT)

if __name__=='__main__':
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    main()
