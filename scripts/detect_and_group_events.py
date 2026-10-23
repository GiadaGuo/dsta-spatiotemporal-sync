#!/usr/bin/env python3
"""DSTA Step 6.2: embedding anomaly detection and candidate event grouping."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INFILE = ROOT/'data/processed/city_state_ae_embeddings.parquet'
OUTDATA = ROOT/'data/processed/embedding_anomaly_states.parquet'
REPORT = ROOT/'outputs/reports'
FIG = ROOT/'outputs/figures'
SEED = 42
CONTAMINATION = .08
MIN_TRAFFIC_OBSERVATIONS = 5
EPS = 1e-7


def connected(a, b):
    # Axis-aligned rectangles overlap or share a boundary (including corner).
    return (a.x0 <= b.x0+b.size+EPS and b.x0 <= a.x0+a.size+EPS
            and a.y0 <= b.y0+b.size+EPS and b.y0 <= a.y0+a.size+EPS)


def main():
    df = pd.read_parquet(INFILE).copy().sort_values(['window_id','grid_id']).reset_index(drop=True)
    needed = ['window_id','grid_id','is_anomaly','observation_count','mean_speed_mps',
              'x0','y0','size','anomaly_score'] + [f'ae_embedding_{i}' for i in range(8)]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}; check AE input/output and earlier steps')
    eligible = df['eligible_for_detection'].fillna(False).astype(bool).to_numpy() if 'eligible_for_detection' in df else df.window_id.between(df.window_id.min()+1,df.window_id.max()-1).to_numpy()
    Z = df[[f'ae_embedding_{i}' for i in range(8)]].to_numpy(dtype=float)
    if not np.isfinite(Z).all():
        raise ValueError('AE embeddings contain NaN or inf')
    # Exploratory unsupervised fit on eligible states; not an out-of-sample accuracy claim.
    scaled = StandardScaler().fit_transform(Z[eligible])
    detector = IsolationForest(n_estimators=200, contamination=CONTAMINATION, random_state=SEED, n_jobs=-1)
    detector.fit(scaled)
    score = -detector.score_samples(scaled)
    flagged = detector.predict(scaled) == -1
    df['ae_anomaly_score'] = np.nan
    df['ae_is_anomaly'] = False
    df.loc[eligible,'ae_anomaly_score'] = score
    df.loc[eligible,'ae_is_anomaly'] = flagged
    df['raw_is_anomaly'] = df['is_anomaly'].fillna(False).astype(bool)
    df['either_anomaly'] = df['raw_is_anomaly'] | df['ae_is_anomaly']
    df['both_anomaly'] = df['raw_is_anomaly'] & df['ae_is_anomaly']
    df['candidate_kind'] = np.where((df.observation_count < MIN_TRAFFIC_OBSERVATIONS) |
                                    (df.mean_speed_mps.isna()), 'observation_quality', 'traffic_pattern')
    REPORT.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(parents=True,exist_ok=True)
    OUTDATA.parent.mkdir(parents=True,exist_ok=True)
    df.to_parquet(OUTDATA,index=False)

    candidates = df.loc[df.either_anomaly].copy().reset_index().rename(columns={'index':'state_row_id'})
    n=len(candidates)
    parent=list(range(n))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(i,j):
        ri,rj=find(i),find(j)
        if ri!=rj: parent[rj]=ri
    # Only connect same/adjacent windows, spatially touching cells.
    for i in range(n):
        a=candidates.iloc[i]
        for j in range(i+1,n):
            b=candidates.iloc[j]
            if int(b.window_id)-int(a.window_id)>1: break
            if abs(int(b.window_id)-int(a.window_id))<=1 and connected(a,b):
                union(i,j)
    groups={}
    for i in range(n): groups.setdefault(find(i),[]).append(i)
    records=[]
    members=[]
    for k,ids in enumerate(sorted(groups.values(), key=lambda ids:(int(candidates.iloc[ids].window_id.min()),-len(ids))),1):
        sub=candidates.iloc[ids]
        event_id=f'EVT_{k:03d}'
        traffic=int((sub.candidate_kind=='traffic_pattern').sum())
        quality=int((sub.candidate_kind=='observation_quality').sum())
        windows=sorted(map(int,sub.window_id.unique()))
        records.append(dict(event_id=event_id,start_window=min(windows),end_window=max(windows),
                            duration_windows=len(windows),state_count=len(sub),grid_count=sub.grid_id.nunique(),
                            observation_sum=int(sub.observation_count.sum()),
                            both_branch_states=int(sub.both_anomaly.sum()),traffic_pattern_states=traffic,
                            observation_quality_states=quality,
                            event_type='traffic_pattern_candidate' if traffic>0 else 'observation_quality_candidate',
                            mean_ae_score=float(sub.ae_anomaly_score.mean(skipna=True)) if sub.ae_anomaly_score.notna().any() else np.nan,
                            x_min=float(sub.x0.min()),y_min=float(sub.y0.min()),
                            x_max=float((sub.x0+sub['size']).max()),y_max=float((sub.y0+sub['size']).max()),
                            grid_ids='|'.join(sorted(map(str,sub.grid_id.unique())))))
        for row in sub.itertuples():
            members.append(dict(event_id=event_id,state_row_id=int(row.state_row_id),
                                window_id=int(row.window_id),grid_id=row.grid_id,
                                raw_is_anomaly=bool(row.raw_is_anomaly),ae_is_anomaly=bool(row.ae_is_anomaly),
                                candidate_kind=row.candidate_kind))
    event_cols=['event_id','start_window','end_window','duration_windows','state_count','grid_count','observation_sum',
                'both_branch_states','traffic_pattern_states','observation_quality_states','event_type',
                'mean_ae_score','x_min','y_min','x_max','y_max','grid_ids']
    events=pd.DataFrame(records,columns=event_cols)
    event_members=pd.DataFrame(members,columns=['event_id','state_row_id','window_id','grid_id',
                                                'raw_is_anomaly','ae_is_anomaly','candidate_kind'])
    events.to_csv(REPORT/'detected_events.csv',index=False)
    event_members.to_csv(REPORT/'event_members.csv',index=False)
    # Spatial map of anomaly states; use blue/orange, not red-green.
    fig,ax=plt.subplots(figsize=(9,8))
    for row in df.loc[df.either_anomaly].itertuples():
        color='#2563eb' if row.candidate_kind=='traffic_pattern' else '#f97316'
        ax.add_patch(Rectangle((row.x0,row.y0),row.size,row.size,facecolor=color,edgecolor='white',lw=.3,alpha=.6))
    ax.set_xlim(3584,3968);ax.set_ylim(2112,2496);ax.set_aspect('equal')
    ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
    ax.set_title('Candidate anomaly locations (all eligible windows)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color='#2563eb',lw=5,label='Traffic-pattern candidate'),
                       Line2D([0],[0],color='#f97316',lw=5,label='Observation-quality candidate')])
    fig.tight_layout();fig.savefig(FIG/'15_anomaly_hotspots.png',dpi=220);plt.close(fig)
    temporal=(df.loc[eligible].groupby('window_id')[['raw_is_anomaly','ae_is_anomaly','both_anomaly']].sum())
    ax=temporal.plot(kind='line',marker='o',figsize=(10,4.5),color=['#2563eb','#f97316','#7c3aed'])
    ax.set_xlabel('Time window (2 s)');ax.set_ylabel('Flagged state count')
    ax.set_title('Anomaly discovery across time');ax.grid(alpha=.2)
    fig=ax.figure;fig.tight_layout();fig.savefig(FIG/'16_anomaly_timeline.png',dpi=220);plt.close(fig)
    summary={'states':len(df),'eligible_states':int(eligible.sum()),'raw_anomaly_states':int(df.raw_is_anomaly.sum()),
             'ae_anomaly_states':int(df.ae_is_anomaly.sum()),'both_branch_states':int(df.both_anomaly.sum()),
             'union_states':n,'candidate_events':len(events),
             'traffic_pattern_candidate_events':int((events.event_type=='traffic_pattern_candidate').sum())}
    (REPORT/'event_detection_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('=== EVENT DISCOVERY SUMMARY ===')
    print(json.dumps(summary,indent=2))
    print('\nTop candidate events:')
    print(events[['event_id','start_window','end_window','state_count','both_branch_states','event_type']].head(10).to_string(index=False))
    print('\nSaved:',OUTDATA,REPORT/'detected_events.csv',REPORT/'event_members.csv')
    print('Figures:',FIG/'15_anomaly_hotspots.png',FIG/'16_anomaly_timeline.png')

if __name__=='__main__':
    main()
