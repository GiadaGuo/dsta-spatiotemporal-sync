"""Shared ROI, grid builders, and reproducible data loading."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Explicit, predeclared study region in DAIR world coordinates (metres).
# These bounds cover the previously inspected example; do not derive them from test data.
ROI = (3584.0, 2112.0, 3968.0, 2496.0)  # xmin, ymin, xmax, ymax
ROOT_SIZE = 64.0


def check_roi(roi=ROI, root_size=ROOT_SIZE):
    xmin, ymin, xmax, ymax = roi
    if xmax <= xmin or ymax <= ymin:
        raise ValueError('Invalid ROI')
    for v in roi:
        if not np.isclose(v / root_size, round(v / root_size)):
            raise ValueError('ROI boundaries must align with root_size')


def fixed_cells(size, roi=ROI):
    xmin, ymin, xmax, ymax = roi
    if (xmax-xmin) % size or (ymax-ymin) % size:
        raise ValueError('Fixed size must divide ROI extent')
    records = []
    for ix, x in enumerate(np.arange(xmin, xmax, size)):
        for iy, y in enumerate(np.arange(ymin, ymax, size)):
            records.append({'grid_id':f'F{size:g}_{ix}_{iy}', 'x0':float(x), 'y0':float(y), 'size':float(size), 'depth':0})
    return pd.DataFrame(records)


def density_cells(train, threshold=30, min_size=2.0, roi=ROI, root_size=ROOT_SIZE, bucket_us=1_000_000):
    """Full-ROI density quadtree; counts unique source/sequence/track per time bucket."""
    check_roi(roi, root_size)
    df = train.copy()
    t0 = int(df.timestamp_us.min())
    df['time_bucket'] = ((df.timestamp_us.astype('int64') - t0) // bucket_us).astype('int64')
    def recurse(part, x0, y0, size, depth, path):
        n = len(part[['time_bucket','source','sequence_id','track_id']].drop_duplicates())
        if n <= threshold or size/2 < min_size:
            return [{'grid_id':path,'x0':x0,'y0':y0,'size':size,'depth':depth,'occupancy':n}]
        half = size/2
        out = []
        for ix in range(2):
            for iy in range(2):
                xx, yy = x0 + ix*half, y0 + iy*half
                child = part[(part.world_x >= xx)&(part.world_x < xx+half)&(part.world_y >= yy)&(part.world_y < yy+half)]
                out.extend(recurse(child, xx, yy, half, depth+1, f'{path}{ix}{iy}'))
        return out
    xmin,ymin,xmax,ymax = roi
    records=[]
    for x in np.arange(xmin,xmax,root_size):
        for y in np.arange(ymin,ymax,root_size):
            part=df[(df.world_x>=x)&(df.world_x<x+root_size)&(df.world_y>=y)&(df.world_y<y+root_size)]
            records.extend(recurse(part,float(x),float(y),float(root_size),0,f'D_{int(x)}_{int(y)}'))
    return pd.DataFrame(records)


def load_split(root, name):
    return pd.read_parquet(Path(root)/'data/processed/splits'/f'{name}.parquet')


def load_train_features(root):
    from dsta.grid_features import prepare_features
    root=Path(root)
    train=load_split(root,'train')
    matches=pd.read_csv(root/'outputs/reports/spatial_matches.csv', dtype={
        'vehicle_frame':str,'infrastructure_frame':str,
        'vehicle_track_id':str,'infrastructure_track_id':str})
    veh=set(zip(train.loc[train.source=='vehicle-side','frame_id'].astype(str),
                train.loc[train.source=='vehicle-side','track_id'].astype(str)))
    inf=set(zip(train.loc[train.source=='infrastructure-side','frame_id'].astype(str),
                train.loc[train.source=='infrastructure-side','track_id'].astype(str)))
    keep=[(str(r.vehicle_frame),str(r.vehicle_track_id)) in veh and
          (str(r.infrastructure_frame),str(r.infrastructure_track_id)) in inf
          for r in matches.itertuples(index=False)]
    return prepare_features(train, matches.loc[keep].copy())
