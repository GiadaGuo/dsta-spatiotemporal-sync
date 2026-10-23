from pathlib import Path
import json
import pandas as pd
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dsta.grid_experiment import ROI,check_roi

def main():
    check_roi()
    tracks=pd.read_parquet(ROOT/'data/processed/world_tracks.parquet')
    tracks['timestamp_us']=tracks.timestamp_us.astype('int64')
    times=tracks.timestamp_us.drop_duplicates().sort_values().to_numpy()
    if len(times)<10: raise ValueError('Too few timestamps')
    cut1=times[int(len(times)*.70)]
    cut2=times[int(len(times)*.85)]
    out=ROOT/'data/processed/splits';out.mkdir(parents=True,exist_ok=True)
    parts={'train':tracks[tracks.timestamp_us<cut1],
           'valid':tracks[(tracks.timestamp_us>=cut1)&(tracks.timestamp_us<cut2)],
           'test':tracks[tracks.timestamp_us>=cut2]}
    xmin,ymin,xmax,ymax=ROI
    summary={'roi':ROI,'train_valid_boundary_us':int(cut1),'valid_test_boundary_us':int(cut2)}
    for name,part in parts.items():
        part.to_parquet(out/f'{name}.parquet',index=False)
        outside=~((part.world_x>=xmin)&(part.world_x<xmax)&(part.world_y>=ymin)&(part.world_y<ymax))
        summary[name]={'rows':len(part),'outside_roi':int(outside.sum())}
        print(name,len(part),'outside ROI:',int(outside.sum()))
    (out/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if any(summary[k]['outside_roi'] for k in parts):
        raise ValueError('ROI does not cover all observations; choose an explicit ROI before optimization')
if __name__=='__main__':main()
