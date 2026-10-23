from dataclasses import dataclass
import numpy as np
import pandas as pd
from dsta.grid_experiment import ROI, ROOT_SIZE, check_roi

@dataclass
class GridParams:
    w_density:float=.5
    w_motion:float=.3
    w_disagreement:float=.2
    split_threshold:float=.45
    min_size:float=2.
    root_size:float=ROOT_SIZE
    time_bucket_us:int=1_000_000

class QualityQuadtree:
    def __init__(self,params:GridParams,roi=ROI):
        self.params=params
        self.roi=roi
        self.scales={}

    @staticmethod
    def q75(s):
        a=np.asarray(s,dtype=float)
        a=a[np.isfinite(a)]
        return float(np.quantile(a,.75)) if len(a) else np.nan

    def features(self,part):
        n=len(part[['time_bucket','source','sequence_id','track_id']].drop_duplicates())
        return {'density':float(np.log1p(n)),
                'motion':self.q75(part.motion_score),
                'disagreement':self.q75(part.disagreement_score)}

    def score(self,part):
        f=self.features(part)
        weights={'density':self.params.w_density,'motion':self.params.w_motion,
                 'disagreement':self.params.w_disagreement}
        keys=[k for k in weights if np.isfinite(f[k])]
        denom=sum(weights[k] for k in keys)
        return sum(weights[k]*min(max(f[k],0)/self.scales[k],1) for k in keys)/denom if denom else 0.

    def fit(self,train):
        check_roi(self.roi,self.params.root_size)
        df=train.copy()
        t0=int(df.timestamp_us.min())
        df['time_bucket']=((df.timestamp_us.astype('int64')-t0)//self.params.time_bucket_us).astype('int64')
        xmin,ymin,xmax,ymax=self.roi
        r=self.params.root_size
        roots=[]
        for x in np.arange(xmin,xmax,r):
            for y in np.arange(ymin,ymax,r):
                part=df[(df.world_x>=x)&(df.world_x<x+r)&(df.world_y>=y)&(df.world_y<y+r)]
                roots.append((float(x),float(y),part))
        for key in ('density','motion','disagreement'):
            vals=[self.features(part)[key] for _,_,part in roots if not part.empty]
            vals=[v for v in vals if np.isfinite(v)]
            self.scales[key]=max(float(np.quantile(vals,.75)),1e-6) if vals else 1.
        def recurse(part,x,y,size,depth,path):
            score=self.score(part) if not part.empty else 0.
            if score<=self.params.split_threshold or size/2<self.params.min_size:
                return [{'grid_id':path,'x0':x,'y0':y,'size':size,'depth':depth,'quality_score':score}]
            half=size/2
            out=[]
            for ix in range(2):
                for iy in range(2):
                    xx,yy=x+ix*half,y+iy*half
                    child=part[(part.world_x>=xx)&(part.world_x<xx+half)&(part.world_y>=yy)&(part.world_y<yy+half)]
                    out.extend(recurse(child,xx,yy,half,depth+1,f'{path}{ix}{iy}'))
            return out
        records=[]
        for x,y,part in roots:
            records.extend(recurse(part,x,y,r,0,f'Q_{int(x)}_{int(y)}'))
        return pd.DataFrame(records)
