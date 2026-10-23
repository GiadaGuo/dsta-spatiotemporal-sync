"""Common grid evaluation, including coverage and grid count."""
import numpy as np
import pandas as pd


def assign_points_to_cells(points, cells):
    x=points.world_x.to_numpy(dtype=float)
    y=points.world_y.to_numpy(dtype=float)
    assigned=np.full(len(points),-1,dtype=np.int64)
    # Rectangles are non-overlapping. Group by size for vectorized index lookup.
    for size, group in cells.groupby('size',sort=False):
        size=float(size)
        lookup={(round(float(r.x0),8),round(float(r.y0),8)):int(idx)
                for idx,r in group.iterrows()}
        xmin=min(lookup, key=lambda t:t[0])[0]
        ymin=min(lookup, key=lambda t:t[1])[1]
        # Lookup by global grid origin, not group-specific origin: coarse cells may be sparse.
        # Group-specific minimum is still aligned to the size lattice.
        xx=np.floor((x-xmin)/size)*size+xmin
        yy=np.floor((y-ymin)/size)*size+ymin
        for i in np.flatnonzero(assigned<0):
            idx=lookup.get((round(float(xx[i]),8),round(float(yy[i]),8)))
            if idx is not None:
                assigned[i]=idx
    result=points.copy()
    result['grid_id']=pd.NA
    result['grid_center_x']=np.nan
    result['grid_center_y']=np.nan
    ok=assigned>=0
    if ok.any():
        selected=cells.loc[assigned[ok]]
        result.loc[ok,'grid_id']=selected.grid_id.to_numpy()
        result.loc[ok,'grid_center_x']=selected.x0.to_numpy()+selected['size'].to_numpy()/2
        result.loc[ok,'grid_center_y']=selected.y0.to_numpy()+selected['size'].to_numpy()/2
    return result


def evaluate_grid(points,cells):
    assigned=assign_points_to_cells(points,cells)
    valid=assigned.grid_id.notna().to_numpy()
    n=int(valid.sum())
    if n==0:
        return {'n_observations':len(points),'covered_observations':0,'coverage_rate':0.,
                'n_cells':len(cells),'occupied_cells':0,'mean_quantization_error_m':None,
                'median_quantization_error_m':None,'p90_quantization_error_m':None,
                'max_quantization_error_m':None}
    d=np.hypot(assigned.loc[valid,'world_x']-assigned.loc[valid,'grid_center_x'],
               assigned.loc[valid,'world_y']-assigned.loc[valid,'grid_center_y'])
    return {'n_observations':len(points),'covered_observations':n,
            'coverage_rate':n/len(points),'n_cells':len(cells),
            'occupied_cells':int(assigned.loc[valid,'grid_id'].nunique()),
            'mean_quantization_error_m':float(d.mean()),
            'median_quantization_error_m':float(d.median()),
            'p90_quantization_error_m':float(d.quantile(.9)),
            'max_quantization_error_m':float(d.max())}
