from pathlib import Path
import json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dsta.grid_experiment import load_split,load_train_features,fixed_cells,density_cells,ROI
from dsta.grid_evaluation import evaluate_grid

# Candidate hyperparameters selected using VALIDATION only.
FIXED_SIZES=(2.,4.,8.,16.,32.,64.)
DENSITY_THRESHOLDS=(10,20,30,50,80,120,200,350,600,1000)

def main():
    train=load_train_features(ROOT)
    valid=load_split(ROOT,'valid')
    test=load_split(ROOT,'test')
    budget=200
    result_path=ROOT/'outputs/reports/grid_optimization_result.json'
    if not result_path.exists():raise FileNotFoundError('Run optimize_grid.py first')
    budget=json.loads(result_path.read_text())['budget']
    options=[]
    for size in FIXED_SIZES:
        cells=fixed_cells(size)
        if len(cells)<=budget:
            m=evaluate_grid(valid,cells)
            if m['coverage_rate']==1: options.append(('Fixed Grid',f'{size:g}m',cells,m))
    for threshold in DENSITY_THRESHOLDS:
        cells=density_cells(train,threshold=threshold)
        if len(cells)<=budget:
            m=evaluate_grid(valid,cells)
            if m['coverage_rate']==1: options.append(('Density Quadtree',f'threshold={threshold}',cells,m))
    selected=[]
    for method in ('Fixed Grid','Density Quadtree'):
        candidates=[x for x in options if x[0]==method]
        if not candidates:raise RuntimeError(f'No feasible {method} under budget {budget}')
        selected.append(min(candidates,key=lambda x:x[3]['mean_quantization_error_m']))
    optimized=pd.read_csv(ROOT/'data/processed/optimized_grid_cells.csv')
    val_metrics=evaluate_grid(valid,optimized)
    if len(optimized)>budget or val_metrics['coverage_rate']<1:
        raise RuntimeError('Optimized grid fails budget or coverage')
    selected.append(('Quality + Optuna','validation-selected',optimized,val_metrics))
    records=[]
    for method,config,cells,vm in selected:
        tm=evaluate_grid(test,cells)
        if tm['coverage_rate']<1:
            raise RuntimeError(f'{method} does not cover test ROI')
        record={'method':method,'configuration':config,'validation_mean_error_m':vm['mean_quantization_error_m'],**tm}
        records.append(record)
        print(method,config,'test mean:',round(tm['mean_quantization_error_m'],4),'cells:',len(cells))
    out=ROOT/'outputs/reports/grid_comparison.csv'
    pd.DataFrame(records).to_csv(out,index=False)
    print('Saved:',out)
if __name__=='__main__':main()
