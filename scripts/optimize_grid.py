from pathlib import Path
import json,sys
import numpy as np
import optuna
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dsta.grid_experiment import load_split,load_train_features,ROI
from dsta.grid_evaluation import evaluate_grid
from dsta.quality_quadtree import GridParams,QualityQuadtree

BUDGET=200
N_TRIALS=80
PENALTY=20.

def main():
    train=load_train_features(ROOT)
    valid=load_split(ROOT,'valid')
    reports=ROOT/'outputs/reports';reports.mkdir(parents=True,exist_ok=True)
    def objective(trial):
        p=GridParams(w_density=trial.suggest_float('w_density',.01,1.),
                     w_motion=trial.suggest_float('w_motion',.01,1.),
                     w_disagreement=trial.suggest_float('w_disagreement',.01,1.),
                     split_threshold=trial.suggest_float('split_threshold',.1,1.0))
        cells=QualityQuadtree(p,roi=ROI).fit(train)
        metrics=evaluate_grid(valid,cells)
        n=metrics['n_cells'];coverage=metrics['coverage_rate']
        trial.set_user_attr('n_cells',n)
        trial.set_user_attr('coverage_rate',coverage)
        trial.set_user_attr('valid_error_m',metrics['mean_quantization_error_m'])
        if coverage<1: return 1e6
        return metrics['mean_quantization_error_m']+PENALTY*max(0.,n/BUDGET-1.)**2
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=True)
    study.trials_dataframe().to_csv(reports/'grid_optimization_trials.csv',index=False)
    feasible=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE
              and t.user_attrs.get('n_cells',float('inf'))<=BUDGET
              and t.user_attrs.get('coverage_rate',0)==1.
              and t.user_attrs.get('valid_error_m') is not None]
    if not feasible:
        print('No feasible trial under budget; try increasing trials or budget. No test evaluation performed.')
        return
    best=min(feasible,key=lambda t:t.user_attrs['valid_error_m'])
    cells=QualityQuadtree(GridParams(**best.params),roi=ROI).fit(train)
    cells.to_csv(ROOT/'data/processed/optimized_grid_cells.csv',index=False)
    result={'budget':BUDGET,'roi':ROI,'best_trial':best.number,'best_params':best.params,
            'validation':evaluate_grid(valid,cells),'train_only_fit':True}
    (reports/'grid_optimization_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('\n=== BEST FEASIBLE VALIDATION RESULT ===')
    print(json.dumps(result,indent=2))
    print('Test remains untouched. Run compare_grids.py for the single final test evaluation.')
if __name__=='__main__':main()
