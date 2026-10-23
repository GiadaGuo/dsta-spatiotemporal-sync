from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'outputs/reports/grid_comparison.csv')
out=ROOT/'outputs/figures';out.mkdir(parents=True,exist_ok=True)
colors=['#2563eb','#d97706','#7c3aed']
fig,ax=plt.subplots(figsize=(9,5))
x=range(len(df))
ax.bar([i-.18 for i in x],df.mean_quantization_error_m,width=.36,label='Mean',color='#2563eb')
ax.bar([i+.18 for i in x],df.p90_quantization_error_m,width=.36,label='P90',color='#d97706')
ax.set_xticks(list(x),df.method,rotation=8)
ax.set_ylabel('Quantization error (m)')
ax.set_title('Held-out Test: Common ROI and Grid Budget')
ax.legend()
fig.tight_layout();fig.savefig(out/'06_grid_test_errors.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,5))
for i,row in df.iterrows():
    ax.scatter(row.n_cells,row.mean_quantization_error_m,s=90,color=colors[i%len(colors)])
    ax.annotate(row.method,(row.n_cells,row.mean_quantization_error_m),xytext=(6,6),textcoords='offset points')
ax.set_xlabel('Total leaf cells (including empty cells)')
ax.set_ylabel('Test mean quantization error (m)')
ax.set_title('Grid Cost vs Quantization Error')
fig.tight_layout();fig.savefig(out/'07_grid_tradeoff.png',dpi=200);plt.close(fig)
print('Saved:',out/'06_grid_test_errors.png',out/'07_grid_tradeoff.png')
