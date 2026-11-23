"""Figures distinguish raw group differences from recurrence evidence."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROLES=['opening','initial','descent','after_loss','destination','comparison','final_fields']
LABELS=['Opening','Initial energy','Descent','After loss','Required climb','Comparison','Final fields']

def render(output):
    output=Path(output);folder=output/'figures';folder.mkdir(exist_ok=True)
    data=[json.loads((output/f'contrasts/gpu{gpu}/index.json').read_text()) for gpu in (0,1)]
    panels=[]
    for d in data:
        byrole={r['role']:r for r in d['summaries']};values=np.full((len(ROLES),35),np.nan)
        support=np.full_like(values,np.nan);coverage=[]
        for row,role in enumerate(ROLES):
            if role not in byrole:coverage.append('No alignment');continue
            r=byrole[role];values[row]=np.array(r['difference_squared_by_merge']).reshape(35,3).sum(axis=1)
            support[row]=np.array(r['supported_positions_by_merge']).reshape(35,3).sum(axis=1)/(3*r['positions'])
            coverage.append(f"{r['positions']} positions; full group {r['full_population_positions']}")
        panels.append((values,support,coverage))
    fig,axs=plt.subplots(2,2,figsize=(18,9),layout='constrained')
    vmax=max(np.nanmax(x[0]) for x in panels)
    for gpu,(values,support,coverage) in enumerate(panels):
        a=axs[gpu,0];im=a.imshow(values,aspect='auto',cmap='viridis',vmin=0,vmax=vmax)
        a.set_title(f'GPU {gpu}: raw difference between the two group means',loc='left')
        a.set_yticks(range(len(ROLES)),LABELS);a.set_xticks(range(0,35,2));a.set_xlabel('Layer (sum of its three measured merges)')
        b=axs[gpu,1];other=b.imshow(support,aspect='auto',cmap='Blues',vmin=0,vmax=1)
        b.set_title('Whole-vector leave-one-out support — not a context-filtered component',loc='left',fontsize=11)
        b.set_yticks(range(len(ROLES)),coverage);b.set_xticks(range(0,35,2));b.set_xlabel('Layer')
    fig.colorbar(im,ax=axs[:,0],label='Mean squared difference in recorded direction changes')
    fig.colorbar(other,ax=axs[:,1],label='Share of aligned position/merge cells with unanimous support')
    fig.suptitle('Successful and failing groups separate along the response\nRaw separation alone does not identify the shared capability contribution',fontsize=17)
    fig.savefig(folder/'raw-overlap-and-coverage.png',dpi=160);fig.savefig(folder/'raw-overlap-and-coverage.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();render(a.output)
