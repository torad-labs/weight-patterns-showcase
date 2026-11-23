"""Show common trace portions together with the population that supports them."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline.overlap.analyze import read,write,finite
from pipeline.native_records.run_declaration import check_native_run

LABELS={'opening':'Opening','initial':'Initial energy','descent':'Descent','after_loss':'After loss','destination':'Final climb','comparison':'Comparison','final_fields':'Final fields'}
COLORS={'opening':'#a3aaa8','initial':'#7184a6','descent':'#3a8d8e','after_loss':'#b89939','destination':'#df694a','comparison':'#855e9c','final_fields':'#325b52'}

def main(output):
    output=Path(output);dest=output/'figures';dest.mkdir(exist_ok=True)
    # Rendering is a view of declared analysis, not another statistical selection.
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
    fig.patch.set_facecolor('#f7f6f1')
    fig.suptitle('The common portion is strongest around the final climb\nCoverage determines how widely that observation holds',fontsize=20,x=.04,ha='left')
    records=[]
    for gpu in (0,1):
        meta=read(output/f'common-coordinates/gpu{gpu}/index.json')
        for row,coverage in enumerate(('all_aligned','full_population')):
            ax=axes[row,gpu];ax.set_facecolor('#f7f6f1')
            for s in meta['summaries']:
                if s['coverage']!=coverage:continue
                v=np.array(s['common_squared_by_merge']).reshape(-1,3).sum(axis=1)
                ax.plot(np.arange(len(v)),v,label=f"{LABELS[s['role']]} ({s['positions']} positions)",color=COLORS[s['role']],lw=1.7)
                raw=sum(s['raw_squared_by_merge']);common=sum(s['common_squared_by_merge'])
                records.append(dict(gpu=gpu,coverage=coverage,passage=s['role'],positions=s['positions'],
                    raw_squared=raw,common_squared=common,squared_norm_ratio=common/raw if raw else None))
            pop=meta['population'];subtitle='All aligned subsets' if row==0 else f"Every member: {pop['failures']} failures + {pop['successes']} successes"
            ax.set_title(f'GPU {gpu} · {subtitle}',loc='left',fontsize=13)
            ax.set_xlabel('Native layer');ax.set_ylabel('Common gap squared · mean over aligned positions')
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.set_xlim(0,34);ax.grid(axis='y',alpha=.2)
            ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8,frameon=False)
    fig.supxlabel('Each curve sums the three measured merge types within a layer. Panels use their own vertical scale.\nA coordinate is retained only when every included cross-group difference has its sign. A smaller or zero gap does not rule out a distributed pattern.',fontsize=10)
    fig.savefig(dest/'common-trace-and-coverage.png',dpi=170);fig.savefig(dest/'common-trace-and-coverage.svg');plt.close(fig)
    with (dest/'passage-summary.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write(dest/'passage-summary.json',finite(records))
    # Separate-groups view at the strongest fully covered observed position/merge.
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained');fig.patch.set_facecolor('#f7f6f1')
    chosen=[]
    for gpu,ax in enumerate(axes):
        directory=output/f'common-coordinates/gpu{gpu}';meta=read(directory/'index.json');metrics=np.load(directory/'metrics.npy')
        ids=np.array([i for i,r in enumerate(meta['rows']) if r['full_population']]);local,merge=np.unravel_index(np.argmax(metrics[ids,:,1]),metrics[ids,:,1].shape);pos=int(ids[local])
        gap=np.load(directory/'gap.npy',mmap_mode='r')[pos,merge];coord=int(np.argmax(abs(gap)))
        bounds={key:np.load(directory/(key+'.npy'),mmap_mode='r')[pos,merge,coord] for key in ('failure_min','failure_max','success_min','success_max')}
        groups=[]
        for label,color,lo,hi in [('Failures','#c26b50',bounds['failure_min'],bounds['failure_max']),('Successes','#248a81',bounds['success_min'],bounds['success_max'])]:
            y=len(groups);ax.plot([lo,hi],[y,y],lw=12,color=color,solid_capstyle='butt');groups.append(label)
        ax.set_yticks([0,1],groups);ax.set_ylim(-.5,1.5);ax.set_facecolor('#f7f6f1');ax.grid(axis='x',alpha=.2);ax.spines[['top','right','left']].set_visible(False)
        row=meta['rows'][pos];ax.set_title(f"GPU {gpu} · {LABELS[row['role']]} · layer {merge//3}\nMerge {merge}, residual coordinate {coord}",loc='left');ax.set_xlabel('Recorded change in this native direction coordinate')
        chosen.append(dict(gpu=gpu,position=pos,merge=int(merge),coordinate=coord,row=row,bounds=bounds,
            selection='Largest common-gap norm with full-population alignment; coordinate with largest absolute common gap there. Exploratory display only, not construction selection.'))
    fig.suptitle('Separate group ranges at the strongest fully covered common-gap locations',fontsize=16)
    fig.supxlabel('Each bar spans every included member. The gap is common to those observed members; it does not identify a cause.\nThe displayed coordinate is selected from the measured map. All coordinates and omission behavior remain in the data.',fontsize=10)
    fig.savefig(dest/'separate-group-ranges.png',dpi=170);fig.savefig(dest/'separate-group-ranges.svg');plt.close(fig)
    write(dest/'range-examples.json',finite(chosen))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();main(args.output)
