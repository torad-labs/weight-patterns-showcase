"""Plot full-passage observations with scope and missing correspondence visible."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pipeline.acquisition.run_a_baseline import ROOT, RUN


STAGES = ['initial','descent','after_loss','destination','comparison','final_fields']
LABELS = ['Initial energy','Descent','After loss / available','Required climb','Comparison','Final fields']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, default=RUN.parents[3]/'studies/04-round-2-overlap/runs/overlap')
    p.add_argument('--output', type=Path, default=ROOT/'figures/signal-trace-repair/A2-full-response')
    args=p.parse_args()
    report=json.loads((args.analysis/'response-map/index.json').read_text())
    groups=[json.loads((args.analysis/f'group-map/gpu{g}-comparison/index.json').read_text()) for g in [0,1]]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'#f6f5ef','axes.facecolor':'#f6f5ef'})
    fig=plt.figure(figsize=(16,11))
    fig.suptitle('Trace differences span the earlier calculation and the climb',
                 fontsize=21, fontweight='bold', x=.06, ha='left', y=.98, color='#234641')
    fig.text(.06,.936,'45 original natural responses · 21,839 predictions · all 105 connected merges in 35 layers',fontsize=12)
    grid=fig.add_gridspec(3,2,left=.15,right=.94,top=.85,bottom=.19,hspace=.66,wspace=.28,
                         height_ratios=[1,1,.92])
    axes=[fig.add_subplot(grid[i,:]) for i in range(2)]
    for gpu,(ax,group) in enumerate(zip(axes,groups)):
        by_role={r['role']:r for r in group['summaries']}
        data=np.full((len(STAGES),35),np.nan)
        text=[]
        for j,role in enumerate(STAGES):
            row=by_role.get(role)
            if row:
                values=np.array(row['direction_squared_by_layer'])
                data[j]=100*values/values.sum()
                text.append(f'{LABELS[j]}  ({row["positions"]})')
            else:
                text.append(f'{LABELS[j]}  (unmatched)')
        cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#dddcd4')
        image=ax.imshow(data,aspect='auto',cmap=cmap,vmin=0,vmax=20,interpolation='none')
        ax.set_yticks(range(6),text);ax.set_xticks(range(0,35,2))
        ax.set_title(f'Card {gpu}: separate failure/success group overlays — share of each passage’s difference',
                     loc='left',pad=12,fontsize=12)
        ax.axvline(9.5,color='#a78546',lw=1.2);ax.axvline(14.5,color='#a78546',lw=1.2)
        ax.set_xlabel('Layer · every cell combines its attention, feed-forward and per-layer-input merges')
        fig.colorbar(image,ax=ax,pad=.015,fraction=.023,label='% of passage difference')
    ax=fig.add_subplot(grid[2,0]); ax2=fig.add_subplot(grid[2,1])
    by_role={r['role']:r for r in report['grouped'] if r['relation']=='natural_failure_success'}
    x=np.arange(len(STAGES))
    direction=[by_role[r]['direction_total'] if r in by_role else np.nan for r in STAGES]
    gain=[by_role[r]['gain_total'] if r in by_role else np.nan for r in STAGES]
    short=['Initial','Descent','After loss','Climb','Compare','Final']
    ax.bar(x,direction,color='#238478',width=.65)
    ax2.bar(x,gain,color='#b18536',width=.65)
    for plot in [ax,ax2]:
        plot.set_xticks(x,short,rotation=25,ha='right');plot.grid(axis='y',alpha=.2)
        plot.set_axisbelow(True)
    ax.set_title('Supporting context comparisons: direction',loc='left',fontsize=12)
    ax.set_ylabel('Mean squared full-vector difference')
    ax2.set_title('Signal-size changes remain visible',loc='left',fontsize=12)
    ax2.set_ylabel('Mean squared difference in log gain')
    fig.text(.06,.105,'The same-card supporting comparisons cover 5,583 of 9,187 failure positions. 3,604 remain unmatched; three failures have no same-card designated success mate.',fontsize=10)
    fig.text(.06,.077,'All 45 responses participate in separate group overlays. Numbers in heatmap labels count aligned reference positions, not independent questions. Gray means unmatched.',fontsize=10)
    fig.text(.06,.049,'These are observed contextual differences, not an edit target or a capability map. Early numerical calculations are correct despite nonzero differences. Full vectors and graph links remain in the records.',fontsize=9.5)
    fig.text(.06,.023,'No new weights or inference. Lower plots weight positions, then mates, then questions; direction and gain use separate units. Source: round-2/overlap.',fontsize=9.5)
    args.output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:
        fig.savefig(args.output/f'full-response-map.{suffix}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
