"""Views of observed component accounting, with context and missing layers visible."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline.overlap.analyze import read
from pipeline.research_guard.verify_round2 import verify,ROOT


def main(output):
    if verify(ROOT):raise ValueError('Round-two verifier refused')
    output=Path(output);data=read(output/'components/summary.json');profiles=np.load(output/'components/profiles.npz');dest=output/'figures'
    fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained');fig.patch.set_facecolor('#f7f6f1')
    colors=['#247f78','#d16b4b']
    for gpu in (0,1):
        r=next(r for r in data['records'] if r['gpu']==gpu and r['layer']==13 and r['kind']=='feedforward' and r['coverage']=='all_aligned' and r['axis']=='raw' and r['role']=='destination')
        v=profiles[r['profile_key']];curve=np.cumsum(np.sort(abs(v))[::-1]);curve/=curve[-1]
        axes[0,0].plot(np.arange(1,len(v)+1),curve,color=colors[gpu],label=f"GPU {gpu}: effective count {r['effective_absolute_components']:.0f} / {len(v)}")
        layers=sorted({x['layer'] for x in data['records']});vals=[];contexts=[]
        for layer in layers:
            row=next(r for r in data['records'] if r['gpu']==gpu and r['layer']==layer and r['kind']=='feedforward' and r['coverage']=='all_aligned' and r['axis']=='raw' and r['role']=='destination')
            vals.append(row['signed_component_sum']);contexts.append(row['context_share'])
        axes[0,1].plot(layers,vals,color=colors[gpu],label=f'GPU {gpu} weighted FF contribution')
        axes[0,1].plot(layers,contexts,color=colors[gpu],ls=':',label=f'GPU {gpu} incoming-vector term')
        roles=['opening','initial','descent','after_loss','comparison','final_fields'];arr=np.full((len(roles),35),np.nan)
        for row in data['sharing']:
            if row['gpu']==gpu and row['kind']=='feedforward' and row['coverage']=='all_aligned' and row['axis']=='raw' and row['passage'] in roles:
                arr[roles.index(row['passage']),row['layer']]=row['signed_component_profile_cosine_to_climb']
        ax=axes[1,gpu];cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#d9dcd7');im=ax.imshow(arr,aspect='auto',vmin=-1,vmax=1,cmap=cmap)
        ax.set_yticks(np.arange(len(roles)),['Opening','Initial energy','Descent','After loss','Comparison','Final fields']);ax.set_xticks(np.arange(0,35,5));ax.set_xlabel('Native layer');ax.set_title(f'GPU {gpu} · similarity to the climb contribution profile',loc='left')
        fig.colorbar(im,ax=ax,label='Cosine of signed unit-contribution profiles',shrink=.8)
    n=len(v);axes[0,0].plot([0,n],[0,1],ls='--',color='#aaa',label='Equal absolute contribution from every unit')
    axes[0,0].set_xlabel('Units, ordered by absolute contribution');axes[0,0].set_ylabel('Cumulative absolute contribution');axes[0,0].set_title('Layer 13 feed-forward · final-climb group contrast',loc='left');axes[0,0].legend(fontsize=9,frameon=False)
    axes[0,1].set_title('The direction metric includes the incoming signal',loc='left');axes[0,1].set_xlabel('Native layer');axes[0,1].set_ylabel('Signed share along observed group contrast');axes[0,1].legend(fontsize=9,frameon=False)
    for ax in axes.flat:ax.set_facecolor('#f7f6f1');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Observed trace differences are distributed across active contributions',fontsize=21,x=.04,ha='left')
    fig.supxlabel('All aligned subsets; populations differ between cards. Contributions use recorded inputs and native weights. No component was ablated.\nThe incoming-vector term is algebra induced by unit(outgoing) − unit(incoming), not another learned branch. Gray cells have no retained component inputs.\nThese component profiles are not speed-controlled; the held-out/equal-speed analysis is reported separately.',fontsize=10)
    fig.savefig(dest/'component-accounting.png',dpi=170);fig.savefig(dest/'component-accounting.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();main(a.output)
