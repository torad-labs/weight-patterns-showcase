"""Render the actual group and individual evidence without selecting an edit."""
import argparse
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import read, write, finite


def view(root,name):
    p=root/name
    return read(p/'summary.json'),read(p/'members.json'),np.load(p/'aligned.npy'),np.load(p/'evidence.npz')


def summarize(root):
    root=Path(root);meta,members,aligned,e=view(root,'calculation')
    pm,_,pa,pe=view(root,'pre-error')
    roles=np.array(meta['roles'],dtype=object)
    failure=[i for i,m in enumerate(members) if m['role']=='failure']
    passages=list(dict.fromkeys(r for r in meta['roles'] if r))
    rows=[];profile=[]
    for i,m in enumerate(members):
        row=dict(id=m['id'],role=m['role'],parameters=m['parameters'],family=m['family'],
            total_predictions=m['total'],aligned_predictions=m['aligned'],contexts=[])
        for role in passages:
            for name,aa,ev,subset in [('all',aligned,e,np.ones(len(roles),bool)),
                                      ('pre_error_peers',pa,pe,np.ones(len(roles),bool)),
                                      ('error_conditioned',aligned,e,np.ones(len(roles),bool))]:
                if name=='error_conditioned':
                    err=m['label']['first_quantitative_error']
                    if err is None:continue
                    subset=aligned[i]>err['first_wrong_operator']['decision']
                ix=np.flatnonzero((roles==role)&(aa[i]>=0)&subset&np.isfinite(ev['margins'][i]).any(1))
                if not len(ix):continue
                v=np.asarray(ev['margins'][i,ix],dtype='f8');bymerge=np.nanmean(v,axis=0)
                row['contexts'].append(dict(role=role,view=name,positions=len(ix),
                    first_prediction=int(aa[i,ix].min()),last_prediction=int(aa[i,ix].max()),
                    mean_margin=float(np.nanmean(v)),median_cell_margin=float(np.nanmedian(v)),
                    positive_merge_means=int((bymerge>0).sum()),valid_merges=int(np.isfinite(bymerge).sum()),
                    margin_by_merge=bymerge))
        rows.append(row)
    for name,mm,ev in [('all',meta,e),('pre_error_peers',pm,pe)]:
        for s in mm['summaries']:
            v=np.array(s['squared_by_merge'],dtype=float)
            peak=int(np.nanargmax(v));layers=np.nansum(v.reshape(-1,3),axis=1)
            profile.append(dict(view=name,role=s['role'],positions=s['positions'],
                counts_min=s['counts_min'],counts_max=s['counts_max'],
                full_population_positions=s['full_population_positions'],peak_merge=peak,
                peak_layer=meta['connections'][peak]['layer'],peak_kind=meta['connections'][peak]['kind'],
                squared_by_layer=layers,total_squared=float(np.nansum(v)),
                strict_squared=float(np.nansum(s['strict_squared_by_merge'])),
                strict_cells=s['strict_cells']))
    result=dict(population=meta['population'],reference=meta['reference'],members=rows,passages=profile,
                scope='Positive margins are descriptive held-out agreement, not a causal test or capability purity. Pre-error view excludes error-conditioned failures from the direction itself.')
    write(root/'trace-summary.json',finite(result))
    return result


def figures(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    root=Path(root);dest=root/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f5','axes.facecolor':'#faf9f5'})
    meta,members,aligned,e=view(root,'calculation');pm,_,pa,pe=view(root,'pre-error')
    passages=[r for r in dict.fromkeys(meta['roles']) if r]
    fig,axs=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    matrices=[]
    for kind in ['all','pre_error_peers']:
        matrix=np.full((len(passages),35),np.nan)
        for s in summary['passages']:
            if s['view']==kind and s['role'] in passages:matrix[passages.index(s['role'])]=s['squared_by_layer']
        matrices.append(matrix)
    hi=max(np.nanmax(x) for x in matrices)
    for ax,m,title in zip(axs,matrices,['Whole answer: both group overlays','Peers still before their own error']):
        im=ax.imshow(m,aspect='auto',vmin=0,vmax=hi,cmap='magma');ax.set_title(title)
        ax.set_yticks(range(len(passages)),[p.replace('_',' ') for p in passages]);ax.set_xlabel('Layer (three connected merges summed)')
    fig.colorbar(im,ax=axs,label='Mean squared difference between group turns')
    fig.suptitle('Where the observed success–failure trace difference lies',fontsize=17)
    fig.savefig(dest/'location.png',dpi=160);plt.close(fig)

    failures=[r for r in summary['members'] if r['role']=='failure']
    fig,axs=plt.subplots(1,2,figsize=(16,10),layout='constrained');matrices=[]
    for kind in ['pre_error_peers','error_conditioned']:
        matrix=np.full((len(failures),35),np.nan)
        for i,r in enumerate(failures):
            cs=[c for c in r['contexts'] if c['role']=='destination' and c['view']==kind]
            if cs:matrix[i]=np.nanmean(np.array(cs[0]['margin_by_merge'],dtype=float).reshape(35,3),axis=1)
        matrices.append(matrix)
    bound=max(np.nanmax(abs(x)) for x in matrices)
    for ax,m,title in zip(axs,matrices,['Climb, only pre-error contributors','Climb, after this failure emitted the error']):
        im=ax.imshow(m,aspect='auto',vmin=-bound,vmax=bound,cmap='BrBG');ax.set_title(title)
        ax.set_yticks(range(len(failures)),[f"{r['id']}  ({r['parameters']['m']}, {r['parameters']['v']}, {r['parameters']['drop']})" for r in failures]);ax.set_xlabel('Layer')
    fig.colorbar(im,ax=axs,label='Held-out own-side margin: positive = agreement')
    fig.suptitle('Every failure: does its own trace carry the group difference?\nLabels include (mass, speed, descent); blanks mean missing alignment',fontsize=15)
    fig.savefig(dest/'failures.png',dpi=160);plt.close(fig)

    pre_climb=[s for s in summary['passages'] if s['view']=='pre_error_peers' and s['role']=='destination']
    if pre_climb:
        merge=pre_climb[0]['peak_merge'];fig,axs=plt.subplots(5,4,figsize=(17,15),layout='constrained')
        for ax,r in zip(axs.flat,failures):
            i=meta['members'].index(r['id']);valid=(aligned[i]>=0)&np.isfinite(e['margins'][i,:,merge]);vp=(pa[i]>=0)&np.isfinite(pe['margins'][i,:,merge])
            ax.plot(aligned[i,valid],e['margins'][i,valid,merge],color='#a97843',alpha=.7,label='all peers')
            ax.plot(pa[i,vp],pe['margins'][i,vp,merge],color='#26756c',label='pre-error peers')
            error=members[i]['label']['first_quantitative_error'];ax.axvline(error['first_wrong_operator']['decision'],color='#9c3f31',linestyle='--');ax.axhline(0,color='#888',linewidth=.6)
            ax.set_title(r['id'],fontsize=9);ax.set_xlabel('Actual generated prediction')
        for ax in axs.flat[len(failures):]:ax.set_visible(False)
        axs.flat[0].legend(fontsize=7)
        fig.suptitle(f"Individual timing at the measured pre-error climb peak: L{pre_climb[0]['peak_layer']} {pre_climb[0]['peak_kind']}\nDashed line: prediction emitting the wrong operator; curves join only retained observations",fontsize=15)
        fig.savefig(dest/'timing.png',dpi=150);plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,parameter in zip(axs,['m','v','drop']):
        for outcome,color in [('failure','#a97843'),('natural_numerical_success','#26756c')]:
            xy=[(r['parameters'][parameter],c['mean_margin']) for r in summary['members'] if r['role']==outcome for c in r['contexts'] if c['view']=='pre_error_peers' and c['role']=='destination']
            if xy:ax.scatter(*zip(*xy),color=color,label=outcome.replace('natural_numerical_',''),alpha=.75)
        ax.axhline(0,color='#888',linewidth=.7);ax.set_xlabel(parameter);ax.set_ylabel('Mean pre-error climb margin')
    axs[0].legend();fig.suptitle('Parameter dependence on these same responses — no new sample or fitted gate')
    fig.savefig(dest/'parameters.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    result=summarize(a.output);figures(a.output,result)
