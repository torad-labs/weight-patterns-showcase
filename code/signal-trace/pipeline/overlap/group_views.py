"""Show independent group overlays before inspecting between-group support.

Reads means, within-group dispersion, native turns, gain and alignment only.
No held-out score, contrast direction or prior overlap supplies these views.
"""
import argparse
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import read,write,finite


def main(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'group-overlays';out.mkdir(exist_ok=True)
    calc=root/'calculation';meta=read(calc/'summary.json');members=read(calc/'members.json')
    aligned=np.load(calc/'aligned.npy');stats=np.load(calc/'evidence.npz')
    metrics=stats['group_metrics'];counts=stats['counts']
    names=['failure','success'];roles=np.array(meta['roles'],dtype=object)
    display=[r for r in dict.fromkeys(meta['roles']) if r]
    means=[np.load(calc/f'{g}-mean.npy',mmap_mode='r') for g in names]
    mean2=[np.sum(x*x,axis=-1) for x in means]
    # Mean squared vector = squared group mean + population dispersion.
    coherence=[np.divide(mean2[g],mean2[g]+metrics[g,:,:,0],
        out=np.full_like(mean2[g],np.nan),where=(mean2[g]+metrics[g,:,:,0])>0) for g in range(2)]
    for g in range(2):coherence[g][counts[:,g]<2]=np.nan
    report=[]
    for g,name in enumerate(names):
        for role in display:
            pp=np.flatnonzero((roles==role)&(counts[:,g]>=2))
            if not len(pp):continue
            m=np.nanmean(mean2[g][pp],axis=0);v=np.nanmean(metrics[g,pp,:,0],axis=0)
            report.append(dict(group=name,role=role,positions=len(pp),members_min=int(counts[pp,g].min()),members_max=int(counts[pp,g].max()),
                mean_turn_squared_by_merge=m,dispersion_by_merge=v,
                coherence_by_merge=m/(m+v),
                mean_log_gain_by_merge=np.nanmean(metrics[g,pp,:,1],axis=0),
                log_gain_variance_by_merge=np.nanmean(metrics[g,pp,:,2],axis=0),
                total_mean_squared=float(np.nansum(m)),total_dispersion=float(np.nansum(v)),
                pooled_coherence=float(np.nansum(m)/np.nansum(m+v)),
                largest_mean_merge=int(np.nanargmax(m)),largest_spread_merge=int(np.nanargmax(v))))
    write(out/'summary.json',finite(dict(definition='Coherence = squared length of group mean turn / mean squared length of member turns. One means identical vectors; lower values indicate disagreement in direction or size. No score cutoff. Dispersion is mean squared Euclidean distance from the group mean.',groups=report,source='calculation/failure-mean.npy and success-mean.npy; group_metrics only from evidence.npz')))
    plt.rcParams.update({'figure.facecolor':'#faf9f5','axes.facecolor':'#faf9f5','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    marks=[(int(np.flatnonzero(roles==r)[0]),r.replace('_',' ')) for r in display]
    marks=[(p,'opening / initial' if r=='opening' else r) for p,r in marks if r!='initial']
    fig,axs=plt.subplots(2,2,figsize=(17,11),layout='constrained')
    vmax=max(np.nanmax(np.sqrt(x)) for x in mean2)
    for column,g in enumerate([1,0]):
        title='25 numerical successes' if g==1 else '20 climb failures'
        im=axs[0,column].imshow(coherence[g].T,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
        axs[0,column].set_title(title+' — agreement within the group')
        mag=axs[1,column].imshow(np.sqrt(mean2[g]).T,origin='lower',aspect='auto',vmin=0,vmax=vmax,cmap='magma')
        axs[1,column].set_title(title+' — mean turn magnitude')
        for ax in axs[:,column]:
            ax.set_yticks([0,40,104],['L0 attention','L13 feed-forward','L34 embedding']);ax.set_ylabel('Actual residual merge (105)')
            ax.set_xticks([p for p,r in marks],[r for p,r in marks],rotation=40,ha='right');ax.set_xlabel('Aligned response position in block-base')
    fig.colorbar(im,ax=axs[0,:],label='Fraction of member turn energy retained by the group mean')
    fig.colorbar(mag,ax=axs[1,:],label='Length of the mean direction change')
    fig.suptitle('The two overlays, separately — every retained position and merge\nAgreement and spread are measured inside each group; no success-minus-failure direction is used',fontsize=16)
    fig.savefig(out/'groups.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(16,5.5),layout='constrained')
    top=float(np.sqrt(np.nanmax(metrics[:,:,:,0])))
    for ax,g in zip(axs,[1,0]):
        dispersion=np.sqrt(metrics[g,:,:,0]).copy();dispersion[counts[:,g]<2]=np.nan
        im=ax.imshow(dispersion.T,origin='lower',aspect='auto',vmin=0,vmax=top,cmap='magma')
        ax.set_title('Successes: within-group spread' if g==1 else 'Failures: within-group spread')
        ax.set_yticks([0,40,104],['L0 attention','L13 feed-forward','L34 embedding'])
        ax.set_xticks([p for p,r in marks],[r for p,r in marks],rotation=35,ha='right')
    fig.colorbar(im,ax=axs,label='RMS distance from that group’s mean turn')
    fig.suptitle('Where the members of each overlay spread out\nPositions with fewer than two members are blank',fontsize=16)
    fig.savefig(out/'spread.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(14,8),layout='constrained',sharex=True)
    for g,color,name in [(1,'#26756c','successes'),(0,'#a97843','failures')]:
        axs[0].plot(counts[:,g],color=color,label=name)
        axs[1].plot(np.nanmean(metrics[g,:,:,1],axis=1),color=color,label=name)
        axs[2].plot(np.sqrt(np.nanmean(metrics[g,:,:,2],axis=1)),color=color,label=name)
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    axs[0].set_ylabel('Members aligned');axs[1].set_ylabel('Mean log gain, across merges');axs[2].set_ylabel('RMS gain spread, across merges')
    axs[2].set_xticks([p for p,r in marks],[r for p,r in marks],rotation=25,ha='right')
    fig.suptitle('Coverage and size changes — the overlays do not contain every member at every position')
    fig.savefig(out/'coverage-gain.png',dpi=160);plt.close(fig)

    # A shared, label-blind projection of observed passage-averaged vector paths.
    # Every member still contributes a full 105 x 1536 vector path before plotting.
    passages=['initial','descent','after_loss','destination'];paths={r:[] for r in passages}
    for i,m in enumerate(members):
        source=Path(m['source']).parent.parent/'exported-original-gpu0'/m['id']
        turns=np.load(source/'turns.npy',mmap_mode='r')
        for role in passages:
            pp=np.flatnonzero((roles==role)&(aligned[i]>=0))
            paths[role].append(np.mean(np.asarray(turns[aligned[i,pp]],dtype='f8'),axis=0) if len(pp) else np.full(turns.shape[1:],np.nan))
    projection=[];fig,axs=plt.subplots(len(passages),2,figsize=(14,19),layout='constrained')
    for row,role in enumerate(passages):
        x=np.array(paths[role]);flat=x.reshape(-1,x.shape[-1]);good=np.isfinite(flat).all(1);center=flat[good].mean(0);z=flat[good]-center
        eigen,vectors=np.linalg.eigh(z.T@z);basis=vectors[:,-2:][:,::-1];coordinates=(x-center)@basis
        explained=float(eigen[-2:].sum()/eigen.sum())
        projection.append(dict(role=role,explained_fraction=explained,coordinates=coordinates,
                               center=center,basis=basis))
        for column,outcome in enumerate(['natural_numerical_success','failure']):
            ax=axs[row,column];ix=[i for i,m in enumerate(members) if m['role']==outcome]
            for i in ix:ax.plot(coordinates[i,:,0],coordinates[i,:,1],color='#26756c' if column==0 else '#a97843',alpha=.22,linewidth=.7)
            mean=np.nanmean(coordinates[ix],axis=0);ax.plot(mean[:,0],mean[:,1],color='#183e3b',linewidth=2)
            for q,label in [(0,'L0 A'),(40,'L13 FF'),(104,'L34 end')]:ax.scatter(*mean[q],s=16,color='#222');ax.annotate(label,mean[q],fontsize=8)
            ax.set_title(f"{role.replace('_',' ')} · {'successes' if column==0 else 'failures'} · two axes retain {explained:.1%}")
            ax.set_xlabel('Shared projection axis 1');ax.set_ylabel('Shared projection axis 2')
            ax.set_xlim(np.nanmin(coordinates[:,:,0]),np.nanmax(coordinates[:,:,0]));ax.set_ylim(np.nanmin(coordinates[:,:,1]),np.nanmax(coordinates[:,:,1]))
    fig.suptitle('Each line is one response’s connected turn profile\nWithin-passage average over its aligned positions; one label-blind projection shared by both groups\nThis is a projection of recorded changes, not a literal hidden geometry or a single token’s trajectory',fontsize=14)
    fig.savefig(out/'vector-overlays.png',dpi=150);plt.close(fig)
    np.savez(out/'vector-projections.npz',**{f'{r["role"]}_{key}':r[key] for r in projection for key in ['coordinates','center','basis']})
    write(out/'projection.json',[dict(role=r['role'],explained_fraction=r['explained_fraction']) for r in projection])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
