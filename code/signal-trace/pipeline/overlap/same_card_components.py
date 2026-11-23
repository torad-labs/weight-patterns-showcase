"""Native component accounting of same-card group overlaps."""
import argparse
import json
from pathlib import Path
import re
import struct
import numpy as np

from pipeline.paths import ROOT
from pipeline.native_records.run_declaration import check_native_run, digest
from pipeline.acquisition.export_stream_traces import StreamRows
from pipeline.overlap.analyze import read, write, finite
from pipeline.overlap.attribute import native_matrix
from pipeline.overlap.components import normalized_parts
from pipeline.overlap.same_card import error_stage


def member_parts(a,c,fa,sa,fc,sc,d,basis,side):
    length=np.linalg.norm(d,axis=-1)
    u=np.divide(d,length[:,None],out=np.zeros_like(d),where=length[:,None]>0)
    parts=side*(a-(fa+sa)/2)*(u@basis)
    context=side*np.sum((c-(fc+sc)/2)*u,axis=-1)
    parts[length==0]=np.nan;context[length==0]=np.nan
    return parts,context


def operators(path):
    """Follow the recorded producer edges, with no prior layer selection."""
    events={}
    with Path(path).open() as source:
        for line in source:
            event=json.loads(line)
            if event['graph_execution']!=1:
                if events:break
                continue
            events[event['event']]=event
    found=[]
    for event in events.values():
        m=re.fullmatch(r'(attn|ffn)_post_norm-(\d+)',event['name'])
        if not m:continue
        kind,layer=m[1],int(m[2])
        gains=[p['weight'] for p in event['inputs'] if p and p.get('weight')]
        ports=[p for p in event['inputs'] if p and p.get('producer_event') is not None]
        if event['operation']!='MUL' or len(gains)!=1 or len(ports)!=1:
            raise ValueError('Unexpected normalized branch')
        norm=events[ports[0]['producer_event']]
        if norm['operation']!='RMS_NORM':raise ValueError('Missing native branch RMS norm')
        proj=events[norm['inputs'][0]['producer_event']]
        weights=[p['weight'] for p in proj['inputs'] if p and p.get('weight')]
        inputs=[p for p in proj['inputs'] if p and p.get('producer_event') is not None]
        if proj['operation']!='MUL_MAT' or len(weights)!=1 or len(inputs)!=1:
            raise ValueError('Missing native output projection')
        signal=f'kqv_out-{layer}' if kind=='attn' else f'ffn_geglu-{layer}'
        if inputs[0]['layout']['name']!=signal:raise ValueError('Unbound projection input')
        row=dict(layer=layer,kind='attention' if kind=='attn' else 'feedforward',
            merge=layer*3+(0 if kind=='attn' else 1),weight=weights[0],gain=gains[0],
            epsilon=struct.unpack('<f',struct.pack('<i',norm['output']['op_params_i32'][0]))[0],
            signal=signal,incoming=('inp_scaled' if layer==0 else f'l_out-{layer-1}') if kind=='attn' else f'attn_out-{layer}',
            branch=event['name'],projection_event=proj['event'],norm_event=norm['event'])
        if kind=='attn':
            shaped=events[inputs[0]['producer_event']]
            layout=shaped['inputs'][0]['layout']
            width,heads=layout['shape'][:2]
            if layout['strides'][:2]!=[4,4*width] or width*heads!=weights[0]['shape'][0]:
                raise ValueError('Native attention head layout changed')
            row['head_width']=width;row['heads']=heads
        found.append(row)
    return sorted(found,key=lambda r:r['merge'])


def group_mean(a,indices,counts):
    return np.divide(np.nansum(a[indices],axis=0),counts[:,None],
                     out=np.full(a.shape[1:],np.nan),where=counts[:,None]>0)


def main(output,pre_error=False):
    output=Path(output).resolve();dest=output/('components-pre-error' if pre_error else 'components')
    check_native_run(dest,output/'declaration.json')
    if dest.exists():raise ValueError('Preserve existing component accounting')
    calc=output/('pre-error' if pre_error else 'calculation');meta=read(calc/'summary.json');members=read(calc/'members.json')
    aligned=np.load(calc/'aligned.npy');data=np.load(calc/'evidence.npz')
    counts=data['counts'];roles=meta['roles'];npos=len(roles)
    means=[np.load(calc/f'{g}-mean.npy',mmap_mode='r') for g in ('failure','success')]
    fi=np.array([i for i,m in enumerate(members) if m['role']=='failure'])
    si=np.array([i for i,m in enumerate(members) if m['role']!='failure'])
    graph=ROOT/'studies/04-round-2-overlap/inputs/acquisition/capture-pilot/block-changed/signal-events.jsonl'
    ops=operators(graph)
    expected=[i for i,c in enumerate(meta['connections']) if c['kind'] in ('attention','feedforward')]
    if [o['merge'] for o in ops]!=expected:raise ValueError('Component coverage must enumerate every native attention/FF merge')
    model=ROOT/'models/gemma4/gemma4-e2b/gguf-bf16-bartowski/google_gemma-4-E2B-it-bf16.gguf'
    model_sha=digest(model)
    if model_sha!=read(output/'declaration.json')['model_sha256']:raise ValueError('Original model identity changed')
    dest.mkdir();write(dest/'operators.json',dict(source=str(graph),sha256=digest(graph),operators=ops))
    streams=[];norms=[];turns=[];input_positions=[]
    for m in members:
        source=Path(m['source']);r=read(source/'result.json')
        positions=[x['input_position'] for x in r['steps']]
        streams.append(StreamRows(source,positions));input_positions.append(positions)
        directory=source.parent.parent/'exported-original-gpu0'/m['id']
        norms.append(np.load(directory/'norms.npy',mmap_mode='r'))
        turns.append(np.load(directory/'turns.npy',mmap_mode='r'))
        print('indexed native inputs '+m['id'],flush=True)
    records=[]
    try:
        for op in ops:
            merge=op['merge'];w=native_matrix(model,op['weight']);gain=native_matrix(model,op['gain']).ravel()
            basis=w*gain[:,None];ncomp=w.shape[1]
            aa=np.full((len(members),npos,ncomp),np.nan)
            cc=np.full((len(members),npos,w.shape[0]),np.nan)
            tt=np.full_like(cc,np.nan);errors=[]
            for i,m in enumerate(members):
                pp=np.flatnonzero(aligned[i]>=0);q=aligned[i,pp]
                native_positions=[input_positions[i][v] for v in q]
                if not len(pp):continue
                x=np.stack([streams[i].values(op['signal'],v) for v in native_positions])
                r=np.stack([streams[i].values(op['incoming'],v) for v in native_positions])
                alpha,context=normalized_parts(x,w,gain,r,norms[i][q,merge],op['epsilon'])
                native=np.asarray(turns[i][q,merge],dtype='f8');reconstructed=alpha@basis.T+context
                error=np.linalg.norm(reconstructed-native,axis=-1)
                relative=error/np.maximum(np.linalg.norm(native,axis=-1),np.finfo(float).tiny)
                errors.append(dict(id=m['id'],max_absolute=float(error.max()),max_relative=float(relative.max())))
                aa[i,pp]=alpha;cc[i,pp]=context;tt[i,pp]=native
            af=group_mean(aa,fi,counts[:,0]);ass=group_mean(aa,si,counts[:,1])
            cf=group_mean(cc,fi,counts[:,0]);cs=group_mean(cc,si,counts[:,1])
            d=np.asarray(means[1][:,merge]-means[0][:,merge]);dn=np.linalg.norm(d,axis=-1)
            axis=np.divide(d,dn[:,None],out=np.full_like(d,np.nan),where=dn[:,None]>0)
            group_parts=(ass-af)*(axis@basis);group_context=np.sum((cs-cf)*axis,axis=-1)
            reconstructed=(ass-af)@basis.T+(cs-cf)
            group_error=np.linalg.norm(reconstructed-d,axis=-1)/np.maximum(dn,np.finfo(float).tiny)
            folder=dest/f"layer-{op['layer']:02d}-{op['kind']}";folder.mkdir()
            if op['kind']=='attention':
                group_parts=group_parts.reshape(npos,op['heads'],op['head_width']).sum(-1)
            np.savez(folder/'group.npz',components=group_parts,context=group_context,difference_norm=dn,
                reconstruction_relative=group_error)
            case_records=[]
            for i,m in enumerate(members):
                pp=np.flatnonzero((aligned[i]>=0)&np.all(counts>=2,axis=1)&np.isfinite(data['margins'][i,:,merge]))
                if not len(pp):continue
                isfailure=m['role']=='failure';nf,ns=counts[pp,0],counts[pp,1]
                fa=(af[pp]*nf[:,None]-aa[i,pp])/(nf-1)[:,None] if isfailure else af[pp]
                sa=ass[pp] if isfailure else (ass[pp]*ns[:,None]-aa[i,pp])/(ns-1)[:,None]
                fc=(cf[pp]*nf[:,None]-cc[i,pp])/(nf-1)[:,None] if isfailure else cf[pp]
                sc=cs[pp] if isfailure else (cs[pp]*ns[:,None]-cc[i,pp])/(ns-1)[:,None]
                ft=(means[0][pp,merge]*nf[:,None]-tt[i,pp])/(nf-1)[:,None] if isfailure else means[0][pp,merge]
                st=means[1][pp,merge] if isfailure else (means[1][pp,merge]*ns[:,None]-tt[i,pp])/(ns-1)[:,None]
                components,context=member_parts(aa[i,pp],cc[i,pp],fa,sa,fc,sc,st-ft,basis,-1 if isfailure else 1)
                reconstruction=np.abs(components.sum(-1)+context-data['margins'][i,pp,merge])
                if op['kind']=='attention':
                    components=components.reshape(len(pp),op['heads'],op['head_width']).sum(-1)
                keys=[(roles[p],error_stage(m['label'],int(aligned[i,p]))) for p in pp]
                rows=[];profiles=[];positive=[]
                for key in dict.fromkeys(keys):
                    ix=np.array([j for j,k in enumerate(keys) if k==key])
                    profiles.append(np.mean(components[ix],axis=0));positive.append(np.sum(components[ix]>0,axis=0))
                    rows.append(dict(role=key[0],stage=key[1],positions=len(ix),reference_positions=pp[ix],
                        context=float(np.mean(context[ix])),margin=float(np.mean(components[ix].sum(-1)+context[ix]))))
                np.savez(folder/(m['id']+'.npz'),profiles=np.array(profiles),positive=np.array(positive),
                         contexts=np.array([r['context'] for r in rows]))
                case_records.append(dict(id=m['id'],contexts=rows,max_margin_reconstruction_absolute=float(np.nanmax(reconstruction))))
            report=dict(operator=op,native_reconstruction=errors,max_group_reconstruction_relative=float(np.nanmax(group_error)),members=case_records)
            write(folder/'index.json',finite(report));records.append(dict(layer=op['layer'],kind=op['kind'],merge=merge,
                path=str(folder),max_native_reconstruction_relative=max(r['max_relative'] for r in errors),
                max_group_reconstruction_relative=report['max_group_reconstruction_relative'],
                max_margin_reconstruction_absolute=max(r['max_margin_reconstruction_absolute'] for r in case_records)))
            print(json.dumps(records[-1]),flush=True)
    finally:
        for stream in streams:stream.close()
    write(dest/'index.json',dict(operators=records,model_sha256=model_sha,
        scope='All native attention-output and FF-down components at aligned positions. Per-member contributions add to a leave-one-out signed margin. This is observational accounting, not a causal intervention. Per-layer-embedding turns are in the full overlap, not decomposed into heads or FF units.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--pre-error',action='store_true');a=parser.parse_args()
    main(a.output,a.pre_error)
