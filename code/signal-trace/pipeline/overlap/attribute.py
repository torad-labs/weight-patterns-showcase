"""Attribute original group differences through recorded weighted signal inputs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import SHARED,read,write,finite
from pipeline.overlap.components import normalized_parts,project_components,project_along
from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROOT


def native_matrix(model,binding):
    if binding['native_type'] not in ('bf16','f32'):raise ValueError('Unrecognized native weight representation')
    dtype='<u2' if binding['native_type']=='bf16' else '<f4'
    raw=np.memmap(model,mode='r',dtype=dtype,offset=binding['offset_bytes'],shape=(binding['bytes']//np.dtype(dtype).itemsize,))
    values=(raw.astype('<u4')<<16).view('<f4') if dtype=='<u2' else np.array(raw)
    width,height=binding['shape'][:2]
    return values.reshape(height,width).astype('f8')


def head_layouts():
    path=SHARED/'acquisition/capture-pilot/block-changed/signal-events.jsonl';layouts={}
    with path.open() as source:
        for line in source:
            event=json.loads(line)
            if event['graph_execution']!=1:break
            name=event.get('name','')
            if name.startswith('kqv_out-') and event['operation']=='RESHAPE':
                layout=event['inputs'][0]['layout'];width,heads=layout['shape'][:2]
                if layout['strides'][:2]!=[4,4*width]:raise ValueError('Head layout is not the recorded contiguous native layout')
                layouts[int(name.split('-')[-1])]=dict(width=width,heads=heads,event=event['event'],source=str(path))
    return layouts


def main(output,model):
    output,model=Path(output),Path(model)
    dest=output/'components';check_native_run(dest,output/'declaration.json')
    if dest.exists():raise ValueError('Component accounting already exists')
    dest.mkdir()
    with model.open('rb') as source:model_sha=hashlib.file_digest(source,'sha256').hexdigest()
    expected=(model.parent/'SHA256SUMS').read_text().split()[0]
    if model_sha!=expected:raise ValueError('Original checkpoint does not match its stored source digest')
    base=SHARED/'observation/original-model-traces';operators=read(base/'native-operators.json')['projections']
    source=read(SHARED/'observation/response-correspondence/index.json')
    cases={c:m for c,m in source['cases'].items() if m['role'] in ('failure','natural_numerical_success')}
    layouts=head_layouts();group_maps={gpu:read(output/f'contrasts/gpu{gpu}/index.json') for gpu in (0,1)}
    original_d={gpu:np.load(output/f'contrasts/gpu{gpu}/difference.npy',mmap_mode='r') for gpu in group_maps}
    common={gpu:np.load(output/f'common-coordinates/gpu{gpu}/gap.npy',mmap_mode='r') for gpu in group_maps}
    all_records=[]
    for operator in operators:
        layer,kind=operator['layer'],operator['kind'];merge=operator['merges'][0]
        w=native_matrix(model,operator['weight']);gain=native_matrix(model,operator['gain']).ravel();basis=w*gain[:,None]
        signal=f'kqv_out-{layer}' if kind=='attention' else f'ffn_geglu-{layer}'
        incoming=('inp_scaled' if layer==0 else f'l_out-{layer-1}') if kind=='attention' else f'attn_out-{layer}'
        coefficients={};contexts={};errors=[]
        for cid in cases:
            directory=base/'matrix-inputs'/cid
            x=np.load(directory/(signal+'.npy'),mmap_mode='r');r=np.load(directory/(incoming+'.npy'),mmap_mode='r')
            norms=np.load(base/'full-response-turns'/cid/'norms.npy',mmap_mode='r')[:,merge]
            alpha,context=normalized_parts(x,w,gain,r,norms,operator['epsilon'])
            native=np.load(base/'full-response-turns'/cid/'turns.npy',mmap_mode='r')[:,merge]
            reproduced=alpha@basis.T+context
            difference=np.linalg.norm(reproduced-native,axis=1);length=np.linalg.norm(native,axis=1)
            error=np.divide(difference,length,out=np.full_like(difference,np.nan),where=length>0)
            errors.append(dict(case=cid,max_absolute=float(difference.max()),max_relative=float(np.nanmax(error)),mean_relative=float(np.nanmean(error))))
            coefficients[cid]=alpha;contexts[cid]=context
        folder=dest/f'layer-{layer:02d}-{kind}';folder.mkdir()
        report=dict(layer=layer,kind=kind,merge=merge,weight=operator['weight'],gain=operator['gain'],epsilon=operator['epsilon'],
            native_reconstruction=errors,head_layout=layouts.get(layer) if kind=='attention' else None,cards=[])
        for gpu,meta in group_maps.items():
            delta=[];residual=[]
            for row in meta['rows']:
                fm,sm=row['failure_members'],row['success_members']
                delta.append(np.mean([coefficients[c][p] for c,p in sm],axis=0)-np.mean([coefficients[c][p] for c,p in fm],axis=0))
                residual.append(np.mean([contexts[c][p] for c,p in sm],axis=0)-np.mean([contexts[c][p] for c,p in fm],axis=0))
            delta,residual=np.array(delta),np.array(residual);d=np.asarray(original_d[gpu][:,merge])
            account=project_components(delta,basis,residual,d)
            axis_account=project_along(delta,basis,residual,d,np.asarray(common[gpu][:,merge]))
            values=account['unit_share']; common_values=axis_account['unit_share']
            if kind=='attention':
                layout=layouts[layer]
                if layout['heads']*layout['width']!=values.shape[1]:raise ValueError('Native head count does not match matrix width')
                values=values.reshape(len(values),layout['heads'],layout['width']).sum(axis=-1)
                common_values=common_values.reshape(len(common_values),layout['heads'],layout['width']).sum(axis=-1)
            np.savez(folder/f'gpu{gpu}.npz',delta_alpha=delta,context_difference=residual,
                raw_component_share=values,raw_context_share=account['context_share'],
                raw_reconstruction_relative=account['reconstruction_relative'],
                common_axis_component_share=common_values,common_axis_context_share=axis_account['context_share'],
                common_axis_raw_projection=axis_account['projected_raw'])
            report['cards'].append(dict(gpu=gpu,positions=len(delta),component_count=values.shape[1],
                max_group_reconstruction_relative=float(np.nanmax(account['reconstruction_relative'])),
                max_common_axis_accounting_error=float(np.nanmax(abs(np.sum(common_values,axis=1)+axis_account['context_share']-1))) if np.isfinite(common_values).any() else None))
        write(folder/'index.json',finite(report));all_records.append(dict(path=str(folder),layer=layer,kind=kind,merge=merge))
        print(json.dumps(dict(layer=layer,kind=kind,max_native_relative=max(e['max_relative'] for e in errors),cards=report['cards'])),flush=True)
    write(dest/'index.json',dict(operators=all_records,original_model=str(model),original_model_sha256=model_sha,
        source=str(base/'native-operators.json'),all_layers_in_trace=sorted({r['layer'] for r in read(base/'full-response-turns/block-changed/index.json')['connections']}),
        input_layers=sorted({r['layer'] for r in operators}),
        scope='Additive accounting through recorded native weighted inputs plus residual and normalization terms. This is an observational contribution, not a causal intervention or a claim that a unit stores the procedure.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('model',type=Path);a=p.parse_args();main(a.output,a.model)
