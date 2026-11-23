"""Resolve recorded attention inputs into explicitly derived history messages.

Fused attention's internal probabilities were not recorded. These higher-
precision diagnostic reconstructions are labeled derived and compared with the
actual recorded outputs. No activation or model weight is modified.
"""
import hashlib
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

import numpy as np

from pipeline.native_records.group_observation import frames,publish
from pipeline.native_records.native_excerpts import ExcerptReader
from pipeline.native_records.native_trace import checkpoint_values
from pipeline.observation.observe_a_turn_fields import anchors
from pipeline.observation.observe_a_weight_transport import MODEL
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT
from pipeline.acquisition.run_a_baseline import RUN

PIECES={}


def initialize(pieces):
    global PIECES
    PIECES=pieces


def cache_positions(cid,context,selection):
    member=json.loads(Path(selection['member_record']).read_text())
    source=member['source']['compressed_events']
    expected={f['execution']:f['input_positions'] for f in context['frames']}
    counts={};versions={};digest=hashlib.sha256();writes=0
    process=subprocess.Popen(['zstd','-d','-c',source['compressed']],stdout=subprocess.PIPE)
    for line in process.stdout:
        digest.update(line)
        if b'"cache_write"' not in line:continue
        event=json.loads(line);w=event['cache_write'];storage=w['storage'];start=counts.get(storage,0)
        if w['slots']!=expected[event['graph_execution']] or w['slots']!=list(range(start,start+len(w['slots']))):
            raise ValueError('Cache positions are not the recorded append-only token positions')
        counts[storage]=start+len(w['slots']);versions[event['event']]=(storage,counts[storage]);writes+=1
    if process.wait()!=0 or digest.hexdigest()!=selection['source_events_sha256']:
        raise ValueError('Full cache metadata differs from the acquired recording')
    return versions,dict(case=cid,writes=writes,stores=counts,events_sha256=digest.hexdigest(),
                         observed='Every actual cache write matches that execution\'s recorded token positions, in append order, without slot reuse.')


def history_labels(case,annotation):
    pieces=[PIECES[t].replace('\u2581',' ') for t in case['tokens']]
    text=''.join(pieces);spans=[];cursor=0
    for piece in pieces:spans.append((cursor,cursor+len(piece)));cursor+=len(piece)
    labels=['prompt_other']*len(pieces)
    patterns={'prompt_speed':r'initially moves at [0-9.]+ m/s',
              'prompt_descent':r'descends [0-9.]+ m along a smooth track',
              'prompt_loss':r'dissipates [0-9.]+ J',
              'prompt_climb':r'destination [0-9.]+ m above the bottom of that climb'}
    matches={}
    for name,pattern in patterns.items():
        found=list(re.finditer(pattern,text))
        if len(found)!=1:raise ValueError(f'Prompt clause requires explicit annotation: {case["id"]}, {name}')
        m=found[0];indices=[i for i,(a,b) in enumerate(spans) if a<m.end() and b>m.start()]
        matches[name]=dict(text=m.group(),positions=indices)
        for i in indices:labels[i]=name
    roles={s['index']:s['role'] for s in annotation['sections']}
    labels.extend('generated_'+str(roles.get(t['section'],'unassigned')) for t in annotation['tokens'])
    return labels,matches


def collect(cid):
    root=OUTPUT/cid;out=root/'attention-messages'
    if (out/'index.json').exists():return cid
    out.mkdir(exist_ok=False)
    selection=json.loads((root/'selection.json').read_text())
    context=json.loads((RUN/'context-index'/(cid+'.json')).read_text())
    annotation=json.loads((RUN/'response-correspondence/cases'/(cid+'.json')).read_text())
    supplied=json.loads((RUN/'cohort-inputs'/(cid+'.json')).read_text());case=supplied[0] if isinstance(supplied,list) else supplied
    labels,prompt_clauses=history_labels(case,annotation)
    positions={k:v for k,v in anchors(annotation).items() if k.endswith('heading_end') or k.startswith('climb_reference_')}
    selected={int(context['prediction_executions'][str(p)]):k for k,p in positions.items()}
    versions,cache_proof=cache_positions(cid,context,selection)
    reader=ExcerptReader(root/'native');rows=[];cursor=0;seen=set();weight_cache={}
    try:
        with (out/'derived.f32').open('xb') as stream:
            def write(values):
                nonlocal cursor
                values=np.asarray(values,dtype='<f4');start=cursor;stream.write(values.tobytes());cursor+=values.size
                return dict(offset=start,shape=list(values.shape),elements=values.size)
            for execution,events in frames(root/'events.jsonl.zst'):
                if execution not in selected:continue
                anchor=selected[execution];prediction=positions[anchor]
                attention=[e for e in events if e['operation']=='FLASH_ATTN_EXT']
                projections={p['weight']['name']:e for e in events if e['operation']=='MUL_MAT' for p in e['inputs'] if p and p.get('weight')}
                if len(attention)!=35:raise ValueError('Expected every layer\'s attention event')
                for layer,event in enumerate(attention):
                    source=event['inputs'];snap=[p['snapshot'] for p in source[:4]]
                    params=np.array(event['output']['op_params_i32'],dtype='<i4').view('<f4')
                    if params[1]!=0 or params[2]!=0 or any(p is not None for p in source[4:]):raise ValueError('Attention reconstruction needs another explicit operator case')
                    q,k,v,mask=[reader.values(s).reshape(s['shape'],order='F') for s in snap]
                    if q.shape[1]!=1 or k.shape[2:]!=(1,1) or v.shape[2:]!=(1,1):raise ValueError('Unexpected attention axes')
                    q=q[:,0,:,0].T.astype('f8');k=k[:,:,0,0].T.astype('f8');v=v[:,:,0,0].T.astype('f8');mask=mask[:,0,0,0].astype('f8')
                    bases=[]
                    for port in [1,2]:
                        operand=source[port];root_layout=operand['cache_root_layout'];stride=root_layout['strides'][1]
                        if operand['cache_view_offset']%stride:raise ValueError('Unaligned cache view')
                        bases.append(operand['cache_view_offset']//stride)
                    if bases[0]!=bases[1]:raise ValueError('K/V rows do not refer to the same token positions')
                    valid=np.flatnonzero(np.isfinite(mask));slots=valid+bases[0]
                    if not len(valid) or any(int(slots.max())>=versions[source[p]['cache_state']][1] for p in [1,2]):raise ValueError('Attention includes an unwritten token')
                    logits=q@k[valid].T*params[0]+mask[valid][None,:]
                    probability=np.exp(logits-logits.max(1,keepdims=True));probability/=probability.sum(1,keepdims=True)
                    result=probability@v[valid]
                    actual=reader.values(event['output']).reshape(event['output']['shape'],order='F')[:,:,0,0].T.astype('f8')
                    error=float(np.linalg.norm(result-actual)/max(np.linalg.norm(actual),1e-30))
                    name=f'blk.{layer}.attn_output.weight';projection=projections[name];binding=next(p['weight'] for p in projection['inputs'] if p and p.get('weight'))
                    if name not in weight_cache:weight_cache[name]=checkpoint_values(MODEL,binding).reshape(binding['shape'][1],binding['shape'][0]).astype('f8')
                    weight=weight_cache[name];actual_projected=reader.values(projection['output']).astype('f8');energy=float(np.dot(actual_projected,actual_projected))
                    groups=sorted({labels[int(s)] for s in slots});messages=[];stats=[]
                    for group in groups:
                        choose=np.array([labels[int(s)]==group for s in slots]);message=probability[:,choose]@v[valid[choose]];projected=weight@message.ravel()
                        messages.append(projected);stats.append(dict(group=group,probability_by_head=probability[:,choose].sum(1).tolist(),
                                                                     projected_norm=float(np.linalg.norm(projected)),projected_signed_share=float(np.dot(projected,actual_projected)/max(energy,1e-30))))
                    projected_error=float(np.linalg.norm(np.sum(messages,axis=0)-actual_projected)/max(np.sqrt(energy),1e-30))
                    rows.append(dict(anchor=anchor,prediction=prediction,execution=execution,layer=layer,event=event['event'],
                                     query=source[0]['snapshot'],key_version=source[1]['cache_state'],value_version=source[2]['cache_state'],
                                     cache_positions=slots.tolist(),probabilities=write(probability),groups=stats,projected_messages=write(messages),
                                     projection_event=projection['event'],projection_weight=binding,
                                     relative_attention_reconstruction_error=error,relative_projection_reconstruction_error=projected_error))
                seen.add(execution)
                if seen==set(selected):break
    finally:reader.close()
    if seen!=set(selected):raise ValueError('Requested attention context missing')
    publish(out/'index.json',dict(case=cid,positions=positions,cache_positions=cache_proof,prompt_clauses=prompt_clauses,rows=rows,
                                 derived=True,probabilities_recorded=False,
                                 definition='softmax(scale * recorded_Q @ recorded_K.T + recorded_mask) @ recorded_V, followed by the actual BF16 attention-output matrix. Groups sum at the linear projection before its nonlinear post-normalization.',
                                 limit='A diagnostic reconstruction from recorded parents, not the fused kernel\'s recorded intermediate arithmetic. Every result carries its discrepancy from actual attention and projected outputs. Attention mass and value-message share are descriptive contributions, not causal importance.'))
    return cid


def main():
    sys.path.insert(0,str(Path('dependencies/llama.cpp-f089c9c/gguf-py')))
    from gguf import GGUFReader
    cases=sorted(p.parent.name for p in OUTPUT.glob('*/retrieved.json'))
    wanted=set()
    for cid in cases:
        d=json.loads((RUN/'cohort-inputs'/(cid+'.json')).read_text());c=d[0] if isinstance(d,list) else d;wanted.update(c['tokens'])
    reader=GGUFReader(str(MODEL),'r');field=reader.fields['tokenizer.ggml.tokens'];pieces={i:bytes(field.parts[field.data[i]]).decode('utf8') for i in wanted}
    with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(pieces,)) as pool:
        for f in as_completed([pool.submit(collect,cid) for cid in cases]):print(f.result(),flush=True)


if __name__=='__main__':main()
