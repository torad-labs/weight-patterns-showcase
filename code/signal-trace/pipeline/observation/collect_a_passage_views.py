"""Keep individual connected branch vectors for inspecting A's recurring progression.

Every residual merge, its actual operands, and the supplying weighted branch are
retained across all layers. These are views of native records, not edited states,
capability labels, or fit targets. All other operations remain in each excerpt.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline.native_records.compare_occurrences import structure
from pipeline.native_records.group_observation import frames, publish
from pipeline.native_records.native_excerpts import ExcerptReader
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT
from pipeline.acquisition.run_a_baseline import RUN


def collect(cid):
    root=OUTPUT/cid
    out=root/'branch-view'
    if (out/'view.json').exists():return json.loads((out/'view.json').read_text())
    out.mkdir(exist_ok=False)
    selected=json.loads((root/'selection.json').read_text())
    context=json.loads((RUN/'context-index'/(cid+'.json')).read_text())
    decisions={int(e):int(p) for p,e in context['prediction_executions'].items()}
    annotation=json.loads((RUN/'response-correspondence/cases'/(cid+'.json')).read_text())
    tokens={t['prediction']:t for t in annotation['tokens']}
    roles={s['index']:s['role'] for s in annotation['sections']}
    reader=ExcerptReader(root/'native')
    file=out/'vectors.f32'
    index=[]; templates={};offset=0; count=0;max_add_error=0.
    try:
        with file.open('xb') as stream:
            for execution,events in frames(root/'events.jsonl.zst',selected['selected_events_sha256']):
                schema=structure(events)
                width=next(e['output']['shape'][0] for e in events if e['name']=='inp_scaled')
                residual_shape=[width,1,1,1]
                additions=[i for i,e in enumerate(events) if e['operation']=='ADD' and e['output']['shape']==residual_shape]
                nodes=set(additions)
                nodes.update(i for i,e in enumerate(events) if e['name'].startswith('l_out-'))
                # Follow each merge operand backward through the branch's norm
                # and scaling to its actual projection; stop at another merge.
                def branch(i):
                    if i in nodes:return
                    nodes.add(i)
                    if schema[i]['operation'] in ('ADD','MUL_MAT','GET_ROWS'):return
                    for port in schema[i]['ports']:
                        if port and port.get('producer') is not None:branch(port['producer'])
                for i in additions:
                    for port in schema[i]['ports']:
                        if port and port.get('producer') is not None:branch(port['producer'])
                nodes=sorted(nodes)
                template=dict(schema=schema,selected_nodes=nodes,merges=additions,width=width)
                digest=hashlib.sha256(json.dumps(template,separators=(',',':')).encode()).hexdigest()
                templates.setdefault(digest,template)
                values={};entries=[];start=offset
                for i in nodes:
                    v=reader.values(events[i]['output']).astype('<f4')
                    values[i]=v
                    stream.write(v.tobytes())
                    entries.append(dict(node=i,offset=offset,elements=v.size,shape=events[i]['output']['shape']))
                    offset+=v.size
                for i in additions:
                    operands=[reader.values(p['snapshot']).astype(np.float64) for p in events[i]['inputs'] if p and p.get('snapshot')][:2]
                    if len(operands)==2 and all(v.size==width for v in operands):
                        error=float(np.linalg.norm(values[i].astype(np.float64)-operands[0]-operands[1]))
                        scale=max(float(np.linalg.norm(values[i])),1e-30)
                        max_add_error=max(max_add_error,error/scale)
                prediction=decisions[execution];t=tokens[prediction]
                index.append(dict(execution=execution,prediction=prediction,role=roles[t['section']],selected=t['selected'],piece=t['piece'],
                                  template=digest,offset=start,entries=entries))
                count+=1
    finally:
        reader.close()
    report=dict(case=cid,group=selected['group'],source_selection=str(root/'selection.json'),native_excerpt=str(root/'native/excerpt.json'),
        executions=index,templates=templates,elements=offset,frames=count,max_native_addition_relative_rounding_error=max_add_error,
        representation='Exact float32 representations of the selected native branch outputs, retained separately for each execution and case. Original full preimages, all other operations and cache connections remain in the excerpt.',
        scope='All residual merges and their supplying projection/norm branches across every layer; no group averaging, edit or capability verdict.')
    publish(out/'view.json',report)
    return dict(case=cid,frames=count,elements=offset,max_native_addition_relative_rounding_error=max_add_error)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',action='append',dest='cases');p.add_argument('--workers',type=int,default=4,choices=range(1,5));args=p.parse_args()
    cases=args.cases or sorted(p.parent.name for p in OUTPUT.glob('*/retrieved.json'))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(collect,cid) for cid in cases]):
            r=f.result();print(json.dumps({k:r[k] for k in ['case','frames','elements','max_native_addition_relative_rounding_error']}),flush=True)


if __name__=='__main__':main()
