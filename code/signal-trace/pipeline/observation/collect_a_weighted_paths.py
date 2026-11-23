"""Retain every layer's actual weighted projection at explicit inspection contexts.

These views connect input signals, native stored weights and observed outputs.
They do not assign a meaning to individual weight values or alter inference.
Complete passages remain available in the verified native excerpts.
"""
import json
from concurrent.futures import ProcessPoolExecutor,as_completed

import numpy as np

from pipeline.native_records.group_observation import frames,publish
from pipeline.native_records.native_excerpts import ExcerptReader
from pipeline.observation.observe_a_turn_fields import anchors
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT
from pipeline.acquisition.run_a_baseline import RUN


def collect(cid):
    root=OUTPUT/cid;out=root/'weighted-paths'
    if (out/'index.json').exists():return cid
    out.mkdir(exist_ok=False)
    ann=json.loads((RUN/'response-correspondence/cases'/(cid+'.json')).read_text())
    positions={k:v for k,v in anchors(ann).items() if k.endswith('heading_end') or k.startswith('climb_reference_')}
    context=json.loads((RUN/'context-index'/(cid+'.json')).read_text())
    selected={int(context['prediction_executions'][str(p)]):k for k,p in positions.items()}
    reader=ExcerptReader(root/'native');records=[];cursor=0;observed=[]
    try:
        with (out/'vectors.f32').open('xb') as stream:
            for execution,events in frames(root/'events.jsonl.zst'):
                if execution not in selected:continue
                anchor=selected[execution];prediction=positions[anchor]
                for local,event in enumerate(events):
                    if event['operation']!='MUL_MAT':continue
                    weights=[p['weight'] for p in event['inputs'] if p and p.get('weight')]
                    inputs=[p for p in event['inputs'] if p and p.get('snapshot')]
                    if len(weights)!=1 or len(inputs)!=1 or not weights[0]['name'].startswith('blk.'):continue
                    weight=weights[0];source=inputs[0]
                    if int(np.prod(source['snapshot']['shape'][1:]))!=1 or int(np.prod(event['output']['shape'][1:]))!=1:
                        raise ValueError('Expected a recorded vector projection')
                    x=reader.values(source['snapshot']);y=reader.values(event['output'])
                    xo=cursor;stream.write(x.astype('<f4').tobytes());cursor+=x.size
                    yo=cursor;stream.write(y.astype('<f4').tobytes());cursor+=y.size
                    records.append(dict(anchor=anchor,prediction=prediction,execution=execution,node=local,event=event['event'],
                                        name=event['name'],operation=event['operation'],weight=weight,
                                        input_offset=xo,input_elements=x.size,output_offset=yo,output_elements=y.size,
                                        source=source,output=event['output']))
                observed.append(execution)
                if len(observed)==len(selected):break
    finally:reader.close()
    if set(observed)!=set(selected):raise ValueError('A requested recorded context is missing')
    publish(out/'index.json',dict(case=cid,positions=positions,records=records,elements=cursor,
                                 native_excerpt=str(root/'native/excerpt.json'),
                                 scope='All layer-bound weighted vector projections at the named context positions, using original native snapshots. This is an inspection subset of the complete captured path, not a complete-response replacement.'))
    return cid


def main():
    cases=sorted(p.parent.name for p in OUTPUT.glob('*/retrieved.json'))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(collect,c) for c in cases]):print(f.result(),flush=True)


if __name__=='__main__':main()
