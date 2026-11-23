"""Compare separate observed groups through explicit execution correspondence.

Full group-vector contrasts remain attached to every recorded operation and its
actual branches. Descriptive group differences are not automatically capability
contributions or a weight-edit prescription. Original individual vectors and
unmatched passages remain in the group source records.
"""


import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from pipeline.correspondence.bind_a_comparisons import bind
from pipeline.native_records.group_observation import GroupObservation, publish
from pipeline.observation.observe_a_groups import storage_reservation
from pipeline.acquisition.run_a_baseline import ROOT, RUN
from pipeline.correspondence.solution_correspondence import correspond


def compare_groups(left, right, mapping, output, storage=None):
    if not left.state['complete'] or not right.state['complete']:
        raise ValueError('Both separate group observations must be complete first')
    if left.layout['identity']['gpu'] != right.layout['identity']['gpu']:
        raise ValueError('Cross-card group differences are not a failure/success contrast')
    if not mapping.get('relation') or not mapping.get('frames'):
        raise ValueError('Explicit group-reference execution correspondence required')
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    required = 0
    for pair in mapping['frames']:
        frame = left.layout['frames'][str(pair['left_execution'])]
        template = left.layout['templates'][frame['template']]
        # Upper bound: every vector, addition change, and its JSON metadata.
        required += 4*(template['elements']+template['change_elements']) + 2048*len(template['outputs'])
    reservation = storage(output, required) if storage else nullcontext()
    compared = unmatched = source_only = offset = 0
    with reservation, (output/'contrasts.f32').open('wb') as values, (output/'locations.jsonl').open('w') as rows:
        def vector(value):
            nonlocal offset
            encoded = value.astype('<f4').tobytes()
            start = offset
            values.write(encoded)
            offset += value.size
            return dict(offset=start, elements=value.size, norm=float(np.linalg.norm(value)))
        for pair in mapping['frames']:
            if not pair.get('reason'):
                raise ValueError('A comparison frame lacks its correspondence reason')
            a,b=pair['left_execution'],pair['right_execution']
            if left.schema(a)!=right.schema(b):
                raise ValueError(f'Actual operation/source-port structure differs for group-reference executions {a}, {b}')
            lf=left.layout['frames'][str(a)];rf=right.layout['frames'][str(b)]
            lo=left.layout['templates'][lf['template']]['outputs']
            ro=right.layout['templates'][rf['template']]['outputs']
            for local,(la,ra) in enumerate(zip(lo,ro)):
                row=dict(connection=dict(left_execution=a,right_execution=b,local_operation=local),
                         name=la['name'],operation=la['operation'],
                         left_count=left.count(a,local),right_count=right.count(b,local))
                if not la['elements'] or not ra['elements']:
                    row.update(status='source_only',left_reason=la['reason'],right_reason=ra['reason'])
                    source_only+=1
                elif la['shape']!=ra['shape']:
                    row.update(status='unmatched',reason='Different native coordinate shapes')
                    unmatched+=1
                elif not row['left_count'] or not row['right_count']:
                    row.update(status='unmatched',reason='No corresponding finite observations in one group')
                    unmatched+=1
                else:
                    lm=left.mean(a,local);rm=right.mean(b,local)
                    delta=rm-lm
                    lv=left.dispersion(a,local);rv=right.dispersion(b,local)
                    spread=np.sqrt((lv+rv)/2)
                    row.update(status='compared',shape=la['shape'],**vector(delta),
                               left_mean_norm=float(np.linalg.norm(lm)),right_mean_norm=float(np.linalg.norm(rm)),
                               left_mean_squared_dispersion=lv,right_mean_squared_dispersion=rv,
                               difference_over_pooled_spread=float(np.linalg.norm(delta)/spread) if spread else None)
                    lc=left.addition_change(a,local);rc=right.addition_change(b,local)
                    if lc is not None and rc is not None:
                        row['addition_change']=dict(**vector(rc-lc),
                            left_mean_change_norm=float(np.linalg.norm(lc)),right_mean_change_norm=float(np.linalg.norm(rc)))
                    compared+=1
                rows.write(json.dumps(row,separators=(',',':'))+'\n')
    report=dict(left_group=str(left.path),right_group=str(right.path),correspondence=mapping,
        left_members=left.state['members'],right_members=right.state['members'],
        compared_locations=compared,unmatched_locations=unmatched,source_only_locations=source_only,
        sign='natural success group mean minus target-failure group mean, in each actual operation coordinate space',
        source_graph='Use each group layout template schema at connection.execution/local_operation for actual incoming producer ports, weights and cache references. Member source records retain every original vector and unmatched passage.',
        scope='Descriptive connected group contrast. No capability identity, exclusive signature, causal repair, or edit direction is established by a large difference between centroids.',
        interpretation_limits=['Context and previous generated tokens differ across questions.',
            'Reference correspondence is an index, not a claim of identical histories or numerical quantities.',
            'Coverage varies by location; counts and individual member source records are retained.',
            'Full native member vectors remain authoritative for checking any small difference in the derived float32 sums.',
            'Cache vectors without explicit group history-row correspondence remain original source observations.'],
        artifacts={})
    for name in ['contrasts.f32','locations.jsonl']:
        with (output/name).open('rb') as f:
            report['artifacts'][name]=dict(bytes=(output/name).stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
    publish(output/'comparison.json',report)
    return report


def reference_mapping(left,right):
    source=RUN/'response-correspondence'
    index=json.loads((source/'index.json').read_text())
    a=left.layout['identity']['reference'];b=right.layout['identity']['reference']
    aa=json.loads((source/'cases'/(a+'.json')).read_text())
    bb=json.loads((source/'cases'/(b+'.json')).read_text())
    pair=correspond(aa,bb)
    pair.update(left=a,right=b,relation='same-card separate-group reference correspondence',
                left_gpu=left.layout['identity']['gpu'],right_gpu=right.layout['identity']['gpu'],same_hardware=True)
    prompts=[json.loads((ROOT/index['cases'][cid]['input']).read_text())[0] for cid in [a,b]]
    contexts=[json.loads((RUN/'context-index'/(cid+'.json')).read_text()) for cid in [a,b]]
    return bind(pair,*contexts,*prompts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--producer-session')
    args=parser.parse_args()
    root=RUN/'group-observation'
    names=['gpu0-failure','gpu0-success','gpu1-failure','gpu1-success']
    while not all((root/n/'observation.json').exists() for n in names):
        if not args.wait:
            raise ValueError('Separate group observations are not complete')
        if args.producer_session and subprocess.run(['tmux','has-session','-t',args.producer_session],
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
            raise RuntimeError('The group observer ended before all four observations were complete')
        time.sleep(10)
    for gpu in [0,1]:
        out=root/f'gpu{gpu}-failure-success'
        if (out/'comparison.json').exists():
            continue
        left=GroupObservation(root/f'gpu{gpu}-failure');right=GroupObservation(root/f'gpu{gpu}-success')
        report=compare_groups(left,right,reference_mapping(left,right),out,storage=storage_reservation)
        print(json.dumps(dict(gpu=gpu,compared_locations=report['compared_locations'],
                             unmatched_locations=report['unmatched_locations'],new_weight_edits=0)),flush=True)


if __name__=='__main__':
    main()
