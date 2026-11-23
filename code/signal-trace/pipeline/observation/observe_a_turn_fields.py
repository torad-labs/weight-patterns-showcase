"""Observe full vector changes of direction at real residual merges.

These are descriptive fields, not latent semantic coordinates or a weight edit.
Each context anchor and neighboring decision remains separate. The original
recorded operands, outputs, operators and weight bindings remain linked.
"""
import json
from pathlib import Path

import numpy as np

from pipeline.native_records.group_observation import publish
from pipeline.observation.inspect_a_connected_progressions import View, branches, OUTPUT, RUN


def anchors(annotation):
    result = {}
    for role in ['after_loss', 'destination']:
        sections = [s for s in annotation['sections'] if s['role'] == role]
        if len(sections) != 1:
            continue
        section = sections[0]
        tokens = [t for t in annotation['tokens'] if t['section'] == section['index']]
        if not tokens:
            continue
        result[role+'_start'] = tokens[0]['prediction']
        # The first generated token after the heading's newline. Its input is
        # the completed heading, not a forced response or an assumed calculation.
        for i,t in enumerate(tokens[:-1]):
            if '\n' in t['piece']:
                result[role+'_heading_end'] = tokens[i+1]['prediction']
                break
        if role == 'destination':
            passage=''.join(t['piece'] for t in tokens)
            phrase='above the bottom of the climb'
            start=passage.find(phrase)
            if start >= 0:
                # This literal-context subset is reported separately; missing
                # formulations are not forced into it or dropped from A.
                end=start+len(phrase);position=0
                for i,t in enumerate(tokens):
                    position+=len(t['piece'])
                    if position >= end:
                        heading_end=passage.find('\n')
                        location='heading' if heading_end<0 or end<=heading_end else 'body'
                        result['climb_reference_'+location]=t['prediction']
                        break
    return result


def written_route(annotation, role):
    section=next(s for s in annotation['sections'] if s['role']=='destination')
    passage=''.join(t['piece'] for t in annotation['tokens'] if t['section']==section['index'])
    if role=='failure':return 'adds_descent_to_climb'
    if 'The object starts the climb' in passage:return 'available_energy_then_direct_height'
    if 'height gained during the climb' in passage:return 'explicit_climb_only_height'
    if 'The destination is' in passage:return 'direct_destination_height'
    return 'explicit_final_height'


def main():
    out = OUTPUT/'turn-fields'
    out.mkdir(exist_ok=True)
    cohort = json.loads(Path('research/experiments/signal-trace-repair/A-cohort.json').read_text())
    records=[]; data=[]; connections=None
    for case in cohort['cases']:
        cid=case['id']
        if not (OUTPUT/cid/'branch-view/view.json').exists():
            continue
        v=View(cid)
        ann=json.loads((RUN/'response-correspondence/cases'/(cid+'.json')).read_text())
        route=written_route(ann,case['role'])
        for anchor,prediction in anchors(ann).items():
            for offset in range(-4,5):
                p=prediction+offset
                if p not in v.frames:
                    continue
                cs=branches(v.template(p))
                if connections is None:
                    connections=cs
                assert [(c['incoming'],c['branch'],c['merged']) for c in cs] == [(c['incoming'],c['branch'],c['merged']) for c in connections]
                incoming=np.stack([v.vector(p,c['incoming']) for c in cs])
                merged=np.stack([v.vector(p,c['merged']) for c in cs])
                normal=lambda x:x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-30)
                # Unit-direction change lives in this actual merge's shared
                # residual coordinates. Do not compare raw coordinates between
                # different projections or interpret its norm as a capability.
                change=normal(merged)-normal(incoming)
                data.append(change.astype('<f4'))
                records.append(dict(case=cid,gpu=case['gpu'],role=case['role'],parameters=case['parameters'],
                                    anchor=anchor,offset=offset,prediction=p,execution=v.frames[p]['execution'],written_route=route,
                                    selected_piece=v.frames[p]['piece'],source=str(v.path/'view.json')))
        del v
    np.save(out/'changes.npy',np.asarray(data,dtype='<f4'))
    publish(out/'index.json',dict(version=2,records=records,connections=connections,shape=[len(data),105,1536],
                                 definition='At each real residual merge: output / ||output|| - incoming / ||incoming||. Original contributions and magnitudes remain in native branch views.',
                                 anchors='Text headings supply reproducible context positions, not hidden calculation onset. Every offset is separate; no passage-averaged vector.',
                                 limit='A repeated directional contrast can include numerical context, wording or general sequence preparation. This observation is not an edit recipe or a capability-specific finding.'))
    print(json.dumps(dict(cases=len({r['case'] for r in records}),frames=len(records),output=str(out))),flush=True)


if __name__=='__main__':main()
