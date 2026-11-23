"""Describe individual recorded branch progressions, without constructing an edit.

All eligible correspondence rows in both complete passages are inspected. A
branch's difference is followed through its real residual merge and layer exit;
native vector references remain available for every scalar summary. No output
tokens, layers, or high-scoring examples are selected by these measurements.
"""
import collections
import json
from pathlib import Path

import numpy as np

from pipeline.native_records.group_observation import publish
from pipeline.acquisition.prepare_a_native_excerpts import OUTPUT
from pipeline.acquisition.run_a_baseline import RUN


METRICS = ['incoming_difference_norm', 'branch_difference_norm',
           'merged_difference_norm', 'incoming_relative_separation',
           'merged_relative_separation', 'relative_separation_gain',
           'incoming_branch_cosine', 'incoming_merged_cosine',
           'merged_native_addition_error']


class View:
    def __init__(self, cid):
        self.cid = cid
        self.path = OUTPUT / cid / 'branch-view'
        self.index = json.loads((self.path / 'view.json').read_text())
        self.frames = {x['prediction']: x for x in self.index['executions']}
        self.entries = {p: {e['node']: e for e in f['entries']} for p, f in self.frames.items()}
        self.data = np.memmap(self.path / 'vectors.f32', dtype='<f4', mode='r')

    def vector(self, prediction, node):
        e = self.entries[prediction][node]
        return self.data[e['offset']:e['offset'] + e['elements']].astype(np.float64)

    def template(self, prediction):
        return self.index['templates'][self.frames[prediction]['template']]


def branches(template):
    schema = template['schema']
    names = {n['name']: i for i, n in enumerate(schema)}
    result = []
    for layer in range(35):
        attention = names[f'attn_out-{layer}']
        feedforward = names[f'pe_in-{layer}']
        exit_node = names[f'l_out-{layer}']
        pe = schema[exit_node]['ports'][0]['producer']
        for kind, merge, residual_port in [('attention', attention, 1),
                                           ('feedforward', feedforward, 1),
                                           ('per_layer_embedding', pe, 0)]:
            assert schema[merge]['operation'] == 'ADD'
            ports = schema[merge]['ports']
            incoming, addition = ports[residual_port]['producer'], ports[1-residual_port]['producer']
            assert incoming is not None and addition is not None
            chain = []
            def ancestors(node):
                entry = schema[node]
                chain.append(dict(node=node, **entry))
                if entry['operation'] in ('MUL_MAT', 'GET_ROWS', 'ADD'):
                    return
                for port in entry['ports']:
                    if port and port.get('producer') is not None:
                        ancestors(port['producer'])
            ancestors(addition)
            result.append(dict(layer=layer, kind=kind, incoming=incoming,
                               branch=addition, merged=merge, layer_exit=exit_node,
                               supplying_chain=chain))
    return result


def measure(left, right, lp, rp, connections):
    def matrix(view, prediction, key):
        return np.stack([view.vector(prediction, c[key]) for c in connections])
    la, ra = matrix(left,lp,'incoming'), matrix(right,rp,'incoming')
    lb, rb = matrix(left,lp,'branch'), matrix(right,rp,'branch')
    ly, ry = matrix(left,lp,'merged'), matrix(right,rp,'merged')
    incoming, branch, merged = la-ra, lb-rb, ly-ry
    norm = lambda v: np.linalg.norm(v,axis=1)
    ni, nb, nm = norm(incoming), norm(branch), norm(merged)
    ri = ni / np.maximum((norm(la)+norm(ra))/2,1e-30)
    rm = nm / np.maximum((norm(ly)+norm(ry))/2,1e-30)
    def divide(a,b):
        return np.divide(a,b,out=np.full_like(a,np.nan),where=b!=0)
    return np.stack([ni,nb,nm,ri,rm,divide(rm,ri),
                     divide(np.einsum('ij,ij->i',incoming,branch),ni*nb),
                     divide(np.einsum('ij,ij->i',incoming,merged),ni*nm),
                     norm(merged-incoming-branch)/np.maximum(nm,1e-30)],axis=1)


def main():
    out = OUTPUT / 'connected-progressions'
    out.mkdir(exist_ok=True)
    cohort_index = json.loads((RUN/'response-correspondence/index.json').read_text())
    views = {p.parent.parent.name: View(p.parent.parent.name) for p in OUTPUT.glob('*/branch-view/view.json')}
    completed, missing = [], []
    for number, pair in enumerate(cohort_index['pairs']):
        name = f'{number:03}-{pair["left"]}--{pair["right"]}'
        target = out/name
        if (target/'index.json').exists():
            completed.append(str(target/'index.json'))
            continue
        if pair['left'] not in views or pair['right'] not in views:
            missing.append(pair)
            continue
        left, right = views[pair['left']], views[pair['right']]
        mapping = json.loads((RUN/'response-correspondence'/pair['file']).read_text())
        observations, values, templates, excluded = [], [], {}, []
        for row in mapping['predictions']:
            lp, rp = row['left_prediction'], row['right_prediction']
            if lp not in left.frames or rp not in right.frames:
                excluded.append(dict(left_prediction=lp, right_prediction=rp, reason='Outside the two complete retrieved passages'))
                continue
            lt, rt = left.template(lp), right.template(rp)
            # Producer topology must agree, not merely the displayed layer name.
            if lt['schema'] != rt['schema']:
                excluded.append(dict(left_prediction=lp, right_prediction=rp, reason='Different recorded producer topology'))
                continue
            tid = left.frames[lp]['template']
            if tid not in templates:
                templates[tid] = branches(lt)
            connections = templates[tid]
            values.append(measure(left, right, lp, rp, connections))
            observations.append(dict(row, template=tid,
                                     left_execution=left.frames[lp]['execution'],
                                     right_execution=right.frames[rp]['execution']))
        target.mkdir(exist_ok=False)
        data = np.asarray(values,dtype='<f4')
        np.save(target/'metrics.npy',data)
        summaries = []
        if observations:
            for role in sorted({x['section'] for x in observations}):
                indices=[i for i,x in enumerate(observations) if x['section']==role]
                for j,c in enumerate(next(iter(templates.values()))):
                    part=data[indices,j]
                    def median(k):
                        v=part[:,k];v=v[np.isfinite(v)]
                        return float(np.median(v)) if len(v) else None
                    summaries.append(dict(section=role,layer=c['layer'],kind=c['kind'],decisions=len(indices),
                                          **{key:median(k) for k,key in enumerate(METRICS)}))
        report=dict(pair=pair,observations=observations,connections=templates,metrics=METRICS,
                    shape=list(data.shape),summaries=summaries,excluded=excluded,
                    unmatched_left=mapping['unmatched_left'],unmatched_right=mapping['unmatched_right'],
                    left_vectors=str(left.path/'view.json'),right_vectors=str(right.path/'view.json'),
                    limit='Descriptive connected differences. Context, input parameters and generated histories differ. No scalar curve or difference vector is itself an identified capability or edit target.')
        publish(target/'index.json',report)
        completed.append(str(target/'index.json'))
        print(json.dumps(dict(pair=name,decisions=len(observations))),flush=True)
    publish(out/'index.json',dict(pairs=completed,missing=missing,source_pairs=len(cohort_index['pairs']),
                                 complete=len(completed)==len(cohort_index['pairs']),metrics=METRICS,
                                 scope='Individual recorded contributions through every residual addition in both complete passages. Question-level summaries only; vectors remain separate by event.'))


if __name__ == '__main__':
    main()
