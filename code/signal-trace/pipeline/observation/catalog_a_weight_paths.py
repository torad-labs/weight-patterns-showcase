"""Bind the full-response map to recorded weighted projections without selecting edits.

The user needs the actual transformations that can affect each observed merge,
including shared cache writers. Preserve all recorded branches, native tensor
identities, source positions, and unbound uses; do not infer semantic responsibility
or full-passage input support from the five-position structural source.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from pipeline.acquisition.run_a_baseline import RUN
from pipeline.native_records.group_observation import publish


def catalog_documents(documents):
    projections, other, templates, cases = {}, {}, {}, {}
    merges = None
    observations = set()
    occurrences = 0
    for document in documents:
        cid, gpu = document['summary']['case'], document['summary']['gpu']
        if cid in cases:
            raise ValueError('Each structural source case must occur once')
        cases[cid] = dict(gpu=gpu, predictions=sorted({r['prediction'] for r in document['records']}))
        for key, template in document['templates'].items():
            shape = [dict(layer=c['layer'], kind=c['kind']) for c in template['connections']]
            if merges is None:
                merges = shape
            if shape != merges:
                raise ValueError('Ordered merge identities differ between source graphs')
            if key not in templates:
                templates[key] = dict(cases=[], nodes=len(template['schema']))
                for node, operation in enumerate(template['schema']):
                    if operation['operation'] == 'MUL_MAT':
                        continue
                    for port in operation['ports']:
                        if port and port.get('weight'):
                            identity = (port['weight'], operation['operation'], node, key)
                            other[identity] = dict(weight=port['weight'], operation=operation['operation'],
                                                   node=node, template=key, name=operation['name'])
            templates[key]['cases'].append(cid)
        for record in document['records']:
            binding = record['weight']
            name = binding['name']
            node = document['templates'][record['template']]['schema'][record['node']]
            if node['operation'] != 'MUL_MAT' or not any(p and p.get('weight') == name for p in node['ports']):
                raise ValueError('Projection record does not bind to its recorded operation')
            first = sorted(set(record['first_merges']))
            if any(m < 0 or m >= len(merges) for m in first):
                raise ValueError('Unknown measured merge')
            if name not in projections:
                projections[name] = dict(weight=binding, first_merges=set(),
                                         cases=defaultdict(list), variants={})
            row = projections[name]
            if row['weight'] != binding:
                raise ValueError('Native binding differs for ' + name)
            row['first_merges'].update(first)
            row['cases'][cid].append(dict(prediction=record['prediction'], execution=record['execution'],
                                          node=record['node'], template=record['template']))
            variant = tuple(first)
            row['variants'][variant] = row['variants'].get(variant, 0) + 1
            observations.add((cid, record['prediction']))
            occurrences += 1
    if merges is None:
        raise ValueError('No recorded projection sources')
    for i, merge in enumerate(merges):
        merge['index'] = i
        merge['projections'] = sorted(name for name, row in projections.items() if i in row['first_merges'])
    for row in projections.values():
        row['first_merges'] = sorted(row['first_merges'])
        row['cases'] = dict(row['cases'])
        row['variants'] = [dict(first_merges=list(k), occurrences=v) for k,v in sorted(row['variants'].items())]
    return dict(projections=projections, merges=merges, other_weighted_uses=list(other.values()),
                templates=templates, cases=cases,
                coverage=dict(cases=len(cases), prediction_observations=len(observations),
                              matrix_occurrences=occurrences, unique_projection_weights=len(projections),
                              connected_projection_weights=sum(bool(r['first_merges']) for r in projections.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, default=RUN/'shared/observation/original-model-traces/cases')
    parser.add_argument('--analysis', type=Path, default=RUN.parents[3]/'studies/04-round-2-overlap/runs/overlap')
    parser.add_argument('--bindings', type=Path, default=RUN/'shared/observation/original-model-traces/native-operators.json')
    parser.add_argument('--output', type=Path, default=RUN.parents[3]/'studies/04-round-2-overlap/runs/overlap/weight-path-catalog.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve the existing analysis generation')
    sources = []
    def documents():
        for path in sorted(args.sources.glob('*/index.json')):
            content=path.read_bytes()
            document=json.loads(content)
            sources.append(dict(case=document['summary']['case'], path=str(path),
                                sha256=hashlib.sha256(content).hexdigest()))
            yield document
    result=catalog_documents(documents())
    if len(result['merges']) != 105 or {m['layer'] for m in result['merges']} != set(range(35)):
        raise ValueError('This source does not cover the declared 35-layer, 105-merge architecture')
    binding_content=args.bindings.read_bytes()
    bindings=json.loads(binding_content)
    for name,row in result['projections'].items():
        if bindings['tensors'].get(name) != row['weight']:
            raise ValueError('Native model manifest disagrees with projection binding: ' + name)
    result['native_bindings']=dict(path=str(args.bindings),sha256=hashlib.sha256(binding_content).hexdigest(),
                                  model=bindings['model'],model_bytes=bindings['model_bytes'],
                                  tensors=bindings['tensors'])
    group_index=json.loads((args.analysis/'group-map/index.json').read_text())
    case_names=set()
    for group in group_index['groups']:
        index=json.loads(Path(group['record']).read_text())
        case_names.update(member['case'] for member in index['members'])
    result['coverage']['original_population']=group_index['members']
    result['coverage']['cases_without_anchor_projection_sources']=sorted(case_names-set(result['cases']))
    map_sources=[]
    for gpu in [0,1]:
        path=args.analysis/f'group-map/gpu{gpu}-comparison/index.json';content=path.read_bytes()
        contrast=json.loads(content)
        map_sources.append(dict(gpu=gpu,path=str(path),sha256=hashlib.sha256(content).hexdigest()))
        for merge in result['merges']:
            merge.setdefault('observed_context_differences',[]).extend(
                dict(gpu=gpu,role=s['role'],positions=s['positions'],
                     direction_squared=s['direction_squared_by_merge'][merge['index']])
                for s in contrast['summaries'])
    result['source_records']=sources
    result['difference_maps']=map_sources
    result['scope']='Structural binding catalog from all existing selected-position projection records, joined to full-response differences. No matrix is selected for editing.'
    result['limits']=[
        'The structural source records cover five climb-anchor predictions per supported case, not all passage inputs. Full-passage input separability and inverse reachability remain unmeasured.',
        'A merge may have many supplying projections and a cache writer may supply many merges. Its discrepancy is not independent credit assigned to every listed weight.',
        'An empty first_merges list concerns this recorded projection use, not every use of that tensor. token_embd also participates in GET_ROWS, retained separately.',
        'Non-MUL_MAT weighted uses and all native manifest tensors remain visible; no unsupported graph edge or input-to-GPU connection is invented.',
        'Recorded first-merge connections preserve cache fanout but do not prove the effect of an edited cache on later tokens or include a fitted downstream response.']
    publish(args.output,result)
    print(json.dumps(result['coverage'],indent=1))


if __name__=='__main__':
    main()
