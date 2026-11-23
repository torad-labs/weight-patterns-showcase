"""Group-first connected overlaps with individual, time-resolved recurrence.

The reference is a coordinate grid, not a desired response. No failure's mate
supplies a target. All descriptive directions are computed from whole groups.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.paths import ROOT
from pipeline.native_records.run_declaration import check_native_run, digest
from pipeline.correspondence.solution_correspondence import annotate, correspond
from pipeline.overlap.analyze import read, write, finite
from pipeline.overlap.statistics import group_statistics
from pipeline.overlap.cross_validation import held_out_difference


def error_stage(label, prediction):
    error = label['first_quantitative_error']
    if error is None:
        return 'no_target_error'
    if prediction < error['statement_first_decision']:
        return 'before_statement'
    wrong = error['first_wrong_operator']['decision']
    if prediction < wrong:
        return 'statement_before_wrong_token'
    if prediction == wrong:
        return 'wrong_token_prediction'
    return 'error_conditioned'


def main(output, pre_error=False, source=None):
    output = Path(output).resolve()
    declaration = output/'declaration.json'
    destination = output/('pre-error' if pre_error else 'calculation')
    check_native_run(destination, declaration)
    if destination.exists():
        raise ValueError('Preserve the existing overlap')
    source = Path(source).resolve() if source else output.parent/'acquisition-single-card'
    cases = read(source/'original-questions.json')
    labels = {r['id']: r for r in read(source/'labels-original-gpu0.json')['rows']}
    ids = [c['id'] for c in cases]
    if set(ids) != set(labels) or len(ids) != len(set(ids)):
        raise ValueError('Original population must occur exactly once')
    annotations, arrays, norms, metadata = {}, {}, {}, {}
    for case in cases:
        cid = case['id']; raw = source/'original-gpu0'/cid
        result = read(raw/'result.json')
        if digest(raw/'result.json') != labels[cid]['result_sha256']:
            raise ValueError('Reviewed response changed')
        if result['teacher_token_count'] or result['conditioned_continuation']:
            raise ValueError('Only natural responses belong here')
        annotations[cid] = annotate(result, prompt_tokens=case['tokens'])
        directory = source/'exported-original-gpu0'/cid
        metadata[cid] = read(directory/'index.json')
        arrays[cid] = np.load(directory/'turns.npy', mmap_mode='r')
        norms[cid] = np.load(directory/'norms.npy', mmap_mode='r')
        if arrays[cid].shape[0] != len(result['steps']):
            raise ValueError('Export must cover the entire response')
    reference = ids[0]
    npos, nmerge, width = arrays[reference].shape
    connections = metadata[reference]['connections']
    if any(m['connections'] != connections for m in metadata.values()):
        raise ValueError('All members must follow the same connected merges')
    destination.mkdir()
    aligned = np.full((len(ids), npos), -1, dtype='i4')
    role = [None]*npos
    for t in annotations[reference]['tokens']:
        if t['section'] is not None:
            role[t['prediction']] = annotations[reference]['sections'][t['section']]['role']
    member_records = []
    for i, case in enumerate(cases):
        cid = case['id']
        mapping = correspond(annotations[reference], annotations[cid])
        for row in mapping['predictions']:
            aligned[i, row['left_prediction']] = row['right_prediction']
        member_records.append(dict(id=cid, role=labels[cid]['role'],
            parameters=case['parameters'], family=case['family'],
            source=str(source/'original-gpu0'/cid),
            annotation=annotations[cid], label=labels[cid],
            correspondence=mapping, aligned=int((aligned[i]>=0).sum()),
            total=len(annotations[cid]['tokens'])))
    if pre_error:
        # A member can be before its error while a peer at the aligned position
        # has already emitted one. Exclude those peers from this entire overlay,
        # not merely from the final report of the held-out member.
        for i,cid in enumerate(ids):
            error=labels[cid]['first_quantitative_error']
            if error is not None:
                aligned[i,aligned[i]>error['first_wrong_operator']['decision']]=-1
        for i,m in enumerate(member_records):m['active_aligned']=int((aligned[i]>=0).sum())
    np.save(destination/'aligned.npy', aligned)
    write(destination/'members.json', member_records)
    shape = (npos, nmerge, width)
    means = {g: np.lib.format.open_memmap(destination/f'{g}-mean.npy', mode='w+',
             dtype='f8', shape=shape) for g in ('failure','success')}
    for v in means.values(): v[:] = np.nan
    member_shape = (len(ids), npos, nmerge)
    margins = np.full(member_shape, np.nan, dtype='f4')
    gain_margins = np.full(member_shape, np.nan, dtype='f4')
    agreement = np.full(member_shape, np.nan, dtype='f4')
    member_delta_cosine = np.full(member_shape, np.nan, dtype='f4')
    lengths = np.full((npos, nmerge), np.nan)
    stability = np.full_like(lengths, np.nan)
    gain_difference = np.full_like(lengths, np.nan)
    group_metrics = np.full((2, npos, nmerge, 3), np.nan)
    counts = np.zeros((npos, 2), dtype='i4')
    strict = np.zeros((npos, nmerge), dtype=bool)
    for p in range(npos):
        groups = [np.array([i for i,c in enumerate(ids) if aligned[i,p]>=0
                    and (labels[c]['role']=='failure') == failure], dtype=int)
                  for failure in (True,False)]
        values, gains = [], []
        for group, (name, ix) in enumerate(zip(('failure','success'), groups)):
            counts[p,group] = len(ix)
            if not len(ix):
                values.append(None); gains.append(None); continue
            x = np.stack([arrays[ids[i]][aligned[i,p]] for i in ix]).astype('f8')
            ns = np.stack([norms[ids[i]][aligned[i,p]] for i in ix])
            if np.any(ns<=0): raise ValueError('Undefined gain')
            g = np.log(ns[:,:,1]/ns[:,:,0])
            stats = group_statistics(x)
            means[name][p] = stats['mean']
            agreement[ix,p] = stats['loo_cosine']
            group_metrics[group,p] = np.stack([stats['dispersion'],g.mean(0),g.var(0)],axis=-1)
            values.append(x); gains.append(g[:,:,None])
        # Each held-out group needs at least one other member. Undefined scores
        # remain missing; this mathematical requirement is not a sampling gate.
        if min(map(len, groups)) < 2: continue
        cv = held_out_difference(*values)
        gm = held_out_difference(*gains)
        order = np.concatenate(groups)
        margins[order,p] = cv['signed_margin']
        gain_margins[order,p] = gm['signed_margin']
        stability[p] = cv['direction_pairwise_cosine']
        d = cv['difference']; lengths[p] = np.sum(d*d,axis=-1)
        gain_difference[p] = gm['difference'][:,0]
        # Direction of each leave-one-out difference relative to the full
        # overlap, separately from the member's own-side evidence.
        nd = np.linalg.norm(d,axis=-1)
        nl = np.linalg.norm(cv['directions'],axis=-1)
        member_delta_cosine[order,p] = np.divide(
            np.sum(cv['directions']*d,axis=-1),nl*nd,
            out=np.full(nl.shape,np.nan),where=nl*nd>0)
        strict[p] = np.all(cv['signed_margin']>0,axis=0)&np.all(agreement[order,p]>0,axis=0)
        if p % 50 == 0: print(f'overlap position {p}/{npos}',flush=True)
    for array in means.values(): array.flush()
    np.savez(destination/'evidence.npz', margins=margins,gain_margins=gain_margins,
        agreement=agreement,member_delta_cosine=member_delta_cosine,
        difference_squared=lengths,stability=stability,gain_difference=gain_difference,
        group_metrics=group_metrics,counts=counts,strict=strict)
    summaries=[];members=[]
    for passage in dict.fromkeys(role):
        selected=np.array([p for p,r in enumerate(role) if r==passage and min(counts[p])>=2])
        if not len(selected): continue
        summaries.append(dict(role=passage,positions=len(selected),
            full_population_positions=int(np.all(counts[selected]==[sum(labels[c]['role']=='failure' for c in ids),sum(labels[c]['role']!='failure' for c in ids)],axis=1).sum()),
            counts_min=counts[selected].min(0),counts_max=counts[selected].max(0),
            squared_by_merge=np.nanmean(lengths[selected],axis=0),
            strict_squared_by_merge=np.nanmean(lengths[selected]*strict[selected],axis=0),
            strict_cells=int(strict[selected].sum()),
            direction_stability=np.nanmean(stability[selected],axis=0)))
    for i, member in enumerate(member_records):
        contexts=[]
        for passage in dict.fromkeys(role):
            for stage in dict.fromkeys(error_stage(labels[ids[i]],int(q)) for q in aligned[i] if q>=0):
                pp=np.array([p for p,r in enumerate(role) if r==passage and aligned[i,p]>=0 and
                    error_stage(labels[ids[i]],int(aligned[i,p]))==stage and np.isfinite(margins[i,p]).any()],dtype=int)
                if not len(pp): continue
                v=margins[i,pp]
                contexts.append(dict(role=passage,stage=stage,positions=len(pp),
                    predictions=[int(aligned[i,p]) for p in pp],reference_positions=pp,
                    margin_by_merge=np.nanmean(v,axis=0),
                    positive_by_merge=np.sum(v>0,axis=0),valid_by_merge=np.sum(np.isfinite(v),axis=0),
                    agreement_by_merge=np.nanmean(agreement[i,pp],axis=0),
                    gain_margin_by_merge=np.nanmean(gain_margins[i,pp],axis=0)))
        members.append(dict(id=ids[i],role=member['role'],parameters=member['parameters'],
            first_error=labels[ids[i]]['first_quantitative_error'],contexts=contexts))
    write(destination/'member-evidence.json',finite(members))
    write(destination/'summary.json',finite(dict(reference=reference,members=ids,
        population=dict(Counter(labels[c]['role'] for c in ids)),roles=role,
        connections=connections,summaries=summaries,
        source_hashes={str(p):digest(p) for p in [source/'original-questions.json',source/'labels-original-gpu0.json']},
        pre_error_peers_only=pre_error,
        scope='Group-first observed direction/gain. Strict support is descriptive, not a purified capability or causal proof.')))
    print('overlap complete',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--pre-error',action='store_true')
    p.add_argument('--source',type=Path,help='folder with original-questions.json, labels-original-gpu0.json, original-gpu0/ and exported-original-gpu0/')
    a=p.parse_args()
    main(a.output,a.pre_error,a.source)
