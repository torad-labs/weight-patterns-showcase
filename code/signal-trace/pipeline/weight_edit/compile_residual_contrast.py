"""Run the declared, source-only common-donor signed recurrence diagnostic."""
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.native_records.native_trace import checkpoint_values
from pipeline.native_records.run_declaration import check_native_run
from pipeline.overlap.complete_overlap import member_means
from pipeline.paths import ROOT
from pipeline.weight_edit import compile_context_targets as compiled
from pipeline.weight_edit import context_targets as c
from pipeline.weight_edit import residual_contrast as rc

ESTIMATORS = ('affine', 'mates')
VARIANTS = ('affine-raw', 'affine-adjusted', 'mates-raw', 'mates-adjusted', 'registration')


def checked(record):
    if c.binding(ROOT/record['record'], ROOT) != record:
        raise ValueError(f"Source binding changed: {record['record']}")
    return ROOT/record['record']


def add_summary(total, value):
    """Complete only if all constituent source-counted terms are complete."""
    for name in ('observed', 'unknown', 'unknown_mass'):
        total[name] = total.get(name, 0) + value[name]
    total['complete'] = total['observed'] if total['unknown'] == 0 else None


def completion_record(result):
    """Publish the required research envelope without altering diagnostic values."""
    return dict(
        prediction_vs_result=dict(
            prediction='A common-donor signed residual recurs beyond same-member squared error and registration distortion.',
            coverage={key: result[key] for key in ('members', 'rows', 'merges')},
            observed='Exact native and signed recurrence readings are retained in the native, recurrence and cross_fold fields.'),
        failures_only=dict(unknown_native_terms=sum(v['unknown'] for v in result['native'].values()),
                           limits=['Fixed fitting folds share estimator sources, so they are not independent trials.',
                                   'Recurrence does not identify capability causality or remove context/wording confounding.',
                                   'No native model edit, repair or no-fit transfer was tested.']),
        what_this_changes='The full-field signed diagnostic is measured under common donors; it does not by itself admit a target or native construction.',
        **result)


def joined_members(registered, native, wide):
    """Retain the complete token identity required by the existing row mapper."""
    if len(registered['members']) != len(native['answers']):
        raise ValueError('Native and registered member counts differ')
    every = []
    for member, row in zip(registered['members'], native['answers']):
        if member['id'] != row['id'] or member['correct'] != row['correct']:
            raise ValueError('Native and registered member identities differ')
        original = wide[member['id']]
        every.append(dict(member, steps=row['steps'], question=row['question_positions'],
                          parameters=original['parameters'], tokens=original['tokens'],
                          recorded=original['recorded_tokens']))
    return every


def load_original(parent) -> dict[str, Any]:
    """Use the completed compilation's exact members, folds and fitting slices."""
    factory = compiled.read_json(parent/'factory.json')
    for key, record in factory['sources'].items():
        if key == 'registration_pairs':
            for entry in record.values():
                checked(entry)
        else:
            checked(record)
    source_decl = compiled.read_json(checked(factory['declaration']))
    sources = {key: ROOT/value for key, value in source_decl['sources'].items()}
    native = compiled.read_json(sources['native_rows'])
    registered = compiled.read_json(sources['registration']/'members.json')
    wide = {m['id']: m for m in compiled.read_json(sources['members'])}
    every = joined_members(registered, native, wide)
    selected = set(factory['rows']['fitting'])
    fitting = np.asarray([m['id'] in selected for m in every])
    members = [m for m, take in zip(every, fitting) if take]
    if [m['id'] for m in members] != factory['rows']['fitting']:
        raise ValueError('Fitting inventory changed')
    rows, spans, total = compiled.selected_rows(every, fitting)
    if spans != factory['rows']['selected_original_spans']:
        raise ValueError('Original fitting row slices changed')
    mapping = c.row_map(members, {})
    schema = compiled.read_json(parent/'observation-schema.json')
    folds = rc.held_sets(members, factory['rows']['folds'])
    pairs = [np.load(sources['registration']/'pairs'/(m['id']+'.npy'), allow_pickle=False) for m in members]
    prompts = [np.asarray(factory['prompt_registration']['pairs'][m['id']], dtype=np.int64) for m in members]
    tensors = compiled.read_json(sources['tensors'])
    weight = checkpoint_values(sources['model'], tensors['output_norm.weight']).astype('<f8').ravel()
    last = native['layers'][-1]
    scale_name = f'blk.{last}.layer_output_scale.weight'
    scale = checkpoint_values(sources['model'], tensors[scale_name]).astype('<f8').ravel()
    terminal = schema['terminal']
    import hashlib
    if hashlib.sha256(weight.tobytes()).hexdigest() != terminal['output_norm_sha256']:
        raise ValueError('Original output norm changed')
    if scale.shape != (1,) or float(scale[0]) != terminal['layer_output_scale'][scale_name]:
        raise ValueError('Original terminal scale changed')
    return dict(factory=factory, sources=sources, members=members, rows=rows, total=total,
                mapping=mapping, schema=schema, folds=folds, pairs=pairs, prompts=prompts,
                weight=weight, scale=float(scale[0]), terminal=terminal,
                keys=source_decl['context_parameters'])


def compile_run(run):
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT/'studies/04-round-2-overlap/runs'):
        raise ValueError('Outputs must live in the declared runs root')
    declaration = check_native_run(run, run/'declaration.json')
    if declaration['method_step'] != 3 or declaration['operation'] != 'signed-residual-contrast':
        raise ValueError('This command implements only the declared source-only diagnostic')
    for entry in declaration['implementation'] + declaration['bindings']:
        checked(entry)
    if (run/'factory.json').exists() or (run/'fields').exists():
        raise FileExistsError('Preserve prior or partial diagnostics; never overwrite a run')
    parent = ROOT/declaration['parent']
    data = load_original(parent)
    factory, members, folds = data['factory'], data['members'], data['folds']
    nodes, width = data['schema']['merges'], len(data['weight'])
    if factory['model_sha256'] != declaration['model_sha256']:
        raise ValueError('Original model binding differs')
    mapping = data['mapping']
    spans = [(slice(m['answer_rows_from'], m['answer_rows_from']+m['steps']),
              slice(m['question_rows_from'], m['question_rows_from']+m['question'])) for m in mapping['answers']]
    segments = [('answer', data['pairs'], factory['answer_registration']['places']),
                ('question', data['prompts'], factory['prompt_registration']['places'])]
    coefficients, coefficient_record = {}, []
    for fold, (recipients, donors) in enumerate(folds):
        contexts = [[1]+[members[i]['parameters'][k] for k in data['keys']] for i in donors]
        entry: dict[str, Any] = dict(fold=fold, recipients=[members[i]['id'] for i in recipients], donors=[members[i]['id'] for i in donors], coefficients={})
        for i in recipients:
            m = members[i]
            values = dict(affine=c.affine_weights(contexts, [1]+[m['parameters'][k] for k in data['keys']]),
                          mates=c.mate_weights([members[j]['parameters'] for j in donors], m['parameters']))
            coefficients[i] = values
            entry['coefficients'][m['id']] = {name: None if alpha is None else [str(v) for v in alpha] for name, alpha in values.items()}
        coefficient_record.append(entry)
    masses = {}
    for segment, (part, correspondence, places) in enumerate(segments):
        masses[part] = np.mean([rc.axis_mass(pairs, m['steps'] if segment == 0 else m['question'], places, m['steps']+m['question'])
                               for m, pairs in zip(members, correspondence)], axis=0)
    # Exact logical shapes of kept statistics, not a sampling/residency cap.
    per_node = sum(len(VARIANTS)*places*(len(folds)*(width+1+4)+len(members)*2)*np.dtype('<f8').itemsize
                   for _, _, places in segments)
    floor = declaration['parameters']['disk_floor']['value_bytes']
    if shutil.disk_usage(run).free - per_node*len(nodes) < floor:
        raise ValueError('Complete diagnostic arrays would cross the existing disk floor')
    c.write_json(run/'factory.json', dict(parent=c.binding(parent/'factory.json', ROOT), source=declaration['bindings'],
                 members=mapping['answers'], folds=coefficient_record, merges=nodes, rows=len(data['rows']),
                 registered_mass={part: mass.tolist() for part, mass in masses.items()},
                 logical_array_bytes=per_node*len(nodes), representation='float64 signed fields and scalar diagnostics',
                 intermediate_vectors='Each member computed completely; only fold vectors and member inner products persisted'))
    (run/'fields').mkdir()
    summaries, native_totals, cross_fold = {}, {}, {}
    for node in nodes:
        index = node['merge']
        if shutil.disk_usage(run).free - per_node < floor:
            raise ValueError('Stopped before the existing disk floor; preserve partial output')
        stem = f"{node['kind']}-{node['layer']}.f32"
        unit, ub = compiled.read_selected(data['sources']['baseline']/('state-'+stem), data['rows'], data['total'], width)
        size, sb = compiled.read_selected(data['sources']['baseline']/('size-'+stem), data['rows'], data['total'], 1)
        source = compiled.read_json(parent/'source-slices'/f'{index}.json')
        if ub != source['unit'] or sb != source['log_size']:
            raise ValueError(f'Merge {index}: original fitting slices changed')
        size = size[:, 0]
        if node['space'] == 'output_norm':
            unit, size = c.terminal_features(unit, size, data['weight'], data['scale'], data['terminal']['eps'])
        unit, size, _ = c.features(unit, size)
        arrays, node_report = {}, dict(merge=index, native={}, recurrence={}, cross_fold={})
        for segment, (part, correspondence, places) in enumerate(segments):
            source_u = [member_means(unit[spans[i][segment]], correspondence[i], places) for i in range(len(members))]
            source_s = [member_means(size[spans[i][segment]], correspondence[i], places) for i in range(len(members))]
            values = {v: {'direction': np.full((len(members), places, width), np.nan), 'size': np.full((len(members), places, 1), np.nan)} for v in VARIANTS}
            for recipients, donors in folds:
                for i in recipients:
                    m, span = members[i], spans[i][segment]
                    coefficient = 1/(len({x['correct'] for x in members})*sum(x['correct'] == m['correct'] for x in members)*(m['steps']+m['question'])*len(nodes))
                    for estimator in ESTIMATORS:
                        alpha = coefficients[i][estimator]
                        pu = c.combine(alpha, [source_u[j] for j in donors]) if alpha is not None else np.full((places, width), np.nan)
                        ps = c.combine(alpha, [source_s[j] for j in donors]) if alpha is not None else np.full(places, np.nan)
                        got = rc.residual_fields(unit[span], size[span], pu, ps, correspondence[i], places)
                        for kind in ('raw', 'adjusted'):
                            values[estimator+'-'+kind]['direction'][i] = got[kind+'_direction']
                            values[estimator+'-'+kind]['size'][i, :, 0] = got[kind+'_size']
                        if estimator == ESTIMATORS[0]:
                            for channel in ('direction', 'size'):
                                field = got['registration_'+channel]
                                values['registration'][channel][i] = field if channel == 'direction' else field[:, None]
                        for name, reading in got['native'].items():
                            if name.startswith('registration') and estimator != ESTIMATORS[0]:
                                continue
                            label = name if name.startswith('registration') else estimator+'-'+name
                            key = m['id']+'/'+label
                            weighed = dict(reading, observed=reading['observed']*coefficient,
                                           complete=None if reading['complete'] is None else reading['complete']*coefficient,
                                           unknown_mass=reading['unknown_mass']*coefficient)
                            add_summary(node_report['native'].setdefault(key, {}), weighed)
                            add_summary(native_totals.setdefault(key, {}), weighed)
            mass = masses[part]/len(nodes)
            for variant in VARIANTS:
                for channel in ('direction', 'size'):
                    key = variant+'/'+part+'/'+channel
                    statistics = [rc.field_statistics(values[variant][channel][recipients], [members[i]['correct'] for i in recipients]) for recipients, _ in folds]
                    for name in ('gap', 'cross_member', 'gap_squared'):
                        arrays[key+'/'+name] = np.stack([s[name] for s in statistics])
                    arrays[key+'/member_dot'] = np.empty((len(members), places))
                    for fold, ((recipients, _), stats) in enumerate(zip(folds, statistics)):
                        arrays[key+'/member_dot'][recipients] = stats['member_dot']
                        for name in ('cross_member', 'gap_squared'):
                            label = key+f'/fold-{fold}/'+name
                            reading = rc.complete_sum(stats[name], mass)
                            node_report['recurrence'][label] = reading
                            add_summary(summaries.setdefault(label, {}), reading)
                    for left in range(len(folds)):
                        for right in range(left+1, len(folds)):
                            label = key+f'/{left}-{right}'
                            reading = rc.complete_sum(np.sum(statistics[left]['gap']*statistics[right]['gap'], axis=-1), mass)
                            node_report['cross_fold'][label] = reading
                            add_summary(cross_fold.setdefault(label, {}), reading)
            del values, source_u, source_s
        np.savez(run/'fields'/f'{index:03d}.npz', **arrays)
        c.write_json(run/'nodes'/f'{index:03d}.json', node_report)
        print(json.dumps(dict(merge=index, of=len(nodes), status='written')), flush=True)
    for label, reading in cross_fold.items():
        key, pair = label.rsplit('/', 1)
        left, right = pair.split('-')
        a = summaries[key+f'/fold-{left}/gap_squared']['complete']
        b = summaries[key+f'/fold-{right}/gap_squared']['complete']
        reading['cosine'] = (reading['complete']/np.sqrt(a*b)
                             if reading['complete'] is not None and a is not None and b is not None and a > 0 and b > 0 else None)
    result = dict(members=len(members), rows=len(data['rows']), merges=len(nodes), fitting=[m['id'] for m in members],
                  folds=coefficient_record, native=native_totals, recurrence=summaries, cross_fold=cross_fold,
                  field_files=[c.binding(run/'fields'/f"{node['merge']:03d}.npz", ROOT) for node in nodes],
                  interpretation='Signed residual recurrence diagnostic only; no capability target, native edit or transfer established.')
    c.write_json(run/'result.json', completion_record(result))
    print(json.dumps(dict(status='complete', members=len(members), rows=len(data['rows']), merges=len(nodes))), flush=True)


if __name__ == '__main__':
    compile_run(sys.argv[1])
