"""Prepare, inspect, and run the source-bound one-pair Final Experiment."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / 'studies/04-round-2-overlap/runs'
EVIDENCE = RUNS / 'vector-chance-final-experiment-installed-348-recovery'
MEASURED = ROOT / 'records/development/2026-11-17-final-experiment/run.py'
RECOVERY = ROOT / 'records/development/2026-11-17-unmeasured-state-recovery/run.py'
MODULE = 'pipeline.weight_edit.final_experiment'
RUNTIME = ['uv', 'run', '--offline', '--python', '3.12', '--with', 'torch==2.15.0',
           '--with', 'numpy==2.5.5', '--with', 'matplotlib==3.11.2', 'python']
ENV = dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TORAD_RESEARCH_TRANSFERS='1',
           PYTHONPATH=str(ROOT / 'code/signal-trace'))
INTENT = ('why_this_direction', 'repeating_pattern_check', 'hypothesis', 'prediction', 'would_weaken_it')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    return {'record': str(path.resolve().relative_to(ROOT)), 'sha256': digest(path)}


def controller(path):
    # The recovery controller differs in checkpoint admission. Both execute the
    # same measured loss, gradient, native install and fresh acquisition.
    if path not in (MEASURED, RECOVERY):
        raise ValueError(f'unknown Final Experiment program {path}')
    spec = importlib.util.spec_from_file_location('weight_patterns_final_controller', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_path(path):
    path = path.resolve()
    if path.parent != RUNS:
        raise ValueError(f'run must be a direct child of {RUNS}')
    return path


def sources(declaration):
    for row in declaration['implementation']:
        if binding(ROOT / row['record'])['sha256'] != row['sha256']:
            raise ValueError(f"changed implementation: {row['record']}")


def prepare(run, predecessor, review, intent, stopping=None, diagnostic=None, prefix_objective=None,
            evaluate_after=None, resume_evaluation_interval=False, *, development_origin=None):
    if run.exists():
        raise FileExistsError(f'new run already exists: {run}')
    if not all(isinstance(intent.get(k), str) and intent[k].strip() for k in INTENT):
        raise ValueError('intent needs nonempty ' + ', '.join(INTENT))
    lineage = read(EVIDENCE / 'declaration.json')
    provenance = None
    if development_origin:
        from pipeline.weight_edit import group_development_origin
        provenance = group_development_origin.checked_origin(binding(development_origin), lineage, predecessor=predecessor)
    former = lineage if provenance else read(predecessor / 'declaration.json')
    from pipeline.weight_edit import joint_stock_lineage
    stock_line = None if provenance else joint_stock_lineage.from_parent(predecessor, lineage)
    previous_selection = former.get('validation_stopping')
    previous_diagnostic = former.get('trace_diagnostic')
    previous_prefix = former.get('prefix_objective')
    if resume_evaluation_interval:
        original_boundary = former.get('prefix_evaluation_boundary')
        if not previous_prefix or not original_boundary:
            raise ValueError('the predecessor has no frozen prefix evaluation interval')
        calibration = original_boundary['calibration']
        if evaluate_after is None:
            evaluate_after = ROOT / calibration['record']
        if binding(evaluate_after) != calibration:
            raise ValueError('the continued interval must use its original calibration')
    if prefix_objective is None and previous_prefix:
        inherited = previous_prefix['policy']
        prefix_objective = ROOT / inherited['record']
        if binding(prefix_objective) != inherited:
            raise ValueError('the inherited prefix objective has changed')
    if diagnostic is None and previous_diagnostic:
        inherited = previous_diagnostic['policy']
        diagnostic = ROOT / inherited['record']
        if binding(diagnostic) != inherited:
            raise ValueError('the inherited trace diagnostic has changed')
    if previous_selection and stopping is None:
        stopping = ROOT / previous_selection['policy']['record']
        if binding(stopping) != previous_selection['policy']:
            raise ValueError('the inherited stopping policy has changed')
    # Keep this release tied to the experiment it reproduces, including its
    # rounded derivative and calibration. Changing the method needs a new declaration.
    for key in ('reference', 'tolerance', 'loss_identity', 'original_model', 'gradient_step',
                'functional_loop', 'native_inputs', 'native_reading', 'eta'):
        if former[key] != lineage[key]:
            raise ValueError(f'predecessor differs from the documented baseline in {key}')
    sources(lineage)
    app = controller(MEASURED)
    previous = provenance['checkpoint'] if provenance else binding(predecessor / 'final-state.json')
    state = app.held.checked(ROOT, previous)
    observation = app.held.checked(ROOT, state['last_measurement'])
    if not app.continuation.validate(state, observation, state['last_measurement'], former['loss_identity']):
        raise ValueError('this entry point continues the complete full-parameter state')
    if (state['model_identity']['run'] != str(predecessor.relative_to(ROOT))
            or state['eta'] != former['eta'] or (not provenance and observation['loss_identity'] != former['loss_identity'])):
        raise ValueError('checkpoint, fresh observation and predecessor disagree')
    if provenance:
        if 'pristine_exit' in provenance:
            exit_record, exit_kind = provenance['pristine_exit'], 'measured_pristine_stock_origin'
        else:
            exit_record, exit_kind = provenance['legacy_checkpoint'], 'materialized_measured_development_origin'
    else:
        exit_record = binding(predecessor / 'launcher-exit.json')
        preparer = app.load_module('weight_patterns_measured_admission', MEASURED.parent / 'prepare_after_comparison.py')
        stopped = app.held.checked(ROOT, exit_record)
        try:
            exit_kind = preparer.validate_launcher_exit(stopped, state, full=True)
        except ValueError:
            if not resume_evaluation_interval:
                raise
            from pipeline.weight_edit import common_prefix_driver
            exit_kind = common_prefix_driver.archived_exit_kind(stopped, state, observation)
    payloads = app.continuation.payloads(ROOT, state, predecessor, app.held.checked)
    manifest = read(payloads[0])
    for path, key in zip(payloads[1:], ('native', 'remainder'), strict=True):
        if path.stat().st_size != manifest[key]['bytes']:
            raise ValueError(f'truncated {key} checkpoint payload {path}')

    dec = copy.deepcopy(former)
    dec.pop('validation_stopping', None)
    dec.pop('trace_diagnostic', None)
    dec.pop('prefix_objective', None)
    dec.pop('prefix_evaluation_boundary', None)
    dec.pop('native_key_anchor', None)
    dec.pop('group_development_origin', None)
    dec.pop('joint_stock_lineage', None)
    dec.pop('group_contrast', None)
    dec['parameters'].pop('evaluation_interval', None)
    dec['parameters'].pop('evaluation_interval_remaining', None)
    for stale in ('unmeasured_recovery', 'provenance_last_measurement', 'scheduled_handoff', 'retry_of',
                  'preparation_whole_build_review', 'previous_whole_build_review'):
        dec.pop(stale, None)
    dec.update({key: intent[key] for key in INTENT})
    dec.update(declared_at=app.now(), status='prepared', declaration_revision=1,
               program=binding(MEASURED), plan=binding(ROOT / 'FINAL-EXPERIMENT.md'),
               whole_build_review=binding(review), outputs=[str(run.relative_to(ROOT))],
               predecessor_exit={'record': exit_record, 'kind': exit_kind},
               revision_reason='Prospective continuation from a measured complete native checkpoint.',
               description='One-pair Final Experiment with the installed-weight learner.',
               start_fresh=state['last_measurement'], resume_checkpoint=previous,
               resume_predecessor_generation=state['iteration'],
               counter_semantics=dict(local_baseline=0,
                   historical_predecessor=(state['iteration'] if provenance else app.continuation.historical_step(former, state)),
                   meaning='Fresh generation at the restored model precedes every new update.'))
    dec['parameters']['eta']['measured_from'] = previous['record']
    _, specs = app.state_specs(app.held.checked(ROOT, dec['tensors']), app.held.checked(ROOT, dec['architecture']))
    retained = app.continuation.remainder_inventory(ROOT, state, app.held.checked)
    dec['parameters']['storage']['value'] = app.storage_budget(specs, retained, dec['native_context'], dec['storage'])
    dec['observability']['entrypoint'] = dec['program']['record']
    dec['observability']['sources'] = [dec['program'] if r['record'] == former['program']['record'] else r
                                        for r in dec['observability']['sources']]
    source_map = {r['record']: r for r in lineage['implementation']}
    launcher = binding(Path(__file__))
    source_map[launcher['record']] = launcher
    from pipeline.weight_edit import retired_admission
    if retired_admission.ARCHIVE.is_file():
        driver = binding(Path(retired_admission.__file__))
        source_map[driver['record']] = driver
        dec['retired_input_admission'] = dict(archive=binding(retired_admission.ARCHIVE), driver=driver)
    if provenance:
        dec['group_development_origin'] = provenance
        source = provenance['implementation']
        source_map[source['record']] = source
        dec['observability']['sources'] = [*dec['observability']['sources'], source]
    if stock_line:
        dec['joint_stock_lineage'] = stock_line
        for module in (joint_stock_lineage, joint_stock_lineage.group_development_origin):
            source = binding(Path(module.__file__))
            source_map[source['record']] = source
    dec['implementation'] = [source_map[k] for k in sorted(source_map)]
    original = app.held.checked(ROOT, dec['reference'])
    acquisition = app.held.checked(ROOT, observation['capture'])
    paths = [predecessor / 'declaration.json', ROOT / state['base_identity']['file'],
             ROOT / dec['original_model']['record'], ROOT / original['trace']['record'],
             ROOT / original['source_definition']['record'], ROOT / acquisition['cases']['record'],
             ROOT / previous['record'], ROOT / exit_record['record'], ROOT / dec['start_fresh']['record'],
             ROOT / observation['capture']['record'], *payloads]
    if dec.get('retired_input_admission'):
        archive = read(ROOT / dec['retired_input_admission']['archive']['record'])
        paths.append(ROOT / archive['input'])
    if provenance:
        core = ('native_inputs', 'native_reading', 'tensors', 'architecture', 'reference', 'tolerance')
        inherited = {dec[key]['record'] for key in core} | {row['record'] for row in provenance['inputs']}
    else:
        inherited = set(former['inputs'])
    if stock_line:
        inherited.update(stock_line[key]['record'] for key in ('origin','parent','group_boundary','group_observation','policy'))
    dec['inputs'] = sorted(inherited | {str(p.relative_to(ROOT)) for p in paths})
    archive = dec['consumed_trace_archive']
    archive['active_runs'] = list(dict.fromkeys([*archive['active_runs'], run.name]))
    dec['commands'] = [shlex.join(['env', f"PYTHONPATH={ENV['PYTHONPATH']}", *RUNTIME,
                                  '-m', MODULE, 'run', str(run)])]
    if stopping:
        from pipeline.weight_edit import validation_driver, validation_stopping, trace_stopping
        policy = binding(stopping)
        spec, rule = validation_driver.load_policy(policy, dec)
        driver = binding(Path(validation_driver.__file__))
        sources_for_policy = [driver, binding(Path(validation_stopping.__file__)), binding(Path(trace_stopping.__file__))]
        if spec['format'] == 'development-operation-stopping/1':
            sources_for_policy.extend([spec['scorer'], *spec['scorer_dependencies']])
        for row in sources_for_policy:
            source_map[row['record']] = row
        dec['implementation'] = [source_map[k] for k in sorted(source_map)]
        history = []
        if previous_selection and previous_selection['policy'] == policy:
            if (predecessor/'validation-decision.json').exists():
                raise ValueError('this stopping policy reached its terminal decision; the study is complete')
            history = list(previous_selection.get('history',[]))
            first = 1 if history else 0
            for n in range(first, state['iteration']+1):
                row = predecessor/'loop'/str(n)/'selection-observation.json'
                history.append(binding(row))
            if (not history or (state['iteration'] > 0 and
                    validation_driver.read(history[-1])['measurement'] != state['last_measurement'])):
                raise ValueError('cannot recover a contiguous stopping history at this measured state')
            dec['inputs'] = sorted(set(dec['inputs']) | {r['record'] for r in history})
        dec['validation_stopping'] = dict(policy=policy, driver=driver, history=history, bindings=rule.bindings)
        study = RUNS.parent
        data_paths = [r['record'] for r in rule.bindings if (ROOT/r['record']).is_relative_to(RUNS)
                      or (ROOT/r['record']).is_relative_to(study/'inputs')]
        dec['inputs'] = sorted(set(dec['inputs']) | set(data_paths))
        dec['stopping'] = 'Synchronous fresh fitted/selection measurements under the bound development policy.'
        dec['observability']['entrypoint'] = driver['record']
        dec['parameters']['selection_schedule'] = dict(value=dict(cadence=rule.cadence, budget=rule.budget),
            measured_from=spec['calibration']['record'], derivation='One more than the longest recovered dry spell; one prior trajectory of steps as the prospective compute allowance.',
            scales_with='The length and recovered drawdowns of the bound same-loss calibration trajectory.')
        fields = dec['observability']['context_fields']
        dec['observability']['context_fields'] = list(dict.fromkeys([*fields,'measurement_role','validation_member']))
    else:
        dec['stopping'] = lineage['stopping']
        dec['parameters'].pop('selection_schedule', None)
    if diagnostic:
        from pipeline.weight_edit import validation_driver, trace_diagnostic, validation_stopping
        policy = binding(diagnostic)
        spec = trace_diagnostic.registration(policy)
        driver = binding(Path(validation_driver.__file__))
        hooks = [driver, binding(Path(trace_diagnostic.__file__)), binding(Path(validation_stopping.__file__)),
                 spec['reader'],*spec['dependencies']]
        for row in hooks:
            source_map[row['record']] = row
        dec['implementation'] = [source_map[k] for k in sorted(source_map)]
        inputs = [policy,spec['artifact'],*spec['inputs']]
        dec['trace_diagnostic'] = dict(policy=policy, driver=driver, bindings=inputs)
        measured = [row['record'] for row in inputs if (ROOT/row['record']).is_relative_to(RUNS)
                    or (ROOT/row['record']).is_relative_to(RUNS.parent/'inputs')]
        dec['inputs'] = sorted(set(dec['inputs']) | set(measured))
        obs = dec['observability']
        obs['entrypoint'] = driver['record']
        known_sources = {r['record']:r for r in [*obs['sources'],*hooks]}
        obs['sources'] = list(known_sources.values())
        obs['additional_metrics'] = list(dict.fromkeys([*obs['additional_metrics'],*trace_diagnostic.METRICS]))
        obs['context_fields'] = list(dict.fromkeys([*obs['context_fields'],'diagnostic_key']))
    if prefix_objective:
        from pipeline.weight_edit import common_prefix_driver
        common_prefix_driver.configure(dec, prefix_objective)
        if evaluate_after:
            common_prefix_driver.configure_evaluation(dec, evaluate_after, continue_interval=resume_evaluation_interval)
    elif evaluate_after:
        raise ValueError('the fixed evaluation interval accompanies a frozen prefix objective')
    return dec


def verify_stock_group_policy(declaration):
    """Bind the group objective to the stock checkpoint selected by this continuation."""
    line = declaration.get('joint_stock_lineage')
    if line:
        from pipeline.weight_edit import joint_stock_lineage
        joint_stock_lineage.check_successor(line, declaration)


def inspect_run(run, dec):
    if dec['outputs'] != [str(run.relative_to(ROOT))] or dec['operation'] != 'final-experiment-full-computational-training':
        raise ValueError('output directory or operation differs from its declaration')
    path = ROOT / dec['program']['record']
    if path not in (MEASURED, RECOVERY) or binding(path) != dec['program']:
        raise ValueError('unknown or modified Final Experiment controller')
    sources(dec)
    if path == MEASURED and 'unmeasured_recovery' in dec:
        raise ValueError('measured controller cannot admit an unmeasured recovery')
    app = controller(path)
    restored_provenance = dec.get('retired_input_admission')
    if restored_provenance:
        from pipeline.weight_edit import retired_admission
        if path != MEASURED or restored_provenance['driver'] != binding(Path(retired_admission.__file__)):
            raise ValueError('unexpected restored-input controller')
        frozen = retired_admission.bind(app, dec)
        frozen(app.held.checked(ROOT, dec['native_inputs']))
    app.held.checked(ROOT, dec['whole_build_review'], parse=False)
    app.held.checked(ROOT, dec['plan'], parse=False)
    state = app.held.checked(ROOT, dec['resume_checkpoint'])
    key = 'start_fresh' if path == MEASURED else 'provenance_last_measurement'
    observation = app.held.checked(ROOT, dec[key])
    if path == MEASURED:
        app.continuation.validate(state, observation, dec[key], dec['loss_identity'])
    else:
        app.recovery.validate(ROOT, dec, state, observation, app.held.checked, app.continuation)
    missing = [name for name in dec['inputs'] if not (ROOT / name).exists()]
    if missing:
        raise FileNotFoundError('missing declared inputs: ' + ', '.join(missing))
    data = app.continuation.payloads(ROOT, state, (ROOT / dec['resume_checkpoint']['record']).parent, app.held.checked)
    manifest = read(data[0])
    for payload, key in zip(data[1:], ('native', 'remainder'), strict=True):
        if payload.stat().st_size != manifest[key]['bytes']:
            raise ValueError(f'truncated {key} payload {payload}')
    case = app.held.checked(ROOT, observation['capture'])['cases']
    app.held.checked(ROOT, case)
    original = app.held.checked(ROOT, dec['reference'])
    engine = app.held.checked(ROOT, dec['native_reading'])['engine']
    for required_file in (original['trace']['record'], state['base_identity']['file'],
                          dec['original_model']['record'], dec['tolerance']['record']):
        if not (ROOT / required_file).is_file():
            raise FileNotFoundError(required_file)
    required = dec['parameters']['storage']['value']['free_required_bytes']
    available = shutil.disk_usage(run if run.exists() else run.parent).free
    stopping = dec.get('validation_stopping')
    if stopping:
        from pipeline.weight_edit import validation_driver
        if path != MEASURED or stopping['driver'] != binding(Path(validation_driver.__file__)):
            raise ValueError('unknown selection driver or unsupported bootstrap')
        _, rule = validation_driver.load_policy(stopping['policy'], dec)
        if rule.bindings != stopping['bindings']:
            raise ValueError('stopping calibration or case input inventory differs from its declaration')
        validation_driver.replay(rule,stopping)
    diagnostic = dec.get('trace_diagnostic')
    if diagnostic:
        from pipeline.weight_edit import trace_diagnostic, validation_driver
        if path != MEASURED or diagnostic['driver'] != binding(Path(validation_driver.__file__)):
            raise ValueError('unknown trace diagnostic driver or unsupported bootstrap')
        spec = trace_diagnostic.registration(diagnostic['policy'])
        if diagnostic['bindings'] != [diagnostic['policy'],spec['artifact'],*spec['inputs']]:
            raise ValueError('trace diagnostic inputs changed after registration')
    prefix = dec.get('prefix_objective')
    if prefix:
        from pipeline.weight_edit import common_prefix_driver
        if path != MEASURED or prefix != common_prefix_driver.configuration(prefix['policy']):
            raise ValueError('changed prefix objective bindings or unsupported bootstrap')
    boundary = dec.get('prefix_evaluation_boundary')
    if boundary and (not prefix or boundary != common_prefix_driver.checked_evaluation(dec, boundary)):
        raise ValueError('changed evaluation boundary calibration or unsupported objective')
    if (dec.get('predecessor_exit', {}).get('kind') == 'completed_fresh_checkpoint_then_archive_readback_failure'
            and not (boundary or {}).get('continuation')):
        raise ValueError('archive readback recovery must complete the original bounded evaluation interval')
    provenance = dec.get('group_development_origin')
    if provenance:
        from pipeline.weight_edit import group_development_origin
        if (not dec.get('group_contrast') or provenance != group_development_origin.checked_origin(provenance['origin'], dec)
                or dec['resume_checkpoint'] != provenance['checkpoint'] or dec['start_fresh'] != provenance['measurement']):
            raise ValueError('changed materialized development source bindings')
    if path == MEASURED and not provenance and (dec.get('group_contrast') or dec.get('joint_stock_lineage')):
        from pipeline.weight_edit import joint_stock_lineage
        parent = (ROOT / dec['resume_checkpoint']['record']).parent
        expected = joint_stock_lineage.from_parent(parent, read(EVIDENCE / 'declaration.json'))
        if expected != dec.get('joint_stock_lineage'):
            raise ValueError('Missing or changed inherited stock lineage')
        verify_stock_group_policy(dec)
    return dict(program=dec['program'], stopping=stopping, trace_diagnostic=diagnostic, prefix_objective=prefix,
                prefix_evaluation_boundary=boundary,
                group_development_origin=provenance,
                verified_retired_input=restored_provenance,
                implementation_files=len(dec['implementation']),
                inputs=len(dec['inputs']), model=dec['original_model'], seed=state['base_identity'],
                reference=dec['reference'], reference_trace=original['trace'], tolerance=dec['tolerance'],
                cases=case, checkpoint=state['checkpoint'], native_payload=manifest['native'],
                remainder_payload=manifest['remainder'], reference_member=original['reference'],
                training_members=dec['functional_loop']['training_members'], eligible_tensors=len(manifest['layout']),
                hardware=dec['hardware'], native_reading=dec['native_reading'], runtime=RUNTIME,
                recorder=engine['recorder'], libraries=engine['libraries'],
                historical_start=dec['counter_semantics']['historical_predecessor'],
                storage_needed_bytes=required, storage_available_bytes=available, storage_ready=available >= required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('prepare', help='draft a measured continuation; stdout only unless --write is supplied')
    plan.add_argument('run', type=Path)
    plan.add_argument('--from-run', type=Path, required=True)
    plan.add_argument('--review', type=Path, required=True)
    plan.add_argument('--intent', type=Path, required=True, help='JSON object with ' + ', '.join(INTENT))
    plan.add_argument('--stopping-policy', type=Path, help='bound whole-trace development stopping policy')
    plan.add_argument('--trace-diagnostic',type=Path,help='observe a frozen computation-derived signal on native/JVP forwards')
    plan.add_argument('--prefix-objective', type=Path, help='train toward a frozen common-prefix turn displacement')
    plan.add_argument('--evaluate-after', type=Path, help='exit at the evaluation interval derived from bound fitted history')
    plan.add_argument('--resume-evaluation-interval', action='store_true',
                      help='complete the interrupted prefix evaluation interval from its last measured state')
    plan.add_argument('--write', action='store_true')
    check = sub.add_parser('check', help='verify declared sources and checkpoint metadata without running')
    check.add_argument('run', type=Path)
    check.add_argument('--declaration', type=Path, help='read a draft for this output directory')
    sub.add_parser('run', help='start the recorded learner in a new declared output').add_argument('run', type=Path)
    args = parser.parse_args()
    run = run_path(args.run)
    if args.action == 'prepare':
        dec = prepare(run, run_path(args.from_run), args.review, read(args.intent), args.stopping_policy,
                      args.trace_diagnostic, args.prefix_objective, args.evaluate_after, args.resume_evaluation_interval)
        if args.write:
            # Validate all dependencies and serialize before creating the output.
            inspect_run(run, dec)
            encoded = json.dumps(dec, indent=2, allow_nan=False) + '\n'
            run.mkdir()
            with (run / 'declaration.json').open('x') as handle:
                handle.write(encoded)
            print(run / 'declaration.json')
        else:
            print(json.dumps(dec, indent=2, allow_nan=False))
        return
    dec = read(getattr(args, 'declaration', None) or run / 'declaration.json')
    summary = inspect_run(run, dec)
    if args.action == 'check':
        print(json.dumps(summary, indent=2))
        return
    if any((run / name).exists() for name in ('working-state.json', 'progress.jsonl', 'loop',
                                              'launcher-exit.json', 'observability', 'owner.lock')):
        parser.error('this output has already been used; continue from its checkpoint in a new declaration')
    if not summary['storage_ready']:
        parser.error(f"checkpoint and capture reserve needs {summary['storage_needed_bytes']} bytes; "
                     f"{summary['storage_available_bytes']} available")
    # Fresh interpreter keeps preparation/test imports outside the reviewed
    # controller's loaded code inventory. Its first action is check_native_run.
    os.chdir(ROOT)
    entry = (dec.get('prefix_objective') or dec.get('trace_diagnostic') or dec.get('validation_stopping')
             or dec.get('retired_input_admission') or {}).get('driver', dec['program'])
    os.execve(sys.executable, [sys.executable, str(ROOT / entry['record']), str(run)], os.environ | ENV)


if __name__ == '__main__':
    main()
