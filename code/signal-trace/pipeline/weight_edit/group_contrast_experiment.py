"""Prepare and launch a source-bound signed group contrast continuation."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import sys

from pipeline.weight_edit import final_experiment as baseline, group_contrast_driver as driver
from pipeline.weight_edit import group_contrast_objective as objective, native_key_experiment as native
from pipeline.weight_edit import common_prefix_driver, common_prefix_objective
from pipeline.weight_edit import group_disk_state, host_pages

DISPLAY_VARIABLES = ('DISPLAY', 'WAYLAND_DISPLAY', 'XAUTHORITY')  # a run outlives the desktop session
MODE = 'final-experiment-signed-group-contrast-training'
MODULE = 'pipeline.weight_edit.group_contrast_experiment'


def loss_stop(rule, policy):
    stock = baseline.read(common_prefix_objective.checked(rule['stock_measurement']))
    reduction = rule['reduction_percent']
    value = stock['measured_contribution']
    if (stock['policy'] != policy or stock['generation'] != 0 or
            not math.isfinite(value) or value <= 0 or not 0 < reduction < 100):
        raise ValueError('The loss threshold needs the original group baseline and a finite percentage reduction')
    return dict(stock_measurement=rule['stock_measurement'], reduction_percent=reduction,
                metric='native.group.measured_contribution', stock_loss=value,
                threshold=value*(100-reduction)/100)


def configuration(declaration, policy):
    spec, inputs = objective.specification(policy)
    [primary] = declaration['functional_loop']['training_members']
    cases = {m['id']: baseline.read(common_prefix_objective.checked(m['case']))[0] for m in spec['members']}
    source = baseline.read(baseline.ROOT/declaration['start_fresh']['record'])
    capture = baseline.read(baseline.ROOT/source['capture']['record'])
    [fitted] = baseline.read(common_prefix_objective.checked(capture['cases']))
    if primary not in cases or cases[primary]['tokens'] != fitted['tokens']:
        raise ValueError('The cohort’s primary acquisition must reproduce the admitted origin prompt')
    ordinary = baseline.read(baseline.ROOT/declaration['storage']['sizing']['record'])['capture_reserve_bytes']
    individual = ordinary + declaration['native_key_anchor']['additional_capture_allowance_bytes']
    context = declaration['native_context']
    events = {member:len(case['tokens'])+case['max_tokens'] for member, case in cases.items()}
    if any(type(case['max_tokens']) is not int or case['max_tokens'] < 1 or events[member] > context
           for member, case in cases.items()):
        raise ValueError('Each cohort case needs a finite cap within the qualified native context')
    per_event = (individual+context-1)//context
    sources = [baseline.binding(Path(m.__file__)) for m in (driver, objective, common_prefix_driver, common_prefix_objective)]
    if declaration.get('disk_working_state'):
        sources.extend(baseline.binding(Path(m.__file__)) for m in (group_disk_state, host_pages))
    sources.extend(entry['program'] for entry in spec.get('operation_matchers', {}).values())
    return dict(policy=policy, primary=primary, members=[m['id'] for m in spec['members']], updates=spec['updates'],
        driver=sources[0], sources=[baseline.binding(Path(__file__)), *sources, spec['registrar']], inputs=inputs,
        individual_capture_reserve_bytes=individual, maximum_events_per_member=events,
        cohort_capture_reserve_bytes=per_event*sum(events.values()))


def prepare(run, prior, review, intent, policy, development_origin=None):
    declaration = native.prepare(run, prior, review, intent, development_origin=development_origin)
    if intent.get('disk_working_state'):
        declaration['disk_working_state'] = dict(plot_backend='Agg')  # completed below, once the cohort is known
    config = configuration(declaration, baseline.binding(policy))
    if intent.get('disk_working_state'):
        declare_disk(declaration)
    observation_led = config['updates'] == objective.OBSERVED_STOP
    declaration.update(operation=MODE, group_contrast=config,
        description='Signed cohort contrast with fresh failing and preservation operations.',
        stopping=('Run until the declared fitted outcome or research stop is observed. '
                  'Use the native graceful stop to persist a measured endpoint.' if observation_led else
                  'Bound measured cohort interval or earlier native terminal condition.'))
    if declaration.get('joint_stock_lineage'):
        baseline.verify_stock_group_policy(declaration)
    declaration.pop('group_loss_stop', None)
    if 'group_loss_stop' in intent:
        if not observation_led:
            raise ValueError('A loss-only endpoint requires a policy without a fixed update count')
        stop = loss_stop(intent['group_loss_stop'], config['policy'])
        declaration['group_loss_stop'] = stop
        declaration['stopping'] = 'Checkpoint at the first freshly measured group training loss at or below the declared threshold.'
        declaration['inputs'] = sorted(set(declaration['inputs']) | {stop['stock_measurement']['record']})
        declaration['parameters']['group_loss_threshold'] = dict(value=stop['threshold'],
            measured_from=stop['stock_measurement']['record'],
            derivation=f"Original measured group loss times (100 - {stop['reduction_percent']}) / 100.",
            scales_with='Original loss under this exact frozen objective.')
    for container, field in ((declaration, 'implementation'), (declaration['observability'], 'sources')):
        container[field] = list({r['record']:r for r in [*container[field], *config['sources']]}.values())
    obs = declaration['observability']
    obs['entrypoint'] = config['driver']['record']
    obs['context_fields'] = list(dict.fromkeys([*obs['context_fields'], 'group_objective_sha256']))
    obs['additional_metrics'] = [*obs.get('additional_metrics', []), 'native.group.objective', *(
        f'native.group.{role}.{key}' for role in objective.ROLES for key in
        ('objective','represented_cells','supported_cells','baseline_required_cells','unmeasured_operations'))]
    declaration['inputs'] = sorted(set(declaration['inputs']) | {row['record'] for row in config['inputs']})
    spec, _ = objective.specification(config['policy'])
    declaration['parameters']['group_updates'] = dict(value=config['updates'], measured_from=spec['update_basis']['record'],
        derivation=(declaration['stopping'] if observation_led else
                    'The declared cohort interval exercises an installed update and the next update with fresh preservation feedback.'),
        scales_with=('Regenerated fitted outcomes, protected operations and observed learning progress.' if observation_led else
                     'The predeclared native calibration interval for this cohort.'))
    support = ('all donor-qualified source cells' if objective.support_policy(spec) == objective.LIVE_SUPPORT
               else 'baseline-represented source cells')
    declaration['parameters']['group_cell_mass'] = dict(value=f'equal roles, members and operations, then fixed mass over {support}',
        measured_from=config['policy']['record'], derivation=f'Divide operation mass over {support} and each cell over its visits.',
        scales_with='Frozen operation inventory, declared support policy and current visits.')
    if objective.support_policy(spec) == objective.LIVE_SUPPORT:
        declaration['parameters']['group_cell_mass'].update(value='equal roles, members and operations, then fixed donor source cells',
            derivation='Divide each operation mass over all frozen donor-qualified cells. For each currently matched cell divide its mass over its current visits.',
            scales_with='Frozen donor-qualified operation geometry and current computational correspondence.')
        obs['additional_metrics'].append('native.group.measured_contribution')
    storage = declaration['storage']
    storage['capture_reserve_bytes'] = config['cohort_capture_reserve_bytes']
    app = baseline.controller(baseline.MEASURED)
    state = app.held.checked(baseline.ROOT, declaration['resume_checkpoint'])
    retained = app.continuation.remainder_inventory(baseline.ROOT, state, app.held.checked)
    _, specs = app.state_specs(app.held.checked(baseline.ROOT, declaration['tensors']),
                               app.held.checked(baseline.ROOT, declaration['architecture']))
    declaration['parameters']['storage']['value'] = app.storage_budget(
        specs, retained, sum(config['maximum_events_per_member'].values()), storage)
    declaration['parameters']['storage']['derivation'] += ' Reserve one complete cohort for each successor acquisition before archival.'
    declaration['commands'] = [shlex.join(['env', f"PYTHONPATH={baseline.ENV['PYTHONPATH']}", *baseline.RUNTIME,
                                           '-m', MODULE, 'run', str(run)])]
    return declaration


def working_limits(declaration):
    """The working files' largest sizes on disk."""
    tensors = baseline.read(baseline.ROOT / declaration['tensors']['record'])
    architecture = baseline.read(baseline.ROOT / declaration['architecture']['record'])
    model = (baseline.ROOT / declaration['original_model']['record']).stat().st_size
    return group_disk_state.file_limits(tensors, architecture, model)


def disk_requirements(declaration):
    state = baseline.read(baseline.ROOT / declaration['resume_checkpoint']['record'])
    manifest = baseline.read(common_prefix_objective.checked(state['checkpoint']))
    limits = working_limits(declaration)
    return dict(model_bytes=limits['model'],
                dense_remainder_bytes=sum(r['remainder']['bytes'] for r in manifest['layout'] if r['remainder']['kind'] == 'dense'),
                working_files=limits, working_files_bytes=group_disk_state.total(limits), plot_backend='Agg')


def declare_disk(declaration):
    """Bind the working files and their disk requirement into a declaration."""
    disk = disk_requirements(declaration)
    declaration['disk_working_state'] = disk
    declaration['parameters']['working_storage_bytes'] = dict(
        value=disk['working_files_bytes'],
        measured_from=declaration['tensors']['record'],
        derivation=('From the stored tensor shapes, with the architecture record and the native model file this '
                    'declaration binds. '
                    'Sum of the working files at their largest, every array in whole pages: the native model file; '
                    'FP32 of every tensor the forward keeps on the host; and FP32 of every eligible dense remainder. '
                    'The direction and the displacement are in RAM and a fresh native trace has no file of its own: '
                    "each merge is read from its capture's stream."),
        scales_with='Stored tensor shapes and the size of the native model file.')
    return disk


def check(run, declaration):
    if declaration['operation'] != MODE:
        raise ValueError('The run does not bind a signed cohort contrast')
    config = configuration(declaration, declaration['group_contrast']['policy'])
    if declaration['group_contrast'] != config:
        raise ValueError('Changed cohort, source or registrar binding')
    if 'group_loss_stop' in declaration:
        stop = declaration['group_loss_stop']
        if stop != loss_stop(stop, config['policy']):
            raise ValueError('The declared group loss threshold changed')
    storage = declaration['storage']
    cap = declaration['parameters']['storage']['value']
    cohort = config['cohort_capture_reserve_bytes']
    individual = config['individual_capture_reserve_bytes']
    if cap['capture_reserve_bytes'] != cohort or storage['capture_reserve_bytes'] != cohort:
        raise ValueError('Native capture reserve does not include the full cohort')
    native_cap = dict(cap, capture_reserve_bytes=individual,
                      free_required_bytes=cap['free_required_bytes']-cohort+individual)
    base = dict(declaration, operation=native.MODE, storage=dict(storage, capture_reserve_bytes=individual),
                parameters=dict(declaration['parameters'], storage=dict(declaration['parameters']['storage'], value=native_cap)))
    preflight = native.check(run, base)
    spec, _ = objective.specification(config['policy'])
    objective.Objective(spec)
    available = shutil.disk_usage(run if run.exists() else run.parent).free
    required = cap['free_required_bytes'] + cohort
    if declaration.get('disk_working_state'):
        disk = disk_requirements(declaration)
        if disk != declaration['disk_working_state']:
            raise ValueError('Disk backing differs from the native seed, the saved remainders or the cohort')
        required += disk['working_files_bytes']
    return dict(preflight, group_contrast=config, storage_available_bytes=available, storage_needed_bytes=required,
                storage_ready=available >= required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('run', type=Path)
    prep.add_argument('--from-run', type=Path, required=True)
    prep.add_argument('--review', type=Path, required=True)
    prep.add_argument('--intent', type=Path, required=True)
    prep.add_argument('--group-objective', type=Path, required=True)
    prep.add_argument('--development-origin', type=Path)
    prep.add_argument('--write', action='store_true')
    verify = commands.add_parser('check')
    verify.add_argument('run', type=Path)
    verify.add_argument('--declaration', type=Path)
    commands.add_parser('run').add_argument('run', type=Path)
    args = parser.parse_args()
    run = baseline.run_path(args.run)
    if args.action == 'prepare':
        dec = prepare(run, baseline.run_path(args.from_run), args.review, baseline.read(args.intent),
                      args.group_objective, args.development_origin)
        if not args.write:
            print(json.dumps(dec, indent=2, allow_nan=False))
            return
        check(run, dec)
        encoded = json.dumps(dec, indent=2, allow_nan=False)+'\n'
        run.mkdir()
        with (run/'declaration.json').open('x') as handle:
            handle.write(encoded)
        print(run/'declaration.json')
        return
    declaration = baseline.read(getattr(args, 'declaration', None) or run/'declaration.json')
    summary = check(run, declaration)
    if args.action == 'check':
        print(json.dumps(summary, indent=2))
        return
    if any((run/file).exists() for file in ('working-state.json','progress.jsonl','loop','launcher-exit.json','observability','owner.lock')):
        parser.error('A used cohort run requires a new continuation output.')
    if not summary['storage_ready']:
        parser.error(f"Cohort continuation needs {summary['storage_needed_bytes']} bytes; {summary['storage_available_bytes']} available.")
    os.chdir(baseline.ROOT)
    entry = baseline.ROOT/declaration['group_contrast']['driver']['record']
    env = {key: value for key, value in os.environ.items() if key not in DISPLAY_VARIABLES} | baseline.ENV
    env['MPLBACKEND'] = 'Agg'
    os.execve(sys.executable, [sys.executable, str(entry), str(run)], env)


if __name__ == '__main__':
    main()
