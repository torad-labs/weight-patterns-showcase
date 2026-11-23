"""Connect the common-prefix residual to the installed one-pair learner."""

from contextlib import ExitStack, contextmanager
from functools import wraps
import inspect
import math
from pathlib import Path
import signal
import sys
import time
from unittest.mock import patch

import torch

from pipeline.overlap import complete_overlap
from pipeline.weight_edit import common_prefix_objective as prefix, final_experiment as baseline, full_graph


def configuration(policy):
    spec = prefix.specification(policy)
    return dict(policy=policy, driver=baseline.binding(Path(__file__)),
                objective=baseline.binding(Path(prefix.__file__)),
                alignment=baseline.binding(Path(complete_overlap.__file__)),
                registrar=spec['registrar'], bindings=[policy, spec['arrays'], *spec.get('inputs', [])])


def configure(declaration, path):
    config = configuration(baseline.binding(path))
    declaration['prefix_objective'] = config
    sources = [config[key] for key in ('driver', 'objective', 'alignment', 'registrar')]
    for container, field in ((declaration, 'implementation'), (declaration['observability'], 'sources')):
        container[field] = list({r['record']: r for r in [*container[field], *sources]}.values())
    declaration['inputs'] = sorted(set(declaration['inputs']) | {r['record'] for r in config['bindings']})
    obs = declaration['observability']
    obs['entrypoint'] = config['driver']['record']
    obs['additional_metrics'] = list(dict.fromkeys([*obs.get('additional_metrics', []), *prefix.METRICS.values(),
                                                   prefix.COMBINED_METRIC]))
    obs['context_fields'] = list(dict.fromkeys([*obs.get('context_fields', []), 'prefix_objective_sha256']))
    declaration['description'] = 'Installed one-pair learner with a frozen common-prefix turn objective.'
    declaration['parameters']['prefix_cell_mass'] = dict(value='1 / (supported source cells × visits to that source event)',
        measured_from=config['policy']['record'],
        derivation='Average squared signed shortfall over the frozen positive source cells, dividing each cell mass across its visits.',
        scales_with='The number of source event–merge cells with mutual donor support and their current matched visits.')
    return config


def evaluation_boundary(calibration, loss_identity):
    from pipeline.weight_edit import validation_stopping

    history = baseline.read(prefix.checked(calibration))['fit_history']
    measurements = [baseline.read(prefix.checked(row)) for row in history]
    if any(not row['complete'] or not row['ended'] or row['loss_identity'] != loss_identity
           or row['step'] != i for i, row in enumerate(measurements)):
        raise ValueError('Evaluation calibration needs contiguous complete fresh measurements of the ordinary objective')
    every = validation_stopping.recovered_delay([row['objective'] for row in measurements])
    return dict(calibration=calibration, completed_updates=every, measurements=history,
                derivation_source=baseline.binding(Path(validation_stopping.__file__)))


def archived_exit_kind(stopped, state, observation):
    if (stopped['exit_code'] != 1 or stopped.get('state_status') != 'complete'
            or state['status'] != 'complete' or not state.get('checkpoint')
            or not stopped.get('reason', '').startswith('RuntimeError: archive readback failed; no local stream released:')
            or stopped['reason'] != state['reason'] or stopped['last_measurement'] != state['last_measurement']
            or stopped['installed_updates'] != state['iteration'] or observation['step'] != state['iteration']
            or observation['model_identity'] != state['model_identity']
            or not observation['complete'] or not observation['ended'] or 'prefix_objective' not in observation):
        raise ValueError('Archive readback recovery requires the interrupted run’s own complete measured prefix checkpoint')
    return 'completed_fresh_checkpoint_then_archive_readback_failure'


def continued_boundary(declaration, base):
    source_state = declaration['resume_checkpoint']
    source_measurement = declaration['start_fresh']
    source_exit = declaration['predecessor_exit']['record']
    predecessor = prefix.checked(source_state).parent
    local = baseline.binding(predecessor/'declaration.json')
    former = baseline.read(predecessor/'declaration.json')
    state = baseline.read(prefix.checked(source_state))
    observed = baseline.read(prefix.checked(source_measurement))
    stopped = baseline.read(prefix.checked(source_exit))
    prior = former['prefix_evaluation_boundary']
    if ({key: prior[key] for key in base} != base
            or former['loss_identity'] != declaration['loss_identity']
            or former['prefix_objective']['policy'] != declaration['prefix_objective']['policy']
            or state['last_measurement'] != source_measurement or stopped['last_measurement'] != source_measurement
            or prefix.checked(source_exit) != predecessor/'launcher-exit.json'
            or prefix.checked(source_measurement) != predecessor/'loop'/str(state['iteration'])/'measurement.json'
            or observed['model_identity']['run'] != str(predecessor.relative_to(baseline.ROOT))
            or state['iteration'] != stopped['installed_updates'] or state['iteration'] != observed['step']
            or state['status'] != 'complete' or stopped['state_status'] != 'complete'
            or observed['model_identity'] != state['model_identity'] or not observed['complete'] or not observed['ended']
            or 'prefix_objective' not in observed):
        raise ValueError('Continuation changed the frozen evaluation interval, objective or measured checkpoint')
    if stopped['exit_code'] != 0:
        if archived_exit_kind(stopped, state, observed) != declaration['predecessor_exit']['kind']:
            raise ValueError('Continuation does not bind the recorded archive readback exit')
    origin = prior.get('continuation', {}).get('origin_declaration', local)
    original = baseline.read(prefix.checked(origin))
    historical_origin = original['counter_semantics']['historical_predecessor']
    start = declaration['counter_semantics']['historical_predecessor']
    elapsed = start - historical_origin
    carried = prior.get('continuation', {}).get('completed_updates_before', 0)
    if (elapsed != carried + state['iteration'] or not 0 < elapsed < base['completed_updates']
            or original['prefix_evaluation_boundary'] != base
            or original['prefix_objective']['policy'] != declaration['prefix_objective']['policy']):
        raise ValueError('Continuation must account for completed updates inside the original frozen interval')
    return dict(base, continuation=dict(origin_declaration=origin, predecessor_declaration=local,
        source_checkpoint=source_state, source_exit=source_exit, source_measurement=source_measurement,
        completed_updates_before=elapsed, remaining_updates=base['completed_updates']-elapsed,
        historical_start=historical_origin, historical_endpoint=historical_origin+base['completed_updates']))


def checked_evaluation(declaration, boundary):
    base = evaluation_boundary(boundary['calibration'], declaration['loss_identity'])
    if 'continuation' in boundary:
        return continued_boundary(declaration, base)
    return base


def configure_evaluation(declaration, path, *, continue_interval=False):
    config = evaluation_boundary(baseline.binding(path), declaration['loss_identity'])
    if continue_interval:
        config = continued_boundary(declaration, config)
        carried = config['continuation']
        declaration['inputs'] = sorted(set(declaration['inputs']) |
            {carried[key]['record'] for key in ('origin_declaration', 'predecessor_declaration')})
        declaration['parameters']['evaluation_interval_remaining'] = dict(value=carried['remaining_updates'],
            measured_from=carried['source_exit']['record'],
            derivation='Original measured interval minus all updates already installed and freshly measured within it.',
            scales_with='The original evaluation interval and the completed update count across its continuations.')
    declaration['prefix_evaluation_boundary'] = config
    source = config['derivation_source']
    for container, field in ((declaration, 'implementation'), (declaration['observability'], 'sources')):
        container[field] = list({r['record']: r for r in [*container[field], source]}.values())
    declaration['inputs'] = sorted(set(declaration['inputs']) |
        {r['record'] for r in [config['calibration'], *config['measurements']]})
    declaration['parameters']['evaluation_interval'] = dict(value=config['completed_updates'],
        measured_from=config['calibration']['record'],
        derivation='One more update than the longest dry spell ending at a new low in this measured fitted trajectory.',
        scales_with='The recovered drawdowns of the bound historical ordinary objective.')
    declaration['stopping'] = 'Exit after the measured evaluation interval or an earlier native terminal condition.'


def weight_gradient(graph, history, reference, pairs, registered, measured):
    started = time.monotonic()

    def objective(forward):
        ordinary, outside = [], 0
        for observed in reference.measure(forward, pairs):
            ordinary.append(observed.residual.double().square().sum())
            outside += int(observed.outside.sum())
        if len(ordinary) != reference.merges or not ordinary:
            raise ValueError('The full objective must account for every declared ordinary merge')
        additional = [residual.square().sum() for _, residual in registered.residuals(forward)]
        original_value = math.fsum(float(term.detach()) for term in ordinary)
        prefix_value = math.fsum(float(term.detach()) for term in additional)
        if prefix_value != measured['prefix_objective']['objective']:
            raise ValueError('The prefix derivative origin differs from the fresh native measurement')
        scalar = torch.stack(ordinary + additional).sum()
        return scalar, dict(objective=original_value + prefix_value, ordinary_objective=original_value,
            prefix_objective=prefix_value, outside=outside, pair_merge_cells=len(pairs)*reference.merges)

    direction, before, norm, diagnostics = full_graph._derivative(graph, history, objective)
    return full_graph.Gradient(direction, before, norm, time.monotonic()-started, diagnostics)


def moments(merge, residual):
    primal, tangent = torch.autograd.forward_ad.unpack_dual(residual)
    r = primal.detach().double()
    q = None if tangent is None else tangent.detach().double()
    return dict(merge=merge, baseline=float(r.square().sum()),
                loss_cross=0. if q is None else 2 * float((r*q).sum()),
                response_norm_squared=0. if q is None else float(q.square().sum()))


def predicted_group(rows, coefficient):
    model = {name: math.fsum(row[name] for row in rows)
             for name in ('baseline', 'loss_cross', 'response_norm_squared')}
    return dict(model, prediction=(model['baseline'] + coefficient * model['loss_cross']
                                    + coefficient**2 * model['response_norm_squared'])
                if coefficient is not None else None)


class Training:
    def __init__(self, app, run, config, objective, boundary=None):
        self.app, self.run, self.config, self.objective = app, run, config, objective
        self.boundary = boundary
        self.current = None
        self.active = None
        self.generation = None

    @contextmanager
    def connected(self):
        app = self.app
        acquire, propose, residuals = app.acquire, app.propose, app.residuals

        @wraps(acquire)
        def fresh(*args, **kwargs):
            invocation = inspect.signature(acquire).bind(*args, **kwargs)
            bound = invocation.arguments
            if bound['run'] != self.run:
                return acquire(*args, **kwargs)
            resumed = (self.boundary or {}).get('continuation')
            if resumed and bound['number'] == 0:
                bound['previous'] = resumed['source_measurement']
            measurement = app.held.turn_correction.measurement
            registered = None

            @wraps(measurement)
            def measure(reference, native, pairs):
                nonlocal registered
                if registered is not None:
                    raise ValueError('A fresh acquisition must bind one prefix registration')
                ordinary = measurement(reference, native, pairs)
                with torch.no_grad():
                    registered = self.objective.register(native, prompt_events=len(bound['case']['tokens']))
                    observed = registered.measure(native)
                result = dict(ordinary, prefix_objective=observed,
                              combined_objective=ordinary['objective'] + observed['objective'])
                if resumed:
                    local = bound['number']
                    interval = resumed['completed_updates_before'] + local
                    result['evaluation_position'] = dict(local_generation=local, completed_updates=interval,
                        historical_step=resumed['historical_start']+interval)
                return result

            with patch.object(app.held.turn_correction, 'measurement', measure):
                result = acquire(*invocation.args, **invocation.kwargs)
            if registered is None:
                raise ValueError('Fresh acquisition omitted the declared prefix measurement')
            self.current = result[3], registered, result[5], result[-2]
            step = self.generation = bound['number']
            path = self.run/'loop'/str(step)/'prefix-correspondence.json'
            app.save(path, dict(measurement=result[5], complete=result[-2], policy=self.config['policy'],
                pairs=[dict(source_event=self.objective.first_source_event + source, query_event=event)
                       for source, event in registered.pairs]))
            recorder = bound.get('recorder')
            if recorder is not None:
                for name, metric in prefix.METRICS.items():
                    recorder.metric(metric, result[3]['prefix_objective'][name], step=step)
                recorder.metric(prefix.COMBINED_METRIC, result[3]['combined_objective'], step=step)
            return result

        def gradient(graph, history, reference, *, pairs):
            measured, registered, _, _ = self.current
            return weight_gradient(graph, history, reference, pairs, registered, measured)

        @wraps(residuals)
        def requested(reference, pairs, ordinary_moments, selected_moments=None):
            ordinary = residuals(reference, pairs, ordinary_moments, selected_moments)
            if self.active is None:
                return ordinary
            phase = self.active['recorder'].context_attributes['response_kind']
            extra_moments = self.active['responses'][phase] = []
            _, registered, _, _ = self.current

            def combined(forward):
                yield from ordinary(forward)
                for merge, residual in registered.residuals(forward):
                    extra_moments.append(moments(merge, residual))
                    yield residual

            return combined

        @wraps(propose)
        def proposal(*args, **kwargs):
            bound = inspect.signature(propose).bind(*args, **kwargs).arguments
            if self.current is None:
                raise ValueError('A prefix update requires a fresh native acquisition')
            measured, _, source, complete = self.current
            if bound['measured'] is not measured or not complete:
                raise ValueError('A prefix update needs its own complete immediately preceding native acquisition')
            self.active = dict(recorder=bound['recorder'], responses={})
            try:
                result, direction = propose(*args, **kwargs)
                groups = {}
                for phase, extra in self.active['responses'].items():
                    continuous = phase == 'continuous_direction'
                    coefficient = result.get('applied_length') if continuous else 1.
                    ordinary = result['continuous_moments' if continuous else 'installed_moments']
                    groups[phase] = dict(ordinary=predicted_group(ordinary, coefficient),
                                         prefix=predicted_group(extra, coefficient), prefix_per_merge=extra)
                return dict(result, gradient_scalar_reduction_difference=(result['before']['objective']
                    - measured['combined_objective']), prefix_objective=dict(policy=self.config['policy'],
                    source_measurement=source, source_coverage=measured['prefix_objective']['coverage'],
                    response_groups=groups)), direction
            finally:
                self.active = None

        with (patch.object(app, 'acquire', fresh), patch.object(app, 'propose', proposal),
              patch.object(app, 'residuals', requested), patch.object(app.full_graph, 'weight_gradient', gradient)):
            yield

    @contextmanager
    def evaluation(self, config):
        app = self.app
        resumed = config.get('continuation')
        requested = resumed['remaining_updates'] if resumed else config['completed_updates']
        execute = app.execute

        @wraps(execute)
        def bounded(run, declaration, recorder, capture, completion):
            archive = app.archive_source.archive_previous
            reached = None

            @wraps(archive)
            def archived(run, previous_step, current_step, declaration, *, stop_requested=None):
                nonlocal reached
                receipt = archive(run, previous_step, current_step, declaration, stop_requested=stop_requested)
                if run == self.run and current_step == requested:
                    if self.generation != current_step or self.current is None or not self.current[-1]:
                        raise ValueError('Evaluation boundary requires its complete fresh native acquisition')
                    reached = self.current[2]
                    signal.raise_signal(signal.SIGUSR1)
                return receipt

            with patch.object(app.archive_source, 'archive_previous', archived):
                result = execute(run, declaration, recorder, capture, completion)
            if reached is not None:
                state = app.binding(run/'final-state.json')
                exit_record = app.binding(run/'launcher-exit.json')
                final = baseline.read(run/'final-state.json')
                terminal = baseline.read(run/'launcher-exit.json')
                if (final['status'] != 'complete' or final['iteration'] != requested or terminal['exit_code'] != 0
                        or final['last_measurement'] != reached or terminal['last_measurement'] != reached):
                    raise ValueError('Evaluation boundary did not finish at its declared clean measured checkpoint')
                receipt = dict(completed_updates=config['completed_updates'], local_completed_updates=requested,
                    measurement=reached, final_state=state, launcher_exit=exit_record, calibration=config['calibration'])
                if resumed:
                    receipt.update(historical_step=resumed['historical_endpoint'], continuation=resumed)
                app.save(run/'evaluation-boundary.json', receipt)
            return result

        with patch.object(app, 'execute', bounded):
            yield


def execute(run):
    run = baseline.run_path(run)
    declaration = baseline.read(run/'declaration.json')
    native_key = declaration.get('native_key_anchor')
    if native_key:
        from pipeline.weight_edit import native_key_driver, native_key_experiment
        native_key_experiment.check(run, declaration)
    else:
        baseline.inspect_run(run, declaration)
    config = declaration['prefix_objective']
    objective = prefix.Objective(prefix.specification(config['policy']))
    app = baseline.controller(baseline.MEASURED)
    recorder = app.Recorder

    def recording(path, attributes):
        return recorder(path, dict(attributes, entrypoint=config['driver']['record'],
                                  prefix_objective_sha256=config['policy']['sha256']))

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, 'argv', [str(__file__), str(run)]))
        stack.enter_context(patch.object(app, 'Recorder', recording))
        if native_key:
            stack.enter_context(native_key_driver.connected(app, declaration))
        training = Training(app, run, config, objective, declaration.get('prefix_evaluation_boundary'))
        stack.enter_context(training.connected())
        if declaration.get('prefix_evaluation_boundary'):
            stack.enter_context(training.evaluation(declaration['prefix_evaluation_boundary']))
        if declaration.get('trace_diagnostic') or declaration.get('validation_stopping'):
            from pipeline.weight_edit import validation_driver
            stack.enter_context(patch.object(baseline, 'controller', return_value=app))
            validation_driver.main()
        else:
            if declaration.get('retired_input_admission'):
                from pipeline.weight_edit import retired_admission
                retired_admission.bind(app, declaration)
            app.main()


if __name__ == '__main__':
    execute(Path(sys.argv[1]))
