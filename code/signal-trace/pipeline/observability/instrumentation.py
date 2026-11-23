"""Explicit, reversible wrappers for future computational research processes.

The original functions, arguments, callback order, return objects and exceptions
are retained. Expensive repeated operations get independent, correlated traces.
"""
from contextlib import contextmanager
from functools import wraps
import importlib
import os
import time


# module, attribute, span name, independent trace. No per-tensor/operator hooks.
TARGETS = (
    ('vector_loop', 'Backend.__init__', 'run.bind', True),
    ('vector_loop', 'Backend._model', 'model.load', False),
    ('vector_loop', 'Backend._state', 'state.verify', False),
    ('vector_loop', 'Backend.observe', 'native.observe', True),
    ('vector_loop', 'Backend._acquire', 'native.acquire', False),
    ('vector_loop', 'Backend.correct', 'correction.step', True),
    ('vector_loop', 'Backend.post_closure', 'closure.record', True),
    ('vector_loop', 'Backend.finish', 'run.finish', True),
    ('vector_loop', 'compact_state', 'state.compact', False),
    ('gap_state', 'Working.make', 'state.copy', True),
    ('gap_state', 'Working.taken_up', 'state.take_up', True),
    ('vector_recovery', 'Backend.observe', 'native.observe_recovery', True),
    ('vector_recovery', 'Backend.__init__', 'run.bind_recovery', True),
    ('vector_device_loop', 'Backend._acquire', 'native.acquire', False),
    ('vector_gradient_loop', 'Backend.correct', 'correction.step', True),
    ('vector_working_loop', 'Backend.__init__', 'run.bind_working', True),
    ('vector_working_loop', 'Backend._state', 'state.verify', False),
    ('vector_working_loop', 'Backend._model', 'model.resident', False),
    ('vector_working_loop', 'Backend._acquire', 'native.acquire', False),
    ('vector_working_loop', 'Backend.observe', 'native.observe', True),
    ('vector_working_loop', 'Backend.install', 'state.install', True),
    ('working_model', 'Working.create', 'state.create_working', True),
    ('working_model', 'Working.install', 'state.install_working', False),
    ('trace_gradient', 'propose', 'proposal.gradient_step', True),
    ('trace_gradient', 'weight_gradient', 'proposal.weight_derivative', True),
    ('model_trace_response', 'measure_model', 'proposal.directional_response', True),
    ('functional_reading', 'Reading.read', 'native.read', False),
    ('turn_artifact', 'load', 'reference.load', False),
    ('turn_reference', 'select', 'reference.select', True),
    ('vector_chance_reference', 'compile', 'reference.calibrate', True),
    ('vector_chance_run', 'compile_original', 'reference.compile', True),
    ('turn_reference_run', 'compile_original', 'reference.compile_turns', True),
    ('turn_reference', 'coverage', 'reference.coverage', True),
    ('turn_pair', 'turns', 'computation.turns', False),
    ('vector_chance', 'turns', 'computation.turns', False),
    ('turn_correction', 'correspondence', 'computation.correspondence', False),
    ('turn_correction', 'measurement', 'computation.measure', False),
    ('turn_correction', 'propose', 'proposal.construct', True),
    ('joint_functional_step', 'inverse', 'solver.joint_inverse', True),
    ('joint_functional_step', 'response', 'solver.direction_response', True),
    ('joint_inverse', 'solve', 'solver.solve', True),
    ('joint_inverse', 'Linearization.__init__', 'solver.linearize', True),
    ('joint_inverse', 'Linearization.push', 'solver.jvp', True),
    ('joint_inverse', 'Linearization.pull', 'solver.vjp', True),
    ('joint_inverse_device', 'solve', 'solver.solve', True),
    ('joint_inverse_device', 'Linearization.__init__', 'solver.linearize', True),
    ('joint_inverse_device', 'Linearization.push', 'solver.jvp', True),
    ('joint_inverse_device', 'Linearization.pull', 'solver.vjp', True),
    ('native_graph', 'Graph.forward', 'model.forward', False),
    ('bf16_direction', 'search', 'proposal.search', True),
    ('bf16_direction', 'rounded', 'proposal.round_bf16', False),
    ('bf16_direction', 'first_change', 'proposal.first_representable_change', False),
    ('bf16_direction', 'previous_change', 'proposal.previous_representable_change', False),
    ('functional_install', 'install', 'state.install', True),
    ('functional_install', 'load', 'state.verify_installation', False),
    ('state_tensors', 'keep', 'state.archive', False),
    ('state_tensors', 'made_again', 'state.verify_rebuild', False),
    ('state_tensors', 'restore', 'state.restore', True),
    ('pair_loop', 'iterate', 'training.loop', True),
)


def _observed(recorder, call):
    try:
        return call()
    except Exception as error:
        recorder.warning(error)
        return None


def model_attributes(record):
    """Explicit scalar fields; a run/update label never becomes a file SHA."""
    result = {}
    identity = record.get('model_identity')
    if identity is not None:
        from pipeline.weight_edit.working_identity import from_record
        identity = from_record(record)
        result.update({'model_identity.kind': identity['kind'],
                       'model_identity.run': identity['run'],
                       'model_identity.original_sha256': identity['original_sha256']})
    elif isinstance(record.get('model_sha256'), str):
        result['model_sha256'] = record['model_sha256']
    generation = record.get('generation', identity.get('generation') if isinstance(identity, dict) else None)
    if type(generation) is int:
        result['model_generation'] = generation
    return result


def wrapped(function, recorder, name, *, detached=False):
    @wraps(function)
    def invoke(*args, **kwargs):
        context = {}
        if name in ('native.observe', 'native.observe_recovery', 'correction.step', 'state.install'):
            number = args[1] if len(args) > 1 else kwargs.get('number')
            if type(number) is int:
                context['training_step'] = number
        if name == 'correction.step':
            observation = args[2] if len(args) > 2 else kwargs.get('observation')
            if isinstance(observation, dict):
                context.update(_observed(recorder, lambda: model_attributes(observation)) or {})
                for field in ('reference_sha256',):
                    if isinstance(observation.get(field), str):
                        context[field] = observation[field]
        elif name == 'state.install_working':
            step = args[3] if len(args) > 3 else kwargs.get('step')
            if type(step) is int:
                context['training_step'] = step
        if name == 'solver.solve':
            context.update(solver_accepted_iteration=0, solver_product_index=0)
        elif name in ('solver.jvp', 'solver.vjp'):
            # Count observed products, not presumed iterations: rejected/restarted
            # directions can perform more products without an accepted callback.
            previous_product = recorder.context_attributes.get('solver_product_index')
            if type(previous_product) is int:
                recorder.annotate_context(solver_product_index=previous_product + 1)
        with recorder.context(**context), recorder.span(name, detached=detached) as outcome:
            call_kwargs = kwargs
            if name == 'solver.solve':
                old_progress = kwargs.get('progress')
                step = recorder.context_attributes.get('training_step', 0)
                previous = _observed(recorder, time.perf_counter)

                def progress(iteration, objective):
                    nonlocal previous
                    if old_progress is not None:
                        old_progress(iteration, objective)
                    recorder.annotate_context(solver_accepted_iteration=iteration)
                    now = _observed(recorder, time.perf_counter)
                    recorder.metric(f'solver.step_{step}.linear_objective', objective,
                                    step=iteration, attributes={'measurement_kind': 'linear_surrogate'})
                    if now is not None and previous is not None:
                        recorder.metric(f'ops.step_{step}.solver_callback_interval_seconds',
                                        now - previous, step=iteration)
                    previous = now
                call_kwargs = kwargs | {'progress': progress}
            elif name == 'proposal.search':
                old_evaluate = kwargs['evaluate']
                evaluation_index = -1

                def evaluate(*inner_args, **inner_kwargs):
                    nonlocal evaluation_index
                    evaluation_index += 1
                    # bf16_direction.search evaluates the unchanged baseline once,
                    # then evaluates each distinct candidate in recorded order.
                    with recorder.context(proposal_evaluation_index=evaluation_index,
                            proposal_evaluation_kind='baseline' if evaluation_index == 0 else 'candidate'), \
                            recorder.span('proposal.evaluate', detached=True) as details:
                        value = old_evaluate(*inner_args, **inner_kwargs)
                        _observed(recorder, lambda: details.update(
                            measurement_kind='held_history_surrogate', estimate=value.estimate,
                            lower=value.lower, upper=value.upper))
                        return value
                call_kwargs = kwargs | {'evaluate': evaluate}
            result = function(*args, **call_kwargs)
            _observed(recorder, lambda: summarize(recorder, name, result, outcome))
            return result
    return invoke


def summarize(recorder, name, result, outcome):
    """Explicit projections only; never stringify or serialize result objects."""
    if name in ('native.observe', 'native.observe_recovery') and isinstance(result, dict):
        outcome.update(model_attributes(result))
        outcome.update({key: result[key] for key in
                        ('complete', 'ended', 'reference_sha256', 'measure')
                        if key in result})
        outcome['acquisition_kind'] = ('retained_remeasured' if 'recovered_from' in result
                                       else 'fresh_native')
        for field in ('objective', 'outside'):
            if field in result:
                outcome['native.' + field] = result[field]
        for field, label in (('pair_merge_cells', 'pair_merge_cells'),
                             ('merges', 'trace_merges'), ('events', 'trace_events')):
            if field in result:
                outcome['ops.' + label] = result[field]
    elif name == 'solver.solve':
        outcome.update(status=result.status, iterations=result.iterations,
                       solver_accepted_iteration=result.iterations,
                       solver_product_index=recorder.context_attributes.get('solver_product_index', 0),
                       measurement_kind='linear_surrogate')
    elif name == 'proposal.search':
        outcome.update(status=result.status, measurement_kind='held_history_surrogate')
        outcome['proposal.held_history_objective'] = result.value.estimate
        if result.length is not None:
            outcome['proposal.length'] = result.length
        outcome['proposal.trials'] = len(result.trials)
    elif name == 'proposal.weight_derivative':
        outcome.update(measurement_kind='complete_computational_loss_derivative',
                       **{'proposal.gradient_norm_squared': result.norm_squared,
                          'proposal.origin_objective': result.before['objective']})
    elif name == 'proposal.directional_response':
        outcome.update(status=result.status, measurement_kind='local_linear_prediction',
                       **{'proposal.response_slope': result.slope,
                          'proposal.response_norm_squared': result.curvature,
                          'proposal.linear_prediction': result.predicted_objective})
        if result.length is not None:
            outcome['proposal.response_scale'] = result.length
    elif name == 'proposal.gradient_step':
        outcome.update(status=result.status, measurement_kind='once_rounded_weight_proposal',
                       **{'proposal.gradient_norm_squared': result.gradient_norm_squared,
                          'proposal.changed_values': result.changed_values})
        for key, value in result.timings.items():
            outcome['ops.' + key] = value
    elif name in ('state.install', 'state.verify_installation', 'state.install_working') and isinstance(result, dict):
        outcome.update(model_attributes(result))
        outcome.update({key: result[key] for key in ('status', 'format') if key in result})
    elif name == 'state.create_working':
        outcome['storage.policy'] = 'one current working model; historical metadata only'
    elif name == 'reference.select':
        outcome['reference'] = result.reference
    elif name == 'training.loop' and isinstance(result, dict):
        outcome.update({key: result[key] for key in ('status', 'closure', 'updates', 'repair_claim')
                        if key in result})


@contextmanager
def instrument(recorder, targets=TARGETS):
    restored = []
    try:
        for module_name, path, name, detached in targets:
            try:
                owner = importlib.import_module('pipeline.weight_edit.' + module_name)
                parts = path.split('.')
                for part in parts[:-1]:
                    owner = getattr(owner, part)
                key = parts[-1]
                # Restore inherited attributes by deletion, not by leaving a copied override.
                owned = key in vars(owner)
                original = vars(owner)[key] if owned else getattr(owner, key)
                if isinstance(original, (classmethod, staticmethod)):
                    replacement = type(original)(wrapped(original.__func__, recorder, name, detached=detached))
                else:
                    replacement = wrapped(original, recorder, name, detached=detached)
                setattr(owner, key, replacement)
                restored.append((owner, key, original, owned))
            except Exception as error:
                recorder.warning(error)
                recorder.metric('ops.instrumentation_setup_error', 1, attributes={'operation': name})
        # Capture command boundaries, but never full argv, environment or captured output.
        import subprocess
        original_run = subprocess.run

        @wraps(original_run)
        def command(*args, **kwargs):
            supplied = args[0] if args else kwargs.get('args')
            executable = 'external-command'
            if type(supplied) in (list, tuple) and supplied and type(supplied[0]) in (str, bytes):
                executable = os.path.basename(supplied[0])
                if isinstance(executable, bytes):
                    executable = executable.decode(errors='replace')
            with recorder.span('process.execute', attributes={'executable': executable}) as outcome:
                result = original_run(*args, **kwargs)
                _observed(recorder, lambda: outcome.update(returncode=result.returncode))
                return result
        subprocess.run = command
        restored.append((subprocess, 'run', original_run, True))
        yield
    finally:
        for owner, key, original, owned in reversed(restored):
            try:
                if owned:
                    setattr(owner, key, original)
                else:
                    delattr(owner, key)
            except Exception as error:
                recorder.warning(error)
