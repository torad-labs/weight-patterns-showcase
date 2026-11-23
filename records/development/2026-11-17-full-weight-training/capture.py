"""Small local operation records around existing calls; never evaluate tensors.

Wrappers retain call arguments, return objects, exceptions and order. Summaries
use explicit scalar fields and tensor metadata only; no input arrays, text,
argv, environments, tensor reductions or extra model/product calls.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import wraps
import json
import math
from pathlib import Path
import subprocess

FORMAT = 'native-continuation-capture/1'


def scalars(values, prefix=''):
    return {prefix + key: value for key, value in values.items()
            if isinstance(key, str) and isinstance(value, (str, int, float, bool))
            and (not isinstance(value, float) or math.isfinite(value))}


def metadata(value, name):
    """Shape/dtype/device inspection never reads an activation or weight value."""
    result = {}
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        result[name + '.shape'] = ','.join(str(int(n)) for n in value.shape)
        result[name + '.dtype'] = str(value.dtype)
        if hasattr(value, 'device'):
            result[name + '.device'] = str(value.device)
    return result


def selected(record, keys, prefix='output.'):
    return scalars({key: record[key] for key in keys if key in record}, prefix)


def result_summary(kind, result):
    if kind == 'proposal':
        record = result[0]
        values = selected(record, ('status', 'changed_values', 'remainder_changed_values',
            'applied_length', 'linear_prediction_at_applied_length', 'actual_bf16_prediction',
            'actual_displacement_squared', 'remainder_before_squared', 'remainder_after_squared',
            'captured_origin_checks', 'full_request_objective', 'gradient_norm_squared',
            'direction_matrices', 'request_merges', 'gradient_scope', 'direction_scope',
            'gradient_scalar_reduction_difference', 'response_scalar_reduction_difference',
            'actual_bf16_response_coefficient'))
        values.update(scalars(record.get('timings', {}), 'output.timings.'))
        return values
    if kind == 'weight_gradient':
        values = selected(result.before, ('objective', 'outside', 'pair_merge_cells'))
        values.update(scalars({'matrices': len(result.direction),
                              'gradient_norm_squared': result.norm_squared,
                              'seconds': result.seconds}, 'output.'))
        for index, (name, value) in enumerate(result.direction.items()):
            values[f'output.direction_{index}.matrix'] = name
            values.update(metadata(value, f'output.direction_{index}'))
        return values
    if kind == 'response':
        return scalars({name: getattr(result, name) for name in
            ('status', 'objective', 'slope', 'curvature', 'length', 'predicted_objective')}, 'output.')
    if kind == 'forward':
        return {'output.merges': len(result['merged']),
                'output.events': int(result['merged'][0].shape[0])}
    if kind == 'turns':
        return {'output.merges': len(result), 'output.events': int(result[0].shape[0]),
                'output.width': int(result[0].shape[1])}
    if kind == 'pairs':
        return metadata(result, 'output.pairs') | {'output.pairs': len(result)}
    if kind == 'rounded':
        return metadata(result[0], 'output.stored') | metadata(result[1], 'output.remainder')
    if kind == 'gradient':
        values = {'output.gradient_count': len(result)}
        for index, value in enumerate(result):
            values.update(metadata(value, f'output.gradient_{index}'))
            values[f'output.gradient_{index}.unused'] = value is None
        return values
    if kind == 'model':
        return {'output.layers': result.layers, 'output.width': result.width,
                'output.editable_matrices': len(result.leaves)}
    if kind == 'bytes':
        return {'output.bytes': len(result)}
    if kind == 'process':
        return {'output.returncode': result.returncode}
    if kind == 'publication':
        return {'output.published': result is not None}
    if kind == 'archive':
        return selected(result, ('phase', 'retained', 'previous_step', 'current_step')) | {
            'output.archives': len(result.get('archives', []))}
    if kind == 'verification':
        return {'output.problems': len(result)}
    if kind == 'release':
        receipt, problems = result
        return {'output.problems': len(problems), 'output.receipt_present': receipt is not None}
    if kind == 'upload':
        return {'output.failed': bool(result[1])}
    if kind == 'matrices':
        return {'output.matrices': len(result)}
    if isinstance(result, dict):
        return selected(result, ('record', 'sha256', 'status', 'format', 'objective', 'outside',
            'pair_merge_cells', 'sum', 'complete', 'ended', 'events', 'merges', 'iteration'))
    return {}


def input_summary(kind, args, kwargs):
    if kind == 'weight_gradient':
        graph, history, reference = args[:3]
        return {'input.events': len(history), 'input.merges': reference.merges,
                'input.pairs': len(kwargs['pairs']), 'input.gradient_targets': len(graph.leaves)}
    if kind == 'response':
        return {'input.matrices': len(args[1]), 'input.events': len(args[2]),
                'input.direction_blocks': len(args[3])}
    if kind == 'forward':
        return {'input.events': len(args[1])}  # bound Graph.forward: self, history
    if kind == 'anchor':
        return {'input.events': len(args[2]), 'input.merges': len(args[3]['merged'])}
    if kind == 'rounded':
        return (metadata(args[0], 'input.native') | metadata(args[1], 'input.remainder')
                | metadata(args[2], 'input.direction') | {'input.applied_length': args[3]})
    if kind == 'gradient':
        inputs = args[1] if len(args) > 1 else kwargs['inputs']
        return {'input.gradient_targets': len(inputs)}
    if kind == 'bytes':
        values = metadata(args[0], 'input.stored') | selected(args[1], ('bytes', 'type'), 'input.')
        name = args[1].get('name')
        if isinstance(name, str) and name:
            values.update({'matrix': name, 'input.matrix': name})
        return values
    if kind == 'process':
        command = args[0] if args else kwargs.get('args')
        return {'input.executable': Path(command[0]).name} if isinstance(command, (tuple, list)) and command else {}
    return {}


class Capture:
    def __init__(self, recorder):
        self.recorder = recorder
        self.calls, self.errors, self.steps = Counter(), Counter(), defaultdict(Counter)
        self.targets, self.setup_errors = [], []

    def observe(self, function):
        try:
            return function()
        except Exception as error:
            self.recorder.warning(error)
            self.recorder.metric('ops.capture_summary_error', 1)
            return {}

    def outcome(self, outcome, kind, result):
        outcome.update(self.observe(lambda: result_summary(kind, result)))

    def wrap(self, function, operation, kind='', *, detached=False):
        @wraps(function)
        def invoke(*args, **kwargs):
            self.calls[operation] += 1
            step = self.recorder.context_attributes.get('training_step')
            self.steps[str(step)][operation] += 1
            attributes = self.observe(lambda: input_summary(kind, args, kwargs))
            attributes['operation_call_index'] = self.steps[str(step)][operation]
            with self.recorder.span(operation, attributes=attributes, detached=detached) as outcome:
                try:
                    result = function(*args, **kwargs)
                except BaseException:
                    self.errors[operation] += 1
                    raise
                self.outcome(outcome, kind, result)
                return result
        return invoke

    @contextmanager
    def instrument(self, app):
        """Only process-local replacements; restore every hook on every exit."""
        targets = [
            (app['trace_gradient'], 'weight_gradient', 'proposal.weight_derivative', 'weight_gradient', False),
            (app['trace_gradient'].torch.autograd, 'grad', 'solver.vjp', 'gradient', True),
            (app['model_trace_response'], 'measure_model', 'solver.jvp', 'response', True),
            (app['native_graph'].Graph, '__init__', 'model.anchor', 'anchor', False),
            (app['native_graph'].Graph, 'forward', 'model.forward', 'forward', False),
            (app['held'].turn_correction, 'correspondence', 'computation.correspondence', 'pairs', False),
            (app['held'].turn_correction, 'measurement', 'computation.measure', '', False),
            (app['held'].turn_correction.turn_pair, 'turns', 'computation.turns', 'turns', False),
            (app['held'].turn_correction.vector_chance, 'turns', 'computation.turns', 'turns', False),
            (app['held'], 'raw_forward', 'native.read', '', False),
            (app['evaluation'], 'loss_map', 'computation.map', '', False),
            (app['factory'], 'capture', 'native.acquire', '', False),
            (app['factory'], 'memory_model', 'state.create_working', '', False),
            (app['factory'], 'apply_changes', 'state.restore', '', False),
            (app['factory'], 'load_changes', 'state.read_checkpoint', '', False),
            (app['held'], 'load_inputs', 'run.bind_inputs', '', False),
            (app['held'].turn_artifact, 'load', 'reference.load', '', False),
            (app['accumulation'], 'load', 'state.restore_remainder', 'matrices', False),
            (app['accumulation'], 'save', 'state.checkpoint_remainder', '', False),
            (app['accumulation'], 'step', 'proposal.round_bf16', 'rounded', False),
            (app['native_encoding'], 'encoded_resident', 'weights.encode', 'bytes', False),
            (app['observation_source'].Observation, 'write_candidate', 'state.prepare_sparse', '', False),
            (app['observation_source'].Observation, 'committed', 'state.commit', '', False),
            (app['archive'], 'published', 'storage.check_publication', 'publication', False),
            (app['archive'].archive, 'bucket', 'storage.transfer', 'upload', False),
            (app['archive'].archive, 'verify', 'storage.verify_remote', 'verification', False),
            (app['archive'].archive, 'release', 'storage.release', 'release', False),
            (app['reclaim'], 'free', 'storage.release_handles', '', False),
            (app, 'native_bits', 'state.read_native_layout', 'matrices', False),
            (app, 'retained_weights', 'state.verify_checkpoint', '', False),
            (app, 'checkpoint', 'state.checkpoint_native', '', False),
            (subprocess, 'run', 'process.execute', 'process', False),
        ]
        restored = []
        try:
            for owner, name, operation, kind, detached in targets:
                self.targets.append(operation)
                try:
                    original = owner[name] if isinstance(owner, dict) else getattr(owner, name)
                    owned = name in (owner if isinstance(owner, dict) else vars(owner))
                    wrapped = self.wrap(original, operation, kind, detached=detached)
                    if isinstance(owner, dict):
                        owner[name] = wrapped
                    else:
                        setattr(owner, name, wrapped)
                    restored.append((owner, name, original, owned))
                except Exception as error:
                    self.setup_errors.append({'operation': operation, 'error_type': type(error).__name__})
                    self.recorder.warning(error)
                    self.recorder.metric('ops.instrumentation_setup_error', 1, attributes={'operation': operation})
            yield
        finally:
            for owner, name, original, owned in reversed(restored):
                if isinstance(owner, dict):
                    if owned:
                        owner[name] = original
                    else:
                        del owner[name]
                elif owned:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)

    def coverage(self, path):
        """Small actual-call inventory, including hooks that were never reached."""
        def write():
            record = dict(format=FORMAT, targets=self.targets, calls=dict(self.calls),
                errors=dict(self.errors), steps={key: dict(value) for key, value in self.steps.items()},
                setup_errors=self.setup_errors,
                not_called=sorted(set(self.targets) - set(self.calls)),
                scope='Actual wrapped calls only; zero means not observed, never presumed successful. Direct phase spans remain in events.jsonl.')
            destination = Path(path)
            pending = destination.with_name(destination.name + '.pending')
            pending.write_text(json.dumps(record, indent=2) + '\n')
            pending.replace(destination)
            return {}
        self.observe(write)
