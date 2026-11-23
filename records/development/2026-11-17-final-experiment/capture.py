"""Small local operation records around existing calls; never evaluate tensors.

Wrappers retain call arguments, return objects, exceptions and order. Summaries
use explicit scalar fields and tensor metadata only; no input arrays, text,
argv, environments, tensor reductions or extra model/product calls.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
from functools import wraps
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import threading

FORMAT = 'native-continuation-capture/1'

# One catalog supplies both runtime installation and the source-bound inventory.
# Root's direct phase spans remain distinct from the actual helper-call spans.
HOOKS = (
    ('full_graph.weight_gradient', 'proposal.differentiate', 'weight_gradient', False),
    ('full_graph.torch.autograd.grad', 'solver.vjp', 'gradient', True),
    ('full_graph.directional_response', 'proposal.directional_response', 'full_response', False),
    ('full_graph.Graph.__init__', 'model.anchor', 'anchor', False),
    ('full_graph.Graph.forward', 'model.forward', 'forward', False),
    ('full_graph.Graph._layer', 'model.layer', 'layer', False),
    ('legacy.precision.Model.feed_forward', 'model.feed_forward', 'layer_operator', False),
    ('legacy.projection_adapter.Model.attention', 'model.attention', 'layer_operator', False),
    ('legacy.projection_adapter.Model.third', 'model.per_layer_merge', 'layer_operator', False),
    ('full_parameters.ParameterModel.__init__', 'model.bind_parameters', 'parameter_model', False),
    ('full_parameters.ParameterModel.rows', 'model.embedding_rows', 'rows', False),
    ('full_parameters.ParameterModel.per_layer_input', 'model.per_layer_input', 'per_layer_input', False),
    ('full_parameters._DeviceWeights.__getitem__', 'model.parameter', 'parameter', False),
    ('full_parameters.inventory', 'model.parameter_inventory', 'specs', False),
    ('full_state.FullState.preview', 'proposal.round_native', 'state_preview', False),
    ('full_state.FullState.begin', 'state.begin_install', '', False),
    ('full_state.FullState.install', 'weights.install_chunk', 'chunk', False),
    ('full_state.FullState.finish', 'state.commit', '', False),
    ('full_state.FullState.restore_legacy', 'state.restore', '', False),
    ('full_state.FullState.rebind_complete_identity', 'state.bind_revision', '', False),
    ('full_state.FullState.checkpoint', 'state.checkpoint_native', '', False),
    ('full_state.FullState.restore', 'state.restore_complete', '', False),
    ('full_state.accumulate', 'weights.encode', 'native_conversion', False),
    ('full_loss_maps.build_loss_maps', 'computation.map', 'loss_maps', False),
    ('full_loss_maps.write_local', 'computation.save_maps', '', False),
    ('full_loss_maps.correspondence_changes', 'computation.correspondence_changes', '', False),
    ('full_loss_maps.gradient_update_inventory', 'computation.parameter_inventory', '', False),
    ('full_loss_maps.response_comparison', 'computation.response_comparison', '', False),
    ('full_loss_maps.render_loss_views', 'computation.render_maps', '', False),
    ('full_loss_maps.render_cell_overlay', 'computation.render_overlay', '', False),
    ('held.turn_correction.correspondence', 'computation.correspondence', 'pairs', False),
    ('held.turn_correction.measurement', 'computation.measure', '', False),
    ('turn_pair.turns', 'computation.turns', 'turns', False),
    ('vector_chance.turns', 'computation.turns', 'turns', False),
    ('held.raw_forward', 'native.read', '', False),
    ('factory.capture', 'native.acquire', '', False),
    ('factory.memory_model', 'state.create_working', '', False),
    ('load_inputs', 'run.bind_inputs', '', False),
    ('archive_source.published', 'storage.check_publication', 'publication', False),
    ('archive_source.legacy.archive.bucket', 'storage.transfer', 'upload', False),
    ('archive_source.legacy.archive.verify', 'storage.verify_remote', 'verification', False),
    ('archive_source.legacy.archive.release', 'storage.release', 'release', False),
    ('legacy.reclaim.free', 'storage.release_handles', '', False),
    ('preview', 'state.preview_installation', 'preview', False),
    ('install', 'state.install_joint', '', False),
    ('subprocess.run', 'process.execute', 'process', False),
)
DYNAMIC_METHODS = (
    ('keys', 'model.keys'), ('attention', 'model.attention'),
    ('feed_forward', 'model.feed_forward'), ('third', 'model.per_layer_merge'),
)
WORKER_OPERATIONS = frozenset(('model.layer','model.keys','model.parameter','model.normalize',
    'model.activation','model.rotary','model.projection','model.shared_projection',
    'model.embedding_rows','model.attention','model.feed_forward','model.per_layer_merge','model.per_layer_input'))


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
        record = result[0] if isinstance(result, tuple) else result
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
    if kind == 'loss_maps':
        return scalars(result.summary, 'output.') | scalars(vars(result.identity), 'output.')
    if kind == 'specs':
        return {'output.stored_tensors': len(result),
                'output.eligible_tensors': sum(row.eligible for row in result)}
    if kind == 'encode':
        return {'output.bytes': len(result)}
    if kind == 'native_conversion':
        return ({'output.bytes': len(result[0]), 'output.displacement_exact_in_f32': result[4]}
                | metadata(result[1], 'output.stored') | metadata(result[2], 'output.remainder')
                | metadata(result[3], 'output.displacement'))
    if kind in ('tensor', 'rows', 'operator', 'per_layer_input'):
        return metadata(result, 'output.value')
    if kind == 'layer':
        return {'output.values': len(result)} | metadata(result[0], 'output.first_merge')
    if kind == 'layer_operator':
        return (metadata(result, 'output.value') if hasattr(result, 'shape') else
                {'output.values': len(result)} | metadata(result[0], 'output.first'))
    if kind in ('linear', 'shared_linear', 'parameter'):
        return metadata(result, 'output.value')
    if kind == 'preview':
        return {'output.displacement_tensors': len(result[0]), 'output.tensor_receipts': len(result[1])}
    if kind in ('response', 'full_response'):
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
        values = {'output.gradient_count': len(result),
                  'output.unused_gradient_outputs': sum(value is None for value in result)}
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
    if kind == 'full_response':
        return {'input.matrices': len(args[0].leaves), 'input.events': len(args[1]),
                'input.direction_blocks': len(args[2])}
    if kind == 'loss_maps':
        identity = kwargs['identity']
        return {'input.merges': kwargs['expected_merges'], 'input.pairs': len(args[3]),
                **scalars(vars(identity)), 'input.declared_objective': kwargs['declared_objective']}
    if kind == 'state_preview':
        name = args[1] if len(args) > 1 else kwargs['name']
        return {'matrix': name, 'input.matrix': name,
                'input.applied_length': args[3] if len(args) > 3 else kwargs['length'],
                **metadata(args[2] if len(args) > 2 else kwargs['direction'], 'input.direction')}
    if kind == 'chunk':
        chunk = args[1] if len(args) > 1 else kwargs['chunk']
        return {'matrix': chunk.name, 'input.matrix': chunk.name,
                'input.start': chunk.start, 'input.elements': len(chunk.stored),
                **metadata(chunk.stored, 'input.native')}
    if kind == 'encode':
        return metadata(args[0], 'input.stored') | {'input.native_dtype': args[1]}
    if kind == 'rows':
        return {'matrix': args[1], 'input.matrix': args[1], 'input.events': len(args[2])}
    if kind == 'parameter':
        return {'matrix': args[1], 'input.matrix': args[1]}
    if kind == 'shared_linear':
        return {'matrix': 'per_layer_model_proj.weight', 'input.matrix': 'per_layer_model_proj.weight',
                **metadata(args[0], 'input.value'), **metadata(args[1], 'input.weight')}
    if kind == 'linear':
        return metadata(args[0], 'input.value') | metadata(args[1], 'input.weight')
    if kind == 'layer_operator':
        return {'layer': args[1], 'block': args[1], **metadata(args[2], 'input.value')}
    if kind == 'layer':
        return {'layer': args[1], 'block': args[1], **metadata(args[2], 'input.value')}
    if kind in ('tensor', 'operator', 'per_layer_input'):
        value = args[1] if kind in ('tensor', 'per_layer_input') else args[0]
        return metadata(value, 'input.value')
    if kind == 'forward':
        return {'input.events': len(args[1])}  # bound Graph.forward: self, history
    if kind == 'anchor':
        return {'input.events': len(args[2]), 'input.merges': len(args[3]['merged'])}
    if kind in ('rounded', 'native_conversion'):
        return (metadata(args[0], 'input.native') | metadata(args[1], 'input.remainder')
                | metadata(args[2], 'input.direction') | {'input.applied_length': args[3]}
                | ({'input.native_dtype': args[4]} if kind == 'native_conversion' else {}))
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
        self.target_calls = Counter()
        self.recompute_stops = Counter()
        self._product_context = None
        self._counter_lock = threading.RLock()
        self.targets = [row[1] for row in HOOKS] + [row[1] for row in DYNAMIC_METHODS]
        self.targets += ['model.normalize','model.activation','model.rotary','model.projection','model.shared_projection']
        self.setup_errors = []
        self._restored, self._installed = [], set()

    def observe(self, function):
        try:
            return function()
        except Exception as error:
            self.recorder.warning(error)
            self.recorder.metric('ops.capture_summary_error', 1)
            return {}

    def outcome(self, outcome, kind, result):
        outcome.update(self.observe(lambda: result_summary(kind, result)))

    def wrap(self, function, operation, kind='', *, detached=False, target=''):
        @contextmanager
        def worker_scope():
            product = self._product_context
            if (operation in WORKER_OPERATIONS and product is not None
                    and threading.get_ident() != product[0] and not self.recorder._stack.get()):
                # GPU checkpoint replay executes Python on an engine worker.
                # Propagate only this Recorder's explicit context/parent stack,
                # not unrelated Torch ContextVars or an inferred historical link.
                token = self.recorder._stack.set(product[2])
                try:
                    with self.recorder.context(**(product[1] | {'capture_worker_context':'autograd_owner_snapshot'})):
                        yield
                finally:
                    self.recorder._stack.reset(token)
            else:
                yield

        @contextmanager
        def entered(args, kwargs):
            attributes = self.observe(lambda: input_summary(kind, args, kwargs))
            context = {k: attributes[k] for k in ('matrix', 'layer', 'block') if k in attributes}
            with self._counter_lock:
                self.calls[operation] += 1
                if target:
                    self.target_calls[target] += 1
                step = self.recorder.context_attributes.get('training_step')
                self.steps[str(step)][operation] += 1
                attributes.update(operation_call_index=self.steps[str(step)][operation], hook_target=target)
            recompute_stop = None
            with self.recorder.context(**context), self.recorder.span(operation, attributes=attributes, detached=detached) as outcome:
                previous_product = self._product_context
                is_product = operation in ('solver.vjp','proposal.directional_response')
                if is_product:
                    self._product_context = (threading.get_ident(), dict(self.recorder.context_attributes),
                                             self.recorder._stack.get())
                try:
                    yield outcome
                except BaseException as error:
                    # Non-reentrant Torch checkpoint deliberately interrupts
                    # replay after its needed intermediates have been saved.
                    # Preserve that exact signal for its caller, outside the
                    # logging span, instead of inventing a failed model call.
                    checkpoint = sys.modules.get('torch.utils.checkpoint')
                    stop_type = getattr(checkpoint, '_StopRecomputationError', None)
                    if stop_type is not None and isinstance(error, stop_type):
                        recompute_stop = error
                        with self._counter_lock:
                            self.recompute_stops[operation] += 1
                        outcome.update({'output.checkpoint_recompute_stopped': True,
                                        'output.returned': False})
                    else:
                        with self._counter_lock:
                            self.errors[operation] += 1
                        raise
                finally:
                    if is_product:
                        self._product_context = previous_product
            if recompute_stop is not None:
                raise recompute_stop

        if inspect.isgeneratorfunction(function):
            @wraps(function)
            def generator(*args, **kwargs):
                with worker_scope(), entered(args, kwargs) as outcome:
                    result = yield from function(*args, **kwargs)
                    outcome['output.generator_exhausted'] = True
                    return result
            return generator
        @wraps(function)
        def invoke(*args, **kwargs):
            with worker_scope(), entered(args, kwargs) as outcome:
                if kind == 'per_layer_input':
                    # Scope the instance function to this call only: retaining an
                    # instance in the run-wide restoration table retains old
                    # compact tables/weights across successive native histories.
                    model, original = args[0], args[0].shared_linear
                    model.shared_linear = self.wrap(original, 'model.shared_projection', 'shared_linear',
                        target='parameter_model.shared_linear')
                    try:
                        result = function(*args, **kwargs)
                    finally:
                        model.shared_linear = original
                else:
                    result = function(*args, **kwargs)
                if kind == 'parameter_model':
                    try:
                        self.bind_operators(args[0])
                    except Exception as error:
                        self.setup_errors.append({'operation':'model.bind_operators',
                                                  'error_type':type(error).__name__})
                        self.recorder.warning(error)
                        self.recorder.metric('ops.instrumentation_setup_error',1)
                self.outcome(outcome, kind, result)
                return result
        return invoke

    @contextmanager
    def instrument(self, app):
        """Only process-local replacements; restore every hook on every exit."""
        self._restored, self._installed = [], set()
        try:
            for path, operation, kind, detached in HOOKS:
                try:
                    parts = path.split('.')
                    owner = app if len(parts) == 1 else (subprocess if parts[0] == 'subprocess' else app[parts[0]])
                    for component in parts[1:-1]:
                        owner = getattr(owner, component)
                    self.hook(owner, parts[-1], operation, kind, detached=detached, target=path)
                except Exception as error:
                    self.setup_errors.append({'operation': operation, 'target': path,
                                              'error_type': type(error).__name__})
                    self.recorder.warning(error)
                    self.recorder.metric('ops.instrumentation_setup_error', 1, attributes={'operation': operation})
            yield
        finally:
            for owner, name, original, owned in reversed(self._restored):
                if isinstance(owner, dict):
                    if owned:
                        owner[name] = original
                    else:
                        del owner[name]
                elif owned:
                    setattr(owner, name, original)
                else:
                    delattr(owner, name)
            self._restored, self._installed = [], set()

    def hook(self, owner, name, operation, kind='', *, detached=False, target=''):
        key = (id(owner), name)
        if key in self._installed:
            return
        owned = name in (owner if isinstance(owner, dict) else vars(owner))
        raw = owner[name] if isinstance(owner, dict) else vars(owner).get(name)
        original = owner[name] if isinstance(owner, dict) else getattr(owner, name)
        descriptor = type(raw) if isinstance(raw, (classmethod, staticmethod)) else None
        if descriptor:
            original = raw.__func__
        replacement = self.wrap(original, operation, kind, detached=detached, target=target)
        if descriptor:
            replacement = descriptor(replacement)
        if isinstance(owner, dict):
            owner[name] = replacement
        else:
            setattr(owner, name, replacement)
        self._restored.append((owner, name, raw if owned else original, owned))
        self._installed.add(key)
        self.targets.append(operation)

    def bind_operators(self, model):
        """Bind the actual executed source class after ParameterModel construction.

        Class descriptors/instance functions are restored when instrumentation
        exits. No constructor, layer or product is called to discover metadata.
        """
        cls = model._model_class
        for name, operation in DYNAMIC_METHODS:
            self.hook(cls, name, operation, 'layer_operator', target='executed_model.' + name)
        environment = cls.per_layer_input.__globals__
        for name, operation in (('rms','model.normalize'), ('gelu','model.activation'), ('rope','model.rotary')):
            self.hook(environment, name, operation, 'operator', target='executed_model.' + name)
        self.hook(environment['F'], 'linear', 'model.projection', 'linear', target='executed_model.F.linear')


    def coverage(self, path):
        """Small actual-call inventory, including hooks that were never reached."""
        def write():
            record = dict(format=FORMAT, targets=self.targets, calls=dict(self.calls),
                target_calls=dict(self.target_calls),
                checkpoint_recompute_stops=dict(self.recompute_stops),
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


def operation_inventory(program=None):
    """Refresh source bindings only; no controller import, instrumentation or run."""
    source = Path(__file__).resolve()
    root = source.parents[3]
    program = source.with_name('run.py') if program is None else Path(program).resolve()
    tree = ast.parse(program.read_text())
    direct = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'span' and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            direct.append(dict(operation=node.args[0].value, line=node.lineno,
                detached=any(k.arg == 'detached' and isinstance(k.value, ast.Constant)
                             and k.value.value is True for k in node.keywords)))
    hooks = [dict(target=p, operation=o, kind=k, detached=d) for p, o, k, d in HOOKS]
    dynamic = [dict(target='executed_model.' + p, operation=o) for p, o in DYNAMIC_METHODS]
    dynamic += [dict(target=p, operation=o) for p, o in (
        ('executed_model.rms','model.normalize'), ('executed_model.gelu','model.activation'),
        ('executed_model.rope','model.rotary'), ('executed_model.F.linear','model.projection'),
        ('parameter_model.shared_linear','model.shared_projection'))]
    def bound(path):
        return dict(record=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(format=FORMAT, program=bound(program), capture=bound(source),
        archive=bound(source.with_name('archive.py')),
        archive_workflow=bound(root/'studies/04-round-2-overlap/runs/vector-chance-loss-rise-diagnosis/empirical-eta-continuation/consumed_trace_archive.py'),
        archive_operator=bound(root/'records/development/2026-11-17-capture-context/archive_inputs.py'),
        operations=sorted({row['operation'] for row in direct + hooks + dynamic}),
        direct=sorted(direct, key=lambda row: row['line']), hooks=hooks, dynamic_hooks=dynamic,
        context_fields=['training_step','source_model_generation','model_generation','target_model_generation',
            'training_member','loss_version','tolerance_sha256','reference_sha256','layer','block','matrix','response_kind'],
        io='Explicit scalar fields and tensor shape/dtype/device only. No arrays, model weights, text, argv or environment values.',
        dynamic_binding='Actual ParameterModel source class after construction; no extra model call. Hook failures and unreached hooks remain explicit.',
        generator_scope='FullState.preview retains lazy generator execution, source context, original yield/send/throw/close semantics.',
        coverage_file='observability/coverage.json',
        refresh='Execute this capture.py directly after finalizing run.py; it only writes operations.json source bindings.')



if __name__ == "__main__":
    Path(__file__).with_name("operations.json").write_text(json.dumps(operation_inventory(), indent=2) + "\n")
