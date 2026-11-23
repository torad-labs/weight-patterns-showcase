"""Tracing preserves research behavior and records only explicit observations."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from pipeline.observability import Recorder
from pipeline.observability.__main__ import validate_spec
from pipeline.observability.instrumentation import TARGETS, instrument, wrapped
from pipeline.observability.recorder import FORMAT
from tests.observability_fixture import record_two_steps


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / 'events.jsonl'
        self.recorder = Recorder(self.path, {'folder': 'fixture'})
        self.addCleanup(self.recorder.close)

    def events(self):
        return [json.loads(line) for line in self.path.read_text().splitlines()]

    def test_nested_spans_and_detached_work_retain_real_relationships(self):
        with self.recorder.context(training_step=3), self.recorder.span('outer'):
            with self.recorder.span('child'):
                pass
            with self.recorder.span('independent', detached=True):
                with self.recorder.span('inner'):
                    pass
        starts = {row['name']: row for row in self.events() if row['event'] == 'span_start'}
        self.assertEqual(starts['child']['parent_id'], starts['outer']['span_id'])
        self.assertEqual(starts['child']['trace_id'], starts['outer']['trace_id'])
        self.assertIsNone(starts['independent']['parent_id'])
        self.assertEqual(starts['independent']['parent_trace_id'], starts['outer']['trace_id'])
        self.assertEqual(starts['inner']['trace_id'], starts['independent']['trace_id'])
        self.assertEqual(starts['inner']['attributes']['training_step'], 3)
        rows = self.events()
        self.assertEqual([row['seq'] for row in rows], list(range(1, len(rows) + 1)))
        self.assertTrue(all(row['duration_seconds'] >= 0 for row in rows if row['event'] == 'span_end'))

    def test_arguments_and_arbitrary_objects_are_never_serialized(self):
        class MustNotRead:
            def __str__(self):
                raise AssertionError('an argument was inspected')
        value = MustNotRead()
        function = wrapped(lambda supplied: supplied, self.recorder, 'opaque')
        self.assertIs(function(value), value)
        with self.recorder.span('metadata', attributes={'opaque': value, 'valid': 7}):
            pass
        self.assertNotIn('opaque":', self.path.read_text())
        self.assertIn('"valid":7', self.path.read_text())

    def test_logging_failure_does_not_replace_original_exception(self):
        class BrokenFile:
            def write(self, _):
                raise OSError('disk unavailable')
            def close(self):
                pass
        failure = LookupError('original research failure')
        def operation():
            raise failure
        with patch.object(self.recorder, '_file', BrokenFile()), self.assertRaises(LookupError) as caught:
            wrapped(operation, self.recorder, 'failing')()
        self.assertIs(caught.exception, failure)

    def test_error_status_without_implicit_exception_message_capture(self):
        with self.assertRaises(ValueError):
            with self.recorder.span('failure'):
                raise ValueError('private response content')
        end = self.events()[-1]
        self.assertEqual(end['status'], 'ERROR')
        self.assertEqual(end['attributes']['error.type'], 'ValueError')
        self.assertNotIn('private response content', self.path.read_text())

    def test_span_setup_failure_runs_research_once(self):
        calls = []
        with patch('pipeline.observability.recorder.uuid.uuid4', side_effect=OSError('entropy unavailable')):
            result = wrapped(lambda: calls.append('ran') or 17, self.recorder, 'operation')()
        self.assertEqual(result, 17)
        self.assertEqual(calls, ['ran'])

    def test_span_teardown_failure_preserves_research_exception(self):
        failure = RuntimeError('original')
        with patch('pipeline.observability.recorder.time.perf_counter', side_effect=[1., OSError('clock')]):
            with self.assertRaises(RuntimeError) as caught:
                with self.recorder.span('operation'):
                    raise failure
        self.assertIs(caught.exception, failure)

    def test_subprocess_pathlike_is_not_stringified_by_observer(self):
        class Executable:
            def __fspath__(self):
                return '/bin/true'
            def __str__(self):
                raise RuntimeError('telemetry must not call str')
        with instrument(self.recorder, targets=()):
            result = subprocess.run([Executable()], check=True)
        self.assertEqual(result.returncode, 0)

    def test_instrumentation_import_failure_does_not_skip_research(self):
        calls = []
        with patch('pipeline.observability.instrumentation.importlib.import_module',
                   side_effect=ImportError('instrumentation unavailable')):
            with instrument(self.recorder, TARGETS[:1]):
                calls.append('research')
        self.assertEqual(calls, ['research'])
        errors = [row for row in self.events() if row.get('name') == 'ops.instrumentation_setup_error']
        self.assertEqual(len(errors), 1)

    def test_metrics_are_finite_explicit_namespaces(self):
        self.recorder.metric('native.objective', .2, step=1)
        self.recorder.metric('answer.correct', 1, step=1)
        self.recorder.metric('native.objective', float('nan'))
        self.recorder.metric('native.objective', True)
        rows = [row for row in self.events() if row['event'] == 'metric']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['name'], 'native.objective')

    def test_solver_callback_order_and_return_identity_are_preserved(self):
        calls = []
        result = SimpleNamespace(status='stationary', iterations=2)
        def solve(residual, parameters, *, progress=None):
            assert progress is not None
            for iteration in range(3):
                progress(iteration, 1. / (iteration + 1))
            return result
        callback = lambda iteration, objective: calls.append((iteration, objective))
        with self.recorder.context(training_step=4):
            actual = wrapped(solve, self.recorder, 'solver.solve')(None, None, progress=callback)
        self.assertIs(actual, result)
        self.assertEqual(calls, [(0, 1.), (1, .5), (2, 1. / 3)])
        metrics = [row for row in self.events() if row.get('name') == 'solver.step_4.linear_objective']
        self.assertEqual([row['step'] for row in metrics], [0, 1, 2])
        self.assertTrue(all(row['attributes']['measurement_kind'] == 'linear_surrogate' for row in metrics))

    def test_original_callback_exception_is_preserved(self):
        failure = RuntimeError('callback failed')
        def callback(*_):
            raise failure
        def solve(*_, progress=None):
            assert progress is not None
            progress(0, 1.)
        with self.assertRaises(RuntimeError) as caught:
            wrapped(solve, self.recorder, 'solver.solve')(progress=callback)
        self.assertIs(caught.exception, failure)

    def test_candidate_evaluation_arguments_and_result_are_preserved(self):
        weights = object()
        value = SimpleNamespace(estimate=.4, lower=.4, upper=.4)
        seen = []
        def evaluate(supplied):
            seen.append(supplied)
            return value
        result = SimpleNamespace(value=value, status='accepted', length=.7, trials=[1])
        def search(*_, evaluate):
            self.assertIs(evaluate(weights), value)
            return result
        actual = wrapped(search, self.recorder, 'proposal.search')(evaluate=evaluate)
        self.assertIs(actual, result)
        self.assertEqual(seen, [weights])
        starts = [row for row in self.events() if row.get('name') == 'proposal.evaluate']
        self.assertEqual(len(starts), 1)

    def test_two_steps_keep_searchable_identity_and_distinct_counters(self):
        self.recorder.attributes.update(original_model_sha256='synthetic-original',
                                        reference_sha256='synthetic-reference',
                                        synthetic_fixture=True)
        callbacks = record_two_steps(self.recorder, wrapped)
        self.assertEqual(callbacks, [(1, 0, 1.), (1, 1, .5), (2, 0, 1.), (2, 1, .5)])
        starts = [row for row in self.events() if row['event'] == 'span_start']
        ends = {row['span_id']: row for row in self.events() if row['event'] == 'span_end'}
        self.assertTrue(all(row['attributes']['folder'] == 'fixture' for row in starts))
        self.assertTrue(all(row['attributes']['original_model_sha256'] == 'synthetic-original'
                            for row in starts))
        for step in (1, 2):
            rows = [row for row in starts if row['attributes'].get('training_step') == step]
            self.assertTrue(rows)
            self.assertTrue(all(row['attributes']['model_sha256'] == f'synthetic-current-model-{step}'
                                for row in rows))
            products = [row['attributes'] for row in rows if row['name'] in ('solver.jvp', 'solver.vjp')]
            self.assertEqual([row['solver_product_index'] for row in products], [1, 2, 3, 4])
            self.assertEqual([row['solver_accepted_iteration'] for row in products], [0, 0, 0, 1])
            self.assertTrue(all(row['phase'] == 'solver' and row['trace_level'] == 'detail' for row in products))
            solve = next(row for row in rows if row['name'] == 'solver.solve')
            self.assertEqual(ends[solve['span_id']]['attributes']['solver_accepted_iteration'], 1)
            self.assertEqual(ends[solve['span_id']]['attributes']['solver_product_index'], 4)
            evaluated = [row['attributes'] for row in rows if row['name'] == 'proposal.evaluate']
            self.assertEqual([row['proposal_evaluation_index'] for row in evaluated], [0, 1])
            self.assertEqual([row['proposal_evaluation_kind'] for row in evaluated], ['baseline', 'candidate'])
            self.assertTrue(all(row['phase'] == 'proposal' and row['trace_level'] == 'detail' for row in evaluated))
            # Inverse counters must not leak into the following candidate search.
            self.assertTrue(all('solver_product_index' not in row for row in evaluated))
        self.assertEqual(self.recorder.context_attributes, {})

    def test_prospective_binding_is_required(self):
        source = {'record': 'code/observer.py', 'sha256': 'example'}
        declaration = dict(implementation=[source], observability=dict(
            format=FORMAT, entrypoint='pipeline.example', events='observability/events.jsonl',
            sources=[source]))
        validate_spec(declaration, 'pipeline.example', [source])
        with self.assertRaises(ValueError):
            validate_spec(declaration, 'pipeline.other', [source])
        declaration['implementation'] = []
        with self.assertRaises(ValueError):
            validate_spec(declaration, 'pipeline.example', [source])

    def test_direct_gradient_spans_preserve_results_and_separate_prediction(self):
        class PrivateWeights:
            def __str__(self):
                raise AssertionError('weight values must not be serialized')
        gradient = SimpleNamespace(norm_squared=2., before={'objective': .5}, direction=PrivateWeights())
        response = SimpleNamespace(status='proposal', slope=-2., curvature=4.,
                                   length=.25, predicted_objective=.25)
        proposal = SimpleNamespace(status='ready', gradient_norm_squared=2., changed_values=7,
                                   weights=PrivateWeights(), timings={'weight_derivative_seconds': .1})
        with self.recorder.context(training_step=2):
            for name, value in [('proposal.weight_derivative', gradient),
                                ('proposal.directional_response', response), ('proposal.gradient_step', proposal)]:
                self.assertIs(wrapped(lambda: value, self.recorder, name)(), value)
        ends = [row['attributes'] for row in self.events() if row['event'] == 'span_end']
        self.assertEqual(ends[0]['proposal.gradient_norm_squared'], 2.)
        self.assertEqual(ends[1]['measurement_kind'], 'local_linear_prediction')
        self.assertEqual(ends[1]['proposal.response_scale'], .25)
        self.assertEqual(ends[2]['status'], 'ready')
        self.assertFalse(any(key.startswith('native.') for attrs in ends for key in attrs))
        self.assertNotIn('PrivateWeights', self.path.read_text())

    def test_gradient_entrypoint_and_its_override_are_prospectively_instrumented(self):
        from pipeline.observability.__main__ import ENTRYPOINTS
        self.assertEqual(ENTRYPOINTS['vector-gradient-loop'], 'pipeline.weight_edit.vector_gradient_loop')
        self.assertIn(('vector_gradient_loop', 'Backend.correct', 'correction.step', True), TARGETS)
        for path in ('weight_gradient', 'propose'):
            self.assertTrue(any(module == 'trace_gradient' and attribute == path
                                for module, attribute, _, _ in TARGETS))

    def test_working_update_and_fresh_observation_keep_revision_without_fake_sha(self):
        revision = dict(kind='native-working-revision/1', run='synthetic/run', generation=2,
                        original_sha256='a' * 64)
        reached = dict(model_identity=revision, model_sha256=None, generation=2, status='ready')
        observed = reached | {'objective': .25, 'outside': 3, 'complete': True, 'ended': True}
        self.assertIs(wrapped(lambda *args: reached, self.recorder, 'state.install')(None, 2, object()), reached)
        self.assertIs(wrapped(lambda *args: observed, self.recorder, 'native.observe')(None, 3), observed)
        ends = [row['attributes'] for row in self.events() if row['event'] == 'span_end']
        for row in ends:
            self.assertEqual(row['model_identity.kind'], revision['kind'])
            self.assertEqual(row['model_identity.run'], revision['run'])
            self.assertEqual(row['model_generation'], 2)
            self.assertNotIn('model_sha256', row)
        self.assertNotIn('native.objective', ends[0])
        self.assertEqual(ends[1]['native.objective'], .25)
        starts = [row['attributes'] for row in self.events() if row['event'] == 'span_start']
        self.assertEqual([row['training_step'] for row in starts], [2, 3])

    def test_working_entrypoint_hooks_overrides_without_double_wrapping_inherited_correction(self):
        from pipeline.observability.__main__ import ENTRYPOINTS
        from pipeline.weight_edit import vector_working_loop, vector_gradient_loop
        self.assertEqual(ENTRYPOINTS['vector-working-loop'], 'pipeline.weight_edit.vector_working_loop')
        targets = [item for item in TARGETS if item[0] == 'vector_working_loop']
        self.assertIn(('vector_working_loop', 'Backend.observe', 'native.observe', True), targets)
        self.assertIn(('vector_working_loop', 'Backend.install', 'state.install', True), targets)
        self.assertFalse(any(item[1] == 'Backend.correct' for item in targets))
        with instrument(self.recorder):
            self.assertIs(vector_working_loop.Backend.correct, vector_gradient_loop.Backend.correct)

    def test_real_joint_inverse_is_unchanged_and_wrappers_restore(self):
        import torch
        from pipeline.weight_edit import joint_inverse
        matrix = torch.tensor([[2., 1.], [1., 3.]])
        target = torch.tensor([1., -2.])
        residual = lambda weights: matrix @ weights - target
        parameters = (torch.zeros(2),)
        expected = joint_inverse.solve(residual, parameters)
        original = joint_inverse.solve
        selected = tuple(item for item in TARGETS if item[0] == 'joint_inverse')
        with instrument(self.recorder, selected):
            actual = joint_inverse.solve(residual, parameters)
        self.assertIs(joint_inverse.solve, original)
        self.assertEqual(actual.objectives, expected.objectives)
        self.assertEqual(actual.status, expected.status)
        for first, second in zip(actual.changes, expected.changes):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_actual_vector_reference_bound_turn_alias_is_instrumented(self):
        import torch
        from pipeline.weight_edit import vector_chance
        reference = vector_chance.Reference([torch.tensor([[-1., 1.]])],
                                            [torch.zeros(1)], expected_merges=1)
        forward = dict(incoming=[torch.tensor([[1., 0.]])],
                       merged=[torch.tensor([[0., 1.]])])
        targets = tuple(item for item in TARGETS if item[0] in ('turn_pair', 'vector_chance'))
        with instrument(self.recorder, targets):
            reading = next(reference.measure(forward, [[0, 0]]))
        self.assertFalse(bool(reading.outside.any()))
        events = [row for row in self.events() if row.get('name') == 'computation.turns']
        self.assertEqual(len(events), 1)

    def test_all_declared_targets_resolve_and_restore(self):
        before = []
        for module, path, _, _ in TARGETS:
            owner = importlib.import_module('pipeline.weight_edit.' + module)
            parts = path.split('.')
            for part in parts[:-1]:
                owner = getattr(owner, part)
            owned = parts[-1] in vars(owner)
            original = vars(owner)[parts[-1]] if owned else getattr(owner, parts[-1])
            before.append((owner, parts[-1], original, owned))
        with instrument(self.recorder):
            pass
        for owner, key, value, owned in before:
            self.assertIs(vars(owner)[key] if owned else getattr(owner, key), value)
            self.assertEqual(key in vars(owner), owned)


if __name__ == '__main__':
    unittest.main()
