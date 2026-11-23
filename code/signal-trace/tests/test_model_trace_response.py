"""The response traverses the actual model graph, on a tiny made-up CPU model."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from pipeline.weight_edit import model_trace_response as response

HERE = Path(__file__).resolve().parents[3] / 'records/development/2026-11-12-path-forward'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('response_forward_fixture', HERE / 'test_torch_forward.py')
assert spec is not None and spec.loader is not None
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class ModelResponseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        file, tensors = fixture.made(self.tmp.name)
        self.model = fixture.tf.Model(file, tensors, fixture.ARCHITECTURE, 'cpu')
        self.names = tuple(name for name, weight in self.model.w.items() if weight.requires_grad)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_merges_and_shared_attention_have_the_same_joint_derivative(self):
        # A stand-in squared residual tests plumbing, not a scientific trace loss.
        residuals = lambda forward: tuple(out - before for before, out in
                                         zip(forward['incoming'], forward['merged']))
        raw = residuals(self.model.forward(fixture.TEXT))
        objective = torch.stack([r.double().square().sum() for r in raw]).sum()
        parameters = tuple(self.model.w[name] for name in self.names)
        gradient = torch.autograd.grad(objective, parameters)
        direction = dict(zip(self.names, (-g for g in gradient)))
        expected_slope = -sum(float(g.double().square().sum()) for g in gradient)
        before_mapping = self.model.w
        step = response.measure_model(self.model, self.names, fixture.TEXT, direction, residuals)
        torch.testing.assert_close(torch.tensor(step.slope), torch.tensor(expected_slope))
        torch.testing.assert_close(torch.tensor(step.objective), objective.float())
        self.assertEqual(step.status, 'proposal')
        self.assertLess(step.predicted_objective, step.objective)
        self.assertIs(self.model.w, before_mapping)
        self.assertTrue(all(self.model.w[name] is weight for name, weight in zip(self.names, parameters)))
        self.assertEqual(len(raw), 3 * fixture.ARCHITECTURE['layers'])

    def test_failed_measurement_restores_the_model_mapping(self):
        original = self.model.w
        direction = {name: torch.ones_like(original[name]) for name in self.names}
        def broken(_forward):
            raise ValueError('fixture observable failed')
        with self.assertRaisesRegex(ValueError, 'fixture observable'):
            response.measure_model(self.model, self.names, fixture.TEXT, direction, broken)
        self.assertIs(self.model.w, original)

    def test_consumer_exception_restores_mapping_while_generator_is_retained(self):
        original = self.model.w
        direction = {name: torch.ones_like(original[name]) for name in self.names}
        retained = []
        real_measure = response.measure
        def capture(function, parameters, directions):
            def observed(*weights):
                generator = function(*weights)
                retained.append(generator)
                return generator
            return real_measure(observed, parameters, directions)
        def residual(forward):
            yield forward['merged'][0]
            yield torch.tensor([float('nan')])
        with patch.object(response, 'measure', side_effect=capture):
            with self.assertRaisesRegex(ValueError, 'must be finite'):
                response.measure_model(self.model, self.names, fixture.TEXT, direction, residual)
        # The reducer still retains the generator object. Outer-call cleanup
        # must close it now, rather than waiting for eventual GC/finalization.
        self.assertTrue(retained)
        self.assertIs(self.model.w, original)
        self.assertTrue(all(generator.gi_frame is None for generator in retained))

    def test_missing_matrix_direction_is_refused_not_an_implicit_partial_update(self):
        direction = {name: torch.ones_like(self.model.w[name]) for name in self.names[1:]}
        with self.assertRaisesRegex(ValueError, 'every declared'):
            response.measure_model(self.model, self.names, fixture.TEXT, direction,
                                   lambda forward: forward['exit'])


if __name__ == '__main__':
    unittest.main()
