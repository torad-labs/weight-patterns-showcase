"""The paper's full turn trace, not stream values or operator matching."""
import tempfile
import unittest

import torch

from pipeline.weight_edit import joint_functional_step as joint, native_graph, turn_pair
from tests.test_native_graph import fixture, rounded_trace


class TurnPairTests(unittest.TestCase):
    def test_different_states_with_the_same_turn_have_no_request(self):
        first = dict(incoming=[torch.tensor([[1., 1., 0.]])],
                     merged=[torch.tensor([[-1., 1., 0.]])])
        second = dict(incoming=[torch.tensor([[1., 0., 1.]])],
                      merged=[torch.tensor([[-1., 0., 1.]])])
        reference = turn_pair.Reference(first, [torch.zeros(1, 3)], expected_merges=1)
        measured = list(reference.measure(second, [[0, 0]]))
        torch.testing.assert_close(measured[0].residual, torch.zeros(1, 3, dtype=torch.float64),
                                   rtol=0, atol=0)
        self.assertFalse(bool(measured[0].outside.any()))

    def test_turn_amount_and_the_incoming_edge_both_matter(self):
        first = dict(incoming=[torch.tensor([[1., 0.]])],
                     merged=[torch.tensor([[0., 1.]])])
        reference = turn_pair.Reference(first, [torch.zeros(1, 2)], expected_merges=1)
        same_direction_smaller_turn = dict(incoming=[torch.tensor([[4., 3.]])],
                                          merged=[torch.tensor([[3., 4.]])])
        same_output_different_input = dict(incoming=[torch.tensor([[0., -1.]])],
                                          merged=first['merged'])
        for current in (same_direction_smaller_turn, same_output_different_input):
            with self.subTest(current=current):
                measured = list(reference.measure(current, [[0, 0]]))
                self.assertTrue(bool(measured[0].outside.any()))
                self.assertGreater(float(measured[0].residual.square().sum()), 0.)

    def test_one_fixed_component_boundary_controls_request_and_membership(self):
        original = dict(incoming=[torch.tensor([[1., 0.]])],
                        merged=[torch.tensor([[0., 1.]])])
        current = dict(incoming=[torch.tensor([[1., 0.]])],
                       merged=[torch.tensor([[-1., 0.]])])
        reference = turn_pair.Reference(original, [torch.tensor([[1., 0.]])], expected_merges=1)
        measured = list(reference.measure(current, [[0, 0]]))[0]
        torch.testing.assert_close(measured.residual, torch.tensor([[0., -1.]], dtype=torch.float64))
        self.assertTrue(bool(measured.outside.all()))

    def test_missing_events_or_merges_cannot_be_ignored(self):
        original = dict(incoming=[torch.eye(2)], merged=[-torch.eye(2)])
        reference = turn_pair.Reference(original, [torch.zeros(2, 2)], expected_merges=1)
        with self.assertRaisesRegex(ValueError, 'complete'):
            list(reference.measure(original, [[0, 0]]))
        with self.assertRaisesRegex(ValueError, 'merge'):
            list(reference.measure(dict(incoming=[], merged=[]), [[0, 0], [1, 1]]))

    def test_repeated_pairs_keep_the_original_mass(self):
        original = dict(incoming=[torch.tensor([[1., 0.], [1., 0.]])],
                        merged=[torch.tensor([[0., 1.], [0., 1.]])])
        reference = turn_pair.Reference(original, [torch.zeros(2, 2)], expected_merges=1)
        totals = []
        for count, pairs in ((2, [[0, 0], [1, 1]]),
                             (3, [[0, 0], [0, 1], [1, 2]])):
            current = dict(incoming=[torch.tensor([[1., 0.]]).repeat(count, 1)],
                           merged=[torch.tensor([[-1., 0.]]).repeat(count, 1)])
            totals.append(torch.stack([m.residual.square().sum()
                                       for m in reference.measure(current, pairs)]).sum())
        torch.testing.assert_close(totals[0], totals[1])

    def test_native_graph_turns_keep_both_edges_and_the_joint_weight_derivative(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            file, tensors = fixture.made(folder)
            model = fixture.tf.Model(file, tensors, fixture.ARCHITECTURE, 'cpu')
            history = fixture.TEXT
            native = rounded_trace(model, history)
            graph = native_graph.Graph(model, history, native)
            actual = turn_pair.turns(graph.forward(history), expected_merges=3 * model.layers)
            expected = fixture.tf.turns(native)
            self.assertEqual(len(actual), 3 * model.layers)
            for ours, theirs in zip(actual, expected):
                torch.testing.assert_close(ours, theirs, rtol=0, atol=0)
            target = [value.detach().clone() for value in actual]
            target[-1] = target[-1] + torch.ones_like(target[-1])
            def residual(forward):
                current = turn_pair.turns(forward, expected_merges=3 * model.layers)
                return tuple(value - wanted for value, wanted in zip(current, target))
            loss = torch.stack([(value - wanted).square().sum()
                                for value, wanted in zip(actual, target)]).sum()
            names = tuple(model.leaves)
            gradients = torch.autograd.grad(loss, tuple(model.w[name] for name in names))
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
            self.assertTrue(any(bool((g != 0).any()) for g in gradients))
            response = joint.response(model, names, {'one': joint.Trace(graph, history, residual)},
                                      dict(zip(names, (-g for g in gradients))), expected_members=['one'])
            expected_slope = -sum(float(g.double().square().sum()) for g in gradients)
            torch.testing.assert_close(torch.tensor(response.slope), torch.tensor(expected_slope))


if __name__ == '__main__':
    unittest.main()
