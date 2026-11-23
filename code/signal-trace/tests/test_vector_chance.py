"""A whole-vector chance boundary has one continuous request and stop rule."""
import os
import unittest

import numpy as np
import torch

from pipeline.weight_edit import joint_inverse, reference_rule, turn_pair, vector_chance


class VectorChanceTests(unittest.TestCase):
    def test_collapsed_interval_keeps_the_identity_response_at_zero_request(self):
        delta = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
        bound = torch.zeros_like(delta)
        result = reference_rule.outside(delta, bound, bound, torch.ones(1))
        gradient, = torch.autograd.grad(result.residual.sum(), delta)
        torch.testing.assert_close(gradient, torch.ones_like(delta))
        self.assertFalse(bool(result.outside.any()))

    def test_one_whole_vector_not_separate_coordinate_limits_defines_agreement(self):
        delta = torch.tensor([[1., 1.], [2., 0.]], dtype=torch.float64)
        result = vector_chance.outside(delta, torch.ones(2), torch.ones(2))
        self.assertFalse(bool(result.outside[0]))
        self.assertTrue(bool(result.outside[1]))
        torch.testing.assert_close(result.residual[0], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(result.residual[1],
                                   torch.tensor([2. - np.sqrt(2.), 0.], dtype=torch.float64))

    def test_zero_chance_retains_the_exact_joint_response_at_agreement(self):
        delta = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
        result = vector_chance.outside(delta, torch.zeros(1), torch.ones(1))
        gradient, = torch.autograd.grad(result.residual.sum(), delta)
        torch.testing.assert_close(gradient, torch.ones_like(delta))
        self.assertFalse(bool(result.outside.any()))

    def test_joint_inverse_keeps_zero_chance_relations_while_fulfilling_another_request(self):
        first, second = torch.zeros(1), torch.zeros(1)
        def residual(a, b):
            delta = torch.stack((torch.cat((a, a * 0)), torch.cat((a + b - 1, b * 0))))
            return vector_chance.outside(delta, torch.zeros(2), torch.ones(2)).residual
        solved = joint_inverse.solve(residual, (first, second))
        torch.testing.assert_close(solved.changes[0], torch.zeros(1))
        torch.testing.assert_close(solved.changes[1], torch.ones(1))

    @unittest.skipUnless(os.environ.get('RUN_CUDA_TURN_FIXTURE') == '1',
                         'run explicitly under the declared GPU lock')
    def test_gpu_graph_uses_the_same_canonical_turns_without_cutting_either_derivative(self):
        before = torch.tensor([[2., 1., 0.]], device='cuda', requires_grad=True)
        after = torch.tensor([[1., 2., 1.]], device='cuda', requires_grad=True)
        actual = turn_pair.turns(dict(incoming=[before], merged=[after]), expected_merges=1)[0]
        expected = turn_pair.turns(dict(incoming=[before.detach().cpu()],
                                       merged=[after.detach().cpu()]), expected_merges=1)[0]
        self.assertEqual(actual.device.type, 'cpu')
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        gradients = torch.autograd.grad(actual.sum(), (before, after))
        self.assertTrue(all(gradient.device.type == 'cuda' and bool(gradient.abs().sum() > 0)
                            for gradient in gradients))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(before.detach(), torch.ones_like(before))
            value = turn_pair.turns(dict(incoming=[dual], merged=[after.detach()]), expected_merges=1)[0]
            _, tangent = torch.autograd.forward_ad.unpack_dual(value)
            assert tangent is not None
            self.assertEqual(tangent.device.type, 'cpu')
            torch.testing.assert_close(tangent.sum().float(), gradients[0].sum().cpu())

    def test_vector_boundary_is_unchanged_by_an_orthogonal_coordinate_change(self):
        delta = torch.tensor([[3., 1.]], dtype=torch.float64)
        rotation = torch.tensor([[.6, -.8], [.8, .6]], dtype=torch.float64)
        first = vector_chance.outside(delta, torch.ones(1), torch.ones(1))
        second = vector_chance.outside(delta @ rotation, torch.ones(1), torch.ones(1))
        torch.testing.assert_close(first.residual @ rotation, second.residual)
        self.assertEqual(bool(first.outside[0]), bool(second.outside[0]))


if __name__ == '__main__':
    unittest.main()
