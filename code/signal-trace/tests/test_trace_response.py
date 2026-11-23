"""Update lengths come from the joint matrix response, not the scalar loss."""
import math
import unittest

import torch

from pipeline.weight_edit import trace_response as response

F = torch.float64


class ResponseTests(unittest.TestCase):
    def test_linear_problem_uses_response_curvature(self):
        w = torch.tensor([0.], dtype=F)
        direction = torch.tensor([1.], dtype=F)
        step = response.measure(lambda x: 3 * x - 6, (w,), (direction,))
        self.assertEqual(step.objective, 36)
        self.assertEqual(step.slope, -36)
        self.assertEqual(step.curvature, 9)
        self.assertEqual(step.length, 2)
        self.assertEqual(step.predicted_objective, 0)
        self.assertEqual(step.status, 'proposal')

    def test_coupled_matrices_contribute_cross_terms_before_squaring(self):
        a, b = (torch.tensor([x], dtype=F) for x in (2., 3.))
        step = response.measure(lambda first, second: first * second - 12,
                                (a, b), (torch.ones_like(a), torch.ones_like(b)))
        # Both matrices move: d(ab)/ds = b + a = 5, not sqrt(a²+b²).
        self.assertEqual(step.curvature, 25)
        self.assertEqual(step.length, 6 / 5)
        self.assertEqual(step.slope, -60)

    def test_full_attention_graph_response_matches_direct_directional_derivative(self):
        x = torch.tensor([[0.3, 1.2], [1.1, -0.2]], dtype=F)
        a = torch.tensor([[0.6, -0.1], [0.2, 0.7]], dtype=F, requires_grad=True)
        b = torch.tensor([[0.4, 0.3], [-0.1, 0.5]], dtype=F, requires_grad=True)
        da, db = torch.ones_like(a), torch.ones_like(b) / 3
        def residual(first, second):
            incoming = torch.softmax(x @ first @ x.T, -1) @ x
            return incoming + torch.tanh(incoming @ second)
        r = residual(a, b)
        g = torch.autograd.grad(r.square().sum(), (a, b))
        expected = ((g[0] * da).sum() + (g[1] * db).sum()).item()
        step = response.measure(residual, (a, b), (da, db))
        self.assertAlmostEqual(step.slope, expected, places=12)

    def test_guide_normalized_merge_response_includes_rms_and_direction(self):
        x = torch.tensor([2., -1.], dtype=F)
        h = torch.tensor([0.2, 0.7], dtype=F)
        gain = torch.tensor([1.1, 0.6], dtype=F)
        w = torch.tensor([[0.8, -0.3], [0.1, 0.9]], dtype=F)
        s = torch.tensor([[0.3, 0.1], [-0.2, 0.4]], dtype=F)
        epsilon = 1e-6  # The model's RMS epsilon, not an optimizer parameter.
        def residual(matrix):
            y = matrix @ x
            m = h + gain * y / torch.sqrt(y.square().mean() + epsilon)
            return m / torch.linalg.vector_norm(m)
        y = w @ x
        rms = torch.sqrt(y.square().mean() + epsilon)
        norm_derivative = torch.eye(2, dtype=F) / rms - torch.outer(y, y) / (2 * rms**3)
        m = h + gain * y / rms
        d = m / torch.linalg.vector_norm(m)
        A = (torch.eye(2, dtype=F) - torch.outer(d, d)) / torch.linalg.vector_norm(m)
        expected = A @ torch.diag(gain) @ norm_derivative @ s @ x
        step = response.measure(residual, (w,), (s,))
        self.assertAlmostEqual(step.curvature, expected.square().sum().item(), places=12)

    def test_partitioned_trace_keeps_the_same_response(self):
        w = torch.tensor([0.5, 1.5], dtype=F)
        d = torch.tensor([0.1, -0.2], dtype=F)
        whole = response.measure(lambda x: x.square() - 2, (w,), (d,))
        pieces = response.measure(lambda x: tuple(x.square() - 2), (w,), (d,))
        self.assertEqual(whole, pieces)

    def test_rescaling_direction_changes_length_not_the_proposed_weights(self):
        w = torch.tensor([0.], dtype=F)
        d = torch.tensor([1.], dtype=F)
        a = response.measure(lambda x: 3 * x - 6, (w,), (d,))
        b = response.measure(lambda x: 3 * x - 6, (w,), (7 * d,))
        assert a.length is not None and b.length is not None
        torch.testing.assert_close(a.length * d, b.length * 7 * d)

    def test_nonzero_residual_with_zero_response_is_a_stall_not_closure(self):
        w = torch.tensor([0.], dtype=F)
        step = response.measure(lambda x: x * 0 + 3, (w,), (torch.ones_like(w),))
        self.assertEqual(step.objective, 9)
        self.assertIsNone(step.length)
        self.assertEqual(step.status, 'zero_response')

    def test_zero_residual_does_not_claim_native_closure(self):
        w = torch.tensor([0.], dtype=F)
        step = response.measure(lambda x: x, (w,), (torch.ones_like(w),))
        self.assertIsNone(step.length)
        self.assertEqual(step.status, 'zero_residual')
        self.assertFalse(hasattr(step, 'closed'))

    def test_uphill_direction_has_no_positive_proposal(self):
        w = torch.tensor([2.], dtype=F)
        step = response.measure(lambda x: x, (w,), (torch.ones_like(w),))
        self.assertEqual(step.status, 'non_descent')
        self.assertIsNone(step.length)

    def test_no_gradient_mutation_or_weight_change(self):
        w = torch.tensor([2.], dtype=F, requires_grad=True)
        w.grad = torch.tensor([7.], dtype=F)
        response.measure(lambda x: x.square() - 9, (w,), (-torch.ones_like(w),))
        self.assertEqual(w.item(), 2)
        self.assertEqual(w.grad.item(), 7)

    def test_empty_nonfinite_and_mismatched_responses_are_refused(self):
        w = torch.tensor([2.], dtype=F)
        for bad in (math.nan, math.inf):
            with self.assertRaisesRegex(ValueError, 'finite'):
                response.measure(lambda x: x * bad, (w,), (torch.ones_like(w),))
        with self.assertRaisesRegex(ValueError, 'shape'):
            response.measure(lambda x: x, (w,), (torch.ones(2, dtype=F),))
        with self.assertRaisesRegex(ValueError, 'empty'):
            response.measure(lambda x: x[:0], (w,), (torch.ones_like(w),))


if __name__ == '__main__':
    unittest.main()
