"""A native-conditioned graph reproduces every observed merge without cutting gradients."""
import math
import tempfile
import unittest

import torch

from pipeline.weight_edit import local_affine, model_local_affine, model_trace_response, native_graph
from tests.test_model_trace_response import fixture


def rounded_trace(model, history):
    """A tiny test engine with explicit BF16 merge rounding, not real model data."""
    with torch.no_grad():
        positions = torch.arange(len(history), device=model.device)
        x = model.rows('token_embd.weight', history) * math.sqrt(model.width)
        per = model.per_layer_input(x, history)
        incoming, merged, cache = [], [], {}
        for layer in range(model.layers):
            if layer < model.keys_from:
                cache[layer] = model.keys(layer, x, positions)
            k, v = cache[model.keys_of(layer)]
            incoming.append(x)
            a = (x + model.attention(layer, x, positions, k, v)).bfloat16().float()
            incoming.append(a)
            b = (a + model.feed_forward(layer, a)).bfloat16().float()
            incoming.append(b)
            c = (b + model.third(layer, b, per[:, layer])).bfloat16().float()
            merged.extend((a, b, c))
            x = c * model.w[f'blk.{layer}.layer_output_scale.weight']
        return dict(incoming=incoming, merged=merged)


class NativeGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/tmp')
        file, tensors = fixture.made(self.tmp.name)
        self.model = fixture.tf.Model(file, tensors, fixture.ARCHITECTURE, 'cpu', cache='float16')
        self.history = fixture.TEXT
        self.native = rounded_trace(self.model, self.history)
        self.graph = native_graph.Graph(self.model, self.history, self.native)

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_merge_is_anchored_to_its_native_observation(self):
        forward = self.graph.forward(self.history)
        for name in ('incoming', 'merged'):
            for actual, expected in zip(forward[name], self.native[name]):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        functions = model_local_affine.conditional_merges(self.graph, self.history, forward)
        for merge, expected in zip(functions, self.native['merged']):
            actual = local_affine.action(merge.function, (merge.query,), (merge.query,))
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_offsets_are_fixed_after_weights_change_not_recomputed_to_hide_movement(self):
        name = next(name for name in self.model.w if name.endswith('.ffn_down.weight'))
        original = self.model.w[name].detach().clone()
        with torch.no_grad():
            self.model.w[name].add_(0.1)
        changed = self.graph.forward(self.history)
        self.assertGreater(float((changed['merged'][1] - self.native['merged'][1]).abs().max().detach()), 0)
        with torch.no_grad():
            self.model.w[name].copy_(original)

    def test_joint_mixed_derivative_matches_reverse_gradient_on_anchored_graph(self):
        names = tuple(name for name, weight in self.graph.w.items() if weight.requires_grad)
        forward = self.graph.forward(self.history)
        at = model_local_affine.conditional_merges(self.graph, self.history, forward)
        probes = [merge.query.detach() + 0.2 for merge in at]
        def residual(current):
            return tuple(local_affine.action(merge.function, (merge.query,), (probe,))
                         for merge, probe in zip(model_local_affine.conditional_merges(self.graph, self.history, current), probes))
        raw = residual(forward)
        objective = torch.stack([x.double().square().sum() for x in raw]).sum()
        gradients = torch.autograd.grad(objective, tuple(self.graph.w[name] for name in names))
        direction = dict(zip(names, (-g for g in gradients)))
        step = model_trace_response.measure_model(self.graph, names, self.history, direction, residual)
        expected = -sum(float(g.double().square().sum()) for g in gradients)
        torch.testing.assert_close(torch.tensor(step.slope), torch.tensor(expected))
        self.assertEqual(step.status, 'proposal')

    def test_trace_of_another_history_or_missing_merge_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'history'):
            self.graph.forward(list(reversed(self.history)))
        with self.assertRaisesRegex(ValueError, 'every merge'):
            native_graph.Graph(self.model, self.history,
                               dict(incoming=self.native['incoming'][:-1], merged=self.native['merged'][:-1]))


if __name__ == '__main__':
    unittest.main()
