"""The streamed derivative hands over exactly the gradients the one-shot derivative returns."""
from types import SimpleNamespace
import unittest

import torch

from pipeline.weight_edit import full_graph


def fixture():
    torch.manual_seed(5)
    weights = {name: torch.randn(*shape) for name, shape in
               (('a', (6, 5)), ('b', (5,)), ('unused', (3, 3)), ('c', (4, 6)))}
    graph = SimpleNamespace(w=weights, leaves=tuple(weights), parameter_rows={},
        forward=lambda history: (lambda w: torch.tanh(w['c'] @ w['a']) @ w['b'] + 0 * w['unused'].detach().sum())(graph.w))
    # the 'unused' leaf never enters the differentiable graph
    return graph


def objective(forward):
    return forward.square().sum(), {}


class StreamedDerivative(unittest.TestCase):
    def test_same_gradients_diagnostics_and_unreached_leaf(self):
        whole = full_graph._derivative(fixture(), (), objective)
        graph, kept, order = fixture(), {}, []

        def sink(name, gradient):
            order.append(name)
            kept[name] = -gradient if name not in kept else kept[name] - gradient
        streamed = full_graph._derivative(graph, (), objective, sink=sink)
        self.assertEqual(set(kept), {'a', 'b', 'c'})
        self.assertEqual(streamed[0], {})
        for name in kept:
            self.assertTrue(torch.equal(kept[name], whole[0][name]), name)
        self.assertEqual(streamed[3]['unused']['gradient_was_none'], True)
        self.assertEqual(streamed[2], whole[2])
        self.assertEqual(streamed[3], whole[3])
        self.assertEqual(streamed[1], whole[1])
        self.assertTrue(all(w.grad is None for w in graph.w.values()))
        self.assertEqual(len(order), 3)


if __name__ == '__main__':
    unittest.main()
