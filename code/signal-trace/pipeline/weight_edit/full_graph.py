"""Complete current-history learned-parameter graph, without loss/state policy.

Create ParameterModel from the current unwrapped CPU torch_forward.Model, wrap
it with the unchanged precision adapters, then Graph(wrapped, history, native).
Graph.w is the RAW CPU candidate mapping (unlike operation-side model.w, which
stages operands). Existing model_trace_response.measure_model therefore works
with all538 native-name leaves, including compact table row arrays. Root owns
reference/tolerance/controller; full_state owns conversion/installation. This
module never generates tokens, installs weights, chooses eta or edits a model.
"""
from dataclasses import dataclass
import math
import time

import torch

from .native_graph import Graph as NativeGraph
from . import model_trace_response


class Graph(NativeGraph):
    def __init__(self, model, history, native):
        self.parameter_model = model.parameter_model
        if tuple(history) != self.parameter_model.history:
            raise ValueError('parameter rows and native graph must bind the same history')
        self.parameter_rows = self.parameter_model.parameter_rows
        self.parameter_specs = self.parameter_model.specs
        super().__init__(model, history, native)

    @property
    def w(self):
        return self.parameter_model.weights

    @w.setter
    def w(self, values):
        self.parameter_model.set_weights(values)

    @property
    def leaves(self):
        return self.parameter_model.leaves


@dataclass
class Gradient:
    direction: dict[str, torch.Tensor]
    before: dict
    norm_squared: float
    seconds: float
    diagnostics: dict[str, dict]


def _derivative(graph, history, objective, *, sink=None):
    """One same-state VJP, preserving None versus a connected numerical zero.

    None signals disconnected/possibly detached, not proof of valid inactivity.
    Structural exclusions live in ParameterSpec; no eligible None is silently
    endorsed. Returned zero directions keep the full inventory inspectable, with
    diagnostics for the controller to reject/investigate before installation.

    With ``sink`` each leaf's gradient is handed over the moment backward has
    finished it, as ``sink(name, gradient)`` with the raw (un-negated) gradient,
    then freed. The full gradient set is never alive at once, so host memory
    follows the largest tensor rather than the parameter count. The returned
    direction is then empty; a leaf the graph never reached is reported with
    ``gradient_was_none`` and never sent to the sink.
    """
    original = graph.w
    names = tuple(graph.leaves)
    weights = tuple(original[name].detach().requires_grad_(True) for name in names)
    delivered = {}
    handles = []
    try:
        with torch.enable_grad():
            graph.w = original | dict(zip(names, weights))
            scalar, before = objective(graph.forward(history))
            if scalar.ndim != 0 or not bool(torch.isfinite(scalar)):
                raise ValueError('the complete objective must be a finite tensor scalar')
            if sink is None:
                gradients = (torch.autograd.grad(scalar, weights, allow_unused=True)
                             if scalar.requires_grad else (None,)*len(weights))
            else:
                for name, weight in zip(names, weights):
                    handles.append(weight.register_post_accumulate_grad_hook(
                        _delivery(name, delivered, sink)))
                if scalar.requires_grad:
                    torch.autograd.backward(scalar, inputs=list(weights))
                gradients = (None,)*len(weights)
    finally:
        for handle in handles:
            handle.remove()
        graph.w = original
    direction, diagnostics, norms = {}, {}, []
    for name, weight, gradient in zip(names, weights, gradients):
        if name in delivered:
            report = delivered[name]
            value = None
        elif gradient is None:
            value, report = torch.zeros_like(weight), dict(
                status='disconnected_or_detached', gradient_was_none=True, norm_squared=0., nonzero_entries=0)
            if sink is not None:
                value = None
        else:
            value = gradient.detach()
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'nonfinite derivative: {name}')
            nonzero = int(torch.count_nonzero(value))
            report = dict(status='connected_nonzero' if nonzero else 'connected_zero', gradient_was_none=False,
                          norm_squared=float(value.double().square().sum()), nonzero_entries=nonzero)
        norms.append(report['norm_squared'])
        diagnostics[name] = dict(status=report['status'], gradient_was_none=report['gradient_was_none'],
            norm_squared=report['norm_squared'], nonzero_entries=report['nonzero_entries'], finite=True,
            shape=list(weight.shape), device=str(weight.device),
            row_ids=list(graph.parameter_rows[name]) if name in graph.parameter_rows else None)
        if value is not None:
            direction[name] = value.neg_()
    total = math.fsum(norms)
    if not math.isfinite(total):
        raise ValueError('nonfinite full gradient norm')
    return direction, before, total, diagnostics


def _delivery(name, delivered, sink):
    def hand_over(weight):
        value = weight.grad
        weight.grad = None
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'nonfinite derivative: {name}')
        nonzero = int(torch.count_nonzero(value))
        delivered[name] = dict(status='connected_nonzero' if nonzero else 'connected_zero',
            gradient_was_none=False, norm_squared=float(value.double().square().sum()), nonzero_entries=nonzero)
        sink(name, value)
    return hand_over


def weight_gradient(graph, history, reference, *, pairs):
    """Same signature/core reduction as trace_gradient; retain route diagnostics.

    The supplied reference owns all105 weighted residuals and fixed tolerance.
    There are no additional examples, targets, normalization or token judges.
    """
    started = time.monotonic()
    def objective(forward):
        terms, outside = [], 0
        for observed in reference.measure(forward, pairs):
            terms.append(observed.residual.double().square().sum())
            outside += int(observed.outside.sum())
        if len(terms) != reference.merges or not terms:
            raise ValueError('the full objective must account for every declared merge')
        values = [float(term.detach()) for term in terms]
        scalar = torch.stack(terms).sum()
        return scalar, dict(objective=math.fsum(values), outside=outside,
                            pair_merge_cells=len(pairs)*reference.merges)
    direction, before, norm, diagnostics = _derivative(graph, history, objective)
    return Gradient(direction, before, norm, time.monotonic()-started, diagnostics)


def directional_response(graph, history, direction, residual):
    """One existing full residual JVP, with outer-call unconditional restoration."""
    return model_trace_response.measure_model(graph, tuple(graph.leaves), history, direction, residual)
