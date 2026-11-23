"""Physical Gemma4 parameter inventory and CPU-staged, shared embedding leaves.

Integration contract: inventory(tensors, architecture, history) classifies EVERY
stored tensor. ParameterModel(source, history, device=..., shared_linear=...)
accepts the *unwrapped* existing torch_forward.Model (normally loaded on CPU).
It retains no source model or dense embedding table. Eligible dense leaves and
compact table leaves live on CPU as F32; weight()/rows() transfer only operands
of the current operator, preserving reverse and forward AD through transfers.
Wrap this model with the existing FFN/projection precision adapters BEFORE
constructing full_graph.Graph. The default shared_linear uses CPU BF16-rounded
operands and F32 products, matching the source operand policy; native reduction
equivalence is not claimed.

Native storage owner consumes physical names and parameter_rows[table] (sorted
unique IDs, optionally unioned with state-required dormant rows). A table leaf
has (len(row_ids), table_width) shape; repeated positions gather the SAME leaf row. Other rows have zero conditional derivative,
not independent parameters or token targets. No installer/optimizer is defined.

The retained model metadata imply 7,508,417,672 bytes of dense F32 leaves;
leaves plus one direction occupy 15,016,835,344 CPU bytes, sharing the source's
promoted dense storage. Largest block operands are about 280 MB and 612-position
native anchors about 790 MB, before workspaces/activations. The inspected machine
has two 16 GB GPUs and 128 GB RAM (~67 GB available at preparation), so retaining
all dense leaves/directions on a 16 GB GPU is not viable. CPU staging is the
implemented policy; these are payload bounds, not a measured GPU peak.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
import operator
from types import MethodType

import torch

from . import host_pages

TABLES = ('token_embd.weight', 'per_layer_token_embd.weight')
GLOBAL = ('rope_freqs.weight', *TABLES, 'output_norm.weight',
          'per_layer_model_proj.weight', 'per_layer_proj_norm.weight')
PARTS = ('attn_norm', 'layer_output_scale', 'ffn_down', 'ffn_gate', 'ffn_up',
         'inp_gate', 'post_attention_norm', 'post_ffw_norm', 'post_norm', 'proj',
         'ffn_norm', 'attn_k_norm', 'attn_output', 'attn_q_norm', 'attn_k',
         'attn_q', 'attn_v')
FORWARD_METHODS = frozenset(('head', 'keys_of', 'rotation', 'masked', 'keys',
                             'attention', 'feed_forward', 'third', 'block', 'forward'))

CPU_POLICY_SOURCES = (
    'dependencies/llama.cpp-f089c9c/src/models/gemma4.cpp',
    'dependencies/llama.cpp-f089c9c/ggml/src/ggml-cpu/ggml-cpu.c',
    'dependencies/llama.cpp-f089c9c/ggml/src/ggml-cpu/vec.cpp',
    'dependencies/llama.cpp-f089c9c/ggml/src/ggml-cpu/llamafile/sgemm.cpp',
    'dependencies/llama.cpp-f089c9c/ggml/src/ggml-impl.h',
)


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    shape: tuple[int, ...]  # complete physical tensor in Torch/output-major order
    native_type: str
    offset_bytes: int
    bytes: int
    eligible: bool
    reason: str
    row_ids: tuple[int, ...] | None = None
    aliases: tuple[str, ...] = ()  # roles sharing storage, never extra leaves

    @property
    def leaf_shape(self):
        return (len(self.row_ids), self.shape[1]) if self.row_ids is not None else self.shape


def _input_ids(values):
    try:
        return tuple(operator.index(value) for value in values)
    except TypeError as error:
        raise ValueError('table/input IDs must be integers, never rounded selectors') from error


def inventory(tensors, architecture, history=None, *, row_ids=None):
    """Account for the exact physical schema; unknown/missing tensors are errors.

    The601/538 counts emerge from the real35-block metadata, not an allowlist
    of70 projections. Smaller same-architecture fixtures are supported.
    """
    layers = architecture['layers']
    producers = layers - architecture['layers_sharing_keys']
    if layers < 2 or not 2 <= producers <= layers or len(architecture['windowed']) != layers:
        raise ValueError('invalid Gemma4 block/cache inventory')
    expected = set(GLOBAL) | {f'blk.{layer}.{part}.weight'
                             for layer in range(layers) for part in PARTS}
    if set(tensors) != expected:
        raise ValueError(f'physical tensor inventory differs: missing={sorted(expected-set(tensors))}, extra={sorted(set(tensors)-expected)}')
    ids = None if history is None else tuple(sorted(set(_input_ids(history))))
    if ids == () or ids is not None and ids[0] < 0:
        raise ValueError('embedding history must contain nonnegative input IDs')
    row_ids = {} if row_ids is None else row_ids
    if set(row_ids) - set(TABLES):
        raise ValueError('row overrides may name only physical embedding tables')
    rows = []
    for name, tensor in tensors.items():
        shape = tuple(tensor['shape'])
        kind = tensor['native_type']
        if (len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape)
                or kind not in ('bf16', 'f32') or tensor['offset_bytes'] < 0
                or tensor['bytes'] != math.prod(shape) * (2 if kind == 'bf16' else 4)):
            raise ValueError(f'invalid native tensor metadata: {name}')
        resident = tuple(n for n in reversed(shape) if n > 1) or (1,)
        eligible, reason = True, 'continuous measured-merge dependency'
        if name == 'rope_freqs.weight':
            eligible, reason = False, 'fixed configuration-derived proportional-RoPE buffer'
        elif name in ('output_norm.weight', f'blk.{layers-1}.layer_output_scale.weight'):
            eligible, reason = False, 'after last measured merge; no held-history loss path'
        elif name.startswith('blk.'):
            _, layer, part, _ = name.split('.')
            if int(layer) >= producers and part in ('attn_k', 'attn_v', 'attn_k_norm'):
                eligible, reason = False, 'unused stored late-KV tensor; actual cache is reused'
        selected = _input_ids(row_ids.get(name, ids)) if name in TABLES and (name in row_ids or ids is not None) else None
        if selected is not None and (not selected or selected != tuple(sorted(set(selected)))
                or selected[0] < 0 or ids is not None and not set(ids) <= set(selected)):
            raise ValueError('table rows must be sorted/distinct and include the full current history')
        if name in TABLES and (len(resident) != 2 or selected is not None and selected[-1] >= resident[0]):
            raise ValueError(f'embedding IDs/table shape differ: {name}')
        rows.append(ParameterSpec(name, resident, kind, tensor['offset_bytes'], tensor['bytes'],
                                  eligible, reason, selected,
                                  ('output.weight',) if name == TABLES[0] else ()))
    return tuple(rows)


def memory_plan(specs, *, events, width, layers):
    """Payload accounting, not a measured peak or promise of GPU qualification.

    Keep leaves and VJP directions on CPU. Checkpointed reverse computation stages
    one layer's operands; forward JVP has no reverse tape. Native anchors remain
    resident. Temporary workspaces, activations and allocator overhead are extra.
    """
    eligible = [row for row in specs if row.eligible]
    if any(row.name in TABLES and row.row_ids is None for row in eligible):
        raise ValueError('bind current history before estimating compact table memory')
    dense = sum(math.prod(row.shape) * 4 for row in eligible if row.name not in TABLES)
    tables = sum(math.prod(row.leaf_shape) * 4 for row in eligible if row.name in TABLES)
    blocks = [sum(math.prod(row.shape) * 4 for row in eligible
                  if row.name.startswith(f'blk.{layer}.')) for layer in range(layers)]
    return dict(dense_f32_leaf_bytes=dense, compact_embedding_leaf_bytes=tables,
                cpu_leaf_and_direction_bytes=2*(dense+tables),
                largest_block_operand_f32_bytes=max(blocks),
                native_anchor_bytes=events*width*4*3*layers*2,
                policy='CPU F32 leaves/directions; per-operator device transfers; layer checkpointing; sparse shared table rows',
                limitation='Payload counts only; native model, captures, activations, gradients during recompute and library workspaces are additional. No native/GPU peak measured.')


def cpu_bf16_round(value):
    """GGML CPU nearest-even conversion, preserving BF16 subnormals/sign zero.

    ggml-impl.h:611–623 adds0x7fff plus the retained parity bit. Unlike the
    historical GPU storage helper, this CPU operand conversion does not flush
    subnormals. Solver inputs/results must remain finite.
    """
    if value.device.type != 'cpu' or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
        raise ValueError('CPU BF16 operands require finite CPU F32 values')
    bits = value.detach().contiguous().view(torch.int32).to(torch.int64) & 0xffffffff
    codes = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    rounded = codes.to(torch.int32).view(torch.float32)
    if not bool(torch.isfinite(rounded).all()):
        raise ValueError('CPU BF16 operand rounding overflow')
    return rounded


def shared_calls_from_index(path, *, events, width):
    """Actual input-layer CPU call positions, not token/word eligibility."""
    import json
    groups = []
    with open(path) as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('label') != 'inp_scaled':
                continue
            positions = tuple(row['positions'])
            if (row['native_type'] != 'f32' or row['shape'] != [width,len(positions),1,1]
                    or row['microbatch_count'] != len(positions) or not positions):
                raise ValueError('shared projection input geometry differs')
            groups.append(positions)
    if sorted(position for group in groups for position in group) != list(range(events)):
        raise ValueError('shared projection geometry must cover each event once')
    return tuple(groups)


class _CpuSharedProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, groups):
        if x.dtype != torch.float32 or weight.dtype != torch.float32 or x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
            raise ValueError('shared projection requires compatible F32 matrices')
        xq, wq = cpu_bf16_round(x.cpu()), cpu_bf16_round(weight.cpu())
        ctx.save_for_backward(xq,wq)
        ctx.save_for_forward(xq,wq)
        ctx.x_device,ctx.w_device,ctx.groups = x.device,weight.device,groups
        return _cpu_grouped_product(xq,wq,groups).to(x.device)

    @staticmethod
    def backward(ctx, adjoint):
        xq,wq = ctx.saved_tensors
        g = adjoint.cpu()
        return ((g @ wq).to(ctx.x_device) if ctx.needs_input_grad[0] else None,
                (g.T @ xq).to(ctx.w_device) if ctx.needs_input_grad[1] else None,None)

    @staticmethod
    def jvp(ctx, dx, dw, _groups):
        xq,wq = ctx.saved_tensors
        out = None if dx is None else _cpu_grouped_product(dx.cpu(),wq,ctx.groups)
        if dw is not None:
            term = _cpu_grouped_product(xq,dw.cpu(),ctx.groups)
            out = term if out is None else out+term
        return None if out is None else out.to(ctx.x_device)


def _cpu_grouped_product(x, weight, groups):
    if groups is None:
        return x @ weight.T
    order = [position for group in groups for position in group]
    if not groups or any(not group for group in groups) or sorted(order) != list(range(len(x))):
        raise ValueError('shared projection groups must cover each event once')
    values = [x[list(group)] @ weight.T for group in groups]
    return torch.cat(values)[torch.argsort(torch.tensor(order))]


def cpu_shared_linear(x, weight, *, groups=None):
    """CPU BF16/BF16 operand policy; explicit F32 VJP/JVP at rounded operands.

    Native ggml-cpu.c BF16 traits select BF16 vec_dot_type. Both generic dot and
    llamafile BF16 GEMM consume the converted BF16 input (including1-column
    calls), returning F32. We compute decoded-rounded operands with CPU Torch
    F32 GEMM and preserve supplied inp_scaled call groups. SIMD/tinyBLAS/BLAS
    accumulation order is NOT reproduced or qualified by this implementation.
    Supply groups=shared_calls_from_index(...) for the actual production trace.
    """
    return _CpuSharedProduct.apply(x,weight,groups)


class _DeviceWeights(Mapping):
    """Read-only operation view; never cache a candidate/dual device operand."""
    def __init__(self, model):
        self.model = model

    def __getitem__(self, name):
        # The host copy may live in a file (host_pages); once on the device its pages go back.
        value = self.model.weights[name]
        host_pages.prefetch(value)
        try:
            return value.to(device=self.model.device, dtype=torch.float32)
        finally:
            host_pages.release(value)

    def __iter__(self):
        return iter(self.model.weights)

    def __len__(self):
        return len(self.model.weights)


class ParameterModel:
    def __init__(self, source, history, *, device=None, shared_linear=cpu_shared_linear, row_ids=None):
        """row_ids optionally binds history∪previously touched rows per table.

        Dormant rows remain candidate/dual leaves with zero current derivative;
        state can still carry/install their existing remainder without deferral.
        """
        if source.device.type != 'cpu' or any(not callable(getattr(type(source), name, None))
                for name in (*FORWARD_METHODS, 'rows', 'per_layer_input')):
            raise ValueError('load the unwrapped torch_forward.Model on CPU before binding full parameters')
        # Bind operations as unbound methods to THIS object; source.weight may be
        # a resident-model closure over another mapping and must never be copied.
        self._model_class = type(source)
        for key, value in vars(source).items():
            if key not in {'w', 'leaves', 'weight', 'rows', 'kept', 'per_layer_input', *FORWARD_METHODS}:
                setattr(self, key, value)
        self.device = torch.device(source.device if device is None else device)
        self.history = _input_ids(history)
        self.specs = inventory(source.tensors, source.a, self.history, row_ids=row_ids)
        self.parameter_rows = {row.name: row.row_ids for row in self.specs if row.name in TABLES}
        self._eligible = tuple(row.name for row in self.specs if row.eligible)
        self.shared_linear = shared_linear
        self.shared_projection_policy = 'CPU BF16 operands/F32 product with explicit STE; actual inp_scaled groups supplied by caller; finite native reduction unqualified'
        self.weights = {}
        for row in self.specs:
            if row.reason.startswith('unused stored'):
                continue
            value = source.rows(row.name, row.row_ids) if row.name in TABLES else source.w[row.name]
            value = value.detach().to(device='cpu', dtype=torch.float32)
            if row.name not in TABLES:
                # Promote once and share persistent CPU storage with the source;
                # its native-install callback updates these same dense values.
                source.w[row.name] = value
            if tuple(value.shape) != row.leaf_shape:
                raise ValueError(f'parameter leaf shape differs: {row.name}')
            self.weights[row.name] = value.requires_grad_(row.eligible)
        self._view = _DeviceWeights(self)
        self._row_index = {name: {token: i for i, token in enumerate(ids)}
                           for name, ids in self.parameter_rows.items()}

    @property
    def parameter_model(self):
        return self

    @property
    def w(self):
        return self._view

    @w.setter
    def w(self, values):
        self.set_weights(values)

    @property
    def leaves(self):
        return {name: self.weights[name] for name in self._eligible}

    def set_weights(self, values):
        if set(values) != set(self.weights):
            raise ValueError('candidate mapping must preserve all learned and fixed entries')
        for name, value in values.items():
            previous = self.weights[name]
            if value.shape != previous.shape or value.dtype != torch.float32 or value.device.type != 'cpu':
                raise ValueError(f'candidate leaf must retain CPU F32 layout: {name}')
        self.weights = values  # preserve mapping identity for unconditional restoration

    def weight(self, name):
        return self.w[name]

    def rows(self, name, tokens):
        if name not in self.parameter_rows:
            raise ValueError('only the two physical shared embedding tables support rows')
        try:
            indices = [self._row_index[name][token] for token in _input_ids(tokens)]
        except KeyError as error:
            raise ValueError('candidate table rows belong to the bound current history') from error
        index = torch.tensor(indices, dtype=torch.long, device='cpu')
        try:
            return self.weights[name].index_select(0, index).to(self.device)
        finally:
            host_pages.release(self.weights[name])

    def per_layer_input(self, x, tokens):
        # Preserve original scaling/order; expose the previously constant shared
        # projection through an explicit operator boundary for native qualification.
        rms = self._model_class.per_layer_input.__globals__['rms']
        own = self.rows(TABLES[1], tokens).view(len(tokens), self.layers, self.per) * math.sqrt(self.per)
        shared = (self.shared_linear(x, self.weights['per_layer_model_proj.weight'])
                  * (1/math.sqrt(self.width))).view(len(tokens), self.layers, self.per)
        host_pages.release(self.weights['per_layer_model_proj.weight'])
        return (rms(shared, self.w['per_layer_proj_norm.weight']) + own) * (1/math.sqrt(2))

    def __getattr__(self, name):
        if name in FORWARD_METHODS:
            return MethodType(getattr(self._model_class, name), self)
        raise AttributeError(name)
