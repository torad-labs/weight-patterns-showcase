"""carried() with the steps inside a block fused: the solver's own IEEE operations on every element, in the solver's own order, launched as three
kernels a coordinate instead of about thirty, and a block's 128 coordinates captured as one CUDA graph.

What is the same as tube.carried (tube2_build.py, sha256 602fd692...): every operation on every element and the order they are taken in.
  - a row is rounded as bf16() rounds it (to float32, then the engine's arithmetic on the bits) and what is left over is the row less that;
  - the leftover's amount along the size direction is the library's own sum of the same 1536 products (torch.sum on the same kind of tensor);
  - what a coordinate leaves over is taken off the block's later rows in the same two steps, the sized part first;
  - the products that carry a block's leftovers on to all later rows are the solver's own two expressions, on the same matrices.
What differs: how many kernels are launched for it, and that a value used twice may be computed twice (the same operation on the same operands
gives the same bits). The kernels write each operation with the device's own single operations (__dmul_rn, __dsub_rn, __ddiv_rn), which the
compiler never merges into a fused multiply-add: the library compiles these kernels with merging allowed, and a merged a - b * c rounds once where
the solver's separate kernels round twice.

Ways to run it, all meant to write the same matrix:
  kernels='jit'    the kernels below through torch.cuda.jiterator (a card is needed)
  kernels='plain'  the same steps as ordinary tensor operations, on any device: it is what checks the rearrangement of the loop on a CPU
  lean=True        three launches a coordinate (row, the library's sum, carried-on); False: five (the leftover and the amount as kernels of their own)
  capture=True     a whole block of BLOCK coordinates replayed as one CUDA graph; False: launched one by one
  tail=True        the two subtractions that carry a block's leftovers on to the later rows as one kernel; False: the solver's two statements
"""
import torch

# bf16(x.float()).double(): the nearest float32 (ties to even), then tube.bf16's arithmetic on its bits, in 64-bit integers as there
FILE_VALUE = r"""
template <typename T> T file_value(T x) {
    float narrow = (float)x;
    unsigned long long bits = (unsigned long long)__float_as_uint(narrow);
    unsigned long long kept = ((bits + 0x7fffULL + ((bits >> 16) & 1ULL)) >> 16) & 0xffffULL;
    kept = (((bits & 0x7f800000ULL) == 0ULL) ? ((bits & 0x80000000ULL) >> 16) : kept) << 16;
    return (T)__uint_as_float((unsigned int)kept);
}
"""
KERNELS = dict(
    # out[i] = bf16(left[i].float()) ; over = left[i] - out[i].double() ; over * toward (summed by the library outside)
    row=(3, FILE_VALUE + r"""
template <typename T> void written_row(T x, T toward, T& rounded, T& over, T& along) {
    rounded = file_value<T>(x);
    over = __dsub_rn(x, rounded);
    along = __dmul_rn(over, toward);
}
"""),
    row_plain=(2, FILE_VALUE + r"""
template <typename T> void written_row_plain(T x, T& rounded, T& over) {
    rounded = file_value<T>(x);
    over = __dsub_rn(x, rounded);
}
"""),
    # (over - amount * toward) / upper[i, i]
    across=(1, r"""
template <typename T> T across_row(T over, T amount, T toward, T diagonal) {
    return __ddiv_rn(__dsub_rn(over, __dmul_rn(amount, toward)), diagonal);
}
"""),
    # over / upper[i, i]
    across_plain=(1, r"""
template <typename T> T across_row_plain(T over, T diagonal) {
    return __ddiv_rn(over, diagonal);
}
"""),
    # left -= (upper_sized[i, j] * amounts[i]) * toward ; then left -= upper[i, j] * across[i]
    on=(1, r"""
template <typename T> T carried_on(T left, T sized_row, T amount_over, T toward, T upper_row, T across) {
    return __dsub_rn(__dsub_rn(left, __dmul_rn(__dmul_rn(sized_row, amount_over), toward)), __dmul_rn(upper_row, across));
}
"""),
    on_plain=(1, r"""
template <typename T> T carried_on_plain(T left, T upper_row, T across) {
    return __dsub_rn(left, __dmul_rn(upper_row, across));
}
"""),
    # the same two subtractions with amounts[i] = amount / upper_sized[i, i] and across[i] = (over - amount * toward) / upper[i, i] taken inside
    on_lean=(1, r"""
template <typename T> T carried_on_lean(T left, T sized_row, T amount, T sized_diagonal, T toward, T upper_row, T over, T diagonal) {
    T amount_over = __ddiv_rn(amount, sized_diagonal);
    T across = __ddiv_rn(__dsub_rn(over, __dmul_rn(amount, toward)), diagonal);
    return __dsub_rn(__dsub_rn(left, __dmul_rn(__dmul_rn(sized_row, amount_over), toward)), __dmul_rn(upper_row, across));
}
"""),
    on_plain_lean=(1, r"""
template <typename T> T carried_on_plain_lean(T left, T upper_row, T over, T diagonal) {
    return __dsub_rn(left, __dmul_rn(upper_row, __ddiv_rn(over, diagonal)));
}
"""),
    # left[last:] -= product ; then left[last:] -= sized_amounts[:, None] * toward[None, :]
    tail=(1, r"""
template <typename T> T carried_tail(T left, T product, T sized_amount, T toward) {
    return __dsub_rn(__dsub_rn(left, product), __dmul_rn(sized_amount, toward));
}
"""),
)
_JIT = {}


def jit():
    """the kernels, compiled the first time each is called (the library keeps them in memory and on disk after that)"""
    if not _JIT:
        from torch.cuda.jiterator import _create_jit_fn, _create_multi_output_jit_fn
        for name, (outputs, code) in KERNELS.items():
            _JIT[name] = _create_jit_fn(code) if outputs == 1 else _create_multi_output_jit_fn(code, num_outputs=outputs)
    return _JIT


def plain(bf16):
    """the same steps as ordinary tensor operations, each in the solver's order (any device); bf16 is the solver's own rounding"""
    def row(x, toward):
        rounded = bf16(x.float()).double()
        over = x - rounded
        return rounded, over, over * toward

    def row_plain(x):
        rounded = bf16(x.float()).double()
        return rounded, x - rounded

    return dict(
        row=row, row_plain=row_plain,
        across=lambda over, amount, toward, diagonal: (over - amount * toward) / diagonal,
        across_plain=lambda over, diagonal: over / diagonal,
        on=lambda left, sized_row, amount_over, toward, upper_row, across: (left - (sized_row * amount_over) * toward) - upper_row * across,
        on_plain=lambda left, upper_row, across: left - upper_row * across,
        on_lean=lambda left, sized_row, amount, sized_diagonal, toward, upper_row, over, diagonal:
            (left - (sized_row * (amount / sized_diagonal)) * toward) - upper_row * ((over - amount * toward) / diagonal),
        on_plain_lean=lambda left, upper_row, over, diagonal: left - upper_row * (over / diagonal),
        tail=lambda left, product, sized_amount, toward: (left - product) - sized_amount * toward)


def block(ops, rows, upper, upper_sized, toward, lean):
    """One block of coordinates. rows (B, m): the block's rows as they stand with every earlier block's leftovers carried onto them; upper and
    upper_sized: the block's own corner of the factors (B, B); toward (m,) or None. Returns what is written (doubles holding the file's values),
    each coordinate's leftover across the size direction over its own diagonal (B, m), and its amount along that direction over the sized diagonal
    (B,), None without a size direction."""
    count, sized = rows.shape[0], toward is not None
    rest, written, overs, summed, across, amounts = rows, [], [], [], [], []
    for i in range(count):
        more = i + 1 < count
        if sized:
            rounded, over, along = ops['row'](rest[0], toward)
            amount = along.sum()
            if lean:
                overs.append(over)
                summed.append(amount)
                if more:
                    rest = ops['on_lean'](rest[1:], upper_sized[i, i + 1:, None], amount, upper_sized[i, i], toward[None, :], upper[i, i + 1:, None], over[None, :], upper[i, i])
            else:
                amount_over = amount / upper_sized[i, i]
                leftover = ops['across'](over, amount, toward, upper[i, i])
                amounts.append(amount_over)
                across.append(leftover)
                if more:
                    rest = ops['on'](rest[1:], upper_sized[i, i + 1:, None], amount_over, toward[None, :], upper[i, i + 1:, None], leftover[None, :])
        else:
            rounded, over = ops['row_plain'](rest[0])
            if lean:
                overs.append(over)
                if more:
                    rest = ops['on_plain_lean'](rest[1:], upper[i, i + 1:, None], over[None, :], upper[i, i])
            else:
                leftover = ops['across_plain'](over, upper[i, i])
                across.append(leftover)
                if more:
                    rest = ops['on_plain'](rest[1:], upper[i, i + 1:, None], leftover[None, :])
        written.append(rounded)
    if lean:   # every coordinate's leftover and amount at once: the same operations on the same operands as inside the kernel
        diagonal = upper.diagonal()[:, None]
        if sized:
            amount = torch.stack(summed)
            return torch.stack(written), ops['across'](torch.stack(overs), amount[:, None], toward[None, :], diagonal), amount / upper_sized.diagonal()
        return torch.stack(written), ops['across_plain'](torch.stack(overs), diagonal), None
    return torch.stack(written), torch.stack(across), torch.stack(amounts) if sized else None


class Captured:
    """A block of exactly B coordinates and m outputs captured once as a CUDA graph: fixed buffers in, fixed buffers out, the same kernels replayed."""

    def __init__(self, ops, B, m, sized, lean, device):
        f = dict(dtype=torch.float64, device=device)
        self.rows, self.upper = torch.zeros((B, m), **f), torch.eye(B, **f)
        self.upper_sized, self.toward = (torch.eye(B, **f), torch.zeros(m, **f)) if sized else (None, None)
        self.written, self.across = torch.zeros((B, m), **f), torch.zeros((B, m), **f)
        self.amounts = torch.zeros(B, **f) if sized else None
        self.ops, self.lean = ops, lean
        side = torch.cuda.Stream(device=device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            for _ in range(3):   # the library asks for warm-up runs before a capture; every kernel a block uses is compiled here
                self.body()
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.body()
        self.pins = [(each, each.data_ptr()) for each in (self.rows, self.upper, self.upper_sized, self.toward, self.written, self.across, self.amounts) if each is not None]

    def body(self):
        written, across, amounts = block(self.ops, self.rows, self.upper, self.upper_sized, self.toward, self.lean)
        self.written.copy_(written)
        self.across.copy_(across)
        if amounts is not None:
            self.amounts.copy_(amounts)

    def stands(self):
        """whether the buffers are where they were when the graph was captured (a graph replays on fixed addresses: a buffer that was moved off
        the card and back is another one)"""
        return all(each.is_cuda and each.data_ptr() == address for each, address in self.pins)


CAPTURED = {}


def forget():
    """Drop every captured block and the card memory its graph keeps. To be called before the card's tensors are parked (tube.Card.given): the
    graph's own memory is not a tensor that can be moved off the card, and its buffers would come back at other addresses."""
    CAPTURED.clear()


def carried(target, measure, bf16=None, BLOCK=128, kernels='jit', capture=True, lean=True, tail=True, single_products=False):
    """tube.carried(target, measure) by the fused steps. bf16: the solver's own rounding (the plain steps use it); BLOCK: the solver's.

    single_products is NOT the solver's arithmetic and is off unless asked for: the product that carries a block's leftovers on to all later rows
    taken in single precision (the factor rounded to float32 once a call), everything else as it is. It is here to read how many weights that
    would write differently and what it would save, which is for the solver's owner to weigh; nothing uses it."""
    upper, along = measure['upper'], measure['along']
    toward, upper_sized = along if along is not None else (None, None)
    sized = toward is not None
    ops = jit() if kernels == 'jit' else plain(bf16)
    rest = target.double() if tail else target.double().clone()   # the rows not written yet, every earlier block's leftovers carried onto them
    single = upper.float() if single_products else None
    out = torch.empty_like(target)
    count, m = target.shape
    for first in range(0, count, BLOCK):
        last = min(first + BLOCK, count)
        size = last - first
        corner = upper[first:last, first:last]
        corner_sized = upper_sized[first:last, first:last] if sized else None
        if capture and size == BLOCK:
            key = (kernels, lean, BLOCK, m, sized, rest.device)
            if key not in CAPTURED or not CAPTURED[key].stands():
                CAPTURED.pop(key, None)
                CAPTURED[key] = Captured(ops, BLOCK, m, sized, lean, rest.device)
            c = CAPTURED[key]
            c.rows.copy_(rest[:size])
            c.upper.copy_(corner)
            if sized:
                c.upper_sized.copy_(corner_sized)
                c.toward.copy_(toward)
            c.graph.replay()
            written, across, amounts = c.written, c.across, c.amounts
        else:
            written, across, amounts = block(ops, rest[:size], corner, corner_sized, toward, lean)
        out[first:last].copy_(written)   # doubles holding float32 values: the copy is exact
        if last == count:
            break
        if single is None:
            product = upper[first:last, last:].T @ across                              # the solver's own expression
        else:
            product = (single[first:last, last:].T @ across.float()).double()
        if tail:
            if sized:
                rest = ops['tail'](rest[size:], product, (upper_sized[first:last, last:].T @ amounts)[:, None], toward[None, :])
            else:
                rest = rest[size:] - product
        else:   # the solver's own two statements
            rest = rest[size:]
            rest -= product
            if sized:
                rest -= (upper_sized[first:last, last:].T @ amounts)[:, None] * toward[None, :]
    return out
