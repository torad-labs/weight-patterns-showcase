"""What each merge of the original model adds to the recurring difference between its wrong and its correct answers, beside what the merge
inherits and what a changing comparison point makes of it. Astra's specification of November 12, 2026, reading A
(records/retrospectives/2026-11-12-inherited-and-added-astra.md), on the held-member recurrence reading of 460c3ebb
(records/development/2026-11-12-correction-objective/README.md). The original's recordings only: no model is read through the engine, no weights
are solved, and nothing is fitted, selected, generated or made a target, a mask or a threshold.

From code/signal-trace, on a card under its lock:
  python -m pipeline.weight_edit.contributions map|deals <run>

The states. The model's residual stream goes through three merges a layer, each an addition (pipeline/acquisition/export_stream_traces.py,
`connections`): attention (the layer's input plus the attention's normed and scaled output), feedforward (that plus the feed-forward's) and
layer (that plus the per-layer embedding's part). The layer's output is its last merge times one positive number: no addition, it turns nothing
and multiplies every term of the next merge's incoming side alike. At a merge, row by row (a row is one answer step): h- the state it adds to,
a what it adds, h+ = h- + a.
  attention, feedforward   h- is the recorded state (input-overlap's inputs). a is rebuilt from the recorded input of the matrix, the matrix and the
                           learned scale, as the engine multiplied it (ladder.Merge, tube.multiplied): g * y / rms(y). The contribution itself was
                           not kept when the sixty answers were recorded.
  layer                    h- is the feed-forward's h+ as rebuilt. h+ is the next layer's recorded input over the layer's output scale; at the
                           last layer no next input is on record, and h+ is each row's recorded direction (the recorded direction after attention
                           plus the two recorded turns that follow it) times its recorded size. a = h+ - h-: the per-layer embedding's part was
                           not kept as a field, and is what the two states leave.
Each answer's rows are laid on the 480 places by the frozen registration and averaged at a place with the registration's own weights
(tube.Layout.means), the same for h-, a and h+, before anything else: H-, A, H+ = H- + A, members x places x 1536. No state is made from a mean
direction and a mean log size.

The steps read. The 105 merges in the model's order, and after them one step that is no merge: `output`, the model's output norm on the last
merge's state (correction.passed_on), which the frozen reference reads the last merge through. It adds nothing to a state: only its features
are read, before and after it, and it has no raw terms. The last merge itself is read here on its own state, like every other merge.

The comparison point. For a correct answer j the reference is the other twenty-nine correct answers, as in 460c3ebb. A tested answer t (each
wrong answer, and j itself) is compared with the mixture of the twenty-nine nearest to it in the joint features: the registered unit state and
log size, |direction|^2 + (w_size / w_direction) size^2 with the frozen reference's own weights w_f = 0.5 / S_f, no radius (mixture.joined).
alpha+ are the coefficients found on the step's OUTGOING features (tube.recorded; the size as the engine recorded it). alpha- are the
coefficients of the INCOMING features and are held fixed through the step: the incoming state of a step is the outgoing state of the step
before it (times the layer's output scale between two layers, which changes no direction and moves every log size alike), so alpha- of a step
are alpha+ of the step before; for the model's first merge they are found on the features of the embedding's rows, the state layer 0's
attention adds to.

The identity, in the registered raw states (vectors, 1536 wide; i runs over the twenty-nine):
  incoming     e- = H-[t] - sum_i alpha-_i H-[i]
  added        D  = A[t]  - sum_i alpha-_i A[i]            what the merge adds to t, less what it adds to the comparison held fixed
  comparison   C  = - sum_i (alpha+_i - alpha-_i) H+[i]    what the change of the nearest mixture makes of it: no contribution of the merge
  outgoing     e+ = H+[t] - sum_i alpha+_i H+[i]  =  e- + D + C
These are vectors: their lengths do not add, and the squares are kept with all their cross products. Beside them, and named apart, the same
split of the FEATURES (direction, size and joint): incoming r-, `through` (the features' change through the step at the comparison held fixed),
comparison, outgoing r+ (at every merge but the last, and at `output`, the leftover of 460c3ebb). A feature split is no raw contribution.

The readings, for every pair of a wrong answer i and a correct answer j, with the exclusions of 460c3ebb: the builders are the wrong answers
other than i, other than a wrong answer to i's case and other than a wrong answer to j's case; neither i nor j helped to make the reference or
the direction they are read against.
  on the outgoing direction   common = the sum of the builders' e+ (one direction for the pair, held fixed). a_term = <term[i], common> / |common|
                              and b_term = <term[j], common> / |common| for each of the four terms: a_outgoing = a_incoming + a_added +
                              a_comparison exactly, and likewise b. This splits what a left-out answer has of the recurring state difference
                              into what it came in with, what the merge added and what the comparison's change made.
  each term's own             common = the sum of the builders' same term: does the added part recur by itself in a left-out wrong answer
                              and not in a left-out correct one (and the incoming part, the comparison's, the outgoing).
  the features                the same four-way split read on the builders' r+, in the direction, in the size and jointly.
Kept per step and place for each reading: the share of the 900 pairs with a over b (`ordered`; 0.5 if the groups did not differ), the shares
resolved either way, the means of a and b, of a / |term[i]| and b / |term[j]|, of the lengths, and the same shares over the 46 pairs whose
questions have the same m, v and drop; per answer, the raw terms' means and how sure they are; for the four pairs that are one case, the terms
themselves.

How sure. A search ends with a bound on what it leaves of its cost over the least cost. The mixture found then lies within
e = sqrt(bound / w_direction) of the true nearest one in the joint features (the mixture is the nearest point of a convex set). A raw term is a
product of the coefficients with other vectors than those features, so its error is bounded through the coefficients: for a difference d of
two sets of coefficients (it sums to nothing), |M'd| <= kappa |Z'd| with kappa(M | Z) the largest of |M'd| / |Z'd| over such d, found from the
two Gram matrices (`kappa`); where the members' features do not determine the coefficients kappa is infinite and the term is undetermined.
incoming: e- kappa(H- | Z-); added: e- kappa(A | Z-); comparison: e+ kappa(H+ | Z+) + e- kappa(H+ | Z-); outgoing: e+ kappa(H+ | Z+). The
feature terms likewise (incoming e-; through: e- kappa(F+ - F- | Z-); comparison: e+ + e- kappa(F+ | Z-); outgoing e+). A pair is resolved
where a and b lie farther apart than these allow, the turn of the common direction included (`compared`); otherwise it is unresolved, which is
neither agreement nor the lack of a difference. kappa is taken over every difference of coefficients, also those a mixture's nonnegative
coefficients could not make: it can overstate, never understate.

`deals`: the same reading with the sixty answers dealt into two thirties at random (seeds 1 to 12), at the merges 460c3ebb's deals read (and
`output`): what the same statistics give on these recordings when the split means nothing. An observed level, not a threshold.

Data roles. All sixty registered original answers are read, the thirty no-fit ones among them, as the frozen reference reads them. Nothing here
is a fitting, a selection or an evaluation; reserved-40b is not read. A merge's response to different inputs is also what it adds: no label of
input against error follows from the identity."""
import json
import os
import re
import sys
import time

import numpy as np
import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ACQUISITION, ROUND_TWO_RUNS
from pipeline.weight_edit import correction, ladder, mixture, tube
from pipeline.weight_edit.tube import DEVICE, TINY

RESOLVED = mixture.FINE ** 0.5   # where a search ends: the summed bound over the summed cost, as in 460c3ebb
TOGETHER = 5                     # how many correct answers' references are searched in one call, as in 460c3ebb: what the card holds
RAW = ('incoming', 'added', 'comparison', 'outgoing')
FEATURE = ('incoming', 'through', 'comparison', 'outgoing')
FORMS = ('direction', 'size', 'joint')
NUMBERS = ('m', 'v', 'drop', 'loss', 'rise')
SQUARES = [(a, b) for k, a in enumerate(RAW) for b in RAW[k:]]   # the ten products of two raw terms
DEALS = (0, 51, 104, 105)        # layer 0's attention, layer 17's attention and the model's last merge, which 460c3ebb's deals read, and `output`
SEEDS = tuple(range(1, 13))
INF = float('inf')


def stages_of(layers):
    """The steps read, in the model's order: three merges a layer, and the output norm after the last."""
    return [(layer, kind) for layer in layers for kind in tube.MERGES] + [(layers[-1], 'output')]


def compared(tb, cb, bb, tt, cc, unsure_t, unsure_c, unsure_b, left):
    """One reading for one correct answer j and every wrong answer i (W of them), place by place.
      tb  places x W x W   <the tested vector of wrong answer i, the builders' vector of wrong answer x>
      cb  places x W       <the tested vector of the correct answer j, the builders' vector of wrong answer x>
      bb  places x W x W   the builders' vectors among themselves
      tt, cc               places x W, places: the squared lengths of the tested vectors
      unsure_t, unsure_c, unsure_b   places x W, places, places x W: how far each vector may lie from the true one
      left  W x W          which wrong answers x are left out of the builders for wrong answer i
    Returns a and b (places x W: what the tested vectors have along the builders' sum, one direction a pair), the tested vectors' lengths, and
    where the pair is resolved: a and b farther apart than the vectors' own uncertainties and the turn of the common direction allow (a unit
    vector turns by no more than twice the change of its vector over its length)."""
    keep = 1 - left
    inner_a, inner_b = torch.einsum('pix,ix->pi', tb, keep), cb @ keep.T
    length = torch.einsum('ix,pxy,iy->pi', keep, bb, keep).clamp(min=0).sqrt()   # |common|
    there = length > 0
    a, b = (torch.where(there, inner / length.clamp(min=TINY), torch.zeros_like(inner)) for inner in (inner_a, inner_b))
    long_a, long_b = tt.clamp(min=0).sqrt(), cc.clamp(min=0).sqrt()[:, None].expand(-1, tt.shape[1])
    summed = unsure_b @ keep.T   # the builders' uncertainties
    turn = 2 * summed / length.clamp(min=TINY)
    known = there & (length > summed)
    within = unsure_t + long_a * turn + unsure_c[:, None] + long_b * turn
    return a, b, long_a, long_b, known & ((a - b).abs() > within)


def zero_sum(count, **kind):
    """count x (count - 1): an orthonormal basis of the differences of two sets of coefficients, which sum to nothing."""
    return torch.linalg.qr(torch.cat([torch.ones((count, 1), **kind), torch.eye(count, **kind)[:, 1:]], 1)).Q[:, 1:]


def rooted(under, basis):
    """The Cholesky factor of the members' Gram matrix (problems x members x members) on the differences of coefficients, and where it exists:
    where it does not, the members' vectors do not determine a difference of coefficients. The factor is kept on the CPU: these are many small
    matrices, and the card's batched routines ask for gigabytes of working space for them."""
    factor, failed = torch.linalg.cholesky_ex((basis.T @ under @ basis).cpu())
    fine = failed == 0
    return torch.where(fine[:, None, None], factor, torch.eye(factor.shape[-1], dtype=factor.dtype).expand_as(factor)), fine


def kappa(of, factor, fine, basis):
    """The largest of |M'd| / |Z'd| over the differences d of coefficients: `of` is the members' Gram matrix under M (problems x members x
    members), `factor` and `fine` are `rooted` of their Gram matrix under Z. Infinite where Z does not determine d. Returned where `of` is."""
    half = torch.linalg.solve_triangular(factor, (basis.T @ of @ basis).cpu(), upper=False)
    whole = torch.linalg.solve_triangular(factor, half.transpose(1, 2), upper=False)
    most = torch.linalg.eigvalsh((whole + whole.transpose(1, 2)) / 2)[:, -1].clamp(min=0).sqrt()
    return torch.where(fine, most, torch.full_like(most, INF)).to(of.device)


def term_rows(before, after):
    """`before` and `after`: places x tested x answers, each tested vector as its own answer less the mixture it is compared with, by the
    incoming coefficients and by the outgoing ones (every row sums to nothing). Returns each raw term as a row over [H- | A], and each
    feature term as a row over [the incoming features | the outgoing features] (the module's identity)."""
    zero, moved = torch.zeros_like(before), after - before
    raw = dict(incoming=torch.cat([before, zero], 2), added=torch.cat([zero, before], 2), comparison=torch.cat([moved, moved], 2), outgoing=torch.cat([after, after], 2))
    feature = dict(incoming=torch.cat([before, zero], 2), through=torch.cat([-before, before], 2), comparison=torch.cat([zero, moved], 2), outgoing=torch.cat([zero, after], 2))
    return raw, feature


def answer_rows(index, total, what, layer, kind):
    """One of input-overlap's recordings at every answer step: a matrix's input (`x`) or the state its merge adds to (`state`)."""
    rows = np.memmap(ladder.INPUTS/f'{what}-{kind}-{layer}.f32', dtype='<f4', mode='r').reshape(total, -1)
    if what == 'x' and rows.shape[1] != index['width'][f'{kind}-{layer}']:
        raise ValueError(f'{what}-{kind}-{layer}: the stored rows are not the listed answers\' steps')
    return torch.from_numpy(np.array(rows)).to(DEVICE)


def output_scale(tensors, layer):
    """The one positive number a layer's last merge is multiplied by to give the layer's output."""
    scale = tube.checkpoint_values(tube.ORIGINAL_MODEL, tensors[f'blk.{layer}.layer_output_scale.weight']).astype('f8').ravel()
    if scale.size != 1 or not scale[0] > 0:
        raise ValueError(f'the output of layer {layer} is not its last merge times one positive number')
    return float(scale[0])


def raw_of(answers, index, tensors, layer, kind):
    """Row by row at every answer step: the state a merge adds to and what it adds; what each is made from; what was checked on the way; and
    the number the state it adds to is the step before's outgoing state times (the module's `states`)."""
    steps = [m['steps'] for m in answers]
    total, first, last, checks = sum(steps), index['layers'][0], index['layers'][-1], {}

    def matrix(which):   # a matrix's merge: the recorded state it adds to, and what it adds as the engine made it
        w, g = ladder.weights_of(tensors, layer, which)[1:]
        state = answer_rows(index, total, 'state', layer, which)
        merge = ladder.Merge(tube.multiplied(answer_rows(index, total, 'x', layer, which), steps, 'as generated'), state, w, g)
        engine = tube.native_log(answers, layer, which)[:, 0].exp()
        checks[f'{which}_rebuilt_size_from_the_engines'] = float((merge.size[:, 0].double() / engine - 1).abs().max())
        return state, merge.g * merge.sent

    if kind == 'attention':
        before, added = matrix('attention')
        recorded = answer_rows(index, total, 'state', layer, 'feedforward')   # the same state as the engine recorded it: what the feed-forward adds to
        checks['rebuilt_outgoing_from_the_recorded'] = float(((before + added - recorded).norm(dim=1) / recorded.norm(dim=1)).max())
        told = dict(incoming='recorded', added='rebuilt from the recorded matrix input', outgoing='incoming plus added')
        return before, added, told, checks, output_scale(tensors, layer - 1) if layer != first else 1.0
    if kind == 'feedforward':
        before, added = matrix('feedforward')
        return before, added, dict(incoming='recorded', added='rebuilt from the recorded matrix input', outgoing='incoming plus added'), checks, 1.0
    state, added = matrix('feedforward')
    before = state + added   # the state the layer's last merge adds to: the feed-forward's merge as rebuilt
    del state, added
    engine = tube.native_log(answers, layer, 'layer')
    told = dict(incoming='the feed-forward merge as rebuilt', added='not kept as a field: outgoing less incoming')
    if layer != last:
        scale = output_scale(tensors, layer)
        after = answer_rows(index, total, 'state', layer + 1, 'attention') / scale
        told.update(outgoing='the next layer\'s recorded input over the layer\'s output scale', layer_output_scale=scale)
    else:
        rebuilt = lambda at: tube.turned_to(answers, at) * tube.native_log(answers, at, 'layer').float().exp()
        after = rebuilt(layer)
        told.update(outgoing='not on record as a state: each row\'s recorded direction (after attention plus the two recorded turns) times its recorded size')
        known = answer_rows(index, total, 'state', layer, 'attention') / output_scale(tensors, layer - 1)   # the layer before, where the state is on record
        checks['the_same_rebuild_one_layer_earlier_from_the_recorded_state'] = float(((rebuilt(layer - 1) - known).norm(dim=1) / known.norm(dim=1)).max())
        del known
    checks['outgoing_size_from_the_engines'] = float((after.norm(dim=1).double() / engine[:, 0].exp() - 1).abs().max())
    return before, after - before, told, checks, 1.0


def mapped(run, folder, chosen, seed=None, members=True, say=print):
    """The module's reading at the steps `chosen` (their numbers in `stages_of`), each written to <folder>/<kind>-<layer>.npz with its line in
    <folder>/log.jsonl. A step before a chosen one is searched for its coefficients and nothing of it is kept. With `seed`, the sixty answers
    are dealt into two thirties at random in the place of the wrong and the correct ones."""
    declaration = json.loads((run/'declaration.json').read_text())
    reference = tube.ROOT/declaration['correction_reference']
    if tube.digest(reference/'index.json') != declaration['correction_reference_index_sha256']:
        raise ValueError('the correction reference is not the one the run declares')
    tensors = json.loads(ladder.TENSORS.read_text())
    scale = json.loads((reference/'index.json').read_text())['scale']
    weigh = {form: 0.5 / scale[form] for form in correction.FORMS}
    ratio = weigh['size'] / weigh['direction']   # the joint reading in the direction's units
    _, answers, index = ladder.cohort(run)
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    pairs = [np.load(ladder.CORRESPONDENCE/'pairs'/f"{m['id']}.npy").astype(np.int64) for m in answers]
    correct = np.array([m['correct'] for m in answers])
    layout = tube.Layout(~correct, [m['steps'] for m in answers], [m['question'] for m in answers], pairs, places)
    total, last = sum(m['steps'] for m in answers), index['layers'][-1]
    norm = correction.terminal(tensors, last)
    stages = stages_of(index['layers'])

    ids = [m['id'] for m in answers]
    case = [re.sub(r'(-channel)?-h\d$', '', each) for each in ids]
    split = np.random.default_rng(seed).permutation(correct) if seed is not None else correct   # which answers are read as the correct ones
    wrong_at, right_at = np.nonzero(~split)[0], np.nonzero(split)[0]
    W, R = len(wrong_at), len(right_at)
    K = R - 1
    double = dict(dtype=torch.float64, device=DEVICE)
    same = torch.tensor([[case[a] == case[b] for b in wrong_at] for a in wrong_at], **double)   # wrong x wrong: itself and a wrong answer to its case
    kin = torch.tensor([[case[a] == case[b] for b in wrong_at] for a in right_at], **double)    # correct x wrong: a wrong answer to the correct answer's case
    order = torch.from_numpy(np.concatenate([wrong_at, right_at])).to(DEVICE)
    others = [torch.tensor([k for k in range(R) if k != j], device=DEVICE) for j in range(R)]
    every = torch.stack(others)   # R x K
    basis = zero_sum(K, **double)
    asked = [json.loads(next(ACQUISITION.glob(f'*/{each}/case.json')).read_text()) for each in ids]
    if not all(list(one['tokens']) == list(m['tokens']) for one, m in zip(asked, answers)):
        raise ValueError('a case read is not the question its answer was recorded on')
    numbers = np.array([[one['parameters'][name] for name in NUMBERS] for one in asked])
    matched = torch.tensor([[bool((numbers[a, :3] == numbers[b, :3]).all()) for b in right_at] for a in wrong_at], **double)   # wrong x correct: the same m, v and drop
    one_case = [(a, b) for a in range(W) for b in range(R) if case[wrong_at[a]] == case[right_at[b]]]

    def searched(gram, across, far, sizes, size_far, start, widest):
        """Problems x members x members and the rest a problem each: the nearest mixtures with no radius (mixture.joined), from `start`."""
        count, k = gram.shape[:2]
        weight, size_weight = (torch.full((count,), weigh[form], **double) for form in correction.FORMS)
        nothing = torch.zeros(count, **double)
        steep = 2 * (weight * widest + size_weight * (sizes ** 2).sum(1))   # the largest eigenvalue of all thirty's products bounds any twenty-nine's
        return mixture.joined(gram, across, far, sizes, size_far, nothing, nothing, weight, size_weight, steep, start, RESOLVED, k ** 2)

    def nearest(gamma, s):
        """The coefficients of a step's features (gamma: places x 60 x 60, the directions' products; s: places x 60, the sizes; the wrong
        answers first, both counted from the correct answers' mean): every wrong answer against each correct answer's other twenty-nine, and
        every correct answer against its own other twenty-nine, with each search's bound, and what every call of the search did. The calls
        are 460c3ebb's, in its order."""
        right, widest = gamma[:, W:, W:], torch.linalg.eigvalsh(gamma[:, W:, W:])[:, -1].clamp(min=0)
        to_wrong, far_wrong, size_wrong, size_right = gamma[:, :W, W:].transpose(0, 1), gamma.diagonal(dim1=1, dim2=2)[:, :W].T, s[:, :W].T, s[:, W:]
        calls = []
        told = lambda name, found: calls.append(dict(search=name, problems=int(found['cost'].numel()), loops=int(found['loops']), loops_allowed=int(found['alpha'].shape[-1]) ** 2,
                                                     bound=float(found['bound'].sum()), cost=float(found['cost'].sum())))
        # every wrong answer against all thirty correct answers: only where the searches below start from
        began_at = searched(right[None].expand(W, -1, -1, -1).reshape(-1, R, R), to_wrong.reshape(-1, R), far_wrong.reshape(-1), size_right[None].expand(W, -1, -1).reshape(-1, R), size_wrong.reshape(-1),
                            torch.full((W * places, R), 1 / R, **double), widest.repeat(W))
        told('every wrong answer against all thirty correct ones (a start only)', began_at)
        began_at = began_at['alpha'].view(W, places, R)
        gram = torch.stack([right[:, o][:, :, o] for o in others])   # correct x places x 29 x 29
        held = torch.stack([size_right[:, o] for o in others])       # correct x places x 29
        own = searched(gram.view(-1, K, K), torch.stack([right[:, j, o] for j, o in enumerate(others)]).view(-1, K), right.diagonal(dim1=1, dim2=2).T.reshape(-1), held.view(-1, K), size_right.T.reshape(-1),
                       torch.full((R * places, K), 1 / K, **double), widest.repeat(R))
        told('every correct answer against the other twenty-nine', own)
        alpha, bound, cost = torch.zeros((R, W, places, K), **double), torch.zeros((R, W, places), **double), torch.zeros((R, W, places), **double)
        for first in range(0, R, TOGETHER):
            group = list(range(first, min(first + TOGETHER, R)))
            starts = [began_at[:, :, others[j]] for j in group]
            starts = [torch.where(each.sum(-1, keepdim=True) > 0, each / each.sum(-1, keepdim=True).clamp(min=TINY), torch.full_like(each, 1 / K)) for each in starts]
            found = searched(torch.cat([gram[j][None].expand(W, -1, -1, -1).reshape(-1, K, K) for j in group]), torch.cat([to_wrong[:, :, others[j]].reshape(-1, K) for j in group]), far_wrong.reshape(-1).repeat(len(group)),
                             torch.cat([held[j][None].expand(W, -1, -1).reshape(-1, K) for j in group]), size_wrong.reshape(-1).repeat(len(group)), torch.cat([each.reshape(-1, K) for each in starts]), widest.repeat(W * len(group)))
            told(f'every wrong answer against the other twenty-nine of correct answers {group[0]} to {group[-1]}', found)
            alpha[group], bound[group], cost[group] = found['alpha'].view(len(group), W, places, K), found['bound'].view(len(group), W, places), found['cost'].view(len(group), W, places)
            del found
        return dict(alpha=alpha, bound=bound.clamp(min=0), cost=cost, own_alpha=own['alpha'].view(R, places, K), own_bound=own['bound'].view(R, places).clamp(min=0), calls=calls)

    def features(n):
        """A step's outgoing features as the frozen reference reads them, laid out places x 60 (the wrong answers first) and counted from
        the correct answers' mean; n = -1 is the state the model's first merge adds to (the embedding's rows)."""
        if n < 0:
            to, log = tube.unit_and_log(answer_rows(index, total, 'state', index['layers'][0], 'attention'))
        else:
            layer, kind = stages[n]
            merge = 'layer' if kind == 'output' else kind
            w, g = ladder.weights_of(tensors, layer, merge)[1:] if merge in tube.KINDS else (None, None)
            to, log, _, _ = tube.recorded(answers, index, layer, merge, w, g, tensors)
            if merge in tube.KINDS:
                log = tube.native_log(answers, layer, merge)
            if kind == 'output':
                to, log = correction.passed_on(to, log, **norm)
        z, sizes = layout.means(to), layout.means(log)[..., 0]
        del to, log
        y, s = z[order].transpose(0, 1), sizes[order].T
        return y - y[:, W:].mean(1, keepdim=True), s - s[:, W:].mean(1, keepdim=True)

    def parts(alpha, own, j):   # each tested vector as its answer less a mixture of j's reference: places x (W + 1) x 60, the wrong answers and then j
        out = torch.zeros((places, W + 1, W + R), **double)
        out[:, :W, :W] = torch.eye(W, **double)
        out[:, :W, W + others[j]] = -alpha.transpose(0, 1)
        out[:, W, W + j] = 1
        out[:, W, W + others[j]] = -own
        return out

    restricted = lambda gram: gram[:, W:, W:][:, every[:, :, None], every[:, None, :]].transpose(0, 1).reshape(-1, K, K)   # the correct answers' block, for each j its other twenty-nine: (R x places) x K x K
    joint = lambda g, sizes: g + ratio * sizes[:, :, None] * sizes[:, None, :]
    cells = ('a', 'b', 'long_a', 'long_b', 'resolved')
    needed = sorted(set(chosen) | {n - 1 for n in chosen})
    folder.mkdir(parents=True, exist_ok=True)
    before, stood, began = None, None, time.time()   # the step before: its features and coefficients; and what its outgoing terms measured
    for n in needed:
        y, s = features(n)
        gamma = y @ y.transpose(1, 2)
        now = dict(nearest(gamma, s), n=n, y=y, s=s, gamma=gamma)
        if n not in chosen:
            before, stood = now, None
            continue
        if before is None or before['n'] != n - 1:
            raise ValueError(f'step {n} is read without the coefficients of the step before it')
        layer, kind = stages[n]
        additive = kind != 'output'
        names = ([f'raw_{term}' for term in RAW] + [f'raw_own_{term}' for term in RAW] if additive else []) + [f'feature_{form}_{term}' for form in FORMS for term in FEATURE]
        mixed = before['y'] @ y.transpose(1, 2)
        gamma2 = torch.cat([torch.cat([before['gamma'], mixed], 2), torch.cat([mixed.transpose(1, 2), gamma], 2)], 1)   # places x 120 x 120 over [the incoming features | the outgoing]
        s2 = torch.cat([before['s'], s], 1)
        # how far a term can lie from the true one for each part of a search's bound: the module's kappa, for each j and place
        z_in, z_out = joint(before['gamma'], before['s']), joint(gamma, s)
        root_in, root_out = rooted(restricted(z_in), basis), rooted(restricted(z_out), basis)
        through = joint(gamma - mixed - mixed.transpose(1, 2) + before['gamma'], s - before['s'])
        reach = {name: kappa(restricted(of), *root, basis).view(R, places) for name, of, root in (('through|Z-', through, root_in), ('F+|Z-', z_out, root_in))}
        del through, z_in, z_out
        source, checks, came_times, G = None, None, 1.0, None
        if additive:
            low, added, source, checks, came_times = raw_of(answers, index, tensors, layer, kind)
            H, A = (layout.means(each)[order].transpose(0, 1) for each in (low, added))   # places x 60 x 1536
            del low, added
            H, A = H - H[:, W:].mean(1, keepdim=True), A - A[:, W:].mean(1, keepdim=True)   # every term's row sums to nothing in each block: this changes no term
            cross = H @ A.transpose(1, 2)
            G = torch.cat([torch.cat([H @ H.transpose(1, 2), cross], 2), torch.cat([cross.transpose(1, 2), A @ A.transpose(1, 2)], 2)], 1)   # places x 120 x 120 over [H- | A]
            del H, A, cross
            g_out = G[:, :W + R, :W + R] + G[:, :W + R, W + R:] + G[:, W + R:, :W + R] + G[:, W + R:, W + R:]   # H+ among themselves
            reach.update({name: kappa(restricted(of), *root, basis).view(R, places) for name, of, root in (
                ('H-|Z-', G[:, :W + R, :W + R], root_in), ('A|Z-', G[:, W + R:, W + R:], root_in), ('H+|Z-', g_out, root_in), ('H+|Z+', g_out, root_out))})
            del g_out
        del root_in, root_out
        torch.cuda.empty_cache()
        kept = {name: {cell: torch.zeros((places, W, R), **double) for cell in cells} for name in names}
        squares = {group: {pair: torch.zeros((places, R), **double) for pair in SQUARES} for group in ('wrong', 'correct')}
        unsure_kept = {group: {term: torch.zeros((places, R), **double) for term in RAW} for group in ('wrong', 'correct')}
        apart, off = 0.0, 0.0
        for j in range(R):
            rows, feature_rows = term_rows(parts(before['alpha'][j], before['own_alpha'][j], j), parts(now['alpha'][j], now['own_alpha'][j], j))
            e_in = (torch.cat([before['bound'][j].T, before['own_bound'][j][:, None]], 1) / weigh['direction']).sqrt()   # places x (W + 1)
            e_out = (torch.cat([now['bound'][j].T, now['own_bound'][j][:, None]], 1) / weigh['direction']).sqrt()
            by = lambda e, name: torch.where(torch.isinf(reach[name][j])[:, None], torch.full_like(e, INF), e * reach[name][j][:, None])
            left = (same + kin[j][None]).clamp(max=1)

            def read(name, tested_with_builders, tested, builders, sure_tested, sure_builders):
                """One reading into `kept`: the products are places x (W + 1) x (W + 1), the wrong answers and then j."""
                found = compared(tested_with_builders[:, :W, :W], tested_with_builders[:, W, :W], builders[:, :W, :W], tested.diagonal(dim1=1, dim2=2)[:, :W], tested[:, W, W],
                                 sure_tested[:, :W], sure_tested[:, W], sure_builders[:, :W], left)
                for cell, value in zip(cells, found):
                    kept[name][cell][:, :, j] = value

            if additive:   # the raw terms
                unsure = dict(incoming=by(e_in, 'H-|Z-'), added=by(e_in, 'A|Z-'), comparison=by(e_out, 'H+|Z+') + by(e_in, 'H+|Z-'), outgoing=by(e_out, 'H+|Z+'))
                carried = {term: rows[term] @ G for term in RAW}
                among = lambda a, b: carried[a] @ rows[b].transpose(1, 2)
                own = {term: among(term, term) for term in RAW}
                for term in RAW:
                    read(f'raw_{term}', own[term] if term == 'outgoing' else among(term, 'outgoing'), own[term], own['outgoing'], unsure[term], unsure['outgoing'])
                    read(f'raw_own_{term}', own[term], own[term], own[term], unsure[term], unsure[term])
                    unsure_kept['wrong'][term][:, j], unsure_kept['correct'][term][:, j] = unsure[term][:, :W].mean(1), unsure[term][:, W]
                product = {}
                for a, b in SQUARES:
                    product[a, b] = own[a].diagonal(dim1=1, dim2=2) if a == b else (carried[a] * rows[b]).sum(2)   # places x (W + 1): <term a, term b> of each tested answer
                    squares['wrong'][a, b][:, j], squares['correct'][a, b][:, j] = product[a, b][:, :W].mean(1), product[a, b][:, W]
                three = [(a, b) for a, b in SQUARES if 'outgoing' not in (a, b)]
                closes = product['outgoing', 'outgoing'] - sum((1 if a == b else 2) * product[a, b] for a, b in three)   # |e+|^2 less the squares and cross products of its three parts
                off = max(off, float((closes.abs() / sum((1 if a == b else 2) * product[a, b].abs() for a, b in three).clamp(min=TINY)).max()))
                del carried, own, product
            # the feature terms, in the direction, in the size and jointly
            unsure_f = dict(incoming=e_in, through=by(e_in, 'through|Z-'), comparison=e_out + by(e_in, 'F+|Z-'), outgoing=e_out)
            carried = {term: feature_rows[term] @ gamma2 for term in FEATURE}
            sized = {term: (feature_rows[term] @ s2[:, :, None])[..., 0] for term in FEATURE}
            formed = dict(direction=lambda a, b: carried[a] @ feature_rows[b].transpose(1, 2), size=lambda a, b: sized[a][:, :, None] * sized[b][:, None, :])
            formed['joint'] = lambda a, b: formed['direction'](a, b) + ratio * formed['size'](a, b)
            for form in FORMS:
                sure = {term: value / ratio ** 0.5 if form == 'size' else value for term, value in unsure_f.items()}
                builders = formed[form]('outgoing', 'outgoing')
                for term in FEATURE:
                    read(f'feature_{form}_{term}', builders if term == 'outgoing' else formed[form](term, 'outgoing'), builders if term == 'outgoing' else formed[form](term, term), builders,
                         sure[term], sure['outgoing'])
                if form == 'joint':
                    apart = max(apart, float((weigh['direction'] * builders.diagonal(dim1=1, dim2=2)[:, :W].T - now['cost'][j]).abs().max() / now['cost'][j].mean()))   # the leftovers as laid here against the search's own cost
            del carried, sized, rows, feature_rows
        arrays = {}
        for name in names:
            a, b, long_a, long_b, resolved = (kept[name][cell] for cell in cells)
            values = dict(ordered=(a > b).double() + 0.5 * (a == b).double(), over=resolved * (a > b), under=resolved * (a < b), wrong_agreement=a / long_a.clamp(min=TINY), correct_agreement=b / long_b.clamp(min=TINY),
                          wrong_along=a, correct_along=b, wrong_length=long_a, correct_length=long_b)
            for cell, value in values.items():
                arrays[f'{name}_{cell}'] = value.mean((1, 2)).float().cpu().numpy()
            for cell in ('ordered', 'over', 'under'):
                arrays[f'{name}_matched_{cell}'] = ((values[cell] * matched).sum((1, 2)) / matched.sum()).float().cpu().numpy()
        for name, value in reach.items():   # over the references, place by place: the median of those determined, and the share not determined
            finite = torch.isfinite(value)
            arrays[f'reach_{name}'] = torch.where(finite, value, torch.zeros_like(value)).median(0).values.float().cpu().numpy()
            arrays[f'reach_{name}_undetermined'] = (~finite).double().mean(0).float().cpu().numpy()
        chain = None
        if additive:
            for group in ('wrong', 'correct'):
                for a, b in SQUARES:
                    arrays[f'raw_{group}_product_{a}_{b}'] = squares[group][a, b].mean(1).float().cpu().numpy()
                for term in RAW:
                    finite = torch.isfinite(unsure_kept[group][term])
                    arrays[f'raw_{group}_unsure_{term}'] = (torch.where(finite, unsure_kept[group][term], torch.zeros_like(unsure_kept[group][term])).sum(1) / finite.sum(1).clamp(min=1)).float().cpu().numpy()
                    arrays[f'raw_{group}_undetermined_{term}'] = (~finite).double().mean(1).float().cpu().numpy()   # the share of references for which the features do not determine the coefficients
            for term in RAW:
                a, b, resolved = (kept[f'raw_{term}'][cell] for cell in ('a', 'b', 'resolved'))
                picked = lambda values, kind: np.stack([values[:, i, j].cpu().numpy().astype(kind) for i, j in one_case]) if one_case else np.zeros((0, places), kind)
                arrays[f'one_case_{term}_wrong_along'], arrays[f'one_case_{term}_correct_along'], arrays[f'one_case_{term}_resolved'] = picked(a, np.float32), picked(b, np.float32), picked(resolved, bool)
            if members:
                for term in RAW:
                    for name in (f'raw_{term}', f'raw_own_{term}'):
                        a, b, long_a, long_b, resolved = (kept[name][cell] for cell in cells)
                        over = resolved * (a > b)
                        for group, axis, value, length in (('wrong', 2, a, long_a), ('correct', 1, b, long_b)):
                            arrays[f'member_{group}_{name}_along'] = value.mean(axis).T.cpu().numpy().astype(np.float32)
                            arrays[f'member_{group}_{name}_resolved_over'] = (255 * over.mean(axis).T).round().cpu().numpy().astype(np.uint8)   # of a wrong answer: the correct answers it is resolved over; of a correct one: the wrong answers resolved over it
                            arrays[f'member_{group}_{name}_unresolved'] = (255 * (1 - resolved).mean(axis).T).round().cpu().numpy().astype(np.uint8)
                            if name == f'raw_{term}':
                                arrays[f'member_{group}_raw_{term}_length'] = length.mean(axis).T.cpu().numpy().astype(np.float32)
            is_now = squares['wrong']['incoming', 'incoming'].mean(1)
            if stood is not None:   # the term this merge came in with against the term the merge before put out, times the layer's output scale between them
                chain = float((is_now - stood * came_times ** 2).abs().max() / is_now.max().clamp(min=TINY))
            stood = squares['wrong']['outgoing', 'outgoing'].mean(1)
        else:
            stood = None
        np.savez_compressed(folder/f'{kind}-{layer}.npz', step=n, **arrays)
        mean = lambda name: float(arrays[name].mean()) if name in arrays else None
        unresolved = lambda name: 1 - mean(f'{name}_over') - mean(f'{name}_under') if f'{name}_over' in arrays else None
        line = dict(step=n, layer=layer, kind=kind, seconds=round(time.time() - began, 1), additive=additive, states=source, checks=checks, incoming_is_the_step_before_times=came_times,
                    searches=now['calls'], searches_of_the_step_before=before['calls'],
                    greatest_bound_over_cost=max(each['bound'] / each['cost'] for each in now['calls'] if each['cost'] > 0), most_loops=max(each['loops'] for each in now['calls']),
                    leftovers_against_costs=apart, outgoing_square_against_its_parts=off if additive else None, incoming_against_the_outgoing_before=chain,
                    reach_median={name: float(np.median(arrays[f'reach_{name}'])) for name in reach}, reach_undetermined={name: mean(f'reach_{name}_undetermined') for name in reach},
                    feature_joint_outgoing_ordered=mean('feature_joint_outgoing_ordered'), feature_joint_outgoing_unresolved=unresolved('feature_joint_outgoing'),
                    raw_outgoing_ordered=mean('raw_outgoing_ordered'), raw_outgoing_unresolved=unresolved('raw_outgoing'), raw_added_ordered=mean('raw_added_ordered'), raw_added_unresolved=unresolved('raw_added'),
                    raw_own_added_ordered=mean('raw_own_added_ordered'), raw_own_added_unresolved=unresolved('raw_own_added'))
        with (folder/'log.jsonl').open('a') as out:
            out.write(json.dumps(line) + '\n')
        say(json.dumps({name: value for name, value in line.items() if name not in ('searches', 'searches_of_the_step_before', 'states')}), flush=True)
        before = now
        del G, gamma2, s2, mixed, kept, squares, unsure_kept, reach, arrays
        torch.cuda.empty_cache()
    return dict(wrong=[ids[k] for k in wrong_at], correct=[ids[k] for k in right_at], steps=[f'{kind}-{layer}' for layer, kind in stages], size_over_direction=ratio, numbers=numbers.tolist(),
                matched_pairs=int(matched.sum()), one_case=[[ids[wrong_at[i]], ids[right_at[j]]] for i, j in one_case],
                left_out_with_a_wrong_answer=same.cpu().numpy().astype(bool).tolist(), left_out_with_a_correct_answer=kin.cpu().numpy().astype(bool).tolist())


def started(run, name):
    folder = run/name
    check_native_run(folder, run/'declaration.json')
    made_with = {each: tube.digest(module.__file__) for each, module in (('contributions', sys.modules[__name__]), ('mixture', mixture), ('correction', correction), ('tube', tube), ('ladder', ladder))}
    return folder, stages_of(json.loads((ladder.INPUTS/'index.json').read_text())['layers']), made_with


def whole_map(run):
    """`map`: every step, the answers' own split. A step whose file exists is not made again, so the command can be run again after a stop;
    STEPS=<n,n,...> before it reads those steps only."""
    folder, stages, made_with = started(run, 'map')
    wanted = [int(n) for n in os.environ['STEPS'].split(',')] if os.environ.get('STEPS') else list(range(len(stages)))
    chosen = [n for n in wanted if not (folder/f'{stages[n][1]}-{stages[n][0]}.npz').exists()]
    if not chosen:
        return
    began = time.time()
    told = mapped(run, folder, chosen)
    tube.save(folder/'index.json', dict(told, read=wanted, made_now=chosen, resolved=RESOLVED, searched_together=TOGETHER, how=mapped.__doc__, reading=__doc__, seconds=round(time.time() - began, 1), made_with=made_with))


def dealt(run):
    """`deals`: the same reading with the answers dealt at random (SEEDS), at the steps DEALS, without the per-answer arrays. A deal whose
    files exist is not made again."""
    folder, stages, made_with = started(run, 'deals')
    correct = {m['id']: m['correct'] for m in json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['members']}
    for seed in SEEDS:
        out = folder/f'deal-{seed}'
        chosen = [n for n in DEALS if not (out/f'{stages[n][1]}-{stages[n][0]}.npz').exists()]
        if not chosen:
            continue
        began = time.time()
        told = mapped(run, out, chosen, seed=seed, members=False)
        tube.save(out/'index.json', dict(told, seed=seed, read=list(DEALS), made_now=chosen, wrong_answers_among_those_read_as_wrong=sum(not correct[each] for each in told['wrong']), resolved=RESOLVED,
                                         searched_together=TOGETHER, seconds=round(time.time() - began, 1), made_with=made_with))


if __name__ == '__main__':
    {'map': whole_map, 'deals': dealt}[sys.argv[1]](ROUND_TWO_RUNS/sys.argv[2])
