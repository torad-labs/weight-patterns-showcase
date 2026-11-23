"""The wrong group's cells aimed at the nearest mixture of the original correct answers' own states. A development diagnostic's objective: nothing
here is read unless a run's declaration names it, and correction.py's cells, sums and commands are what they were without it.

Astra's specification of November 12, 2026 (records/retrospectives/2026-11-12-correct-mixture-astra.md), on this session's readings of where each
fitting answer stands between the two patterns (records/development/2026-11-12-correction-objective/README.md).

What it changes, and only that. correction.py aims the wrong group's mean, at every place of a merge, at one point: the mean of the original correct
answers there, within a radius. The original correct answers do not stand at one point. Here the wrong group's mean is aimed at the nearest
point of what they span: with c_k and l_k an original correct answer's registered direction and log size at a place (the full original correct
cohort, as the correction reference reads them), b and s the wrong group's reference there, d and h the wrong group's mean movement on the rows
read (counted as correction.py counts it, from the original), u = b + d and v = s + h, a cell's cost is

    F(alpha) = w_direction max(|u - sum_k alpha_k c_k| - r_direction, 0)^2 + w_size max(|v - sum_k alpha_k l_k| - r_size, 0)^2

at the coefficients alpha (none negative, summing to one) that make it least: ONE set of coefficients for the direction and the size together,
found from the whole cost with its radii, not from the direction alone. The weights and radii are the reference's own, unchanged. The mixed
direction is not made a unit vector again: the set aimed at is the members' convex hull in the features as registered. Where every member
has the same features this is correction.py's cell with that target. Each correct member's own error, which stands in the correct group's place
in correction-members-at-17-original's sum, is not touched: these are cells, and that term is the part's (correction.charged).

Finite work, and what a reading says of it. The coefficients are found by a finite search (`joined`), so a reading holds an ATTAINED cost, which
is never less than the least one, and a BOUND on how much less the least one can be: for a cost that is convex and has a slope in the
coefficients, sum_k alpha_k slope_k - min_k slope_k (Astra's specification: 'Preserve the attained value and that bound; its sum brackets the
nested objective'). correction.read passes the bound on, and correction.solve takes a written step only when what it leaves, as attained, is
under what stood less its bound: then the least cost itself has fallen, whatever the search left. With the coefficients fixed where a cell
stands (`Mixed.fixed`), the cells are correction.py's own with other targets, never under these cells' cost anywhere and equal to it there, up
to the bound.

A hull holds combinations no answer was observed at. A cell that stands inside it, within its radius, reads no discrepancy THIS relaxation can
see; that is not a statement that the computation there is correct."""
import copy
import json
import sys
import time

import numpy as np
import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROUND_TWO_RUNS
from pipeline.weight_edit import correction, ladder, tube
from pipeline.weight_edit.correction import Cells
from pipeline.weight_edit.tube import DEVICE, EFFORT, EPS, PATH

FINE = float(torch.finfo(torch.float64).eps)   # the rounding of the arithmetic the coefficients are found in
LEAST = float(torch.finfo(torch.float64).tiny)


def simplex(v):
    """Each row's nearest point among those with no negative part and a sum of one."""
    u = torch.sort(v, dim=1, descending=True).values
    over = (u.cumsum(1) - 1) / torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype)
    last = (u > over).sum(1, keepdim=True) - 1   # u_i > over_i holds for the first of them and for no later one
    return (v - over.gather(1, last)).clamp(min=0)


def joined(gram, across, far, sizes, size_far, radius, size_radius, weight, size_weight, steep, start, resolved, loops):
    """The coefficients of the mixture each cell stands nearest, for its direction and its size together and with their radii (the module's F).

    Everything is counted from the members' own mean, where a mixture's coefficients sum to one: with e the cell's direction less that mean and
    D_k a member's offset from it, |e - sum_k alpha_k D_k|^2 = far - 2 alpha.across + alpha.gram.alpha, and likewise for the size.
      gram, across, far   places x members x members (D_k . D_l), places x members (D_k . e), places (|e|^2)
      sizes, size_far     places x members (the members' size offsets from their own mean), places (the cell's size from that mean)
      radius, size_radius, weight, size_weight   places each: the cell's own
      steep               places: a bound on how fast the cost's slope turns, 2 (weight x gram's largest eigenvalue + size_weight x |sizes|^2)
      start               places x members: coefficients to start from (a point of the simplex)
      resolved            the search ends where the bounds, summed over the places, are no more than this part of the costs summed
      loops               and after this many loops at the latest, whatever is left: the bound says what

    A loop, at every place at once: Wolfe's nearest-point search, with the cost's own curvature in the place of a least-squares solve, since
    the radii make the cost no quadratic. A place has a face: the members that may hold something, at the start those that do. On the face
    the step is the one the cost's curvature gives there, its coefficients summing to nothing (with a lift of the root of the arithmetic's
    rounding times the curvature's mean, so that a flat direction gives no step). It is taken as far as the first member that would have to
    give more than it holds, which then holds nothing and leaves the face, and it is halved while it leaves no less of the cost. Once a
    whole step has been taken on a face, the member the slope favours most joins it. A place where no part of a step leaves less takes one
    step straight down the slope, kept on the simplex, which never leaves more than it found (`steep`), and its face is what that step
    holds. The search ends where the bounds are resolved, where no place moves, or at the last loop allowed. Whatever it does, the bound is
    taken from the coefficients it returns.
    Returns alpha, the cost attained per place with its two forms apart, the bound per place, how far each cell stands from its mixture (the
    direction's distance; the size's signed difference), and the loops used."""
    places, members = gram.shape[:2]
    rows, nothing, endless = torch.arange(places, device=gram.device), torch.zeros_like(start), torch.full_like(start, float('inf'))

    def at(alpha):
        carried = (gram @ alpha[:, :, None])[..., 0]
        root = (far - 2 * (alpha * across).sum(1) + (alpha * carried).sum(1)).clamp(min=0).sqrt()
        out = (root - radius).clamp(min=0)
        off = size_far - (sizes * alpha).sum(1)
        grown = (off.abs() - size_radius).clamp(min=0)
        pull = torch.where(out > 0, out / root.clamp(min=LEAST), torch.zeros_like(out))   # (1 - radius / distance), nothing inside
        lean = across - carried   # D_k . (e - the mixture)
        slope = -2 * (weight * pull)[:, None] * lean - 2 * (size_weight * grown * off.sign())[:, None] * sizes
        return dict(alpha=alpha, direction=weight * out ** 2, size=size_weight * grown ** 2, slope=slope, root=root, out=out, off=off, grown=grown, pull=pull, lean=lean)

    cost = lambda told: told['direction'] + told['size']
    lower = lambda one, other: {name: torch.where((cost(other) < cost(one)).view(-1, *[1] * (value.dim() - 1)), other[name], value) for name, value in one.items()}

    def stepped(now, free):   # the step the cost's curvature gives on a face, its coefficients summing to nothing
        curve = 2 * (weight * now['pull'])[:, None, None] * gram \
            + 2 * (weight * torch.where(now['out'] > 0, radius / now['root'].clamp(min=LEAST) ** 3, torch.zeros_like(radius)))[:, None, None] * now['lean'][:, :, None] * now['lean'][:, None, :] \
            + 2 * (size_weight * (now['grown'] > 0))[:, None, None] * sizes[:, :, None] * sizes[:, None, :]
        curve = torch.where(free[:, :, None] & free[:, None, :], curve, torch.zeros_like(curve))
        mean = curve.diagonal(dim1=1, dim2=2).sum(1) / free.sum(1)
        lift = torch.where(mean > 0, FINE ** 0.5 * mean, torch.ones_like(mean))
        system = torch.zeros((places, members + 1, members + 1), dtype=gram.dtype, device=gram.device)
        system[:, :members, :members] = curve + torch.diag_embed(torch.where(free, lift[:, None].expand(-1, members), torch.ones_like(nothing)))
        system[:, :members, members] = system[:, members, :members] = free.to(gram.dtype)
        wanted = torch.cat([torch.where(free, -now['slope'], nothing), nothing[:, :1]], 1)
        return torch.linalg.solve(system, wanted[:, :, None])[:, :members, 0]

    now, used = at(start), 0
    face, joins = start > 0, torch.ones(places, dtype=torch.bool, device=gram.device)
    while True:
        alpha, slope = now['alpha'], now['slope']
        bound = ((alpha * slope).sum(1) - slope.min(1).values).clamp(min=0)
        if not float(bound.sum()) > resolved * float(cost(now).sum()) or used == loops:
            break
        used += 1
        face[rows[joins], slope.argmin(1)[joins]] = True
        step = stepped(now, face)
        asked = face & (alpha == 0) & (step < 0)   # a member that holds nothing is asked to give: it leaves, and the step is taken again without it
        if bool(asked.any()):
            face = face & ~asked
            step = stepped(now, face)
        reach = torch.where(step < 0, alpha / -step, endless)
        part = reach.min(1).values.clamp(max=1)
        best, whole = now, torch.zeros_like(joins)
        while True:
            gone = (reach <= part[:, None]) & (step < 0)   # the members that give all they hold at this part of the step
            tried = torch.where(gone, nothing, (alpha + part[:, None] * step).clamp(min=0))
            tried = at(tried / tried.sum(1, keepdim=True))
            less = cost(tried) < cost(best)
            best, whole = lower(best, tried), torch.where(less, part == 1, whole)
            part = torch.where(cost(tried) < cost(now), torch.zeros_like(part), part / 2)   # a place that has its step asks for no other
            if not bool((part * step.abs().max(1).values > FINE).any()):
                break
        moved = cost(best) < cost(now)
        stuck = ~moved & (bound > 0)
        if bool(stuck.any()):   # one step straight down the slope, for the places no part of a step served
            down = at(simplex(alpha - slope / steep.clamp(min=LEAST)[:, None]))
            served = stuck & (cost(down) < cost(now))
            best = {name: torch.where(served.view(-1, *[1] * (value.dim() - 1)), down[name], value) for name, value in best.items()}
            moved = moved | served
        if not bool(moved.any()):
            break
        face = torch.where(moved[:, None], best['alpha'] > 0, face)
        joins = whole | ~moved
        now = best
    return dict(alpha=now['alpha'], cost=cost(now), direction=now['direction'], size=now['size'], bound=bound, distance=now['root'], size_off=now['off'], loops=used)


class Mixed(Cells):
    """A merge's Cells with the wrong group's target, at every place, the mixture of the original correct answers' own states that the group's
    mean stands nearest (the module's F, by `joined`). The correct group's cells, every radius and every weight are the reference's own, shared
    and never changed.

    cells    the reference's own Cells at the merge
    members  members x places x width: each original correct answer's registered direction there, less the wrong group's reference (c_k - b)
    sizes    members x places: the same of the log of its size (l_k - s)
    Their mean over the members is the reference's gap, so the mixture that weighs every member alike is the reference's own target.

    A reading (`read`) is Cells' own at the targets found, with the bound of the search (`bound`) and what the search did (`mixture`). The
    coefficients found last are where the next search starts, whichever answers it is for: they are a start, and the bound is taken from where
    the search ends. Every search made is counted with its loops, wherever it was made from (a reading, an aim, an error), until `searched`
    is asked: the declared diagnostic mixture-at-17-original kept the loops of its readings' searches and not of those inside its rounds."""

    def __init__(self, cells, members, sizes, resolved=EPS, loops=None):
        for name in ('target', 'radius', 'weight', 'size_target', 'size_radius', 'size_weight', 'places'):
            setattr(self, name, getattr(cells, name))
        offsets, sizes = members.double().to(DEVICE).transpose(0, 1).contiguous(), sizes.double().to(DEVICE).T.contiguous()   # places x members x width, places x members
        if offsets.shape[0] != self.places or sizes.shape != offsets.shape[:2]:
            raise ValueError('the members\' features are not laid on these cells\' places')
        self.middle, self.size_middle = offsets.mean(1), sizes.mean(1)   # where the members' own mean stands from the wrong group's reference: the pooled target
        self.spread, self.size_spread = offsets - self.middle[:, None], sizes - self.size_middle[:, None]
        self.gram = self.spread @ self.spread.transpose(1, 2)
        self.widest = torch.linalg.eigvalsh(self.gram)[:, -1].clamp(min=0)
        self.resolved = resolved   # the cells are read in the rows' own arithmetic: its rounding is what a reading resolves
        self.loops = self.spread.shape[1] ** 2 if loops is None else loops   # a stop no search is meant to reach (`joined`); what one leaves is in its bound
        self.alpha = torch.full(self.spread.shape[:2], 1 / self.spread.shape[1], dtype=torch.float64, device=DEVICE)
        self.kept = None
        self.searches = []   # the loops of every search made since `searched` was last asked

    def weighed(self, weight, size_weight):
        """The same cells with other weights: new ones, with a search of their own; the members are shared."""
        other = copy.copy(self)
        other.weight, other.size_weight, other.kept, other.searches = weight.float().to(DEVICE), size_weight.float().to(DEVICE), None, []
        return other

    def searched(self):
        """What the searches made since this was last asked did, and they are held no longer: how many were made, the loops they used together,
        the most one of them used, and how many used every loop allowed (`loops`: such a search was stopped, resolved or not, and its bound says
        what it left). A search is one run of `joined`: a reading and the aims taken where the last search was made are that search (`mixed`)
        and are not counted again. A solve's record keeps these round by round (correction.searched)."""
        made, self.searches = self.searches, []
        return dict(searches=len(made), loops=sum(made), most_loops=max(made, default=0), used_every_loop_allowed=sum(1 for each in made if each == self.loops))

    def mixed(self, mean, size):
        """Where the wrong group's cells stand (mean: 2 x places x width, size: 2 x places; the wrong group first), the mixture each stands
        nearest: its coefficients, the targets they make for both forms, the cost attained and the bound. Kept, so that a reading and the aims
        taken at the same place are one search."""
        moved, grown = mean[0], size[0]
        if self.kept is not None and torch.equal(moved, self.kept['moved']) and torch.equal(grown, self.kept['grown']):
            return self.kept
        weight, size_weight = self.weight[0].double(), self.size_weight[0].double()
        away, size_away = moved.double() - self.middle, grown.double() - self.size_middle
        found = joined(self.gram, (self.spread @ away[:, :, None])[..., 0], (away ** 2).sum(1), self.size_spread, size_away, self.radius[0].double(), self.size_radius[0].double(),
                       weight, size_weight, 2 * (weight * self.widest + size_weight * (self.size_spread ** 2).sum(1)), self.alpha, self.resolved, self.loops)
        self.alpha = found['alpha']
        self.searches.append(found['loops'])
        target = self.middle + (found['alpha'][:, None, :] @ self.spread)[:, 0]
        size_target = self.size_middle + (self.size_spread * found['alpha']).sum(1)
        self.kept = dict(found, moved=moved.clone(), grown=grown.clone(), target=torch.stack([target.float(), self.target[1]]), size_target=torch.stack([size_target.float(), self.size_target[1]]))
        return self.kept

    def fixed(self, mean, size):
        """Cells with the wrong group's targets held at the mixture nearest where it stands here: correction.py's own cells with other targets.
        Read anywhere else they are never under these cells' cost there, and here they are these cells' reading."""
        found = self.mixed(mean, size)
        return Cells(found['target'], self.radius, self.weight, found['size_target'], self.size_radius, self.size_weight)

    def errors(self, mean, size):
        return self.fixed(mean, size).errors(mean, size)

    def aims(self, mean, size):
        return self.fixed(mean, size).aims(mean, size)

    def read(self, mean, size):
        told, found = self.fixed(mean, size).read(mean, size), self.mixed(mean, size)
        held = (found['alpha'] > 0).sum(1)
        return dict(told, bound=float(found['bound'].sum()),
                    mixture=dict(loops=found['loops'], cost_in_the_search=float(found['cost'].sum()), places_with_a_cost=int((found['cost'] > 0).sum()),
                                 members_held=dict(least=int(held.min()), mean=float(held.double().mean()), most=int(held.max()))))


def mixed_of(cells, folder, reference, kind, layer):
    """A merge's Mixed cells, from the reference's own Cells there and the original correct answers' features the run's `members` command kept
    at that merge (folder: the run's correct-members), counted from the wrong group's reference pattern."""
    frozen, kept = np.load(reference/f'{kind}-{layer}.npz'), np.load(folder/f'members-{kind}-{layer}.npz')
    return Mixed(cells, torch.from_numpy(kept['members']).double() - torch.from_numpy(frozen['pattern'][0]).double(),
                 torch.from_numpy(kept['sizes']).double() - torch.from_numpy(frozen['pattern_size'][0]).double())


def stood_at(answers, index, tensors, layout, groups, reference, layer, kind, norm):
    """One merge of `members`: the original correct answers' registered features there (members x places x width in float32, as the reference keeps
    a pattern; members x places), the arrays `members` writes for the merge, and its line of the record. `norm`: what the output norm passes
    on (correction.terminal), for the model's last merge; None at any other."""
    w, g = ladder.weights_of(tensors, layer, kind)[1:] if kind in tube.KINDS else (None, None)
    to, log, _, _ = tube.recorded(answers, index, layer, kind, w, g, tensors)
    if kind in tube.KINDS:
        log = tube.native_log(answers, layer, kind)
    if norm is not None:
        to, log = correction.passed_on(to, log, **norm)
    z, sizes = layout.means(to), layout.means(log)[..., 0]
    del to, log
    frozen = np.load(reference/f'{kind}-{layer}.npz')
    pattern, pattern_size = (torch.from_numpy(frozen[name]).to(DEVICE) for name in ('pattern', 'pattern_size'))
    apart = dict(direction=max(float((z[mask].mean(0).float() - pattern[k]).abs().max()) for k, mask in enumerate(groups)),
                 size=max(float((sizes[mask].mean(0) - pattern_size[k]).abs().max()) for k, mask in enumerate(groups)))
    if max(apart.values()) > EPS:   # the rounding of the arithmetic the reference keeps a pattern in
        raise ValueError(f'the groups\' means at {kind}-{layer} are not the reference\'s patterns ({apart}): these rows are read another way than the reference read them')
    kept, kept_sizes = z[groups[1]].float(), sizes[groups[1]]
    del z, sizes
    cells = correction.cells_of(reference, kind, layer)
    mixed = Mixed(cells, kept.double() - pattern[0].double(), kept_sizes - pattern_size[0])
    still = torch.zeros_like(cells.target), torch.zeros_like(cells.radius)
    pooled, told, found = cells.read(*still), mixed.read(*still), mixed.mixed(*still)
    turned, grown = cells.errors(*still)
    arrays = dict(alpha=found['alpha'], direction=found['direction'], size=found['size'], bound=found['bound'], distance=found['distance'], size_off=found['size_off'],
                  pooled_direction=(cells.weight[0] * turned[0] ** 2).double(), pooled_size=(cells.size_weight[0] * grown[0] ** 2).double(),
                  pooled_distance=cells.target[0].double().norm(dim=-1), pooled_size_off=cells.size_target[0].double(), radius=cells.radius[0].double(), size_radius=cells.size_radius[0].double())
    held = (found['alpha'] > 0).sum(1)[found['direction'] + found['size'] > 0]
    line = dict(layer=layer, kind=kind, pooled=dict(direction=pooled['direction'][0], size=pooled['size'][0]), mixture=dict(direction=told['direction'][0], size=told['size'][0]),
                bound=told['bound'], loops=found['loops'], places_with_a_cost=dict(pooled=int(((turned[0] > 0) | (grown[0] > 0)).sum()), mixture=told['mixture']['places_with_a_cost']),
                members_held_where_a_cost_is_left=dict(least=int(held.min()), mean=float(held.double().mean()), most=int(held.max())) if len(held) else None,
                the_groups_means_from_the_references_patterns=apart)
    return kept, kept_sizes, {name: value.cpu().numpy() for name, value in arrays.items()}, line


def members(run):
    """The original correct answers' own states at every merge of the model, as the run's correction reference reads them, and what the
    original model has of this module's objective there. Written once into the run's correct-members and only read after.

    At each merge (<kind>-<layer>.npz), from the original recordings as correction.reference reads them (tube.recorded; the size as the engine
    recorded it; the model's last merge as the output norm passes it on) and the reference's own cells, where the wrong group's reference
    stands with nothing moved: the coefficients of the nearest mixture at every place (alpha: places x members, the members in index.json's
    order), the cost attained with its two forms apart, its bound, how far the reference stands from the mixture, and beside them the same
    cells' cost with the pooled target (the reference's own: the wrong side of correction-members-at-17-original's sum). Every count is the
    source's number of answers: no fitting, no-fit or reserved role enters. Where a place's cost is nothing its coefficients are not
    determined: they are where the search stopped, from the mixture that weighs every member alike.

    The members' features themselves (members-<kind>-<layer>.npz: members x places x width in float32, as the reference keeps a pattern, and the
    logs of their sizes) are kept at the merges the run's solve reads. At any other merge this command makes them again from the same
    recordings, in the seconds a merge takes. index.json: the members' ids in the arrays' order, the reference read, and merge by merge what
    the arrays sum to."""
    folder = run/'correct-members'
    declaration = check_native_run(folder, run/'declaration.json')
    made_with = {name: tube.digest(module.__file__) for name, module in (('mixture', sys.modules[__name__]), ('correction', correction), ('tube', tube))}   # as they are when the command starts
    at = correction.started_from(run)
    reference, tensors = at['reference'], at['tensors']
    if tube.digest(reference/'index.json') != declaration['correction_reference_index_sha256']:
        raise ValueError('the correction reference is not the one the run declares')
    _, answers, index = ladder.cohort(run)
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    pairs = [np.load(ladder.CORRESPONDENCE/'pairs'/f"{m['id']}.npy").astype(np.int64) for m in answers]
    correct = np.array([m['correct'] for m in answers])
    layout = tube.Layout(~correct, [m['steps'] for m in answers], [m['question'] for m in answers], pairs, places)
    groups = [torch.from_numpy(mask).to(DEVICE) for mask in (~correct, correct)]   # the wrong group first
    last = index['layers'][-1]
    norm = correction.terminal(tensors, last)
    solved = [(at['layer'], each) for each in [at['kind']] + (['layer'] if at['kind'] == 'feedforward' else [])]   # the merges the run's solve reads
    folder.mkdir(parents=True)
    report, started = [], time.time()
    for layer in index['layers']:
        for kind in tube.MERGES:
            kept, kept_sizes, arrays, line = stood_at(answers, index, tensors, layout, groups, reference, layer, kind, norm if (layer, kind) == (last, 'layer') else None)
            np.savez(folder/f'{kind}-{layer}.npz', **arrays)
            if (layer, kind) in solved:
                np.savez(folder/f'members-{kind}-{layer}.npz', members=kept.cpu().numpy(), sizes=kept_sizes.cpu().numpy())
                line['members_kept'] = f'members-{kind}-{layer}.npz'
            report.append(line)
            print(json.dumps(line), flush=True)
            del kept, kept_sizes, arrays
            torch.cuda.empty_cache()
    if len(report) != len(tube.MERGES) * len(index['layers']):
        raise ValueError('The members\' reference does not cover every merge of the model')
    total = lambda which, form: sum(each[which][form] for each in report)
    record = dict(members=[m['id'] for m in answers if m['correct']], wrong_reference_from=[m['id'] for m in answers if not m['correct']], places=places, merges=report,
                  the_originals_objective=dict(pooled={form: total('pooled', form) for form in correction.FORMS}, mixture={form: total('mixture', form) for form in correction.FORMS},
                                               bound=sum(each['bound'] for each in report), most_loops=max(each['loops'] for each in report)),
                  reference=dict(folder=str(reference.relative_to(tube.ROOT)), sha256=tube.digest(reference/'index.json'),
                                 its_own_objective=json.loads((reference/'index.json').read_text())['the_originals_objective']),
                  resolved=EPS, how=members.__doc__, objective=__doc__, specification=declaration.get('specification'), seconds=round(time.time() - started, 1), made_with=made_with)
    tube.save(folder/'index.json', record)
    return record


VARIANTS = ('members', 'mixture')   # the matched run's own sum, and the same sum with the wrong group's cells aimed at the nearest mixture


def walked(run):
    """The run's paths: at its one matrix, from the original, on the rows the matched run was solved on, the matched run's own sum (VARIANTS:
    `members`) and this module's (`mixture`: the same sum with the wrong group's cells Mixed), each under the matched run's recorded work.
    That work is read from the matched run's own record, not copied: its unit of the ridge, its first ridge and the ridge it selected, every
    ridge between halved in turn, every stage run to its own end under the same EFFORT, the same preconditioner and writing measure (the
    reference's own cells). For each sum: one path on all the fitting answers, and one per fold, solved on the other two folds and read on it.
    Nothing is selected: every path ends at the matched run's selected ridge. paths/<variant>-<path>.npz is the matrix a path ended on (float32
    values, each one the file's BF16 holds) and .json its record; a path whose record exists is not walked again. No model is read through the
    engine and none is written."""
    folder = run/'paths'
    declaration = check_native_run(folder, run/'declaration.json')
    made_with = {name: tube.digest(module.__file__) for name, module in (('mixture', sys.modules[__name__]), ('correction', correction), ('tube', tube))}
    at = correction.started_from(run)
    asked = declaration['correct_members']
    requests, replaced = tuple(asked['requests']), bool(asked['replaces_the_correct_groups_cells'])
    if not replaced or any(correction.REQUESTS[name]['members'] != 'correct' for name in requests):
        raise ValueError('the sum compared is the matched run\'s: each correct member\'s own error in the correct group\'s place, and nothing asked of a wrong member')
    other = ROUND_TWO_RUNS/declaration['matched_work']['run']
    matched = json.loads((other/'candidate'/'record.json').read_text())
    theirs = json.loads((other/'declaration.json').read_text())
    same = lambda name: theirs['planned_starting_state'][name] == at['state'][name]
    if not all(same(name) for name in ('model_sha256', 'matrix', 'source_rows', 'source_rows_sha256')) or theirs['correct_members']['requests'] != list(requests) \
            or tube.digest(at['reference']/'index.json') != matched['reference']['sha256']:
        raise ValueError('the matched run was not solved from this matrix of this model on these rows against this reference with this sum')
    if matched['work'] != dict(EFFORT, resolved=tube.RESOLVED, path=PATH) or matched['selected_round'] != EFFORT['rounds']:
        raise ValueError('the matched run\'s work is not this file\'s: a stage run to its own end here would not be a stage of its')
    unit, first, last = matched['unit'], matched['first_ridge_power'], matched['selected_ridge_power']
    on = correction.laid(at, np.load(tube.ROOT/at['state']['source_rows']))
    trace, frames, w, g, stands, used, fitting, groups, places = (on[name] for name in ('trace', 'frames', 'w', 'g', 'stands', 'used', 'fitting', 'groups', 'places'))
    mixed = [mixed_of(cells, run/'correct-members', at['reference'], each, at['layer']) for cells, each in zip(frames, on['merges'])]
    folds, everyone = [np.asarray(each[fitting]) for each in groups], np.ones(len(used), bool)

    def part_of(pick):   # these answers with each correct member's own error beside the cells, weighed by the reference's own cells
        part = trace.part(pick, places)
        part.quiet = correction.charged(part, frames, requests)
        return part

    own_unit = correction.unit_of(part_of(everyone), frames, stands, g)
    if abs(own_unit / unit - 1) > tube.RESOLVED:
        raise ValueError(f'the unit of the ridge on these rows ({own_unit}) is not the matched run\'s ({unit})')
    ids = lambda pick: [m['id'] for m, take in zip(used, pick) if take]
    folder.mkdir(exist_ok=True)
    order = [(variant, 'all', everyone, None) for variant in VARIANTS] + [(variant, f'fold-{k}', everyone & ~folds[k], folds[k]) for k in range(len(folds)) for variant in VARIANTS]
    for variant, name, solved_on, read_on in order:
        out = folder/f'{variant}-{name}'
        if out.with_suffix('.json').exists():
            print(json.dumps(dict(variant=variant, path=name, kept='already walked')), flush=True)
            continue
        started = time.time()
        summed = [correction.without_correct(cells) for cells in (mixed if variant == 'mixture' else frames)]   # new cells for every path: no search starts from another path's
        say = lambda told, variant=variant, name=name: print(json.dumps(dict(variant=variant, path=name, **{key: (round(value, 12) if isinstance(value, float) else value) for key, value in told.items()})), flush=True)
        walk = correction.Descent(part_of(solved_on), summed, w, g, stands, unit, EFFORT, beside=[] if read_on is None else [part_of(read_on)], say=say, geometry=frames)
        power = first
        while walk.ended is None and power >= last:
            walk.stage(power)
            power -= PATH
        walk.record['ended'] = walk.ended or 'the ridge the matched run selected'
        kept = walk.weights
        if not bool((tube.bf16(kept) == kept).all()):
            raise ValueError('a path ended on a matrix the file does not hold')
        np.savez(out.with_suffix('.npz'), matrix=kept.cpu().numpy())   # kept the moment it exists; the record follows and says the path is whole
        record = dict(run=run.name, variant=variant, path=name, tensor=at['state']['matrix'], solved_on=ids(solved_on), read_on=None if read_on is None else ids(read_on),
                      matched_to=dict(run=other.name, unit=unit, first_ridge_power=first, last_ridge_power=last, the_unit_on_these_rows=own_unit), work=dict(EFFORT, resolved=tube.RESOLVED, path=PATH),
                      objective_before=walk.record['before']['objective'], objective=walk.record['stages'][-1]['objective'] if walk.record['stages'] else walk.record['before']['objective'],
                      change_over_matrix=float((kept - w).norm() / w.norm()), weights_changed=int((kept != stands).sum()), seconds=round(time.time() - started, 1), made_with=made_with)
        if (variant, name) == ('members', 'all'):   # the matched run's own last solve, walked again by this file: its candidate, or the two differ somewhere
            theirs_kept = torch.from_numpy(np.load(other/'candidate'/f"{at['state']['matrix']}.npy")).to(DEVICE)
            record['the_matched_runs_candidate'] = dict(weights_that_differ=int((kept != theirs_kept).sum()), its_objective=matched['objective'], its_change_over_matrix=matched['change_over_matrix'])
            del theirs_kept
        tube.save(out.with_suffix('.json'), dict(record, walk=walk.record))
        print(json.dumps(record), flush=True)
        del walk, summed, kept
        torch.cuda.empty_cache()
        with tube.CARD.given():   # another project's waiting job gets in between two paths
            time.sleep(5)


if __name__ == '__main__':
    tube.CARD.take()
    {'members': members, 'paths': walked}[sys.argv[1]](ROUND_TWO_RUNS/sys.argv[2])
