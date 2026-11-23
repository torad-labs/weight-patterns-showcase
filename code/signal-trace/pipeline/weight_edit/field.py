"""One matrix's part of a correction of the whole gap: every row has its own request, and the matrix is solved against all of them at once.

A row is one answer step, or one position of a question, of one fitting answer. At the matrix's merge it asks how far the unit state the
merge leaves should move (a vector as wide as the state) and how far the log of that state's size should (a number). Where the computation
already agrees the row asks for nothing, and it is held at nothing in the same sum: there is no second objective and no bound. Which rows ask
what is not this file's: the requests come with the rows, from whoever fixed them from the original model. This file is what a matrix does
with them, and what it costs: nothing here chooses a layer, a row or a size of change.

The arithmetic for one matrix derives from the historical per-row inverse documented in
records/retrospectives/2026-11-10-d1-a2-calibration/READINESS.md (snapshot 67559f4a); it is implemented here, not imported from a removed
package. That construction had recorded repairs outside its fit. One part is taken out and another put in its place.

  goal       For each row, the output of the matrix that leaves the least of the row's request among the outputs as long as the row's own.
             The block's norm divides an output by its root mean square, so where an output points reaches the state and its length
             almost does not: the norm's epsilon keeps a small part of the length, and keeping every row's output as long as its own
             leaves that part out (an approximation of what a matrix can reach, not of what is read). The states such an output can
             leave are an ellipsoid around the incoming state. A row is aimed at the state its request names, to first order through the
             unit state and the log of the size, and the ellipsoid's nearest point to that aim is found from its one-dimensional
             equation (`nearest`); the aim is taken again from there until the row gains nothing, and a step that would raise the row's
             error is shortened instead (`goal`).
  shared     One least squares over the rows: the change of the matrix whose rows' state changes lie nearest the ones their goals ask,
             to first order. An output change reaches a row's unit state over the output's root mean square and the state's size, and
             coordinate by coordinate times the block's learned scale there. Each row is weighed by its own weight and the first
             (`reaches`), each coordinate of the output by the second, squared (`Matrix.counts`), with a ridge on the change. The normal
             equations are then one symmetric matrix as wide as the matrix's input, the same for every coordinate of the output but
             for that coordinate's count: it is decomposed once, and every ridge after that is a division, a coordinate's ridge being
             the ridge over its count (`Shared`). A coordinate the scale nearly ignores is held where it is by that: its output would
             have to change most to move the state least. Until November 14 this file counted every coordinate alike, and at matrices
             whose scale nearly ignores a few dozen coordinates most of what the rows asked of the matrix was a change of those
             (records/development/2026-11-14-whole-gap-solver/README.md, "What the coordinates the scale ignores were asked").
  taken out  the cap on a matrix's change that construction borrowed, which decided its ridge.
  put in     the ridge is read (`path`). The fitting answers are in groups. Each group in turn is left out of the least squares, and the
             matrix the others give is written as the file's BF16 holds it and read through the merge itself on the group left out, at
             every ridge from one that moves no weight of the file down to one the rows' own decomposition no longer resolves. What a ridge
             leaves of the requests of answers it was not solved on, and what it does to their rows that asked for nothing, is the reading
             a ridge is taken by. Which ridge that reading takes is the caller's rule: `path` chooses none, and `written_at` writes the one
             it is told.
  last merge A layer ends in a third merge that none of these matrices writes: there the engine adds the per-layer embedding to the
             state the feed-forward's merge leaves, a fixed function of that state on a text that stays the same (`Final`). A
             feed-forward's rows can ask there as well as at their own merge. Both requests are then in one sum, each with its own
             weight: a row's goal is the output of its own length that leaves the least of the two together, sought through the last
             merge's own function, and every reading goes through that merge as the engine rounds it. Neither request stands in for the
             other, and a row whose weight at a merge is nothing has no request there, which is not a request for nothing. The last
             merge does not read a step alike in every direction, so the search behind it first aims by the slope over twice a bound
             on how sharply the two errors bend together (`Final.stretch`), and after that by the slope over the bend the row's last
             step itself showed; a row is at rest when the bound's own step gains it nothing. Its steps are the ellipsoid's exact
             nearest points all the same, and it ends at the same kind of place. Behind the last layer's feed-forward lies the
             model's last merge, which is read as what the output norm passes on of it (`Terminal`).

Nothing here is iterated to a tolerance that decides a result. The search for a row's goal stops the row when an aiming gains it less than
the float32 state can hold of the row's error where it stands (`unheld`) or no shortened step moves its state any more; the path's ends
come from the decomposition. A matrix is read, compared and
returned only as the file's BF16 holds it, weight by weight to the nearest (tube.bf16): what one weight cannot hold is not made up for by
the others here, and every reading says how many weights moved.

The sums, the decompositions and every ridge's division are in double precision, on the card or the processor the rows are on. The two
products a ridge then costs are in single precision: input width x input width by input width x 1536 to make its change (`Shared.change`)
and a third of the rows by that matrix to read it, as the engine multiplies (`Matrix.out`). For an input w wide and r rows the normal
equations take r w^2 products, every row summed once: each of a path's groups has its own sums, and every least squares is some of them
added (`Matrix.parted`). Each decomposition takes about 9 w^3: three for the groups and one for the matrix that is written, which at
the widest inputs is most of a matrix's time (records/development/2026-11-14-whole-gap-solver/README.md).

A matrix can be solved in single precision instead (`Matrix.in_single`), which a card does several times faster: each block's two
products of the sums are then single and the blocks are added in double (`normal_single`), the decomposition is single with every
vector kept, each ridge's division stays double, and the readings are what they were. A single decomposition knows a value no nearer
than rounding times its largest, so its path ends there (`Shared.resolved`), some fifteen powers of two above where the double one's
ends, and near that end its changes are no longer the least squares' own. What that does to a path's reading and to the ridge a rule
takes, on a real request, is read in records/development/2026-11-14-whole-gap-solver/README.md ("One matrix's ridges read both ways");
`unsolved` says of every matrix written this way how far its change is from solving its own least squares."""
import math

import torch

from pipeline.weight_edit import tube
from pipeline.weight_edit.a2 import NORM_EPSILON

F8 = torch.float64
EPS = float(torch.finfo(F8).eps)
EPS4 = float(torch.finfo(torch.float32).eps)
FLOOR = lambda width: width * float(torch.finfo(torch.float32).eps) ** 2   # what a float32 state of that width cannot hold of a squared distance between unit states that agree
# Rows multiplied or summed at once. It puts no row into a sum and takes none out: it orders a sum's additions and bounds the work's
# memory. The number is the narrowest input among the matrices a sweep solves, so that at no matrix a block of rows (BLOCK x input
# width) is larger than the sums it is added into (input width x input width); a sweep holds it to its own matrices and refuses
# another (gap_run.block_held).
BLOCK = 2048


def unheld(width, error):
    """What a float32 state of that width cannot hold of a row's error where the row stands. FLOOR is the square of a length: how far
    apart two unit states can be that such a state does not tell apart. A row's error is a squared distance d^2 (between unit states,
    and of the log of the size), and a state that is uncertain by that length leaves it uncertain by (d + length)^2 - d^2. Where the
    row's request is met d is nothing and this is FLOOR; where a request cannot be met it is what stops the row, at a gain the
    float32 state it is read through could not show."""
    return FLOOR(width) + 2 * math.sqrt(FLOOR(width)) * torch.sqrt(error)


def left(incoming, raw, gain):
    """What a merge leaves of some rows: the unit state and its size, from the state the matrix's output is added to (rows x width), the
    output and the block's learned scale. The block's norm takes the output's root mean square out first."""
    state = incoming + gain * raw / torch.sqrt((raw * raw).mean(1, keepdim=True) + NORM_EPSILON)
    size = state.norm(dim=1)
    return state / size[:, None], size


def asked_of(incoming, raw, gain, unit, log_size):
    """A row's request from where its state should stand after the merge (its unit state and the log of its size) and where the matrix as it
    stands leaves it: the two differences."""
    to, size = left(incoming, raw, gain)
    return unit - to, log_size - torch.log(size)


def errors(incoming, raw, changed, gain, direction, size):
    """What is left of each row's request when the matrix's output goes from `raw` to `changed`: the squared distance of the unit state from
    where it was asked to stand, and the squared distance of the log of the size from where it was asked to."""
    to, was = left(incoming, raw, gain)
    now, is_ = left(incoming, changed, gain)
    return ((now - to - direction) ** 2).sum(1), (torch.log(is_ / was) - size) ** 2


def reaches(raw, gain, size):
    """How much of an output change reaches each row's state, to first order and as one number a row: the learned scale's root mean square,
    over the output's root mean square and the state's size. A coordinate's own part of it is `Matrix.counts`: the learned scale there,
    squared, over its mean square. The two condition the least squares; what a matrix leaves is read by `errors`."""
    return torch.sqrt((gain * gain).mean()) / (torch.sqrt((raw * raw).mean(1) + NORM_EPSILON) * size)


def goal(incoming, raw, gain, direction, size, final=None, share=None):
    """For each row the output that leaves the least of its request among the outputs as long as its own, and what the search did. A row
    that asks nothing keeps its output as it is, bit for bit. With `final` (a feed-forward's layer's last merge, Final) a row's request
    there is in the same least: `share` is how much of a row's error is the last merge's (rows, 0 to 1), the rest its own merge's."""
    if bool((gain == 0).any()):
        raise ValueError('a learned scale of zero: no output reaches that coordinate of the state, and this search divides by the scale')
    asks = ((direction * direction).sum(1) + size * size) > 0
    if final is not None:
        asks = (asks & (share < 1)) | ((((final.direction * final.direction).sum(1) + final.size * final.size) > 0) & (share > 0))
    out = raw.clone()
    told = dict(rows=int(raw.shape[0]), asking=int(asks.sum()), passes=0, largest_step_along_the_sphere_left=0.0)
    there = asks & (share > 0) if final is not None else torch.zeros_like(asks)   # the rows whose error holds the last merge's part
    for some, behind in ((asks & ~there, False), (there, True)):
        if bool(some.any()):
            more = _Behind(final, some.nonzero().squeeze(1), incoming[some], raw[some], gain, share[some]) if behind else None
            out[some], search = _on_the_sphere(incoming[some], raw[some], gain, direction[some], size[some], more)
            told.update(passes=max(told['passes'], search['passes']),
                        largest_step_along_the_sphere_left=max(told['largest_step_along_the_sphere_left'], search['largest_step_along_the_sphere_left']))
            if behind:
                told.update(largest_stretch_of_the_last_merge=float(more.stretch.max()))
    return out, told


class _Behind:
    """Some rows' request at the layer's last merge, read from the states their feed-forward's merge leaves: the error there and how
    sharply it bends, for a row's goal. The last merge is taken with nothing rounded here, so that it has a slope everywhere. `stretch`
    bounds, for each row, how much longer the last merge makes any step of the state the row starts from (`Final.stretch`)."""

    def __init__(self, final, index, x, z, gain, share):
        self.final, self.index, self.share = final, index, share
        state = x + gain * z / torch.sqrt((z * z).mean(1, keepdim=True) + NORM_EPSILON)
        to, self.was, _ = final.smooth(state, index)
        self.wanted, self.h = to + final.direction[index], final.size[index]
        self.stretch = final.stretch(state, index)

    def __call__(self, state, rows):
        to, size, bend = self.final.smooth(state, self.index[rows])
        grown = torch.log(size / self.was[rows]) - self.h[rows]
        return ((to - self.wanted[rows]) ** 2).sum(1) + grown * grown, bend


def nearest(x, scale, radius, point):
    """For each row, the output as long as `radius` whose state x + scale * output lies nearest `point`: the least of a quadratic on a
    sphere, from its one-dimensional equation. With t = point - x that output is scale t / (scale^2 + u), for the one u above minus the
    row's least scale squared at which it is as long as the radius. Its length falls as u rises, and the length's inverse root is
    nearly a straight line in u: Newton's steps on it from below (where the least-scaled coordinates' own terms are already as long
    as the radius) only rise, and are taken until a step no longer raises the double. From one last place either side of where
    they end the two ends are moved out, twice as far each time, until the length says they hold u, and bisected until a double
    holds no point between them: one binary search, outward and then inward, with no number of its own. The length never rises
    with u as it is computed, so there is one pair of neighbouring doubles at which the output stops being longer than the radius,
    and the search ends on it wherever it began: the end the bisection alone reaches (`_bisected`), in about a third of its
    readings of the length (records/development/2026-11-10-speed-review/solve-arithmetic/nearest_by_newton.py read that on
    19,751 real rows when the steps still ended at a multiple of float64's epsilon and the ends began at another). A row with
    nothing pulling at its least-scaled coordinate has no start from below and is left to the bisection."""
    t = point - x
    pulled = (scale * t) ** 2
    square = scale * scale
    over = square - square.min(dim=1).values[:, None]   # nothing at a row's least scaled coordinate; v below is u + least, above nothing
    wanted = radius * radius
    length = lambda v: (pulled / (over + v[:, None]) ** 2).sum(1)
    v = torch.sqrt((pulled * (over == 0)).sum(1) / wanted)   # no shorter than the radius there
    plainly = ~(v > 0)
    live = ~plainly
    while bool(live.any()):
        deep = over + v[:, None]
        each = pulled / (deep * deep)
        long, slope = each.sum(1), (each / deep).sum(1)
        step = (long / slope) * (torch.sqrt(long / wanted) - 1)
        live = live & (long > wanted) & (v + step > v)
        v = torch.where(live, v + step, v)
    low, high, wide = v.clone(), v.clone(), torch.nextafter(v, torch.full_like(v, math.inf)) - v
    below, above = ~plainly, ~plainly
    while bool((below | above).any()):   # the two ends are moved out, twice as far each time, until the length says they hold u
        low = torch.where(below, v - wide, low)
        high = torch.where(above, v + wide, high)
        below = below & (low > 0) & ~(length(low) > wanted)
        above = above & (length(high) > wanted)
        wide = wide * 2
    while True:
        middle = (low + high) / 2
        between = (middle > low) & (middle < high)
        if not bool(between.any()):
            break
        longer = length(middle) > wanted
        low = torch.where(between & longer, middle, low)
        high = torch.where(between & ~longer, middle, high)
    out = scale * t / (over + high[:, None])
    out = out * (radius / out.norm(dim=1))[:, None]
    if bool(plainly.any()):
        out[plainly] = _bisected(x[plainly], scale[plainly], radius[plainly], point[plainly])
    return out


def _bisected(x, scale, radius, point):
    """`nearest` by halving and bisection alone: u is bracketed by halving a bound and found by bisection until a double holds no
    point between the two ends."""
    t = point - x
    pulled = (scale * t) ** 2
    least = (scale * scale).min(dim=1).values
    over = scale * scale - least[:, None]   # nothing at a row's least scaled coordinate; v below is u + least, above nothing
    wanted = radius * radius
    length = lambda v: (pulled / (over + v[:, None]) ** 2).sum(1)
    high = torch.sqrt(pulled.sum(1) / wanted)   # no longer than the radius there: every term is at most pulled / v^2
    low = high.clone()
    searching = high > 0
    while bool(searching.any()):
        low = torch.where(searching, low / 2, low)
        searching = searching & (low > 0) & (length(low) < wanted)
    high = torch.minimum(high, 2 * low)
    while True:
        middle = (low + high) / 2
        between = (middle > low) & (middle < high)
        if not bool(between.any()):
            break
        longer = length(middle) > wanted
        low = torch.where(between & longer, middle, low)
        high = torch.where(between & ~longer, middle, high)
    out = scale * t / (over + high[:, None])
    return out * (radius / out.norm(dim=1))[:, None]


def _on_the_sphere(x, z, gain, d, h, behind=None):
    radius = z.norm(dim=1)
    if bool((radius == 0).any()):
        raise ValueError('a row whose output is nothing: the norm gives it no direction to turn')
    scale = gain / torch.sqrt((z * z).mean(1, keepdim=True) + NORM_EPSILON)   # what an output of this length is multiplied by on its way to the state
    base = x + scale * z
    base_size = base.norm(dim=1)
    wanted = base / base_size[:, None] + d

    def read(c, rows, aimed=False):
        """A row's error at an output, and with `aimed` the state it leaves, the step its request asks of that state, and the state's size."""
        state = x[rows] + scale[rows] * c
        if aimed:
            state.requires_grad_(True)
        norm = state.norm(dim=1)
        unit = state / norm[:, None]
        grown = torch.log(norm / base_size[rows]) - h[rows]
        value, bend = ((unit - wanted[rows]) ** 2).sum(1) + grown * grown, 1 / (norm * norm)
        if behind is not None:
            there, bend_there = behind(state, rows)
            value, bend, bend_there = (1 - behind.share[rows]) * value + behind.share[rows] * there, (1 - behind.share[rows]) * bend, behind.share[rows] * bend_there
        if not aimed:
            return value
        with torch.enable_grad():
            down, = torch.autograd.grad(value.sum(), state)
        value, state, down, bend, norm = value.detach(), state.detach(), down.detach(), bend.detach(), norm.detach()
        across = c / scale[rows]   # the ellipsoid's own direction at the state: a step along the ellipsoid has no part of it
        # The step the request asks of the state, to first order. At the matrix's own merge the unit state and the log of the size read
        # a step alike in every direction (the same across the state and along it), so the slope over twice that bend is the step.
        # Behind a feed-forward the last merge reads a step through its own slopes, as at most `stretch` times as long as it is: the
        # two merges' error together then bends no more sharply than bend + bend_there stretch^2 in any direction, and the slope over
        # twice that is a step the first-order reading cannot overshoot. That bound's step is a row's first there, and the measure
        # of its rest; between the two the loop below counts the step from the bend the row's last step showed. Either way the
        # trial is the ellipsoid's nearest point to where the step leads: the least, on the ellipsoid itself, of a reading that
        # bends alike in every direction.
        if behind is not None:
            bend = bend + bend_there.detach() * behind.stretch[rows] ** 2
        aim = -down / (2 * bend[:, None])
        return value, state, aim, norm, aim - ((across * aim).sum(1) / (across * across).sum(1))[:, None] * across

    c = z.clone()
    part = torch.ones(z.shape[0], dtype=z.dtype, device=z.device)   # how much of an aimed step a row's next trial takes
    active = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
    passes = 0
    last = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)   # the step along the ellipsoid each row was last asked, beside its state: how near rest it ended
    # Behind a feed-forward the bound's step is far too short for a row the last merge stretches much in a few directions and little
    # in the rest: at feed-forward 8 the merge stretches a step up to 74 times for a few rows of 7,916 (4.6 at the median), and by
    # the bound's step alone those rows were still moving after 17,000 passes (records/development/2026-11-14-whole-gap-solver/
    # README.md). So there a row's step is the slope over the bend its last step itself showed: how much the slope changed along
    # that step, over the step's length squared (the secant's step, Barzilai and Borwein's). A trial that does not lower the row's
    # error is shortened as everywhere. The bound's own step is taken at a row's first pass, where the last step showed no bend,
    # and after a pass that gained the row nothing: a row is at rest only when the bound's step gains it nothing.
    if behind is not None:
        stood, aimed = torch.zeros_like(z), torch.zeros_like(z)   # where each row stood at its last pass, and the bound's step asked of it there
        seen, doubted = torch.zeros_like(active), torch.zeros_like(active)
    while bool(active.any()):
        passes += 1
        rows = active.nonzero().squeeze(1)
        found, state, aim, norm, along = read(c[rows], rows, aimed=True)
        if behind is not None:
            step, turned = state - stood[rows], aimed[rows] - aim   # the bound's step is the slope over twice the bound, turned about: its change is the slope's
            bent, length = (step * turned).sum(1), (step * step).sum(1)
            shown = seen[rows] & ~doubted[rows] & (bent > 0)
            part[rows] = torch.where(shown, length / bent.clamp(min=torch.finfo(F8).tiny), torch.ones_like(bent))
            stood[rows], aimed[rows], seen[rows] = state, aim, True
        far = aim.norm(dim=1)
        last[rows] = along.norm(dim=1) / norm
        gained = torch.zeros_like(found)
        waiting = far > 0
        # A row's step is shortened until it lowers the row's error, or no longer moves its state. Why by a half: a trial at part p
        # that does not lower the row's error e shows e(p) >= e(0), while along the ellipsoid the error falls at first (the aimed
        # step is the first-order request's own: its slope there is s = -2 bend |along|^2). The quadratic through e(0), s and e(p)
        # has its least at -s p^2 / (2 (e(p) - e(0) - s p)), which is p / 2 where e(p) = e(0) and less for every worse trial.
        # Half is the largest part a failed trial leaves standing, whatever it failed by; the search takes it and uses nothing
        # else of the trial. At a matrix's own merge a part that was halved and then gains is doubled back, the same factor
        # undone, never past the whole step.
        while bool(waiting.any()):
            at = waiting.nonzero().squeeze(1)
            here = rows[at]
            trial = nearest(x[here], scale[here], radius[here], state[at] + part[here][:, None] * aim[at])
            tried = read(trial, here)
            take = tried < found[at]
            c[here[take]] = trial[take]
            gained[at[take]] = found[at[take]] - tried[take]
            waiting[at[take]] = False
            rest = at[~take]
            part[rows[rest]] *= 0.5
            waiting[rest] = part[rows[rest]] * far[rest] > EPS * norm[rest]
        floor = unheld(z.shape[1], found)   # of each row's error as this pass found it
        if behind is None:
            active[rows] = gained > floor
            again = rows[gained > floor]
            part[again] = (part[again] * 2).clamp(max=1)
        else:
            active[rows] = (gained > floor) | shown
            doubted[rows] = shown & (gained <= floor)
    return c, dict(passes=passes, largest_step_along_the_sphere_left=float(last.max()))


class Shared:
    """The least squares of some rows, decomposed: the change of the matrix (input width x output width, as the rows multiply it) whose
    rows' output changes lie nearest their targets, each row times its pull and each coordinate of the output times its count (`counts`,
    one number a coordinate), with any ridge on the change. A count multiplies a coordinate's whole least squares, so the coordinate's
    change at a ridge is the plain one at the ridge over its count. The sums, the decomposition and every ridge's division are in double
    precision. The one product a ridge then costs (the decomposition's vectors by the divided targets, input width x input width by
    input width x output width) is in single precision unless `double` asks for it: on a card that product in double was nearly all of a
    path's time, and what it makes is added to a matrix the file holds to eight bits a weight.

    With `single` the decomposition itself is made in single precision, of sums given in single precision (`Matrix.in_single`). Its
    values are then known to rounding times the largest and no nearer: `resolved` is that, or the size of a value found under nothing
    where one is larger (the sums have none), and the path ends there. No vector is cut: a value under `resolved` is kept, as nothing
    where it was found under nothing, and a ridge above it is what it is divided by. `rank` counts the values over `resolved`."""

    def __init__(self, gram, moment, rows, counts, double=False, single=False):
        width = gram.shape[0]
        if single:
            if double:
                raise ValueError('a least squares is decomposed in single precision, or its ridges\' products are made in double: not both')
            values, vectors = torch.linalg.eigh(gram.to(torch.float32))
            values = values.to(F8)
            top = float(values[-1])
            # What of this decomposition is rounding. The sums have no value under nothing, so one found there is rounding, and a value
            # as far above nothing is known no better; and no value is known nearer than rounding times the largest (LAPACK's own
            # bound for a symmetric decomposition's values). Every vector is kept all the same: at a ridge above this a value under it
            # is divided by the ridge, and what the rows ask along its vector is still moved.
            self.resolved = max(-float(values[0]), EPS4 * top)
            self.values, self.counts = values.clamp(min=0), counts.to(F8)
            self.projected = (vectors.T @ moment.to(torch.float32)).to(F8)
            self.vectors = vectors
            self.unit = float(torch.diagonal(gram).to(F8).sum()) / width   # the same unit the double decomposition counts in: its values' sum is the sums' diagonal's
            self.rank, self.width, self.rows, self.top = int((values > self.resolved).sum()), width, rows, top
            return
        values, vectors = torch.linalg.eigh(gram)
        top = float(values[-1])
        self.resolved = max(rows, width) * EPS * top   # below it a value of the decomposition is rounding
        keep = values > self.resolved
        if not bool(keep.all()):
            vectors = vectors[:, keep]
        self.values, self.counts = values[keep], counts.to(F8)
        self.projected = vectors.T @ moment
        self.vectors = vectors if double else vectors.to(torch.float32)
        self.unit = float(values.clamp(min=0).sum()) / width   # the rows' mean square, pulled, per coordinate of the input: what a ridge is counted in
        self.rank, self.width, self.rows, self.top = int(keep.sum()), width, rows, top

    def powers(self):
        """The decomposition's own ridges as powers of two of the unit, from its largest value down to where it stops resolving. `path` goes
        on above the first until a ridge moves no weight of the file."""
        if not self.rank:
            return []
        return list(range(math.ceil(math.log2(self.top / self.unit)), math.floor(math.log2(max(self.resolved, torch.finfo(F8).tiny) / self.unit)) - 1, -1))

    def on(self, device):
        """The same least squares with what it holds moved to a card or to the processor: kept off the card until a rule names its ridge."""
        self.values, self.counts, self.projected, self.vectors = (each.to(device) for each in (self.values, self.counts, self.projected, self.vectors))
        return self

    def change(self, power):
        """The change at the ridge of that power of two of the unit, in the precision the vectors are kept in."""
        ridge = self.unit * 2.0 ** power / self.counts
        return self.vectors @ (self.projected / (self.values[:, None] + ridge[None, :])).to(self.vectors.dtype)


def normal(x, pull, target, rows=None):
    """The normal equations of some rows (all, or those `rows` marks): the sum of pull^2 x x' and of pull^2 x target', in double precision."""
    width = x.shape[1]
    gram = torch.zeros((width, width), dtype=F8, device=x.device)
    moment = torch.zeros((width, target.shape[1]), dtype=F8, device=x.device)
    index = torch.arange(x.shape[0], device=x.device) if rows is None else torch.nonzero(rows).ravel()
    for first in range(0, len(index), BLOCK):
        take = index[first:first + BLOCK]
        pulled = x[take].to(F8) * pull[take, None]
        gram += pulled.T @ pulled
        moment += pulled.T @ (target[take] * pull[take, None])
    return gram, moment, len(index)


def normal_single(x, pull, target, rows=None):
    """`normal` with each block's two products made in single precision and the blocks added in double: what a block's rounding leaves
    is not carried from block to block."""
    width = x.shape[1]
    gram = torch.zeros((width, width), dtype=F8, device=x.device)
    moment = torch.zeros((width, target.shape[1]), dtype=F8, device=x.device)
    index = torch.arange(x.shape[0], device=x.device) if rows is None else torch.nonzero(rows).ravel()
    for first in range(0, len(index), BLOCK):
        take = index[first:first + BLOCK]
        pulled = x[take].to(torch.float32) * pull[take, None].to(torch.float32)
        gram += pulled.T @ pulled
        moment += pulled.T @ (target[take] * pull[take, None]).to(torch.float32)
    return gram, moment, len(index)


class Final:
    """A layer's last merge, behind its feed-forward's: there the engine adds the per-layer embedding,
        last = state + scale * norm((gelu(state @ gate) * t) @ proj),
    with `state` what the feed-forward's merge leaves and t the row's token's own per-layer input (tube.Behind, tube.per_layer_inputs). On a
    text that stays the same it is a fixed function of that state. None of the attention-output and feed-forward-down matrices writes it;
    the feed-forward's matrix reaches it through this function, so a feed-forward's rows ask there as well as at their own merge, and
    both requests stay in one sum: neither is asked in the other's place.
      gate, proj, scale   the layer's gate (width x small) and projection (small x width) as the file holds them, and the learned scale
      t, together         each row's per-layer input (rows x small), and which rows the engine reads several tokens at once: there it
                          rounds what goes into the gate and into the projection to the file's BF16
      direction, size, weight   each row's request at this merge (how far the unit state should move, how far the log of its size) and
                          its weight; a weight of nothing leaves the row's request there out of every sum"""

    def __init__(self, gate, proj, scale, t, together, direction, size, weight):
        self.gate, self.proj, self.t, self.together = gate, proj, t, together
        self.scale, self.direction, self.size, self.weight = (each.to(F8) for each in (scale, direction, size, weight))
        self.gate8, self.proj8, self.t8 = gate.to(F8), proj.to(F8), t.to(F8)

    def through(self, state, index):
        """The state the merge leaves of some rows' states, with nothing rounded: the same function with a slope everywhere, which a row's
        goal is sought through."""
        sent = (tube.gelu(state @ self.gate8) * self.t8[index]) @ self.proj8
        return state + self.scale * sent / torch.sqrt((sent * sent).mean(1, keepdim=True) + NORM_EPSILON)

    def read(self, out):
        """Of the states the merge leaves (rows x width), what is read: the unit state, its size, and a bound on how sharply the two
        bend with a step of that state. The unit state and the log of the size read a step alike in every direction, as one over
        the size, so the bound is one over the size squared, and it is met."""
        size = out.norm(dim=1)
        return out / size[:, None], size, 1 / (size * size)

    def against(self, out):
        """What a step of the state the merge leaves is read against, row by row: a step that long moves what is read by about one, in
        a direction like any other. For the unit state and the log of the size it is the state's size. It conditions the least
        squares (`reaches`) and reads nothing."""
        return out.norm(dim=1)

    def smooth(self, state, index):
        """What is read after the merge (`read`), with nothing rounded (`through`)."""
        return self.read(self.through(state, index))

    def stretch(self, state, index):
        """For some rows' states, a bound on how much longer this merge makes any step of the state. The merge is the state itself and a
        part that passes through the gate, so no step grows by more than one plus the largest that part makes of a step. With V what a
        change of the gate's output does to the state the merge leaves (scale N proj' c: c the gelu's slope times the row's per-layer
        input over the root mean square of what is sent, N the block norm's slope), that largest is the root of the largest value of
        gate'gate times V'V, both as wide as the gate's output: V'V is the projection's own products and the row's, and nothing as wide
        as the state is decomposed."""
        gate, proj, width = self.gate8, self.proj8, self.gate8.shape[0]
        lower = torch.linalg.cholesky(gate.T @ gate)
        squared = (proj * self.scale ** 2) @ proj.T
        out = torch.empty(state.shape[0], dtype=F8, device=state.device)
        many = gate.shape[1]   # rows at once: as many as the matrices decomposed here are wide. It changes no row's bound
        for first in range(0, state.shape[0], many):
            take = slice(first, first + many)
            at, t = state[take] @ gate, self.t8[index[take]]
            sent = (tube.gelu(at) * t) @ proj
            rms = torch.sqrt((sent * sent).mean(1) + NORM_EPSILON)
            normed = sent / rms[:, None]
            c = tube.gelu(at, slope=True) * t / rms[:, None]
            p, crossed = normed @ proj.T, (normed * self.scale ** 2) @ proj.T
            within = squared[None] - (crossed[:, :, None] * p[:, None, :] + p[:, :, None] * crossed[:, None, :]) / width
            within = within + ((normed * normed * self.scale ** 2).sum(1) / width ** 2)[:, None, None] * (p[:, :, None] * p[:, None, :])
            within = lower.T[None] @ (c[:, :, None] * within * c[:, None, :]) @ lower[None]
            out[take] = 1 + torch.sqrt(torch.linalg.eigvalsh((within + within.transpose(1, 2)) / 2)[:, -1].clamp(min=0))
        return out

    def engine(self, state, index):
        """The states the merge leaves of some rows' states as the engine computes it: the products in float32, and what goes into the
        gate and into the projection rounded to the file's BF16 at the rows read with others."""
        merged, together = state.to(torch.float32), self.together[index]
        sent = tube.rounded_where(tube.gelu(tube.rounded_where(merged.clone(), together) @ self.gate) * self.t[index], together)
        sent = (sent @ self.proj).to(F8)
        return merged.to(F8) + self.scale * sent / torch.sqrt((sent * sent).mean(1, keepdim=True) + NORM_EPSILON)

    def left(self, state, index):
        """What is read after the merge as the engine computes it (`engine`, `read`): the unit state and its size. Every reading of a
        matrix goes through this."""
        return self.read(self.engine(state, index))[:2]

    def asked_of(self, state, unit, log_size):
        """Every row's request here from where its state should stand after this merge, and where the matrix as it stands leaves it."""
        to, size = self.left(state, torch.arange(state.shape[0], device=state.device))
        return unit - to, log_size - torch.log(size)


class Terminal(Final):
    """The model's last merge, behind the last layer's feed-forward, read as what the output norm passes on of it: the observable the
    model's last state is compared in. With z the merge's unit state, |h| its size, w the output norm's learned scale, n the width
    and c = n eps / (the last layer's output scale)^2, what is passed on has the direction w z / |w z| and a size whose log is
        log |w z| + log(n) / 2 - log1p(c / |h|^2) / 2
    (correction.passed_on, whose arithmetic this is: the raw size enters through eps alone, and that term is kept). The merge itself,
    how the engine rounds it and its stretch are Final's.
      passes_on   correction.terminal's: the output norm's learned scale (`weight`), the last layer's output scale (`scale`, one
                  positive number) and the norm's epsilon (`eps`)

    How sharply this reading bends with a step dh of the merge's state. The unit state moves by dz and the log of the size by ds, with
    |dz|^2 + ds^2 = |dh|^2 / |h|^2. With a = w dz / |w z|, no longer than m |dz| for m the largest |w| over |w z|, the passed-on
    direction moves by a's part across it and the log of the passed-on size by a's part along it plus k ds, k = c / (|h|^2 + c). The
    two together are then no longer than m |dz| + k |ds|, whose square is at most (m^2 + k^2) |dh|^2 / |h|^2: the bound `read` gives.
    In a direction like any other a is about r |dz| long, r the root mean square of w over |w z|: what `against` counts by."""

    def __init__(self, gate, proj, scale, t, together, direction, size, weight, passes_on):
        super().__init__(gate, proj, scale, t, together, direction, size, weight)
        self.passed_scale = passes_on['weight'].to(self.scale.device, F8)
        self.c = self.passed_scale.shape[0] * float(passes_on['eps']) / float(passes_on['scale']) ** 2

    def read(self, out):
        squared = (out * out).sum(1)
        turned = out / torch.sqrt(squared)[:, None] * self.passed_scale
        length = turned.norm(dim=1)
        log = torch.log(length) + 0.5 * math.log(out.shape[1]) - 0.5 * torch.log1p(self.c / squared)
        most, k = self.passed_scale.abs().max() / length, self.c / (squared + self.c)
        return turned / length[:, None], torch.exp(log), (most * most + k * k) / squared

    def against(self, out):
        size = out.norm(dim=1)
        return size * (out / size[:, None] * self.passed_scale).norm(dim=1) / torch.sqrt((self.passed_scale ** 2).mean())


def written(w, change):
    """A matrix of the file with a change, as the file's BF16 holds it weight by weight, and how much of it moved."""
    out, size = tube.bf16((w.to(F8) + change).to(torch.float32)), float(torch.linalg.vector_norm(w, dtype=F8))
    return out, dict(weights_moved=int((out != w).sum()), change_over_matrix=float((out.to(F8) - w).norm()) / size,
                     asked_change_over_matrix=float(torch.linalg.vector_norm(change, dtype=F8)) / size)


class Matrix:
    """One matrix with the rows it is solved on: its input at every row as the engine multiplies it (rows x input width, float32), the state
    its output is added to (rows x width), the matrix as the file holds it (input width x width, float32 values the BF16 holds), the block's
    learned scale, each row's request (direction: rows x width; size: rows), its weight, and which answer it belongs to (a number a row).
    A row whose weight is nothing is in no sum: its request is not known, which is not a request for nothing, and `unknown` counts such
    rows. For a feed-forward matrix `final` holds the same rows' requests at the layer's last merge (Final), and every sum holds both."""

    def __init__(self, x, incoming, w, gain, direction, size, weight, answer, final=None):
        if not bool((tube.bf16(w) == w).all()):
            raise ValueError('the matrix holds a value the file\'s BF16 does not hold: a solve stands on a matrix of the file')
        self.x, self.w, self.answer, self.final = x, w, answer, final
        self.incoming, self.gain, self.direction, self.size, self.weight = (each.to(F8) for each in (incoming, gain, direction, size, weight))
        self.raw = self.out(w)
        self.w_size = float(torch.linalg.vector_norm(w, dtype=F8))
        self.every = torch.arange(x.shape[0], device=x.device)
        asks = (((self.direction * self.direction).sum(1) + self.size * self.size) > 0) & (self.weight > 0)
        _, size_now = left(self.incoming, self.raw, self.gain)
        reach = self.weight * reaches(self.raw, self.gain, size_now)
        self.counts = self.gain ** 2 / (self.gain ** 2).mean()   # each coordinate of the output by the learned scale there
        if final is None:
            self.unknown = int((self.weight == 0).sum())
            self.asking, self.pull = asks, reach
            self.goal, self.sought = goal(self.incoming, self.raw, self.gain, self.direction * asks[:, None], self.size * asks)
        else:
            self.unknown = dict(own=int((self.weight == 0).sum()), final=int((final.weight == 0).sum()))
            asks_there = (((final.direction * final.direction).sum(1) + final.size * final.size) > 0) & (final.weight > 0)
            both = self.weight ** 2 + final.weight ** 2
            share = torch.where(both > 0, final.weight ** 2 / both.clamp(min=torch.finfo(F8).tiny), torch.zeros_like(both))
            self.asking = asks | asks_there
            there = final.engine(self.merged(self.raw), self.every)
            self.final_to, self.final_was = final.read(there)[:2]
            self.pull = torch.sqrt(reach ** 2 + (final.weight * reaches(self.raw, self.gain, final.against(there))) ** 2)
            self.goal, self.sought = goal(self.incoming, self.raw, self.gain, self.direction * asks[:, None], self.size * asks, final, share)
        self.target = self.goal - self.raw
        self.parts = None   # the rows' sums, part by part (`parted`): made when a least squares is first asked for
        self.masks = None   # which rows each part holds
        self.single = False   # the sums' products and the decompositions in single precision (`in_single`)

    def in_single(self):
        """From here the rows' sums are multiplied and their least squares decomposed in single precision (`normal_single`, `Shared`).
        Asked before any sum is made: a matrix's sums are all of one kind."""
        if self.parts is not None:
            raise ValueError('the rows\' sums are already made')
        self.single = True
        return self

    def in_double(self):
        """Back to double precision: sums made in single precision are let go, and the next least squares sums the rows again."""
        if self.single:
            self.parts = self.masks = None
        self.single = False
        return self

    def parted(self, parts=()):
        """Sums every row once, part by part. `parts` are masks of rows that share none (a path's groups left out); the rows no part
        marks are one part more, the last. Every row's sums are then the parts' own added, and the least squares without a part is
        the other parts' own added (`summed`): no row is summed twice and no sum is taken away from another. The sums are kept off
        the card between decompositions: at 12288 wide one is 1.1 GiB, and a decomposition asks eight."""
        marked = torch.zeros(self.x.shape[0], dtype=torch.long, device=self.x.device)
        for part in parts:
            marked += part
        if bool((marked > 1).any()):
            raise ValueError('two parts share a row: its sums would be counted twice')
        rest = marked == 0
        self.parts, self.masks = [], []
        for part in (*parts, *([rest] if bool(rest.any()) else [])):
            if self.single:   # kept where the rows are, in single precision: a part is half a double one's room and its decomposition asks no eight of them
                gram, moment, count = normal_single(self.x, self.pull, self.target, rows=part)
                self.parts.append((gram.to(torch.float32), moment.to(torch.float32), count))
            else:
                gram, moment, count = normal(self.x, self.pull, self.target, rows=part)
                self.parts.append((gram.cpu(), moment.cpu(), count))
            self.masks.append(part)

    def summed(self, without=None):
        """The sums of every row, or of every row outside one part of `parted` (its number), on the rows' card or processor."""
        if self.parts is None:
            self.parted()
        taken = [each for number, each in enumerate(self.parts) if number != without]
        if not taken:
            raise ValueError('no row is left to solve on')
        device = self.x.device
        gram, moment = taken[0][0].to(device, copy=True), taken[0][1].to(device, copy=True)
        for more in taken[1:]:
            gram += more[0].to(device)
            moment += more[1].to(device)
        return gram, moment, sum(each[2] for each in taken)

    @property
    def gram(self):
        """Every row's sum of pull^2 x x', off the card."""
        return self.summed()[0].cpu()

    @property
    def moment(self):
        """Every row's sum of pull^2 x target'."""
        return self.summed()[1]

    def merged(self, out, rows=None):
        """The states the matrix's own merge leaves of some rows' outputs."""
        incoming = self.incoming if rows is None else self.incoming[rows]
        return incoming + self.gain * out / torch.sqrt((out * out).mean(1, keepdim=True) + NORM_EPSILON)

    def out(self, w, rows=None):
        """The rows' outputs through a matrix, multiplied in float32 as the engine does."""
        index = torch.arange(self.x.shape[0], device=self.x.device) if rows is None else torch.nonzero(rows).ravel()
        return torch.cat([(self.x[index[first:first + BLOCK]] @ w).to(F8) for first in range(0, len(index), BLOCK)]) if len(index) else torch.zeros((0, w.shape[1]), dtype=F8, device=w.device)

    def read(self, w, rows=None):
        """What a matrix leaves of the requests of some rows (all, or those marked), each row's error times its weight squared: summed over
        the rows that ask and over the rows that ask nothing, and answer by answer."""
        rows = torch.ones(self.x.shape[0], dtype=torch.bool, device=self.x.device) if rows is None else rows
        changed = self.out(w, rows)
        turned, grown = errors(self.incoming[rows], self.raw[rows], changed, self.gain, self.direction[rows], self.size[rows])
        cost = self.weight[rows] ** 2 * (turned + grown)
        told = {}
        if self.final is not None:   # the same rows at the layer's last merge, through the merge as the engine computes it
            now, is_ = self.final.left(self.merged(changed, rows), self.every[rows])
            there = self.final.weight[rows] ** 2 * (((now - self.final_to[rows] - self.final.direction[rows]) ** 2).sum(1)
                                                    + (torch.log(is_ / self.final_was[rows]) - self.final.size[rows]) ** 2)
            told = dict(own_merge=float(cost.sum()), last_merge=float(there.sum()))
            cost = cost + there
        asking, answer = self.asking[rows], self.answer[rows]
        by_answer = {int(each): float(cost[answer == each].sum()) for each in torch.unique(answer)}
        return dict(asking=float(cost[asking].sum()), still=float(cost[~asking].sum()), all=float(cost.sum()), **told, by_answer=by_answer)

    def written(self, change):
        """The matrix with a change, as the file's BF16 holds it weight by weight, and how much of it moved (`written`)."""
        return written(self.w, change)

    def shared(self, without=None, double=False):
        """The least squares of every row, or of every row outside one part: the number of a part of `parted` (a path's group left
        out), whose rows were summed once with the others'; or a mask of rows, which are then summed here apart from the rest."""
        if without is not None and not isinstance(without, int):
            self.parted([without])
            without = 0
        gram, moment, count = self.summed(without)
        return Shared(gram, moment, count, self.counts, double, single=self.single)


def path(matrix, groups, say=None, systems=None):
    """Every ridge read on answers it was not solved on. `groups` are the fitting answers' groups (lists of answers' numbers that share
    none). For each group: the least squares of the other groups' rows, and at every ridge of its path the matrix it gives, written, read on
    the rows of the group left out. A stage that moves no weight of the file is read once (the matrix as it stands) and the path goes on
    below it. Returns, per group, its answers, the least squares' rank and unit, what the matrix as it stands leaves there, and the stages.
    Nothing is chosen. Each group's rows are summed once (`Matrix.parted`), and a matrix written after its path (`written_at`) is
    solved on those same sums added. With `systems` (a list) each group's least squares is put there, off the card, where it is
    otherwise let go: a caller that will write the matrix a group's others give, at a ridge named later, keeps it (`Shared.on`)."""
    told = []
    helds = [torch.isin(matrix.answer, torch.as_tensor(sorted(group), device=matrix.answer.device)) for group in groups]
    for held in helds:
        if not bool(held.any()) or bool(held.all()):
            raise ValueError('a group left out holds no row, or every row: nothing is read, or nothing is solved')
    matrix.parted(helds)   # every row summed once: a group's least squares is the other parts' own sums added
    for number, (group, held) in enumerate(zip(groups, helds)):
        system = matrix.shared(without=number)
        stands = matrix.read(matrix.w, held)
        stages = []

        def stage_at(power):
            written, moved = matrix.written(system.change(power))
            stage = dict(power=power, **moved, **(matrix.read(written, held) if moved['weights_moved'] else stands))
            if say:
                say(dict(group=number, **{key: value for key, value in stage.items() if key != 'by_answer'}))
            return stage

        powers = system.powers()
        for power in powers:
            stages.append(stage_at(power))
        while stages and stages[0]['weights_moved']:   # upward from the decomposition's largest, until a ridge moves no weight of the file
            stages.insert(0, stage_at(stages[0]['power'] + 1))
        told.append(dict(answers=sorted(int(each) for each in group), rows=int(held.sum()), rank=system.rank, unit=system.unit, top=system.top, resolved=system.resolved,
                         **({'arithmetic': 'single'} if matrix.single else {}), as_it_stands=stands, stages=stages))
        if systems is not None:
            systems.append(system.on('cpu'))
        del system
    return told


def written_at(matrix, power, without=None):
    """The matrix every fitting row gives at the ridge of one power of two of its own unit, as the file's BF16 holds it, with what it leaves
    of every row's request beside what the matrix as it stands leaves. With `without`, the matrix the rows outside that part give
    (`Matrix.shared`): what it leaves is read on every row all the same, the part's included."""
    system = matrix.shared(without)
    change = system.change(power)
    written, moved = matrix.written(change)
    told = dict(power=power, ridge=system.unit * 2.0 ** power, unit=system.unit, rank=system.rank, **moved, sought=matrix.sought)
    if matrix.single:
        told.update(arithmetic='single', top=system.top, resolved=system.resolved, unsolved=unsolved(matrix, change, system.unit * 2.0 ** power, without))
    return written, dict(told, before=matrix.read(matrix.w), after=matrix.read(written))


def unsolved(matrix, change, ridge, without=None):
    """How far a change is from solving the least squares it was made for, as one number: what it leaves of the rows' normal equations at
    that ridge, over their right-hand side. For rows X pulled by P, targets T and each coordinate's ridge r (the ridge over its count) that
    is X'P'P(T - X change) - change r against X'P'P T, each by its size. It is read on the rows themselves, where the least squares'
    own error T - X change is formed before anything is multiplied back, so that no two large sums are taken from each other: the
    products in single precision, the sums in double. A change that solves its least squares leaves rounding. `without` is the part of
    `parted` whose rows the change was not solved on."""
    rows = torch.ones(matrix.x.shape[0], dtype=torch.bool, device=matrix.x.device)
    if without is not None:
        rows = ~matrix.masks[without]
    index = torch.nonzero(rows).ravel()
    made = change.to(torch.float32)
    left = torch.zeros(change.shape, dtype=F8, device=change.device)
    asked = torch.zeros(change.shape, dtype=F8, device=change.device)
    for first in range(0, len(index), BLOCK):
        take = index[first:first + BLOCK]
        pulled = matrix.x[take].to(torch.float32) * matrix.pull[take, None].to(torch.float32)
        target = matrix.target[take] * matrix.pull[take, None]
        short = target - (pulled @ made).to(F8)
        left += pulled.T @ short.to(torch.float32)
        asked += pulled.T @ target.to(torch.float32)
    left -= change.to(F8) * (ridge / matrix.counts)[None, :]
    size = float(asked.norm())
    return float(left.norm()) / size if size else 0.0
