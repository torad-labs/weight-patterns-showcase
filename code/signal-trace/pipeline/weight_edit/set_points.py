"""The point of what several points span that lies nearest a row's own point.

A reference that is a set says, for a row, not one place its state should stand but every place some given points span: every mixture of
them with coefficients that are not negative and add to one (their convex hull). How far a row stands from that set is its distance to the
set's nearest point, and that point is what the row is asked for. This file finds it, and nothing else: which points make a row's set,
what a point is made of and what is done with the distance are the caller's. A comparison that measures a model against such a
reference and a solver that corrects toward it both call this one arithmetic, so the sum one measures and the sum the other lowers
cannot differ by how each found its points.

A point here is a row of numbers of any width. Where a state's unit direction and the log of its size are compared together the caller
puts the two side by side in one row (the direction's coordinates, then the log of the size): one set of coefficients then mixes both, and
the mixed direction is as long as the mixture makes it, never brought back to unit length.

The arithmetic. With y_p a set's points, c their mean, a_p = y_p - c and b = x - c for the row's own point x, the nearest point is
c + sum_p l_p a_p for the coefficients l on the simplex that leave |sum_p l_p a_p - b|^2 least. Only the products H = a a' and h = a b
enter, and they are made about the set's own mean so that points that lie close together far from the row keep the precision of what
they differ by. The least is found by Wolfe's method for the nearest point of a polytope (P. Wolfe, 1976): a subset of the points is
kept whose own flat is searched exactly, a point enters it while some point of the set still lies nearer the row than the flat's own
nearest does, and points leave it when their coefficient would turn negative. Every step of it is an exact solve of a small system; no
step length, rate or count is chosen.

What ends it, and what is returned of how well. With r = h - H l, the number g = max_p r_p - l.r is nothing at the nearest point and
bounds how far any other mixture is from it: |q - q*|^2 <= g for the q returned and the true nearest q*, and the squared distance
returned is above the true one by at most 2 g. A pair stops when g is under what rounding alone makes of products of this size (`bound`:
2 (n + 2 P) eps times the set's spread times the spread and the row's distance from the set's mean together, for points n wide and P of
them; the first-order rounding of an n-term product summed over P coefficients), or when a whole round no longer lowers the sum, which in
exact arithmetic cannot happen before the nearest point. `gap` is g as the pair ended, in the points' own squared units: it is the
accuracy of this arithmetic for that pair and says nothing about what the distance means.

Everything is in double precision, on the device the points are on.
"""
from typing import Any

import torch

F8 = torch.float64
EPS = float(torch.finfo(F8).eps)
# Bytes of a set's points held at once in `onto`: a block of pairs x the points of a set x the width, twice (the points and what they
# are about their mean). It bounds the work's memory and orders nothing a result depends on: a pair's point is found from its own
# points alone.
ROOM = 2 ** 30


def nearest(points, standing, valid=None) -> dict[str, Any]:
    """For each pair, the point of its set nearest its own point.

      points    pairs x most x width: each pair's set, a point a row. Where sets differ in size, `most` is the largest and `valid`
                says which rows are points
      standing  pairs x width: each pair's own point
      valid     pairs x most, true where `points` holds a point (all of them, when None); every pair has one at least

    Returned, a pair each: `point` (pairs x width: the nearest point), `squared` (the squared distance from `standing` to it),
    `coefficients` (pairs x most: the mixture that makes the point, nothing negative, adding to one, nothing at a row that is no
    point), `gap` and `bound` (see the module: how far the point can still be from the true nearest, squared, and what rounding alone
    makes of that), and `rounds`, one number: how many times the loop went round for the pair that took longest."""
    points, standing = points.to(F8), standing.to(F8)
    if points.ndim != 3 or standing.ndim != 2 or standing.shape != (points.shape[0], points.shape[2]):
        raise ValueError(f'points are pairs x most x width and standing pairs x width: {tuple(points.shape)} and {tuple(standing.shape)}')
    pairs, most, width = points.shape
    device = points.device
    valid = torch.ones(pairs, most, dtype=torch.bool, device=device) if valid is None else valid.to(device)
    if valid.shape != (pairs, most) or not bool(valid.any(1).all()):
        raise ValueError('valid marks pairs x most, and every pair has a point at least')
    holds = valid.to(F8)
    centre = (points * holds[:, :, None]).sum(1) / holds.sum(1, keepdim=True)
    about = (points - centre[:, None, :]) * holds[:, :, None]
    away = standing - centre
    spread = (about * about).sum(2).max(1).values                     # the farthest point from the set's mean, squared
    unit = torch.where(spread > 0, spread, torch.ones_like(spread))
    products = about @ about.transpose(1, 2) / unit[:, None, None]    # H, in units of the spread
    toward = (about @ away[:, :, None]).squeeze(2) / unit[:, None]    # h
    bound = 2 * (width + 2 * most) * EPS * (1 + away.norm(dim=1) / torch.sqrt(unit))
    found = from_products(products, toward, bound, valid)
    mixture = found['coefficients']
    point = centre + (mixture[:, :, None] * about).sum(1)
    return dict(point=point, squared=((standing - point) ** 2).sum(1), coefficients=mixture,
                gap=found['gap'] * unit, bound=bound * unit, rounds=found['rounds'])


def from_products(products, toward, bound, valid=None) -> dict[str, Any]:
    """The same simplex solve from supplied Gram/cross products.

    The caller owns centering, scale and the full-coordinate numerical account.
    This avoids rebuilding one original reference geometry for each recipient.
    No eigendecomposition or reference dimensional truncation is needed.
    """
    products, toward, bound = products.to(F8), toward.to(F8), bound.to(F8)
    if (products.ndim != 3 or products.shape[1] != products.shape[2]
            or toward.shape != products.shape[:2] or bound.shape != products.shape[:1]):
        raise ValueError('Gram, cross products and numerical bounds have incompatible shapes')
    if not all(bool(torch.isfinite(value).all()) for value in (products, toward, bound)):
        raise ValueError('the projection products and numerical bounds must be finite')
    pairs, most = toward.shape
    device = products.device
    valid = torch.ones(pairs, most, dtype=torch.bool, device=device) if valid is None else valid.to(device)
    if valid.shape != toward.shape or not bool(valid.any(1).all()):
        raise ValueError('each projection needs a nonempty reference set')
    start = torch.where(valid, products.diagonal(dim1=1, dim2=2) - 2 * toward, torch.full_like(toward, float('inf'))).argmin(1)
    mixture = torch.zeros(pairs, most, dtype=F8, device=device)
    mixture[torch.arange(pairs, device=device), start] = 1
    kept = mixture > 0                                                # the subset whose flat is searched
    ended = torch.zeros(pairs, dtype=torch.bool, device=device)
    within = torch.zeros(pairs, dtype=torch.bool, device=device)      # the subset has changed and its flat is not searched yet
    least = torch.full((pairs,), float('inf'), dtype=F8, device=device)
    rounds = 0
    while True:
        live = (~ended).nonzero().squeeze(1)
        if live.numel() == 0:
            break
        rounds += 1
        H, l, held = products[live], mixture[live], kept[live]
        moved = (H @ l[:, :, None]).squeeze(2)
        r = toward[live] - moved
        outside = ~within[live]
        if bool(outside.any()):
            top = torch.where(valid[live], r, torch.full_like(r, -float('inf'))).max(1)
            gap = top.values - (l * r).sum(1)
            value = (l * moved).sum(1) - 2 * (l * toward[live]).sum(1)          # the sum being lowered, less what no mixture changes
            stops = outside & ((gap <= bound[live]) | ~(value < least[live]))
            enters = outside & ~stops
            ended[live[stops]] = True
            entering = live[enters]
            kept[entering, top.indices[enters]] = True
            least[entering] = value[enters]
            within[entering] = True
        goes = within[live] & ~ended[live]
        if not bool(goes.any()):
            continue
        some = live[goes]
        H, l, held, r = H[goes], l[goes], kept[some], r[goes]
        inside = held.to(F8)
        system = torch.zeros(len(some), most + 1, most + 1, dtype=F8, device=device)
        system[:, :most, :most] = H * inside[:, :, None] * inside[:, None, :] + torch.diag_embed(1 - inside)
        system[:, :most, most] = inside
        system[:, most, :most] = inside
        asked = torch.cat([r * inside, torch.zeros(len(some), 1, dtype=F8, device=device)], 1)
        solved, failed = torch.linalg.solve_ex(system, asked[:, :, None])
        solved = solved.squeeze(2)
        again = (failed != 0) | ~torch.isfinite(solved).all(1)
        if bool(again.any()):   # points of the subset that lie in one another's flat: the shortest of the steps that search it
            solved[again] = (torch.linalg.pinv(system[again], hermitian=True) @ asked[again][:, :, None]).squeeze(2)
        flat = l + solved[:, :most] * inside                          # the nearest point of the subset's flat, as a mixture
        whole = ((flat > 0) | ~held).all(1)
        taken = some[whole]
        mixture[taken] = flat[whole] / flat[whole].sum(1, keepdim=True)
        within[taken] = False
        short = ~whole
        if bool(short.any()):
            l, flat, held = l[short], flat[short], held[short]
            blocks = held & (flat <= 0)
            share = torch.where(blocks, l / (l - flat), torch.full_like(l, float('inf')))
            share = torch.nan_to_num(share, nan=0.0, posinf=float('inf'))
            step = share.min(1).values.clamp(max=1)
            there = l + step[:, None] * (flat - l)
            leaves = blocks & (share <= step[:, None])
            there = torch.where(leaves | ~held, torch.zeros_like(there), there).clamp(min=0)
            mixture[some[short]] = there / there.sum(1, keepdim=True)
            kept[some[short]] = held & ~leaves
    moved = (products @ mixture[:, :, None]).squeeze(2)
    r = toward - moved
    gap = torch.where(valid, r, torch.full_like(r, -float('inf'))).max(1).values - (mixture * r).sum(1)
    return dict(coefficients=mixture, gap=gap, rounds=rounds)


def onto(table, offsets, members, at, standing, room=ROOM):
    """Each pair's nearest point, where the sets are drawn from one table of points.

      table     points x width: every point a set may hold, a row each
      offsets, members   set j is the rows members[offsets[j]:offsets[j + 1]] of the table, each row its own point (a row named twice
                is two points at one place, and changes nothing)
      at        pairs: the set of each pair
      standing  pairs x width: each pair's own point

    Returned as `nearest` returns them, a pair each: `point`, `squared`, `gap`, `bound`; `support` (how many of the set's points carry
    the nearest point); `coefficients` with `first`: pair p's are coefficients[first[p]:first[p + 1]], in the order of its set's rows;
    and `rounds`. Pairs whose sets are the same size are worked together, so no set is padded."""
    device = standing.device
    offsets, members, at = (torch.as_tensor(each, device=device).long() for each in (offsets, members, at))
    if at.ndim != 1 or standing.shape != (len(at), table.shape[1]):
        raise ValueError(f'standing is pairs x width: {tuple(standing.shape)} for {len(at)} pairs of points {table.shape[1]} wide')
    if len(at) and (int(at.min()) < 0 or int(at.max()) >= len(offsets) - 1):
        raise ValueError('a pair names a set that is not there')
    sizes = (offsets[1:] - offsets[:-1])[at]
    if bool((sizes <= 0).any()):
        raise ValueError(f'{int((sizes <= 0).sum())} pairs are given a set with no point')
    pairs, width = len(at), table.shape[1]
    point = torch.empty(pairs, width, dtype=F8, device=device)
    squared, gap, bound = (torch.empty(pairs, dtype=F8, device=device) for _ in range(3))
    support = torch.empty(pairs, dtype=torch.long, device=device)
    first = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(sizes, 0)])
    coefficients = torch.empty(int(first[-1]), dtype=F8, device=device)
    rounds = 0
    for size in torch.unique(sizes).tolist():
        which = (sizes == size).nonzero().squeeze(1)
        along = torch.arange(size, device=device)
        block = max(1, room // (2 * size * width * 8))
        for begin in range(0, len(which), block):
            some = which[begin:begin + block]
            rows = members[offsets[at[some]][:, None] + along[None, :]]
            found = nearest(table[rows], standing[some])
            point[some], squared[some], gap[some], bound[some] = found['point'], found['squared'], found['gap'], found['bound']
            support[some] = (found['coefficients'] > 0).sum(1)
            coefficients[(first[some][:, None] + along[None, :]).reshape(-1)] = found['coefficients'].reshape(-1)
            rounds = max(rounds, found['rounds'])
    return dict(point=point, squared=squared, gap=gap, bound=bound, support=support, coefficients=coefficients, first=first, rounds=rounds)
