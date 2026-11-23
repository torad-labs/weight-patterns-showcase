"""One matrix of a correction-only training: one objective, and nothing beside it.

The round's direction of November 12, 2026, and before it of November 10: protection is no objective of its own, the training knows where the
overlap exists, and nothing applies where the computation already agrees. The words it was given in are kept in the repository's journal
(journal/2026-11-12-correction-not-protection.md, journal/2026-11-10-controlled-training.md). They are not repeated here: this file is sent to
machines that are not ours, and nobody's name or words go there with it.

The trace is read in cells: a group's mean at a place of the axis, of the direction of the state and of the log of its size, at a merge. Every
cell has a target, a radius and a weight, all fixed from the original model alone before any solve (a specification: Cells). A cell's error is
how far its mean stands from its target beyond the radius, and the objective is the sum over the cells of weight x error squared. A cell inside
its radius adds nothing and pushes on nothing: where the computation already agrees, nothing is applied. A cell that an earlier change has carried
out of its radius is an error like any other, in the same sum. There is no bound to hold, no bringing back and no second objective: tube.py's
brought_back, its bounded families and its drawing in are not called from here.

What is Astra's and only carried out here (records/retrospectives/2026-11-12-correction-objective-astra.md): which cells there are and each one's
target, radius and weight, made by `reference` from the original model's own recordings and read back by `cells_of`; and which matrix of a path
is kept (`chosen`). `path` returns every stage's end and chooses none.

What it shares with tube.py, unchanged: the rows and their layout (Trace, Part), the merges and what an output change does to them to first order
(Merge, Stand), the layer's last merge behind a feed-forward's (Behind), and the writing of a wanted matrix as one the file's BF16 holds, in the
solve's own measure (written, carried, measure_of). Every matrix that is read, compared or returned is such a matrix itself.

A round's step. With the merges taken to first order at the matrix the solve stands on, the objective is convex in the step, and it is brought down
by aiming: every cell is aimed at the nearest point of its own radius' ball (itself, where it stands inside), the step is the one whose cells lie
nearest their aims in the weights given, with the ridge on how far the matrix stands from where the path started, and the aims are taken again
from where that step puts the cells. The square of a cell's distance to its ball is never more than the square of its distance to any point of the
ball, so each pass leaves the first-order objective no higher than it found it. A written part of the step is taken when the written matrix,
read through the merges themselves, leaves less of the objective."""
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import types

import numpy as np
import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROUND_TWO_RUNS
from pipeline.weight_edit import ladder, tube
from pipeline.weight_edit.tube import DEVICE, EFFORT, PATH, RESOLVED, STILL, TINY

# The solver's two files as this process read them. A command can wait an hour for its card while the files change beside it (November 13,
# 2026: two finishings waited for card 0 across a change of this file): a record names what ran, not what lies there when the card comes.
AS_READ = dict(sha256=tube.digest(__file__), tube_sha256=tube.digest(tube.__file__))

FORMS = ('direction', 'size')   # the two forms a cell is read in: the unit state, and the log of its size
TOLD = ('loss', 'direction', 'size', 'cells_outside', 'furthest')   # what a record keeps of a merge's reading
HELD = 'measured, the roundings held'   # solve's `damping`: the measured second ridge, read with the roundings as they stand at the round's matrix


class Cells:
    """What the original model fixes at one merge for the correction, per group (the wrong one first) and place: for the direction, the target of
    the group's mean movement from where the original stood (2 x places x width), its radius and its weight (2 x places each); and the same three
    for the log of the size (2 x places each)."""

    def __init__(self, target, radius, weight, size_target, size_radius, size_weight):
        self.target, self.radius, self.weight = (each.float().to(DEVICE) for each in (target, radius, weight))
        self.size_target, self.size_radius, self.size_weight = (each.float().to(DEVICE) for each in (size_target, size_radius, size_weight))
        self.places = self.radius.shape[1]

    def errors(self, mean, size):
        """How far each cell stands from its target beyond its radius: nothing inside."""
        return ((mean - self.target).norm(dim=-1) - self.radius).clamp(min=0), ((size - self.size_target).abs() - self.size_radius).clamp(min=0)

    def aims(self, mean, size):
        """The nearest point of each cell's ball: the cell itself where it stands inside."""
        away = mean - self.target
        scale = (self.radius[..., None] / away.norm(dim=-1, keepdim=True).clamp(min=TINY)).clamp(max=1)
        return self.target + away * scale, self.size_target + (size - self.size_target).clamp(-self.size_radius, self.size_radius)

    def read(self, mean, size):
        """The objective's part at this merge, with its two forms and its two groups apart, and how many cells stand outside their radius."""
        turned, grown = self.errors(mean, size)
        direction, sized = (self.weight * turned ** 2).double(), (self.size_weight * grown ** 2).double()
        return dict(loss=float(direction.sum() + sized.sum()), direction=[float(each) for each in direction.sum(1)], size=[float(each) for each in sized.sum(1)],
                    cells_outside=dict(direction=[int(each) for each in (turned > 0).sum(1)], size=[int(each) for each in (grown > 0).sum(1)]),
                    furthest=dict(direction=float((turned / self.radius.clamp(min=TINY)).max()), size=float((grown / self.size_radius.clamp(min=TINY)).max())))

    def weighed(self, weight, size_weight):
        """The same cells with other weights, as new ones: these are not changed. Cells that find their targets another way (mixture.Mixed)
        stay what they are under it."""
        return Cells(self.target, self.radius, weight, self.size_target, self.size_radius, size_weight)


def without_correct(cells):
    """The same cells with the correct group's own cost left out: its targets and radii stand, and its weights are nothing. For an objective in
    which each correct member's own error stands in the correct group's place (`charged`), so that one error is not counted a second time
    through the group's mean (Astra, November 12: records/retrospectives/2026-11-12-correct-members-native-astra.md). These are new Cells:
    the reference's own are not changed, and what a member's error weighs is taken from those, never from these, whose weight for it is
    nothing. A reading still tells how many of the correct group's cells stand outside their radius; with these that is a description
    and no part of the sum."""
    kept = torch.tensor([1.0, 0.0], device=cells.weight.device)[:, None]   # the wrong group's weight, and none of the correct group's
    return cells.weighed(cells.weight * kept, cells.size_weight * kept)


def without_groups(cells):
    """The same cells with both groups' own cost left out: targets and radii stand, and every weight is nothing. For an objective whose terms
    are all members' own: the row-level requests of the answers asked to move (`asked_of`) and each correct member's own error (`charged`).
    The wrong group's mean is then asked for nothing by itself: what was the full cohort's mean gap at every place is no target here, and a
    cell with no member term is an unsupported cell, which the sum does not hold (Astra, November 12, 2026:
    records/retrospectives/2026-11-12-fork-request-decision-astra.md, round four). These are new Cells: the reference's own are not changed,
    and what a member's term weighs is taken from those."""
    return cells.weighed(torch.zeros_like(cells.weight), torch.zeros_like(cells.size_weight))


def means(part, turned, grown, stood=None):
    """Each group's mean, over its members, of each member's mean at each place of its steps' movement of the unit state (2 x places x width) and of
    their change of the log of the size (2 x places); the wrong group first. With `stood` (members x places x width, members x places) the rows
    given are states as they are, not movements, and each member's mean at a place is counted from where that member stood there."""
    return grouped(part, *member_means(part, turned, grown, stood))


def member_means(part, turned, grown, stood=None):
    """Each member's mean at each place of its steps' movement of the unit state (members x places x width) and of their change of the log of the
    size (members x places); with `stood`, of the states as they are, counted from where each member stood there (`means`)."""
    members = torch.zeros((part.n * part.places, turned.shape[1]), device=DEVICE).index_add_(0, part.cells, turned[part.at] * part.weight[:, None]).view(part.n, part.places, -1)
    sizes = torch.zeros(part.n * part.places, device=DEVICE).index_add_(0, part.cells, grown[part.at][:, 0] * part.weight).view(part.n, part.places)
    if stood is not None:
        members, sizes = members - stood[0], sizes - stood[1]
    return members, sizes


def grouped(part, members, sizes):
    """The two groups' means over their members, the wrong group first."""
    groups = lambda values: torch.stack([values[part.wrong].mean(0), values[~part.wrong].mean(0)])
    return groups(members), groups(sizes)


def quiet_of(part, frames, members='all', places=True, prompt=True):
    """What an answer's own movement weighs, in the same sum as the cells, where a cell asks for no move. Made as a DIAGNOSTIC's requests
    (runs/separable-at-17-original; Astra's assessment of November 12 in records/retrospectives/2026-11-12-next-loop-astra.md). Since
    that run, a correct member's own error at every place and prompt position is the correct group's part of one declared objective
    (runs/correction-members-at-17-original; records/retrospectives/2026-11-12-correct-members-native-astra.md), there in the place of
    the correct group's cells (`without_correct`) and not beside them. The wrong members' requests stay a diagnostic's.

    A cell charges a group's mean only beyond its radius, so inside the radius a solve moves a mean freely, and it does (rows_moved.py:
    correction-at-19's candidate carried the correct group's mean from 0.11 of its radius, the median cell, to 0.89, and the prompt's positions,
    which no cell reads, as far as the sign's line). Here, per merge read, a member's own mean movement at a place where its group's cell asks
    for no move (the target lies inside the radius: every cell of the correct group, and the wrong group's where the gap is not resolved) is
    charged whole: its square by the cell's own weight over the group's number of members, so that a group's members together weigh what the
    group's cell weighs; and every position of a member's prompt likewise, as a place of that member alone. Movement is counted from the
    original, as the cells count. No radius lies under these and no number of ours is in them.

    members  whose movement is charged: 'correct' (a correct member's original computation on the same text is a known correct reference),
             'wrong' (an unresolved wrong cell is not established agreement, and a wrong question's prompt state is no known correct
             reference: charged only to see what the request costs), or 'all'
    places   the answer's places, where the member's group's cell asks for no move
    prompt   the member's prompt positions
    Returns, per merge: direction and size (members x places), prompt and prompt_size (question rows)."""
    out = []
    counts = torch.where(part.wrong, part.wrong.sum(), (~part.wrong).sum()).float()   # the size of each member's group
    group = (~part.wrong).long()
    charged = dict(all=torch.ones_like(counts), correct=(~part.wrong).float(), wrong=part.wrong.float())[members]
    for cells in frames:
        still = (cells.target.norm(dim=-1) <= cells.radius).float(), (cells.size_target.abs() <= cells.size_radius).float()   # 2 x places: no move is asked
        per = lambda weight, asked_none: (weight * asked_none)[group] * (charged / counts)[:, None] * (1.0 if places else 0.0)
        at_prompt = lambda weight: (weight[group, 0] * charged / counts)[part.asker] * (1.0 if prompt else 0.0)
        out.append(dict(direction=per(cells.weight, still[0]), size=per(cells.size_weight, still[1]), prompt=at_prompt(cells.weight), prompt_size=at_prompt(cells.size_weight)))
    return out


REQUESTS = dict(correct_places=dict(members='correct', prompt=False), correct_prompt=dict(members='correct', places=False),
                wrong_prompt=dict(members='wrong', places=False), wrong_unresolved=dict(members='wrong', prompt=False))   # quiet_of's, one by one


def charged(part, frames, names):
    """The requests named (REQUESTS) as one, for this part's answers: per merge, the sum of their weights. None where none is named.
    `frames` are the reference's own cells, whose weights the requests take: never cells a group's cost has been left out of
    (`without_correct`), which would weigh that group's members at nothing."""
    if not names:
        return None
    each = [quiet_of(part, frames, **REQUESTS[name]) for name in names]
    return [{key: sum(one[k][key] for one in each) for key in each[0][k]} for k in range(len(frames))]


def asked_of(part, merges, requests):
    """A member's own target at a place, in the same sum as the cells: the row-level request (Astra, November 12, 2026:
    records/retrospectives/2026-11-12-fork-request-decision-astra.md). Off unless a part is given one: a part without `asked` is read as before.

    A cell asks a group's mean for a move, and `quiet_of` charges a member's own mean movement where no move is asked. Here one member is asked
    a move of its own at one place of the axis; where the member has one row there, that is the row's own target. A member that stands at its
    target costs nothing, and its error is the square of how far its mean movement lies from the target, by the weight given. Each request names
      member, place         whose mean, at which place: the member's number in this part and the place's number on the axis
      merge                 which of the `merges` merges read, in the frames' order
      direction             the movement asked of the member's mean unit state there, counted from where the original stood (width); None for none
      size                  the movement asked of the log of its size; None for none
      weight, size_weight   what the square of the error weighs, for each form that is asked
      source                the record that the target and its weight are from: a request without one is refused
    Returns `asked`, per merge: the targets (direction: members x places x width; size: members x places; nothing where none is asked) and
    which cells are asked (at, at_size); and `quiet`, per merge: what each asked cell weighs, in `quiet_of`'s form, for `requested` to add to
    what a part is charged. Two requests of one cell and form are refused, as a target without a weight is: one row is not given two targets."""
    width = None
    for each in requests:
        if not str(each.get('source') or '').strip():
            raise ValueError('a row-level request names the record its target is from')
        if each.get('direction') is None and each.get('size') is None:
            raise ValueError('a row-level request asks a direction, a size or both')
        if each.get('direction') is not None:
            width = each['direction'].shape[-1] if width is None else width
            if each['direction'].shape != (width,):
                raise ValueError('the directions asked are not of one width')
    asked = [dict(direction=None if width is None else torch.zeros((part.n, part.places, width), device=DEVICE), size=torch.zeros((part.n, part.places), device=DEVICE),
                  at=torch.zeros((part.n, part.places), dtype=torch.bool, device=DEVICE), at_size=torch.zeros((part.n, part.places), dtype=torch.bool, device=DEVICE)) for _ in range(merges)]
    prompt = len(part.asker) if hasattr(part, 'asker') else 0
    quiet = [dict(direction=torch.zeros((part.n, part.places), device=DEVICE), size=torch.zeros((part.n, part.places), device=DEVICE),
                  prompt=torch.zeros(prompt, device=DEVICE), prompt_size=torch.zeros(prompt, device=DEVICE)) for _ in range(merges)]
    for each in requests:
        k, m, q = each['merge'], each['member'], each['place']
        if not (0 <= k < merges and 0 <= m < part.n and 0 <= q < part.places):
            raise ValueError(f'no merge {k}, member {m} or place {q} in this part')
        for form, target, where, weight in (('direction', 'direction', 'at', 'weight'), ('size', 'size', 'at_size', 'size_weight')):
            if each.get(target) is None:
                continue
            if bool(asked[k][where][m, q]):
                raise ValueError(f'member {m} at place {q} is given two targets for its {form} at merge {k}')
            if not each.get(weight, 0) > 0:
                raise ValueError(f'member {m} at place {q} is asked a {form} with no weight')
            asked[k][where][m, q] = True
            asked[k][target][m, q] = each[target].float().to(DEVICE) if form == 'direction' else float(each[target])
            quiet[k][form][m, q] = float(each[weight])
    return dict(asked=asked, quiet=quiet, sources=sorted({each['source'] for each in requests}))


def requested(part, made, quiet=None):
    """Puts the row-level requests `asked_of` made on a part, beside what the part is charged already (`quiet`: charged's; None for nothing):
    the part's `quiet` becomes the two together and its `asked` the targets. Refused where a cell that is asked a move is already charged for
    any movement: a decision the original got wrong and a correction overrode is no computation to hold in place, and one row is not asked to
    stay and to move (Astra, November 12, 2026). Which rows are held and which are asked is the declaration's to say, with each one's source."""
    if quiet is not None:
        for k, (held, new) in enumerate(zip(quiet, made['asked'])):
            for form, where in (('direction', 'at'), ('size', 'at_size')):
                both = (held[form] > 0) & new[where]
                if bool(both.any()):
                    m, q = (int(v) for v in both.nonzero()[0])
                    raise ValueError(f'member {m} at place {q} is asked to stay and to move in its {form} at merge {k}')
    part.asked = made['asked']
    part.quiet = made['quiet'] if quiet is None else [{key: held[key] + new[key] for key in held} for held, new in zip(quiet, made['quiet'])]
    return part


def dressed(part, frames, charges=(), asks=None, picked=None):
    """A part with what the sum holds beside the cells: the requests named (`charged`) and the row-level requests of the answers it holds.
      asks     row-level requests by answer: each names `answer` (its number among the fitting answers), place, merge, direction and/or
               size, and source. None for none, which leaves the part as `charged` alone makes it.
      picked   which of the fitting answers this part holds (a mask, in their order): an answer's number in the part is its rank there
    A request weighs what a member's own term weighs in `quiet_of`: the reference's own cell of the wrong group at that merge and place, over
    the number of wrong members in this part, so that the part's wrong members together weigh what their group's cell weighs there. A
    request of an answer the part does not hold is not this part's."""
    quiet = charged(part, frames, charges) if charges else None
    if quiet is not None:
        part.quiet = quiet
    if asks:
        local = {int(answer): member for member, answer in enumerate(np.flatnonzero(np.asarray(picked)))}
        wrong = int(part.wrong.sum())
        mine = [dict(member=local[each['answer']], place=each['place'], merge=each['merge'], direction=each.get('direction'), size=each.get('size'), source=each['source'],
                     weight=float(frames[each['merge']].weight[0, each['place']]) / wrong, size_weight=float(frames[each['merge']].size_weight[0, each['place']]) / wrong)
                for each in asks if each['answer'] in local]
        if mine:
            requested(part, asked_of(part, len(frames), mine), quiet)
    return part


def rested(part, moved):
    """What `quiet_of` weighs, where the rows stand (Stand.moved): per merge, each member's mean movement at each place and of the log size, and
    every prompt position's own. Where the part has row-level requests (`asked_of`), a member's movement at an asked place is counted from its
    target: what a step does to it (`rested_change`) is what it was, so the same sum is the square of how far the member lies from the target."""
    stood, asked = getattr(part, 'stood', None), getattr(part, 'asked', None)
    out = []
    for k, (turned, grown, turned_q, grown_q) in enumerate(moved):
        members, sizes = member_means(part, turned, grown, None if stood is None else stood[k])
        if asked is not None:
            members = members if asked[k]['direction'] is None else members - asked[k]['direction']
            sizes = sizes - asked[k]['size']
        out.append((members, sizes, turned_q, grown_q[:, 0]))
    return out


def asked_sum(part, at):
    """The row-level requests' own part of what `quiet_sum` sums (at: `rested`): per merge, the weighted squares at the asked cells alone."""
    return [float((q['direction'] * a['at'] * (members ** 2).sum(-1)).sum() + (q['size'] * a['at_size'] * sizes ** 2).sum())
            for q, a, (members, sizes, _, _) in zip(part.quiet, part.asked, at)]


def asked_members(part, at):
    """The same weighted squares member by member, the merges summed: what each member's own asked cells keep of its request (at: `rested`).
    One number for each member of the part, in the part's own order (an answer's rank among the fitting answers the part holds), and None
    for a member that is asked nothing. They add up to the sum of `asked_sum`. A group's sum does not say which of its members a matrix
    carries: every sign-row solve up to November 13, 2026 left one fold's held builders three to five times what the other two folds' kept,
    and no record said which builders (records/development/2026-11-12-correction-objective/README.md, who is carried)."""
    total = sum((q['direction'] * a['at'] * (members ** 2).sum(-1)).sum(1) + (q['size'] * a['at_size'] * sizes ** 2).sum(1)
                for q, a, (members, sizes, _, _) in zip(part.quiet, part.asked, at))
    asked = sum((a['at'] | a['at_size']).any(1).int() for a in part.asked) > 0
    return [float(value) if bool(is_asked) else None for value, is_asked in zip(total, asked)]


def rested_change(part, moves):
    """What a step does, to first order, to what `rested` reads (Stand.moves): nothing of it is counted from anywhere."""
    return [(*member_means(part, turned, grown), turned_q, grown_q[:, 0]) for turned, grown, turned_q, grown_q in moves]


def quiet_sum(quiet, at, halves=False):
    """The quiet places' part of the objective where they stand (`at`: rested, or what a step does to them): per merge, the weighted squares."""
    total = [(q['direction'] * (members ** 2).sum(-1)).sum() + (q['size'] * sizes ** 2).sum() + (q['prompt'] * (turned_q ** 2).sum(1)).sum() + (q['prompt_size'] * grown_q ** 2).sum()
             for q, (members, sizes, turned_q, grown_q) in zip(quiet, at)]
    return [0.5 * each for each in total] if halves else total


def seen(part, moved):
    """The groups' means at every merge a stand reads (Stand.moved), each counted as the part's rows are: from the row of the original that each
    row is read beside, or, for rows of another text, from where each answer stood in the original place by place (Trace, `stood`)."""
    stood = getattr(part, 'stood', None)
    return [means(part, turned, grown, None if stood is None else stood[k]) for k, (turned, grown, _, _) in enumerate(moved)]


def nothing_left(part):
    """How a solve or a walk says that its sum is zero: with the cells alone, every cell stands inside its radius; with requests charged
    beside them, that and no charged movement."""
    return 'every cell stands inside its radius' if getattr(part, 'quiet', None) is None else 'nothing of the objective is left: every cell it holds stands inside its radius and nothing charged has moved'


def read(part, frames, weights, g, stand=None):
    """The objective of these answers at these weights, through the merges themselves: at the matrix's own merge and, behind a feed-forward's, at the
    layer's last (frames: the Cells of each, in that order). Where cells find their targets by a finite search (mixture.Mixed), `loss` is what
    the search attained and `bound` how much less the objective itself can be; with the cells of this file there is no such entry."""
    with torch.no_grad():
        stand = standing(part, weights, g) if stand is None else stand
        merges = [cells.read(*each) for cells, each in zip(frames, seen(part, stand.moved))]
        bound = dict(bound=sum(each.get('bound', 0.0) for each in merges)) if any('bound' in each for each in merges) else {}
        quiet = getattr(part, 'quiet', None)
        if quiet is not None:   # development: what is applied where nothing is asked (quiet_of) and what is left of a member's own target (asked_of), in the same sum
            rest = rested(part, stand.moved)
            applied = [float(each) for each in quiet_sum(quiet, rest)]
            own = dict(asked=asked_sum(part, rest), asked_by_member=asked_members(part, rest)) if getattr(part, 'asked', None) is not None else {}   # the row-level requests' part of `quiet`, told apart, and member by member
            return dict(loss=sum(each['loss'] for each in merges) + sum(applied), merges=merges, cells=sum(each['loss'] for each in merges), quiet=applied, **own, **bound)
    return dict(loss=sum(each['loss'] for each in merges), merges=merges, **bound)


def by_group(reading):
    """A reading's two groups apart, the wrong one first: each group's cells, the row-level requests with the wrong group (they are its
    answers' own targets) and what else the part is charged with the correct group (in a candidate, each correct member's own error:
    `candidate` takes no other charge). The two add up to the reading's loss."""
    cells = [sum(each['direction'][k] + each['size'][k] for each in reading['merges']) for k in range(2)]
    asked, charged_all = sum(reading.get('asked', [])), sum(reading.get('quiet', []))
    return [cells[0] + asked, cells[1] + charged_all - asked]


def by_member(readings):
    """What each of these readings keeps of its row-level requests member by member (read's `asked_by_member`), as an entry of a record:
    `beside_members`, one list for each reading, in the order of the answers that reading holds. Nothing at all where no reading has a
    request, so that the record of a sum without requests is what it was."""
    told = [each.get('asked_by_member') for each in readings]
    return dict(beside_members=told) if any(each is not None for each in told) else {}


def searched(frames, name):
    """What the finite searches of the cells that find their targets by one did since this was last asked (mixture.Mixed.searched), as an
    entry of a record under `name`: one item for each of `frames` in their order, None for cells that make no search. Nothing at all where
    none of the cells search, so that a record of this file's own cells is what it was. Asking empties the cells' count, so every search
    stands in one entry: the entry that follows the work it was made in."""
    told = [cells.searched() if hasattr(cells, 'searched') else None for cells in frames]
    return {name: told} if any(each is not None for each in told) else {}


def basis_of(part, stand, frames, width):
    """What speeds a round's linear solves and what its writings are nearest in: the cells' action on the rows' inputs without the merges' two
    projections, as tube.action_of takes the bounds' (each cell answers to one row of inputs, the mean of its steps' inputs, each as far as an
    output change reaches its state), and the size cells' action along the one direction of the outputs that the sizes are most read off. It
    changes how fast a solve gets there and which leftover a writing prefers, not what is solved.

    The cells' action is a gram, B'B: B has one row of inputs for each cell of each merge read (`rows`: 1,920 at a feed-forward read at two
    merges), each times the root of its cell's weight. The action itself is never made. Its values that are not nothing and their vectors are
    read off B: the squares of B's singular values and its right singular vectors, as many as B has rows, or as the inputs are wide where
    that is fewer (then they are every direction of the inputs). On every direction of the inputs outside the vectors kept the action is
    nothing exactly, and a linear solve and a writing's measure are the ridge alone there (solve, tube.measure_of).

    Until November 13, 2026 the action was made as a matrix and factored: 12288 wide it is 1.1 GiB in double precision and its factoring asks
    4.5 GiB of work space beside it, which a 16 GB card did not hold beside a solve's rows (candidate:29 of sign-chain-29-34-original). And a
    factoring of the product cannot tell a small value from none where B's own can: at that link 1,142 of the 1,920 values were resolved by the
    product's factoring and all 1,920 by B's (records/development/2026-11-13-rented-solve/small_side.py, Performance; the vectors of B B'
    brought back through B do not do, they are orthonormal only within 4e-4).

    `rebuilt_within` and `orthonormal_within` say how far the vectors as they are kept, in single precision, stand from the action's own: what
    they make of inputs with no pattern in them against B'B of the same inputs, over its size, and their products with each other against
    one and nothing."""
    stack, moment, sizes = [], torch.zeros((width, width), device=DEVICE, dtype=torch.float64), []
    share = part.weight / part.members
    for cells, (now, _) in zip(frames, stand.reaches()):
        weight = cells.weight.flatten().double()
        if bool((weight < 0).any()):
            raise ValueError('a cell weighs less than nothing: the cells\' action is then no gram of rows of inputs')
        stack.append(tube.mean_inputs(part, share * now.ratio[part.at, 0]).double() * weight.sqrt()[:, None])
        scaled = now.g * now.to
        reach = (scaled - now.sent * (now.sent * scaled).sum(1, keepdim=True) / now.width) / (now.rms * now.size)
        per_row = torch.zeros(part.x.shape[0], device=DEVICE).index_add_(0, part.at, cells.size_weight[part.group, part.place] * share)
        moment += (reach * (per_row * (part.x ** 2).sum(1))[:, None]).double().T @ reach.double()
        sizes.append((reach, cells.size_weight.flatten()))
        del scaled
    spectrum, directions = torch.linalg.eigh(moment)
    toward = directions[:, -1].float()
    sized = None
    for reach, strength in sizes:   # each merge's part is made on the card and added off it, in the order it was added on it: one is 1.1 GiB at an input 12288 wide
        one = tube.gram(tube.mean_inputs(part, share * (reach @ toward)[part.at]), strength).cpu()
        sized = one if sized is None else sized.add_(one)
        del one
    leading = float(spectrum[-1] / spectrum.sum()) if float(spectrum.sum()) > 0 else None
    del moment, sizes, spectrum, directions
    stacked = torch.cat(stack)   # B: rows of inputs x inputs, in double precision (0.18 GiB at 1,920 rows of 12,288 inputs)
    del stack
    _, heights, across = torch.linalg.svd(stacked, full_matrices=False)
    values, vectors = (heights ** 2).flip(0).float(), across.flip(0).T.contiguous().float()   # ascending, as a factoring of the action itself gives them
    del heights, across
    trace = float((stacked ** 2).sum())   # the action's own trace: the sum of the squares of B
    kept = vectors.double()
    probe = torch.randn((stacked.shape[1], 4), generator=torch.Generator().manual_seed(0), dtype=torch.float64).to(DEVICE)
    exact = stacked.T @ (stacked @ probe)
    rebuilt = float(((kept * values.double()) @ (kept.T @ probe) - exact).norm() / exact.norm()) if float(exact.norm()) > 0 else 0.0
    alike = kept.T @ kept
    alike.diagonal().sub_(1)
    orthonormal = float(alike.abs().max())
    rows = int(stacked.shape[0])
    del stacked, kept, probe, exact, alike
    return dict(values=values, vectors=vectors, size_direction=toward, size_action=sized, size_direction_share=leading, trace=trace, size_trace=float(sized.diagonal().sum()),
                rank=int(values.shape[0]), rows=rows, rebuilt_within=rebuilt, orthonormal_within=orthonormal)


def with_a_term(part, frames, stand):
    """Which of a part's rows the sum reads at all, and their inputs: the answer steps and the question positions whose movement some term of
    the sum weighs (a cell's, a member's own error's, a row-level request's). Where the groups' cells are left out (without_groups) a wrong
    answer's rows are read at its requested row alone, and half the rows multiply nothing: a step is then multiplied into the rows read
    only (solve), which changes nothing of the sum, of its slope or of a step, since what a step does at a row no term reads is weighed by
    nothing. Found from the sum itself, not from how it was put together: the slope of the sum's own quadratic, with respect to an output
    change at every row, is nothing exactly at the rows no term reads (each is multiplied by weights that are nothing) and something at
    every other for a change with no pattern in it. False where every row is read: then every row is multiplied, as before.
    Returns (answer rows read, their inputs, question rows read, their inputs) or False."""
    made = torch.Generator().manual_seed(0)
    width = part.h.shape[1]
    dy = torch.randn((part.x.shape[0], width), generator=made).to(DEVICE).requires_grad_(True)
    dy_q = torch.randn((part.xq.shape[0], width), generator=made).to(DEVICE).requires_grad_(True)
    slope, slope_q = torch.autograd.grad(weighed_change(part, frames, stand.moves(dy, dy_q)), (dy, dy_q), allow_unused=True)
    read_at = lambda each, count: torch.zeros(count, dtype=torch.bool, device=DEVICE) if each is None else (each != 0).any(1)
    rows, rows_q = read_at(slope, part.x.shape[0]), read_at(slope_q, part.xq.shape[0])
    if bool(rows.all()) and bool(rows_q.all()):
        return False
    rows, rows_q = rows.nonzero()[:, 0], rows_q.nonzero()[:, 0]
    found = rows, part.x[rows], rows_q, part.xq[rows_q]
    with torch.no_grad():   # held against every row multiplied, on one step with no pattern in it: the sum's quadratic is the same
        step = torch.randn((part.x.shape[1], width), generator=made).to(DEVICE)
        whole = float(weighed_change(part, frames, stand.moves(part.x @ step, part.xq @ step)))
        mine = float(weighed_change(part, frames, stand.moves(*multiplied_where_read(part, found, step))))
    if not abs(whole - mine) <= RESOLVED * abs(whole):
        raise ValueError(f'the rows found to carry a term do not carry the whole sum: its quadratic is {whole} on every row and {mine} on them')
    part.term_rows_told = dict(answer_rows_read=int(rows.numel()), of_answer_rows=int(part.x.shape[0]), question_rows_read=int(rows_q.numel()), of_question_rows=int(part.xq.shape[0]),
                               quadratic_on_every_row=whole, quadratic_on_the_rows_read=mine)
    return found


def weighed_change(part, frames, moves):
    """The sum's own quadratic of a change (Stand.moves): every cell's weight on the square of what the change does to its mean, and what the
    part is charged beside the cells on the squares of what it does there."""
    value = 0
    for cells, (turned, grown, _, _) in zip(frames, moves):
        mean, size = means(part, turned, grown)
        value = value + (cells.weight * (mean ** 2).sum(-1)).sum() + (cells.size_weight * size ** 2).sum()
    quiet = getattr(part, 'quiet', None)
    if quiet is not None:
        value = value + sum(quiet_sum(quiet, rested_change(part, moves)))
    return value


def multiplied_where_read(part, live, p):
    """A step's output change at every row of a part: the rows' inputs times the step at the rows the sum reads (with_a_term), and nothing at
    the others, which are not multiplied."""
    if live is False:
        return part.x @ p, part.xq @ p
    rows, x, rows_q, xq = live
    blank = lambda count: torch.zeros((count, p.shape[1]), device=p.device, dtype=p.dtype)
    return blank(part.x.shape[0]).index_copy(0, rows, x @ p), blank(part.xq.shape[0]).index_copy(0, rows_q, xq @ p)


def unit_of(part, frames, w, g):
    """The ridge that a path of ridges is counted from: the mean, over the directions of a change, of the objective's own curvature on these rows
    at the matrix as it stands in the original (the cells' weights on the rows' inputs, each as far as an output change reaches its state; the
    size cells along the one direction of the outputs their sizes are read off)."""
    with torch.no_grad():
        found = basis_of(part, standing(part, w, g), frames, w.shape[1])
    return (found['trace'] * float((g ** 2).mean()) + found['size_trace'] / w.shape[1]) / w.shape[0]


def strongest(part, frames, g, stands):
    """The strongest ridge at which a stage can be told from none: the one where a first step, taken to first order, brings the objective down
    by RESOLVED of itself, which is the fall a solve's rounds end at. Far above the objective's own curvature the step at a ridge is the
    objective's slope over the ridge, and it brings the objective down by twice the slope's square over the ridge (the slope being that of half
    the objective, as the solve writes it): so the ridge sought is 2 |slope|^2 / (RESOLVED x objective), at the matrix the path starts from.
    Nothing where no cell stands outside its radius."""
    loss = read(part, frames, stands, g)['loss']
    if loss == 0:
        return None
    stand = standing(part, stands, g)
    with torch.no_grad():
        base = seen(part, stand.moved)
        aims = [cells.aims(mean, size) for cells, (mean, size) in zip(frames, base)]
    step = torch.zeros_like(stands).requires_grad_(True)
    value = 0
    moves = stand.moves(part.x @ step, part.xq @ step)
    for cells, (at, at_size), (turned, grown, _, _), (aim, aim_size) in zip(frames, base, moves, aims):
        mean, size = means(part, turned, grown)
        value = value + 0.5 * (cells.weight * ((at + mean - aim) ** 2).sum(-1)).sum() + 0.5 * (cells.size_weight * (at_size + size - aim_size) ** 2).sum()
    quiet = getattr(part, 'quiet', None)
    if quiet is not None:
        with torch.no_grad():
            rest = rested(part, stand.moved)
        value = value + sum(quiet_sum(quiet, [tuple(a + b for a, b in zip(here, there)) for here, there in zip(rest, rested_change(part, moves))], halves=True))
    slope = torch.autograd.grad(value, step)[0]
    return 2 * float((slope.double() ** 2).sum()) / (RESOLVED * loss)


def first_power(part, frames, g, stands, unit):
    """The power of two, over the unit, that a path of ridges starts at: the last one not above `strongest`. None where nothing is to be done."""
    most = strongest(part, frames, g, stands)
    return None if most is None else math.floor(math.log2(most / unit))

def solve(part, frames, w, g, stands, ridge, origin=None, effort=EFFORT, basis=None, beside=(), say=None, damping=0.0, geometry=None):
    """The rounds at one ridge, from the matrix as it stands (on the file's grid; w is the original, which a change is reported from).

    origin  the matrix the ridge counts from: the one the path of ridges started from. Without it the ridge counts from w.
    beside  answers read after every step that is taken and never used by a step
    A written part of a step is taken when its matrix, read through the merges, leaves less of the objective. Where a reading is an attained
    value with a bound under it (read: cells that find their targets by a finite search), less means under the standing value less its
    bound; a part that leaves less only inside that bound is refused, and the round's record says which parts were.
    damping a second ridge, on how far a round's step goes from the matrix the round stands on (the ridge itself holds the matrix to where the
            path started, and holds nothing of a step along a direction the cells barely read). Nothing by default: the declared solves of
            correction-at-19 and correction-at-17-original had none. A number is that ridge at every round (damped_step.py). 'measured' takes
            it from the merges themselves. A round's step p is solved in a model that takes the merges to first order: it brings down the
            objective to first order plus ridge x |start to p|^2 plus second ridge x |p|^2. What the first order leaves out is read at the
            step: (the objective through the merges at the step as solved, before any writing, less the objective the model gave there)
            over |p|^2. That is the least second ridge whose own term makes up the difference, so that the model the step was solved in lies
            no lower than the merges at the step, and what brings the model down brings the objective down. Where the reading is larger than
            the second ridge a step was solved with, the step is solved again with the reading. The second ridge only rises, within a solve
            and from one solve of a path to the next (kept in `basis`; nothing at a path's start): a solve that began again from nothing
            took an unheld step at its new ridge, whose reading was 79 units where the path had needed 0.3 to 2 (measured_step.py, the
            first reading). A difference between the merges and the model that is no more than RESOLVED of the objective is not read. No
            number of ours is in it. Development only until a run declares it.
            HELD is the same rule with one thing changed in what it reads. Behind a feed-forward matrix the layer's last merge is read
            too, and where the engine reads several tokens at once it rounds what goes into the per-layer embedding to the file's BF16
            (tube.Behind). A rounding that turns over under a step has no slope: what it adds to the objective does not shrink with the
            square of the step, so over |p|^2 it reads as a second ridge that grows as the step shrinks, and no second ridge makes up
            for it. Read as 'measured' reads it, one such reading at a strong ridge held the whole path that followed
            (sign-chain-29-34-original, link 33: records/development/2026-11-12-correction-objective/README.md, the rounding read as
            curvature). With HELD the objective at the step is read with every rounding holding what it added at the round's matrix
            (tube.Stand, `rounded`): at that matrix those are the engine's own rows, their change with a step is to first order the
            one the round's model follows, slopes at the rounded states and all, and what the model leaves out is what the first order
            leaves out of a function that has a slope. Which written step is taken is read through the merges themselves, as ever, and
            every round's record keeps both readings at the step. At a matrix with no merge behind its own nothing is rounded with a
            step, and the two are one.
    geometry the cells that what speeds the linear solves, and what a writing is nearest in, are made from (basis_of), where they are not
            the cells the sum holds: the reference's own, for a sum that some of them are left out of (without_correct). Without it, `frames`.
    Returns the last matrix reached, its reading, the readings of the answers beside, the record of every round and why the rounds ended."""
    weights, spread = stands, float((g ** 2).mean())
    quiet = getattr(part, 'quiet', None)   # development: quiet_of
    measured = isinstance(damping, str)
    if measured and damping not in ('measured', HELD):
        raise ValueError(f'damping is a number, \'measured\' or {HELD!r}, not {damping!r}')
    mu = (float(basis.get('damping', 0.0)) if basis is not None else 0.0) if measured else float(damping)
    now = read(part, frames, weights, g)
    others = [read(each, frames, weights, g) for each in beside]
    before = searched(frames, 'searches_before_the_rounds')   # the two readings above, and whatever read these cells since they were last asked
    rounds, status, inverse, measure, made_at, vectors, scaled, source = [], 'the rounds allowed were used', None, None, None, None, None, None
    live = getattr(part, 'term_rows', None)   # the rows the sum reads at all (with_a_term): found at a part's first round and kept on it

    def prepared(second):   # what speeds the linear solves, and what a writing is nearest in, with this second ridge beside the stage's own
        under = scaled + ridge + second
        if vectors.shape[1] == vectors.shape[0]:   # the eigenvectors are every direction of the inputs
            return (lambda left, under=under: vectors @ ((vectors.T @ left) / under)), tube.measure_of([source], spread, ridge + second)
        # The action reads only the kept eigenvectors V (basis_of, `rank`); on every other direction of the inputs the same operator is the
        # ridge alone. V ((V' left) / under) + (left - V V' left) / ridge = left / ridge + V ((V' left) (1 / under - 1 / ridge)), and
        # 1 / under - 1 / ridge = -scaled / (under x ridge), written so that a value far under the ridge is not lost in a difference.
        flat = ridge + second
        apart = -scaled / (under * flat)
        return (lambda left, apart=apart, flat=flat: left / flat + vectors @ ((vectors.T @ left) * apart)), tube.measure_of([source], spread, ridge + second)

    for number in range(effort['rounds']):
        if now['loss'] == 0:
            status = nothing_left(part)
            break
        started = time.time()
        stand = standing(part, weights, g)
        with torch.no_grad():
            base = seen(part, stand.moved)
            total = weights - (w if origin is None else origin)
            rest = None if quiet is None else rested(part, stand.moved)
            roundings = stand.roundings() if damping == HELD else None   # what each rounding on the way to the merge behind adds at the round's matrix, which a step's reading holds
        if vectors is None:
            with torch.no_grad():
                found = basis_of(part, stand, frames if geometry is None else geometry, w.shape[1]) if basis is None or 'values' not in basis else basis
                if basis is not None and 'values' not in basis:
                    basis.update({name: value.cpu() if torch.is_tensor(value) else value for name, value in found.items()})
                # What speeds the linear solves: the cells' action on the inputs, in its own eigenvectors, times what an output change reaches of the
                # state through the learned scale, output by output (g squared, which the merge's two projections are left out of), plus the ridge:
                # divided through coordinate by coordinate. tube.fit takes the mean of g squared for every output, which holds every bound's solve it
                # was measured on; here nothing else stiffens the system, and at a small ridge the outputs' own scales are what is left.
                kept = int(found.get('rank', found['values'].shape[0]))   # the directions of the inputs the action reads at all (basis_of); the values ascend
                vectors = found['vectors'][:, found['values'].shape[0] - kept:].to(DEVICE).contiguous()
                scaled = found['values'][found['values'].shape[0] - kept:].to(DEVICE)[:, None] * (g.flatten() ** 2)[None, :]
                source = found if basis is None else basis
                inverse, measure = prepared(mu)
                made_at = mu
                if vectors.shape[1] < vectors.shape[0] and basis is not None and 'linear_solves' not in basis:
                    # Held once a path. Outside the vectors kept the linear solves' speed is the ridge alone, which is the action's own inverse
                    # there if the vectors make the action and are orthonormal. Both are read where the vectors are made, against the rows of
                    # inputs the action is the gram of (basis_of), with no factoring of a product in between.
                    apart = max(found['rebuilt_within'], found['orthonormal_within'])
                    if not apart <= RESOLVED:
                        raise ValueError(f"the vectors the linear solves are sped in are not the action's own: what they make stands {found['rebuilt_within']} of the action from it, "
                                         f"and their products with each other {found['orthonormal_within']} from one and nothing")
                    basis['linear_solves'] = dict(directions_kept=kept, of_inputs=int(vectors.shape[0]), rows_of_inputs_the_action_is_the_gram_of=found.get('rows'),
                                                  the_vectors_make_the_action_within=found['rebuilt_within'], the_vectors_are_orthonormal_within=found['orthonormal_within'], at_ridge=ridge + mu)
                if basis is None and not measured:
                    source = None   # nothing is made again without a second ridge that moves
                del found

        if live is None:
            live = part.term_rows = with_a_term(part, frames, stand)

        def changed(p):   # what a step does to every cell's mean at each merge read, to first order; and, where the part has quiet places, to those
            moves = stand.moves(*multiplied_where_read(part, live, p))   # nothing at the rows no term of the sum reads, which are not multiplied
            if quiet is None:
                return [means(part, turned, grown) for turned, grown, _, _ in moves], None
            each = rested_change(part, moves)
            return [grouped(part, members, sizes) for members, sizes, _, _ in each], each

        def curved(p):   # the quadratic's second derivative applied to a step: the ridge, and the cells' weights on what the step does to them
            p = p.detach().requires_grad_(True)
            value = 0.5 * (ridge + mu) * (p ** 2).sum()
            cells_change, quiet_change = changed(p)
            for cells, (mean, size) in zip(frames, cells_change):
                value = value + 0.5 * (cells.weight * (mean ** 2).sum(-1)).sum() + 0.5 * (cells.size_weight * size ** 2).sum()
            if quiet is not None:
                value = value + sum(quiet_sum(quiet, quiet_change, halves=True))
            return torch.autograd.grad(value, p)[0]

        def slope(p, aims):   # of the quadratic that aims every cell at `aims`
            p = p.detach().requires_grad_(True)
            value = 0.5 * ridge * ((total + p) ** 2).sum() + 0.5 * mu * (p ** 2).sum()
            cells_change, quiet_change = changed(p)
            for cells, (at, at_size), (mean, size), (aim, aim_size) in zip(frames, base, cells_change, aims):
                value = value + 0.5 * (cells.weight * ((at + mean - aim) ** 2).sum(-1)).sum() + 0.5 * (cells.size_weight * (at_size + size - aim_size) ** 2).sum()
            if quiet is not None:
                value = value + sum(quiet_sum(quiet, [tuple(a + b for a, b in zip(here, there)) for here, there in zip(rest, quiet_change)], halves=True))
            return torch.autograd.grad(value, p)[0]

        def modelled(p):   # the objective at a step to first order, the same with the ridge's part (what the passes bring down), and each cell's mean there
            with torch.no_grad():
                cells_change, quiet_change = changed(p)
                cells_at = [(at + mean, at_size + size) for (at, at_size), (mean, size) in zip(base, cells_change)]
                loss = sum(cells.read(mean, size)['loss'] for cells, (mean, size) in zip(frames, cells_at))
                if quiet is not None:
                    loss += float(sum(quiet_sum(quiet, [tuple(a + b for a, b in zip(here, there)) for here, there in zip(rest, quiet_change)])))
                return loss, 0.5 * loss + 0.5 * ridge * float(((total + p).double() ** 2).sum()) + 0.5 * mu * float((p.double() ** 2).sum()), cells_at

        def proposed():   # a round's step, with the second ridge as it stands: the aiming passes
            step = torch.zeros_like(weights)
            first_slope, iterations, each_pass = None, 0, []
            meant, whole, cells_at = now['loss'], 0.5 * now['loss'] + 0.5 * ridge * float((total.double() ** 2).sum()), base
            # Where no cell has a radius every aim is its target whatever the step, so the round's subproblem is one quadratic: it is solved in one
            # run of conjugate gradients with the same work a round of aiming passes has, and nothing is begun again (Astra, November 12).
            aimed = any(float(cells.radius.max()) > 0 or float(cells.size_radius.max()) > 0 for cells in frames)
            for count in range(1, (effort['passes'] if aimed else 1) + 1):
                with torch.no_grad():
                    aims = [cells.aims(mean, size) for cells, (mean, size) in zip(frames, cells_at)]
                left = -slope(step, aims)
                if first_slope is None:
                    first_slope = left.norm()
                lead = inverse(left)
                direction, inner = lead.clone(), (left * lead).sum()
                for _ in range(effort['back_iterations'] * (1 if aimed else effort['passes'])):
                    if left.norm() <= first_slope * RESOLVED:
                        break
                    bent = curved(direction)
                    rate = inner / (direction * bent).sum()
                    step += rate * direction
                    left -= rate * bent
                    lead = inverse(left)
                    inner, before = (left * lead).sum(), inner
                    direction = lead + (inner / before) * direction
                    iterations += 1
                before, (meant, whole, cells_at) = whole, modelled(step)
                each_pass.append(dict(objective_to_first_order=meant, with_the_ridge=whole, step_over_matrix=float(step.norm() / w.norm()), slope_left=float(left.norm() / first_slope)))
                if not before - whole > RESOLVED * before:   # the aims have stopped bringing the first-order objective down
                    break
            return step, meant, each_pass, iterations

        tried, iterations = [], 0
        while True:
            if made_at != mu:
                # The last second ridge's two are let go before this one's are made. A measure is two factors as wide as the inputs squared (1.1 GiB
                # each at 12288 wide), and made beside the last one a solve stood 2.3 GiB higher on the card than a solve whose second ridge
                # stays: 12.77 GiB against 10.43 (sign-33-only-original, November 13, out of memory on a 16 GB card at ridge 2^3).
                inverse = measure = None
                (inverse, measure), made_at = prepared(mu), mu
            step, meant, each_pass, used = proposed()
            iterations += used
            length = float((step.double() ** 2).sum())
            if not measured or not math.isfinite(length) or length == 0:
                break
            there = read(part, frames, weights + step, g)['loss']   # through the merges themselves, at the step as solved and before any writing
            held = there   # HELD: the same with the roundings held as they stand at the round's matrix, which is then what the second ridge is read from
            if roundings is not None:
                with torch.no_grad():
                    held = read(part, frames, None, g, stand=standing(part, weights + step, g, rounded=roundings))['loss']
            needed = (held - meant) / length if held - meant > RESOLVED * now['loss'] else 0.0
            tried.append(dict(damping=mu, needed=needed, objective_to_first_order=meant, objective_at_the_step=there,
                              **({} if roundings is None else dict(objective_at_the_step_with_the_roundings_held=held)), step_over_matrix=float(step.norm() / w.norm()), iterations=used))
            if needed <= mu or len(tried) == effort['passes']:
                break
            mu = needed
        if measured and basis is not None:
            basis['damping'] = mu   # where the next solve starts: the second ridge this one stands at
        in_step = searched(frames, 'searches_in_the_step')   # the aiming passes, the first-order reading after each, and a measured second ridge's readings at the step
        if not bool(torch.isfinite(step).all()):
            rounds.append(dict(round=number, taken=False, passes=len(each_pass), iterations=iterations, **in_step))
            status = 'the linear solve gave no finite step'
            break

        inside = []   # parts of the step whose written matrix left less as attained, by no more than the standing reading's bound

        def holds(share):
            stored = tube.written(weights + share * step, measure)
            if bool((stored == weights).all()):
                return STILL
            trial = read(part, frames, stored, g)
            # Where the standing reading is an attained value with a bound under it (read), a fall is one only below that bound: then the
            # objective itself has fallen, whatever a finite search left. With no bound this is `trial['loss'] < now['loss']`, as before.
            if trial['loss'] < now['loss'] - now.get('bound', 0.0):
                return trial, stored
            if trial['loss'] < now['loss']:
                inside.append(share)
            return None

        share, trial, halvings = tube.halved(holds)
        record = dict(round=number, objective_before=now['loss'], objective_of_the_step_to_first_order=meant, passes=len(each_pass), iterations=iterations, each_pass=each_pass,
                      step_over_matrix=float(step.norm() / w.norm()), part_of_the_step_taken=share, halvings=halvings,
                      **(dict(damping=mu, dampings_tried=tried) if measured else dict(damping=mu) if mu else {}),
                      **(dict(bound_before=now['bound'], parts_refused_inside_the_bound=inside) if 'bound' in now else {}),
                      **in_step, **searched(frames, 'searches_in_the_writings'))   # the writings: each written part of the step, read through the merges
        if trial is None or trial is STILL:
            rounds.append(dict(record, taken=False, seconds=round(time.time() - started, 1)))
            status = 'the whole step moves no weight the file holds' if trial is STILL else 'no part of a step left less of the objective'
            break
        fell = now['loss'] - trial[0]['loss']
        now, weights = trial
        others = [read(each, frames, weights, g) for each in beside]
        rounds.append(dict(record, taken=True, objective=now['loss'], merges=[{name: each[name] for name in ('loss', 'direction', 'size', 'cells_outside', 'furthest')} for each in now['merges']],
                           beside=[each['loss'] for each in others], beside_groups=[by_group(each) for each in others], **by_member(others), change_over_matrix=float((weights - w).norm() / w.norm()),
                           change_from_the_start_squared=float(((weights - (w if origin is None else origin)).double() ** 2).sum()), seconds=round(time.time() - started, 1),
                           **(searched(frames, 'searches_beside') if beside else {})))   # the readings of the answers beside, at the matrix just taken
        if say:
            say({name: rounds[-1][name] for name in ('round', 'objective_before', 'objective', 'objective_of_the_step_to_first_order', 'part_of_the_step_taken', 'step_over_matrix', 'passes', 'iterations', 'beside', 'seconds')})
        if not fell > RESOLVED * record['objective_before']:
            status = 'the objective stopped falling at this ridge'
            break
    return dict(weights=weights, reading=now, beside=others, rounds=rounds, status=status, **before)


FLOOR = lambda power: 2.0 ** power < RESOLVED   # a ridge that is less of the unit than a linear solve resolves


def placed(part, device):
    """A part's rows by the matrix's inputs, on the device named: the answer steps' and the question positions', and the same rows where the sum
    reads only some of them (with_a_term). They are nearly all a part holds on the card (0.95 GiB for thirty answers at an input 12288 wide),
    nothing writes to them, and the folds' paths and the last solve each hold their own: off the card while another path's stage is solved, a
    solve of three folds holds one path's rows there at a time (`chosen`). Nothing of the part is changed but where these lie."""
    part.x, part.xq = part.x.to(device), part.xq.to(device)
    live = getattr(part, 'term_rows', None)
    if live:
        part.term_rows = (live[0], live[1].to(device), live[2], live[3].to(device))
    return part


class Descent:
    """One path of ridges being walked, a stage at a time: the matrix it stands on, its record so far, and what its linear solves share. Every
    stage's ridge is counted from the matrix the path started from. A stage that writes no improving step leaves the matrix where it was and the
    path can go on: no motion at one ridge says nothing of a smaller one (Astra, November 12). The walk itself ends only when nothing of the
    objective is left or a linear solve gives no finite step (`ended`)."""

    def __init__(self, part, frames, w, g, stands, unit, effort=EFFORT, beside=(), say=None, damping=0.0, geometry=None):
        if not bool((tube.bf16(stands) == stands).all()):
            raise ValueError('the matrix to start from is not one the file holds')
        self.part, self.frames, self.w, self.g, self.stands, self.unit, self.effort, self.beside, self.say = part, frames, w, g, stands, unit, effort, list(beside), say
        self.damping = damping   # every stage's second ridge on a round's step (solve): nothing unless a caller names one
        self.geometry = geometry   # the cells the linear solves' speed and a writing's measure are made from, where not `frames` (solve)
        start = read(part, frames, stands, g)
        beside_start = [read(each, frames, stands, g) for each in self.beside]
        self.record = dict(before=dict(objective=start['loss'], merges=[{name: each[name] for name in TOLD} for each in start['merges']],
                                       beside=[each['loss'] for each in beside_start], beside_groups=[by_group(each) for each in beside_start], **by_member(beside_start),
                                       **({'bound': start['bound']} if 'bound' in start else {}),
                                       **searched(frames, 'searches_up_to_the_first_stage')),   # the start's readings, and whatever read these cells before the walk
                           unit=unit, stages=[])
        self.basis, self.weights = dict(), stands
        self.ended = None if start['loss'] > 0 else nothing_left(part)

    def rests(self):
        """The walk's rows off the card until its next stage (`wakes`): the answers it is solved on and the answers read beside."""
        for each in (self.part, *self.beside):
            placed(each, 'cpu')

    def wakes(self):
        for each in (self.part, *self.beside):
            placed(each, DEVICE)

    def stage(self, power, rounds=None):
        """One more stage, at the ridge unit x 2^power; with `rounds`, of at most that many rounds. Returns its record, which is also kept."""
        effort = self.effort if rounds is None else dict(self.effort, rounds=rounds)
        run = solve(self.part, self.frames, self.w, self.g, self.weights, self.unit * 2.0 ** power, origin=self.stands, effort=effort, basis=self.basis, beside=self.beside,
                    say=None if self.say is None else lambda told: self.say(dict(ridge_power=power, **told)), damping=self.damping, geometry=self.geometry)
        self.weights = run['weights']
        stage = dict(ridge_power=power, ridge=self.unit * 2.0 ** power, status=run['status'], rounds=len(run['rounds']), rounds_taken=sum(1 for each in run['rounds'] if each['taken']),
                     objective=run['reading']['loss'], merges=[{name: each[name] for name in TOLD} for each in run['reading']['merges']], beside=[each['loss'] for each in run['beside']],
                     beside_groups=[by_group(each) for each in run['beside']], **by_member(run['beside']),
                     change_over_matrix=float((self.weights - self.w).norm() / self.w.norm()), change_from_the_start_over_matrix=float((self.weights - self.stands).norm() / self.w.norm()),
                     change_from_the_start_squared=float(((self.weights - self.stands).double() ** 2).sum()), each_round=run['rounds'],
                     **({'bound': run['reading']['bound']} if 'bound' in run['reading'] else {}),
                     **({'searches_before_the_rounds': run['searches_before_the_rounds']} if 'searches_before_the_rounds' in run else {}))
        if rounds is not None:
            stage['rounds_allowed_here'] = rounds
        self.record['stages'].append(stage)
        for name, told in (('linear_solves', self.basis.get('linear_solves')), ('rows_read', getattr(self.part, 'term_rows_told', None))):
            if told is not None and name not in self.record:   # what the path's linear solves are sped in and which rows a step is multiplied into, once
                self.record[name] = told
        if self.say:
            self.say({name: stage[name] for name in ('ridge_power', 'status', 'rounds_taken', 'objective', 'beside', 'change_from_the_start_over_matrix')})
        if run['reading']['loss'] == 0:
            self.ended = nothing_left(self.part)
        elif run['status'] == 'the linear solve gave no finite step':
            self.ended = run['status']
        return stage


def path(part, frames, w, g, stands, unit, effort=EFFORT, beside=(), say=None, first=None, last=None):
    """A path of ridges from the matrix as it stands, walked to its end: the first stage at the power `first` over the unit (without it, the
    strongest ridge a stage can be told from none at: first_power), every later stage at half the ridge from where the last one ended, down to
    where a halved ridge would be less of the unit than a linear solve resolves (or to the power `last`). Every stage's end is returned, off the
    card, with the record; none is chosen here."""
    walk = Descent(part, frames, w, g, stands, unit, effort, beside, say)
    power, ends = first_power(part, frames, g, stands, unit) if first is None else first, []
    while walk.ended is None:
        walk.stage(power)
        ends.append(walk.weights.cpu())
        if walk.ended is None and last is not None and power - PATH < last:
            walk.ended = 'the last ridge asked for'
        elif walk.ended is None and FLOOR(power - PATH):
            walk.ended = 'the ridge is less of the unit than a linear solve resolves'
        power -= PATH
    walk.record['ended'] = walk.ended
    return ends, walk.record


def least(table):
    """Which setting of a cross-validation's table is taken: the least score; on an exact tie the smaller mean, over the folds, of the squared
    change of the matrix from the start, then the stronger ridge, then the earlier round (Astra, November 12). The start is a row like the
    others (ridge_power None, round 0, no change), so it wins a tie with any written state."""
    return min(table, key=lambda row: (row['score'], row['change_squared'], -(float('inf') if row['ridge_power'] is None else row['ridge_power']), row['round']))


def held_score(losses, counts, groups=None, shares=None):
    """One setting's score, from the folds' readings on the answers each was not solved on.

    Where every fold holds the two groups in the cohort's own proportion (`shares` None, as tube.picks' folds do): the folds' mean, each by
    its number of answers. A fold's reading weighs each group's members by one over that group's number in the fold, so this mean is the
    whole per-member objective over the held answers.

    Where the folds do not (case families kept in one fold): S = the sum over groups g and folds k of n[g, k] / N[g] x E[g, k], with E[g, k]
    the fold's reading of that group (`by_group`), n[g, k] the group's answers in the fold and N[g] in all the folds (`shares`: per fold,
    the two n / N). It is the same whole per-member objective; the folds' mean by their numbers of answers would weigh a group's members
    differently from fold to fold. A group with no answer in a fold adds nothing there. For proportional folds the two are one number
    (Astra, November 12, 2026: records/retrospectives/2026-11-12-ridge-end-astra.md)."""
    if shares is None:
        return float(sum(count * each for count, each in zip(counts, losses)) / sum(counts))
    return float(sum(share[k] * each[k] for share, each in zip(shares, groups) for k in range(2)))


def stalled_after(stalled, power, a_new_least, armed=False):
    """How many ridges below the unit, in a row, have scored no setting under the least score before them, once this ridge is made (Astra,
    November 12, 2026: records/retrospectives/2026-11-12-ridge-end-astra.md). Every accepted setting of the ridge is looked at, at any
    of its rounds, in the score that selects (`a_new_least`); a new least starts the count again, a fold reading less by itself does not,
    and a ridge at which no state was written has no setting and is one more without a new least. Ridges from the first down to the unit
    are not counted: that part of a path is always made.

    `armed` (his decision of November 13, 2:20 AM CT, in the same record, for a declaration that scopes it: chosen's `above`): a setting
    made before this ridge scored under the unchanged start. From then on every ridge counts, above the unit too, so a count begun above
    the unit goes on across it; until then only the ridges below the unit count, as without it."""
    if power >= 0 and not armed:
        return stalled
    return 0 if a_new_least else stalled + 1


def path_end(named, link=None):
    """The declaration's allowance of ridges as `chosen` takes it: `patience` (solver.allowance_of_ridges) and `above`, whether at this link
    the allowance also counts above the unit. A declaration scopes the second by link: solver.allowance_above_the_unit = {from_link: L}
    holds for the links from L on and for no earlier one, which keep the end they were declared and solved with (Astra, November 13, 2026:
    records/retrospectives/2026-11-12-ridge-end-astra.md). Without it the allowance counts below the unit only, as before."""
    patience, scope = named.get('allowance_of_ridges'), named.get('allowance_above_the_unit')
    if scope is None:
        return dict(patience=patience, above=False)
    if patience is None or not isinstance(scope, dict) or type(scope.get('from_link')) is not int:
        raise ValueError('an allowance above the unit widens a declared allowance of ridges and names the first link it holds from (from_link)')
    return dict(patience=patience, above=link is not None and link >= scope['from_link'])


def second_ridge(named, link=None):
    """The declaration's second ridge as `chosen` takes it (`damping`): solver.second_ridge, which is a number, 'measured' or HELD. A
    declaration scopes the held reading by link: solver.second_ridge_holds_the_roundings = {from_link: L}, beside a second ridge that is
    'measured', reads the links from L on with the roundings held (HELD) and every earlier one as it was declared and solved. Without it,
    the declared second ridge at every link (none where the declaration names none)."""
    damping, scope = named.get('second_ridge', 0.0), named.get('second_ridge_holds_the_roundings')
    if scope is None:
        return damping
    if damping != 'measured' or not isinstance(scope, dict) or type(scope.get('from_link')) is not int:
        raise ValueError('holding the roundings is a way to read a measured second ridge and names the first link it holds from (from_link)')
    return HELD if link is not None and link >= scope['from_link'] else damping


def settings_of(power, stages, before, counts, shares=None):
    """The settings one ridge adds to a cross-validation's table, from the three folds' stages there: one for every number of rounds taken, up to
    the most any fold took. A fold whose stage ended by itself before that many rounds stands at its stage's end, which is where that many rounds
    leave it; before its first round it stands where it entered the stage (`before`: each fold's read score and squared change there, with
    `shares` its reading's two groups, and its held members' own requests where it has any). Each setting's score is `held_score` of the
    folds' readings. `read_members`, where the folds' readings hold requests: what each held member keeps of its own, fold by fold, in the
    order of the fold's answers (None for an answer asked nothing): read only, it enters no score."""
    taken = [[each for each in stage['each_round'] if each['taken']] for stage in stages]
    rows = []
    for number in range(1, max(len(each) for each in taken) + 1):
        last = [rounds[min(number, len(rounds)) - 1] if rounds else None for rounds in taken]
        at = [(each['beside'][0], each['change_from_the_start_squared'], each.get('beside_groups', [None])[0], each.get('beside_members', [None])[0]) if each is not None
              else tuple(entered) + (None,) * (4 - len(entered)) for each, entered in zip(last, before)]
        rows.append(dict(ridge_power=power, round=number, read=[each[0] for each in at], score=held_score([each[0] for each in at], counts, [each[2] for each in at], shares),
                         change_squared=float(np.mean([each[1] for each in at])), rounds_each_fold_took=[len(each) for each in taken],
                         **(dict(read_groups=[each[2] for each in at]) if shares is not None else {}),
                         **(dict(read_members=[each[3] for each in at]) if any(each[3] is not None for each in at) else {})))
    return rows


def chosen(trace, frames, w, g, stands, folds, places, effort=EFFORT, say=None, charges=(), damping=0.0, replaced=False, asks=None, members_only=False, patience=None, above=False):
    """The candidate at one matrix (Astra's specification of November 12, section 4, and his ruling of the same night on what a candidate is).

    The fitting answers' three folds cross-validate this one objective. Three paths go down the ridges in step, each solved on two folds and read
    on the third. EVERY written state a path accepts is a candidate setting, named by its ridge and by how many rounds had been taken at that
    ridge; a setting's score is the mean of the three folds' readings there, each fold by its number of answers (settings_of). The setting with
    the least score is taken, the start among them (least). All the fitting answers are then solved on once, down the same ridges to that one
    and no further than that many rounds at it, and where that solve stands is the candidate.

    One unit of the ridge and one first ridge serve the three folds and the last solve: the objective's own curvature, and the strongest ridge a
    stage can be told from none at, on all the fitting answers at the matrix the path starts from.

    Where the paths end is a budget of computation, tested here on this objective, and no statement about weaker ridges. From the first ridge
    down to the unit every ridge is made: there the ridge is at least the objective's own mean curvature, a stage's step is held by it and costs
    seconds, and the first stages move the readings by about what a solve resolves, so a reading that does not fall there says nothing. Below
    the unit the paths end at the first ridge at which states were written and no fold reads less than it has read before, at any of its
    rounds, nor does the folds' mean; or at the floor (a halved ridge less of the unit than a linear solve resolves); or where every fold's own
    walk has ended. A fold whose linear solve gave no finite step ends the scoring there: no setting below is scored on fewer folds. The record
    says how the paths ended and whether the setting taken lies at the last ridge made.

    charges  requests (REQUESTS) in the same sum as the cells, on the answers solved on and on the answers read alike. The whole sum is what
             every reading here is: the start's, a step's acceptance, a stage's end, a setting's score on answers it was not solved on, and
             the last solve's. The first ridge is found from the whole sum and its slope at the start. None by default, which is what
             correction-at-19 and correction-at-17-original had.
    replaced the correct group's own cells are left out of the sum (without_correct): each correct member's own error, named in `charges`,
             stands in their place and is not counted again through the group's mean. What a request weighs, the unit of the ridge and
             what speeds the linear solves stay the reference's own cells', unmodified: the unit is then a scale the search is counted
             in, the same as the cells-only objective's on these rows, and no claim about this sum's own curvature, which the members'
             terms add to (Astra's specification of November 12).
    damping  every solve's second ridge on a round's step (solve). None by default.
    asks     row-level requests by answer (`dressed`), in the same sum, on the answers solved on and on the answers read alike: a fold is
             read on its own answers' requests, which no solve it is scored on has seen. None by default.
    members_only  both groups' own cells are left out of the sum (without_groups): its terms are the row-level requests and each correct
             member's own error, and nothing else. It needs both named. The unit, the weights and the linear solves' speed stay the
             reference's own cells', as with `replaced`.
    patience an allowance of ridges, beside the ends above and never in place of one: below the unit the paths also end once this many
             ridges in a row have scored no setting under the least score before them (`stalled_after`). None by default, which is how the
             three earlier candidates were made. It is a budget of exploration: a path ended by it can miss a later ridge that would
             have scored less, and the record says which end ended the paths (Astra, November 12, 2026:
             records/retrospectives/2026-11-12-ridge-end-astra.md).
    above    the allowance also counts above the unit, once a setting has scored under the unchanged start: from then on every ridge
             without a new least counts, at any power, a new least starts the count again, and the paths end where the count reaches
             `patience`. Before the first setting under the start only the ridges below the unit count, as without it, so the flat first
             ridges cannot use the allowance up. Every other end stands as it is, in its place. False by default. A path it ends above the
             unit never reached the unit: a budget, and no statement that a weaker ridge scores no less (Astra, November 13, 2026, 2:20 AM
             CT, same record).
    The folds need not hold the two groups in the cohort's proportion (case families kept together): a setting's score is then
    `held_score`'s sum by group, the same in the selection and in the allowance.
    Returns the candidate (the start itself where nothing is taken) and the record."""
    everyone = np.logical_or.reduce([np.asarray(each) for each in folds])
    if replaced and not charges:
        raise ValueError('the correct group\'s cells are left out only where its members\' own error is named in their place (charges)')
    if members_only and not (replaced and asks):
        raise ValueError('both groups\' cells are left out only where the correct members\' own error and row-level requests stand in their place')
    if above and patience is None:
        raise ValueError('the allowance above the unit widens an allowance of ridges: none is given (patience)')
    summed = [without_groups(cells) for cells in frames] if members_only else [without_correct(cells) for cells in frames] if replaced else frames   # the cells the sum holds; `frames` stay the reference's own

    def part_of(pick):   # these answers, with the requests named beside the cells: weighed by the reference's own cells, whatever the sum holds
        return dressed(trace.part(pick, places), frames, charges, asks, pick)

    whole = part_of(everyone)
    unit = unit_of(whole, frames, stands, g)   # the reference's own cells on these rows
    most = strongest(whole, summed, g, stands)   # of the whole sum, from its reading and its slope at the start
    first = None if most is None else math.floor(math.log2(most / unit))
    if first is None:
        nothing = 'nothing of the fitting answers\' objective is left at the start' if charges or asks else 'every cell of the fitting answers stands inside its radius at the start'
        return stands, dict(unit=unit, selected=f'no change: {nothing}', status='unchanged')
    told = lambda name: None if say is None else (lambda line: say(dict(solve=name, **line)))
    counts = [int(np.asarray(each).sum()) for each in folds]
    wrong = np.asarray(trace.wrong)
    held = [[int((np.asarray(each) & wrong).sum()), int((np.asarray(each) & ~wrong).sum())] for each in folds]   # each fold's wrong and correct answers
    totals = [sum(each[k] for each in held) for k in range(2)]
    proportional = all(each[0] * totals[1] == each[1] * totals[0] for each in held)
    shares = None if proportional else [[each[k] / totals[k] if totals[k] else 0.0 for k in range(2)] for each in held]
    geometry = frames if replaced else None
    # One path's rows on the card at a time: each fold's walk rests once it is made and between its stages, and the rows of all the fitting
    # answers rest until the last solve. Three walks' rows and the whole's beside a stage's own state did not fit a 16 GB card at an input
    # 12288 wide (November 13, 2026); the speed review of November 10 had passed the same wall one path at a time.
    placed(whole, 'cpu')
    walks = []
    for k, each in enumerate(folds):
        walks.append(Descent(part_of(everyone & ~np.asarray(each)), summed, w, g, stands, unit, effort, beside=[part_of(each)], say=told(f'read fold {k}'), damping=damping, geometry=geometry))
        walks[-1].rests()
    # where each fold stands: its read score, its squared change from the start, its two groups, and its held members' own requests where it has any
    stood = [(walk.record['before']['beside'][0], 0.0, walk.record['before']['beside_groups'][0], walk.record['before'].get('beside_members', [None])[0]) for walk in walks]
    mean = lambda at: held_score([each[0] for each in at], counts, [each[2] for each in at], shares)
    table = [dict(ridge_power=None, round=0, read=[each[0] for each in stood], score=mean(stood), change_squared=0.0, **(dict(read_groups=[each[2] for each in stood]) if shares is not None else {}),
                  **(dict(read_members=[each[3] for each in stood]) if any(each[3] is not None for each in stood) else {}))]
    lowest, lowest_mean, power, ended, stalled, explored = [each[0] for each in stood], table[0]['score'], first, None, 0, []
    while ended is None:
        stages = []
        for walk in walks:
            walk.wakes()
            stages.append(walk.stage(power))
            walk.rests()
        rows = settings_of(power, stages, stood, counts, shares)
        table.extend(rows)
        stood = [(stage['beside'][0], stage['change_from_the_start_squared'], stage['beside_groups'][0], stage.get('beside_members', [None])[0]) for stage in stages]
        fell = [min([row['read'][k] for row in rows], default=lowest[k]) < lowest[k] for k in range(len(walks))]
        fell_mean = min([row['score'] for row in rows], default=lowest_mean) < lowest_mean
        armed = above and lowest_mean < table[0]['score']   # a setting made before this ridge scored under the unchanged start
        stalled = stalled_after(stalled, power, fell_mean, armed)
        explored.append(dict(ridge_power=power, settings=len(rows), least_score_here=min([row['score'] for row in rows], default=None), least_score_before=lowest_mean, a_new_least=fell_mean,
                             **(dict(a_setting_before_it_scored_under_the_start=armed, counted=power < 0 or armed, ridges_in_a_row_without_a_new_least=stalled) if above else
                                dict(ridges_in_a_row_without_one_below_the_unit=stalled))))
        lowest = [min([lowest[k]] + [row['read'][k] for row in rows]) for k in range(len(walks))]
        lowest_mean = min([lowest_mean] + [row['score'] for row in rows])
        if say:
            say(dict(ridge_power=power, read=[each[0] for each in stood], score=mean(stood), a_fold_read_less_than_before=fell, the_mean_read_less_than_before=fell_mean,
                     **(dict(ridges_in_a_row_without_a_new_least=stalled) if patience is not None else {})))
        if any(walk.ended == 'the linear solve gave no finite step' for walk in walks):
            ended = 'a fold\'s linear solve gave no finite step: no weaker ridge is scored'
        elif all(walk.ended for walk in walks):
            ended = 'nothing of any fold\'s objective is left' if charges or asks else 'every fold\'s cells stand inside their radii'
        elif power < 0 and rows and not any(fell) and not fell_mean:
            ended = 'below the unit, no fold and no mean of the folds read less at this ridge than before it'
        elif patience is not None and (power < 0 or above) and stalled >= patience:
            ended = (f'{patience} ridges in a row scored no setting under the least score before them, counted from the first setting under the unchanged start, above the unit too: '
                     'the allowance of ridges is used' if above else
                     f'below the unit, {patience} ridges in a row scored no setting under the least score before them: the allowance of ridges is used')
        elif FLOOR(power - PATH):
            ended = 'the ridge is less of the unit than a linear solve resolves'
        else:
            power -= PATH
    best = least(table)
    record = dict(unit=unit, first_ridge_power=first, strongest_ridge=most,
                  **(dict(asked=dict(requests=len(asks), answers=sorted({int(each['answer']) for each in asks}), sources=sorted({each['source'] for each in asks}),
                                     the_wrong_groups_cells='left out: the row-level requests are the wrong answers\' only terms' if members_only else 'kept, with the requests beside them')) if asks else {}),
                  **(dict(charged=list(charges), the_correct_groups_cells='left out: each correct member\'s own error stands in their place' if replaced else 'kept, with the requests beside them',
                          unit_from='the reference\'s own cells, unmodified, on all the fitting rows at the matrix the path starts from',
                          strongest_ridge_from='the whole sum on all the fitting rows: its reading and its slope at the start') if charges else {}),
                  **(dict(second_ridge=damping) if damping else {}), folds=[dict(answers_read=count, wrong=each[0], correct=each[1]) for count, each in zip(counts, held)],
                  held_score=dict(how='the folds\' mean, each by its number of answers: every fold holds the two groups in the cohort\'s proportion' if shares is None else
                                  'the sum over groups and folds of n[g, k] / N[g] x the fold\'s reading of that group (held_score): the folds do not hold the two groups in the cohort\'s proportion',
                                  **(dict(shares=shares, answers_of_each_group=dict(wrong=totals[0], correct=totals[1])) if shares is not None else {})),
                  **(dict(allowance_of_ridges=patience) if patience is not None else {}),
                  **(dict(allowance_above_the_unit=dict(counts='every ridge after the first accepted setting that scored under the unchanged start, at any power; before one, the ridges below the unit only',
                                                        ridges_allowed=patience, an_end_by_it='a budget of exploration: no statement that a weaker ridge scores no less',
                                                        decided_in='records/retrospectives/2026-11-12-ridge-end-astra.md (Astra, November 13, 2026, 2:20 AM CT)')) if above else {}),
                  explored=explored, cross_validation=table,
                  selected_ridge_power=best['ridge_power'], selected_round=best['round'], selected_score=best['score'], score_at_the_start=table[0]['score'],
                  the_paths_ended=ended, last_ridge_power_made=power, the_setting_taken_is_at_the_last_ridge_made=best['ridge_power'] == power,
                  fold_paths=[walk.record for walk in walks])
    del walks
    torch.cuda.empty_cache()
    if say:
        say(dict(selected_ridge_power=best['ridge_power'], selected_round=best['round'], selected_score=best['score'], score_at_the_start=table[0]['score'], the_paths_ended=ended))
    if best['ridge_power'] is None:
        return stands, dict(record, selected='no change: no written state of the paths reads less on the folds it was not solved on than the start does', status='unchanged')
    walk, at = Descent(placed(whole, DEVICE), summed, w, g, stands, unit, effort, say=told('all the fitting answers'), damping=damping, geometry=geometry), first
    while walk.ended is None and at >= best['ridge_power']:
        walk.stage(at, rounds=best['round'] if at == best['ridge_power'] else None)
        at -= PATH
    walk.record['ended'] = walk.ended or 'the setting the folds chose'
    kept = walk.weights
    return kept, dict(record, selected='the solve on all the fitting answers, down the same ridges to the one the folds chose and no further than the rounds they chose at it',
                      status='changed' if bool((kept != stands).any()) else 'unchanged', refit=walk.record, objective_before=walk.record['before']['objective'],
                      objective=walk.record['stages'][-1]['objective'], rounds_taken_at_the_selected_ridge=walk.record['stages'][-1]['rounds_taken'],
                      change_over_matrix=float((kept - w).norm() / w.norm()), change_from_the_start_over_matrix=float((kept - stands).norm() / w.norm()))


def resolved(z, sizes, again, again_sizes, groups):
    """What the original answers fix at one merge, from each member's mean at each place of its steps' unit states (z: members x places x width) and
    of the logs of their sizes (sizes: members x places), as recorded while the model generated, and the same two read in the pass of the same text
    through the original file (again, again_sizes). groups: which members are wrong, which correct, in that order. Returns, per place,
      pattern, pattern_size   each group's mean over its members (the wrong group first); the correct group's is the target of both groups
      gap, gap_size           the correct group's pattern less the wrong group's
      nu, nu_size             V_wrong / N_wrong + V_correct / N_correct, V a group's unbiased variance over its members (for the direction, the
                              trace of the covariance): how far the contrast of balanced halves of the two groups lies from the full groups'
                              contrast, squared, on average over all such halves
      q, q_size               the square of how far the contrast read in the pass lies from the recorded one: each group's mean change between
                              the two is taken first, as a vector, the two are subtracted, and only then squared
      radius, radius_size     the root of the larger of the two: what the original observations resolve of the pattern there. No floor of ours is
                              under it: where both are nothing, the radius is nothing."""
    counts = [int(mask.sum()) for mask in groups]
    pattern, pattern_size = (torch.stack([values[mask].mean(0) for mask in groups]) for values in (z, sizes))
    nu = sum(z[mask].var(0, unbiased=True).sum(-1) / count for mask, count in zip(groups, counts))
    nu_size = sum(sizes[mask].var(0, unbiased=True) / count for mask, count in zip(groups, counts))
    off, off_size = again - z, again_sizes - sizes
    q = ((off[groups[1]].mean(0) - off[groups[0]].mean(0)) ** 2).sum(-1)
    q_size = (off_size[groups[1]].mean(0) - off_size[groups[0]].mean(0)) ** 2
    return dict(pattern=pattern, pattern_size=pattern_size, gap=pattern[1] - pattern[0], gap_size=pattern_size[1] - pattern_size[0], nu=nu, nu_size=nu_size, q=q, q_size=q_size,
                radius=torch.maximum(nu, q).sqrt(), radius_size=torch.maximum(nu_size, q_size).sqrt())


def passed_on(to, log, weight, scale, eps):
    """What the model's output norm passes on from its last merge, row by row: the direction and the log of the size of
    weight * x / sqrt(mean(x^2) + eps), where x is the last merge's state times the last layer's output scale, from that state's unit direction
    (to: rows x width) and the log of its size (log: rows x 1). That is the engine's own operation after the last layer (build_norm with
    LLM_NORM_RMS: ggml_rms_norm, then the product with output_norm.weight), and the state the head reads.

    With z the direction and s the log of the size, the passed-on state is weight * z * sqrt(n) / sqrt(1 + n eps / (scale^2 e^(2s))): its direction
    is that of weight * z, whatever the size, and the log of its size is log|weight * z| + log(n)/2 - log1p(n eps / (scale^2 e^(2s)))/2. The raw
    size enters through eps alone, and that term is kept, not assumed away."""
    turned = to.double() * weight.to(to.device)
    size = turned.norm(dim=-1, keepdim=True)
    width = to.shape[-1]
    passed_log = size.log() + 0.5 * math.log(width) - 0.5 * torch.log1p(width * eps * torch.exp(-2 * log.double()) / scale ** 2)
    return (turned / size).to(to.dtype), passed_log.to(log.dtype)


def terminal(tensors, last):
    """What `passed_on` needs of the original file: output_norm's learned scale, the last layer's output scale (one positive number) and the
    norm's epsilon, with where each is from. No matrix a sweep writes is among them, so they are the same for every model made from the file."""
    weight = tube.checkpoint_values(tube.ORIGINAL_MODEL, tensors['output_norm.weight']).astype('f8').ravel()
    scale = tube.checkpoint_values(tube.ORIGINAL_MODEL, tensors[f'blk.{last}.layer_output_scale.weight']).astype('f8').ravel()
    if scale.size != 1 or not scale[0] > 0:
        raise ValueError(f'the output of layer {last} is not its last merge times one positive number')
    return dict(weight=torch.from_numpy(weight), scale=float(scale[0]), eps=tube.NORM_EPSILON)


class Ending(tube.Trace):
    """The rows at the last layer's feed-forward matrix. Behind its own merge lies the model's last merge, which the correction reference reads
    as what the output norm passes on from it (`reference`, `passed_on`). Every part of this trace says so to the stand that reads it
    (`standing`), with what `passed_on` needs of the original file (`terminal`, set as `passes_on` when the trace is laid)."""

    def part(self, pick, places):
        part = super().part(pick, places)
        part.passes_on = self.passes_on
        return part


def standing(part, weights, g, rounded=True):
    """A part's rows at a matrix, as every reading and every step here takes them (tube.Stand). At the last layer's feed-forward matrix the
    merge behind its own is the model's last, and it is read as what the output norm passes on (`Passed`). `rounded` is tube.Stand's: True,
    as the engine rounds, or the roundings another stand of these rows kept (its `roundings`), held here as they stood there (solve, HELD):
    no reading of the engine's own away from that stand."""
    stand = tube.Stand(part, weights, g, rounded)
    return stand if getattr(part, 'passes_on', None) is None else Passed(stand, part.passes_on)


class Passed:
    """A stand at the last layer's feed-forward matrix (tube.Stand) with its second merge, the model's last, read as what the output norm
    passes on from it: the form the reference's cells of that merge are in (`reference`, `passed_on`). The feed-forward's own merge is the
    stand's own, untouched. The raw last merge enters nothing here, so the model's last state is counted once, in the passed-on form.

    With z the last merge's unit state, s the log of its size, w output_norm's learned scale, tau = |w z| and c = n eps / scale^2, the
    passed-on state has the direction u = w z / tau and the log of the size L = log tau + log(n)/2 - log1p(c e^(-2s))/2 (`passed_on`). A
    movement dz of the unit state and a change ds of the log of the size move them, to first order, by
        du = (w dz - u (u . w dz)) / tau          dL = (u . w dz) / tau + k ds,     k = c e^(-2s) / (1 + c e^(-2s)).
    That is the final normalization's whole dependence on the changed state: the direction goes through w and is made a unit again, and
    the raw size reaches the passed-on size through eps alone, a term that is kept (Astra, November 12, 2026,
    records/retrospectives/2026-11-12-late-chain-decision-astra.md: a direct score with the denominator held fixed is not the objective).
    Where the rows stand is counted from what the norm passes on of the original file's own last merge at the same row.

    On a text of the model's own (the part's `stood`) no row of the original lies beside an answer's step, and nothing is passed on of
    nothing: an answer's steps stand here as the norm passes them on, counted from nothing, and each answer's means at its places are
    counted from where the norm passes on the original's own rows of that answer (`stood_of` with `passes_on`, as the reference takes
    every row of this merge through the norm before a member's means). On the recordings' own text that is the same count as row by
    row. The question's positions are the original's input token for token on whatever text the answer goes on: they keep the
    original's rows (loop.questions_from_the_original), and a part whose positions were given none is refused.

    norm  `terminal`'s: output_norm's learned scale, the last layer's output scale and the norm's epsilon."""

    def __init__(self, stand, norm):
        self.stand, self.part, self.now, self.asking, self.norm = stand, stand.part, stand.now, stand.asking, norm
        later, _, later_q, _ = stand.later
        there = stand.part.behind
        self.weight = norm['weight'].to(later.to.device, later.to.dtype)
        self.at = [self.read_through(each) for each in (later, later_q)]
        if getattr(stand.part, 'stood', None) is None:
            origin = [passed_on(to, log, **norm) for to, log in ((there.origin, there.log_origin), (there.origin_q, there.log_origin_q))]
        else:   # a text of the model's own: the answer's steps have no row of the original, the question's positions keep theirs
            if there.origin_q.numel() and not bool(there.origin_q.any()):
                raise ValueError('behind the last feed-forward, on a text of the model\'s own, the question\'s positions are counted from what the norm passes on of the '
                                 'original\'s own rows there, and this part was given none (loop.questions_from_the_original lays them)')
            origin = [(torch.zeros_like(self.at[0]['to']), torch.zeros_like(self.at[0]['log'])), passed_on(there.origin_q, there.log_origin_q, **norm)]
        self.moved = [stand.moved[0], (self.at[0]['to'] - origin[0][0], self.at[0]['log'] - origin[0][1], self.at[1]['to'] - origin[1][0], self.at[1]['log'] - origin[1][1])]

    def read_through(self, merge):
        """What the norm passes on of a merge's rows, and what its first-order change needs of them: tau and k, row by row."""
        log = merge.size.log()
        to, passed_log = passed_on(merge.to, log, **self.norm)
        tau = (merge.to.double() * self.norm['weight'].to(merge.to.device)).norm(dim=1, keepdim=True)
        thin = merge.to.shape[1] * self.norm['eps'] * torch.exp(-2 * log.double()) / self.norm['scale'] ** 2
        return dict(to=to, log=passed_log, tau=tau.to(to.dtype), eased=(thin / (1 + thin)).to(to.dtype))

    def roundings(self):
        """The stand's own (tube.Stand.roundings): the norm rounds nothing, and what is held on the way to the last merge is held before it."""
        return self.stand.roundings()

    def carried(self, at, turned, grown):
        """A movement of the last merge's unit state and a change of the log of its size, as the movement of the passed-on unit state and the
        change of the log of its size, to first order."""
        sent = turned * self.weight
        along = (at['to'] * sent).sum(1, keepdim=True)
        return (sent - at['to'] * along) / at['tau'], along / at['tau'] + at['eased'] * grown

    def moves(self, dy, dy_q):
        """What output changes do to first order at each merge read (tube.Stand.moves), the second as the norm passes it on."""
        own, (turned, grown, turned_q, grown_q) = self.stand.moves(dy, dy_q)
        return [own, (*self.carried(self.at[0], turned, grown), *self.carried(self.at[1], turned_q, grown_q))]

    def reaches(self):
        """Stand-ins for `basis_of`, as tube.Stand.reaches gives them, the second for the passed-on state. An output change reaches its
        direction through w: by the root of w's mean square over tau, of what it reaches of the raw unit state. It reaches the log of its
        size along (w u) / tau - (1 - k) z, which is dL over a change of the raw state, times the raw state's size."""
        own, raw = self.stand.reaches()
        spread = float((self.weight.double() ** 2).mean().sqrt())
        return [own, tuple(types.SimpleNamespace(g=each.g, sent=each.sent, rms=each.rms, width=each.width, size=each.size, ratio=each.ratio * spread / at['tau'],
                                                 to=self.weight * at['to'] / at['tau'] - (1 - at['eased']) * merge.to)
                           for each, at, merge in zip(raw, self.at, (self.stand.later[0], self.stand.later[2])))]


def last_merge_held(on, rows):
    """For the last layer's feed-forward matrix, before its solve: this file's own expression of the model's last merge at the matrix the
    solve starts from, made from the rows kept (the matrix's input and the state its merge adds to, through the feed-forward's merge and
    the per-layer embedding's: tube.Behind), against the engine's own output there in the same pass (the rows' third array), as the norm
    passes both on and raw. A description kept with the candidate, and no gate: the two lie apart where the engine rounds a matrix's
    input the other way at the rows it reads several tokens at once (tube.Behind says by how much on the original)."""
    trace = on['trace']
    part = trace.part(np.ones(len(on['used']), bool), on['places'])
    told = lambda apart: dict(median=float(apart.median()), largest=float(apart.max()))
    out = {}
    with torch.no_grad():
        stand = standing(part, on['stands'], on['g'])
        engine = torch.from_numpy(rows['rows2']).to(DEVICE)
        for name, picked, later, at in (('answer_steps', part.rows, stand.stand.later[0], stand.at[0]), ('prompt_positions', part.asking, stand.stand.later[2], stand.at[1])):
            to, log = tube.unit_and_log(engine[picked])
            passed_to, passed_log = passed_on(to, log, **trace.passes_on)
            out[name] = dict(rows=int(picked.numel()),
                             as_the_norm_passes_it_on=dict(direction_apart=told((at['to'] - passed_to).norm(dim=1)), log_size_apart=told((at['log'] - passed_log).abs()[:, 0])),
                             raw=dict(direction_apart=told((later.to - to).norm(dim=1)), log_size_apart=told((later.size.log() - log).abs()[:, 0])))
    return out


def reference(run, folder=None):
    """What the original model fixes for the correction at every one of its merges, written once and only read after: <kind>-<layer>.npz at each
    merge and index.json, in the run's reference folder (Astra's specification of November 12, sections 1 and 2; `resolved` says what each
    array is).

    Everything is the original file's own: its rows as recorded while it generated (`tube.recorded`, the form the state overlap's patterns are in;
    the size as the engine itself recorded it there), and, for what the readings resolve, the engine's own output at the same merge in the pass
    of the same text through the original file (the baseline of the run the declaration names under original_observations_from). Every count is
    the source's number of answers; no fitting, no-fit or reserved role enters. index.json holds the scale of each form, S = the sum over every
    merge and place of |gap|^2 + nu, which every cell's weight is counted over (`cells_of`), the counts, what was read and its hashes, and, merge
    by merge, what each form adds to S and how much of the objective the original model itself has there (its wrong group's cells; its correct
    group's are nothing), before S divides it.

    The model's last merge (the last layer's, kind 'layer') is read as what the output norm passes on from it (`passed_on`): the next operation
    divides the raw size out, so the raw size there is no error of the computation, and the direction the head reads is the one output_norm's
    learned scale leaves (Astra, November 12, before any candidate). Every row of that merge, recorded and read in the pass alike, goes through
    `passed_on` before a member's means are taken. The same merge read raw is kept beside it as a description (raw-layer-<last>.npz and the
    merge's `raw` entry in the index) and enters no scale and no cell. A reading of any model against this reference passes that merge's rows
    through `passed_on` the same way (index.json, `terminal`)."""
    folder = run/'reference' if folder is None else folder
    declaration = check_native_run(folder, run/'declaration.json')
    made_with = dict(solver='code/signal-trace/pipeline/weight_edit/correction.py', sha256=AS_READ['sha256'])   # the file as this process read it
    source, members, index = ladder.cohort(run)
    tensors = json.loads(ladder.TENSORS.read_text())
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    pairs = [np.load(ladder.CORRESPONDENCE/'pairs'/f"{m['id']}.npy").astype(np.int64) for m in members]
    overlap = np.load(source/'recurrence.npz')
    correct = np.array([m['correct'] for m in members])
    steps = sum(m['steps'] for m in members)
    layout = tube.Layout(~correct, [m['steps'] for m in members], [m['question'] for m in members], pairs, places)
    groups = [torch.from_numpy(mask).to(DEVICE) for mask in (~correct, correct)]   # the wrong group first
    counts = [int(mask.sum()) for mask in groups]
    own = tube.reading_of(run)
    last, norm = index['layers'][-1], None
    folder.mkdir(parents=True)
    scale, report = dict.fromkeys(FORMS, 0.0), []
    for layer in index['layers']:
        for kind in tube.MERGES:
            w, g = ladder.weights_of(tensors, layer, kind)[1:] if kind in tube.KINDS else (None, None)
            to, log, _, _ = tube.recorded(members, index, layer, kind, w, g, tensors)
            if kind in tube.KINDS:   # the size the engine itself recorded there while it generated, as the frozen reference of a sweep takes it
                log = tube.native_log(members, layer, kind)
            given, given_log = (torch.from_numpy(np.array(v[:steps])).to(DEVICE) for v in tube.kept(own, kind, layer))
            raw = None
            if (layer, kind) == (last, 'layer'):   # the model's last merge: what the output norm passes on from it; the raw reading is kept as a description
                norm = terminal(tensors, last)
                raw = resolved(layout.means(to), layout.means(log)[..., 0], layout.means(given), layout.means(given_log)[..., 0], groups)
                (to, log), (given, given_log) = passed_on(to, log, **norm), passed_on(given, given_log, **norm)
            found = resolved(layout.means(to), layout.means(log)[..., 0], layout.means(given), layout.means(given_log)[..., 0], groups)
            del to, log, given, given_log
            apart, nu, q = (found['gap'] ** 2).sum(-1), found['nu'], found['q']
            if kind in tube.KINDS:   # the gap and its resolution are the state overlap's own, or something here reads the rows another way
                position = 2 * layer + tube.KINDS[kind][-1]
                for name, mine in (('gap', apart), ('chance', nu)):
                    if not torch.allclose(mine, torch.from_numpy(overlap[name][:, position]).to(DEVICE), rtol=1e-3, atol=1e-7):
                        raise ValueError(f'{name} at {kind}-{layer} is not the state overlap\'s')
            added = dict(direction=float((apart + nu).sum()), size=float((found['gap_size'] ** 2 + found['nu_size']).sum()))
            for form in FORMS:
                scale[form] += added[form]
            np.savez(folder/f'{kind}-{layer}.npz', **{name: (value.float() if name in ('pattern', 'gap') else value).cpu().numpy() for name, value in found.items()})
            beyond = dict(direction=(apart.sqrt() - found['radius']).clamp(min=0), size=(found['gap_size'].abs() - found['radius_size']).clamp(min=0))
            report.append(dict(layer=layer, kind=kind, added_to_the_scale=added,
                               gap_squared=dict(direction=float(apart.sum()), size=float((found['gap_size'] ** 2).sum())), nu=dict(direction=float(nu.sum()), size=float(found['nu_size'].sum())),
                               q=dict(direction=float(q.sum()), size=float(found['q_size'].sum())),
                               cells_whose_radius_is_the_readings=dict(direction=int((q > nu).sum()), size=int((found['q_size'] > found['nu_size']).sum())),
                               cells_with_no_radius=dict(direction=int((found['radius'] == 0).sum()), size=int((found['radius_size'] == 0).sum())),
                               cells_the_original_has_outside={form: int((beyond[form] > 0).sum()) for form in FORMS},
                               the_originals_objective_before_the_scale={form: float((beyond[form] ** 2).sum()) for form in FORMS}))
            if raw is not None:
                np.savez(folder/f'raw-{kind}-{layer}.npz', **{name: (value.float() if name in ('pattern', 'gap') else value).cpu().numpy() for name, value in raw.items()})
                past = dict(direction=((raw['gap'] ** 2).sum(-1).sqrt() - raw['radius']).clamp(min=0), size=(raw['gap_size'].abs() - raw['radius_size']).clamp(min=0))
                report[-1].update(read_as='what the output norm passes on from this merge (passed_on)',
                                  raw=dict(what='the same merge read raw, as every other merge is; a description, in no scale and no cell',
                                           would_add_to_the_scale=dict(direction=float(((raw['gap'] ** 2).sum(-1) + raw['nu']).sum()), size=float((raw['gap_size'] ** 2 + raw['nu_size']).sum())),
                                           gap_squared=dict(direction=float((raw['gap'] ** 2).sum()), size=float((raw['gap_size'] ** 2).sum())),
                                           nu=dict(direction=float(raw['nu'].sum()), size=float(raw['nu_size'].sum())),
                                           cells_the_original_has_outside={form: int((past[form] > 0).sum()) for form in FORMS},
                                           the_originals_objective_before_the_scale={form: float((past[form] ** 2).sum()) for form in FORMS}))
                del raw, past
            print(json.dumps(report[-1]), flush=True)
            del found, apart, nu, q, beyond
            torch.cuda.empty_cache()
    if len(report) != len(tube.MERGES) * len(index['layers']):
        raise ValueError('The correction reference does not cover every merge of the model')
    if not all(scale[form] > 0 for form in FORMS):
        raise ValueError(f'A form\'s scale over the whole original trace is not positive ({scale}): the reference is degenerate and is not repaired with a number of ours')
    share = {name: count / sum(counts) for name, count in zip(('wrong', 'correct'), counts)}
    record = dict(
        scale=scale, answers=dict(wrong=counts[0], correct=counts[1]), group_share=share, places=places, merges=report,
        the_originals_objective={form: share['wrong'] * sum(each['the_originals_objective_before_the_scale'][form] for each in report) / scale[form] for form in FORMS},
        how=reference.__doc__, arrays=resolved.__doc__, specification=declaration.get('specification'),
        state_overlap=str((source/'recurrence.npz').relative_to(tube.ROOT)), state_overlap_sha256=tube.digest(source/'recurrence.npz'),
        the_pass_read_from=str(own.relative_to(tube.ROOT)), the_pass_read_through_sha256=json.loads((own/'index.json').read_text())['read_through_sha256'],
        the_pass_was_read=json.loads((own/'index.json').read_text()).get('passes_read'), recorded_rows_rebuilt_from='the matrix\'s input as the engine multiplied it',
        terminal=dict(merge=dict(layer=last, kind='layer'), read_as=passed_on.__doc__,
                      operator='l_out = pe_out * layer_output_scale; result_norm = l_out / sqrt(mean(l_out^2) + eps) * output_norm.weight',
                      source="this project's llama.cpp build: src/models/gemma4.cpp (the build_norm of model.output_norm after the last layer) and src/llama-graph.cpp (build_norm, LLM_NORM_RMS)",
                      output_norm='output_norm.weight', output_norm_sha256=hashlib.sha256(norm['weight'].numpy().tobytes()).hexdigest(),
                      layer_output_scale={f'blk.{last}.layer_output_scale.weight': norm['scale']}, eps=norm['eps'],
                      the_files_own_eps='gemma4.attention.layer_norm_rms_epsilon reads 9.999999974752427e-07 in the original file (float32 of 1e-6), read November 12, 2026',
                      held_against_the_engine='records/development/2026-11-12-correction-objective/terminal-check.out'),
        made_with=made_with)
    tube.save(folder/'index.json', record)
    if folder == run/'reference':   # the run's own reference: its record is also kept beside the folder, where the declaration names it
        tube.save(run/'reference-record.json', record)
    return record


def cells_of(folder, kind, layer):
    """A merge's Cells from a correction reference. Both groups answer to the correct pattern: counted as movement from where the original stood,
    the wrong group's mean is aimed at the gap and the correct group's at no movement. The radius is what the original observations resolve of
    that pattern, the same for both groups. A cell's weight is its group's share of the source's answers over its form's scale, one scale for
    the whole trace: no cell is weighed by its own radius."""
    values, index = np.load(folder/f'{kind}-{layer}.npz'), json.loads((folder/'index.json').read_text())
    share = torch.tensor([index['group_share']['wrong'], index['group_share']['correct']], dtype=torch.float64)[:, None]
    gap, gap_size = torch.from_numpy(values['gap']), torch.from_numpy(values['gap_size'])
    twice = lambda values: torch.from_numpy(values)[None].expand(2, -1)
    return Cells(torch.stack([gap, torch.zeros_like(gap)]), twice(values['radius']), (share / index['scale']['direction']).expand(2, gap.shape[0]),
                 torch.stack([gap_size, torch.zeros_like(gap_size)]), twice(values['radius_size']), (share / index['scale']['size']).expand(2, gap.shape[0]))


def linked(run, chain, link):
    """The state a link of a chain starts from (a declaration's `chain`: Astra's decision of November 12, 2026,
    records/retrospectives/2026-11-12-late-chain-decision-astra.md). A chain is one correction over several matrices, made early to late in
    one file. Its first link starts from the chain's first state, the original file. A later link starts from the chain's own file as the
    last written link before it left it, with the hash that link's record holds (links/<link>/written.json); a link whose candidate
    changed nothing wrote nothing, and the state it found stands. No link is started before every earlier one is finished.

    Where the declaration names them (chain.first_rows: path and sha256), the first link's rows are the ones another run kept in a pass
    of the same recorded text through the same first state at the same matrix: the link reads those and makes none of its own (`rows`
    refuses), and `laid` holds the file's own record of the state and the matrix it was read at against the link's."""
    links = [int(each) for each in chain['links']]
    if link not in links:
        raise ValueError(f'{link} is not a link of the chain ({links})')
    first = chain['first_state']
    state = dict(source_run=chain['source_run'], matrix=chain['matrix'].format(layer=link), source_rows=f"{(run/'links'/str(link)).relative_to(tube.ROOT)}/rows.npz",
                 model=first['model'], model_sha256=first['model_sha256'], links_before=[])
    kept = chain.get('first_rows')
    if kept is not None and link == links[0]:
        if tube.digest(tube.ROOT/kept['path']) != kept['sha256']:
            raise ValueError(f"{kept['path']} is not the rows the chain declares its first link reads ({kept['sha256']})")
        state['source_rows'] = kept['path']
    for earlier in links[:links.index(link)]:
        left = run/'links'/str(earlier)/'written.json'
        if not left.exists():
            raise ValueError(f'link {earlier} is not finished: a chain is made early to late')
        told = json.loads(left.read_text())
        state['links_before'].append(dict(link=earlier, written=bool(told['written'])))
        if told['written']:
            state.update(model=told['model'], model_sha256=told['model_sha256'])
    return state


def started_from(run, link=None):
    """What a run's edit starts from, by its declaration (planned_starting_state): the sweep whose state it is, that state's model file, with its
    hash checked, the one matrix to be changed, and the rows kept at that matrix from a pass through that state. The correction reference it is
    read against is the run's own, or the one another run made from the same original, where the declaration names it (correction_reference).
    The text the rows are read on is the recordings' own, or, where the declaration names one (planned_starting_state.text), the answers a model
    wrote by itself, as a capture keeps them (`read_on`). For a chain (the declaration's `chain`) the state is the named link's (`linked`),
    and what the link keeps lies in links/<link>/."""
    declaration = json.loads((run/'declaration.json').read_text())
    chain = declaration.get('chain')
    if (chain is None) != (link is None):
        raise ValueError('a chain\'s commands name the link they work at, and a run of one matrix names none')
    state = declaration['planned_starting_state'] if chain is None else linked(run, chain, link)
    model = tube.ROOT/state['model']
    if tube.digest(model) != state['model_sha256']:
        raise ValueError(f"{model.name} is not the state the run declares it starts from ({state['model_sha256']})")
    source = ROUND_TWO_RUNS/state['source_run']
    tensors = json.loads(ladder.TENSORS.read_text())
    layer, kind = next((layer, kind) for _, layer, kind in tube.matrices(json.loads((ladder.INPUTS/'index.json').read_text())['layers'])
                       if f'blk.{layer}.{tube.KINDS[kind][1]}.weight' == state['matrix'])
    binding = json.loads((source/'model'/'bindings.json').read_text())[state['matrix']]
    reference = tube.ROOT/declaration['correction_reference'] if 'correction_reference' in declaration else run/'reference'   # its own, or the one it names
    text = tube.ROOT/state['text'] if state.get('text') else None
    return dict(declaration=declaration, state=state, model=model, source=source, tensors=tensors, layer=layer, kind=kind, binding=binding, reference=reference, text=text,
                chain=chain, link=link, folder=run if chain is None else run/'links'/str(link))


def read_on(at, members):
    """The fitting answers as a run's rows read them, and where each one's steps lie on the round's axis.

    Without a text of its own the run reads the recordings: each answer's recorded tokens, on the frozen registration. With one
    (planned_starting_state.text: the `own` folder of a capture of a model's own answers: <id>/answer.json with the tokens the model wrote and
    <id>/pairs-axis.npy with that text's own registration on the same axis, which tube.registered writes) each answer
    is what the model wrote there by itself: the same question, other tokens and another number of steps. Every other field of an answer is
    the source's."""
    fitting, _ = tube.picks(members)
    used = [m for m, take in zip(members, fitting) if take]
    if at['text'] is None:
        return used, [np.load(ladder.CORRESPONDENCE/'pairs'/f"{m['id']}.npy").astype(np.int64) for m in used]
    own = []
    for m in used:
        answer = json.loads((at['text']/m['id']/'answer.json').read_text())
        if answer['id'] != m['id'] or len(answer['tokens']) != answer['steps']:
            raise ValueError(f"{m['id']}: the capture's answer is not this answer's, or its tokens are not its steps")
        own.append(dict(m, recorded=[int(token) for token in answer['tokens']], steps=int(answer['steps'])))
    pairs = [np.load(at['text']/m['id']/'pairs-axis.npy').astype(np.int64) for m in own]
    for m, each in zip(own, pairs):
        if each[:, 0].max() >= m['steps'] or each[:, 0].min() < 0:
            raise ValueError(f"{m['id']}: the capture's registration names steps the answer does not have")
    return own, pairs


def stood_of(source, members, kind, layer, places, passes_on=None):
    """Where each fitting answer stood at a merge in the original file's own reading of its recorded text (the source sweep's baseline), place by
    place on the frozen registration: its mean of the unit state (answers x places x width) and of the log of the size (answers x places). An
    answer a model wrote by itself has no row of the original to be counted from, row by row; its movement at a place is counted from here.
    On the recordings' own text this is the same count as row by row: an answer's mean at a place of its rows' movements is its mean there
    less the original's mean there.

    passes_on  for the model's last merge (`terminal`): every row goes through the output norm before an answer's means are taken
               (`passed_on`), as every row of that merge does in the reference and under a stand (Passed). The answer then stood where
               the norm passes its rows on, which is what a stand at the last feed-forward counts a text of the model's own from."""
    fitting, _ = tube.picks(members)
    picked = [i for i, take in enumerate(fitting) if take]
    begins = np.cumsum([0] + [m['steps'] for m in members])
    rows = np.concatenate([np.arange(begins[i], begins[i + 1]) for i in picked])
    to, log = (torch.from_numpy(np.ascontiguousarray(values[rows])).to(DEVICE) for values in tube.kept(tube.reading_of(source), kind, layer))
    if passes_on is not None:
        to, log = passed_on(to, log, **passes_on)
    pairs = [np.load(ladder.CORRESPONDENCE/'pairs'/f"{members[i]['id']}.npy").astype(np.int64) for i in picked]
    layout = tube.Layout([not members[i]['correct'] for i in picked], [members[i]['steps'] for i in picked], [members[i]['question'] for i in picked], pairs, places)
    return layout.means(to).float(), layout.means(log)[..., 0].float()


def rows(run, link=None):
    """The rows a candidate is solved on, for a run that keeps its own: the fitting answers' text once through the model the run starts from, at
    the one matrix it declares, read the way the source sweep's passes read, and kept where the declaration names them (source_rows, inside this
    run). A run that starts from rows another run kept reads none. The model is only read. For a chain's link: through the state that link
    starts from, at that link's matrix."""
    check_native_run(run/'passes', run/'declaration.json')
    at = started_from(run, link)
    out = tube.ROOT/at['state']['source_rows']
    if run not in out.parents:
        raise ValueError('The run starts from rows another run kept (source_rows); it reads none of its own')
    if out.exists():
        raise ValueError('The run already has its rows')
    source, layer, kind = at['source'], at['layer'], at['kind']
    tube.declared(source)   # how that sweep's passes read an answer's text
    _, members, index = ladder.cohort(source)
    used, _ = read_on(at, members)   # the recordings' text, or the model's own where the declaration names it
    labels = ladder.labels_of(layer, kind)
    if kind == 'feedforward' and layer == index['layers'][-1]:   # behind it lies the model's last merge: the engine's own output there is kept beside, for `last_merge_held`
        labels = (*labels, f"{tube.OUTPUT['layer']}-{layer}")
    found = tube.passed(at['model'], run, f"rows-{2 * layer + tube.KINDS[kind][-1]}-{kind}", labels, used)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as file:
        np.savez(file, **{f'rows{k}': np.ascontiguousarray(v) for k, v in enumerate(found)}, labels=np.array(labels), model=at['state']['model'], model_sha256=at['state']['model_sha256'])
    print(json.dumps(dict(run=run.name, tensor=at['state']['matrix'], layer=layer, kind=kind, labels=list(labels), rows=[list(v.shape) for v in found], answers=len(used),
                          read_through=at['state']['model'], read_through_sha256=at['state']['model_sha256'], text=at['state'].get('text', 'the recordings\' own'),
                          kept=at['state']['source_rows'])), flush=True)


def laid(at, rows):
    """What a solve at a run's one matrix stands on (`at`: started_from; `rows`: the rows kept there, the matrix's input and the state it adds to):
    the fitting answers' trace, the Cells of every merge it is read at (the matrix's own and, behind a feed-forward's, the layer's last), the
    matrix as the original holds it with its learned scale, the matrix the run starts from, and the answers with their folds. Behind the last
    layer's feed-forward the merge read is the model's last, as what the output norm passes on from it (Ending, Passed). The answers are
    read on the text the run declares (`read_on`). On the recordings' own text every row is counted from the original file's own output at the
    same row; on a model's own text no row of the original lies beside a row, and each answer is counted from where it stood in the original's
    reading, place by place (`stood_of`): behind the last layer's feed-forward, from where the norm passes that on."""
    source, layer, kind, tensors = at['source'], at['layer'], at['kind'], at['tensors']
    tube.declared(source)   # how that sweep's passes read an answer's text, which is how the kept rows were read
    _, members, index = ladder.cohort(source)
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    fitting, groups = tube.picks(members)
    used, pairs = read_on(at, members)
    wrong, steps, questions = np.array([not m['correct'] for m in used]), [m['steps'] for m in used], [m['question'] for m in used]
    ending = kind == 'feedforward' and layer == index['layers'][-1]   # behind it lies the model's last merge, read as what the output norm passes on (Passed)
    norm = terminal(tensors, layer) if ending else None
    if rows['rows0'].shape[0] != sum(steps) + sum(questions):
        raise ValueError(f"the rows kept ({rows['rows0'].shape[0]}) are not these answers' steps and question positions ({sum(steps) + sum(questions)}): they were read on another text")
    named = getattr(rows, 'files', None) or list(rows)   # rows kept by `rows` say the state they were read through and the outputs kept
    if 'model_sha256' in named and str(rows['model_sha256']) != at['state']['model_sha256']:
        raise ValueError('the rows kept were read through another state than the one this solve starts from')
    if 'labels' in named and [str(each) for each in rows['labels']][:2] != list(ladder.labels_of(layer, kind)):
        raise ValueError(f"the rows kept are another matrix's ({[str(each) for each in rows['labels']]}), not {list(ladder.labels_of(layer, kind))}")
    merges = [kind] + (['layer'] if kind == 'feedforward' else [])
    if at['text'] is None:
        origin_of, stood = tube.counted_from(source, tube.rows_of(members, fitting)), None
    else:
        count = sum(steps) + sum(questions)
        origin_of = lambda kind, layer: [torch.zeros((count, width), device=DEVICE) for width in (rows['rows1'].shape[1], 1)]
        stood = [stood_of(source, members, each, layer, places, norm if each == 'layer' else None) for each in merges]   # the model's last merge: where the norm passes each answer on
    _, w, g = ladder.weights_of(tensors, layer, kind)
    _, question_bounds = tube.bounds_of(source, kind, layer, fitting)   # the trace's layout carries them; no cell of the correction reads them
    origin, log_origin = origin_of(kind, layer)
    behind = tube.behind_of(source, layer, used, tensors, fitting, origin_of) if kind == 'feedforward' else None
    trace = (Ending if ending else tube.Trace)(torch.from_numpy(rows['rows0']).to(DEVICE), torch.from_numpy(rows['rows1']).to(DEVICE), origin, log_origin, wrong, steps, questions, pairs, question_bounds,
                                               behind=behind, stood=stood)
    if ending:
        trace.passes_on = norm
    return dict(trace=trace, frames=[cells_of(at['reference'], each, layer) for each in merges], merges=merges, w=w, g=g, stands=ladder.matrix_in(at['model'], at['binding']),
                used=used, fitting=fitting, groups=groups, places=places,
                counted_from='the original file\'s own output at the same row' if stood is None else 'where each answer stood in the original\'s reading, place by place')


def named_case(name):
    """An answer's case by its name: the name less the repeat it ends in (-h1, -h3) and, where the name has one, the channel before that."""
    return re.sub(r'(-channel)?-h\d$', '', name)


def case_folds(used, folds, case_of=None):
    """The folds with every case's answers in one: each answer goes to the fold of its case's first answer in the fitting answers' own
    order (Astra, November 12, 2026: records/retrospectives/2026-11-12-late-chain-decision-astra.md, section 3, point 4). tube.picks deals
    the folds inside each original group, which can part two answers to one case; a path would then be read on an answer whose case it
    was solved on. An answer's case is its name's (`named_case`), unless another reading of a name is given."""
    if case_of is None:
        case_of = named_case
    own = [next(k for k, each in enumerate(folds) if each[i]) for i in range(len(used))]
    first = {}
    for i, m in enumerate(used):
        first.setdefault(case_of(m['id']), own[i])
    return [np.array([first[case_of(m['id'])] == k for m in used]) for k in range(len(folds))]


def rows_asked(at, used):
    """None where a run's declaration names no row-level request (row_requests), which is what `dressed` takes for none. A declaration that names
    one is refused. Until November 15, 2026 this loaded a request made at the row where an answer writes a sign: the step-3 run's file or, for
    a chain's link, what this file's own command made from the link's rows. A request placed by a token is a request on the output and not on
    the computation, so its maker and its loader are out of this file. The runs that used them are closed, their records stay, and the
    repository's history holds the code. What a request feeds (`asked_of`, `requested`, `dressed`) takes any cell and is unchanged."""
    if at['declaration'].get('row_requests'):
        raise ValueError('the declaration names row-level requests placed at the row that writes a sign: their maker and their loader left this file on November 15, 2026, '
                         'and such a run is not solved from the tree')
    return None


def declared_as(run, sha256):
    """A run's declaration as it read when its file had this hash: the file as it stands, or as a commit of this repository holds it. With
    which it was: 'the file as it stands' or the commit."""
    path = run/'declaration.json'
    if tube.digest(path) == sha256:
        return json.loads(path.read_text()), 'the file as it stands'
    inside = str(path.relative_to(tube.ROOT))
    for commit in subprocess.run(['git', '-C', str(tube.ROOT), 'log', '--format=%H', '--', inside], capture_output=True, text=True, check=True).stdout.split():
        text = subprocess.run(['git', '-C', str(tube.ROOT), 'show', f'{commit}:{inside}'], capture_output=True, check=True).stdout
        if hashlib.sha256(text).hexdigest() == sha256:
            return json.loads(text), f'as commit {commit[:8]} holds it'
    raise ValueError(f'no commit of {inside} has the hash the solve was laid from ({sha256})')


def returned(back, at, run, link, used, folds, asks, stands):
    """A candidate's solve made on another machine and returned as files (records/development/2026-11-13-rented-solve/bundle.py, Performance's:
    its `lay` is the head of `candidate` on this machine, the verifier first, kept as one file of what `chosen` takes; its `solve` runs
    `chosen` on that file where a card holds it). The folder holds the matrix `chosen` returned (<tensor>.npy), the record it returned
    (solved.json), what it was solved with (solved_with.json) and what it said (solve.out).

    It is this command's own problem or it is refused: the run, the link and the matrix; the rows' file and the request's by their hashes;
    the fitting answers and the folds' sizes; the end of the paths and the second ridge the problem was laid with, which are this link's as
    the declaration scopes them (`path_end`, `second_ridge`); the declaration the problem was laid from, which is the file as it stands or a committed
    one that differs from it only in `revisions` (a dated note changes no field a solve reads); the solver's two files, the same where
    the problem was laid and where it was solved; the two returned files by the hashes kept beside them; and the number of weights
    that differ from the state this command starts from, counted again here. Returns the matrix, the record, and where and with what
    it was solved, for the candidate's record."""
    if not (back/'solved.json').exists():
        raise ValueError(f'{back.relative_to(tube.ROOT)} holds no solved.json: a solve of this problem is under way on another machine, or it stopped there (solve.out)')
    told = json.loads((back/'solved_with.json').read_text())
    was, matrix = told['laid'], at['state']['matrix']
    same = dict(run=run.name, link=link, tensor=matrix, layer=at['layer'], kind=at['kind'], rows=at['state']['source_rows'], rows_sha256=tube.digest(tube.ROOT/at['state']['source_rows']),
                asked_sha256=tube.digest(at['folder']/'asked.npz') if asks else None, fitting_answers=[m['id'] for m in used], fold_sizes=[int(each.sum()) for each in folds],
                rows_asked=len(asks) if asks else 0)
    apart = {key: (was.get(key), value) for key, value in same.items() if was.get(key) != value}
    if apart:
        raise ValueError(f'the solve returned is of another problem than this command\'s: {apart}')
    named = at['declaration'].get('solver')
    named = named if isinstance(named, dict) else {}
    ends = path_end(named, link)   # a file laid before a declaration could scope the allowance names none: below the unit only
    if was.get('paths_end', dict(ends, above=False)) != ends:
        raise ValueError(f"the solve returned was laid with another end of the paths than this command's: {was.get('paths_end')} against {ends}")
    reading, laid_reading = second_ridge(named, link), was.get('second_ridge', named.get('second_ridge', 0.0))   # a file laid before a declaration could scope the reading names none: the declared second ridge as it is
    if laid_reading != reading:
        raise ValueError(f"the solve returned was laid with another second ridge than this command's: {laid_reading!r} against {reading!r}")
    then, which = declared_as(run, was['declaration_sha256'])
    without = lambda declaration: {key: value for key, value in declaration.items() if key != 'revisions'}
    if without(then) != without(at['declaration']):
        raise ValueError('the declaration the solve was laid from differs from the one that stands in more than its dated revisions')
    laid_with, solved_with = ({key: each[key] for key in ('correction_sha256', 'tube_sha256')} for each in (was, told['solver_as_imported_here']))
    if laid_with != solved_with:
        raise ValueError(f'the problem was laid with one solver and solved with another: {laid_with} against {solved_with}')
    if tube.digest(back/f'{matrix}.npy') != told['matrix_sha256'] or tube.digest(back/'solved.json') != told['solved_sha256']:
        raise ValueError('the matrix or the record returned is not the one its solve kept the hash of')
    kept = torch.from_numpy(np.load(back/f'{matrix}.npy')).to(DEVICE)
    if kept.shape != stands.shape or int((kept != stands).sum()) != told['weights_changed']:
        raise ValueError('the matrix returned does not differ from the state this command starts from where its solve says it does')
    record = json.loads((back/'solved.json').read_text())
    return kept, record, dict(
        how='the head of this command was run on this machine (the verifier, the starting state, the rows, the request, the folds, the declaration\'s solver settings) and what `chosen` takes was kept as one file; '
            '`chosen` ran on that file on another machine; the readings below and this record were made here from the matrix and the record it returned',
        laid_and_solved_by='records/development/2026-11-13-rented-solve/bundle.py', card=told['card'], card_memory_gib=told.get('card_memory_gib'), torch=told['torch'], numpy=told['numpy'],
        solver=dict(file='code/signal-trace/pipeline/weight_edit/correction.py', sha256=was['correction_sha256'], tube_sha256=was['tube_sha256'], where='as it laid the problem here and as it was imported where it was solved'),
        laid_on=was.get('laid_on'), laid_at=was.get('laid_ct'), declaration_sha256=was['declaration_sha256'], declaration=which, bundle_sha256=told['bundle_sha256'],
        paths_end=was.get('paths_end', 'not named by the file that laid it: the allowance of ridges below the unit only'),
        started=told['started_ct'], ended=told['ended_ct'], seconds=told['seconds'], card_memory_at_its_most=dict(allocated=told['max_memory_allocated'], reserved=told['max_memory_reserved']),
        matrix_sha256=told['matrix_sha256'], solved_sha256=told['solved_sha256'], said=str((back/'solve.out').relative_to(tube.ROOT)))


def candidate(run, link=None):
    """The run's candidate: one matrix of the state the run starts from, corrected against the run's own reference by `chosen`, on the rows kept
    at that matrix. candidate/<tensor>.npy is the matrix (float32 values, each one the file's BF16 holds) and candidate/record.json everything
    the choice was made from. No model is read through the engine and none is written. For a chain's link the same, in links/<link>/: the
    matrix of that link, of the state the link starts from.

    The declaration's `solver`, where it names them: second_ridge (solve's `damping`) and, by link, second_ridge_holds_the_roundings
    (`second_ridge`), allowance_of_ridges and, by link, allowance_above_the_unit (chosen's `patience` and `above`: `path_end`) and folds
    ('case families': `case_folds`). A run that names none is solved as the three earlier candidates were.

    Where the command's folder holds `elsewhere/` (a solve of this very problem made on another machine: `returned`), nothing is solved
    here: the matrix and the record `chosen` returned there are taken once they are held against this command's own problem, and the
    readings and the record are made here. While that folder holds no solve yet the command refuses, so that a link is solved once."""
    check_native_run(run/'candidate', run/'declaration.json')
    at = started_from(run, link)
    folder = at['folder']/'candidate'
    if folder.exists():
        raise ValueError('The run already has a candidate' if link is None else 'The link already has a candidate')
    made_with = dict(solver='code/signal-trace/pipeline/weight_edit/correction.py', **AS_READ,   # the two files as this process read them
                     card=torch.cuda.get_device_name(DEVICE) if torch.cuda.is_available() else None, card_lock=os.environ.get('CORRECTION_CARD_LOCK') or 'card 0\'s shared lock (tube.CARD)')
    asked = at['declaration'].get('correct_members')   # each correct member's own error, in the correct group's place where the declaration says so
    requests, replaced = (tuple(asked['requests']), bool(asked['replaces_the_correct_groups_cells'])) if asked else ((), False)
    if any(REQUESTS[name]['members'] != 'correct' for name in requests):
        raise ValueError('a candidate\'s objective takes a correct member\'s own error only: the wrong members\' requests are a diagnostic\'s')
    kept_rows = np.load(tube.ROOT/at['state']['source_rows'])
    on = laid(at, kept_rows)
    trace, frames, w, g, stands, used, fitting, groups, places = (on[name] for name in ('trace', 'frames', 'w', 'g', 'stands', 'used', 'fitting', 'groups', 'places'))
    last_merge = None
    if isinstance(trace, Ending):   # the model's last merge lies behind this matrix: how this file expresses it is held against the engine's own output before the solve
        last_merge = dict(read_as='what the output norm passes on from it (passed_on), with the norm\'s dependence on the changed state in every step (Passed)',
                          terminal=dict(output_norm_sha256=hashlib.sha256(trace.passes_on['weight'].numpy().tobytes()).hexdigest(), layer_output_scale=trace.passes_on['scale'], eps=trace.passes_on['eps']),
                          **(dict(held_against_the_engines_own_output=last_merge_held(on, kept_rows)) if 'rows2' in kept_rows.files else {}))
        print(json.dumps(dict(the_models_last_merge=last_merge)), flush=True)
    del kept_rows
    asks = rows_asked(at, used)   # None; a declaration that names row-level requests is refused there
    members_only = bool(asks) and bool(at['declaration']['row_requests'].get('replaces_the_wrong_groups_cells'))
    named = at['declaration'].get('solver')
    named = named if isinstance(named, dict) else {}
    folds = [np.asarray(each[fitting]) for each in groups]
    if named.get('folds') == 'case families':
        folds = case_folds(used, folds)
    elif named.get('folds') is not None:
        raise ValueError(f"the declaration names folds this file does not know: {named['folds']!r}")
    started = time.time()
    back = at['folder']/'elsewhere'   # this command's own problem solved on another machine and returned as files (`returned`)
    if back.exists():
        kept, record, elsewhere = returned(back, at, run, link, used, folds, asks, stands)
        made_with['solved_elsewhere'] = elsewhere
        print(json.dumps(dict(solved_elsewhere=elsewhere)), flush=True)
    else:
        on_card = lambda: dict(gib_allocated_at_its_most=round(torch.cuda.max_memory_allocated(DEVICE) / 2 ** 30, 2), gib_reserved_at_its_most=round(torch.cuda.max_memory_reserved(DEVICE) / 2 ** 30, 2),
                               gib_of_the_card=round(torch.cuda.get_device_properties(DEVICE).total_memory / 2 ** 30, 2)) if torch.cuda.is_available() else None

        def say(told):   # every line of the solve as it is made; a ridge's line (the folds' readings there) with the card's memory so far
            line = {key: (round(value, 9) if isinstance(value, float) else value) for key, value in told.items()}
            if 'score' in told and on_card() is not None:
                line['card_gib_allocated_at_its_most'] = on_card()['gib_allocated_at_its_most']
            print(json.dumps(line), flush=True)

        kept, record = chosen(trace, frames, w, g, stands, folds, places, charges=requests, replaced=replaced, asks=asks, members_only=members_only,
                              damping=second_ridge(named, link), **path_end(named, link), say=say)
        made_with['card_memory'] = on_card()   # of this process, up to the solve's end: what a solve at this width takes of the card
    folder.mkdir(parents=True)
    np.save(folder/f"{at['state']['matrix']}.npy", kept.cpu().numpy())   # kept the moment it exists; the record follows
    tube.save(folder/'solved.json', record)   # the solve's own record as `chosen` returned it, kept until the whole record is: hours of solving are not lost to what follows
    everyone = np.ones(len(used), bool)
    whole = trace.part(everyone, places)
    alone = dict(before=read(whole, frames, stands, g), after=read(whole, frames, kept, g))   # the reference's own cells, both groups', with nothing beside them
    if requests or asks:   # the whole sum the candidate was made on
        whole = dressed(trace.part(everyone, places), frames, requests, asks, everyone)
        summed = [without_groups(cells) for cells in frames] if members_only else [without_correct(cells) for cells in frames] if replaced else frames
        complete = dict(before=read(whole, summed, stands, g), after=read(whole, summed, kept, g))
    record.update(run=run.name, tensor=at['state']['matrix'], layer=at['layer'], kind=at['kind'], **(dict(link=link, links_before=at['state']['links_before']) if link is not None else {}),
                  started_from=dict(model=at['state']['model'], model_sha256=at['state']['model_sha256'], rows=at['state']['source_rows'], text=at['state'].get('text', 'the recordings\' own'),
                                    counted_from=on['counted_from']),
                  reference=dict(folder=str(at['reference'].relative_to(tube.ROOT)), sha256=tube.digest(at['reference']/'index.json'), scale=json.loads((at['reference']/'index.json').read_text())['scale']),
                  fitting_answers=[m['id'] for m in used], folds=[dict(each, answers=[m['id'] for m, take in zip(used, fold) if take]) for each, fold in zip(record.get('folds', [{}] * 3), folds)],
                  **(dict(folds_dealt=named['folds']) if named.get('folds') else {}),
                  work=dict(tube.EFFORT, resolved=RESOLVED, path=PATH),
                  **(dict(seconds=round(time.time() - started, 1)) if not back.exists() else dict(seconds=made_with['solved_elsewhere']['seconds'], seconds_finishing_here=round(time.time() - started, 1))),
                  all_the_fitting_answers=complete if requests or asks else alone, **(dict(the_references_cells_alone=alone) if requests or asks else {}),
                  **(dict(row_requests=dict(at['declaration']['row_requests'], target_source=at['declaration']['target_source'],
                                            asked=[dict(answer=used[each['answer']]['id'], place=each['place'], merge=on['merges'][each['merge']],
                                                        direction_length=None if each['direction'] is None else float(each['direction'].norm()), size=each['size']) for each in asks])) if asks else {}),
                  **(dict(the_models_last_merge=last_merge) if last_merge is not None else {}),
                  weights_changed=int((kept != stands).sum()), made_with=made_with)
    tube.save(folder/'record.json', record)
    (folder/'solved.json').unlink()   # the whole record holds it
    print(json.dumps(dict(status=record['status'], selected_ridge_power=record.get('selected_ridge_power'), selected_round=record.get('selected_round'), weights_changed=record['weights_changed'],
                          objective=[record['all_the_fitting_answers'][name]['loss'] for name in ('before', 'after')], seconds=record['seconds'])), flush=True)


def write_link(run, at):
    """A chain's link written into the chain's one file (`linked`): the first written link makes the file, a copy of the chain's first state
    inside the run's folder, and every link writes its candidate where its matrix lies. The first state's own file is only read. After the
    writing the file is checked whole: this link's matrix and every earlier written link's hold their candidates value for value, and every
    byte outside those matrices is the first state's. links/<link>/written.json then says what was written and the file's hash, which the
    next link starts from. A link whose candidate changes nothing writes nothing, and its record names the state that stands. The file
    is a state of a chain under construction until `finished` lays model/written.json: no reader starts on it before."""
    chain, link, folder = at['chain'], at['link'], at['folder']
    record, out = json.loads((folder/'candidate'/'record.json').read_text()), folder/'written.json'
    if out.exists():
        raise ValueError('The link is already written')
    standing = dict(model=at['state']['model'], model_sha256=at['state']['model_sha256'])
    if record['status'] == 'unchanged':
        told = dict(written=False, link=link, tensor=at['state']['matrix'], why=record['selected'], the_state_that_stands=standing)
        tube.save(out, told)
        print(json.dumps(told), flush=True)
        return
    name = at['state']['matrix']
    kept, binding = torch.from_numpy(np.load(folder/'candidate'/f'{name}.npy')), at['binding']
    if not bool((tube.bf16(kept) == kept).all()):
        raise ValueError('the candidate is not a matrix the file holds')
    bits = tube.to_bf16(kept.numpy().ravel(order='F'))
    if binding['native_type'] != 'bf16' or bits.nbytes != binding['bytes']:
        raise ValueError('the stored matrix has another type or size')
    first, model = tube.ROOT/chain['first_state']['model'], tube.ROOT/chain['model']
    if run not in model.parents:
        raise ValueError('a chain writes its file inside its own run')
    if at['model'] == first:   # no earlier link wrote: the chain's file is made now
        if model.exists():
            raise ValueError('the chain\'s file exists before its first written link')
        model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(first, model)
    elif at['model'] != model:
        raise ValueError('the link starts from a file that is not the chain\'s')
    with model.open('r+b') as file:
        file.seek(binding['offset_bytes'])
        file.write(bits.tobytes())
        file.flush()
        os.fsync(file.fileno())
    held = lambda file, binding: torch.from_numpy(ladder.checkpoint_values(file, binding).reshape(binding['shape'][:2], order='F').astype('f4'))   # as ladder.matrix_in reads it, off the card
    bindings = json.loads((at['source']/'model'/'bindings.json').read_text())
    written, spans = [each['link'] for each in at['state']['links_before'] if each['written']] + [link], []
    for each in written:   # every written link's matrix is its candidate, value for value
        tensor = chain['matrix'].format(layer=each)
        mine = torch.from_numpy(np.load(run/'links'/str(each)/'candidate'/f'{tensor}.npy'))
        if not bool((held(model, bindings[tensor]) == mine).all()):
            raise ValueError(f'the chain\'s file does not hold link {each}\'s candidate at {tensor}')
        spans.append((bindings[tensor]['offset_bytes'], bindings[tensor]['offset_bytes'] + bindings[tensor]['bytes']))
    if first.stat().st_size != model.stat().st_size:
        raise ValueError('the chain\'s file is not the first state\'s size')
    around = []   # every byte outside the written matrices, in both files
    for each in (first, model):
        found, begin = hashlib.sha256(), 0
        with each.open('rb') as file:
            for start, end in sorted(spans) + [(each.stat().st_size, None)]:
                file.seek(begin)
                while begin < start:
                    block = file.read(min(1 << 24, start - begin))
                    found.update(block)
                    begin += len(block)
                begin = end
        around.append(found.hexdigest())
    if around[0] != around[1]:
        raise ValueError('the chain\'s file differs from its first state outside the matrices its links wrote')
    stood = held(first, binding)
    told = dict(written=True, link=link, tensor=name, model=str(model.relative_to(tube.ROOT)), model_sha256=tube.digest(model), started_from=standing, first_state=chain['first_state'],
                links_written=written, weights_changed=int((kept != stood).sum()), change_over_matrix=float((kept - stood).norm() / stood.norm()),
                every_written_link_holds_its_candidate=True, every_other_byte_is_the_first_states=True, selected_ridge_power=record['selected_ridge_power'], selected_round=record['selected_round'],
                standing='a state of a chain under construction: no candidate until model/written.json is laid')
    tube.save(out, told)
    print(json.dumps(told), flush=True)


def closed(run, whole=True):
    """Closes a chain. Finished (every link has its record): model/written.json, the one file a reader of the candidate waits for, with the
    file's hash and every link's record; with no link written, it says that nothing was. Stopped before its last link (`whole` False):
    model/partial.json instead, naming the links that are finished and the state that stands, and no model/written.json: a stopped chain
    is a prefix and no candidate (Astra, November 12, 2026: records/retrospectives/2026-11-12-late-chain-decision-astra.md)."""
    declaration = check_native_run(run/'model', run/'declaration.json')
    chain = declaration['chain']
    done = []
    for link in chain['links']:
        left = run/'links'/str(link)/'written.json'
        if not left.exists():
            break
        done.append(json.loads(left.read_text()))
    out = run/'model'/('written.json' if whole else 'partial.json')
    if (run/'model'/'written.json').exists() or out.exists():
        raise ValueError('The chain is already closed')
    if whole and len(done) != len(chain['links']):
        raise ValueError(f"the chain is not finished: {len(done)} of {len(chain['links'])} links have their record")
    written = [each for each in done if each['written']]
    model = tube.ROOT/chain['model']
    if written and tube.digest(model) != written[-1]['model_sha256']:
        raise ValueError('the chain\'s file is not what its last written link left')
    told = dict(written=bool(written), final=whole, links=[int(each) for each in chain['links']], links_finished=[each['link'] for each in done], links_written=[each['link'] for each in written],
                tensors=[each['tensor'] for each in written], copied_from=chain['first_state']['model'], copied_from_sha256=chain['first_state']['model_sha256'],
                **(dict(model=str(model.relative_to(tube.ROOT)), model_sha256=written[-1]['model_sha256'], every_written_link_holds_its_candidate=True, every_other_byte_is_the_first_states=True)
                   if written else dict(why='no link changed its matrix: the first state stands')),
                each_link=done, standing='the chain\'s one candidate' if whole else 'a stopped chain: the prefix that stands, and no candidate')
    out.parent.mkdir(parents=True, exist_ok=True)
    tube.save(out, told)
    print(json.dumps({name: value for name, value in told.items() if name != 'each_link'}), flush=True)


def write(run, link=None):
    """The run's model: a copy of the state the run starts from, inside the run's folder, with the candidate matrix written where that matrix
    lies, as a build writes one. The state's own file is only read. The copy is then checked: it holds the candidate at that matrix, value for
    value, and every other byte of it is the state's. model/written.json says what was written and both files' hashes. With a candidate that
    changes nothing, no copy is made. A chain's link is written by `write_link`."""
    check_native_run(run/'model', run/'declaration.json')
    at = started_from(run, link)
    if at['chain'] is not None:
        return write_link(run, at)
    record = json.loads((run/'candidate'/'record.json').read_text())
    kept = torch.from_numpy(np.load(run/'candidate'/f"{at['state']['matrix']}.npy"))
    if record['status'] == 'unchanged':
        print(json.dumps(dict(written=False, why=record['selected'])), flush=True)
        return
    binding, model = at['binding'], run/'model'/f'{run.name}.gguf'
    if not bool((tube.bf16(kept) == kept).all()):
        raise ValueError('the candidate is not a matrix the file holds')
    bits = tube.to_bf16(kept.numpy().ravel(order='F'))
    if binding['native_type'] != 'bf16' or bits.nbytes != binding['bytes']:
        raise ValueError('the stored matrix has another type or size')
    if model.exists():
        raise ValueError('The run already has a model')
    model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(at['model'], model)
    with model.open('r+b') as file:
        file.seek(binding['offset_bytes'])
        file.write(bits.tobytes())
        file.flush()
        os.fsync(file.fileno())
    held = lambda file: torch.from_numpy(ladder.checkpoint_values(file, binding).reshape(binding['shape'][:2], order='F').astype('f4'))   # as ladder.matrix_in reads it, off the card
    if not bool((held(model) == kept).all()):
        raise ValueError('the copy does not hold the candidate at that matrix')
    around = []   # every byte outside that matrix, in both files
    for each in (at['model'], model):
        found = hashlib.sha256()
        with each.open('rb') as file:
            for begin, end in ((0, binding['offset_bytes']), (binding['offset_bytes'] + binding['bytes'], each.stat().st_size)):
                file.seek(begin)
                while begin < end:
                    block = file.read(min(1 << 24, end - begin))
                    found.update(block)
                    begin += len(block)
        around.append(found.hexdigest())
    if around[0] != around[1]:
        raise ValueError('the copy differs from the state it was copied from outside the one matrix')
    stood = held(at['model'])
    told = dict(written=True, model=str(model.relative_to(tube.ROOT)), model_sha256=tube.digest(model), copied_from=at['state']['model'], copied_from_sha256=at['state']['model_sha256'],
                tensor=at['state']['matrix'], weights_changed=int((kept != stood).sum()), change_over_matrix=float((kept - stood).norm() / stood.norm()),
                every_other_byte_is_the_states=True, selected_ridge_power=record['selected_ridge_power'], selected_round=record['selected_round'])
    tube.save(run/'model'/'written.json', told)
    print(json.dumps(told), flush=True)


def card_taken(command):
    """The lock of the card a command computes on, held until the process ends: card 0's shared lock (tube.CARD), under which every pass
    through the model is made, unless the command is `candidate` and CORRECTION_CARD_LOCK names another. A candidate reads no model
    through the engine: it may be solved on another card of this machine under that card's own lock (the lock file's path), or on a
    machine whose card no other job takes ('none'). The card is the one CUDA_VISIBLE_DEVICES names, and the candidate's record says which
    it was and under which lock (made_with). Nothing of what is solved depends on it; a card's arithmetic is its own to the last digits,
    as two runs on one card already are (records/development/2026-11-12-correction-objective/README.md, the same solve made twice)."""
    named = os.environ.get('CORRECTION_CARD_LOCK')
    if command != 'candidate' or not named:
        tube.CARD.take()
        return None
    if named == 'none':
        return None
    held = open(named, 'a')
    fcntl.flock(held, fcntl.LOCK_EX)
    return held


if __name__ == '__main__':
    # write, finished and stopped read and write files. rows is one pass through the model and nothing else on the card: the pass takes
    # the card's lock by itself (run_a_observation.run, under tube.passed), so this process takes none. It used to take the lock at its start,
    # give it up for the pass and take it again after it, only to save the rows: three waits for the card where one pass needs one, and a
    # reading that found the lock free after the pass held the finished rows back (the overlap chain's rows:32, November 13: a pass of 93 s in
    # a command of 7 minutes).
    if sys.argv[1] not in ('rows', 'write', 'finished', 'stopped'):
        lock = card_taken(sys.argv[1])
    link = [int(sys.argv[3])] if len(sys.argv) > 3 else []   # a chain's commands name the link they work at
    {'reference': reference, 'rows': rows, 'candidate': candidate, 'write': write, 'finished': closed, 'stopped': lambda run: closed(run, whole=False)}[sys.argv[1]](ROUND_TWO_RUNS/sys.argv[2], *link)
