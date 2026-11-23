"""One matrix of the controlled training: the wrong pattern is moved into the correct pattern's own resolution where the two diverge, and every part of
the trace is held inside its own bound. The objective and the acceptance are Astra's of November 10, 2026; the solve is this file.

At one matrix's merge, with everything counted from the original model:
  target       where the two original patterns diverge (gap > 2 chance), the nearest point of the ball around the correct pattern whose radius is
               sqrt(2 chance), what two halves of one group differ by. Inside the ball nothing pulls.
  progress     the wrong group's mean state at each place (each member's mean over its steps there, then the mean over the members)
  bounds       at each place, each group's rows: per member the mean over its steps of the squared movement of the unit state, then the mean over the
               members; the wrong group's after taking their shared movement along the gap out where a move is asked
               at each place, each group's change of the log of the merged state's size, squared the same way
               each question's own positions: the mean squared movement of the unit state, and of the log of the size

fit      A first-order step through the merge, repeated. In a step every bounded part pushes back in proportion to its movement, with a strength that
         grows as the part nears its bound: the strength over what is left of the bound. After the step the matrix as the file would hold it goes
         through the merge itself, every row's actual movement is measured, and a part of the step that keeps every bound is found by halving: as
         finely as a linear solve resolves until a part holds, and to an eighth of the step between that part and one that fails.
         A solve starts inside every bound and ends when a step gains nothing.
brought_back  A merge that stands outside a bound when its matrix is reached (an earlier matrix's change put it there) is brought back before
         anything pulls: the least step from the matrix as it stands whose rows hold every bound to first order (Astra, November 11, with nothing
         pulling), written and read like any step, and repeated until every bound of the three groups and of all the fitting answers together
         is inside. A part of a step is taken when the written matrix reads less outside in all; the parts a step brings back are aimed inside
         their bounds by the most a writing of the solve was measured to add there; and the ridge starts at the path's floor and is
         halved, down to the smallest float32 tells from the unit, after every round that did not halve what is outside:
         with the bounds as constraints the ridge does not set where the least step lies, but a ridge that is large against a bound's own
         stiffness keeps a step from reaching it, and a small one leaves a larger change in the matrix
         (records/development/2026-11-10-controlled-training/README.md, the position2 rows and the ridge of a bringing back). A matrix that was brought back may pull like any other, and one at which no pull is kept stays as it was brought
         back (Astra, November 11).
behind   A feed-forward matrix is the last of ours before its layer's last merge, where the engine adds the per-layer embedding: a fixed function
         of the feed-forward's merge. So a feed-forward's solve reads that merge too, for every trial, through the engine's own arithmetic, and
         holds its bounds like its own merge's: in a bringing back, in every step of a pull, and in what is kept (Behind). What it left outside
         there nothing later could bring back (controlled-bf16-2 ended there).
written  The model stays in the file's own BF16, and so does every state of a solve: a solve stands on a matrix the file holds exactly, and what a step
         adds is written from there. BF16 keeps a weight to about one part in 256 and most of what a step asks of a weight is less than half of that,
         so a weight written by itself is either left where it was or moved a whole step: what the rows then read is not the step. A step is written
         as a whole instead (carried): a BF16 matrix near the wanted weights in the measure the solve's own linear solve is sped by, what one weight
         cannot hold made up for by the others. That measure sees what the direction bounds see; along the one output direction the size bounds
         see most, what they see is added to it (one direction, an approximation: at layer 3's attention it holds 0.82 of what they see). Every
         written matrix is then read through the merge like any trial, and only one that holds every bound and gains is stood on. On the stored rows of the matrices read so far, a step written this way holds the bounds the step was solved to
         hold, and written weight by weight the rounding alone put bounded parts several times past them
         (records/development/2026-11-10-controlled-training/bf16-writing.jsonl). Nothing in a solve keeps a second, finer copy of the weights.
choose   Which matrix is kept, decided inside the fitting answers. They are in three groups, and three branches go along one path: in each, one group
         is solved on, one must keep its bounds at every step (in a build its bounds push back in the solve as well, with no pull of its own), and
         one is only read after each step (it never enters a solve). The ridge starts at
         the unit, also after a bringing back, and is halved stage by stage, each branch starting a stage from where its last one ended, so a change grows a little at a time. A
         branch stops when its read group leaves a bound and keeps its last matrix inside; the others go on. What is kept is the candidate (a
         branch's stage end, a stopped branch's last matrix inside, or a stage's mean written as a whole) that holds every bound on each group and
         on all the fitting answers together and leaves all of them together the least tube loss.

What counts as inside is always read from the merge itself at the written weights against the bounds as they are. The numbers that only bound the
solver's work (EFFORT, BRACKET, PATH) decide how close a solve gets, never that.

A merge is read in two ways that are one arithmetic. A solve reads its trial changes through the merge rebuilt from the matrix's input and the state it
adds to: nothing else can say what a change not yet written would do. What a change that is written did is read from the engine's own output of that
merge: at the next read of a build, and in the sweep's acceptance. Both are counted from the engine's own output of the original file. The rebuilt
merge is the engine's output to the last digits of float32 once the matrix's input is taken as the engine multiplies it (`multiplied`): where the
engine reads several tokens at once, at the question's positions and each answer's first step, it rounds what goes into a matrix to the file's BF16
first. Until November 11 a solve took those rows as recorded and counted them from a rebuilt merge of the same kind; what it read of a matrix was then
not what the engine read of it at the answers' first place and at the questions
(records/development/2026-11-10-controlled-training/README.md, the engine's reading of position 2).

A run's commands, from code/signal-trace. No command has to come before another for its own sake: each needs only the files it reads, says which one
is missing, and never writes over one that exists. What is read "before any change" is read through the original file, whenever it is read:
  uv run -q --with torch --with numpy python -m pipeline.weight_edit.tube <command> <run>
  start       the run's own model file, to write into   baseline    the original file, read
  reproduced  the original file read a second time      reference   the patterns and bounds, written once
  own-start   the original file's second answers, and the budgets a sweep's own answers are held to
  build       the sweep                              composed    the finished model on the answers no fit used, their own tokens fed back
  own         the test: the finished model's own answers
  verdict     what the two readings say together, joined from their records (composed and own each end with it)"""
import contextlib
import copy
import difflib
import fcntl
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch

from pipeline.acquisition.export_stream_traces import StreamRows
from pipeline.native_records.native_trace import checkpoint_values
from pipeline.native_records.run_declaration import digest
from pipeline.overlap.sent_overlap import matrices
from pipeline.paths import ORIGINAL_MODEL, ROOT, ROUND_TWO_RUNS
from pipeline.weight_edit import fused, ladder
from pipeline.weight_edit.a2 import NORM_EPSILON, save, to_bf16
from pipeline.weight_edit.a2_pattern import KINDS, WIDE, model_of
from pipeline.weight_edit.ladder import DEVICE, Merge

FAMILIES = ('direction', 'size', 'question', 'question_size')
# What a run's bounds hold of the registered steps, as its declaration names it under `allowance` and its reference is made (frozen).
#   'the rows': at each place a group's rows may be displaced, on average, by the variance of its members' means over the number of members: one bound
#       a group and place for the direction and one for the log of the size (every run before November 11, and any run that names nothing).
#   'the mean and the rows apart' (the direction of November 11, journal/2026-11-11-mean-and-row-preservation.md; Astra,
#       records/retrospectives/2026-11-11-mean-and-row-preservation-astra.md): the group's mean movement at a place is held to that variance, and the
#       rows' average displacement to the original rows' own spread about their group's mean there. Two bounds a group and place for each of the
#       direction and the size (the rows' under the old names, the mean's under MEANS). It holds a group's pattern and how far its rows are
#       displaced on average; it makes no claim for any one answer or for the group's spread.
ALLOWANCES = ('the rows', 'the mean and the rows apart')
MEANS = ('direction_mean', 'size_mean')
ROTATIONS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))   # which group is solved on, which must keep its bounds, which is only read
EPS = float(torch.finfo(torch.float32).eps)
TINY = float(torch.finfo(torch.float32).tiny)   # a bound of nothing allows nothing: what moves there at all is past it, and what does not is not
RESOLVED = EPS ** 0.5   # where a linear solve stops, as a part of the slope it started with; also the least part of a bound that is counted as left when a part is weighed
# The caps on a solve's work. 80 iterations of a linear solve: at layer 3's attention 40 leave a quarter to a half of the slope at the two ridges where
# the bounds bind and 80 leave an eighth to a fifth, and the fitting answers are left 0.477 of their tube loss at 80 against 0.488 to 0.493 at 40
# (records/development/2026-11-10-controlled-training/bf16-choose.jsonl, lines 14, 17 and 19 against 8, 16 and 18).
# 10 passes of a bringing-back step, of 40 iterations of its linear solve each (back_iterations). A round's work was 5 passes of 80: at layer 0's
# feed-forward as controlled-bf16-2 met it, with the layer's last merge held as well and the ridge at the smallest float32 tells from the unit, 5
# passes bring every bound inside in 2 rounds and 53 s, 1 pass has not ended after 12 rounds, and 80 passes do not end by themselves, take 352 s for
# the first round and leave it 6.46 outside where 5 leave 0.019, which is not explained (records/development/2026-11-10-controlled-training/
# back-behind-ridge-23-passes-5.jsonl, -passes-1.jsonl, -passes-80.out). The same 400 iterations as 10 passes of 40 take fewer rounds and leave a
# smaller change, at two ridges and on both solvers (weight-research-perfomance, records/development/2026-11-10-speed-review/calibration/: from
# 2^-17, 3 rounds and 2.28% of the matrix against 5 rounds and 2.33%): the splitting needs its passes more than the linear solve its iterations.
# With the ridge halved as brought_back has it, 10 of 40 end that case in 10 rounds and 268 s with 2.21% changed and the case that ended
# controlled-bf16-1 in 3 rounds and 24 s with 0.97% (back-behind-halved.out, position2-behind-halved.out). The number was 80 passes in
# controlled-bf16-2, set at the ridge of the path's floor on one case (position2-near-passes.out).
EFFORT = dict(rounds=40, iterations=80, passes=10, back_iterations=40)
# How closely the part of a step that holds is bracketed in a solve, once a part that holds is found: to an eighth of the step. It decides how much of a
# step is used, never whether the part used holds: that is read exactly. A solve goes on from the part it took, so a part taken short costs it a round and
# not the result: at layer 3's attention the fitting answers are left the same with every search bracketed to 1/8, 1/64, 1/362 and 1/2896 (0.487, 0.485,
# 0.488, 0.493: lines 18, 20, 16 and 8 of the same record). Until a part holds, the search goes on halving as finely as a linear solve resolves (FOUND):
# a solve ends when no part holds, and a part of 1/16 can hold and gain where 1/8 does not, so a solve is never ended on the coarse bracket (Astra,
# November 10).
BRACKET = 1 / 8
FOUND = RESOLVED
# A change taken back in a build is bracketed as closely as a linear solve resolves: nothing goes on from there, so what is given up stays given up.
TAKEN_BACK = RESOLVED
PATH = 1   # powers of two between one stage's ridge and the next: the ridge is halved


# The 70 matrices are kept in the file's own BF16 and nothing else: the model is not converted, the solver writes into the format the model has
# (the direction of November 10, 2026: journal/2026-11-10-controlled-training.md).
STILL = 'no weight the file holds moved'   # what a part of a step answers when the matrix it writes is the one the solve already stands on
BLOCK = 128   # coordinates written before their leftovers are carried on in one product: it orders sums, bounds memory traffic, and changes no rule
# How a pass reads an answer's own text, which a run's declaration says as passes_read. 'given text' appends the answer to the question, so the engine reads
# all of it several tokens at once. 'as generated' reads the question that way and then decodes the answer's tokens one at a time, as the model did when it
# answered. The two are not the same arithmetic: reading several tokens at once, the engine rounds the input of a BF16 matrix to BF16 before multiplying, and
# decoding one token it does not (records/development/2026-11-10-controlled-training/operator_cpu.out). The patterns were recorded while the model generated.
READING = 'given text'
# Whether a pass ends each compute once the last output it asked for is written (the recorder's opt-in stop_after_last_label, for a run whose declaration
# asks for it): the outputs asked for are byte for byte those of a whole compute, and a pass that reads the first layers does not run the other thirty
# (records/development/2026-11-10-speed-review/README.md, recorder-early-stop). A run declares it only with the byte comparison made on its own card.
STOPS = False
# each answer's native recording of the original model: norms.npy, float64 [answer step, 105 merges, 2], column 1 the size of the state after the merge
NATIVE = ROUND_TWO_RUNS/'wide-overlap'/'exported'


class Card:
    """The card's shared lock (the one every project's job on card 0 takes). A command holds it while it computes on the card and gives it up around
    every pass through the model, which takes it by itself: one job on the card at a time, and another project's waiting job gets in at every pass.
    Before it gives the lock up, everything this process holds on the card is moved off it, and moved back once the lock is taken again, so a job
    that needs the whole card finds none of this process's rows there (only what the driver itself keeps for a process that has used the card)."""

    def __init__(self):
        self.file = None

    def take(self):
        if self.file is None:
            self.file = Path(f'/run/user/{os.getuid()}/gpu0.lock').open('a')
            fcntl.flock(self.file, fcntl.LOCK_EX)

    @contextlib.contextmanager
    def given(self):
        """Around a pass, and the reading of its stream into memory, which uses no card: what is on the card is moved off it and the lock, if held,
        is given up; after, the lock is taken again and the same tensors are moved back, each holding the values it held. Two tensors that shared their memory come back as two copies of it: nothing
        here writes through one to be read through another across a pass."""
        held = self.file is not None
        with warnings.catch_warnings():   # looking at every object wakes the deprecation notes of modules that are only being looked at
            warnings.simplefilter('ignore')
            parked = [each for each in gc.get_objects() if isinstance(each, torch.Tensor) and each.is_cuda]
        for tensor in parked:
            tensor.data = tensor.data.cpu()
        torch.cuda.empty_cache()
        if held:
            fcntl.flock(self.file, fcntl.LOCK_UN)
            self.file.close()
            self.file = None
        try:
            yield
        finally:
            if held:
                self.take()
            for tensor in parked:
                tensor.data = tensor.data.to(DEVICE)


CARD = Card()


def bf16(values):
    """Each value as the nearest one the file's BF16 holds, by the engine's own arithmetic on the bits (a2.to_bf16 and from_bf16: round to the nearest
    even, what is too small for a normal float32 to a signed zero)."""
    bits = values.contiguous().view(torch.int32).long() & 0xffffffff
    kept = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff
    kept = torch.where((bits & 0x7f800000) == 0, (bits & 0x80000000) >> 16, kept) << 16
    return (kept - ((kept >> 31) << 32)).to(torch.int32).view(torch.float32)


def multiplied(x, steps, reading=None):
    """A matrix's input as the engine multiplies it, at every answer step and then every question position (or at the answer steps alone). The rows
    given are changed where they are. `reading` is how the rows were read; without it, the way the run's passes read (READING). The rows recorded
    while the original model generated were read as generated, whatever a run's passes do.

    Where the engine reads several tokens at once it rounds what goes into a matrix to the file's BF16 before multiplying, and where it decodes one
    token it does not. The merge rebuilt that way is the engine's own output to the last digits of float32; rebuilt the other way it lies 7e-5 to 7e-4
    from it (records/development/2026-11-10-controlled-training: operator_cpu.out for the original file, position2-native-rows.out for a changed one).
    Read as generated, the question's positions and each answer's first step, which is decided at the prompt's last position, are read at once, and
    every later step alone; read as given text, everything is read at once. This is measured with this engine build on these prompts; with another
    engine, or prompts of a few tokens, the original file's reading has to show it again before it is relied on."""
    return rounded_where(x, read_together(x.shape[0], steps, reading, x.device))


def read_together(count, steps, reading=None, device=None):
    """Which of some rows (every answer step, then every question position) the engine reads several tokens at once: read as given text, all of them;
    read as generated, the question's positions and each answer's first step."""
    together = torch.ones(count, dtype=torch.bool, device=device or DEVICE)
    if (reading or READING) == 'as generated':
        begins = np.cumsum([0] + list(steps))
        together[:begins[-1]] = False
        together[torch.from_numpy(begins[:-1]).to(together.device)] = True
    return together


def rounded_where(x, together):
    """The rows given, rounded to the file's BF16 where `together` says, changed where they are."""
    for first in range(0, x.shape[0], 4096):   # in blocks: the rounding works on the bits, in wider integers
        block = together[first:first + 4096]
        if bool(block.any()):
            x[first:first + 4096][block] = bf16(x[first:first + 4096][block])
    return x


def carried(target, measure):
    """The BF16 matrix nearest to the wanted weights as the rows read them, not weight by weight. It is written one input coordinate at a time, in the
    file's order: a coordinate's weights go to the nearest values the file holds, and what that leaves over is carried onto the coordinates not yet
    written, so that all the leftovers together are small in the measure (measure_of). Weights that are on the grid already come back as they are:
    they leave nothing over, so nothing is carried.

    The measure has two parts, because the bounds do. How far a row's state turns is read off every output of the matrix alike, so a leftover is
    carried by the rows' inputs alone. How much a row's state grows or shrinks is read off one direction of the outputs only (the row's merged state,
    through the norm and the learned scale), and the size bounds are far tighter than the turn's: carried by the inputs alone, the leftovers along
    that direction are left as they fall, and at layer 3's attention every writing then moved the size at one place of the overlap by a third to
    all of its bound while the step itself moved it by a two-hundredth (bf16-choose.jsonl). So the leftover's part along that direction is carried
    in its own measure, the one with the size bounds in it, and the rest as before.

    What this is and is not (Astra, November 10): each coordinate's rounding is fixed once made, and what is carried makes up for it in a local
    quadratic stand-in for the trace, so the matrix is not the best one on the grid for the trace objective itself, and it holds no bound by being
    written this way. Whether it does is read from the merge, like any other trial. The file's order of coordinates is a fixed order, kept because it
    is the file's; another order writes another matrix (at layer 3's attention, most-seen-first read the same). Nothing is drawn at random. If the
    measure cannot be factored, this fails: nothing is added to it to make it factor.

    BLOCK coordinates are written before what they left over is carried on to all the later ones in one product. It is the same recursion with its
    sums taken in another order: a coordinate is always written with every earlier leftover already carried onto it.

    On a card the same operations are taken in the same order by fused.carried, three kernels a coordinate in place of about thirty: it writes this
    loop's matrix bit for bit at all four widths of the model's matrices (records/development/2026-11-10-speed-review/fused-writing/card.log,
    weight-research-perfomance, November 11), and this loop stays as what it is checked against and as what runs without a card."""
    if target.is_cuda:
        return fused.carried(target, measure, BLOCK=BLOCK, capture=False)
    left = target.double().clone()
    out = torch.empty_like(target)
    upper, along = measure['upper'], measure['along']
    toward, upper_sized = along if along is not None else (None, None)
    count = left.shape[0]
    for first in range(0, count, BLOCK):
        last = min(first + BLOCK, count)
        across = torch.empty((last - first, left.shape[1]), dtype=left.dtype, device=left.device)   # each coordinate's leftover across the size direction, over its own diagonal
        amounts = torch.zeros(last - first, dtype=left.dtype, device=left.device)                    # and its amount along that direction, over the sized diagonal
        for i in range(first, last):
            out[i] = bf16(left[i].float())
            over = left[i] - out[i].double()
            if toward is not None:   # the part of the leftover that changes the states' size is carried in the measure that has the size bounds in it
                amount = (over * toward).sum()
                amounts[i - first] = amount / upper_sized[i, i]
                over = over - amount * toward
                if i + 1 < last:
                    left[i + 1:last] -= (upper_sized[i, i + 1:last] * amounts[i - first])[:, None] * toward[None, :]
            across[i - first] = over / upper[i, i]
            if i + 1 < last:
                left[i + 1:last] -= upper[i, i + 1:last, None] * across[i - first][None, :]
        if last < count:
            left[last:] -= upper[first:last, last:].T @ across
            if toward is not None:
                left[last:] -= (upper_sized[first:last, last:].T @ amounts)[:, None] * toward[None, :]
    return out


def measure_of(bases, spread, ridge):
    """What a writing is nearest in: how much the rows and the ridge see of a leftover in the weights. It is the measure a solve's own linear solve is
    sped by (the bounds' action on the solve's rows, by its inputs, plus the stage's ridge), the mean of them over several solves, and is returned as
    the upper factor of its inverse (inverse = upper' upper), the form carried() reads, with the same for the size bounds' part beside it. It is that
    preconditioner and not the objective's own second derivative: it leaves out the merge's two projections and the pull, and it takes the rows'
    sizes as read off one direction of the outputs, the one that holds most of them (how much is on record in each solve, size_direction_share). At
    a large ridge it is mostly the ridge, and a writing in it mostly keeps each weight near its own value: making up for a leftover costs more
    there, as the ridge says it should."""
    def factor(h):
        # the upper factor of h's inverse from one factorization: h with its coordinates in reverse order is low low', and the inverse of low, turned
        # back round, is that factor (the same matrix as factoring h, inverting and factoring again, by fewer roundings and a third of the room)
        turned = (h + h.T).div_(2).flip(0, 1)
        low = torch.linalg.cholesky(turned)
        del turned
        inverse = torch.eye(h.shape[0], dtype=h.dtype, device=h.device)
        torch.linalg.solve_triangular(low, inverse, upper=False, out=inverse)
        del low
        return inverse.flip(0, 1)

    # a basis may wait off the card (fit keeps it there between the solves that use it); each is brought on by itself, and the sums are taken in the
    # order they are written in, one matrix at a time: at an input 12288 wide one such matrix is 1.1 GiB in double precision
    double = lambda value: value.to(DEVICE, torch.float64)
    whole = None
    for each in bases:
        rank, count = each.get('rank'), each['values'].shape[0]
        thin = each['vectors'].shape[1] < each['vectors'].shape[0]   # only the directions the action reads are kept (correction.basis_of, from the rows it is the gram of)
        if thin or (rank is not None and rank < count):
            # The action is a gram of `rank` rows of inputs (correction.basis_of), so it reads that many directions of the inputs and no
            # other: its part of the measure is made from the vectors of its `rank` largest values, and the ridge stands on every
            # direction. It is the same measure as the product over every eigenvector, without what a factoring's rounding leaves in the
            # values that are nothing.
            rank = count if thin else rank
            vectors = double(each['vectors'][:, count - rank:])
            one = (vectors * (double(each['values'][count - rank:]) * spread)) @ vectors.T
            one.diagonal().add_(ridge)
        else:
            vectors = double(each['vectors'])
            one = (vectors * (double(each['values']) * spread + ridge)) @ vectors.T
        del vectors
        whole = one if whole is None else whole.add_(one)
        del one
    whole.div_(len(bases))
    # the size bounds' own part: their action on the rows' inputs, along the one direction of the outputs that the rows' sizes are most read off.
    # Each solve has its own; for several solves one common direction is taken (theirs, laid the same way round, averaged), and each solve's
    # action counts by how much of its own direction lies along the common one, squared (Astra, November 10: the projected rank-one average).
    # What of a solve's size bounds lies across the common direction is left out, as what lies across its own already is.
    first = double(bases[0]['size_direction'])
    toward = sum(double(each['size_direction']) * torch.sign((double(each['size_direction']) * first).sum()) for each in bases)
    toward = toward / toward.norm()
    along = [float((double(each['size_direction']) * toward).sum()) for each in bases]
    upper = factor(whole)
    sized = None
    for cosine, each in zip(along, bases):
        one = cosine ** 2 * double(each['size_action'])
        sized = one if sized is None else sized.add_(one)
        del one
    whole.add_(sized.div_(len(bases)))   # from here on it is the measure with the size bounds' part in it
    del sized
    return dict(upper=upper, along=(toward, factor(whole)), size_directions_alike=min(abs(cosine) for cosine in along))


def written(target, measure=None):
    """Wanted weights as a matrix the file can hold: written as a whole in a measure (carried), or without one every weight to its own nearest value.
    What comes back is the matrix itself, and from here on it is the thing that is read, compared, kept and written to the file. No matrix is ever
    made again from the original and a difference: a difference of two float32 matrices does not hold every BF16 matrix (Astra, November 10: with
    w = 1 and a stored 2^-30, both on the grid, the difference is -1 and the sum 0). A difference is used inside a solve's objective, for the
    ridge, and in records, where it is a size."""
    return bf16(target) if measure is None else carried(target, measure)


def allowance_of(declaration):
    """The allowance a run's declaration names (ALLOWANCES); the first where it names none."""
    named = declaration.get('allowance', ALLOWANCES[0])
    if named not in ALLOWANCES:
        raise ValueError(f'The declaration names an allowance this file does not know: {named!r}')
    return named


def limits_of(bounds, question_bounds):
    """Every family's bounds at one merge: the registered steps', then the questions', then, where the mean and the rows are held apart, the means'."""
    out = dict(direction=bounds.direction, size=bounds.size, **question_bounds)
    if bounds.split:
        out.update({family: getattr(bounds, family) for family in MEANS})
    return out


def shares(parts):
    """Each group's share of all the groups' wrong answers, and of their correct ones: 2 x groups."""
    wrong = torch.tensor([float(each.wrong.sum()) for each in parts], device=DEVICE)
    right = torch.tensor([float((~each.wrong).sum()) for each in parts], device=DEVICE)
    return torch.stack([wrong / wrong.sum(), right / right.sum()])


def mean_inputs(part, values):
    """For each group and place of a part (the wrong group's places first), the sum over its registered steps of a value times the step's input: the
    one row of inputs a bound on that group's mean there answers to. 2 places x inputs."""
    out, pair = torch.zeros((2 * part.places, part.x.shape[1]), device=DEVICE), part.group * part.places + part.place
    for start in range(0, len(pair), 4096):
        out.index_add_(0, pair[start:start + 4096], part.x[part.at[start:start + 4096]] * values[start:start + 4096, None].float())
    return out


class Bounds:
    """What the original model fixes at one matrix's merge, frozen before any fit: where the wrong pattern is to move, and how far everything else may."""

    def __init__(self, gap, chance, direction, size, direction_mean=None, size_mean=None):
        self.gap, self.length = gap.float(), gap.double().norm(dim=1).float()   # places x width: the correct pattern minus the wrong one
        self.radius = (2 * chance.double()).sqrt().float()
        self.asked = (self.length > self.radius).float()                      # the same places as gap > 2 chance
        self.along = torch.where(self.length[:, None] > 0, self.gap / self.length.clamp(min=torch.finfo(torch.float32).tiny)[:, None], torch.zeros_like(self.gap))
        self.direction, self.size = direction.float(), size.float()           # 2 x places each: the wrong group's bounds, then the correct group's
        # With the mean and the rows held apart (ALLOWANCES), those two bound the rows' average displacement and these the group's mean movement.
        self.split = direction_mean is not None
        if self.split:
            self.direction_mean, self.size_mean = direction_mean.float(), size_mean.float()
        self.places = gap.shape[0]
        self.whole = float((self.asked * (self.length - self.radius) ** 2).sum())   # the tube loss of the original model
        # the strength at which every place at its bound costs as much as every ask unmet; with nothing asked there is no ask to weigh the bounds against
        self.unit = self.whole / (2 * self.places) if self.whole else 1.0

    def target(self, mean):
        """The nearest point of the ball, as a movement of the wrong pattern from where it started, and the places still outside it."""
        away = self.gap - mean
        far = away.norm(dim=1)
        return self.gap - self.radius[:, None] * away / far.clamp(min=torch.finfo(torch.float32).tiny)[:, None], self.asked * (far > self.radius).float()


class Layout:
    """Which registered step of which answer lies at which place of the axis, for some answers numbered in the order given, and which positions are whose question."""

    def __init__(self, wrong, steps, questions, pairs, places):
        at, cells, weights, start = [], [], [], 0
        for slot, (count_steps, each) in enumerate(zip(steps, pairs)):
            count = np.bincount(each[:, 1], minlength=places)
            at.append(start + each[:, 0])
            cells.append(slot * places + each[:, 1])
            weights.append(1 / count[each[:, 1]])   # one over the number of the member's steps at that place
            start += count_steps
        self.at, self.cells = (torch.from_numpy(np.concatenate(v)).to(DEVICE) for v in (at, cells))
        self.weight = torch.from_numpy(np.concatenate(weights)).float().to(DEVICE)
        self.wrong = torch.from_numpy(np.asarray(wrong)).to(DEVICE)
        self.n, self.places = len(steps), places
        self.asker = torch.from_numpy(np.repeat(np.arange(len(steps)), questions)).to(DEVICE)
        self.positions = torch.tensor(list(questions), device=DEVICE, dtype=torch.float32)
        # which bound and how many members each registered step answers to, for a solve's weights
        slot, self.place = self.cells // places, self.cells % places
        self.of_wrong = self.wrong[slot]
        self.group = (~self.of_wrong).long()
        self.members = torch.where(self.of_wrong, self.wrong.sum(), (~self.wrong).sum())

    def means(self, values):
        """Each member's mean at each place of the values of its steps there: members x places x width, in double precision."""
        out = torch.zeros((self.n * self.places, values.shape[1]), device=DEVICE, dtype=torch.float64)
        return out.index_add_(0, self.cells, values[self.at].double() * self.weight[:, None].double()).view(self.n, self.places, -1)

    def squares(self, values):
        """Each group's mean, over its members, of each member's mean at each place of the squares of its steps' values: 2 x places, the wrong group first."""
        per = torch.zeros(self.n * self.places, device=DEVICE).index_add_(0, self.cells, values[self.at][:, 0] ** 2 * self.weight).view(self.n, self.places)
        return torch.stack([per[self.wrong].mean(0), per[~self.wrong].mean(0)])

    def spreads(self, values, means):
        """Each group's mean, over its members, of each member's mean at each place of its steps' squared distance from the member's own mean there
        (`means`: members x places x width, as `means` gives it): 2 x places, the wrong group first, in double precision."""
        per, own = torch.zeros(self.n * self.places, device=DEVICE, dtype=torch.float64), means.reshape(self.n * self.places, -1)
        for start in range(0, len(self.at), 8192):
            at, cells = self.at[start:start + 8192], self.cells[start:start + 8192]
            per.index_add_(0, cells, ((values[at].double() - own[cells]) ** 2).sum(1) * self.weight[start:start + 8192].double())
        per = per.view(self.n, self.places)
        return torch.stack([per[self.wrong].mean(0), per[~self.wrong].mean(0)])

    def measured(self, turned, grown, turned_q, grown_q, bounds, more=None):
        """From every row's movement of the unit state and of the log of the merged state's size, at the answer steps and at the question's positions: the
        wrong group's mean movement at every place, and each family's movement in the form its bounds are in. Squares are taken per row, before any
        averaging inside a place or a question. Where the bounds hold the mean and the rows apart, each group's mean movement at a place is measured
        as well, as the square of the mean of its members' means there (the wrong group's after its shared movement along the gap is taken out where
        a move is asked), for the direction and for the log of the size; `more`, a dict if given, then receives the correct group's mean movement
        and both groups' mean change of the log of the size, which `pooled` makes the mean of several groups from."""
        moved = turned[self.at]
        members = torch.zeros((self.n * self.places, moved.shape[1]), device=DEVICE).index_add_(0, self.cells, moved * self.weight[:, None]).view(self.n, self.places, -1)
        mean = members[self.wrong].mean(0)
        shared = (bounds.asked * (mean * bounds.along).sum(-1))[:, None] * bounds.along
        left = moved - shared[self.place] * self.of_wrong[:, None]
        per = torch.zeros(self.n * self.places, device=DEVICE).index_add_(0, self.cells, (left ** 2).sum(1) * self.weight).view(self.n, self.places)
        sized = torch.zeros(self.n * self.places, device=DEVICE).index_add_(0, self.cells, grown[self.at][:, 0] ** 2 * self.weight).view(self.n, self.places)
        groups = lambda values: torch.stack([values[self.wrong].mean(0), values[~self.wrong].mean(0)])
        asked = lambda squares: torch.zeros(self.n, device=DEVICE).index_add_(0, self.asker, squares) / self.positions
        out = dict(direction=groups(per), size=groups(sized), question=asked((turned_q ** 2).sum(1)), question_size=asked(grown_q[:, 0] ** 2))
        if bounds.split:
            other = members[~self.wrong].mean(0)
            grew = groups(torch.zeros(self.n * self.places, device=DEVICE).index_add_(0, self.cells, grown[self.at][:, 0] * self.weight).view(self.n, self.places))
            out.update(direction_mean=torch.stack([((mean - shared) ** 2).sum(-1), (other ** 2).sum(-1)]), size_mean=grew ** 2)
            if more is not None:
                more.update(mean_right=other, size_means=grew)
        return mean, out


class Trace:
    """Some answers' rows at one matrix: its input, as the engine multiplies it (`multiplied`: the rows given are changed where they are), and the state it
    adds to at every answer step, then at every question position; the direction and the log
    of the size each row's merged state is counted from; each answer step's registered places; and each question's own bounds (question: the direction at
    its positions; question_size: the log of the size there).

    stood  for rows that are NOT the text the original was read on (a model's own answers): where each answer stood in the original's reading, place
           by place, at each merge the rows are read at (a list, the matrix's own merge first: answers x places x width for the unit state, answers
           x places for the log of the size). The rows' own origin is then nothing, and a reading that takes an answer's means at its places counts
           them from here (correction.means). No row of another text has a row of the original to be counted from."""

    def __init__(self, x, h, origin, log_origin, wrong, steps, questions, pairs, question_bounds, behind=None, stood=None):
        self.x, self.h, self.origin, self.log_origin = multiplied(x, steps), h, origin, log_origin
        self.behind = behind   # for a feed-forward matrix: the layer's last merge, which lies behind its own (behind_of)
        self.stood = stood
        self.wrong, self.steps, self.questions, self.pairs = np.asarray(wrong), list(steps), list(questions), pairs
        self.question_bounds = {family: value.float() for family, value in question_bounds.items()}
        self.begins = np.cumsum([0] + self.steps)
        self.question_begins = self.begins[-1] + np.cumsum([0] + self.questions)

    def part(self, pick, places):
        return Part(self, [int(i) for i in np.flatnonzero(np.asarray(pick))], places)

    def against(self, state, bounds, weights, g, picks, places):
        """For a feed-forward matrix's trace, once the matrix `weights` is in the model and the engine's own output at the layer's last merge has been
        read at this trace's rows (`state`): the bounds a solve holds there are drawn in by how far the engine's reading stands past this file's own
        reading of the same matrix, bound by bound, the furthest over the picked groups of answers (`picks`), on top of what earlier calls drew in.
        The two readings differ where the engine reads several tokens at once (Behind): what was measured is taken off the bound the solve works to,
        (sqrt(b) - (sqrt(engine) - sqrt(own)))^2, never added to one, and what counts stays the engine's own output against the bounds as they are
        (build). Returns the most any bound was drawn in by, as a part of the bound's root."""
        to, log = unit_and_log(state)
        there = self.behind
        drawn = {family: torch.zeros_like(value) for family, value in limits_of(there['bounds'], there['question_bounds']).items()}
        for pick in picks:
            part = self.part(pick, places)
            own, held = part.realized(bounds, weights, g)['behind']['over'], part.behind
            engines = part.views()[1].read(held.bounds, to[part.rows], log[part.rows], to[part.asking], log[part.asking], held.origin, held.log_origin, held.origin_q, held.log_origin_q)['over']
            for family in drawn:
                apart = (engines[family].sqrt() - own[family].sqrt()).clamp(min=0)
                if apart.dim() == 2:
                    drawn[family] = torch.maximum(drawn[family], apart)
                else:
                    drawn[family][part.ids] = torch.maximum(drawn[family][part.ids], apart)
        inward = lambda value, family: (value * (1 - drawn[family]).clamp(min=0) ** 2).clamp(min=TINY)
        tighter = copy.copy(there['bounds'])
        for family in ('direction', 'size', *(MEANS if tighter.split else ())):
            setattr(tighter, family, inward(getattr(there['bounds'], family), family))
        self.behind = dict(there, bounds=tighter, question_bounds={family: inward(value, family) for family, value in there['question_bounds'].items()})
        return max(float(value.max()) for value in drawn.values())


class Part(Layout):
    """The rows of some of a trace's answers, numbered within the part."""

    def __init__(self, trace, ids, places):
        super().__init__(trace.wrong[ids], [trace.steps[i] for i in ids], [trace.questions[i] for i in ids], [trace.pairs[i] for i in ids], places)
        rows = torch.from_numpy(np.concatenate([np.arange(trace.begins[i], trace.begins[i + 1]) for i in ids])).to(DEVICE)
        asking = torch.from_numpy(np.concatenate([np.arange(trace.question_begins[i], trace.question_begins[i + 1]) for i in ids])).to(DEVICE)
        self.rows, self.asking, self.ids = rows, asking, torch.tensor(ids, device=DEVICE)
        self.x, self.h, self.origin, self.log_origin = trace.x[rows], trace.h[rows], trace.origin[rows], trace.log_origin[rows]
        self.xq, self.hq, self.origin_q, self.log_origin_q = trace.x[asking], trace.h[asking], trace.origin[asking], trace.log_origin[asking]
        self.question_bounds = {family: value[torch.tensor(ids, device=DEVICE)] for family, value in trace.question_bounds.items()}
        self.behind = None if trace.behind is None else Behind(trace.behind, rows, asking, self.ids)
        self.stood = None if getattr(trace, 'stood', None) is None else [(to[self.ids], log[self.ids]) for to, log in trace.stood]

    def bounds_of(self, bounds):
        return limits_of(bounds, self.question_bounds)

    def frames(self, bounds):
        """The bounds of every merge this part's rows are read at: the matrix's own, and behind a feed-forward's the layer's last."""
        return [bounds] if self.behind is None else [bounds, self.behind.bounds]

    def views(self):
        """This part as each of those merges has it: the same rows and places, with that merge's bounds of the questions."""
        if self.behind is None:
            return [self]
        later = copy.copy(self)
        later.question_bounds, later.behind = self.behind.question_bounds, None
        return [self, later]

    def read(self, bounds, to, log, to_q, log_q, origin, log_origin, origin_q, log_origin_q):
        """A reading of states as they are at one merge: the tube loss left there, and every bounded movement over its bound."""
        more = {}
        mean, moved = self.measured(to - origin, log - log_origin, to_q - origin_q, log_q - log_origin_q, bounds, more)
        beyond = ((bounds.gap - mean).norm(dim=1) - bounds.radius).clamp(min=0) * bounds.asked
        return dict(tube=float((beyond ** 2).sum()), inside=int(((beyond == 0) & (bounds.asked > 0)).sum()), mean=mean, shared=bounds.asked * (mean * bounds.along).sum(-1),
                    over={family: moved[family] / limit.clamp(min=TINY) for family, limit in self.bounds_of(bounds).items()}, **more)

    def realized(self, bounds, weights, g):
        """What the given weights do to this part's rows, through the merge itself: the tube loss left, and every bounded movement over its bound. Behind a
        feed-forward's merge the layer's last merge is read as well, through the engine's own arithmetic from the one to the other (Behind), and is in
        the reading as `behind`: a reading of the same form against that merge's own bounds, which `outside`, `worst` and `pooled` count in."""
        now, asking = Merge(self.x, self.h, weights, g), Merge(self.xq, self.hq, weights, g)
        reading = self.read(bounds, now.to, now.size.log(), asking.to, asking.size.log(), self.origin, self.log_origin, self.origin_q, self.log_origin_q)
        if self.behind is not None:
            there = self.behind
            (later, _), (later_q, _) = there.merges(self, now, asking)
            reading['behind'] = dict(self.views()[1].read(there.bounds, later.to, later.size.log(), later_q.to, later_q.size.log(), there.origin, there.log_origin, there.origin_q, there.log_origin_q),
                                     bounds=there.bounds)
        return reading


GELU = (0.79788456080286535587989211986876, 0.044715)   # the engine's GELU on a card (ggml-cuda/unary.cuh): 0.5 x (1 + tanh(a x (1 + b x^2)))


def gelu(x, slope=False):
    """The engine's GELU, or its slope."""
    a, b = GELU
    inner = torch.tanh(a * x * (1.0 + b * x * x))
    if slope:
        return 0.5 * (1.0 + inner) + 0.5 * x * (1.0 - inner * inner) * a * (1.0 + 3.0 * b * x * x)
    return 0.5 * x * (1.0 + inner)


def per_layer_inputs(members, layer, tensors, together):
    """Each row's own per-layer input at one layer (rows: every answer step, then every question position): what the engine multiplies into the gated
    state before the per-layer embedding is projected and added (llama.cpp src/models/gemma4.cpp, build_inp_per_layer and project_per_layer_inputs).
    It is the token's alone: its row of the per-layer table, and its scaled embedding through the model projection and that projection's norm. No
    matrix of ours writes any of it, so it is made from the original file. Where the engine reads several tokens at once it rounds what goes into the
    model projection to the file's BF16, like the input of any BF16 matrix (`multiplied`); with that, the layer's last merge rebuilt from the
    engine's own output of the feed-forward's merge is the engine's own output there to the last digits of float32 at the rows read alone
    (records/development/2026-11-10-controlled-training/README.md, layer-map)."""
    tokens = np.array([(m['tokens'] + m['recorded'][:-1])[m['question'] + s] for m in members for s in range(m['steps'])] + [m['tokens'][p] for m in members for p in range(m['question'])])
    unique, inverse = np.unique(tokens, return_inverse=True)
    width, small = tensors['token_embd.weight']['shape'][0], tensors['per_layer_proj_norm.weight']['shape'][0]

    def table(name, columns):   # the used tokens' rows of a token table, which the file keeps as BF16: the upper half of a float32
        binding = tensors[name]
        if binding['native_type'] != 'bf16' or binding['shape'][0] != columns:
            raise ValueError(f'{name} is not the token table expected')
        rows = np.memmap(ORIGINAL_MODEL, dtype='<u2', mode='r', offset=binding['offset_bytes'], shape=(binding['shape'][1], columns))
        return torch.from_numpy((np.ascontiguousarray(rows[unique]).astype(np.uint32) << 16).view(np.float32)).to(DEVICE)

    def held(name):
        binding = tensors[name]
        return torch.from_numpy(checkpoint_values(ORIGINAL_MODEL, binding).reshape(binding['shape'][:2], order='F').astype('f4')).to(DEVICE)

    count = tensors['per_layer_token_embd.weight']['shape'][0] // small
    scaled = table('token_embd.weight', width) * float(np.sqrt(np.float32(width)))
    own = table('per_layer_token_embd.weight', small * count)[:, layer * small:(layer + 1) * small] * float(np.sqrt(np.float32(small)))
    project = held('per_layer_model_proj.weight')[:, layer * small:(layer + 1) * small]
    scale = torch.from_numpy(checkpoint_values(ORIGINAL_MODEL, tensors['per_layer_proj_norm.weight']).astype('f4').ravel()).to(DEVICE)

    def of(given):
        projected = (given @ project) * float(np.float32(1.0) / np.sqrt(np.float32(width)))
        normed = projected / torch.sqrt((projected * projected).mean(1, keepdim=True) + NORM_EPSILON) * scale
        return (normed + own) * float(np.float32(1.0) / np.sqrt(np.float32(2.0)))

    pick = torch.from_numpy(inverse).to(DEVICE)
    return torch.where(together[:, None], of(bf16(scaled.clone()))[pick], of(scaled)[pick])


def behind_of(run, layer, members, tensors, pick, origin_of):
    """What a feed-forward matrix's trace needs of the layer's last merge, for the picked answers `members`: the engine's fixed arithmetic from the
    feed-forward's merge to it (the gate, each row's per-layer input, the projection, its norm's learned scale), where its rows are counted from, and
    its frozen bounds."""
    held = lambda name: torch.from_numpy(checkpoint_values(ORIGINAL_MODEL, tensors[name]).reshape(tensors[name]['shape'][:2], order='F').astype('f4')).to(DEVICE)
    together = read_together(sum(m['steps'] + m['question'] for m in members), [m['steps'] for m in members])
    bounds, question_bounds = bounds_of(run, 'layer', layer, pick)
    origin, log_origin = origin_of('layer', layer)
    return dict(gate=held(f'blk.{layer}.inp_gate.weight'), proj=held(f'blk.{layer}.proj.weight'),
                scale=torch.from_numpy(checkpoint_values(ORIGINAL_MODEL, tensors[f'blk.{layer}.post_norm.weight']).astype('f4').ravel()).to(DEVICE),
                t=per_layer_inputs(members, layer, tensors, together), together=together, origin=origin, log_origin=log_origin, bounds=bounds, question_bounds=question_bounds)


class Behind:
    """For some rows of a feed-forward matrix: the layer's last merge, which lies behind the matrix's own and which no matrix of ours writes. The
    engine adds the per-layer embedding there: pe_out = pe_in + scale * norm((gelu(pe_in @ gate) * t) @ proj), with t the row's own per-layer input
    (per_layer_inputs). It is a fixed function of the feed-forward's merge, so a solve of the feed-forward matrix reads it for every trial and holds
    its bounds like its own merge's: the feed-forward matrix is the last of ours before it, and what is left outside there nothing later brings back
    (controlled-bf16-2 ended there). The gate and the projection are BF16 matrices, so where the engine reads several tokens at once what goes into
    each is rounded to the file's BF16 first, as in `multiplied`. Those roundings are of states this file rebuilds, which are the engine's to the last
    digits of float32 only: where a value lies that close to where BF16 rounds the other way, the engine and this file round apart, and the row's state
    here lies up to 1e-4 from the engine's (about one row in twenty of those read together, by 1e-5 or less in most; the rows read alone agree to
    1e-6). What the engine itself puts out there is read after the matrix is written, and is what counts (build)."""

    def __init__(self, whole, rows, asking, ids):
        self.gate, self.proj, self.scale, self.bounds = whole['gate'], whole['proj'], whole['scale'], whole['bounds']
        self.t, self.t_q, self.together, self.together_q = whole['t'][rows], whole['t'][asking], whole['together'][rows], whole['together'][asking]
        self.origin, self.log_origin, self.origin_q, self.log_origin_q = whole['origin'][rows], whole['log_origin'][rows], whole['origin'][asking], whole['log_origin'][asking]
        self.question_bounds = {family: value.float()[ids] for family, value in whole['question_bounds'].items()}

    def merges(self, part, now, asking, rounded=True):
        """The layer's last merge at the answer steps and at the question's positions, from the feed-forward's merges there, each with what the gate
        put out (the slope of the GELU is taken there). `rounded` is True, as the engine rounds, or what each rounding added at other merges of
        these rows (`roundings`): then nothing is rounded here, and in each rounding's place stands what it added there. Away from those
        merges that is not what the engine puts out: it is the same function with the roundings held, which has a slope everywhere (Stand)."""
        out = []
        for k, (merge, state, t, together) in enumerate(((now, part.h, self.t, self.together), (asking, part.hq, self.t_q, self.together_q))):
            merged = state + merge.g * merge.sent
            if rounded is True:
                gated = rounded_where(merged.clone(), together) @ self.gate
                sent = rounded_where(gelu(gated) * t, together)
            else:
                gated = (merged + rounded[k][0]) @ self.gate
                sent = gelu(gated) * t + rounded[k][1]
            out.append((Merge(sent, merged, self.proj, self.scale), gated))
        return out

    def roundings(self, part, now, asking):
        """What each rounding on the way to the layer's last merge adds at these merges, row by row, at the answer steps and then at the
        question's positions: to what goes into the gate, and to what goes into the projection (nothing at a row read alone). At the same
        merges, `merges` with these makes the engine's own values again: a value and its rounding lie within a part in a hundred of each
        other, so float32 holds their difference exactly, and the value plus that difference is the rounding itself."""
        out = []
        for merge, state, t, together in ((now, part.h, self.t, self.together), (asking, part.hq, self.t_q, self.together_q)):
            merged = state + merge.g * merge.sent
            into = rounded_where(merged.clone(), together)
            sent = gelu(into @ self.gate) * t
            out.append((into - merged, rounded_where(sent.clone(), together) - sent))
        return out


class Stand:
    """A group's rows at the matrix a solve stands on: its merges at the answer steps and at the question's positions, where its rows stand against
    what they are counted from, and what output changes do to them to first order; one of each for every merge the rows are read at (Part.frames).

    `rounded` is True, as the engine rounds, or what each rounding added where another stand of these rows stood (`roundings`): then the
    layer's last merge is rebuilt with the roundings held there (Behind.merges). Where the engine reads several tokens at once it rounds what
    goes into the per-layer embedding's gate and projection to the file's BF16, and a rounding that turns over under a step has no slope:
    what it adds to a row does not shrink with the step. With the roundings held where the first stand stood, the rows at that matrix are
    the engine's own, and their change with the matrix is to first order the one `moves` gives there, whose slopes are taken at the rounded
    states. What a step does past `moves` is then what the first order leaves out of a function that has a slope; between two stands as
    the engine rounds them it also carries every rounding that turned."""

    def __init__(self, part, weights, g, rounded=True):
        self.part = part
        self.now, self.asking = Merge(part.x, part.h, weights, g), Merge(part.xq, part.hq, weights, g)
        self.moved = [(self.now.to - part.origin, self.now.size.log() - part.log_origin, self.asking.to - part.origin_q, self.asking.size.log() - part.log_origin_q)]
        self.later = self.through = None
        if part.behind is not None:
            there = part.behind
            self.through = (there.gate, there.proj)
            (later, gated), (later_q, gated_q) = there.merges(part, self.now, self.asking, rounded)
            self.later = (later, gelu(gated, slope=True) * there.t, later_q, gelu(gated_q, slope=True) * there.t_q)
            self.moved.append((later.to - there.origin, later.size.log() - there.log_origin, later_q.to - there.origin_q, later_q.size.log() - there.log_origin_q))

    def roundings(self):
        """What each rounding on the way to the layer's last merge adds where this stand stands (Behind.roundings), for a stand at another
        matrix to hold (`rounded`). None where no merge lies behind the matrix's own: nothing is rounded with a step there."""
        return None if self.part.behind is None else self.part.behind.roundings(self.part, self.now, self.asking)

    def grain(self):
        """How finely the size of each row is read, laid like a movement (moved): the spacing of float32 at the log of the row's size, at each merge
        read, with nothing for the directions. A row's change of the log of its size is read as a whole number of these, so a written step's
        reading can lie this far from the step's own model at every row, whichever way. (A direction is read over the state's whole width, where
        the roundings of its many parts do not add up to a step of the reading; the log of a size is one number a row.)"""
        spacing = lambda size: (lambda log: torch.nextafter(log, torch.full_like(log, float('inf'))) - log)(size.log())
        out = [(torch.zeros_like(self.now.to), spacing(self.now.size), torch.zeros_like(self.asking.to), spacing(self.asking.size))]
        if self.later is not None:
            later, _, later_q, _ = self.later
            out.append((torch.zeros_like(later.to), spacing(later.size), torch.zeros_like(later_q.to), spacing(later_q.size)))
        return out

    def moves(self, dy, dy_q):
        """What output changes at the answer steps and at the question's positions do to first order at each merge read: the movement of the unit state
        and the change of the log of the size, at the answer steps and then at the question's positions."""
        full, full_q = state_change(self.now, dy), state_change(self.asking, dy_q)
        out = [(*split(self.now, full), *split(self.asking, full_q))]
        if self.later is not None:
            later, slope, later_q, slope_q = self.later
            out.append((*split(later, self.carried_on(later, slope, full)), *split(later_q, self.carried_on(later_q, slope_q, full_q))))
        return out

    def carried_on(self, later, slope, full):
        """What a change of the feed-forward's merged state adds to the layer's last, to first order: itself, and what it changes of the per-layer
        embedding (the roundings are left out: they have no slope)."""
        gate, proj = self.through
        return full + state_change(later, ((full @ gate) * slope) @ proj)

    def reaches(self):
        """For each merge read: stand-ins for the merges at the answer steps and the question's positions that say how much of an output change reaches
        the state there, without what the per-layer embedding adds to it (action_of reads only that)."""
        out = [(self.now, self.asking)]
        if self.later is not None:
            out.append(tuple(types.SimpleNamespace(g=own.g, sent=own.sent, rms=own.rms, width=own.width, to=later.to, size=later.size, ratio=1 / (later.size * own.rms))
                             for own, later in ((self.now, self.later[0]), (self.asking, self.later[2]))))
        return out



def state_change(merge, dy):
    """What an output change adds to the merged state, to first order, through the norm and the learned scale."""
    return merge.g * ((dy - merge.sent * (merge.sent * dy).sum(1, keepdim=True) / merge.width) / merge.rms)


def split(merge, full):
    """A change of a merged state as the movement of its unit state and the change of the log of its size."""
    along = (merge.to * full).sum(1, keepdim=True)
    return (full - merge.to * along) / merge.size, along / merge.size


def movement(merge, dy):
    """The two parts of one output change, to first order: the movement of the unit state (Merge.forward) and the change of the log of the merged state's size."""
    return split(merge, state_change(merge, dy))


def gram(x, scale):
    """The rows' inputs against themselves, each row weighted, in double precision, a block of rows at a time."""
    total = torch.zeros((x.shape[1], x.shape[1]), device=DEVICE, dtype=torch.float64)
    for start in range(0, x.shape[0], 4096):
        block = x[start:start + 4096].double()
        total += (block * scale[start:start + 4096, None].double()).T @ block
    return total


def halved(holds, bracket=None):
    """A part of a step, between none and all of it, at which the reading holds: all of it when that holds, and otherwise found by halving between a part that
    holds and one that does not, until the two are within the bracket of each other (BRACKET unless another is given). While no part has held yet the
    halving goes on until they are within FOUND, so a solve is not told that no part holds because the search stopped early. Past the part returned there
    may be others that hold; they are not looked for. A part can also be too small to move any weight the file holds (STILL). That is not a part that fails: what holds, if anything does, lies above
    it, so the search goes on above it; and when all of the step moves nothing, no part of it will, and STILL is the answer. Returns the part, its
    reading, and how many halvings it took."""
    reading = holds(1.0)
    if reading is not None:
        return 1.0, reading, 0
    low, high, part, best, count = 0.0, 1.0, 0.0, None, 0
    bracket = BRACKET if bracket is None else bracket
    while high - low > (bracket if best is not None else min(bracket, FOUND)):
        middle, count = (low + high) / 2, count + 1
        reading = holds(middle)
        if reading is None:
            high = middle
        elif reading is STILL:
            low = middle
        else:
            low, part, best = middle, middle, reading
    return part, best, count


def together_of(parts, means, squares, bounds):
    """Several groups at the same weights as one reading of all their answers together: the wrong pattern's mean movement, its shared amount along the gap,
    and the direction and size squares. A group's movement at a place is a mean over its members, so the groups' means weigh by their members; where a move
    is asked the wrong answers' movement is counted after taking out their shared movement along the gap, which all together share one amount of and each
    group its own, so the spread of the groups' amounts comes on top."""
    by = shares(parts)   # 2 x groups
    out = {family: sum(by[:, k, None] * each[family] for k, each in enumerate(squares)) for family in ('direction', 'size')}
    shared = torch.stack([bounds.asked * (mean * bounds.along).sum(-1) for mean in means])   # groups x places
    amount = (by[0][:, None] * shared).sum(0)
    out['direction'] = torch.stack([out['direction'][0] + (by[0][:, None] * (shared - amount) ** 2).sum(0), out['direction'][1]])
    return sum(by[0][k] * mean for k, mean in enumerate(means)), amount, out


def pooled(parts, readings, bounds):
    """Several groups' readings at the same weights, as the one reading of all their answers together (together_of). Every question keeps its own bounds.
    Where the bounds hold the mean and the rows apart, the mean movement of all the answers together is the square of their one mean, the wrong
    answers' after the amount all of them share along the gap is taken out: never the mean of the groups' squared means."""
    limits = dict(direction=bounds.direction.clamp(min=TINY), size=bounds.size.clamp(min=TINY))
    mean, amount, squares = together_of(parts, [reading['mean'] for reading in readings], [{family: reading['over'][family] * limits[family] for family in limits} for reading in readings], bounds)
    over = {family: squares[family] / limits[family] for family in limits}
    over.update({family: torch.cat([reading['over'][family] for reading in readings]) for family in ('question', 'question_size')})
    beyond = ((bounds.gap - mean).norm(dim=1) - bounds.radius).clamp(min=0) * bounds.asked
    out = dict(tube=float((beyond ** 2).sum()), inside=int(((beyond == 0) & (bounds.asked > 0)).sum()), mean=mean, shared=amount, over=over)
    if bounds.split:
        by = shares(parts)
        other = sum(by[1][k] * reading['mean_right'] for k, reading in enumerate(readings))
        sizes = sum(by[:, k, None] * reading['size_means'] for k, reading in enumerate(readings))
        over.update(direction_mean=torch.stack([((mean - amount[:, None] * bounds.along) ** 2).sum(-1), (other ** 2).sum(-1)]) / bounds.direction_mean.clamp(min=TINY),
                    size_mean=sizes ** 2 / bounds.size_mean.clamp(min=TINY))
        out.update(mean_right=other, size_means=sizes)
    if 'behind' in readings[0]:   # the layer's last merge behind a feed-forward's, against its own bounds
        there = readings[0]['behind']['bounds']
        out['behind'] = dict(pooled(parts, [reading['behind'] for reading in readings], there), bounds=there)
    return out


def outside(readings, families=None):
    """How far past their bounds the parts of some readings are, in all: at the merge read, and where a reading has the layer's last merge behind it
    (`behind`), there as well. Every family a reading has counts (the means' too, where its bounds hold the mean and the rows apart), or `families`."""
    every = [*readings, *(reading['behind'] for reading in readings if 'behind' in reading)]
    return float(sum((reading['over'][family] - 1).clamp(min=0).sum() for reading in every for family in (families or reading['over'])))


def worst(reading):
    """The furthest any part of a reading is along or past its bound, at the merge read or at the layer's last merge behind it."""
    return max(float(each['over'][family].max()) for each in (reading, *([reading['behind']] if 'behind' in reading else [])) for family in each['over'])


def nearest(reading):
    """The family whose part is furthest along or past its bound."""
    return max(reading['over'], key=lambda family: float(reading['over'][family].max()))


def furthest(reading, bounds):
    """The part of a reading that is furthest along or past its bound: its family, how far, and for a direction or a size the group, the place and
    whether a move is asked there; for a question, which one."""
    if 'behind' in reading and worst(reading['behind']) > max(float(reading['over'][family].max()) for family in reading['over']):
        return dict(furthest(reading['behind'], reading['behind']['bounds']), merge='the layer\'s last')
    family = nearest(reading)
    over = reading['over'][family]
    index = int(over.argmax())
    if over.dim() == 2:
        group, place = divmod(index, over.shape[1])
        return dict(family=family, over=float(over[group, place]), group=('wrong', 'correct')[group], place=place, a_move_is_asked_there=bool(bounds.asked[place]))
    return dict(family=family, over=float(over[index]), question=index)


def left_outside(reading, bounds):
    """Every part of a reading that stands past its bound, the furthest first: its family, how far, and for a direction or a size its bound, the
    group, the place and whether a move is asked there; for a question, which one. A part at the layer's last merge behind the merge read says so."""
    out = []
    for family, over in reading['over'].items():
        for index in (over.flatten() > 1).nonzero().flatten().tolist():
            if over.dim() == 2:
                group, place = divmod(index, over.shape[1])
                out.append(dict(family=family, over=float(over[group, place]), bound=float(getattr(bounds, family)[group, place]), group=('wrong', 'correct')[group], place=place,
                                a_move_is_asked_there=bool(bounds.asked[place])))
            else:
                out.append(dict(family=family, over=float(over[index]), question=index))
    if 'behind' in reading:
        out += [dict(each, merge='the layer\'s last') for each in left_outside(reading['behind'], reading['behind']['bounds'])]
    return sorted(out, key=lambda each: -each['over'])


def top_cells(reading, bounds, family, count=5):
    """The parts of one family of a reading that are furthest along or past their bounds, the furthest first: group, place, and whether a move is asked there."""
    over = reading['over'][family]
    values, flat = over.flatten().topk(count)
    return [dict(group=('wrong', 'correct')[int(i) // over.shape[1]], place=int(i) % over.shape[1], over=float(v), a_move_is_asked_there=bool(bounds.asked[int(i) % over.shape[1]]))
            for v, i in zip(values, flat)]


def summary(reading, bounds, families=None):
    """A reading in plain numbers, for a record: every family the reading has, or `families`."""
    out = dict(tube_loss_left=reading['tube'] / bounds.whole if bounds.whole else None, places_inside_the_ball=reading['inside'], places_asked=int(bounds.asked.sum()))
    for family in (families or reading['over']):
        over = reading['over'][family].flatten().double()
        out[family] = dict(parts=int(over.numel()), over_their_bound=int((over > 1).sum()), median=float(over.median()), largest=float(over.max()))
    if 'behind' in reading:
        out['the_layers_last_merge'] = summary(reading['behind'], reading['behind']['bounds'], families)
    return out


def laid(parts, moves, bounds, together=False):
    """The rows each bound of some groups measures, from every row's movement of the unit state and of the log of its size (answer steps, then question
    positions, group by group), and each group's mean movement of the wrong pattern. Per group four sets of rows: the direction at every registered
    step (the wrong answers' after their shared movement along the gap is taken out where a move is asked), the size there, the direction at every
    question position, the size there. With `together`, two more for all the groups as one reading (together_of): every group's direction rows and
    how far each group's shared amount lies from the amount all share, and every group's size rows. Where the bounds hold the mean and the rows
    apart, per group two more: one row a group and place (the wrong group's places first), the group's mean movement there (the wrong answers'
    after their shared movement along the gap is taken out where a move is asked), and its mean change of the log of the size; and with
    `together` two more again, the same of all the groups' answers as one (the wrong answers' after the amount all share is taken out). It is
    linear in the movements: the rows of a sum are the sum of the rows."""
    means, rows, own = [], [], []
    for each, (turned, more, turned_q, more_q) in zip(parts, moves):
        moved = turned[each.at]
        members = torch.zeros((each.n * each.places, moved.shape[1]), device=DEVICE, dtype=moved.dtype).index_add_(0, each.cells, moved * each.weight[:, None]).view(each.n, each.places, -1)
        mean = members[each.wrong].mean(0)
        amount = bounds.asked * (mean * bounds.along).sum(-1)
        left = moved - (amount[:, None] * bounds.along)[each.place] * each.of_wrong[:, None]
        means.append(mean)
        own.append((left, more[each.at], amount))
        rows += [[left], [own[-1][1]], [turned_q], [more_q]]
        if bounds.split:
            grew = torch.zeros((each.n * each.places, 1), device=DEVICE, dtype=more.dtype).index_add_(0, each.cells, own[-1][1] * each.weight[:, None]).view(each.n, each.places, 1)
            own[-1] += (members[~each.wrong].mean(0), grew[each.wrong].mean(0), grew[~each.wrong].mean(0))
            rows += [[torch.cat([mean - amount[:, None] * bounds.along, own[-1][3]])], [torch.cat(own[-1][4:])]]
        del members
    if together:
        wrong = torch.stack([each.wrong.sum() for each in parts]).float()
        common = sum(wrong[k] / wrong.sum() * each[2] for k, each in enumerate(own))
        rows.append([each[0] for each in own] + [(each[2] - common)[:, None] for each in own])
        rows.append([each[1] for each in own])
        if bounds.split:
            by = shares(parts)
            rows.append([torch.cat([sum(by[0][k] * mean for k, mean in enumerate(means)) - common[:, None] * bounds.along, sum(by[1][k] * each[3] for k, each in enumerate(own))])])
            rows.append([torch.cat([sum(by[g][k] * each[4 + g] for k, each in enumerate(own)) for g in range(2)])])
    return means, rows


def laid_out(parts, bounds, together=False):
    """For the rows `laid` gives: the bounds of each set, and for each of its row arrays which bound a row answers to and its share in that bound's mean
    of squares, so that a bound's measure is the sum over its rows of share times the row's square (Layout.measured, together_of). `of` says
    what the rows of each array are: a group's registered steps, its question positions, or one row a place; `family` whose bounds they are."""
    blocks = []
    one_a_bound = lambda: (torch.arange(2 * bounds.places, device=DEVICE), torch.ones(2 * bounds.places, device=DEVICE))   # a mean's bound measures its one row whole
    for k, each in enumerate(parts):
        pair, share, asked = each.group * each.places + each.place, each.weight / each.members, 1 / each.positions[each.asker]
        limits = each.bounds_of(bounds)
        blocks += [dict(family='direction', limit=limits['direction'].flatten(), rows=[(pair, share)], of=[('steps', k)]), dict(family='size', limit=limits['size'].flatten(), rows=[(pair, share)], of=[('steps', k)]),
                   dict(family='question', limit=limits['question'], rows=[(each.asker, asked)], of=[('question', k)]),
                   dict(family='question_size', limit=limits['question_size'], rows=[(each.asker, asked)], of=[('question', k)])]
        if bounds.split:
            blocks += [dict(family=family, limit=limits[family].flatten(), rows=[one_a_bound()], of=[('means', k)]) for family in MEANS]
    if together:
        counts = torch.stack([torch.stack([each.wrong.sum(), (~each.wrong).sum()]) for each in parts]).float()   # groups x (wrong, correct)
        totals = counts.sum(0)
        every = [(each.group * each.places + each.place, each.weight / totals[each.group]) for each in parts]
        places = torch.arange(bounds.places, device=DEVICE)
        steps = [('steps', k) for k in range(len(parts))]
        blocks.append(dict(family='direction', limit=bounds.direction.flatten(), of=steps + [('places', k) for k in range(len(parts))],
                           rows=every + [(places, torch.full((bounds.places,), float(counts[k, 0] / totals[0]), device=DEVICE)) for k in range(len(parts))]))
        blocks.append(dict(family='size', limit=bounds.size.flatten(), rows=every, of=steps))
        if bounds.split:
            blocks += [dict(family=family, limit=getattr(bounds, family).flatten(), rows=[one_a_bound()], of=[('means', None)]) for family in MEANS]
    for block in blocks:
        if not bool((block['limit'] > 0).all()):
            raise ValueError('a bound of nothing is a linear equality, which this solve does not take')
    return blocks


def measures(block, rows):
    """Each bound's measure of its rows: the sum of every row's share times its square."""
    total = torch.zeros_like(block['limit'])
    for (cell, share), each in zip(block['rows'], rows):
        total.index_add_(0, cell, (each ** 2).sum(1) * share)
    return total


def onto(blocks, rows):
    """The nearest rows that hold every bound, in each bound's own measure: the rows of a bound that is passed, scaled back onto it; the others as they are."""
    out = []
    for block, some in zip(blocks, rows):
        scale = (block['limit'] / measures(block, some).clamp(min=TINY)).sqrt().clamp(max=1)
        out.append([each * scale[cell][:, None] for (cell, share), each in zip(block['rows'], some)])
    return out



def action_of(solved, now, asking, weight, width):
    """What speeds a linear solve and what a writing is nearest in, from one round's strengths: the bounds' action without the merge's two projections
    (each row's input, weighted by its parts' strengths and by how much of an output change reaches its state), as its eigenvectors and values, and
    the size bounds' action along the one direction of the outputs that the rows' sizes are most read off. It changes how fast a solve gets there,
    not where. `solved` are the groups whose bounds are in it, `now` and `asking` their merges at the matrix stood on (answer steps and question
    positions), `weight` each group's strength per bound, `width` the matrix's outputs."""
    action = torch.zeros((solved[0].x.shape[1], solved[0].x.shape[1]), device=DEVICE, dtype=torch.float64)
    moment, sizes, mean_sizes = torch.zeros((width, width), device=DEVICE, dtype=torch.float64), [], []
    for k, each in enumerate(solved):
        per_row = torch.zeros(each.x.shape[0], device=DEVICE).index_add_(0, each.at, weight[k]['direction'][each.group, each.place] / each.members * each.weight)
        both = gram(each.x, per_row * now[k].ratio[:, 0] ** 2)
        both += gram(each.xq, (weight[k]['question'] / each.positions)[each.asker] * asking[k].ratio[:, 0] ** 2)
        if 'direction_mean' in weight[k]:   # a bound on a group's mean at a place acts on one row: the mean of its steps' inputs, each as far as it reaches
            both += gram(mean_inputs(each, each.weight / each.members * now[k].ratio[each.at, 0]), weight[k]['direction_mean'].flatten())
        action += both
        del both
        # the size bounds, for the writing: each row's strength, and the direction of the outputs its size is read off (movement's `along`)
        per_row = torch.zeros(each.x.shape[0], device=DEVICE).index_add_(0, each.at, weight[k]['size'][each.group, each.place] / each.members * each.weight)
        for x, merge, strength in ((each.x, now[k], per_row), (each.xq, asking[k], (weight[k]['question_size'] / each.positions)[each.asker])):
            scaled = merge.g * merge.to
            reach = (scaled - merge.sent * (merge.sent * scaled).sum(1, keepdim=True) / merge.width) / (merge.rms * merge.size)
            moment += (reach * (strength * (x ** 2).sum(1))[:, None]).double().T @ reach.double()
            sizes.append((x, strength, reach))
            if x is each.x and 'size_mean' in weight[k]:
                mean_sizes.append((each, reach, weight[k]['size_mean'].flatten()))
            del scaled
    torch.cuda.empty_cache()
    spectrum, directions = torch.linalg.eigh(moment)
    toward = directions[:, -1].float()
    sized = None
    for x, strength, reach in sizes:
        one = gram(x, strength * (reach @ toward) ** 2)
        sized = one if sized is None else sized.add_(one)
        del one
    for each, reach, strength in mean_sizes:   # the bounds on a group's mean size at a place, along the same one direction of the outputs
        sized.add_(gram(mean_inputs(each, each.weight / each.members * (reach @ toward)[each.at]), strength))
    share = float(spectrum[-1] / spectrum.sum())
    del moment, sizes, mean_sizes, spectrum, directions
    # It is kept as it was summed, in double precision, and off the card, where the eigenvectors need the room at the widest matrices. Rounded
    # to float32 it is no longer the matrix the measure is made of: at layer 0's feed-forward its entries reach 2.6e7, the rounding left
    # eigenvalues down to -0.64, and the measure could not be factored once the ridge was under that (controlled-bf16-1, November 11).
    sized = sized.cpu()
    torch.cuda.empty_cache()
    values, vectors = torch.linalg.eigh(action)
    del action
    found = dict(values=values.clamp(min=0).float(), vectors=vectors.float(), size_direction=toward, size_action=sized, size_direction_share=share)
    del values, vectors, sized
    torch.cuda.empty_cache()
    return found


def brought_back(parts, bounds, w, g, stands, ridge, effort=EFFORT, say=None, least=None):
    """A matrix on the file's grid at which every bound of these groups, and of all their answers together, is back inside, starting from the matrix
    as it stands (on the grid as well; w is the original, which a change is told from). Nothing pulls. All the answers together are a reading of
    their own, with the bounds every group has: each group can be inside while all together are not, because the groups' shared amounts along the
    gap differ. That reading is held in the step, counts as outside, and has to be inside for the solve to end. Behind a feed-forward's merge the
    layer's last merge is read and held with it, bound for bound (Part.realized, Behind): its rows are among the rows of every step.

    A round's step (Astra, November 11, "A well-posed subproblem at a binding bound", with nothing pulling): with the merges taken to first order
    at the matrix the solve stands on, the least step p from that matrix whose rows hold every bound as the bound's own rows measure it,
    ||r_c + C_c p||^2 <= b_c. It is found by splitting the rows from the step: the rows are aimed at the nearest rows that hold every bound
    (onto), the step is the one whose rows lie nearest that aim (one linear system, the same in every pass), and what the step's rows keep
    missing the aim by is carried into the next aim. The passes end when the step's rows are inside the bounds aimed at and the aim has stopped
    moving, both as closely as a linear solve resolves, or at the passes allowed.

    What was measured on the case that ended controlled-bf16-1, layer 1's attention with its parts outside standing 387 times their bounds past
    them in all (a sum, not a count) and the furthest 34.6 times its bound (records/development/2026-11-10-controlled-training/README.md, the
    position2 rows), and is built in here:
      the step counts from the matrix stood on, not from the original: counted from the original every round also asked to undo what the last
        writing's rounding left in the weights, which no row sees and the grid cannot hold;
      a part of a step is taken when the written matrix reads less outside in all (the sum over every bound of how far past it the part
        stands). A written step leaves parts that were inside a little past their bounds, by its rounding and by the first order; they are
        outside in the next round's reading and are brought back with the rest. The solve ends inside only with every bound inside;
      the bound a part is brought back to is its bound less the most a writing of this solve was measured to add there: with E what the grid
        left over of a wanted matrix and nu the bound's own measure of E's rows to first order, rows that held (sqrt(b) - sqrt(nu))^2 before
        that writing held b after it (the triangle inequality). The most over the solve's writings, not the last one's: after a small step
        the last one's is next to nothing, a part left 0.3% past its bound was then aimed so little inside it that no weight the file holds
        took it there, and one run of four on the case below ended that way (position2-native-stalled.out). For the next writing this is
        headroom to aim at, not a certificate (Astra, November
        11, records/retrospectives/2026-11-11-restoration-and-progress-astra.md): its leftover, its first order and what lies beyond the first
        order are its own, and where nu is more than b no aim makes up for it. A part that stands inside is held where it stands, not
        drawn inward by this, until a round of which no part left less outside (below);
      and less how finely its rows are read (Stand.grain). A row's change of the log of its size is read in float32, as a whole number of
        the spacing of float32 at that log, so a size reading stands on a grid. At layer 4's attention, on the rows kept after
        controlled-bf16-3, the spacing is 4.77e-7 at every row and the root of the smallest size bounds is 21 to 27 spacings: one row one
        spacing nearer moves such a reading by up to 3.2% of its bound, and at bounds near 1e-7 by 0.03% to 0.10%
        (records/development/2026-11-10-controlled-training/position8-grain-*.json). That sweep ended there: one size part 0.2% past its
        bound through thirteen rounds while the ridge was halved to its floor. The least step that brings such a part to its bound changes
        its rows' logs by less than a spacing, the written matrix reads as it did, and no part of the step leaves less outside. In the
        trials below that named them, every part outside at a round of which no part left less outside was a size part, 0.005% to 1.0%
        past a bound of 4.8e-11 to 1.2e-7. So an aim lies inside its bound by the grain as well, every row one spacing off, in the bound's own measure: by
        the triangle inequality again a written step then reads inside the bound where its own model holds the aim, whichever way each row
        is rounded;
      after a round of which no part left less outside, every part, inside or outside, is aimed inside its bound, by the grain and by the
        most a whole written step of this solve has since read past its own model at that bound (short: what the grid left over, what lies
        past the first order and what the reading resolves, together, as read), and the round is made again at the same ridge before any
        smaller one. What was tried, on the rows kept after controlled-bf16-3 and on eight matrices kept where solves on them stalled
        (records/development/2026-11-10-controlled-training/README.md, position 8, the end): with the most any writing of the solve left
        over as that margin (commit 4e5b354f), a stall took up to eight rounds with steps of a thousandth of the matrix; with what the
        stalled step's writing left over to first order, one solve of fourteen ended outside; with what the whole step read past its
        model and no grain, forty solves of forty from the kept rows ended inside and four of sixteen from the eight kept matrices ended
        outside, one size part 0.006% and 0.05% past a bound of 9.3e-8 after 14 to 16 rounds; with the grain in every aim, sixteen of
        sixteen from the eight matrices ended inside in 2 to 4 rounds, and twelve solves of twelve from the kept rows in 3 to 4 rounds
        with every step taken whole, where eleven of the forty before had a round of which no part was taken. Whether a written matrix is
        inside is read from the merge against the bounds as they are, as always;
      where such a round comes with no smaller ridge left, it is made again at that ridge with the margins it has just grown, while rounds
        are left; the solve ends there only when the round grew no margin, since the same round would then be made again.
        controlled-bf16-4 ended at such a round, its nineteenth of forty, at layer 9's feed-forward, with one part 0.07% past its bound. On
        the rows kept there (records/development/2026-11-10-controlled-training/README.md, position 19, the end; card 0) the same solve
        ends inside five times of five. Of 59 rounds played at the smallest ridge from states those solves kept, one took no part of its
        step, and that one ended its solve: the whole step, 0.05% of the matrix, brought the one part outside back in and put two others
        0.04% and 0.0006% past theirs. That state was played eighteen times, and the first round, the same under either rule, took no
        part of its step in six. Ending there (four of the eighteen), the three solves whose round took none ended outside. With the
        round made again (fourteen), all fourteen ended inside in one to four rounds, the three whose first round took none after one,
        one and three such rounds. What this does not reach: a state where the least step is too small for the file to hold. One was
        made on card 1 by beginning a solve again from the matrix it stood on, 0.07% outside, at the starting ridge with no margin
        kept: the step was 0.0007% of the matrix and left the part where it stood, in thirteen rounds of fifteen and then, made again
        at the smallest ridge, in twenty-six rounds of twenty-six, twice. So beginning again is not built in, though sixteen solves of
        sixteen begun from kept matrices ended inside at layer 4's attention; and in such a state this rule spends the rounds left and
        ends outside as before;
      the ridge is the caller's (choose): with the bounds as constraints it does not set where the least step lies, only how a writing splits
        what it leaves over between the weights and the rows and how fast the splitting gets there. How fast is most of it: a bound's rows are
        held with a stiffness of one over the bound, so against one ridge the loose bounds are held far more weakly than the tight ones, and a
        step reaches a passed bound only as far as its stiffness outweighs the ridge. A small ridge reaches every bound at once and leaves a
        larger change in the matrix. So with `least`, the smallest ridge it may go down to, the ridge is halved for the next round after a round that
        was taken and did not halve what is outside, and after one of which no part left less outside;
    The step is found on all the rows. It was tried on the rows of the bounds that are passed alone, the others joining when the step would pass
    them (the least step lies in the span of the inputs of the rows whose bounds bind): at layer 0's feed-forward the 42 bounds passed lie on 60
    rows, and the step found there would pass 1,379 others, whose rows with the first are more than the matrix has inputs
    (records/development/2026-11-10-controlled-training/README.md, the row span).
    `say`, if given, is told every round as it ends. A round's record keeps, pass by pass, the linear solve's iterations, how far the step's
    rows stood past the bounds aimed at, and how far the aim still moved.
    Returns the last matrix reached, its readings (one per group), the record of every round, why the solve ended, and whether every bound is
    inside."""
    spread = float((g ** 2).mean())
    weights = stands
    readings = [each.realized(bounds, weights, g) for each in parts]
    together = len(parts) > 1
    whole = (lambda readings: [pooled(parts, readings, bounds)]) if together else (lambda readings: [])
    beyond = lambda readings: outside(readings) + outside(whole(readings))
    rho = bounds.unit / len(parts)   # how hard a bound's rows are held to their aim inside a round's step: it sets how fast that step is found, not which
    frames = parts[0].frames(bounds)   # the bounds of every merge read: the matrix's own, and behind a feed-forward's the layer's last
    views = [[each.views()[m] for each in parts] for m in range(len(frames))]
    blocks = [block for m, frame in enumerate(frames) for block in laid_out(views[m], frame, together)]
    laid_all = lambda moves: [rows for m, frame in enumerate(frames) for rows in laid(views[m], [each[m] for each in moves], frame, together)[1]]
    stiff = [[rho * share / block['limit'][cell] for cell, share in block['rows']] for block in blocks]
    weigh = lambda rows: float(sum((hard * (each ** 2).sum(1)).sum() for some, hards in zip(rows, stiff) for each, hard in zip(some, hards))) ** 0.5
    rounds, status, inverse, measure, added, halved_ridge = [], 'the rounds allowed were used', None, None, None, 0
    inward = False   # whether the parts inside are aimed inside their bounds too: from the first round of which no part left less outside
    margin = None    # from that round on: the most a written step of this solve read past its own model at each bound (short, below)
    lower = lambda: least is not None and ridge * 2.0 ** -PATH >= least   # whether there is a smaller ridge to go on at
    for number in range(effort['rounds']):
        if beyond(readings) == 0:
            break
        started = time.time()
        stood = [Stand(each, weights, g) for each in parts]
        rows_now = laid_all([each.moved for each in stood])

        def answer(p):
            """What a step does, to first order, to the rows every bound measures."""
            return laid_all([stand.moves(each.x @ p, each.xq @ p) for each, stand in zip(parts, stood)])

        def slope(p, aim=None):
            """With an aim: the slope at p of the step's own size and of every bound's rows held to their aim. Without one: that function's second
            derivative times p, from the terms that are quadratic in p alone, so that no large constant slope is subtracted from another (Astra,
            November 11)."""
            p = p.detach().requires_grad_(True)
            rows = answer(p)
            value = 0.5 * ridge * (p ** 2).sum()
            if aim is None:
                value = value + 0.5 * sum((hard * (each ** 2).sum(1)).sum() for some, hards in zip(rows, stiff) for each, hard in zip(some, hards))
            else:
                value = value + 0.5 * sum((hard * ((start + each - to) ** 2).sum(1)).sum()
                                          for some, starts, tos, hards in zip(rows, rows_now, aim, stiff) for each, start, to, hard in zip(some, starts, tos, hards))
            return torch.autograd.grad(value, p)[0]

        if inverse is None:
            # What speeds the linear solves and what the steps are written nearest in, made once from the bounds' own stiffness at every merge read:
            # the system is the same in every pass and nearly so in every round.
            reach = [each for stand in stood for each in stand.reaches()]
            found = action_of([each for each in parts for _ in frames], [a for a, _ in reach], [b for _, b in reach],
                              [{family: rho / limit.clamp(min=TINY) for family, limit in view.bounds_of(frame).items()} for each in parts for view, frame in zip(each.views(), frames)], w.shape[1])
            vectors, under = found['vectors'], found['values'] * spread + ridge
            measure = measure_of([found], spread, ridge)
            del found, reach
            toward, sized = measure['along']
            near = toward.float()

            def inverse(s):
                # through the eigenvectors (as one float32 matrix it loses the small ridges: card1-probe-table.out), and along the one direction of the
                # outputs the sizes are read off, the measure with the size bounds in it
                first = vectors @ ((vectors.T @ s) / under[:, None])
                return first + ((sized.T @ (sized @ (s.double() @ toward))).float() - first @ near)[:, None] * near[None, :]

        with torch.no_grad():   # how finely each bound's rows are read at this matrix, in the bound's own measure: every row one spacing off
            grain = [measures(block, rows) for block, rows in zip(blocks, laid_all([each.grain() for each in stood]))]
        tight = []   # the bounds this round's step aims at
        for k, (block, some) in enumerate(zip(blocks, rows_now)):
            here = measures(block, some)
            edge = margin if inward else added
            reach = block['limit'].sqrt() - grain[k].sqrt() - (0 if edge is None else edge[k].sqrt())
            less = (reach.clamp(min=0) ** 2).clamp(min=TINY)
            tight.append(dict(block, limit=torch.where(here > block['limit'], less, less if inward else torch.minimum(torch.maximum(less, here), block['limit']))))
        step, aim, count, passes, each_pass = torch.zeros_like(weights), onto(tight, rows_now), 0, 0, []
        owed = [[torch.zeros_like(each) for each in some] for some in rows_now]
        first_slope, past, still = None, None, None
        for passes in range(1, effort['passes'] + 1):
            left = -slope(step, [[to - over for to, over in zip(tos, overs)] for tos, overs in zip(aim, owed)])
            if first_slope is None:
                first_slope = left.norm()
            lead = inverse(left)
            direction, inner, before_pass = lead.clone(), (left * lead).sum(), count
            for _ in range(effort.get('back_iterations', effort['iterations'])):
                if left.norm() <= first_slope * RESOLVED:
                    break
                count += 1
                bent = slope(direction)
                rate = inner / (direction * bent).sum()
                step += rate * direction
                left -= rate * bent
                lead = inverse(left)
                inner, before = (left * lead).sum(), inner
                direction = lead + (inner / before) * direction
            with torch.no_grad():
                reached = [[start + each for start, each in zip(starts, some)] for starts, some in zip(rows_now, answer(step))]
            past = max(float((measures(block, some) / block['limit']).max()) for block, some in zip(tight, reached)) ** 0.5 - 1
            wanted = [[each + over for each, over in zip(some, overs)] for some, overs in zip(reached, owed)]
            new = onto(tight, wanted)
            owed = [[each - to for each, to in zip(some, tos)] for some, tos in zip(wanted, new)]
            still = weigh([[to - old for to, old in zip(tos, olds)] for tos, olds in zip(new, aim)]) / max(weigh(new), TINY)
            aim = new
            del reached, wanted, new
            each_pass.append(dict(iterations=count - before_pass, slope_left=float(left.norm() / first_slope), past_a_bound_in_the_model=past, aim_still_moving=still))
            if not bool(torch.isfinite(step).all()) or (past <= RESOLVED and still <= RESOLVED):
                break
        del aim, owed
        record = dict(round=number, ridge_halvings=halved_ridge, every_part_aimed_inside=inward, iterations=count, passes=passes, slope_left=float(left.norm() / first_slope), past_a_bound_in_the_model=past, aim_still_moving=still,
                      step_over_matrix=float(step.norm() / w.norm()), outside_before=beyond(readings), each_pass=each_pass, seconds_of_the_step=round(time.time() - started, 1))
        if not bool(torch.isfinite(step).all()):
            rounds.append(dict(record, taken=False, halvings=0))
            status = 'the linear solve gave no finite step'
            break

        whole_step = []   # the whole step as the file holds it (halved tries all of a step first)

        def short(stored, meant):
            """How far past the step's own model a written matrix reads, at each bound, in the bound's own measure: the root of what the merge
            itself reads there at the written matrix, less the root of what the step as meant gives to first order, squared where it is
            more. It holds everything between the two at once: what the grid left over, what lies past the first order, and what a reading
            in float32 resolves at that bound."""
            model = [measures(block, [start + each for start, each in zip(starts, some)]) for block, starts, some in zip(blocks, rows_now, answer(meant))]
            read = [measures(block, some) for block, some in zip(blocks, laid_all([Stand(each, stored, g).moved for each in parts]))]
            return [(now.sqrt() - then.sqrt()).clamp(min=0) ** 2 for now, then in zip(read, model)]

        def holds(share):
            stored = written(weights + share * step, measure)
            if share == 1.0:
                whole_step.append(stored)
            if bool((stored == weights).all()):
                return STILL
            trial = [each.realized(bounds, stored, g) for each in parts]
            return (trial, stored) if beyond(trial) < beyond(readings) else None

        share, trial, halvings = halved(holds)
        record.update(part_of_the_step_taken=share, halvings=halvings, seconds=round(time.time() - started, 1))
        if trial is None or trial is STILL:
            status = 'the whole step moves no weight the file holds' if trial is STILL else 'no part of a step left less outside'
            # The solve has stalled: the parts still outside are a little past bounds so small that the step to them, once written, reads no
            # better. What the whole step just tried read past its own model at each bound is measured, and the next step aims every part,
            # inside or outside, that far inside its bound: first at the same ridge, then at smaller ones.
            with torch.no_grad():
                tried = short(whole_step[0], step)
            grown = margin is None or any(bool((new > old).any()) for new, old in zip(tried, margin))
            margin = tried if margin is None else [torch.maximum(old, new) for old, new in zip(margin, tried)]
            rounds.append(dict(record, taken=False, the_whole_step_read_past_its_model_at_most=max(float((nu / block['limit']).max()) for block, nu in zip(blocks, tried)),
                               parts_outside=[dict(each, read=name) for name, reading in zip([*range(len(readings)), 'all together'], [*readings, *whole(readings)])
                                              for each in left_outside(reading, bounds)]))
            if say:
                say(dict(bringing_back_round=number, taken=False, **{name: record[name] for name in ('ridge_halvings', 'outside_before', 'passes', 'iterations', 'past_a_bound_in_the_model', 'halvings', 'seconds')}))
            if not inward or (trial is STILL and grown):   # aimed again at the same ridge, with what a writing of this size was read to add
                inward, status = True, 'the rounds allowed were used'
                continue
            if lower():   # at a smaller ridge a step reaches further, and moves weights the file holds
                ridge, inverse, halved_ridge, status = ridge * 2.0 ** -PATH, None, halved_ridge + 1, 'the rounds allowed were used'
                continue
            if least is not None and grown:   # no smaller ridge is left: the round is made again at this one, with the margins it has just grown (the docstring)
                status = 'the rounds allowed were used'
                continue
            break
        readings, stored = trial
        if inward:
            with torch.no_grad():
                margin = [torch.maximum(old, new) for old, new in zip(margin, short(stored, share * step))]
        with torch.no_grad():   # what this writing left over, as each bound's rows measure it (to first order, at the matrix the step was taken from)
            left_over = [measures(block, rows) for block, rows in zip(blocks, answer(stored - (weights + share * step)))]
        added = left_over if added is None else [torch.maximum(old, new) for old, new in zip(added, left_over)]   # the most a writing of this solve left at each bound
        weights = stored   # the solve stands on the matrix that was written and read, itself, and on nothing finer
        rounds.append(dict(record, taken=True, outside=beyond(readings), largest_over_its_bound=max(worst(r) for r in readings + whole(readings)),
                           change_over_matrix=float((weights - w).norm() / w.norm()),
                           the_writing_added_at_most=max(float((nu / block['limit']).max()) for block, nu in zip(blocks, left_over))))
        if say:
            say(dict(bringing_back_round=number, **{name: rounds[-1][name] for name in ('ridge_halvings', 'outside_before', 'outside', 'largest_over_its_bound', 'part_of_the_step_taken', 'passes',
                                                                                         'iterations', 'past_a_bound_in_the_model', 'halvings', 'seconds')}))
        # A round that does not halve what is outside has its ridge halved for the next: a bound's rows are held with a stiffness of one over the
        # bound, and a step reaches a passed bound only as far as that outweighs the ridge. (Asked instead whether the rounds left would end
        # inside at the round's rate, 14 of 19 rounds at layer 0's feed-forward went by at 0.64 to 0.85 a round before the ridge had come down:
        # weight-research-perfomance's reading of choose-behind.jsonl.)
        now, before = rounds[-1]['outside'], record['outside_before']
        if now > 0 and lower() and now > before * 2.0 ** -PATH:
            ridge, inverse, halved_ridge = ridge * 2.0 ** -PATH, None, halved_ridge + 1
        torch.cuda.empty_cache()
    if beyond(readings) == 0:
        status = 'every bound of these answers is back inside'
    return dict(weights=weights, readings=readings, rounds=rounds, status=status, feasible=beyond(readings) == 0)


def fit(part, bounds, w, g, stands, ridge, effort=EFFORT, hold=(), watch=(), audit=None, basis=None, origin=None):
    """A matrix on the file's grid that moves this part's wrong pattern toward the ball, starting from the matrix as it stands (on the grid as well; w
    is the original, which a change is reported from).

    origin  the matrix the ridge counts from: the one the path of ridges started from (choose), which is the matrix as the earlier changes left it
            or as it was brought back. From the original itself the two are one. Without it the ridge counts from w.

    part    the answers solved on: the pull is theirs, and their bounds push back
    hold    answers whose bounds push back in the solve as well, without any pull of theirs
    watch   answers whose bounds every step must keep, read only
    audit   answers read after every step that is taken, and never used by a step. The solve ends with the first step after which they, or all the
            answers together, are outside a bound, once they have been inside: what comes after a first leaving is never reached.
    The solve starts inside every bound of the answers it is given and stays there: a matrix with a bound outside is brought back first
    (brought_back), and from one that is not this takes no step.
    Returns the last matrix reached, its readings, the record of every round, why the solve ended, and the audit's first and last readings."""
    solved = [part, *hold]
    watched = solved + list(watch)
    frames = part.frames(bounds)
    spread = float((g ** 2).mean())
    weights = stands
    readings = [each.realized(bounds, weights, g) for each in watched]
    rounds, status, inverse, measure = [], 'the rounds allowed were used', None, None
    first, last, been_inside, heard = None, None, False, None
    largest = lambda some: {family: [float(each['over'][family].max()) for each in some] for family in some[0]['over']}   # per family: the solved group, the others, then the read group

    def audited(weights, readings, number):
        """Reads the audited answers at a matrix. False at the first one where they or all the answers together are outside a bound after having been
        inside."""
        nonlocal first, last, been_inside, heard
        if audit is None:
            return True
        seen = heard = audit.realized(bounds, weights, g)
        together = pooled([*watched, audit], [*readings, seen], bounds)
        last = dict(round=number, tube=seen['tube'], largest_over_its_bound=worst(seen), all_together_largest_over_its_bound=worst(together), family=nearest(seen), all_together_family=nearest(together),
                    furthest=furthest(seen, bounds))
        if first is None:
            first = last
        if outside([seen, together, *readings]) > 0:
            return not been_inside
        been_inside = True
        return True

    # every bound inside on the answers solved on, the others and the audited ones, and on all of them together
    all_inside = lambda readings: outside(readings) == 0 and (audit is None or max(last['largest_over_its_bound'], last['all_together_largest_over_its_bound']) <= 1)
    audited(weights, readings, -1)
    settled = weights if all_inside(readings) else None   # the last matrix of this solve at which that held
    for number in range(effort['rounds']):
        if outside(readings) > 0:
            status = 'a bound of these answers is outside: it is brought back before anything pulls'
            break
        target, pull = bounds.target(readings[0]['mean'])
        if not bool(pull.any()):
            status = 'the wrong pattern is inside the ball at every place asked'
            break
        strength = readings[0]['tube'] / (2 * bounds.places * len(solved))   # every bounded place of the solve at its bound costs as much as the asks still unmet
        # How far the matrix stands from where the path started, for the ridge: a size inside the objective, never a way back to a matrix. Until
        # November 11 it was counted from the original. After a bringing back the ridge's least point was then the original, which is outside:
        # the pull's first step was the way back to it at every ridge (0.9989 to 1.0000 of the bringing back's change at twelve ridges in
        # records/development/2026-11-10-controlled-training/choose-behind.jsonl; weight-research-perfomance read it there), every such step was
        # cut at a bound, and layer 0's feed-forward kept no pull (layer 1's attention kept one that left 0.95491 of the tube loss where the
        # bringing back left 0.95492: position2-behind-solver.jsonl).
        total = weights - (w if origin is None else origin)
        stood = [Stand(each, weights, g) for each in solved]
        # per group, per merge read (its own, and behind a feed-forward's the layer's last): each bound's strength, which grows as its part nears it
        weight = [[{family: strength / (limit.clamp(min=TINY) * (1 - over[family]).clamp(min=RESOLVED)) for family, limit in view.bounds_of(frame).items()}
                   for view, frame, over in zip(each.views(), frames, (reading['over'], *([reading['behind']['over']] if 'behind' in reading else [])))]
                  for each, reading in zip(solved, readings)]

        def objective(p):
            value = 0.5 * ridge * ((total + p) ** 2).sum()
            for k, (each, stand) in enumerate(zip(solved, stood)):
                for m, (frame, moved, change) in enumerate(zip(frames, stand.moved, stand.moves(each.x @ p, each.xq @ p))):
                    mean, parts = each.measured(*(a + b for a, b in zip(moved, change)), frame)
                    if k == 0 and m == 0:   # the pull is at the matrix's own merge
                        value = value + 0.5 * (pull * ((mean - target) ** 2).sum(-1)).sum()
                    value = value + 0.5 * sum((weight[k][m][family] * parts[family]).sum() for family in weight[k][m])
            return value

        def slope(p):
            p = p.detach().requires_grad_(True)
            return torch.autograd.grad(objective(p), p)[0]

        if inverse is None:
            # what speeds the linear solve, built once from the first round's strengths (action_of)
            if basis is None or 'values' not in basis:
                reach = [each for stand in stood for each in stand.reaches()]
                found = action_of([each for each in solved for _ in frames], [a for a, _ in reach], [b for _, b in reach], [each for per in weight for each in per], w.shape[1])
                del reach
                if basis is not None:   # it waits off the card between the solves that use it: three of them at 12288 wide are 3.4 GiB
                    basis.update({name: value.cpu() if torch.is_tensor(value) else value for name, value in found.items()})
            else:
                found = basis
            vectors, values = found['vectors'].to(DEVICE), found['values'].to(DEVICE)
            inverse = (vectors / (values * spread + ridge)) @ vectors.T
            del vectors, values
            measure = measure_of([found], spread, ridge)   # what this solve's steps are written nearest in; it brings each part of the basis on by itself
            del found
        step, at_rest = torch.zeros_like(total), slope(torch.zeros_like(total))
        left, first_slope = -at_rest, at_rest.norm()
        lead = inverse @ left
        direction, inner, count = lead.clone(), (left * lead).sum(), 0
        for count in range(1, effort['iterations'] + 1):
            if left.norm() <= first_slope * RESOLVED:
                break
            bent = slope(direction) - at_rest
            rate = inner / (direction * bent).sum()
            step += rate * direction
            left -= rate * bent
            lead = inverse @ left
            inner, before = (left * lead).sum(), inner
            direction = lead + (inner / before) * direction

        if not bool(torch.isfinite(step).all()):
            rounds.append(dict(round=number, strength=strength, places_pulled=int(pull.sum()), iterations=count, taken=False, halvings=0))
            status = 'the linear solve gave no finite step'
            break

        def holds(share):
            stored = written(weights + share * step, measure)
            if bool((stored == weights).all()):
                return STILL
            trial = [each.realized(bounds, stored, g) for each in watched]
            # nothing may leave its bound, and the part taken must leave less tube loss
            return (trial, stored) if outside(trial) == 0 and trial[0]['tube'] < readings[0]['tube'] else None

        share, trial, halvings = halved(holds)
        record = dict(round=number, strength=strength, places_pulled=int(pull.sum()), iterations=count, slope_left=float(left.norm() / first_slope),
                      step_over_matrix=float(step.norm() / w.norm()), part_of_the_step_taken=share, halvings=halvings, tube_loss_left_before=readings[0]['tube'] / bounds.whole if bounds.whole else None)
        if trial is None or trial is STILL:
            rounds.append(dict(record, taken=False))
            status = 'the whole step moves no weight the file holds' if trial is STILL else 'no part of a step held every bound and left less tube loss'
            break
        readings, stored = trial
        # what the part of the step taken would read had every weight held it as it is, beside what the written matrix reads: what the writing cost
        meant = [each.realized(bounds, weights + share * step, g) for each in [*watched, *([audit] if audit is not None else [])]]
        # the solve stands on the matrix that was written and read, itself, and on nothing finer: what the next step adds, it adds to that
        weights = stored
        rounds.append(dict(record, taken=True, tube_loss_left=readings[0]['tube'] / bounds.whole if bounds.whole else None,
                           change_over_matrix=float((weights - w).norm() / w.norm()), largest_over_its_bound=max(worst(r) for r in readings)))
        inside = audited(weights, readings, number)
        rounds[-1].update(the_step_as_meant=dict(tube_loss_left=meant[0]['tube'] / bounds.whole if bounds.whole else None, largest_over_its_bound=largest(meant)),
                          the_step_as_written=dict(largest_over_its_bound=largest([*readings, *([heard] if audit is not None else [])])),
                          read_furthest=last['furthest'] if audit is not None else None)
        del meant
        if all_inside(readings):
            settled = weights
        if not inside:
            status = 'the audited answers left a bound'
            break
    return dict(weights=weights, last_inside=settled, readings=readings, rounds=rounds, status=status, feasible=outside(readings) == 0, audit_first=first, audit_last=last, audit_reading=heard,
                size_direction_share=None if basis is None else basis.get('size_direction_share'))


def choose(trace, bounds, w, g, stands, groups, unit, effort=EFFORT, held=False, say=None):
    """The matrix that is kept, chosen inside the fitting answers by the rule in this file's head, starting from the matrix as it stands. Both are
    matrices the file holds, and so is every one in between: what is read, compared and returned is always such a matrix itself. `groups` are the
    three groups of the fitting answers; w is the original matrix, which a change is counted from.

    Bringing back comes first. If, with the matrix as it stands, any group or all the fitting answers together are outside a bound (an earlier change put
    them there), they are brought back before anything pulls (brought_back): the three groups' bounds and the bounds of all of them together are held
    and nothing pulls. Its ridge starts at the path's floor (2^-11 of the unit, what a linear solve resolves) and is halved inside the bringing back,
    down to the smallest float32 tells from the unit (2^-23), after every round that did not halve what is outside (brought_back). At the floor
    alone the ridge outweighed the loose bounds that were passed at layer 0's feed-forward, whose stiffness is five orders under the unit's: a
    step reached them by a few percent a round, 22 rounds and 88 minutes. At 2^-23 alone it took 2 rounds and 53 seconds there with the layer's
    last merge held as well, but changed the matrix by 4.0%, and layer 1's attention by 11.7%. Halved as said, it ends at 2^-15 after 10 rounds
    and 268 seconds with 2.21% changed, and layer 1's attention at the floor after 3 rounds and 24 seconds with 0.97%
    (records/development/2026-11-10-controlled-training/README.md, the ridge of a bringing back). The pull that follows counts its ridge from
    the matrix its path starts from, which after a bringing back is the matrix as it was brought back (fit, origin). If that does not end
    with everything inside, nothing is changed and the matrix is reported as outside. Every fitting
    answer has then had a part in where the path starts: a group that is only read after it was not solved on, which is less than not used.

    Then three branches go along one path of ridges, from the matrix as it stands or as it was brought back: in each, one group is solved on (the
    pull is theirs, and their bounds push back), one must keep its bounds at every step, and one is only read after each step. With `held`, as a
    build has it, the second group's bounds push back in the solve as well, with no pull of theirs; without it they only refuse a step that leaves
    them. The first stage is at `unit`; if at that ridge a solve's first step has to be cut at a bound, or the stage takes a read group out of one, at the first ridge above
    it, doubling, where neither happens, as far up as the unit is still more of the ridge than a linear solve resolves. After a bringing back the
    first stage is at the first ridge, from the unit up, at which some branch ends inside having taken a step (the reason is at the code). Every later stage halves the
    ridge and starts each branch from where its own last stage ended. A branch stops with the first step after which its read group, or all its
    answers together, are outside a bound, and keeps the last matrix before that; the other branches go on. The path ends when every branch has
    stopped, when a stage takes no step, or where a halved ridge would be less of the unit than a linear solve resolves.

    What is kept is chosen among candidates (Astra's rule of November 10, ROUND-2.md). Each is a matrix the file holds, read as itself on each of
    the three groups and on all the fitting answers together: the matrix the path started from; every branch's end of every stage that ended
    inside; a stopped branch's last matrix inside; and for every stage whose three ends are all inside, their mean, written as a whole in the three
    solves' measures together at that stage's ridge. A candidate counts only if every bound holds in all four readings. Of those, the one that
    leaves all the fitting answers together the least tube loss is kept; on an exact tie the one whose worst group is left less, then the smaller
    change from where the path started, then the earlier one. Nothing is kept unless it leaves all the fitting answers together strictly less tube
    loss than where the path started. The three groups are not three tests of what is kept: every fitting answer has a part in choosing it.

    Returns the matrix kept, its record, its reading on all the fitting answers together, the measure it was written nearest in (None when no pull
    is kept), and the matrix the path started from: the matrix as it stood, or as it was brought back."""
    places = bounds.places
    if not bool((bf16(stands) == stands).all()):
        raise ValueError('the matrix to start from is not one the file holds')
    parts = [trace.part(each, places) for each in groups]
    left_of = lambda tube: tube / bounds.whole if bounds.whole else None
    floor = lambda power: 2.0 ** power < RESOLVED   # a ridge this small against the unit is less than a linear solve resolves
    ceiling = lambda power: 2.0 ** power > 1 / RESOLVED   # and against a ridge this large, the unit is
    everyone = trace.part(np.asarray(groups[0]) | np.asarray(groups[1]) | np.asarray(groups[2]), places)
    start = everyone.realized(bounds, stands, g)
    at_rest = [each.realized(bounds, stands, g) for each in parts]
    steady = outside(at_rest) == 0 and outside([pooled(parts, at_rest, bounds)]) == 0
    record = dict(before=dict(summary(start, bounds), holds_every_bound=outside([start]) == 0, every_group_and_all_together_inside=steady), ridge_power=None, ridge=None)
    base, power = stands, 0
    if not steady:
        first, least = 0, 0   # the ridges of a bringing back, as powers of two of the unit: from the path's floor down to the smallest float32 tells from the unit
        while not floor(first - PATH):
            first -= PATH
        while 2.0 ** (least - PATH) >= EPS:
            least -= PATH
        run = brought_back(parts, bounds, w, g, stands, unit * 2.0 ** first, effort, say, least=unit * 2.0 ** least)
        inside = run['feasible']
        base, at_rest = run['weights'], run['readings']
        together = pooled(parts, at_rest, bounds)
        lists = dict(halvings='halvings', ridge_halvings='ridge_halvings', passes='passes', iterations='iterations', seconds='seconds', outside_before_each_round='outside_before', each_rounds_passes='each_pass',
                     parts_of_the_steps='part_of_the_step_taken', steps_over_matrix='step_over_matrix')
        back = [dict(ridge_power=first, least_ridge_power=least, ridge_power_at_the_end=first - PATH * (run['rounds'][-1]['ridge_halvings'] if run['rounds'] else 0), every_bound_inside=inside, status=run['status'], rounds=len(run['rounds']), rounds_taken=sum(r['taken'] for r in run['rounds']),
                     **{name: [r.get(key) for r in run['rounds']] for name, key in lists.items()}, outside_left=outside([*at_rest, together]),
                     largest_over_its_bound=max(worst(r) for r in at_rest), all_together_largest_over_its_bound=worst(together),
                     parts_left_outside=[dict(each, read=name) for name, reading in zip([*range(len(at_rest)), 'all together'], [*at_rest, together]) for each in left_outside(reading, bounds)],
                     change_over_matrix=float((base - w).norm() / w.norm()))]
        if say:
            say(dict(bringing_back=True, **{name: value for name, value in back[-1].items() if name not in lists}))
        del run
        torch.cuda.empty_cache()
        record.update(brought_back=dict(stages=back, every_bound_inside=inside))
        if not inside:
            return stands, dict(record, selected='no change: a bound is outside and no feasible restoration was found under this search', status='outside'), start, None, stands
    stood = base
    bases = [dict() for _ in ROTATIONS]
    spread = float((g ** 2).mean())

    def stage(power, starts):
        """The solves at one ridge, one for each branch that is still going, each from its own start (None: a branch that has stopped)."""
        solves = []
        for k, (a, b, c) in enumerate(ROTATIONS):
            if starts[k] is None:
                solves.append(None)
                continue
            run = fit(parts[a], bounds, w, g, starts[k], unit * 2.0 ** power, effort, hold=[parts[b]] if held else (), watch=() if held else [parts[b]], audit=parts[c], basis=bases[k], origin=base)
            last = run['audit_last']
            inside = run['feasible'] and max(last['largest_over_its_bound'], last['all_together_largest_over_its_bound']) <= 1
            taken = [r for r in run['rounds'] if r['taken']]
            solves.append(dict(moved=run['weights'] is not starts[k], last_moved=run['last_inside'] is not None and run['last_inside'] is not starts[k],
                               weights=run['weights'], last_inside=run['last_inside'], left=last['tube'], inside=inside, taken=len(taken),
                               record=dict(solved_on=a, kept_inside=b, read=c, status=run['status'], rounds=len(run['rounds']), rounds_taken=len(taken),
                                           iterations=[r['iterations'] for r in run['rounds']], slope_left=[r.get('slope_left') for r in run['rounds']],
                                           halvings=[r['halvings'] for r in run['rounds']],
                                           parts_of_the_steps_taken=[r.get('part_of_the_step_taken') for r in run['rounds']],
                                           solved_tube_loss_left=left_of(run['readings'][0]['tube']), read_tube_loss_left=left_of(last['tube']), every_bound_inside=inside,
                                           largest_over_its_bound=dict(solved=worst(run['readings'][0]), others=max([worst(r) for r in run['readings'][1:]], default=None),
                                                                       read=last['largest_over_its_bound'], all_together=last['all_together_largest_over_its_bound']),
                                           family_nearest_its_bound=dict(solved=nearest(run['readings'][0]), read=last['family'], all_together=last['all_together_family']),
                                           furthest_along_its_bound=dict(solved=furthest(run['readings'][0], bounds), read=last['furthest']),
                                           read_groups_furthest_size_cells=top_cells(run['audit_reading'], bounds, 'size'), size_direction_share=run['size_direction_share'],
                                           steps_taken=[{name: r[name] for name in ('tube_loss_left', 'part_of_the_step_taken', 'step_over_matrix', 'the_step_as_meant', 'the_step_as_written', 'read_furthest')}
                                                        for r in run['rounds'] if r['taken']])))
        ran = [each for each in solves if each is not None]
        found = dict(power=power, solves=solves, inside=all(each['inside'] for each in ran), taken=sum(each['taken'] for each in ran))
        if say:
            say(dict(ridge_power=power, every_bound_inside=[None if each is None else each['inside'] for each in solves], read_groups_tube_loss_left=[None if each is None else left_of(each['left']) for each in solves],
                     rounds_taken=[None if each is None else each['taken'] for each in solves], status=[None if each is None else each['record']['status'] for each in solves]))
        return found

    def aside(found):
        """An earlier stage's matrices wait off the card until the candidates are read: the card does not hold a whole path of the widest matrices beside
        a solve."""
        for solve in found['solves']:
            if solve is not None:
                solve.update(weights=solve['weights'].cpu(), last_inside=None if solve['last_inside'] is None else solve['last_inside'].cpu())

    path, passed = [], []
    # a first stage is one in which every solve's first step is taken whole and which ends inside: where a first step has to be cut at a bound, or a read
    # group is taken out, the ridge is already past where the change starts gently, and the start is looked for above it, again from where the path starts
    gentle = lambda found: found['inside'] and all(solve['record']['halvings'][:1] in ([], [0]) for solve in found['solves'])
    # After a bringing back the parts lie next to their bounds, where the bringing back left them, and a start that is gentle for every branch is
    # not there to find: controlled-bf16-3 passed over 10, 11, 11, 11, 8, 11 and 11 starts of the 11 there are at its seven matrices brought back,
    # and what it kept from the stages up there took at most 3.4e-5 of the tube loss. Under the mean and the rows apart, at layer 0's feed-forward,
    # two branches' read groups left a bound at every start up to the last, the third branch's stages from 2^11 down to 2^1 took 0.0002, and its
    # stages from the unit down to 2^-4 took 0.018; the stages above the unit were 651 of the choice's 945 seconds
    # (records/development/2026-11-10-controlled-training/README.md, the split's choice). So there the path starts at the first ridge, from the
    # unit up, at which some branch ends inside having taken a step; a branch whose read group leaves a bound in that stage has stopped.
    moving = lambda found: any(solve['inside'] and solve['taken'] for solve in found['solves'])
    begun = gentle if steady else moving
    first = stage(power, [base] * 3)
    while not begun(first) and first['taken'] and not ceiling(power + PATH):
        passed.append(first)
        aside(first)
        power += PATH
        first = stage(power, [base] * 3)
    path.append(first)
    # a branch goes on from its own end while that end is inside; one that left a bound has stopped where it was last inside, and the others go on
    going = lambda found: [solve['weights'] if solve is not None and solve['inside'] else None for solve in found['solves']]
    while any(each is not None for each in going(path[-1])) and path[-1]['taken'] and not floor(power - PATH):
        power -= PATH
        path.append(stage(power, going(path[-1])))
        aside(path[-2])
    row = lambda each: dict(ridge_power=each['power'], ridge=unit * 2.0 ** each['power'], every_bound_inside=each['inside'],
                            read_groups_tube_loss_left=[None if solve is None else left_of(solve['left']) for solve in each['solves']],
                            solves=[None if solve is None else solve['record'] for solve in each['solves']])
    record.update(starts_passed_over=[row(each) for each in passed], stages=[row(each) for each in path],
                  ended='every branch left a bound' if not any(each is not None for each in going(path[-1])) else 'no solve took a step' if not path[-1]['taken']
                        else 'the ridge is less of the unit than a linear solve resolves')

    def measure_at(power, branches):   # what a matrix of these branches at this ridge is written nearest in
        known = [bases[k] for k in branches if 'values' in bases[k]]
        return measure_of(known, spread, unit * 2.0 ** power) if known else None

    # the candidates: where the path started, every branch's end of every stage that ended inside, a stopped branch's last matrix inside, and the
    # mean of a stage's three ends where all three are inside, written as a whole
    candidates = [dict(name='the matrix the path started from', power=None, branches=(), weights=stood)]
    for found in path:
        ends = []
        for k, solve in enumerate(found['solves']):
            if solve is None:
                continue
            c = ROTATIONS[k][2]
            if solve['inside']:
                ends.append(solve['weights'])
                if solve['moved']:
                    candidates.append(dict(name=f'the end of the branch that read group {c}', power=found['power'], branches=(k,), weights=solve['weights']))
            elif solve['last_moved']:
                candidates.append(dict(name=f'the last matrix inside of the branch that read group {c}, before it left a bound', power=found['power'], branches=(k,), weights=solve['last_inside']))
        if len(ends) == len(ROTATIONS):
            measure = measure_at(found['power'], range(len(ROTATIONS)))
            candidates.append(dict(name='the mean of the stage\'s three ends', power=found['power'], branches=tuple(range(len(ROTATIONS))),
                                   weights=written(sum(each.to(DEVICE).double() for each in ends).div(len(ends)).float(), measure).cpu(), alike=None if measure is None else measure['size_directions_alike']))
            del measure
            torch.cuda.empty_cache()
    for order, each in enumerate(candidates):   # each as the matrix itself, on all the fitting answers together and on each group
        matrix = each['weights'].to(DEVICE)
        cohort, thirds = everyone.realized(bounds, matrix, g), [third.realized(bounds, matrix, g) for third in parts]
        each.update(order=order, reading=cohort, holds=outside([cohort, *thirds]) == 0, worst_group=max(third['tube'] for third in thirds), largest_in_a_group=max(worst(third) for third in thirds),
                    size=float((matrix - stood).norm()))
        del matrix
    record.update(candidates=[dict(candidate=each['name'], ridge_power=each['power'], holds_every_bound=each['holds'], tube_loss_left=left_of(each['reading']['tube']),
                                   worst_groups_tube_loss_left=left_of(each['worst_group']), largest_over_its_bound=worst(each['reading']), largest_over_its_bound_in_a_group=each['largest_in_a_group'],
                                   change_from_the_start_over_matrix=each['size'] / float(w.norm()),
                                   **({'the_three_solves_size_directions_alike_at_least': each['alike']} if each.get('alike') is not None else {})) for each in candidates])
    best = min([each for each in candidates if each['holds']], key=lambda each: (each['reading']['tube'], each['worst_group'], each['size'], each['order']), default=None)

    def only_back(why):   # nothing beyond the bringing back, or nothing at all
        if steady:
            return stands, dict(record, selected=f'no change: {why}', status='unchanged'), start, None, stands
        return base, dict(record, selected=f'the bringing back alone: {why}', status='changed', after=summary(candidates[0]['reading'], bounds)), candidates[0]['reading'], None, base

    if best is None or best['power'] is None or not best['reading']['tube'] < candidates[0]['reading']['tube']:
        return only_back('no candidate that holds every bound on every group and on all the fitting answers together leaves them less tube loss than where the path started')
    record.update(ridge_power=best['power'], ridge=unit * 2.0 ** best['power'])
    measure = measure_at(best['power'], best['branches'])   # goes with the matrix: what a part of it taken back would be written nearest in
    del bases
    torch.cuda.empty_cache()
    return best['weights'].to(DEVICE), dict(record, selected=best['name'], status='changed', after=summary(best['reading'], bounds)), best['reading'], measure, stood

MERGES = (*KINDS, 'layer')   # the three merges of a layer in path order: after attention, after the feed-forward, and the layer's last, where the per-layer embedding is added and which no matrix of ours writes


def picks(members):
    """The fitting answers (every other answer of each original group, in the answers' own order), and their three groups (every third of those, within each
    original group again). The other answers are never used by a fit."""
    correct = np.array([m['correct'] for m in members])
    every = lambda among, k, n: np.isin(np.arange(len(members)), np.concatenate([np.flatnonzero(among & (correct == flag))[k::n] for flag in (True, False)]))
    fitting = every(np.ones(len(members), bool), 0, 2)
    return fitting, [every(fitting, k, 3) for k in range(3)]


def native_log(members, layer, kind):
    """The log of the size of the state after a merge at every answer step, as the engine itself recorded it while the original model generated."""
    merge = 3 * layer + MERGES.index(kind)
    return torch.from_numpy(np.concatenate([np.log(np.load(NATIVE/m['id']/'norms.npy', mmap_mode='r')[:m['steps'], merge, 1]) for m in members])[:, None]).to(DEVICE)


def prompt_places(members):
    """Which place of the prompt each question position of each answer lies at. The places are the positions of the prompts of the most common length, and
    a prompt of that length lies on them one to one. A prompt of another length is laid on the first prompt of the common length by its tokens, without
    regard to any answer: runs of equal tokens one to one, and two runs that differ spread over each other in order, so that every position has a place
    and every place a position. Returns the number of places, per answer its [position, place] rows, and what the layout rests on.

    That prompts of one length lie on each other one to one is not shown by their length. It holds for these answers because they are one template:
    the record counts the positions at which every prompt of the common length holds the same token, and names the prompt the others are laid on. On
    other questions that count has to be looked at before this layout is used."""
    lengths = [m['question'] for m in members]
    common = max(sorted(set(lengths)), key=lengths.count)
    first = next(m for m in members if m['question'] == common)
    template = first['tokens'][:common]
    alike = np.array([m['tokens'][:common] for m in members if m['question'] == common])
    told = dict(laid_on=first['id'], prompts_of_the_common_length=int(len(alike)), positions_holding_one_token_in_all_of_them=int((alike == alike[0]).all(0).sum()))
    laid = []
    for m in members:
        if m['question'] == common:
            laid.append(np.stack([np.arange(common)] * 2, 1))
            continue
        rows = []
        for _, a0, a1, b0, b1 in difflib.SequenceMatcher(None, template, m['tokens'][:m['question']], autojunk=False).get_opcodes():
            a, b = a1 - a0, b1 - b0
            if not a or not b:
                raise ValueError(f"{m['id']}: a run of its prompt has no counterpart in the common prompt")
            rows += [(b0 + k * b // max(a, b), a0 + k * a // max(a, b)) for k in range(max(a, b))]
        laid.append(np.array(sorted(set(rows))))
    return common, laid, told


def frozen(members, pairs, places, to, log, asking, asking_log, given=None, given_log=None, native=None, prompt=None, split=False):
    """What the original model fixes at one merge, from all its answers.

    to, log   the unit state after the merge and the log of its size at every answer step, as recorded while the model generated the answers;
              asking, asking_log the same at every question position
    given     the same rows (answer steps, then question positions) as the engine itself put them out in a pass of the answers' own text through the
              original file, read the way the run declares; given_log the log of the size there. Without them no bound is raised to a reading.
    native    for a matrix's merge, the log of the size the engine itself recorded there while generating (native_log)
    prompt    the unit state and the log of its size at every question position as the engine itself put them out; without it, asking and asking_log
    The gap and its chance are the state overlap's. A group's bound at a place is the variance of its members' means there over the number of members, for
    the direction and for the log of the size (the engine's own, where given). A question's bounds are taken the same way from the prompt: at every place
    of the prompt, the variance of its group's questions there over the number of members; over the question's positions, the mean of that at each
    position's places. A bound is never less than what the readings themselves resolve, each measured as the bound is, squares first: how far the pass's
    reading of the original file lies from the recorded rows, and, for the size, how far this file's expression of the merge lies from the size the engine
    recorded. The groups are the original ones, and every count is the original number of members.

    split     the mean and the rows are held apart (ALLOWANCES). With N a group's original number of members, V the variance of its members' means at
              a place and A the mean, over its members, of each member's mean squared distance of its steps there from the member's own mean: the
              group's mean movement there is bounded by V / N (direction_mean, size_mean), which is the bound above, and its rows' average
              displacement by T = (N - 1) V / N + A (direction, size), which is the original rows' own mean squared distance from their group's
              mean there. Each of the two is never less than what the same readings resolve in its own form: the square of the group's mean of the
              difference for the first, the group's mean of the squares for the second, each measured by itself and neither made from the other.
              The questions' bounds are as above. What each of the four limits was made of is given beside them under names that begin raw_ (the
              variance over the members, the members' own spread, and what each reading resolves in each form): no bound, solve or verdict reads
              those; they tell a limit set by the original answers' sampling from one set by a floor (Astra, November 11,
              records/retrospectives/2026-11-11-opening-cells-and-split-astra.md)."""
    correct = np.array([m['correct'] for m in members])
    steps = sum(m['steps'] for m in members)
    layout = Layout(~correct, [m['steps'] for m in members], [m['question'] for m in members], pairs, places)
    right, wrong = torch.from_numpy(correct).to(DEVICE), torch.from_numpy(~correct).to(DEVICE)
    z, sizes = layout.means(to), layout.means(log if native is None else native)[..., 0]
    gap = z[right].mean(0) - z[wrong].mean(0)
    natural = torch.stack([z[mask].var(0, unbiased=True).sum(-1) / int(mask.sum()) for mask in (wrong, right)])
    size = torch.stack([sizes[mask].var(0, unbiased=True) / int(mask.sum()) for mask in (wrong, right)]).float()
    if split:   # T: what is left of the variance of the members' means with the mean's own share taken off, and every member's steps about its own mean
        less = torch.tensor([[int(mask.sum()) - 1] for mask in (wrong, right)], device=DEVICE, dtype=torch.float64)
        raw = dict(direction_variance_over_members=natural, direction_within_members=layout.spreads(to, z),
                   size_variance_over_members=torch.stack([sizes[mask].var(0, unbiased=True) / int(mask.sum()) for mask in (wrong, right)]),
                   size_within_members=layout.spreads(log if native is None else native, sizes[..., None]))
        rows = natural * less + raw['direction_within_members']
        size_mean = size
        size = (raw['size_variance_over_members'] * less + raw['size_within_members']).float()
    del z, sizes
    record = dict(size_bounds_raised_by_the_expression=None)
    if native is not None:
        expressed = layout.squares((log.double() - native).float())
        record.update(size_bounds_raised_by_the_expression=int((expressed > size).sum()))
        size = torch.maximum(size, expressed)
        if split:   # the same difference as the mean's bound measures it: the square of the group's mean of it
            off = layout.means((log.double() - native).float())[..., 0]
            expressed_mean = (torch.stack([off[wrong].mean(0), off[right].mean(0)]) ** 2).float()
            record.update(size_mean_bounds_raised_by_the_expression=int((expressed_mean > size_mean).sum()))
            size_mean = torch.maximum(size_mean, expressed_mean)
            raw.update(size_expression_rows=expressed, size_expression_mean=expressed_mean)
    # the questions: each one's state at every place of the prompt, then each group's variance there over its members
    common, laid, told = prompt_places(members)
    begins = np.cumsum([0] + [m['question'] for m in members])
    row = torch.from_numpy(np.concatenate([begins[i] + each[:, 0] for i, each in enumerate(laid)])).to(DEVICE)
    cell = torch.from_numpy(np.concatenate([i * common + each[:, 1] for i, each in enumerate(laid)])).to(DEVICE)
    share = (1 / torch.zeros(len(members) * common, device=DEVICE, dtype=torch.float64).index_add_(0, cell, torch.ones(len(cell), device=DEVICE, dtype=torch.float64)))[cell]
    question = {}
    for family, values in zip(('question', 'question_size'), (asking, asking_log) if prompt is None else prompt):
        at = torch.zeros((len(members) * common, values.shape[1]), device=DEVICE, dtype=torch.float64).index_add_(0, cell, values[row].double() * share[:, None]).view(len(members), common, -1)
        spread = torch.stack([at[mask].var(0, unbiased=True).sum(-1) / int(mask.sum()) for mask in (wrong, right)]).cpu().numpy()   # 2 x places of the prompt
        per = []
        for i, each in enumerate(laid):   # over the question's positions, the mean of its group's spread at each position's places
            weight = 1 / np.bincount(each[:, 0], minlength=members[i]['question'])[each[:, 0]]
            per.append(float((spread[int(correct[i]), each[:, 1]] * weight).sum() / members[i]['question']))
        question[family] = torch.tensor(per, device=DEVICE, dtype=torch.float32)
        del at
    first = dict(direction=rows, size=size, direction_mean=natural, size_mean=size_mean) if split else dict(direction=natural, size=size)   # before any reading raises one
    draft = Bounds(gap, natural.sum(0), **first)
    record.update(places_asked=int(draft.asked.sum()), places_of_the_prompt=common, prompt_layout=told,
                  prompts_laid_by_their_tokens={m['id']: each.tolist() for m, each in zip(members, laid) if m['question'] != common})
    if given is None:
        record.update(bounds_raised_to_a_reading=None)
        return dict(gap=gap.float(), chance=natural.sum(0).float(), **{family: value.float() for family, value in first.items()}, **question,
                    **({f'raw_{name}': value.float() for name, value in raw.items()} if split else {})), record
    grown = torch.zeros((steps, 1), device=DEVICE) if given_log is None else given_log[:steps] - log
    grown_q = torch.zeros_like(asking_log) if given_log is None else given_log[steps:] - asking_log
    _, apart = layout.measured(given[:steps] - to, grown, given[steps:] - asking, grown_q, draft)
    record.update(bounds_raised_to_a_reading=dict(direction=int((apart['direction'] > first['direction']).sum()), size=None if given_log is None else int((apart['size'] > size).sum()),
                                                  question=int((apart['question'] > question['question']).sum()), question_size=None if given_log is None else int((apart['question_size'] > question['question_size']).sum()),
                                                  **({'direction_mean': int((apart['direction_mean'] > natural).sum()), 'size_mean': None if given_log is None else int((apart['size_mean'] > size_mean).sum())} if split else {})),
                  reading_apart_rms={family: float(value.double().mean().sqrt()) for family, value in apart.items()})
    if split:
        raw.update({f'{family}_reading_{form}': apart[family if form == 'rows' else f'{family}_mean'] for family in ('direction', 'size') for form in ('rows', 'mean')})
    return dict(gap=gap.float(), chance=natural.sum(0).float(), **{family: torch.maximum(value.float(), apart[family]) for family, value in first.items()},
                **{family: torch.maximum(value, apart[family]) for family, value in question.items()}, **({f'raw_{name}': value.float() for name, value in raw.items()} if split else {})), record


def unit_of(part, bounds, w, g):
    """The ridge that the ladder of ridges is counted from: the mean, over the directions of a change, of the bounds' own push-back on these rows before
    anything has moved (each row's input, weighted by its bound's strength and by how much of an output change reaches its state)."""
    per_row = torch.zeros(part.x.shape[0], device=DEVICE).index_add_(0, part.at, (bounds.unit / bounds.direction)[part.group, part.place] / part.members * part.weight)
    ratio = Merge(part.x, part.h, w, g).ratio[:, 0]
    reach = ratio ** 2
    total = float((per_row.double() * reach.double() * (part.x.double() ** 2).sum(1)).sum())
    if bounds.split:   # and the push-back of the bounds on each group's mean at a place: each one's strength on its one row, the mean of its steps' inputs
        total += float(((bounds.unit / bounds.direction_mean).flatten().double() * (mean_inputs(part, part.weight / part.members * ratio[part.at]).double() ** 2).sum(1)).sum())
    return total / part.x.shape[1] * float((g ** 2).mean())


def recorded(members, index, layer, kind, w=None, g=None, tensors=None):
    """The unit state after a merge and the log of its size at every answer step, and the same at the question's positions, from what was recorded while
    the original model generated.

    A matrix's merge: rebuilt from the matrix's recorded input, as the engine multiplied it (`multiplied`: rounded to the file's BF16 at the question's
    positions and at each answer's first step, which the engine read at once while it generated), and the state it adds to, at every row. That is the
    form the state overlap's patterns are in, and it is the engine's own output there to the last digits of float32, which is what a pass's reading is
    compared with. Until November 11 the input was taken as recorded at every row; controlled-bf16-1's reference and the first state overlap are in that
    form and stay as they are (Astra, records/retrospectives/2026-11-11-native-reference-decision-astra.md).
    A layer's last merge (kind 'layer'), where the per-layer embedding is added: the engine's own. Its direction is the direction of the layer's output,
    which is that state times one number and is on record as the state the next layer's attention adds to; its size at the answer steps is the size the
    engine recorded, and at the question's positions the size of the layer's output over that number. The last layer has no layer after it: its
    direction at the answer steps is the recorded direction after attention plus the two recorded turns that follow it, and nothing is on record at its
    question's positions (None)."""
    steps = sum(m['steps'] for m in members)
    if kind == 'layer':
        log = native_log(members, layer, kind).float()
        if layer != index['layers'][-1]:
            rows = [np.fromfile(ladder.INPUTS/f'{prefix}state-attention-{layer + 1}.f32', dtype='<f4').reshape(-1, 1536) for prefix in ('', 'question-')]
            scale = checkpoint_values(ORIGINAL_MODEL, tensors[f'blk.{layer}.layer_output_scale.weight']).astype('f8').ravel()
            if scale.size != 1 or not scale[0] > 0:
                raise ValueError(f'the output of layer {layer} is not its last merge times one positive number')
            asking, asking_log = unit_and_log(torch.from_numpy(rows[1]).to(DEVICE))
            return unit_and_log(torch.from_numpy(rows[0]).to(DEVICE))[0], log, asking, asking_log - float(np.log(scale[0]))
        return turned_to(members, layer), log, None, None
    x, h = (torch.from_numpy(np.ascontiguousarray(v)).to(DEVICE) for v in ladder.stored(members, index, layer, kind))
    merge = Merge(multiplied(x, [m['steps'] for m in members], 'as generated'), h, w, g)
    return merge.to[:steps], merge.size.log()[:steps], merge.to[steps:], merge.size.log()[steps:]


def turned_to(members, layer):
    """The direction after a layer's last merge at every answer step, from the recorded direction after its attention's merge and the two recorded turns
    that follow it (each turn is the direction after a merge minus the direction before it)."""
    after = torch.from_numpy(np.fromfile(ladder.INPUTS/f'state-feedforward-{layer}.f32', dtype='<f4').reshape(-1, 1536)).to(DEVICE)
    turns = np.concatenate([np.load(NATIVE/m['id']/'turns.npy', mmap_mode='r')[:m['steps'], 3 * layer + 1:3 * layer + 3].sum(1) for m in members])
    return unit_and_log(after)[0] + torch.from_numpy(turns).to(DEVICE)


def kept(folder, kind, layer, to=None, log=None, rebuilt=False, again=False):
    """A merge's unit state and the log of its size at every row, written or read back: the engine's own output there, or, with `rebuilt`, the merge
    rebuilt from the matrix's input and the state it adds to, or, with `again`, the engine's own output at the question's positions alone in a second
    reading of the prompts."""
    paths = [folder/f"{'rebuilt-' if rebuilt else 'again-' if again else ''}{what}-{kind}-{layer}.f32" for what in ('state', 'size')]
    if to is not None:
        to.cpu().numpy().astype('<f4').tofile(paths[0])
        log.cpu().numpy().astype('<f4').tofile(paths[1])
        return None
    return np.memmap(paths[0], dtype='<f4', mode='r').reshape(-1, 1536), np.memmap(paths[1], dtype='<f4', mode='r').reshape(-1, 1)


def same_program(installed, wanted):
    """Whether the installed recorder is the declared program in every byte outside the name of its library folder: the second, exact identity
    a run's passes_read_with is held to (Performance, November 13, 2026: pipeline/acquisition/recorder_identity.py). The recorder is linked
    with the folder its engine libraries are looked for in written into the file; a copy that runs on another machine has that one string
    rewritten in place, so its sha256 is another although the program, and the stop that was compared byte for byte in it, are the same. It
    holds when the record kept beside the recorder (observe-stream.identity.json, made from the declared binary and committed) names the
    declared sha256 and the installed file's digest with that string zeroed is the record's. A record that is missing or of another binary, a
    file the string cannot be found in exactly, or a machine without that module is no identity. The libraries the program loads from that
    folder were never part of this check: a capture names them in its own provenance."""
    try:
        from pipeline.acquisition.recorder_identity import without_library_folder
        told = json.loads(Path(str(installed) + '.identity.json').read_text())
        return told['sha256'] == wanted and without_library_folder(installed) == told['sha256_without_library_folder']
    except (ImportError, OSError, ValueError, KeyError):
        return False


def declared(run):
    """Sets, from the run's declaration, how its passes read an answer's text, and returns the declaration. A run declared on a converted copy of the
    model is refused: no command here works on anything but the file's own BF16."""
    global READING, STOPS
    declaration = json.loads((run/'declaration.json').read_text())
    if declaration.get('matrices_kept_as', 'bf16') != 'bf16':
        raise ValueError(f"{run.name} declares its matrices kept as {declaration['matrices_kept_as']!r}: the model is kept in the file's own BF16 and no command works on a converted copy")
    READING = declaration['passes_read']
    if READING not in ('given text', 'as generated'):
        raise ValueError(f'passes_read is {READING!r}')
    STOPS = bool(declaration.get('stop_after_last_label'))
    if STOPS and READING != 'as generated':
        raise ValueError('stop_after_last_label is for passes read as generated: the recorder stops only where an answer\'s own tokens are supplied')
    if STOPS:
        installed, wanted = ROOT/'code/signal-trace/recorder/observe-stream', (declaration.get('recorder_revision') or {}).get('passes_read_with')
        if digest(installed) != wanted and not same_program(installed, wanted):
            raise ValueError('The installed recorder is not the one this run declares for its passes (recorder_revision, passes_read_with), neither byte for byte nor in every byte outside '
                             'the name of its library folder (observe-stream.identity.json beside it): the stop was compared byte for byte for that program only')
    allowance_of(declaration)
    if (anchor(run)/'reference'/'index.json').exists():   # once the reference it is counted from is made, the run's allowance has to be that reference's
        allowance_held(run)
    return declaration


def passed(model, run, name, wanted, members, started=None, each=None, keep=False):
    """The answers' own text through the model, read the way the run declares. Returns each wanted output at every answer step, then at every question
    position. The decision for generated token s is computed at position question + s, whose input is generated token s - 1: either way of reading puts the
    same tokens at the same positions, and the last recorded token is never an input.
    Three things a sweep's passes ask (gap_late), none of which changes a pass that does not ask: with `started`, a recorder's name and every answer's
    seeds, that recorder runs the pass and each case names its seeds (a pass started late: code/signal-trace/recorder/STREAM-RECORDER.md); `each(answer,
    label, rows)` is given every wanted output of every answer in position order as it is read; with `keep` the raw streams are left to the caller."""
    from pipeline.acquisition.run_a_observation import run as observe
    raw, cases = run/'passes'/name, run/'passes'/f'{name}-cases.json'
    cases.parent.mkdir(parents=True, exist_ok=True)
    text = (lambda m: dict(tokens=m['tokens'], teacher_tokens=m['recorded'])) if READING == 'as generated' else (lambda m: dict(tokens=m['tokens'] + m['recorded'][:-1], max_tokens=1))
    late = (lambda m: dict(late_start=dict(seeds=started[1][m['id']]))) if started else (lambda m: {})
    cases.write_text(json.dumps([dict(id=m['id'], **text(m), stream_labels='|'.join(wanted), **(dict(stop_after_last_label=True) if STOPS else {}), **late(m)) for m in members]))
    with CARD.given():
        observe(model, cases, raw, 0, 'conditional', 'capture', 'projection', run/'declaration.json', program=started[0] if started else None)
        answer, question = {label: [] for label in wanted}, {label: [] for label in wanted}
        for m in members:
            count = m['question'] + m['steps']
            stream = StreamRows(raw/m['id'], set(range(count)))
            for label in wanted:
                rows = np.stack([stream.values(label, p) for p in range(count)])
                if each is not None:
                    each(m, label, rows)
                question[label].append(rows[:m['question']])
                answer[label].append(rows[m['question']:])
            stream.close()
            if not keep:
                shutil.rmtree(raw/m['id'])   # the rows are in memory; the raw stream is not kept
        cases.unlink()
        return [np.concatenate(answer[label] + question[label]) for label in wanted]


def asked_again(model, run, name, label, members):
    """The prompts alone through a model, nothing supplied after them: one output of the engine at every question position, in the answers' order. The
    prompt is read exactly as every pass and every answer of the model reads it."""
    from pipeline.acquisition.run_a_observation import run as observe
    raw, cases = run/'passes'/name, run/'passes'/f'{name}-cases.json'
    cases.parent.mkdir(parents=True, exist_ok=True)
    cases.write_text(json.dumps([dict(id=m['id'], tokens=m['tokens'], max_tokens=1, stream_labels=label) for m in members]))
    with CARD.given():
        observe(model, cases, raw, 0, 'natural', 'capture', 'projection', run/'declaration.json')
        rows = []
        for m in members:
            stream = StreamRows(raw/m['id'], set(range(m['question'])))
            rows.append(np.stack([stream.values(label, p) for p in range(m['question'])]))
            stream.close()
            shutil.rmtree(raw/m['id'])
        cases.unlink()
        return np.concatenate(rows)


def start(run):
    """The run's own model file, before any change: a copy of the original file, or of the finished model of the sweep the run starts from.
    model/bindings.json says where each matrix lies in it."""
    declaration = declared(run)
    _, _, index = ladder.cohort(run)
    tensors = json.loads(ladder.TENSORS.read_text())
    names = [f'blk.{layer}.{KINDS[kind][1]}.weight' for _, layer, kind in matrices(index['layers'])]
    partial, earlier = model_of(run).with_suffix('.partial'), declaration.get('starts_from')
    if partial.exists() or model_of(run).exists():
        raise ValueError('The run already has a model')
    partial.parent.mkdir(parents=True, exist_ok=True)
    if earlier:
        declared(ROUND_TWO_RUNS/earlier)
        shutil.copyfile(model_of(ROUND_TWO_RUNS/earlier), partial)
        where = json.loads((ROUND_TWO_RUNS/earlier/'model'/'bindings.json').read_text())
    else:
        shutil.copyfile(ORIGINAL_MODEL, partial)
        where = {name: tensors[name] for name in names}
    save(run/'model'/'bindings.json', where)
    print(json.dumps(dict(model=partial.name, matrices=len(where), bytes=partial.stat().st_size, starts_from=earlier)), flush=True)


def unit_and_log(state):
    """A state's direction and the log of its size, row by row."""
    size = state.norm(dim=1, keepdim=True)
    return state / size, size.log()


OUTPUT = dict(attention=KINDS['attention'][5], feedforward=KINDS['feedforward'][5], layer='pe_out')   # the engine's name for what each merge puts out


def passes(model, run, name, members, layers, weights):
    """The answers' own text through a model, read the way the run declares, five layers a pass (which only bounds the raw stream on disk). For each of the
    three merges of each of those layers: the engine's own output there (the unit state and the log of its size at every answer step, then every question
    position), and, for a matrix's merge, the same merge rebuilt from the matrix's input, as the engine multiplies it, and the state it adds to, which is
    how a solve reads it and the form the reference's rows are in.
    `weights(layer, kind)` gives the matrix the model holds there and its learned scale."""
    for first in range(0, len(layers), 5):
        group = layers[first:first + 5]
        wanted = list(dict.fromkeys(label for layer in group for kind in MERGES for label in (*(ladder.labels_of(layer, kind) if kind in KINDS else ()), f'{OUTPUT[kind]}-{layer}')))
        started = time.time()
        rows = dict(zip(wanted, passed(model, run, f'{name}-{group[0]}', wanted, members)))
        seconds = round(time.time() - started, 1)
        on_card = lambda label: torch.from_numpy(np.ascontiguousarray(rows[label])).to(DEVICE)
        for layer in group:
            for kind in MERGES:
                rebuilt = None
                if kind in KINDS:
                    w, g = weights(layer, kind)
                    x, h = (on_card(label) for label in ladder.labels_of(layer, kind))
                    merge = Merge(multiplied(x, [m['steps'] for m in members]), h, w, g)
                    rebuilt = merge.to, merge.size.log()
                    del merge, w, g, x, h
                yield layer, kind, *unit_and_log(on_card(f'{OUTPUT[kind]}-{layer}')), rebuilt, seconds
                del rebuilt
                torch.cuda.empty_cache()
        del rows


def anchor(run):
    """The run whose reference a run is counted from: itself, or, for a sweep that starts from an earlier sweep's model (starts_from), the first sweep
    of that chain. The patterns, every bound and what its own answers are read against stay the ones of that first sweep: they are made from the
    original file alone, and nothing of the first sweep's own model is in them. A sweep that starts from an earlier model has to leave less tube
    loss than it."""
    earlier = json.loads((run/'declaration.json').read_text()).get('starts_from')
    return anchor(ROUND_TWO_RUNS/earlier) if earlier else run


def observed(run):
    """The run that holds the observations of the original file a chain of sweeps is counted from: its first sweep, or the earlier run that sweep's
    declaration names under original_observations_from. The observations are the original file's own output at every merge in one reading (baseline)
    and its second answers (own/original): what the engine put out with nothing changed, which stays what it is whatever is made from it. They are
    read there and never written. What is made from them, the reference and the floors of the own answers' budgets, is the first sweep's own and is
    never taken from the run that holds the observations (Astra, November 11: the observations' provenance apart from the reference's lineage)."""
    first = anchor(run)
    earlier = json.loads((first/'declaration.json').read_text()).get('original_observations_from')
    return ROUND_TWO_RUNS/earlier if earlier else first


def reading_of(run):
    """Where the original file's reading is kept: the baseline of the run that holds the chain's observations of the original file."""
    return observed(run)/'baseline'


def baseline(run):
    """Every answer's own text read once through the original file, the way the run declares, in the reading every pass of the build uses. At every
    merge the engine's own output is kept: everything is counted from it, a solve's trial changes and what a written change did alike, and the reference
    holds it against the rows recorded while the model generated for its floors. For a matrix's merge the merge rebuilt from its input as the engine
    multiplies it is kept as well: the reference says how far a solve's expression of the merge lies from the engine's output on the same pass.
    (controlled-bf16-1's baseline, made November 10, keeps that merge rebuilt from the input as recorded; its index says so.)

    Nothing was recorded at the last layer's last merge at the question's positions while the original model generated, so there is nothing to hold its
    reading against. The original file reads the prompts once more for that one output: how far its two readings lie apart is the floor of the questions'
    bounds there."""
    declaration = declared(run)
    if observed(run) != run:
        raise ValueError('The original file\'s reading this sweep is counted from is another run\'s (starts_from, original_observations_from); it reads none of its own')
    _, members, index = ladder.cohort(run)
    tensors = json.loads(ladder.TENSORS.read_text())
    original = lambda layer, kind: ladder.weights_of(tensors, layer, kind)[1:]
    folder, report = run/'baseline', []
    folder.mkdir()
    for layer, kind, to, log, rebuilt, seconds in passes(ORIGINAL_MODEL, run, 'baseline', members, index['layers'], original):
        kept(folder, kind, layer, to, log)
        if rebuilt is not None:
            kept(folder, kind, layer, *rebuilt, rebuilt=True)
        report.append(dict(layer=layer, kind=kind, rows=int(to.shape[0]), pass_seconds=seconds))
        print(json.dumps(report[-1]), flush=True)
    last = index['layers'][-1]
    kept(folder, 'layer', last, *unit_and_log(torch.from_numpy(asked_again(ORIGINAL_MODEL, run, 'baseline-prompts-again', f"{OUTPUT['layer']}-{last}", members)).to(DEVICE)), again=True)
    save(folder/'index.json', dict(read_through=str(ORIGINAL_MODEL.relative_to(ROOT)), read_through_sha256=digest(ORIGINAL_MODEL), passes_read=READING, merges=report,
         rebuilt_from='the matrix\'s input as the engine multiplies it',
         meaning='state-<kind>-<layer>.f32: float32 [rows, 1536], the direction of the state right after that merge, '
         'as the engine itself put it out; size-<kind>-<layer>.f32: float32 [rows, 1], the log of its size; rebuilt-state and rebuilt-size: the same merge rebuilt from the '
         'matrix\'s input as the engine multiplies it (rounded to BF16 where it reads several tokens at once) and the state it adds to. Rows: every answer step of every answer in the correspondence\'s order, then every question position, read from the '
         'answers\' own text the way passes_read says. kind layer is the layer\'s last merge, where the per-layer embedding is added. again-state and again-size: the last '
         'layer\'s last merge at every question position, from a second reading of the prompts alone.'))


def reference(run):
    """What the original model fixes at every one of its merges, written once, from the original model alone, and only read after: reference/<kind>-<layer>.npz.

    Every pattern, bound and floor is the original file's: its rows as recorded while it generated (a matrix's merge rebuilt from its input as the engine
    multiplied it, `recorded`; a layer's last merge as the engine put it out), and, for the floors, the engine's own output at the same merge in a pass
    of the same text through the original file (`baseline`, of this run or of the run its declaration names under original_observations_from). The
    questions' bounds are taken from the engine's own output at the prompt in that reading; at the last layer's last merge, where no prompt row is on
    record, their floor is how far the original file's two readings of the prompts lie apart. The gap and its chance have to be the declared state
    overlap's, which is therefore one made in the same form. Beside them, and in none of them, the record says against the bounds how far the merge
    rebuilt in that pass (in the form the baseline's index names) lies from the engine's own output there."""
    declaration = declared(run)
    split = allowance_of(declaration) == ALLOWANCES[1]
    if anchor(run) != run:
        raise ValueError('A sweep that starts from an earlier one is counted from the first sweep\'s reference; it makes none of its own')
    source, members, index = ladder.cohort(run)
    tensors = json.loads(ladder.TENSORS.read_text())
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    pairs = [np.load(ladder.CORRESPONDENCE/'pairs'/f"{m['id']}.npy").astype(np.int64) for m in members]
    overlap = np.load(source/'recurrence.npz')
    steps = sum(m['steps'] for m in members)
    layout = Layout([not m['correct'] for m in members], [m['steps'] for m in members], [m['question'] for m in members], pairs, places)
    folder, report = run/'reference', []
    folder.mkdir()
    own = reading_of(run)
    on_card = lambda pair: tuple(torch.from_numpy(np.ascontiguousarray(v)).to(DEVICE) for v in pair)
    told = lambda v: dict(parts=int(v.numel()), over_their_bound=int((v > 1).sum()), median=float(v.flatten().double().median()), largest=float(v.max()))
    for layer in index['layers']:
        for kind in MERGES:
            w, g = ladder.weights_of(tensors, layer, kind)[1:] if kind in KINDS else (None, None)
            to, log, asking, asking_log = recorded(members, index, layer, kind, w, g, tensors)
            given, given_log = on_card(kept(own, kind, layer))   # the engine's own output in the pass through the original file
            prompt = given[steps:], given_log[steps:]
            unrecorded = asking is None   # the last layer's last merge at the question's positions: nothing was recorded there while the model generated
            if unrecorded:
                asking, asking_log = on_card(kept(own, kind, layer, again=True))
            values, record = frozen(members, pairs, places, to, log, asking, asking_log, given, given_log, native_log(members, layer, kind) if kind in KINDS else None, prompt, split=split)
            record.update(questions_floor_measured_against='a second reading of the prompts through the original file' if unrecorded else 'the rows recorded while the original model generated')
            if kind in KINDS:   # the gap and its chance are the state overlap's own, or something here reads the rows another way
                position = 2 * layer + KINDS[kind][-1]
                for name, mine in (('gap', (values['gap'].double() ** 2).sum(-1)), ('chance', values['chance'].double())):
                    if not torch.allclose(mine, torch.from_numpy(overlap[name][:, position]).to(DEVICE), rtol=1e-3, atol=1e-7):
                        raise ValueError(f'{name} at {kind}-{layer} is not the state overlap\'s')
            elif layer != index['layers'][-1]:   # how well the last layer's way of finding this direction does where the direction itself is on record
                record.update(direction_from_the_recorded_turns_apart_rms=float(((turned_to(members, layer) - to) ** 2).sum(1).double().mean().sqrt()))
            del given, given_log, prompt
            bounds = Bounds(values['gap'], values['chance'], values['direction'], values['size'], values.get('direction_mean'), values.get('size_mean'))
            limits = limits_of(bounds, dict(question=values['question'], question_size=values['question_size']))
            between = lambda a, b: {family: told(moved / limits[family].clamp(min=TINY)) for family, moved in
                                    layout.measured(a[0][:steps] - b[0][:steps], a[1][:steps] - b[1][:steps], a[0][steps:] - b[0][steps:], a[1][steps:] - b[1][steps:], bounds)[1].items()}
            if kind in KINDS:   # how far the merge rebuilt in that pass lies from the engine's own output there
                record.update(the_passes_rebuilt_merge_against_the_engines_output=between(on_card(kept(own, kind, layer, rebuilt=True)), on_card(kept(own, kind, layer))))
            np.savez(folder/f'{kind}-{layer}.npz', **{name: value.cpu().numpy() for name, value in values.items()})
            report.append(dict(layer=layer, kind=kind, **record, direction_bound_rms=[float(v.mean().sqrt()) for v in values['direction']], size_bound_rms=[float(v.mean().sqrt()) for v in values['size']],
                               question_bound_rms=float(values['question'].mean().sqrt()), question_size_bound_rms=float(values['question_size'].mean().sqrt()),
                               **{f'{family}_bound_rms': [float(v.mean().sqrt()) for v in values[family]] for family in MEANS if family in values}))
            print(json.dumps(report[-1]), flush=True)
            del to, log, asking, asking_log, values, bounds, limits
            torch.cuda.empty_cache()
    if len(report) != len(MERGES) * len(index['layers']):
        raise ValueError('The reference does not cover every merge of the model')
    save(folder/'index.json', dict(merges=report, allowance=allowance_of(declaration), how=frozen.__doc__, made_how=reference.__doc__, floors_read_from=str(own.relative_to(ROOT)),
                                   floors_read_from_a_pass_through_sha256=json.loads((own/'index.json').read_text())['read_through_sha256'],
                                   the_passes_rebuilt_merge_is_from=json.loads((own/'index.json').read_text()).get('rebuilt_from', 'the matrix\'s input as recorded'),
                                   recorded_rows_rebuilt_from='the matrix\'s input as the engine multiplied it', state_overlap=str((source/'recurrence.npz').relative_to(ROOT)),
                                   state_overlap_sha256=digest(source/'recurrence.npz'), passes_read=READING))


def reference_sha256(run):
    """One sha256 for the frozen reference a run is counted from: of the name and sha256 of every file in the first sweep's reference folder, in order."""
    return hashlib.sha256(''.join(f'{path.name} {digest(path)}\n' for path in sorted((anchor(run)/'reference').iterdir())).encode()).hexdigest()


def allowance_held(run):
    """The allowance a run's bounds are held under, which is its reference's: the one the first sweep of its chain (`anchor`) made its reference
    with, frozen with it (the reference's index names it; a reference made before the allowances had names has the first). What the run itself
    declares has to agree before any bound is read. A declaration that names an allowance names the reference's. One that names none has the
    first where the reference is the run's own, and its chain's where the run starts from an earlier sweep: a later sweep cannot change what its
    chain is held to, so it either says the same or says nothing. (Astra's review of November 11,
    records/retrospectives/2026-11-11-mean-row-implementation-review-astra.md: read from the arrays a reference's files hold alone, a sweep
    declared with the mean and the rows apart over a chain whose reference has the rows' bounds only was held to those, and the reverse, without
    a word.)"""
    first = anchor(run)
    held = json.loads((first/'reference'/'index.json').read_text()).get('allowance', ALLOWANCES[0])
    if held not in ALLOWANCES:
        raise ValueError(f"{first.name}'s reference names an allowance this file does not know: {held!r}")
    declaration = json.loads((run/'declaration.json').read_text())
    named = allowance_of(declaration) if 'allowance' in declaration or first.resolve() == run.resolve() else held
    if named != held:
        raise ValueError(f"{run.name} declares the allowance {named!r} and the reference it is counted from, {first.name}'s, was made with {held!r}: "
                         'a reference is frozen once made, and no bound is read under another allowance than its own')
    return held


def bounds_of(run, kind, layer, pick):
    """A merge's frozen bounds, and the bounds of the picked answers' questions: the first sweep's, for every sweep after it, under the allowance
    that reference was made with, which the run's declaration has to agree with (allowance_held)."""
    apart = allowance_held(run) == ALLOWANCES[1]
    values = np.load(anchor(run)/'reference'/f'{kind}-{layer}.npz')
    if (MEANS[0] in values.files) != apart:   # a reference made with the mean and the rows apart has the means' bounds, and no other has
        raise ValueError(f"{anchor(run).name}'s reference at {kind}-{layer} {'has no' if apart else 'has'} bounds of the means, against the allowance its index names")
    names = ('gap', 'chance', 'direction', 'size', *(MEANS if apart else ()))
    return Bounds(*(torch.from_numpy(values[name]).to(DEVICE) for name in names)), {family: torch.from_numpy(values[family][pick]).to(DEVICE) for family in ('question', 'question_size')}


def read_state(layout, bounds, question_bounds, to, log, origin, log_origin):
    """A reading, as Part.realized gives it, of states as they are: the rows' unit state and log size against where they are counted from."""
    steps = to.shape[0] - int(layout.asker.numel())
    more = {}
    mean, moved = layout.measured(to[:steps] - origin[:steps], log[:steps] - log_origin[:steps], to[steps:] - origin[steps:], log[steps:] - log_origin[steps:], bounds, more)
    beyond = ((bounds.gap - mean).norm(dim=1) - bounds.radius).clamp(min=0) * bounds.asked
    return dict(tube=float((beyond ** 2).sum()), inside=int(((beyond == 0) & (bounds.asked > 0)).sum()), mean=mean, shared=bounds.asked * (mean * bounds.along).sum(-1),
                over={family: moved[family] / limit.clamp(min=TINY) for family, limit in limits_of(bounds, question_bounds).items()}, **more)


def rows_of(members, pick):
    """Where the picked answers' rows sit among all the answers' rows: their answer steps, then their question positions."""
    begins = np.cumsum([0] + [m['steps'] for m in members])
    asking = begins[-1] + np.cumsum([0] + [m['question'] for m in members])
    ids = np.flatnonzero(pick)
    return np.concatenate([np.arange(begins[i], begins[i + 1]) for i in ids] + [np.arange(asking[i], asking[i + 1]) for i in ids])


def counted_from(run, rows):
    """Where the picked rows' movement is counted from at a merge: the original file's own output there in the first sweep's reading (`baseline`), so
    that it is always the whole distance from the original model. A solve's rebuilt merge and the engine's own output of a written change are both
    counted from it: with the matrix's input as the engine multiplies it, the rebuilt merge of the original file is that output."""
    own = reading_of(run)
    return lambda kind, layer: [torch.from_numpy(np.ascontiguousarray(v[rows])).to(DEVICE) for v in kept(own, kind, layer)]


def build(run):
    """One sweep: the 70 matrices in path order, into the run's own model file (`start` makes it).

    Before each matrix the fitting answers' own text goes through the model as it stands, and the matrix's input and the state it adds to are read: so every
    reading at its merge has every earlier change in it. Every movement is the whole distance from the original model at that merge, on the same text:
    what earlier sweeps and earlier matrices of this sweep moved, and what this matrix adds. The patterns, the
    bounds and where movement is counted from are the first sweep's, whatever sweep this is.

    What an earlier change left outside a bound at this matrix's own merge is this matrix's to bring back first, inside `choose`. If it cannot, the pull
    of the matrix written just before is taken back by halving toward where that pull started (that matrix as it stood, or as it was brought back), each
    trial read through the model, until this matrix's merge is inside with this matrix as it stands and that matrix's own merge still holds on its rows;
    this matrix is then chosen again.

    The same read also brings what the engine itself put out at the merges just above this matrix, which no matrix after them can move: after attention,
    the attention's merge; before the next layer's attention, the feed-forward's merge and the layer's last merge. If a bound is outside there, the pull
    of the matrix written just before is taken back the same way. After the last matrix one more read does this for the last two merges. Where nothing
    can be taken back, or no part of it holds, the sweep ends, not accepted.

    The command can be cut off and run again: it goes on after the last matrix whose record is in build.jsonl. Before a position writes anything, the
    matrices it may write are kept as the file holds them (model/undo) until its record is in the log; a position that was cut off is put back from
    them first, so the model file is always what the log describes. A matrix written before a restart can no longer be taken back: its rows are gone."""
    declared(run)
    _, members, index = ladder.cohort(run)
    tensors = json.loads(ladder.TENSORS.read_text())
    where = json.loads((run/'model'/'bindings.json').read_text())
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    fitting, groups = picks(members)
    used = [m for m, take in zip(members, fitting) if take]
    pairs = [np.load(ladder.CORRESPONDENCE/'pairs'/f"{m['id']}.npy").astype(np.int64) for m in used]
    wrong, steps, questions = np.array([not m['correct'] for m in used]), [m['steps'] for m in used], [m['question'] for m in used]
    thirds, everyone = [each[fitting] for each in groups], np.ones(len(used), bool)
    layout = Layout(wrong, steps, questions, pairs, places)
    origin_of = counted_from(run, rows_of(members, fitting))
    partial, log = model_of(run).with_suffix('.partial'), run/'build.jsonl'
    if model_of(run).exists():
        raise ValueError('The edited model exists; inspect it, never overwrite it')
    if not partial.exists():
        raise ValueError('The run has no model file to write into (start makes it)')
    done = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    undo = run/'model'/'undo'   # the matrices a position may write, as the file held them, until the position's record is in the log

    def on_disk(path):
        """Returns once a file or a folder's listing is on the disk itself: what is written next relies on it surviving a power cut."""
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    if (undo/'position.json').exists():   # an earlier run of this command was cut off inside a position
        cut = json.loads((undo/'position.json').read_text())
        if cut['position'] >= len(done):   # its record never reached the log: what it wrote is taken out, so the file is what the log describes
            with partial.open('r+b') as model:
                for each in cut['tensors']:
                    model.seek(each['offset_bytes'])
                    model.write((undo/each['name']).read_bytes())
                model.flush()
                os.fsync(model.fileno())
            print(json.dumps(dict(cut_off_at_position=cut['position'], put_back_as_they_stood=[each['name'] for each in cut['tensors']])), flush=True)
    shutil.rmtree(undo, ignore_errors=True)

    def keep(position, bindings):
        """Before a position writes anything: the matrices it may write, as the file holds them, kept beside the model."""
        undo.mkdir()
        with partial.open('rb') as model:
            for each in bindings:
                model.seek(each['offset_bytes'])
                (undo/each['name']).write_bytes(model.read(each['bytes']))
                on_disk(undo/each['name'])
        (undo/'position.json').write_text(json.dumps(dict(position=position, tensors=[{name: each[name] for name in ('name', 'offset_bytes', 'bytes')} for each in bindings])))
        for path in (undo/'position.json', undo, undo.parent):
            on_disk(path)

    def write(binding, weights):
        values = weights.cpu().numpy().ravel(order='F')
        if binding['native_type'] != 'bf16':
            raise ValueError(f"{binding['name']} is kept as {binding['native_type']} in this file, not BF16")
        bits = to_bf16(values)
        if bits.nbytes != binding['bytes']:
            raise ValueError(f"{binding['name']}: the stored matrix has another size")
        with partial.open('r+b') as model:
            model.seek(binding['offset_bytes'])
            model.write(bits.tobytes())
            model.flush()
            os.fsync(model.fileno())

    before = None   # the matrix this run of the command wrote last, with its rows: what is taken back when what follows it is left outside
    for position, layer, kind in [*matrices(index['layers']), (2 * len(index['layers']), index['layers'][-1] + 1, None)]:
        if position < len(done):
            continue
        started = time.time()
        closing = kind is None   # after the last matrix: only the engine's own output at the last two merges is read
        # the merges just above, as the engine itself puts them out, counted from the original file's own output there
        above = [('attention', layer)] if kind == 'feedforward' else [('feedforward', layer - 1), ('layer', layer - 1)] if layer > 0 else []
        frames = {merge: (bounds_of(run, *merge, fitting), origin_of(*merge)) for merge in above}
        labels = list(dict.fromkeys([*(() if closing else ladder.labels_of(layer, kind)), *(f'{OUTPUT[name]}-{at}' for name, at in above)]))
        if not closing:
            original, w, g = ladder.weights_of(tensors, layer, kind)
            binding = where[original['name']]
            bounds, question_bounds = bounds_of(run, kind, layer, fitting)
            origin, log_origin = origin_of(kind, layer)
            stands = ladder.matrix_in(partial, binding)   # the matrix as the model holds it: the original, or an earlier sweep's
        keep(position, [*([before['binding']] if before is not None and before['changed'] else []), *(() if closing else [binding])])
        count = [0]

        def through(labels):
            """One pass through the model as it stands: the outputs asked for, at the fitting answers' rows."""
            count[0] += 1
            while (run/'passes'/f"{position}-{kind or 'closing'}-{count[0]}").exists():   # an earlier run of the command read under this name
                count[0] += 1
            return dict(zip(labels, (torch.from_numpy(v).to(DEVICE) for v in passed(partial, run, f"{position}-{kind or 'closing'}-{count[0]}", labels, used))))

        def read():
            found = through(labels)
            trace = None if closing else Trace(*(found[label] for label in ladder.labels_of(layer, kind)), origin, log_origin, wrong, steps, questions, pairs, question_bounds,
                                               behind=behind_of(run, layer, used, tensors, fitting, origin_of) if kind == 'feedforward' else None)
            return trace, {(name, at): read_state(layout, *frames[name, at][0], *unit_and_log(found[f'{OUTPUT[name]}-{at}']), *frames[name, at][1]) for name, at in above}

        def say(line):   # the choice as it goes: every round of a bringing back and every stage of a path, as it ends
            print(json.dumps(dict(position=position, seconds_since_the_position_started=round(time.time() - started, 1), **line)), flush=True)

        inside = lambda engine: all(outside([reading]) == 0 for reading in engine.values())
        told = lambda engine: {f'{name}-{at}': dict(summary(reading, frames[name, at][0][0]), holds_every_bound=outside([reading]) == 0) for (name, at), reading in engine.items()}

        def stands_inside(trace):   # with this matrix as it stands, every group and all the fitting answers together inside every bound of its merge
            parts = [trace.part(each, places) for each in thirds]
            readings = [each.realized(bounds, stands, g) for each in parts]
            return outside(readings) == 0 and outside([pooled(parts, readings, bounds)]) == 0

        def take_back(fine, why):
            """The pull of the matrix written before this one, halved toward where that pull started (the matrix as it stood, or as it was brought back: a
            bringing back is not taken back, the matrix's own merge is outside without it), until `fine` holds of what is then read here and that
            matrix's own merge still holds on its rows, in every group and all together. Returns what was read at the part kept, or None with that
            matrix left as it was written."""
            def holds(share):
                if share == 1.0:   # the change as written is what was just read
                    return None
                weights = written(before['was'] + share * (before['weights'] - before['was']), before['measure'])
                if bool((weights == before['weights']).all()):   # this part takes back no weight the file holds: it is what was just read
                    return None
                if outside([each.realized(before['bounds'], weights, before['g']) for each in before['rows']]) > 0:
                    return None
                write(before['binding'], weights)
                found = read()
                return (found, weights) if fine(found) else None
            share, kept, halvings = halved(holds, TAKEN_BACK)
            found = None
            if kept is not None:
                found, before['weights'] = kept   # the matrix that was written and read, itself
            write(before['binding'], before['weights'])
            taken.append(dict(tensor=before['binding']['name'], why=why, halvings=halvings, part_of_its_pull_kept=None if found is None else share,
                              taken_back_toward='the matrix as it was brought back' if before['brought_back'] else 'the matrix as it stood'))
            return found

        trace, engine = read()
        record, taken = dict(position=position, layer=None if closing else layer, kind=kind or 'closing', tensor=None if closing else binding['name']), []
        can_take_back = before is not None and before['changed']
        if engine:
            record.update(engines_output_above=told(engine))
            if not inside(engine):
                found = take_back(lambda found: inside(found[1]), 'the engine\'s own output above had a bound outside') if can_take_back else None
                if found is None:
                    record.update(status='outside', selected='the engine\'s own output above has a bound outside and ' + ('no part of the change before it holds' if can_take_back else 'no change of this run to take back'))
                else:
                    trace, engine = found
                    record.update(engines_output_above_after_taking_back=told(engine))
        if closing:
            record.setdefault('status', 'read')
        elif record.get('status') != 'outside':
            cohort = trace.part(everyone, places)
            unit = unit_of(cohort, bounds, w, g)
            record.update(as_the_earlier_changes_left_it=dict(summary(cohort.realized(bounds, stands, g), bounds)))
            del cohort
            torch.cuda.empty_cache()
            realized, chosen, reading, measure, stood = choose(trace, bounds, w, g, stands, thirds, unit, held=True, say=say)
            if chosen['status'] == 'outside' and can_take_back:
                found = take_back(lambda found: inside(found[1]) and stands_inside(found[0]), 'nothing brought this matrix\'s own merge back inside')
                if found is not None:
                    trace, engine = found
                    record.update(first_choice=dict(selected=chosen['selected'], brought_back=chosen.get('brought_back')))
                    realized, chosen, reading, measure, stood = choose(trace, bounds, w, g, stands, thirds, unit, held=True, say=say)
            if not bool((realized.view(torch.int32) == ladder.rounded(realized).view(torch.int32)).all()):   # the engine's own writer gives these weights back bit for bit
                raise ValueError(f"{binding['name']}: the weights read here are not ones the file holds")
            changed, behind_outside = bool((realized != stands).any()), False
            if changed:
                write(binding, realized)
            if kind == 'feedforward' and changed:
                # Behind a feed-forward the solve read the layer's last merge through its own arithmetic, which is the engine's except where the
                # engine reads several tokens at once (Behind). So the engine's own output at the two merges behind the written matrix is read now,
                # while this matrix can still be chosen again: where the engine reads a bound outside that the solve read inside, the bounds the solve
                # works to there are drawn in by what was measured (Trace.against) and the matrix is chosen again from the one written, until the
                # engine's own reading is inside or the rounds allowed are used. What counts is the engine's reading against the bounds as they are.
                behind = [('feedforward', layer), ('layer', layer)]
                behind_frames = {merge: (bounds_of(run, *merge, fitting), origin_of(*merge)) for merge in behind}
                wanted, again = [f'{OUTPUT[name]}-{at}' for name, at in behind], []
                for attempt in range(EFFORT['rounds'] + 1):
                    found = through(wanted)
                    engines = {(name, at): read_state(layout, *behind_frames[name, at][0], *unit_and_log(found[f'{OUTPUT[name]}-{at}']), *behind_frames[name, at][1]) for name, at in behind}
                    readings = {f'{name}-{at}': dict(summary(each, behind_frames[name, at][0][0]), holds_every_bound=outside([each]) == 0) for (name, at), each in engines.items()}
                    if inside(engines) or attempt == EFFORT['rounds'] or outside([engines['feedforward', layer]]) > 0:
                        break
                    drawn = trace.against(found[wanted[1]], bounds, realized, g, (everyone, *thirds), places)
                    say(dict(the_engine_reads_the_layers_last_merge_outside=readings[f'layer-{layer}'], its_bounds_drawn_in_by_at_most=drawn))
                    earlier = dict(selected=chosen['selected'], status=chosen['status'], the_engines_own_output_behind=readings, the_layers_last_merges_bounds_drawn_in_by_at_most=drawn)
                    realized, chosen, reading, measure, stood = choose(trace, bounds, w, g, realized, thirds, unit, held=True, say=say)
                    again.append(dict(earlier, chosen_again=chosen['selected']))
                    if not bool((realized.view(torch.int32) == ladder.rounded(realized).view(torch.int32)).all()):
                        raise ValueError(f"{binding['name']}: the weights read here are not ones the file holds")
                    write(binding, realized)
                record.update(the_engines_own_output_behind=readings, **(dict(chosen_again=again) if again else {}))
                behind_outside = not inside(engines)
                del found, engines
            record.update(chosen, ridge_unit=unit, written=changed, change_over_matrix=float((realized - w).norm() / w.norm()),
                          change_in_this_sweep_over_matrix=float((realized - stands).norm() / w.norm()))
            if behind_outside:   # the matrix stays as written, for inspection; the sweep ends at this position
                record.update(status='outside', selected=f"{chosen['selected']}; the engine's own output behind the written matrix has a bound outside")
            # every matrix keeps its rows until the next one has been read
            # what can be taken back of it is its pull: from where the pull started, which holds every bound of its own merge on these rows
            before = dict(binding=binding, w=w, g=g, was=stood, weights=realized, measure=measure, changed=bool((realized != stood).any()), brought_back=stood is not stands, bounds=bounds,
                          rows=[trace.part(each, places) for each in (everyone, *thirds)])
        if taken:
            record.update(taken_back=taken)
        record.update(passes=count[0], seconds=round(time.time() - started, 1))
        with log.open('a') as file:
            file.write(json.dumps(record) + '\n')
            file.flush()
            os.fsync(file.fileno())
        on_disk(run)
        shutil.rmtree(undo)   # the record is in the log: the file is what it describes
        print(json.dumps({name: value for name, value in record.items() if name not in ('stages', 'starts_passed_over', 'brought_back', 'first_choice')}), flush=True)
        if record['status'] == 'outside':
            raise SystemExit(f"{record['tensor'] or 'the last merges'}: {record['selected']}. The sweep ends here, not accepted; {partial.name} is left as it stands for inspection.")
        del trace, engine, frames
        torch.cuda.empty_cache()
    partial.rename(model_of(run))
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    save(run/'edit.json', dict(model=str(model_of(run).relative_to(run)), model_sha256=digest(model_of(run)), passes_read=READING, counted_from=str(anchor(run).relative_to(ROOT)),
                               fitting=[m['id'] for m in used], groups=[[m['id'] for m, take in zip(members, each) if take] for each in groups],
                               matrices_written=sum(line.get('written', False) for line in lines), matrices_taken_back=[each for line in lines for each in line.get('taken_back', [])], matrices=lines))


def read_merges(run, model, name, pick, origin_of):
    """The picked answers' own text through a model, and at every merge of the model the reading of the engine's own output against the frozen reference,
    counted from where `origin_of` says; beside each matrix's merge, what the rebuilt merge reads there, counted from the same place. Every merge has
    to be there: a reading with one missing is not a reading of the model."""
    _, members, index = ladder.cohort(run)
    tensors = json.loads(ladder.TENSORS.read_text())
    where = tensors if model == ORIGINAL_MODEL else json.loads((run/'model'/'bindings.json').read_text())
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    picked = [m for m, take in zip(members, pick) if take]
    layout = Layout([not m['correct'] for m in picked], [m['steps'] for m in picked], [m['question'] for m in picked],
                    [np.load(ladder.CORRESPONDENCE/'pairs'/f"{m['id']}.npy").astype(np.int64) for m in picked], places)
    weights = lambda layer, kind: (ladder.matrix_in(model, where[f'blk.{layer}.{KINDS[kind][1]}.weight']), ladder.weights_of(tensors, layer, kind)[2])
    merges = []
    for layer, kind, to, log, rebuilt, seconds in passes(model, run, name, picked, index['layers'], weights):
        bounds, question_bounds = bounds_of(run, kind, layer, pick)
        reading = read_state(layout, bounds, question_bounds, to, log, *origin_of(kind, layer))
        merges.append(dict(layer=layer, kind=kind, tube_loss=reading['tube'], tube_loss_of_the_original=bounds.whole, holds_every_bound=outside([reading]) == 0, **summary(reading, bounds)))
        if rebuilt is not None:
            solved = read_state(layout, bounds, question_bounds, *rebuilt, *origin_of(kind, layer))
            merges[-1].update(as_the_rebuilt_merge_reads_it=dict(tube_loss=solved['tube'], holds_every_bound=outside([solved]) == 0, largest_over_its_bound=worst(solved)))
        print(json.dumps(merges[-1]), flush=True)
    if len(merges) != len(MERGES) * len(index['layers']):
        raise ValueError('Not every merge of the model was read')
    return picked, merges


def verdict(run):
    """What a sweep's two readings say together, joined from their records whichever was made first; nothing is read or answered again.

    The replay (`composed`) and the model's own answers (`own`) are two verdicts and stay two; neither is needed for the other to be made. The sweep is
    eligible for the next one when both are true of the same model, the same frozen reference, the same first sweep and the same answers no fit used
    (Astra, November 10). While one of the two is not made, or the two records are not bound to the same sources, that is not said: it is unknown, not
    false. The own answers holding while the replay fails is a finding about the model's own traces, with the failed replay beside it; it is not an
    eligible sweep, and it does not show the answers were damaged. The repair of the pattern is whole when the sweep is eligible and its own answers
    leave no asked place outside its ball. Beside them, the tube loss the original file's second answers read: how much tube loss moves with nothing
    changed, which a smaller loss has to be seen against."""
    records = {name: json.loads((run/f'{name}.json').read_text()) if (run/f'{name}.json').exists() else None for name in ('composed', 'own')}
    replayed, answered_own = records['composed'], records['own']
    missing = [name for name, record in records.items() if record is None]
    apart = [] if missing else [field for field in ('model_sha256', 'reference_sha256', 'counted_from', 'answers_no_fit_used') if not replayed.get(field) or replayed.get(field) != answered_own.get(field)]
    said = not missing and not apart
    eligible = bool(replayed['replay_passes'] and answered_own['own_answers_hold']) if said else None
    told = dict(how=verdict.__doc__, readings_not_made=missing, the_two_records_differ_in=apart,
                replay_passes=replayed['replay_passes'] if replayed else None, own_answers_hold=answered_own['own_answers_hold'] if answered_own else None, eligible=eligible,
                pattern_repair_whole=bool(eligible and answered_own['asked_places_left_outside_their_ball'] == 0) if said else None,
                replay={name: replayed[name] for name in ('every_bound_holds', 'tube_loss_left', 'tube_loss_left_where_the_sweep_started')} if replayed else None,
                own_answers={name: answered_own[name] for name in ('every_bound_holds', 'every_answer_whole_as_its_recording_was', 'every_answer_ended_by_itself', 'tube_loss_left', 'tube_loss_left_where_the_sweep_started',
                                                                    'tube_loss_left_of_the_original_files_second_answers', 'asked_places_left_outside_their_ball')} if answered_own else None,
                **({name: answered_own[name] for name in ('model_sha256', 'reference_sha256', 'counted_from')} if said else {}))
    save(run/'verdict.json', told)
    print(json.dumps({name: value for name, value in told.items() if name != 'how'}), flush=True)


def composed(run):
    """The finished model read once on the answers no fit used: their own text through it, the way the run's passes read, and at every one of the model's
    merges every bound and the tube loss, from the engine's own output there counted from the original file's. The replay passes if every bound holds at
    every merge and the tube loss, summed over the merges, is less than that of the model the sweep started from, by the same frozen reference (the
    original model for a first sweep, whose tube loss is the whole of it; the earlier sweep's model after that). Nothing is
    chosen, scaled or tuned from this reading. It does not accept the training, and the model's own answers are read whatever it says. Beside each
    matrix's merge the record says what the rebuilt merge reads there, the form a solve reads in; it decides nothing."""
    declared(run)
    _, members, _ = ladder.cohort(run)
    fitting, _ = picks(members)
    others, merges = read_merges(run, model_of(run), 'composed', ~fitting, counted_from(run, rows_of(members, ~fitting)))
    left, original = sum(m['tube_loss'] for m in merges), sum(m['tube_loss_of_the_original'] for m in merges)
    earlier = json.loads((run/'declaration.json').read_text()).get('starts_from')
    was = json.loads((ROUND_TWO_RUNS/earlier/'composed.json').read_text())['tube_loss_left'] * original if earlier else original
    holds = all(m['holds_every_bound'] for m in merges)
    save(run/'composed.json', dict(how=composed.__doc__, answers=[m['id'] for m in others], answers_no_fit_used=[m['id'] for m in others], passes_read=READING,
                                   counted_from=str(anchor(run).relative_to(ROOT)), model_sha256=digest(model_of(run)), reference_sha256=reference_sha256(run), merges_read=len(merges),
                                   every_bound_holds=holds, merges_with_a_bound_outside=[f"{m['kind']}-{m['layer']}" for m in merges if not m['holds_every_bound']],
                                   tube_loss_left=left / original, tube_loss_left_where_the_sweep_started=was / original, replay_passes=bool(holds and left < was), merges=merges))
    print(json.dumps(dict(every_bound_holds=holds, tube_loss_left=left / original, tube_loss_left_where_the_sweep_started=was / original, replay_passes=bool(holds and left < was))), flush=True)
    verdict(run)


def reproduced(run):
    """The original file read a second time, on the answers no fit uses, against the baseline's reading of it: how far two readings of the same
    unchanged model lie apart at every merge, against the bounds. It enters no bound: a model's own noise never widens what it is allowed."""
    declared(run)
    _, members, _ = ladder.cohort(run)
    fitting, _ = picks(members)
    rows = rows_of(members, ~fitting)
    first = lambda kind, layer: [torch.from_numpy(np.ascontiguousarray(v[rows])).to(DEVICE) for v in kept(run/'baseline', kind, layer)]
    others, merges = read_merges(run, ORIGINAL_MODEL, 'reproduced', ~fitting, first)
    save(run/'reproduced.json', dict(how=reproduced.__doc__, answers=[m['id'] for m in others], passes_read=READING, read_through_sha256=digest(ORIGINAL_MODEL), merges=merges))


def answered(model, run, name):
    """A model answers the overlap's questions on its own: the prompt alone and nothing supplied after it, with the same greedy choice and the same limit
    of tokens as the original recordings. Kept for each answer, under own/<name>/<id>:
      turns.npy, norms.npy   the turn and the sizes at all 105 merges of every decision the model made, as the original recordings are kept
      start.npy              the direction of the state before the first merge, from which the turns add up to the direction after every merge
      prompt.npy             105 x 2: over the question's positions, the mean squared movement of the unit state and of the log of its size after each
                             merge, from the original file's reading of the same prompt (the prompt holds the same tokens whatever the model, so its rows
                             are compared one to one)
      answer.json            what it wrote, and whether it ended by itself. How it stopped when it did not is in the capture's own record (`endings`).
    Each answer's first decision is computed where the prompt is read, several tokens at once; it is kept and read like every other."""
    from pipeline.acquisition.export_stream_traces import export_case
    from pipeline.acquisition.run_a_observation import run as observe
    _, members, index = ladder.cohort(run)
    limit = {m['id']: m['max_tokens'] for m in json.loads((WIDE/'members.json').read_text())}
    first = reading_of(run)
    asking = sum(m['steps'] for m in members) + np.cumsum([0] + [m['question'] for m in members])
    merges = [(layer, kind) for layer in index['layers'] for kind in MERGES]
    folder = run/'own'/name
    todo = [(i, m) for i, m in enumerate(members) if not (folder/m['id']/'answer.json').exists()]
    for begin in range(0, len(todo), 10):   # ten answers at a time only bounds the raw stream on disk
        batch = todo[begin:begin + 10]
        raw, cases = folder/f"raw-{batch[0][1]['id']}", folder/f"raw-{batch[0][1]['id']}-cases.json"
        cases.parent.mkdir(parents=True, exist_ok=True)
        cases.write_text(json.dumps([dict(id=m['id'], tokens=m['tokens'], max_tokens=limit[m['id']]) for _, m in batch]))
        with CARD.given():
            observe(model, cases, raw, 0, 'natural', 'capture', 'residual', run/'declaration.json')
            for i, m in batch:
                result = json.loads((raw/m['id']/'result.json').read_text())
                export_case(raw/m['id'], folder/m['id'])
                ladder.first_direction(raw/m['id'], folder/m['id'])
                stream, moved = StreamRows(raw/m['id'], set(range(m['question']))), np.zeros((len(merges), 2))
                for number, (layer, kind) in enumerate(merges):
                    rows = np.stack([stream.values(f'{OUTPUT[kind]}-{layer}', p) for p in range(m['question'])]).astype('f8')
                    size = np.linalg.norm(rows, axis=1, keepdims=True)
                    to, log = (v[asking[i]:asking[i + 1]] for v in kept(first, kind, layer))
                    moved[number] = ((rows / size - to) ** 2).sum(1).mean(), ((np.log(size) - log) ** 2).mean()
                stream.close()
                np.save(folder/m['id']/'prompt.npy', moved)
                for file in ('stream.f32', 'stream.jsonl'):
                    (raw/m['id']/file).unlink()
                save(folder/m['id']/'answer.json', dict(id=m['id'], steps=len(result['steps']), ended_by_itself=bool(result['ended']) and not result['stopped_limit'], limit=limit[m['id']],
                                                        tokens=[step['selected'] for step in result['steps']], content=result['content']))
                print(json.dumps(dict(answered=m['id'], through=name, steps=len(result['steps']), ended_by_itself=bool(result['ended']) and not result['stopped_limit'])), flush=True)


def endings(folder, members):
    """How each of a model's own answers ended, beside how the original recording of the same question ended, from the captures' own records.

    An answer is whole as its recording was when it ended by itself within the recording's limit of tokens; or, where the recording itself ran into that
    limit, when it did that or ran into the same limit. Either way every decision it made has to be kept, at every merge the recording keeps (the
    rows of turns.npy and norms.npy). An answer that stopped any other way, or whose capture keeps fewer rows than it made decisions or fewer merges
    than the recording, is never whole: a failed or partial capture is not an answer that ran into the limit. Three of the 60 recordings ran into their limit of 768 tokens, so asking every answer to end by itself asked of a
    model what the recordings it is read against never did (Astra, November 11: reference-relative completeness of the observations; the members, the
    limit and every allowance stay as they were).

    Returns the literal facts and the predicate apart: whether every answer ended by itself is said as it is, whatever the recordings did."""
    limit = {m['id']: m['max_tokens'] for m in json.loads((WIDE/'members.json').read_text())}

    def stopped(root, name):
        found = sorted(root.glob(f'*/{name}/result.json'))
        if len(found) != 1:
            raise ValueError(f'{len(found)} capture records of {name} under {root}')
        result = json.loads(found[0].read_text())
        return dict(steps=len(result['steps']), ended_by_itself=bool(result['ended']) and not result['stopped_limit'], ran_into_the_limit=bool(result['stopped_limit']) and not result['ended'])

    each = []
    for m in members:
        then, now, wrote = stopped(WIDE/'capture', m['id']), stopped(folder, m['id']), json.loads((folder/m['id']/'answer.json').read_text())
        if (wrote['steps'], wrote['ended_by_itself'], wrote['limit']) != (now['steps'], now['ended_by_itself'], limit[m['id']]):
            raise ValueError(f"answer.json of {m['id']} under {folder} does not say what its capture's record says")
        merges = int(np.load(NATIVE/m['id']/'norms.npy', mmap_mode='r').shape[1])
        kept = all(tuple(np.load(folder/m['id']/name, mmap_mode='r').shape[:2]) == (now['steps'], merges) for name in ('turns.npy', 'norms.npy'))
        by_itself = now['ended_by_itself'] and now['steps'] <= limit[m['id']]
        to_the_limit = now['ran_into_the_limit'] and now['steps'] == limit[m['id']]
        recording_to_the_limit = then['ran_into_the_limit'] and then['steps'] == limit[m['id']]
        each.append(dict(id=m['id'], ended_by_itself=now['ended_by_itself'], ran_into_the_limit=to_the_limit, recording_ran_into_the_limit=recording_to_the_limit,
                         every_decision_kept=kept, whole=bool(kept and (by_itself or (recording_to_the_limit and to_the_limit)))))
    ids = lambda test: [one['id'] for one in each if test(one)]
    return dict(every_answer_ended_by_itself=all(one['ended_by_itself'] for one in each),
                answers_that_ran_into_the_limit=ids(lambda one: one['ran_into_the_limit']),
                recordings_that_ran_into_the_limit=ids(lambda one: one['recording_ran_into_the_limit']),
                answers_that_ran_into_the_limit_where_the_recording_ended_by_itself=ids(lambda one: one['ran_into_the_limit'] and not one['recording_ran_into_the_limit']),
                answers_that_stopped_another_way_or_keep_fewer_rows_than_decisions=ids(lambda one: not one['every_decision_kept'] or not (one['ended_by_itself'] or one['ran_into_the_limit'])),
                answers_not_whole_as_their_recording_was=ids(lambda one: not one['whole']),
                every_answer_whole_as_its_recording_was=all(one['whole'] for one in each))


def registered(folder, members):
    """Each of a model's own answers laid on the frozen axis, and on the original model's answer to the same question, by the computation alone: the least
    summed distance between the turns at all 105 merges, over a whole path that keeps every step of both sides, which is how the overlap itself was
    registered (complete_overlap.distance_path). What an answer says, which group it was in and how any bound reads enter nowhere, and a path is made
    once and never made again. Written beside each answer: pairs-axis.npy [step, place of the axis] and pairs-original.npy [step, step of the original
    answer]. Returns, per answer, how many steps each side has and how often one is matched."""
    from pipeline.overlap.complete_overlap import distance_path, unit_steps
    record = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())
    grid = unit_steps(np.load(NATIVE/record['axis']/'turns.npy', mmap_mode='r'))
    told = []
    for m in members:
        if not (folder/m['id']/'pairs-original.npy').exists():
            mine = unit_steps(np.load(folder/m['id']/'turns.npy', mmap_mode='r'))
            theirs = unit_steps(np.load(NATIVE/m['id']/'turns.npy', mmap_mode='r'))
            if len(theirs) != m['steps']:
                raise ValueError(f"{m['id']}: the original answer's recording has another number of steps")
            np.save(folder/m['id']/'pairs-axis.npy', distance_path(mine @ grid.T))
            np.save(folder/m['id']/'pairs-original.npy', distance_path(mine @ theirs.T))
        on_axis, on_original = np.load(folder/m['id']/'pairs-axis.npy'), np.load(folder/m['id']/'pairs-original.npy')
        told.append(dict(id=m['id'], steps=int(on_axis[-1, 0]) + 1, steps_of_the_original=m['steps'], most_places_at_one_step=int(np.bincount(on_axis[:, 0]).max()),
                         most_steps_at_one_place=int(np.bincount(on_axis[:, 1]).max()), most_original_steps_at_one_step=int(np.bincount(on_original[:, 0]).max()),
                         most_steps_at_one_original_step=int(np.bincount(on_original[:, 1]).max())))
    return told


OWN = ('direction', 'size', 'spread', 'size_spread', 'question', 'question_size')   # what a model's own answers are held to (own_read)


def own_states(folder, start, merges):
    """The direction of the state after the given merges at every step of one answer: the direction before the first merge plus the recorded turns up to
    each (a turn is the direction after a merge less the direction before it)."""
    return (start.astype('f8')[:, None] + np.cumsum(np.load(folder/'turns.npy', mmap_mode='r'), axis=1, dtype='f8'))[:, merges]


def place_means(values, path, places):
    """Each place's mean of the values of the steps that a whole path puts there. On such a path the steps of one place lie together."""
    first = np.flatnonzero(np.r_[True, np.diff(path[:, 1]) > 0])
    if len(first) != places:
        raise ValueError('The path does not reach every place of the axis')
    return np.add.reduceat(values[path[:, 0]], first, axis=0) / np.diff(np.r_[first, len(path)]).reshape((-1,) + (1,) * (values.ndim - 1))


def own_original(members, merges, places, picked):
    """What the original model's own answers fix for the reading of any model's own answers, from the original recordings and the frozen paths alone
    (Astra, November 10, 2026: records/retrospectives/2026-11-10-stable-training-review-astra.md, "Own-generation acceptance correction").

    Per answer, at each place of the axis and each of the given merges: z, the mean unit state of its steps there, and l, the mean log of the size. Per
    group: the centre of z and of l over its members, each member's squared distance from the centre (r for z, t for l), and the four variances over the
    group's members that the budgets are made of: of z (summed over its coordinates), of l, of r and of t. Nothing here is raised to a floor. Also the
    sums of z and of l over each pick's members of each group, which a model's own answers to the same questions are counted from."""
    entering = np.memmap(ladder.INPUTS/'state-attention-0.f32', dtype='<f4', mode='r').reshape(-1, 1536)   # the state before the first merge, as recorded while the original generated
    begins = np.cumsum([0] + [m['steps'] for m in members])
    group = np.array([int(m['correct']) for m in members])   # 0 the wrong group, 1 the correct one, as the bounds are ordered
    n, shape = np.bincount(group, minlength=2), (2, places, len(merges))

    def means(i):
        start, path = entering[begins[i]:begins[i + 1]].astype('f8'), np.load(ladder.CORRESPONDENCE/'pairs'/f"{members[i]['id']}.npy")
        folder = NATIVE/members[i]['id']
        return (place_means(own_states(folder, start / np.linalg.norm(start, axis=1, keepdims=True), merges), path, places),
                place_means(np.log(np.load(folder/'norms.npy', mmap_mode='r')[:, merges, 1]), path, places))
    centre, log_centre = np.zeros(shape + (1536,)), np.zeros(shape)
    base, log_base = {label: np.zeros(shape + (1536,)) for label in picked}, {label: np.zeros(shape) for label in picked}
    for i in range(len(members)):
        z, l = means(i)
        centre[group[i]] += z / n[group[i]]
        log_centre[group[i]] += l / n[group[i]]
        for label, pick in picked.items():
            if pick[i]:
                base[label][group[i]] += z
                log_base[label][group[i]] += l
    r, t = (np.zeros((len(members), places, len(merges))) for _ in range(2))
    for i in range(len(members)):
        z, l = means(i)
        r[i], t[i] = ((z - centre[group[i]]) ** 2).sum(-1), (l - log_centre[group[i]]) ** 2
    variance = dict(direction=np.stack([r[group == g].sum(0) / (n[g] - 1) for g in (0, 1)]), size=np.stack([t[group == g].sum(0) / (n[g] - 1) for g in (0, 1)]),
                    spread=np.stack([r[group == g].var(0, ddof=1) for g in (0, 1)]), size_spread=np.stack([t[group == g].var(0, ddof=1) for g in (0, 1)]))
    return dict(centre=centre, log_centre=log_centre, base=base, log_base=log_base, r=r, t=t, variance=variance)


def own_measured(folder, members, merges, places, picked, original, along, prompts_kept=True):
    """What a model's own answers are at the given merges (numbered among the 105), in the form the original's are kept in (own_original). Each answer is
    laid on the axis by its own path. Kept: the sums of z and of l over each pick's members of each group; per answer, the squared distance of z from
    its original group's centre with its part along the gap (so that a shared movement along the gap can be taken out afterwards), and the squared
    distance of l from the centre of the sizes; and each answer's prompt rows against the original's (`answered`).

    Beside it, and bounding nothing: each answer held against the original answer to the same question, row by row, through the whole path between the
    two. Every matched pair of steps gives one movement of the unit state and of the log of the size, squared; an own step's matches weigh one in all;
    a pair lies at the places its original step lies at by the frozen path; a place's value is the weighted mean there. Two answers of one group differ
    this way by about twice the group's variance, so this says how far an answer is from the one the original gave, not whether a pattern was kept.

    Both sides are the engine's own: the direction before the first merge plus its recorded turns, and the sizes it recorded. Only answers in one of
    the picks are read. `prompts_kept` is off only where no prompt rows were kept (a check on the original recordings themselves)."""
    begins = np.cumsum([0] + [m['steps'] for m in members])
    entering = np.memmap(ladder.INPUTS/'state-attention-0.f32', dtype='<f4', mode='r').reshape(-1, 1536)
    group, shape = np.array([int(m['correct']) for m in members]), (2, places, len(merges))
    total, log_total = {label: np.zeros(shape + (1536,)) for label in picked}, {label: np.zeros(shape) for label in picked}
    apart, along_gap, log_apart, rows, row_sizes = (np.zeros((len(members), places, len(merges))) for _ in range(5))
    prompts = np.zeros((len(members), len(merges), 2))
    for i, m in enumerate(members):
        if not any(pick[i] for pick in picked.values()):
            continue
        mine = folder/m['id']
        edited, log_edited = own_states(mine, np.load(mine/'start.npy'), merges), np.log(np.load(mine/'norms.npy', mmap_mode='r')[:, merges, 1])
        z, l = (place_means(values, np.load(mine/'pairs-axis.npy'), places) for values in (edited, log_edited))
        for label, pick in picked.items():
            if pick[i]:
                total[label][group[i]] += z
                log_total[label][group[i]] += l
        z -= original['centre'][group[i]]
        apart[i], along_gap[i], log_apart[i] = (z ** 2).sum(-1), (z * along).sum(-1), (l - original['log_centre'][group[i]]) ** 2
        del z
        # row by row against the original answer: every matched pair of steps, at each place its original step lies at
        start = entering[begins[i]:begins[i + 1]].astype('f8')
        theirs = own_states(NATIVE/m['id'], start / np.linalg.norm(start, axis=1, keepdims=True), merges)
        log_theirs = np.log(np.load(NATIVE/m['id']/'norms.npy', mmap_mode='r')[:, merges, 1])
        on_original, frozen_path = np.load(mine/'pairs-original.npy'), np.load(ladder.CORRESPONDENCE/'pairs'/f"{m['id']}.npy")
        many = np.bincount(frozen_path[:, 0], minlength=m['steps'])   # the places of each original step, which the frozen path lists together
        times = many[on_original[:, 1]]
        own_step, their_step = np.repeat(on_original[:, 0], times), np.repeat(on_original[:, 1], times)
        weight = np.repeat(1 / np.bincount(on_original[:, 0])[on_original[:, 0]], times)
        place = frozen_path[(np.cumsum(many) - many)[their_step] + np.arange(len(own_step)) - np.repeat(np.cumsum(times) - times, times), 1]
        if len(np.unique(place)) != places:
            raise ValueError(f"{m['id']}: the paths do not carry every step to the axis")
        for a in range(0, len(own_step), 128):   # 128 pairs at a time only bounds the memory
            b = slice(a, a + 128)
            np.add.at(rows[i], place[b], weight[b, None] * ((edited[own_step[b]] - theirs[their_step[b]]) ** 2).sum(-1))
            np.add.at(row_sizes[i], place[b], weight[b, None] * (log_edited[own_step[b]] - log_theirs[their_step[b]]) ** 2)
        for each in (rows, row_sizes):
            each[i] /= np.bincount(place, weights=weight, minlength=places)[:, None]
        if prompts_kept:
            prompts[i] = np.load(mine/'prompt.npy')[merges]
        del edited, theirs
    return dict(total=total, log_total=log_total, apart=apart, along_gap=along_gap, log_apart=log_apart, rows=rows, row_sizes=row_sizes, prompts=prompts)


def own_read(measured, original, members, label, pick, frames, floors=None):
    """The readings, merge by merge, of a model's own answers for the picked answers, counted from the original model's own answers to the same questions.

    With q the movement of a group's pattern (the mean over the picked members of z, less the original's) and a its amount along the gap where a move
    is asked (the wrong group's only):
      the tube     on the wrong pattern's whole movement, against the frozen gap and ball
      direction    |q - a gap direction|^2                                  size         (the movement of the mean of l)^2
      spread       (the mean change of the members' squared distance from the original centre, the wrong group's after taking a out)^2
      size_spread  the same for l
      question, question_size   each question's prompt rows against the original's, row by row: the prompt holds the same tokens in every model
    Each of the first four is over its own budget: twice the original variance of that quantity over the number of picked members of the group, never
    less than the same reading of the original file's second set of answers (`floors`, when one was made: without it the budget is the sampling
    term alone, which is not a measured floor of nothing). The count is the number of answers in the mean that is read, not the 30 the variance was
    estimated from. Beside each merge, bounding nothing: the median over the places of each group's row-by-row movement (own_measured)."""
    group = np.array([int(m['correct']) for m in members])
    of = [pick & (group == g) for g in (0, 1)]
    count = np.array([int(each.sum()) for each in of])
    tensor = lambda values: torch.from_numpy(np.ascontiguousarray(values)).float().to(DEVICE)
    readings = []
    for number, (bounds, question_bounds) in enumerate(frames):
        along, asked = bounds.along.cpu().numpy().astype('f8'), bounds.asked.cpu().numpy().astype('f8')
        q = (measured['total'][label][:, :, number] - original['base'][label][:, :, number]) / count[:, None, None]   # 2 x places x width
        grown = (measured['log_total'][label][:, :, number] - original['log_base'][label][:, :, number]) / count[:, None]
        mean = tensor(q[0])
        beyond = ((bounds.gap - mean).norm(dim=1) - bounds.radius).clamp(min=0) * bounds.asked
        shared = asked * (q[0] * along).sum(-1)
        q[0] = q[0] - shared[:, None] * along
        # |z - a g - centre|^2 = |z - centre|^2 - 2 a (z - centre).g + a^2, the gap's direction g being of length one where a move is asked
        distance = measured['apart'][:, :, number] + (group == 0)[:, None] * (shared ** 2 - 2 * shared * measured['along_gap'][:, :, number])
        moved = dict(direction=(q ** 2).sum(-1), size=grown ** 2,
                     spread=np.stack([(distance[each] - original['r'][each][:, :, number]).mean(0) ** 2 for each in of]),
                     size_spread=np.stack([(measured['log_apart'][each][:, :, number] - original['t'][each][:, :, number]).mean(0) ** 2 for each in of]))
        budget = {family: 2 * original['variance'][family][:, :, number] / count[:, None] for family in moved}
        raised = None if floors is None else {family: int((floors[number][family] > budget[family]).sum()) for family in moved}
        if floors is not None:
            budget = {family: np.maximum(value, floors[number][family]) for family, value in budget.items()}
        over = {family: tensor(moved[family] / budget[family].clip(min=TINY)) for family in moved}
        over.update({family: tensor(measured['prompts'][pick][:, number, k]) / question_bounds[family].clamp(min=TINY) for k, family in enumerate(('question', 'question_size'))})
        readings.append(dict(tube=float((beyond ** 2).sum()), inside=int(((beyond == 0) & (bounds.asked > 0)).sum()), mean=mean, shared=tensor(shared), over=over, moved=moved,
                             budgets_raised_to_the_original_files_second_answers=raised,
                             row_by_row_median=dict(direction=[float(np.median(measured['rows'][each][:, :, number].mean(0))) for each in of],
                                                    size=[float(np.median(measured['row_sizes'][each][:, :, number].mean(0))) for each in of])))
    return readings


def own_frames(run, members, index):
    """The merges, the picks of answers a model's own answers are read on, each pick's frozen frames at every merge, and the gap's direction at every
    place and merge. The answers no fit used decide; the other two picks are read beside them."""
    fitting, _ = picks(members)
    merges = [(layer, kind) for layer in index['layers'] for kind in MERGES]
    picked = {'the answers no fit used': ~fitting, 'the fitting answers': fitting, 'all the answers': np.ones(len(members), bool)}
    frames = {label: [bounds_of(run, kind, layer, pick) for layer, kind in merges] for label, pick in picked.items()}
    return merges, picked, frames, np.stack([bounds.along.cpu().numpy() for bounds, _ in frames['all the answers']], 1)


def own_told(merges, frames, readings):
    """A model's own answers on one pick, in plain numbers: every merge, and over the merges."""
    rows = [dict(layer=layer, kind=kind, tube_loss=reading['tube'], tube_loss_of_the_original=bounds.whole, holds_every_bound=outside([reading], OWN) == 0, **summary(reading, bounds, OWN),
                 budgets_raised_to_the_original_files_second_answers=reading['budgets_raised_to_the_original_files_second_answers'], row_by_row_median=reading['row_by_row_median'])
            for (layer, kind), (bounds, _), reading in zip(merges, frames, readings)]
    return dict(every_bound_holds=all(row['holds_every_bound'] for row in rows), merges_with_a_bound_outside=[f"{row['kind']}-{row['layer']}" for row in rows if not row['holds_every_bound']],
                tube_loss_left=sum(row['tube_loss'] for row in rows) / sum(row['tube_loss_of_the_original'] for row in rows),
                asked_places_left_outside_their_ball=sum(row['places_asked'] - row['places_inside_the_ball'] for row in rows), merges=rows)


def along_the_gap(sums, start, frames, count):
    """The wrong pattern's movement along the gap where a move is asked, as a part of the gap, over every place and merge together."""
    asked = np.stack([bounds.asked.cpu().numpy() for bounds, _ in frames], 1)[..., None]
    gap = np.stack([bounds.gap.cpu().numpy() for bounds, _ in frames], 1).astype('f8') * asked
    return float(((sums - start) * gap).sum() / (gap ** 2).sum() / count)


def own_start(run):
    """What a chain of sweeps' own answers are read against, made once in its first sweep from the original file alone. No edited model enters,
    whenever this is made.

    The original file answers the questions again, on its own; or, where the sweep's declaration names an earlier run under
    original_observations_from, the second answers that run keeps are read, as they are, and nothing is answered again: they are what the original
    file answered, whatever is counted from them. Everything below is counted here from those captures and this sweep's own reference, and written
    into this sweep's folder; nothing of the earlier run's record of them is taken over, and nothing is written into its folder.

    Its first answers are the recordings every pattern was made from. How far its second
    answers lie from the first, in each quantity a model's own answers are held to and on each pick, is the floor of that quantity's budget
    (own/original-floors.npz): what the original model's own answers do not repeat, carried through a free answer and the same registration, is not
    held against an edited model. The second answers have to be made as the recordings were, by what the execution records of both carry: the card and
    its driver, the three recorded settings of the environment, what is kept, the kind of run, the checkpoint's size and the model. The recorder itself
    may differ from the recordings' only as the run's declaration says under recorder_revision (the two programs by their sha256, and why answers
    with nothing supplied are read the same by both); any other recorder is refused, and both are on record. The record also says how many second
    answers come out token for token as recorded, and the tube loss they read, which is how much tube loss moves with nothing changed. The four
    variances the budgets are made of are written here too (own/original-variances.npz), with the sha256 of the model that answered.

    Every second answer has to be whole as its recording was (`endings`): ended by itself within the recording's limit of tokens, or, where the
    recording itself ran into the limit, that or run into the same limit, with every decision kept. No floor is counted from a set that has an answer
    which is not; the captures are kept as they are and the command ends saying so. Whether every answer ended by itself is on record as the plain
    fact, apart from that. The record of an earlier attempt that was not whole is kept beside this one (own/start.stopped-N.json), never written over."""
    declaration = declared(run)
    if anchor(run) != run:
        raise ValueError('What a chain of sweeps is read against is made once, in its first sweep')
    _, members, index = ladder.cohort(run)
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    captures = observed(run)/'own'/'original'
    (run/'own').mkdir(exist_ok=True)
    if observed(run) == run:
        if any(not (captures/m['id']/'answer.json').exists() for m in members):
            answered(ORIGINAL_MODEL, run, 'original')
            save(captures/'answered-by.json', dict(model=str(ORIGINAL_MODEL.relative_to(ROOT)), sha256=digest(ORIGINAL_MODEL)))
        save(captures/'registered.json', registered(captures, members))
    elif any(not (captures/m['id']/name).exists() for m in members for name in ('answer.json', 'pairs-axis.npy', 'pairs-original.npy')):
        raise ValueError(f'{captures.relative_to(ROOT)} does not hold every second answer of the original file, laid on the axis')
    answered_by = json.loads((captures/'answered-by.json').read_text())
    if answered_by['sha256'] != declaration.get('model_sha256') or answered_by['sha256'] != digest(ORIGINAL_MODEL):
        raise ValueError('The second answers were not given by the original file this run declares')
    wrote = [json.loads((captures/m['id']/'answer.json').read_text()) for m in members]
    ended = endings(captures, members)
    cut = ended['answers_not_whole_as_their_recording_was']
    earlier = run/'own'/'start.json'
    if earlier.exists() and not json.loads(earlier.read_text()).get('complete'):   # an earlier attempt that was not whole: its record is kept as it is
        earlier.rename(earlier.with_name(f"start.stopped-{len(list(earlier.parent.glob('start.stopped-*.json'))) + 1}.json"))
    # how the second answers were made, against how the recordings were
    made = lambda folder: [json.loads(path.read_text()) for path in sorted(folder.glob('*/execution.json'))]
    said = lambda records, field: sorted({json.dumps(record.get(field), sort_keys=True) for record in records})
    then, now = made(WIDE/'capture'), made(captures)
    differs = [field for field in ('hardware', 'environment', 'retention', 'kind', 'checkpoint_bytes', 'model_sha256', 'observer_sha256') if said(then, field) != said(now, field)]
    kept_there = sorted(path for m in members for path in (captures/m['id']).iterdir() if path.is_file())
    record = dict(how=own_start.__doc__, answers=len(members), answered_by=answered_by, answers_read_from=str(captures.relative_to(ROOT)),
                  answers_sha256=hashlib.sha256(''.join(f'{path.relative_to(captures)} {digest(path)}\n' for path in kept_there).encode()).hexdigest(),
                  reference_sha256=reference_sha256(run), **ended,
                  earlier_attempts_kept=sorted(path.name for path in earlier.parent.glob('start.stopped-*.json')),
                  the_original_files_second_answers_token_for_token_the_recorded_ones=sum(a['tokens'] == m['recorded'] for a, m in zip(wrote, members)),
                  made_differently_from_the_recordings=differs, recorder_then=said(then, 'observer_sha256'), recorder_now=said(now, 'observer_sha256'))
    revision = declaration.get('recorder_revision') or {}
    if 'observer_sha256' in differs:   # only the pair of recorders the declaration names, with its reason, is taken as reading these answers the same
        if (said(then, 'observer_sha256'), said(now, 'observer_sha256')) != ([json.dumps(revision.get('recordings_made_with'))], [json.dumps(revision.get('this_run_reads_with'))]):
            differs.append('a recorder the declaration does not name')
        record.update(recorder_revision=revision)
    if [field for field in differs if field != 'observer_sha256'] or cut:
        save(run/'own'/'start.json', dict(record, complete=False))
        raise SystemExit('The original file\'s second answers are not a repeat of the recordings (' + ', '.join(differs + [f"{len(cut)} not whole as their recording was"] * bool(cut)) + '): no floor is made from them.')
    merges, picked, frames, along = own_frames(run, members, index)
    numbers, wrong = list(range(len(merges))), np.array([not m['correct'] for m in members])
    original = own_original(members, numbers, places, picked)
    np.savez(run/'own'/'original-variances.npz', **original['variance'])
    measured = own_measured(captures, members, numbers, places, picked, original, along)
    floors, second = {}, {}
    for k, (label, pick) in enumerate(picked.items()):
        readings = own_read(measured, original, members, label, pick, frames[label])
        count = np.array([int((pick & flag).sum()) for flag in (wrong, ~wrong)])
        floors.update({f'{k}-{family}': np.stack([reading['moved'][family] for reading in readings], -1) for family in OWN[:4]})   # each 2 x places x merges
        second[label] = dict(answers_in_each_mean=count.tolist(), tube_loss_left=sum(reading['tube'] for reading in readings) / sum(bounds.whole for bounds, _ in frames[label]),
                             cells_where_the_floor_is_above_the_sampling_term={family: int((floors[f'{k}-{family}'] > 2 * original['variance'][family] / count[:, None, None]).sum()) for family in OWN[:4]},
                             cells=int(floors[f'{k}-direction'].size))
    np.savez(run/'own'/'original-floors.npz', **floors)
    del measured
    record.update(picks=list(picked), the_original_files_second_answers=second, variances_sha256=digest(run/'own'/'original-variances.npz'), floors_sha256=digest(run/'own'/'original-floors.npz'),
                  model_sha256=declaration.get('model_sha256'), complete=True)
    save(run/'own'/'start.json', record)
    print(json.dumps({key: value for key, value in record.items() if key != 'how'}), flush=True)


def own(run):
    """The test of a sweep: the finished model answers the questions on its own, and the traces of those answers are read against the original model's.

    Nothing is supplied but the prompts. Each answer is laid on the frozen axis by its computation alone (`registered`) and read at every one of the 105
    merges (`own_measured`, `own_read`): the tube on the wrong pattern's whole movement from the original model; each group's pattern and the spread of
    its members around the original centre, for the state and for its size, each against its own budget from the original answers alone; each question's
    prompt rows against theirs. The patterns, the budgets with their floors, the groups and the paths of the original answers are fixed from the
    original model alone and written once (`reference`, `own_start`). No text, token or score of an answer enters, and how far an answer lies row by row from the original's is on
    record without bounding anything: another answer of the same group lies about twice the group's variance away.

    The model's own answers are read whenever the model is finished, whatever its replay read and whether it has been read: the replay is a verdict
    of its own, not a step that has to come first, and what the two say together is `verdict`'s. The own answers hold if every answer is whole as
    its recording was (`endings`: ended by itself within the recording's limit of tokens, or, where the recording itself ran into the limit, that or
    run into the same limit, with every decision kept; whether every answer ended by itself is on record as the plain fact beside it),
    every budget and prompt bound holds at every merge on the answers no fit used, and their tube loss, summed over the merges, is less than that of
    the model the sweep started from, read the same way. That is progress; the repair of the pattern is whole only when no asked place is left outside
    its ball as well. It is a statement about the recurring pattern of the
    trace on these answers, not that every single answer is right. Beside it, deciding nothing: the same reading on the fitting answers and on all the
    answers, each with the count of its own means; the tube loss the original file's second answers read, which is how much tube loss moves with nothing
    changed; and how much of the gap the wrong pattern moved along."""
    declaration = declared(run)
    first = anchor(run)
    start = json.loads((first/'own'/'start.json').read_text())
    if not start['complete']:
        raise ValueError('What a sweep\'s own answers are read against is not whole (own/start.json of the first sweep): nothing is judged from it')
    _, members, index = ladder.cohort(run)
    places = json.loads((ladder.CORRESPONDENCE/'members.json').read_text())['axis_steps']
    answered(model_of(run), run, 'edited')
    paths = registered(run/'own'/'edited', members)
    merges, picked, frames, along = own_frames(run, members, index)
    numbers, wrong = list(range(len(merges))), np.array([not m['correct'] for m in members])
    if list(picked) != start['picks'] or digest(first/'own'/'original-variances.npz') != start['variances_sha256'] or digest(first/'own'/'original-floors.npz') != start['floors_sha256']:
        raise ValueError('The variances and floors of the original answers are not the ones own-start wrote')
    original = own_original(members, numbers, places, picked)
    written, floors = np.load(first/'own'/'original-variances.npz'), np.load(first/'own'/'original-floors.npz')
    if any(not np.array_equal(written[family], value) for family, value in original['variance'].items()):
        raise ValueError('The original answers do not give the variances own-start wrote')
    measured = own_measured(run/'own'/'edited', members, numbers, places, picked, original, along)
    answers = [json.loads((run/'own'/'edited'/m['id']/'answer.json').read_text()) for m in members]
    earlier, out = declaration.get('starts_from'), {}
    for k, (label, pick) in enumerate(picked.items()):
        given = [{family: floors[f'{k}-{family}'][:, :, number] for family in OWN[:4]} for number in numbers]
        told = own_told(merges, frames[label], own_read(measured, original, members, label, pick, frames[label], given))
        started = json.loads((ROUND_TWO_RUNS/earlier/'own.json').read_text())['readings'][label]['edited']['tube_loss_left'] if earlier else 1.0   # the original model's is the whole of it
        out[label] = dict(edited=told, answers_in_each_mean=[int((pick & wrong).sum()), int((pick & ~wrong).sum())], tube_loss_left_where_the_sweep_started=started,
                          tube_loss_left_of_the_original_files_second_answers=start['the_original_files_second_answers'][label]['tube_loss_left'],
                          wrong_pattern_movement_along_the_gap_over_the_gap=along_the_gap(measured['total'][label][0], original['base'][label][0], frames[label], int((pick & wrong).sum())))
        print(json.dumps({label: {key: value for key, value in told.items() if key != 'merges'}}), flush=True)
    decides = out['the answers no fit used']
    ended = endings(run/'own'/'edited', members)
    holds = bool(ended['every_answer_whole_as_its_recording_was'] and decides['edited']['every_bound_holds'] and decides['edited']['tube_loss_left'] < decides['tube_loss_left_where_the_sweep_started'])
    save(run/'own.json', dict(how=own.__doc__, counted_from=str(first.relative_to(ROOT)), model_sha256=digest(model_of(run)), reference_sha256=reference_sha256(run),
                              variances_sha256=start['variances_sha256'], floors_sha256=start['floors_sha256'],
                              answers_no_fit_used=[m['id'] for m, take in zip(members, picked['the answers no fit used']) if take], merges_read=len(merges), answers=len(answers),
                              **ended, paths=paths, readings=out,
                              own_answers_hold=holds, every_bound_holds=decides['edited']['every_bound_holds'], tube_loss_left=decides['edited']['tube_loss_left'],
                              tube_loss_left_where_the_sweep_started=decides['tube_loss_left_where_the_sweep_started'],
                              tube_loss_left_of_the_original_files_second_answers=decides['tube_loss_left_of_the_original_files_second_answers'],
                              asked_places_left_outside_their_ball=decides['edited']['asked_places_left_outside_their_ball']))
    verdict(run)


if __name__ == '__main__':
    if sys.argv[1] not in ('start', 'verdict'):   # those two do nothing on the card
        CARD.take()
    {'start': start, 'baseline': baseline, 'reproduced': reproduced, 'reference': reference, 'own-start': own_start, 'build': build, 'composed': composed, 'own': own, 'verdict': verdict}[sys.argv[1]](ROUND_TWO_RUNS/sys.argv[2])
