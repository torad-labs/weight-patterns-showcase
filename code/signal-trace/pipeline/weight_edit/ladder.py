"""The ladder's edit: one construction for every rung, read in the trace and carried to the weights by the model's own math.

From code/signal-trace:
  uv run -q --with torch --with numpy python -m pipeline.weight_edit.ladder states|baseline|ordinary|build|own|overlay|travel <run>

What is asked, at every matrix and every row (a row is one position of one answer's own computation):
  the direction the state had right after that matrix's merge in the original model, plus the shared part of the gap
  between the correct and the wrong pattern of that state where the two diverge. Rows where the patterns follow each
  other, and the positions of the question, are asked to keep the state they had. The rungs differ only in whose rows
  are fitted.

  The target is the state after the merge, not the turn the merge makes. The turns of successive merges write and
  take back, and part of what a matrix turns answers inputs that enter at every layer and that no weight changes; the
  shared turn gaps, each delivered in full, add up to a state further from the correct pattern than no edit from the
  middle layers on (state-overlap keeps that record). The state after the merge is what the next matrix reads.

How it reaches the weights:
  a matrix's output y is normed, scaled by the learned scale g and added to the state h; the turn is
  unit(h + g*y/rms(y)) - unit(h). A change dW of the matrix changes the turn of row r by A_r(x_r dW) to first order,
  where A_r is the exact derivative of that expression. For a given ridge the change is the one that leaves the least of
  mean |wanted turn - turn at the changed weights|^2 + ridge |dW|^2 over the fitted rows, the turn read by the model's
  own expression: conjugate gradients on that expression, with A_r giving the gradient and the size of each step, and
  every step kept only once the expression itself confirms it. The ridge is the one whose edit, fitted on half the
  answers and rounded to the model's BF16, leaves the least on the other half; leaving the matrix as it is competes.

The path:
  the matrices are solved in path order. Before each one, the answers' own text goes through the model as it stands,
  with every earlier edit in it, and that matrix's inputs and incoming state are read. So each edit sees what the edits
  before it did, and is asked for what is still missing at its own merge, what the matrices before it left included. These passes build the edit. They are not its test: the test
  is the trace of the edited model's own answers.

states    The overlap of the states (its own declared run): at each of the 70 matrices, the direction of the state right
          after its merge, for every generated step of the 60 answers, laid on the axis by the registered
          correspondence: the two patterns, their gap, its chance and the shared part.
baseline  The same pass on the original model, for every matrix: each row's original state after the merge, read the way
          every later pass reads it, so that a row asked to keep its state is never asked to undo a difference between
          two ways of running the model. Beside it, how far these are from the rows recorded while the model generated
          the text token by token.
build     The edit, matrix by matrix along the path, into one copy of the model.
own       The edited model answers the 60 questions on its own again, with the turns of its computation recorded.
overlay   Those traces laid along the overlap's axis by the computation alone, every step kept, and set against the
          original correct and wrong patterns: do the answers that were wrong now follow the correct pattern where the
          two diverged, do they keep their own pattern elsewhere, and how far did the answers that were correct move.
          This is the test of an edit.
ordinary  The D rung's background, before its build: the original model answers the run's ordinary requests on its own.
          The build then weighs every matrix's inputs by what those requests send into it (weighing, below).

Removed on November 14, 2026: the commands that read an edit by its answers (answer, score, steps, spill, parting) and the C rung's
build (cancel), whose rows ended at the first token an answer changed. An edit is read by its trace, and nothing here measures by
tokens, values or words. The records those commands wrote stay in their runs as history, and git keeps the code.
"""
import json
import shutil
import sys
import time

import numpy as np
import torch

from pipeline.acquisition.export_stream_traces import StreamRows
from pipeline.native_records.native_trace import checkpoint_values
from pipeline.native_records.run_declaration import digest
from pipeline.overlap.complete_overlap import member_means, native_targets
from pipeline.overlap.sent_overlap import matrices
from pipeline.paths import ORIGINAL_MODEL, ROOT, ROUND_TWO_RUNS
from pipeline.weight_edit.a2 import NORM_EPSILON, from_bf16, save, to_bf16
from pipeline.weight_edit.a2_pattern import KINDS, WIDE, model_of

INPUTS = ROUND_TWO_RUNS/'input-overlap'/'inputs'
TENSORS = ROUND_TWO_RUNS/'sent-overlap'/'wrong-inputs'/'tensors.json'
# which step of each answer lies at which place of the axis, and the overlap of the turns that the test reads the model's own traces against
CORRESPONDENCE = ROUND_TWO_RUNS/'trace-overlap-complete-registered'
# run -> whose rows are fitted. The rest of the construction is the same for every rung.
RUNGS = {'a2-path': 'the wrong answers', 'b2-path': 'the wrong answers, and the correct answers a2-path broke, held where they were',
         'd2-path': 'the wrong answers, with the size of the change measured by what it does to the inputs of ordinary requests'}
DEVICE = torch.device('cuda:0')  # the solver's card is whichever CUDA_VISIBLE_DEVICES names; the model's passes always run on GPU0, through the recorder
EPS = float(torch.finfo(torch.float32).eps)  # the arithmetic of the solver and of the recorded rows


def cohort(run):
    """The answers in the correspondence's order, with their tokens and where their rows sit in the stored files; and the overlap the run takes its target from."""
    declaration = json.loads((run/'declaration.json').read_text())
    source = (ROOT/declaration['target_source']).parent if declaration.get('target_source') else None
    members = json.loads((CORRESPONDENCE/'members.json').read_text())['members']
    index = json.loads((INPUTS/'index.json').read_text())
    if [m['id'] for m in members] != [a['id'] for a in index['answers']]:
        raise ValueError('The correspondence and the recorded inputs list different answers')
    wide = {m['id']: m for m in json.loads((WIDE/'members.json').read_text())}
    for m, a in zip(members, index['answers']):
        m.update(steps=a['steps'], question=a['question_positions'], tokens=wide[m['id']]['tokens'], recorded=wide[m['id']]['recorded_tokens'])
        if len(m['tokens']) != m['question'] + 1 or len(m['recorded']) != m['steps']:
            raise ValueError(f"{m['id']}: the recorded rows do not match the answer's tokens")
    return source, members, index


def baseline_of(run):
    """Where the original model's states after each matrix's merge are kept: read once, by the first rung, and declared by the others."""
    declared = json.loads((run/'declaration.json').read_text()).get('baseline')
    return ROOT/declared if declared else run/'baseline'


def held_of(run):
    """The correct answers a rung holds where they were: those the rung before it is measured to break, by its own record. None for the A rung."""
    source = json.loads((run/'declaration.json').read_text()).get('protected_source')
    if not source:
        return set()
    return {r['id'] for r in json.loads((ROOT/source).read_text()) if r['set'] == 'overlap' and r['group'] == 'correct' and not r['edited']['complete']}


def depth_of(run, order):
    """The matrices a rung edits: those of the A rung it follows, by that rung's record; all of them when none is declared."""
    source = json.loads((run/'declaration.json').read_text()).get('matrices_source')
    return order[:json.loads((ROOT/source).read_text())['matrices_edited']] if source else order


def labels_of(layer, kind):
    matrix_input, _, _, incoming, *_rest = KINDS[kind]
    state = ('inp_scaled' if layer == 0 else f'l_out-{layer - 1}') if incoming is None else f'{incoming}-{layer}'
    return f'{matrix_input}-{layer}', state


def stored(members, index, layer, kind):
    """The rows input-overlap recorded while the original model generated each answer: one matrix's input and the state it adds to,
    at every answer step, then at every question position."""
    out = []
    for what in ('x', 'state'):
        width = index['width'][f'{kind}-{layer}'] if what == 'x' else 1536
        blocks = [np.memmap(INPUTS/f'{prefix}{what}-{kind}-{layer}.f32', dtype='<f4', mode='r').reshape(-1, width) for prefix in ('', 'question-')]
        if blocks[0].shape[0] != sum(m['steps'] for m in members) or blocks[1].shape[0] != sum(m['question'] for m in members):
            raise ValueError(f'{what}-{kind}-{layer}: stored rows do not match the listed answers')
        out.append(np.concatenate(blocks))
    return out


def passed(model, run, name, wanted, members):
    """The answers' own text through the model as given text. Returns each wanted output at every answer step, then at every question position."""
    from pipeline.acquisition.run_a_observation import run as observe
    raw = run/'passes'/name
    cases = run/'passes'/f'{name}-cases.json'
    cases.parent.mkdir(parents=True, exist_ok=True)
    # The decision for generated token s is computed at position question + s, whose input is generated token s - 1.
    cases.write_text(json.dumps([dict(id=m['id'], tokens=m['tokens'] + m['recorded'][:-1], max_tokens=1, stream_labels='|'.join(wanted)) for m in members]))
    observe(model, cases, raw, 0, 'conditional', 'capture', 'projection', run/'declaration.json')
    answer, question = {label: [] for label in wanted}, {label: [] for label in wanted}
    for m in members:
        count = m['question'] + m['steps']
        stream = StreamRows(raw/m['id'], set(range(count)))
        for label in wanted:
            rows = np.stack([stream.values(label, p) for p in range(count)])
            question[label].append(rows[:m['question']])
            answer[label].append(rows[m['question']:])
        stream.close()
        shutil.rmtree(raw/m['id'])  # the rows are in memory; the raw stream is not kept
    cases.unlink()
    return [np.concatenate(answer[label] + question[label]) for label in wanted]


def moves(source, position, members):
    """For every answer step, the shared part of the gap in the state after this matrix's merge, where the two patterns diverge. Nothing for a correct answer."""
    z = np.load(source/'recurrence.npz')
    gap, chance = z['gap'][:, position], z['chance'][:, position]
    share = np.clip(1 - chance / np.where(gap > 0, gap, np.inf), 0, 1)
    diverged = gap > 2 * chance  # the two groups are further apart here than two halves of one group are
    d = np.load(source/'correct-pattern.npy', mmap_mode='r')[:, position].astype('f8') - np.load(source/'wrong-pattern.npy', mmap_mode='r')[:, position].astype('f8')
    place = (share * diverged)[:, None] * d
    out = [np.zeros((m['steps'], place.shape[1])) if m['correct'] else native_targets(place, np.load(CORRESPONDENCE/'pairs'/f"{m['id']}.npy"), m['steps']) for m in members]
    return np.concatenate(out).astype('f4'), float(diverged.mean())


def states(run):
    """The overlap of the states: at each of the 70 matrices, the direction of the state right after its merge, at every generated
    step of the 60 answers as input-overlap recorded them, laid on the axis by the registered correspondence.

    For each place and matrix: the correct and the wrong pattern, their gap, its chance, how alike the gaps of two halves
    are, and the shared part, by the formulas of the overlap of the turns. Beside them, as a record of why the target is
    the state: what the shared turn gaps of the registered overlap add up to at the same boundary and what they would
    leave of the state gap if each were delivered in full, as declared and with the first merge taken without the gap
    of the token embeddings; and that gap of the embeddings itself.

    The state is rebuilt from the matrix's recorded input as the engine multiplied it (tube.multiplied): at each answer's
    first step, which the engine decided at the prompt's last position while it read the prompt at once, the input is
    rounded to the file's BF16, and at every later step it is as recorded. That is the engine's own output there to the
    last digits of float32. The first state overlap (state-overlap, November 10) took the input as recorded at every step
    and stays as it is (Astra, records/retrospectives/2026-11-11-native-reference-decision-astra.md)."""
    from pipeline.weight_edit.tube import multiplied
    if (run/'recurrence.npz').exists():
        raise ValueError('Inspect the existing overlap; never overwrite it')
    _, members, index = cohort(run)
    tensors = json.loads(TENSORS.read_text())
    record = json.loads((CORRESPONDENCE/'members.json').read_text())
    places, order = record['axis_steps'], list(matrices(index['layers']))
    steps = np.cumsum([0] + [m['steps'] for m in members])
    correct = np.array([m['correct'] for m in members])
    even = np.zeros(len(members), dtype=bool)
    for selected in (correct, ~correct):  # alternating within each group, as in the overlap of the turns
        even[np.flatnonzero(selected)[::2]] = True
    pairs = [np.load(CORRESPONDENCE/'pairs'/f"{m['id']}.npy") for m in members]
    patterns = {name: np.lib.format.open_memmap(run/f'{name}-pattern.npy', mode='w+', dtype='f4', shape=(places, len(order), 1536)) for name in ('correct', 'wrong')}
    gap, chance, halves, left_declared, left_corrected, summed = (np.zeros((places, len(order))) for _ in range(6))

    def overlap(rows):
        means = np.stack([member_means(rows[steps[i]:steps[i + 1]], pairs[i], places) for i in range(len(members))])
        groups = means[correct], means[~correct]
        d = groups[0].mean(0) - groups[1].mean(0)
        d1 = means[correct & even].mean(0) - means[~correct & even].mean(0)
        d2 = means[correct & ~even].mean(0) - means[~correct & ~even].mean(0)
        size = np.linalg.norm(d1, axis=-1) * np.linalg.norm(d2, axis=-1)
        return groups[0].mean(0), groups[1].mean(0), (d * d).sum(-1), sum(np.var(group, axis=0, ddof=1).sum(-1) / len(group) for group in groups), \
            np.divide((d1 * d2).sum(-1), size, out=np.zeros(places), where=size > 0)
    turns = np.load(CORRESPONDENCE/'recurrence.npz')
    turn_share = np.clip(1 - turns['chance'] / np.where(turns['gap'] > 0, turns['gap'], np.inf), 0, 1) * (turns['gap'] > 2 * turns['chance'])
    turn_patterns = [np.load(CORRESPONDENCE/f'{name}-pattern.npy', mmap_mode='r') for name in ('correct', 'wrong')]
    sums = {}
    region = slice(142, 314)
    rms = lambda squared: float(np.sqrt(squared[region].mean()))
    for position, layer, kind in order:
        _, w, g = weights_of(tensors, layer, kind)
        x, h = (torch.from_numpy(np.ascontiguousarray(v[:steps[-1]])).to(DEVICE) for v in stored(members, index, layer, kind))
        ca, cb, gap[:, position], chance[:, position], halves[:, position] = overlap(Merge(multiplied(x, [m['steps'] for m in members], 'as generated'), h, w, g).to.cpu().numpy())
        patterns['correct'][:, position], patterns['wrong'][:, position] = ca, cb
        d = ca - cb
        merge = 3 * layer + KINDS[kind][-1]
        asked = turn_share[:, merge, None] * (turn_patterns[0][:, merge].astype('f8') - turn_patterns[1][:, merge].astype('f8'))
        if position == 0:
            if merge != 0:
                raise ValueError('The first matrix on the path is not the first merge')
            _, _, embedding_gap, embedding_chance, _ = overlap((h / h.norm(dim=1, keepdim=True)).cpu().numpy())
            first = np.clip(1 - chance[:, 0] / np.where(gap[:, 0] > 0, gap[:, 0], np.inf), 0, 1) * (gap[:, 0] > 2 * chance[:, 0])
            sums = dict(declared=asked, corrected=first[:, None] * d)
        else:
            sums = {name: value + asked for name, value in sums.items()}
        summed[:, position] = (sums['declared'] ** 2).sum(-1)
        left_declared[:, position] = ((d - sums['declared']) ** 2).sum(-1)
        left_corrected[:, position] = ((d - sums['corrected']) ** 2).sum(-1)
        diverged = gap[:, position] > 2 * chance[:, position]
        print(json.dumps(dict(position=position, layer=layer, kind=kind, places_142_313=dict(
            state_gap_rms=rms(gap[:, position]), gap_over_chance=float(gap[region, position].sum() / chance[region, position].sum()),
            places_diverged=int(diverged[region].sum()), halves_alike_median=float(np.median(halves[region, position])),
            shared_turn_gaps_summed_rms=rms(summed[:, position]), state_gap_left_if_they_were_delivered_rms=rms(left_declared[:, position]),
            the_same_with_the_first_merge_taken_without_the_embeddings_rms=rms(left_corrected[:, position])),
            all_places=dict(gap_over_chance=float(gap[:, position].sum() / chance[:, position].sum()), places_diverged=int(diverged.sum())))), flush=True)
        del x, h, w, g
        torch.cuda.empty_cache()
    for pattern in patterns.values():
        pattern.flush()
    share = np.clip(1 - chance / np.where(gap > 0, gap, np.inf), 0, 1)
    np.savez(run/'recurrence.npz', gap=gap, chance=chance, halves=halves, shared_part=share, shared_turn_gaps_summed=summed,
             left_if_shared_turn_gaps_delivered=left_declared, left_if_delivered_with_first_merge_without_embeddings=left_corrected,
             embedding_gap=embedding_gap, embedding_chance=embedding_chance)
    save(run/'members.json', dict(axis=record['axis'], axis_steps=places, correspondence=str(CORRESPONDENCE.relative_to(ROOT)),
                                  matrix_input='as the engine multiplied it: rounded to BF16 at each answer\'s first step, as recorded at every later step',
                                  members=[dict(id=m['id'], correct=bool(m['correct'])) for m in members], matrices=[dict(position=p, layer=l, kind=k) for p, l, k in order],
                                  meaning='correct-pattern.npy and wrong-pattern.npy: float32 [place, matrix, 1536], the mean over a group\'s answers of the direction of the state '
                                          'right after that matrix\'s merge, each answer first averaged over its steps laid on the place. recurrence.npz: per place and matrix, '
                                          'the squared gap of the two patterns, its chance, how alike two halves\' gaps are, the shared part; and the record of what the shared '
                                          'turn gaps of the registered overlap add up to there.'))
    print(json.dumps(dict(matrices=len(order), places=places, gap_over_chance=float(gap.sum() / chance.sum()), gap_over_chance_142_313=float(gap[region].sum() / chance[region].sum()),
                          cells_diverged=float((gap > 2 * chance).mean()), median_halves_alike=float(np.median(halves)),
                          embedding_gap_rms=float(np.sqrt(embedding_gap.mean())), first_merge_turn_gap_rms=float(np.sqrt(turns['gap'][:, 0].mean())), first_merge_state_gap_rms=float(np.sqrt(gap[:, 0].mean())))), flush=True)


class Merge:
    """The turn a matrix's output gives each row at the given weights, and the exact first-order action of an output change on that turn."""

    def __init__(self, x, h, w, g):
        y = x @ w
        self.rms = torch.sqrt((y * y).mean(1, keepdim=True) + NORM_EPSILON)
        self.sent = y / self.rms
        merged = h + g * self.sent
        self.size = merged.norm(dim=1, keepdim=True)
        self.to = merged / self.size
        self.turn = self.to - h / h.norm(dim=1, keepdim=True)
        self.g, self.width = g, y.shape[1]
        self.ratio = 1 / (self.size * self.rms)  # how much of an output change reaches the turn, before the two projections

    def forward(self, dy):
        d = (dy - self.sent * (self.sent * dy).sum(1, keepdim=True) / self.width) / self.rms  # the norm takes out a change of size
        d = self.g * d
        return (d - self.to * (self.to * d).sum(1, keepdim=True)) / self.size                 # a change along the merged state turns nothing

    def adjoint(self, v):
        u = (v - self.to * (self.to * v).sum(1, keepdim=True)) / self.size
        u = self.g * u
        return (u - self.sent * (self.sent * u).sum(1, keepdim=True) / self.width) / self.rms


def gram(x, scale):
    """x' diag(scale) x, in blocks of rows."""
    out = torch.zeros((x.shape[1], x.shape[1]), device=x.device)
    for start in range(0, x.shape[0], 8192):
        block = x[start:start + 8192]
        out += block.T @ (scale[start:start + 8192] * block)
    return out


def output_action(merge, weight):
    """What an output change does to the turn, averaged over rows into one matrix over the 1536 outputs.

    For one row it is B = P_y g P_m g P_y: the norm takes out a change of size (P_y), the learned scale weighs each
    output (g), and a change along the merged state turns nothing (P_m). weight says how much each row counts."""
    c = weight / weight.sum()
    y, g2, d = merge.sent, merge.g ** 2, merge.width
    second = y.T @ (c * y)
    u = merge.g * merge.to
    v = u - y * (y * u).sum(1, keepdim=True) / d
    b = torch.diag(g2) - (second * g2[None, :] + g2[:, None] * second) / d + y.T @ (c * (y * y * g2).sum(1, keepdim=True) * y) / d ** 2 - v.T @ (c * v)
    return (b + b.T) / 2


def rounded(w):
    """The weights as the model file keeps them: BF16, by the engine's own rounding."""
    return torch.from_numpy(from_bf16(to_bf16(w.detach().cpu().numpy().ravel())).reshape(w.shape).copy()).to(w.device)


def spectrum(x, merge, s=None):
    """The system over these rows as (its inputs' part) x (its outputs' part), each in its own eigenvectors.

    Exactly, the system is the mean over rows of a_r^2 (x_r' x_r) (x) B_r. Taken as (mean a_r^2 x_r' x_r) (x) (B_r averaged by
    a_r^2 |x_r|^2) it has the same trace over the inputs and a closed-form inverse. That form preconditions the exact solve
    and says where to start looking for the ridge. It is never the edit and never chooses it. With a weighing s the inputs'
    part is taken in the weighed coordinates, s (x' x) s."""
    scale = merge.ratio ** 2
    inputs = gram(x, scale) / x.shape[0]
    if s is not None:
        inputs = s @ inputs @ s
        inputs = (inputs + inputs.T) / 2
    values, vectors = torch.linalg.eigh(inputs)
    del inputs
    out_values, out_vectors = torch.linalg.eigh(output_action(merge, scale * (x * x).sum(1, keepdim=True)))
    return values.clamp(min=0), vectors, out_values.clamp(min=0), out_vectors


def closed(basis, core, ridge):
    values, vectors, out_values, out_vectors = basis
    return vectors @ (core / (values[:, None] * out_values[None, :] + ridge)) @ out_vectors.T


def refine(x, h, w, g, wanted, ridge, start=None, s=None):
    """The change of one matrix that leaves the least over these rows, mean |wanted - state|^2 + ridge |change|^2, with state the
    direction of the state after the merge, read by the model's own expression. (For a fixed incoming state a change of
    the turn and a change of that direction are the same thing, so the first-order operator is the turn's.)

    With a weighing s (symmetric, over the inputs) the change of the matrix is s @ change and the ridge still counts
    |change|^2: the size of a change is then measured by what it does on the inputs the weighing was taken from, not by its
    weights. The matrix's own product x @ w is never rewritten; only the change passes through s. What is returned is
    change, in the coordinates it was solved in.

    Preconditioned conjugate gradients on that expression itself. At every step: the exact gradient at the current
    weights; a step along the conjugate direction, sized by the exact first-order operator there; kept only once the
    model's own expression at the new weights confirms it, and halved until it does. It ends when no step, however
    short, lowers what float32 can state. What it returns says how many steps were taken, how many had to be shortened,
    how it ended and how much of the gradient is left."""
    n = x.shape[0]
    back = (lambda change: change) if s is None else (lambda change: s @ change)

    def at(change):
        merge = Merge(x, h, w + back(change), g)
        return merge, float(((wanted - merge.to) ** 2).sum() / n) + ridge * float((change ** 2).sum())

    def downhill(merge, change):  # minus half the gradient of what is left
        return back(x.T @ merge.adjoint(wanted - merge.to)) / n - ridge * change
    change = torch.zeros_like(w)
    merge, left = at(change)
    gradient_at_no_change = float(downhill(merge, change).norm())
    if start is not None:  # a start that leaves more than no change is not used
        other, value = at(start)
        if value < left:
            change, merge, left = start.clone(), other, value
    basis = spectrum(x, merge, s)  # the preconditioner, built at this call's starting weights and again whenever the conjugate direction fails
    r = downhill(merge, change)
    z = closed(basis, basis[1].T @ r @ basis[3], ridge)
    direction, rz, fresh = z, float((r * z).sum()), True
    steps, shortened, restarts, ended = 0, 0, 0, 'no gradient left'
    while np.isfinite(rz) and rz > 0:
        along = float((r * direction).sum())
        curve = float((merge.forward(x @ back(direction)) ** 2).sum() / n) + ridge * float((direction ** 2).sum())
        step = along / curve if np.isfinite(curve) and curve > 0 and along > 0 else 0.0
        trial, value, halved = merge, left, False
        if step > 0:
            trial, value = at(change + step * direction)
            while not value < left and value != left and float((step * direction).abs().max()) > 0:
                step, halved = step / 2, True
                trial, value = at(change + step * direction)
        if not value < left:
            if fresh:
                ended = 'no step along the gradient lowers what float32 can state'
                break
            # the conjugate direction failed: the preconditioner is rebuilt at the weights as they are, and the plain preconditioned gradient is tried before ending
            basis = spectrum(x, merge, s)
            z = closed(basis, basis[1].T @ r @ basis[3], ridge)
            direction, rz, fresh, restarts = z, float((r * z).sum()), True, restarts + 1
            continue
        change, merge, left = change + step * direction, trial, value
        steps, shortened = steps + 1, shortened + halved
        r_new = downhill(merge, change)
        z_new = closed(basis, basis[1].T @ r_new @ basis[3], ridge)
        beta = max(0.0, float((r_new * (z_new - z)).sum()) / rz)
        r, z, rz = r_new, z_new, float((r_new * z_new).sum())
        direction, fresh = z + beta * direction, beta == 0
    return change, dict(steps=steps, steps_shortened=shortened, restarts=restarts, ended=ended,
                        gradient_left_over_gradient_at_no_change=float(r.norm()) / gradient_at_no_change if gradient_at_no_change else 0.0, left_and_ridge=left)


def balance(folds, w, g, s=None):
    """The ridge at which a fit on one half of the fitted answers leaves the least on the other half, or None when no change leaves less.
    With a weighing s the fits run in its coordinates; what is rounded and judged is always the matrix itself.

    The candidate at each ridge is the edit as it would be applied: each half fitted by refine, rounded to BF16, and read on
    the other half with the model's own expression. The search starts at the system's strongest direction, where the edit
    is small, and walks by factors of two towards a weaker ridge while that leaves less (towards a stronger one if the
    first weaker step does not), each fit started from the one before it. It then narrows between the neighbours of the
    best by golden section, until its two inner candidates differ by less than rounding to BF16 moves that same number:
    a rule to end the search, not a statement that the candidates are the same. What is returned is the best candidate
    evaluated, not a global minimum; every candidate is recorded with how its fits ended."""
    nothing = sum(float(((wanted - Merge(x, h, w, g).to) ** 2).sum()) for x, h, wanted in folds)
    if nothing == 0:
        return None, dict(candidates=[], selected='no change: nothing is asked of this matrix'), None
    x, h, _ = folds[0]
    basis = spectrum(x, Merge(x, h, w, g), s)
    top = float(basis[0].max() * basis[2].max())
    del basis
    limit = int(np.floor(-np.log2(EPS)))  # past this float32 cannot tell the ridge from none beside the strongest direction
    tried = {}

    def at(k):
        if k not in tried:
            started = time.time()
            near = [change.to(w.device) for change in tried[min(tried, key=lambda other: abs(other - k))]['changes']] if tried else [None, None]
            left, plain, changes, sizes, fits = 0.0, 0.0, [], [], []
            for i, (x, h, wanted) in enumerate(folds):
                change, status = refine(x, h, w, g, wanted, top * 2.0 ** -k, near[i], s)
                actual = change if s is None else s @ change
                realized = rounded(w + actual)
                ox, oh, owanted = folds[1 - i]
                left += float(((owanted - Merge(ox, oh, realized, g).to) ** 2).sum())
                plain += float(((owanted - Merge(ox, oh, w + actual, g).to) ** 2).sum())
                changes.append(change.cpu())  # a candidate's change waits in main memory; only the one being fitted is on the card
                sizes.append(realized - w)
                del change, actual, realized
                fits.append({name: status[name] for name in ('steps', 'steps_shortened', 'restarts', 'ended', 'gradient_left_over_gradient_at_no_change')})
            norms = [float(size.norm()) for size in sizes]
            tried[k] = dict(changes=changes, order=len(tried), left=left / nothing, rounding=abs(left - plain) / nothing, unchanged=max(norms) == 0,
                            change_over_matrix=float(np.mean(norms)) / float(w.norm()),
                            halves_agree=float((sizes[0] * sizes[1]).sum()) / (norms[0] * norms[1]) if min(norms) > 0 else None,
                            fits=fits, seconds=round(time.time() - started, 1))
            print(json.dumps(dict(candidate_ridge_over_strongest=2.0 ** -k, **{name: value for name, value in tried[k].items() if name != 'changes'})), file=sys.stderr, flush=True)
            del near, sizes
            torch.cuda.empty_cache()
        return tried[k]['left']
    way = 0
    if at(1) < at(0):
        way = 1
    elif at(-1) < at(0):
        way = -1
    k = way
    while way:
        if way == 1 and k == limit:
            break
        if way == -1 and tried[k]['unchanged']:  # towards a stronger ridge, until the rounded edit is no edit
            break
        if not at(k + way) < at(k):
            break
        k += way
    best = min(tried, key=lambda v: tried[v]['left'])
    if best - 1 in tried and best + 1 in tried:
        golden = (np.sqrt(5) - 1) / 2
        a, b = best - 1.0, best + 1.0
        c, d = b - golden * (b - a), a + golden * (b - a)
        while c < d and abs(at(c) - at(d)) > max(tried[c]['rounding'], tried[d]['rounding']):
            if at(c) < at(d):
                b, d = d, c
                c = b - golden * (b - a)
            else:
                a, c = c, d
                d = a + golden * (b - a)
        best = min(tried, key=lambda v: tried[v]['left'])
    table = [dict(ridge_over_strongest_direction=2.0 ** -v, other_half_left_over_asked=tried[v]['left'],
                  **{name: tried[v][name] for name in ('order', 'rounding', 'change_over_matrix', 'halves_agree', 'fits', 'seconds')}) for v in sorted(tried)]
    if not tried[best]['left'] < 1:
        return None, dict(candidates=table, selected='no change: no candidate leaves less on the other half than the original matrix'), None
    record = dict(candidates=table, ridge_over_strongest_direction=2.0 ** -best, other_half_left_over_asked=tried[best]['left'],
                  halves_agree=tried[best]['halves_agree'], selected='the best candidate evaluated')
    return top * 2.0 ** -best, record, ((tried[best]['changes'][0] + tried[best]['changes'][1]) / 2).to(w.device)


def fit(x, h, w, g, wanted, folds, s=None):
    """The change of one matrix over the fitted rows (x, h, wanted state direction after the merge), or none. folds are the two halves of those rows, by answer.
    s is the weighing of the inputs, when the rung has one; the ridge is then stated in its coordinates."""
    ridge, record, start = balance(folds, w, g, s)
    if ridge is None:
        return torch.zeros_like(w), record
    change, more = refine(x, h, w, g, wanted, ridge, start, s)
    record.update(ridge=ridge, **more)
    return change if s is None else s @ change, record


def ordinary_of(run):
    """The ordinary requests a rung weighs its inputs by, with the original model's own answer to each: its question's tokens and the tokens it wrote.
    Empty for a rung that declares none."""
    declared = json.loads((run/'declaration.json').read_text()).get('ordinary')
    if not declared:
        return []
    out = []
    for case in json.loads((ROOT/declared['cases']).read_text()):
        if case['group'] not in (declared['weighed'], declared['read']):
            continue
        folder = ROOT/declared['original']/case['id']
        tokens = json.loads((folder/'case.json').read_text())['tokens']
        written = [int(step['selected']) for step in json.loads((folder/'result.json').read_text())['steps']]
        out.append(dict(id=case['id'], weighed=case['group'] == declared['weighed'], tokens=tokens, recorded=written, question=len(tokens) - 1, steps=len(written)))
    if not any(m['weighed'] for m in out):
        raise ValueError('No ordinary request is declared to weigh the inputs by')
    return out


def request_rows(requests):
    """Which request each row of a pass over them belongs to: every answer step of every request, then every question position, as passed() returns them."""
    return np.concatenate([np.repeat(np.arange(len(requests)), [m[key] for m in requests]) for key in ('steps', 'question')])


def weighing(x, merge, request_of, weighed):
    """How the D rung measures the size of a change of one matrix: by what it does on the inputs ordinary requests send into that matrix.

    Q is the second moment of the matrix's inputs over the weighed requests, every position of a request counting the
    same, every request counting the same, each row taken with the same gain the fit gives its own rows (how much of an
    output change reaches the turn, merge.ratio). The size of a change dW is then tr(dW' (Q + floor) dW): the mean squared
    output it adds on ordinary text, plus a floor for the directions these requests do not measure.

    Q is an input-side proxy: the model's own expression for the state also applies the learned scale and the two
    projections to the output, which Q does not contain. Whether ordinary text stays where it was is read on the realized
    states and on the model's own answers, never on Q. Q, the floor and the gains are taken once for a matrix and are
    fixed while it is fitted.

    The floor is measured, not chosen: it is the one at which the moment of all the requests but one best predicts the
    request left out, summed over the requests, by the score of an equal-request, zero-mean Gaussian model: the sum over
    directions of log(value + floor) + (the left-out request's energy there) / (value + floor), in the eigenvectors of
    the other requests' moment. The directions the other requests leave empty would alone choose the left-out request's
    mean energy per direction; the directions they do measure move the common value too. The search walks by factors of
    two from the moment's mean eigenvalue while that lowers the sum, then narrows between the neighbours of the best by
    golden section until float64 cannot tell its two inner candidates apart. It returns the best candidate evaluated,
    not a proven optimum. When no positive floor can be stated (an empty moment, a score that is not finite, or a walk
    that reaches the edge of float64 without an interior best) it stops and says so; no stand-in value is used.

    Returns s = sqrt(mean eigenvalue + floor) (Q + floor)^-1/2: a direction carrying the moment's mean eigenvalue is
    left as it is. The scaling only moves the ridge: the coefficient of tr(dW' (Q + floor) dW) in what a fit minimises
    is its ridge divided by (mean eigenvalue + floor), which the record keeps as metric_scale."""
    started = time.time()
    ids = [i for i in range(len(weighed)) if weighed[i]]
    scale = (merge.ratio ** 2).double()
    moments = []
    for i in ids:
        rows = torch.from_numpy(np.flatnonzero(request_of == i)).to(x.device)
        part = x[rows].double()
        moments.append(part.T @ (scale[rows] * part) / len(rows))
        del part
    q = sum(moments) / len(moments)
    held = []
    for moment in moments:  # the others' moment in its own eigenvectors, and the left-out request's energy along each
        values, vectors = torch.linalg.eigh((len(moments) * q - moment) / (len(moments) - 1))
        held.append((values.clamp(min=0), ((vectors.T @ moment) * vectors.T).sum(1).clamp(min=0)))
        del vectors
    del moments
    left = lambda k: float(sum((torch.log(values + mean * 2.0 ** k) + energy / (values + mean * 2.0 ** k)).sum() for values, energy in held))
    mean = float(torch.trace(q)) / q.shape[0]
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError('The ordinary requests send nothing into this matrix: no moment to weigh by')
    edge = int(np.floor(-np.log2(np.finfo(np.float64).eps)))  # past this float64 cannot tell the floor from none beside the mean eigenvalue
    tried = {}

    def at(k):
        if k not in tried:
            tried[k] = left(k)
            if not np.isfinite(tried[k]):
                raise ValueError(f'The left-out score is not finite at a floor of 2^{k} mean eigenvalues')
        return tried[k]
    way = -1 if at(-1) < at(0) else (1 if at(1) < at(0) else 0)
    k = way
    while way and at(k + way) < at(k):
        k += way
        if abs(k) >= edge:
            raise ValueError('No interior floor: the left-out score keeps falling to the edge of float64, so these requests do not determine one')
    best = min(tried, key=tried.get)
    if best - 1 in tried and best + 1 in tried:
        golden = (np.sqrt(5) - 1) / 2
        a, b = best - 1.0, best + 1.0
        c, d = b - golden * (b - a), a + golden * (b - a)
        while c < d and at(c) != at(d):
            if at(c) < at(d):
                b, d = d, c
                c = b - golden * (b - a)
            else:
                a, c = c, d
                d = a + golden * (b - a)
        best = min(tried, key=tried.get)
    floor = mean * 2.0 ** best
    values, vectors = torch.linalg.eigh(q)
    values = values.clamp(min=0)
    s = (vectors * ((mean + floor) / (values + floor)).sqrt()) @ vectors.T
    if not bool(torch.isfinite(s).all()):
        raise ValueError('The weighing is not finite')
    record = dict(requests=len(ids), rows=int(np.isin(request_of, ids).sum()), width=int(q.shape[0]), mean_eigenvalue=mean, floor=floor, floor_over_mean_eigenvalue=2.0 ** best, metric_scale=mean + floor,
                  floor_over_strongest_direction=floor / float(values.max()), directions_above_the_floor=int((values > floor).sum()),
                  strongest_direction_share_of_the_moment=float(values.max() / values.sum()), floor_candidates=len(tried),
                  left_out_sum_at_the_floor_minus_at_the_mean_eigenvalue=tried[best] - tried[0], seconds=round(time.time() - started, 1))
    return ((s + s.T) / 2).float(), record


def read(merge_before, merge_after, wanted, sets):
    """Construction readings of one matrix, on the realized BF16 weights: what was asked of the state after the merge and how far it moved."""
    out = {}
    done = merge_after.to - merge_before.to
    ask = wanted - merge_before.to
    for name, rows in sets.items():
        n = float(rows.sum())
        if n == 0:
            continue
        asked = float((rows * ask * ask).sum())
        out[name] = dict(rows=int(n), asked_rms=float(np.sqrt(asked / n)), moved_rms=float(torch.sqrt((rows * done * done).sum() / n)),
                         share_of_ask_reached=(1 - float((rows * (done - ask) ** 2).sum()) / asked) if asked > 0 else None)
    return out


def build(run):
    if run.name not in RUNGS:
        raise ValueError('No rung is defined for this run')
    source, members, index = cohort(run)
    tensors = json.loads(TENSORS.read_text())
    partial, log = model_of(run).with_suffix('.partial'), run/'build.jsonl'
    if model_of(run).exists():
        raise ValueError('The edited model exists; inspect it, never overwrite it')
    done = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    if not partial.exists():
        if done:
            raise ValueError('A build log without its model')
        partial.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGINAL_MODEL, partial)
    correct = np.array([m['correct'] for m in members])
    held = np.array([m['id'] in held_of(run) for m in members])
    if (held & ~correct).any() or held.sum() != len(held_of(run)):
        raise ValueError('The held answers are not correct answers of this overlap')
    half = np.zeros(len(members), dtype=bool)
    for group in (~correct, held, correct & ~held):  # alternating within each group, as in the overlap
        half[np.flatnonzero(group)[::2]] = True
    answer_of = np.concatenate([np.repeat(np.arange(len(members)), [m[key] for m in members]) for key in ('steps', 'question')])
    steps_total = sum(m['steps'] for m in members)
    on = lambda mask: torch.from_numpy(mask.astype('f4')[:, None]).to(DEVICE)
    where = lambda mask: torch.from_numpy(np.flatnonzero(mask)).to(DEVICE)
    fitted_rows = (~correct | held)[answer_of]  # a held answer's rows are asked for the state they had; nothing else about the fit changes
    fitted, halves = where(fitted_rows), [where(fitted_rows & half[answer_of]), where(fitted_rows & ~half[answer_of])]
    order = depth_of(run, list(matrices(index['layers'])))
    requests = ordinary_of(run)
    request_of = request_rows(requests)
    for position, layer, kind in order:
        if position < len(done):
            continue
        started = time.time()
        binding, w, g = weights_of(tensors, layer, kind)
        s, background, ordinary = None, None, None
        if requests:  # the ordinary requests' own text through the model as it stands, read at this matrix like the answers' text
            ordinary = tuple(torch.from_numpy(v).to(DEVICE) for v in passed(partial, run, f'{layer}-{kind}-ordinary', labels_of(layer, kind), requests))
            s, background = weighing(ordinary[0], Merge(*ordinary, w, g), request_of, [m['weighed'] for m in requests])
            torch.cuda.empty_cache()
        original = torch.from_numpy(np.fromfile(baseline_of(run)/f'state-{kind}-{layer}.f32', dtype='<f4').reshape(-1, 1536)).to(DEVICE)
        move, diverged = moves(source, position, members)
        wanted = original.clone()
        wanted[:steps_total] += torch.from_numpy(move).to(DEVICE)
        moving = np.zeros(len(answer_of), dtype=bool)
        moving[:steps_total] = (move != 0).any(1)
        # where a move is asked: the move itself, beside how far the same-mode baseline is from the token-by-token recording of the same rows
        generated = Merge(*(torch.from_numpy(np.ascontiguousarray(v)).to(DEVICE) for v in stored(members, index, layer, kind)), w, g).to
        modes = dict(move_rms=float(np.sqrt((move[moving[:steps_total]] ** 2).sum(1).mean())) if moving.any() else 0.0,
                     baseline_apart_from_generated_rms=float(torch.sqrt(((original - generated)[where(moving)] ** 2).sum(1).mean())) if moving.any() else 0.0)
        del generated
        torch.cuda.empty_cache()
        x, h = (torch.from_numpy(v).to(DEVICE) for v in passed(partial, run, f'{layer}-{kind}', labels_of(layer, kind), members))
        before = Merge(x, h, w, g)
        change, record = fit(x[fitted], h[fitted], w, g, wanted[fitted], [(x[rows], h[rows], wanted[rows]) for rows in halves], s)
        realized = rounded(w + change)
        after = Merge(x, h, realized, g)
        left = lambda merge: float(((wanted[fitted] - merge.to[fitted]) ** 2).sum())
        # The matrix is written only if, as the model file will keep it, it leaves less over the fitted rows than the matrix it replaces.
        written = bool((realized != w).any()) and left(after) < left(before)
        if not written:
            realized, after = w, before
        if requests:  # how far the realized change moves the state after this merge on ordinary text, the requests it was weighed by and the ones only read
            gone = ((Merge(*ordinary, realized, g).to - Merge(*ordinary, w, g).to) ** 2).sum(1).cpu().numpy()
            of = np.array([m['weighed'] for m in requests])[request_of]
            background.update(moved_rms_on_the_requests_weighed_by=float(np.sqrt(gone[of].mean())), moved_rms_on_the_requests_only_read=float(np.sqrt(gone[~of].mean())) if (~of).any() else None,
                              rows_only_read=int((~of).sum()))
            record.update(ordinary=background)
            del ordinary, s, gone
        sets = {'wrong answers, where the patterns diverge': on(moving), 'wrong answers, the rest of their path': on(~correct[answer_of] & ~moving),
                'correct answers held': on(held[answer_of]), 'correct answers': on((correct & ~held)[answer_of])}
        record.update(position=position, layer=layer, kind=kind, tensor=binding['name'], cells_diverged=diverged, written=written, where_a_move_is_asked=modes,
                      state_apart_from_original_before_this_edit_rms=float(torch.sqrt(((before.to - original) ** 2).sum(1).mean())),
                      turn_rms=float(torch.sqrt((before.turn ** 2).sum(1).mean())),
                      change_over_matrix=float((realized - w).norm() / w.norm()), intended_change_over_matrix=float(change.norm() / w.norm()),
                      readings=read(before, after, wanted, sets), seconds=round(time.time() - started, 1))
        if written:
            stored_bits = to_bf16(realized.cpu().numpy().ravel(order='F'))
            with partial.open('r+b') as model:
                model.seek(binding['offset_bytes'])
                if len(model.read(binding['bytes'])) != stored_bits.nbytes:
                    raise ValueError(f"{binding['name']}: the stored matrix has another size")
                model.seek(binding['offset_bytes'])
                model.write(stored_bits.tobytes())
                model.flush()
        with log.open('a') as file:
            file.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        del x, h, w, g, before, after, change, realized, wanted, original
        torch.cuda.empty_cache()
    partial.rename(model_of(run))
    save(run/'edit.json', dict(model=str(model_of(run).relative_to(run)), model_sha256=digest(model_of(run)), matrices_edited=len(order), matrices_declared=len(order),
                               held=sorted(held_of(run)), matrices=[json.loads(line) for line in log.read_text().splitlines()]))


def matrix_in(model, binding):
    return torch.from_numpy(checkpoint_values(model, binding).reshape(binding['shape'][:2], order='F').astype('f4')).to(DEVICE)


def weights_of(tensors, layer, kind):
    binding = tensors[f'blk.{layer}.{KINDS[kind][1]}.weight']
    g = torch.from_numpy(checkpoint_values(ORIGINAL_MODEL, tensors[f'blk.{layer}.{KINDS[kind][2]}.weight']).astype('f4').ravel()).to(DEVICE)
    w = torch.from_numpy(checkpoint_values(ORIGINAL_MODEL, binding).reshape(binding['shape'][:2], order='F').astype('f4')).to(DEVICE)
    return binding, w, g


def baseline(run):
    _, members, index = cohort(run)
    tensors = json.loads(TENSORS.read_text())
    folder = run/'baseline'
    folder.mkdir()
    layers, report = index['layers'], []
    for first in range(0, len(layers), 5):  # five layers a pass only bounds the raw stream on disk
        group = layers[first:first + 5]
        wanted = [label for layer in group for kind in KINDS for label in labels_of(layer, kind)]
        started = time.time()
        rows = dict(zip(wanted, passed(ORIGINAL_MODEL, run, f'baseline-{group[0]}', wanted, members)))
        seconds = round(time.time() - started, 1)
        for layer in group:
            for kind in KINDS:
                _, w, g = weights_of(tensors, layer, kind)
                both = [Merge(torch.from_numpy(np.ascontiguousarray(x)).to(DEVICE), torch.from_numpy(np.ascontiguousarray(h)).to(DEVICE), w, g)
                        for x, h in ([rows[label] for label in labels_of(layer, kind)], stored(members, index, layer, kind))]
                both[0].to.cpu().numpy().astype('<f4').tofile(folder/f'state-{kind}-{layer}.f32')
                apart = both[0].to - both[1].to
                report.append(dict(layer=layer, kind=kind, rows=int(apart.shape[0]), turn_rms=float(torch.sqrt((both[0].turn ** 2).sum(1).mean())),
                                   apart_from_generated_rms=float(torch.sqrt((apart ** 2).sum(1).mean())), largest_row_apart=float(apart.norm(dim=1).max()),
                                   pass_seconds=seconds, outputs_in_pass=len(wanted)))
                print(json.dumps(report[-1]), flush=True)
                del both, apart, w, g
                torch.cuda.empty_cache()
        del rows
    save(folder/'index.json', dict(matrices=report,
         meaning='state-<kind>-<layer>.f32: float32 [rows, 1536], the direction of the state right after that matrix\'s merge in the original model, for every answer step of '
                 'every answer in the correspondence\'s order, then every question position, read with the answers\' own text given as text. apart_from_generated_rms is how '
                 'far these are from those of the rows input-overlap recorded while the model generated the same text token by token.'))


def ordinary(run):
    """The original model answers the run's ordinary requests on its own, before the build that weighs by them."""
    from pipeline.acquisition.run_a_observation import run as observe
    declared = json.loads((run/'declaration.json').read_text())['ordinary']
    observe(ORIGINAL_MODEL, ROOT/declared['cases'], ROOT/declared['original'], 0, 'natural', 'plain', 'none', run/'declaration.json')
    for case in json.loads((ROOT/declared['cases']).read_text()):
        result = json.loads((ROOT/declared['original']/case['id']/'result.json').read_text())
        print(json.dumps(dict(id=case['id'], group=case['group'], question_tokens=result['prompt_tokens'], steps=len(result['steps']), stopped_at_limit=result['stopped_limit'])), flush=True)


def own(run):
    from pipeline.acquisition.export_stream_traces import export_case
    from pipeline.acquisition.run_a_observation import run as observe
    _, members, _ = cohort(run)
    todo = [m for m in members if not (run/'own-traces'/m['id']).exists()]
    for start in range(0, len(todo), 10):  # ten answers at a time only bounds the raw stream on disk
        batch = todo[start:start + 10]
        raw = run/'own'/f"batch-{batch[0]['id']}"
        cases = run/'own'/f"batch-{batch[0]['id']}-cases.json"
        cases.parent.mkdir(parents=True, exist_ok=True)
        cases.write_text(json.dumps([dict(id=m['id'], tokens=m['tokens'], max_tokens=json.loads((WIDE/'members.json').read_text())[0]['max_tokens']) for m in batch]))
        observe(model_of(run), cases, raw, 0, 'natural', 'capture', 'residual', run/'declaration.json')
        for m in batch:
            export_case(raw/m['id'], run/'own-traces'/m['id'])
            first_direction(raw/m['id'], run/'own-traces'/m['id'])
            for file in ('stream.f32', 'stream.jsonl'):
                (raw/m['id']/file).unlink()
            print(json.dumps(dict(answered=m['id'], steps=len(json.loads((raw/m['id']/'result.json').read_text())['steps']))), flush=True)


def first_direction(raw, exported):
    """The direction each step's state has before the first merge. The turns add up from it to the state after every merge, exactly."""
    index = json.loads((exported/'index.json').read_text())
    stream = StreamRows(raw, index['input_positions'])
    rows = np.stack([stream.values(index['connections'][0]['incoming'], p) for p in index['input_positions']]).astype('f8')
    stream.close()
    np.save(exported/'start.npy', (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype('f4'))


def overlay(run):
    """The edited model's own answers, laid along the axis by their computation alone, against the original patterns. Two readings of the same traces:
    the turns (the registered overlap of the turns), and the state after each matrix's merge (the overlap the edit takes its target from), which the
    turns add up to. The original state of each answer is the baseline reading of its rows. Beside them, the size of the state, which no target holds."""
    from pipeline.overlap.complete_overlap import distance_path, member_means, unit_steps
    target, members, index = cohort(run)
    record = json.loads((CORRESPONDENCE/'members.json').read_text())
    places = record['axis_steps']
    grid = unit_steps(np.load(WIDE/'exported'/record['axis']/'turns.npy', mmap_mode='r'))
    correct = np.array([m['correct'] for m in members])
    order = list(matrices(index['layers']))
    at_matrix = [3 * layer + ('attention', 'feedforward').index(kind) for _, layer, kind in order]  # each matrix's merge among the three merges of its layer
    baseline_rows = [np.memmap(baseline_of(run)/f'state-{kind}-{layer}.f32', dtype='<f4', mode='r').reshape(-1, 1536) for _, layer, kind in order]
    begins = np.cumsum([0] + [m['steps'] for m in members])
    readings = {}
    for name, source in (('turns', CORRESPONDENCE), ('states', target)):
        z = np.load(source/'recurrence.npz')
        c0, w0 = (np.asarray(np.load(source/f'{group}-pattern.npy', mmap_mode='r'), dtype='f8') for group in ('correct', 'wrong'))
        # each place and merge counts by its shared gap, where the patterns diverge
        readings[name] = dict(gap=z['gap'], weight=np.clip(z['gap'] - z['chance'], 0, None) * (z['gap'] > 2 * z['chance']), c0=c0, w0=w0, d=c0 - w0,
                              sums={True: 0.0, False: 0.0}, original=[], edited=[])
    first = readings['turns']['weight'][:, :1].copy()
    readings['turns']['weight'][:, 0] = 0  # the first merge's turn is mostly the token's own embedding (first_merge); it is read apart, below, and is not part of the summary
    sizes = {when: {True: 0.0, False: 0.0} for when in ('original', 'edited')}
    rows = []
    for number, m in enumerate(members):
        folder = dict(original=WIDE/'exported'/m['id'], edited=run/'own-traces'/m['id'])
        turns = {when: np.load(folder[when]/'turns.npy', mmap_mode='r') for when in folder}
        likeness = {when: unit_steps(turns[when]) @ grid.T for when in folder}
        pairs = dict(original=np.load(CORRESPONDENCE/'pairs'/f"{m['id']}.npy"), edited=distance_path(likeness['edited']))
        states = dict(original=np.stack([file[begins[number]:begins[number + 1]] for file in baseline_rows], axis=1),
                      edited=(np.load(folder['edited']/'start.npy').astype('f8')[:, None] + np.cumsum(turns['edited'], axis=1, dtype='f8'))[:, at_matrix])
        row = dict(id=m['id'], correct=bool(m['correct']))
        for when in folder:
            steps = np.bincount(pairs[when][:, 0])
            row[f'steps_{when}'] = int(turns[when].shape[0])
            row[f'likeness_{when}'] = float(np.mean(np.bincount(pairs[when][:, 0], weights=likeness[when][pairs[when][:, 0], pairs[when][:, 1]]) / steps))
            sizes[when][row['correct']] = sizes[when][row['correct']] + member_means(np.load(folder[when]/'norms.npy', mmap_mode='r')[:, :, 1:], pairs[when], places)[..., 0]
            for name, values in (('turns', turns), ('states', states)):
                r = readings[name]
                mean = member_means(values[when], pairs[when], places)
                # where this answer's own trace lies between the two patterns: 0 on the wrong pattern, 1 on the correct one
                r[when].append((((mean - r['w0']) * r['d']).sum(-1) / np.where(r['gap'] > 0, r['gap'], np.inf)).astype('f4'))
                if when == 'edited':
                    r['sums'][row['correct']] = r['sums'][row['correct']] + mean
        rows.append(row)
        print(json.dumps(row), flush=True)
    for name, sel in (('were wrong', ~correct), ('were correct', correct)):
        pick = [r for r, s in zip(rows, sel) if s]
        print(f"{name:12s} steps of the answer: original median {int(np.median([r['steps_original'] for r in pick]))}, edited median {int(np.median([r['steps_edited'] for r in pick]))} "
              f"({min(r['steps_edited'] for r in pick)} to {max(r['steps_edited'] for r in pick)}) | how alike each step is to the axis: original {np.mean([r['likeness_original'] for r in pick]):.2f}, "
              f"edited {np.mean([r['likeness_edited'] for r in pick]):.2f}")
    sections = (('whole path', slice(0, places)), ('before 142', slice(0, 142)), ('places 142-313', slice(142, 314)), ('places 314-456', slice(314, 457)), ('places 457-479', slice(457, places)))
    size = {flag: (sizes['edited'][flag] / sizes['original'][flag]).astype('f4') for flag in (True, False)}
    for name, r in readings.items():
        gap0, d = r['gap'], r['d']
        scale = np.where(gap0 > 0, gap0, np.inf)
        c1, w1 = r['sums'][True] / correct.sum(), r['sums'][False] / (~correct).sum()
        along = ((w1 - r['w0']) * d).sum(-1) / scale
        r['towards'] = along
        out = dict(gap_before=gap0, gap_after=((c1 - w1) ** 2).sum(-1), wrong_to_original_correct=((r['c0'] - w1) ** 2).sum(-1), wrong_towards_correct=along,
                   wrong_sideways=(((w1 - r['w0']) - along[..., None] * d) ** 2).sum(-1) / scale,
                   correct_moved=((c1 - r['c0']) ** 2).sum(-1) / scale, correct_towards_wrong=-((c1 - r['c0']) * d).sum(-1) / scale)
        np.savez(run/('overlay.npz' if name == 'turns' else 'overlay-states.npz'), weight=r['weight'], ids=np.array([row['id'] for row in rows]),
                 original=np.stack(r['original']), edited=np.stack(r['edited']), size_edited_over_original_correct=size[True], size_edited_over_original_wrong=size[False], **out)
        print('the turns of every merge but the first, against the overlap of the turns:' if name == 'turns' else 'the state after each matrix, against the overlap of the states (what the edit aims at):')
        for label, sel in sections:
            w = r['weight'][sel]
            mean = lambda v: float((v[sel] * w).sum() / w.sum())
            gaps = lambda v: float(np.sqrt((v[sel] * gap0[sel]).sum() / gap0[sel].sum()))
            print(f"  {label:15s} answers that were wrong: towards the correct pattern {mean(out['wrong_towards_correct']):5.2f}, sideways {gaps(out['wrong_sideways']):5.2f} gaps, "
                  f"left to the original correct pattern / before {np.sqrt(out['wrong_to_original_correct'][sel].sum() / gap0[sel].sum()):5.2f} | "
                  f"answers that were correct: travelled {gaps(out['correct_moved']):5.2f} gaps | gap between the two groups after / before {np.sqrt(out['gap_after'][sel].sum() / gap0[sel].sum()):5.2f}")
    print(f"the first merge alone, places 142-313 (its turn carries the token embeddings, so it follows the tokens written): answers that were wrong, towards the correct pattern "
          f"{float((readings['turns']['towards'][142:314, :1] * first[142:314]).sum() / first[142:314].sum()):5.2f}")
    for label, flag in (('were wrong', False), ('were correct', True)):
        print(f"size of the state after each merge, edited over original, answers that {label}: median {np.median(size[flag]):.3f}, from {size[flag].min():.3f} to {size[flag].max():.3f}; "
              f"places 142-313 median {np.median(size[flag][142:314]):.3f}; after the last merge median {np.median(size[flag][:, -1]):.3f}")
    part = lambda name, when, i: float((readings[name][when][i][142:314] * readings[name]['weight'][142:314]).sum() / readings[name]['weight'][142:314].sum())
    print('each answer, where its own trace lies between the two patterns over places 142-313 (0 = the wrong pattern, 1 = the correct one), original -> edited:')
    for i, row in enumerate(rows):
        print(f"  {'was correct' if row['correct'] else 'was wrong  '} {row['id']:28s} turns {part('turns', 'original', i):5.2f} -> {part('turns', 'edited', i):5.2f} | "
              f"states {part('states', 'original', i):5.2f} -> {part('states', 'edited', i):5.2f} | steps {row['steps_original']} -> {row['steps_edited']}")


def travel(run):
    """How far each answer's own trace is from where the same answer was in the original model, place by place and matrix by matrix:
    the state after each matrix's merge on the edited model's own answer, laid by the computation alone, minus that answer's own state
    in the original model at the same place. The whole distance is kept, not only its part along the gap, in the units of the state's
    direction, with where each step of the edited answer was laid, so a passage of its text can be found on the axis."""
    from pipeline.overlap.complete_overlap import distance_path, member_means, unit_steps
    target, members, index = cohort(run)
    record = json.loads((CORRESPONDENCE/'members.json').read_text())
    places = record['axis_steps']
    grid = unit_steps(np.load(WIDE/'exported'/record['axis']/'turns.npy', mmap_mode='r'))
    order = list(matrices(index['layers']))
    at_matrix = [3 * layer + ('attention', 'feedforward').index(kind) for _, layer, kind in order]
    baseline_rows = [np.memmap(baseline_of(run)/f'state-{kind}-{layer}.f32', dtype='<f4', mode='r').reshape(-1, 1536) for _, layer, kind in order]
    begins = np.cumsum([0] + [m['steps'] for m in members])
    c0 = np.asarray(np.load(target/'correct-pattern.npy', mmap_mode='r'), dtype='f8')
    d = c0 - np.asarray(np.load(target/'wrong-pattern.npy', mmap_mode='r'), dtype='f8')
    moved = np.zeros((len(members), places, len(order)), dtype='f4')   # squared distance from the answer's own original state
    off, off_before = np.zeros_like(moved), np.zeros_like(moved)       # squared distance from the original correct pattern, on the edited model and on the original: an answer's own text is in both
    along = np.zeros_like(moved)                                        # the part of it along the gap, in units of the state's direction (towards the correct pattern is positive)
    laid = {}
    for number, m in enumerate(members):
        turns = np.load(run/'own-traces'/m['id']/'turns.npy', mmap_mode='r')
        pairs = distance_path(unit_steps(turns) @ grid.T)
        edited = member_means((np.load(run/'own-traces'/m['id']/'start.npy').astype('f8')[:, None] + np.cumsum(turns, axis=1, dtype='f8'))[:, at_matrix], pairs, places)
        original = member_means(np.stack([file[begins[number]:begins[number + 1]] for file in baseline_rows], axis=1), np.load(CORRESPONDENCE/'pairs'/f"{m['id']}.npy"), places)
        gone = edited - original
        moved[number] = (gone ** 2).sum(-1)
        off[number], off_before[number] = ((edited - c0) ** 2).sum(-1), ((original - c0) ** 2).sum(-1)
        along[number] = (gone * d).sum(-1) / np.sqrt(np.where((d ** 2).sum(-1) > 0, (d ** 2).sum(-1), np.inf))
        laid[m['id']] = np.bincount(pairs[:, 0], weights=pairs[:, 1]) / np.bincount(pairs[:, 0])  # the place each step of the edited answer lies at (the mean of its places)
        print(json.dumps(dict(id=m['id'], correct=bool(m['correct']), moved_rms=float(np.sqrt(moved[number].mean())), moved_rms_places_142_313=float(np.sqrt(moved[number, 142:314].mean())))), flush=True)
    np.savez(run/'travel.npz', ids=np.array([m['id'] for m in members]), correct=np.array([m['correct'] for m in members]), moved=moved, along=along, off=off, off_before=off_before,
             gap=(d ** 2).sum(-1).astype('f4'), **{f'laid-{name}': value.astype('f4') for name, value in laid.items()})


if __name__ == '__main__':
    dict(states=states, baseline=baseline, ordinary=ordinary, build=build, own=own, overlay=overlay, travel=travel)[sys.argv[1]](ROUND_TWO_RUNS/sys.argv[2])
