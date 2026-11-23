"""trace-overlap: overlap the traces by the computation itself. No token, value or answer text lines anything up.

From code/signal-trace:
  python3 -m pipeline.overlap.trace_overlap align|overlap

align    Every answer's trace is matched, in order, to one member's trace used as the axis. Two steps match by how
         alike their computation is: the cosine between their turns at all 105 merges. The match is the monotonic
         path with the greatest total likeness.
overlap  Along that axis, the shared pattern of the correct answers, the shared pattern of the wrong answers, the gap
         between them at every merge, and how much of the gap is shared rather than particular to single answers.
"""
import json
import sys

import numpy as np

from pipeline.paths import ROUND_TWO_RUNS

SOURCE = ROUND_TWO_RUNS/'wide-overlap'
RUN = ROUND_TWO_RUNS/'trace-overlap'


def members():
    rows = json.loads((SOURCE/'members.json').read_text())
    return [dict(id=m['id'], correct=m['role'] == 'correct') for m in rows]


def trace(member):
    return np.load(SOURCE/'exported'/member['id']/'turns.npy', mmap_mode='r')


def unit_rows(turns):
    flat = np.asarray(turns, dtype='f4').reshape(turns.shape[0], -1)
    return flat / np.linalg.norm(flat, axis=1, keepdims=True)


def best_path(likeness):
    """Monotonic match of two traces with the greatest total likeness. Returns the (step, axis step) pairs."""
    n, m = likeness.shape
    total = np.full((n + 1, m + 1), -np.inf)
    total[0, 0] = 0
    move = np.zeros((n + 1, m + 1), dtype='i1')
    for i in range(1, n + 1):
        row, above = total[i], total[i - 1]
        for j in range(1, m + 1):
            options = (above[j - 1], above[j], row[j - 1])
            k = int(np.argmax(options))
            row[j] = options[k] + likeness[i - 1, j - 1]
            move[i, j] = k
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        pairs.append((i - 1, j - 1))
        k = move[i, j]
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return pairs[::-1]


def lay_along(turns, grid):
    """One trace laid along the axis: for every place of the axis, the step of the trace matched to it and how alike the two are."""
    s = unit_rows(turns) @ grid.T
    best = {}
    for step, place in best_path(s):
        if place not in best or s[step, place] > s[best[place], place]:
            best[place] = step
    return [best[p] for p in range(grid.shape[0])], [s[best[p], p] for p in range(grid.shape[0])]


def align():
    group = members()
    lengths = [trace(m).shape[0] for m in group]
    axis = int(np.argsort(lengths)[len(lengths) // 2])  # the member of median length: an axis, not a model answer
    grid = unit_rows(trace(group[axis]))
    aligned = np.zeros((len(group), grid.shape[0]), dtype='i4')
    likeness = np.zeros(aligned.shape, dtype='f4')
    for i, member in enumerate(group):
        aligned[i], likeness[i] = lay_along(trace(member), grid)
        print(json.dumps(dict(member=member['id'], steps=lengths[i], mean_likeness=float(likeness[i].mean()))), flush=True)
    np.save(RUN/'aligned.npy', aligned)
    np.save(RUN/'likeness.npy', likeness)
    (RUN/'members.json').write_text(json.dumps(dict(axis=group[axis]['id'], axis_steps=int(grid.shape[0]), members=group), indent=1) + '\n')


def overlap():
    group = json.loads((RUN/'members.json').read_text())['members']
    overlap_of([trace(m) for m in group], np.load(RUN/'aligned.npy'), np.array([m['correct'] for m in group]), RUN)


def overlap_of(traces, aligned, correct, folder):
    """The two shared patterns, the gap between them and how much of it is shared, for traces[member][step, merge, width] laid along aligned[member, place]."""
    places, merges, width = aligned.shape[1], traces[0].shape[1], traces[0].shape[2]
    pattern = {name: np.lib.format.open_memmap(folder/f'{name}-pattern.npy', mode='w+', dtype='f4', shape=(places, merges, width))
               for name in ('correct', 'wrong')}
    gap = np.zeros((places, merges))        # squared size of the gap between the two shared patterns
    chance = np.zeros((places, merges))     # the squared size two patterns of the same group would differ by, from their spread
    halves = np.zeros((places, merges))     # agreement of the gap between two halves of each group that share no answer
    within = np.zeros((2, places, merges))  # how alike the members of a group are: share of their energy in the shared pattern
    even = np.arange(len(traces)) % 2 == 0
    for p in range(places):
        x = np.stack([traces[i][aligned[i, p]] for i in range(len(traces))]).astype('f8')
        a, b = x[correct], x[~correct]
        pattern['correct'][p], pattern['wrong'][p] = a.mean(0), b.mean(0)
        d = a.mean(0) - b.mean(0)
        gap[p] = (d * d).sum(-1)
        spread = [((g - g.mean(0)) ** 2).sum(-1).sum(0) / (len(g) - 1) for g in (a, b)]
        chance[p] = spread[0] / len(a) + spread[1] / len(b)
        for k, g in enumerate((a, b)):
            within[k, p] = (g.mean(0) ** 2).sum(-1) / (g ** 2).sum(-1).mean(0)
        d1 = x[correct & even].mean(0) - x[~correct & even].mean(0)
        d2 = x[correct & ~even].mean(0) - x[~correct & ~even].mean(0)
        halves[p] = (d1 * d2).sum(-1) / (np.linalg.norm(d1, axis=-1) * np.linalg.norm(d2, axis=-1))
        if p % 50 == 0:
            print(f'place {p}/{places}', flush=True)
    for array in pattern.values():
        array.flush()
    np.savez(folder/'recurrence.npz', gap=gap, chance=chance, halves=halves, within=within)


if __name__ == '__main__':
    dict(align=align, overlap=overlap)[sys.argv[1]]()
