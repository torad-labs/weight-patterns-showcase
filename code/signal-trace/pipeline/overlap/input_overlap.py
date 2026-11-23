"""input-overlap: the overlap of trace-overlap's 60 answers, read at what arrives at each matrix.

From code/signal-trace:
  python3 -m pipeline.overlap.input_overlap record|overlap

record   The same 60 answers again on the original model, keeping, for each of the 70 attention-output and
         feed-forward-down matrices, its input and the state its contribution is added to, at every generated token
         of every answer and at the positions of the question. Nothing depends on an alignment.
overlap  For each matrix and place: the gap between the correct group's average input and the wrong group's, against
         what two patterns of one and the same group would differ by, and whether two halves that share no answer
         agree on it. The same two readings as trace-overlap and sent-overlap.
"""
import json
import sys

import numpy as np

from pipeline.acquisition.export_stream_traces import StreamRows
from pipeline.overlap.sent_overlap import matrices
from pipeline.paths import ORIGINAL_MODEL, ROUND_TWO_RUNS
from pipeline.weight_edit.a2 import save
from pipeline.weight_edit.a2_pattern import BATCH, KINDS, WIDE

OVERLAP = ROUND_TWO_RUNS/'trace-overlap'
SENT = ROUND_TWO_RUNS/'sent-overlap'
RUN = ROUND_TWO_RUNS/'input-overlap'


def stores(folder, layers, mode):
    """One flat float32 file per matrix and kind of row: every answer's rows one after another, in trace-overlap's order."""
    names = [(layer, kind) for _, layer, kind in matrices(layers)]
    return {(what, layer, kind): (folder/f'{what}-{kind}-{layer}.f32').open(mode)
            for layer, kind in names for what in ('x', 'state', 'question-x', 'question-state')}


def record():
    from pipeline.acquisition.run_a_observation import run as observe
    group = json.loads((OVERLAP/'members.json').read_text())['members']
    wide = {m['id']: m for m in json.loads((WIDE/'members.json').read_text())}
    folder = RUN/'inputs'
    folder.mkdir()
    tensors = json.loads((SENT/'wrong-inputs'/'tensors.json').read_text())
    layers = json.loads((SENT/'wrong-inputs'/'index.json').read_text())['layers']
    files = stores(folder, layers, 'wb')
    index = []
    for start in range(0, len(group), BATCH):
        batch = group[start:start + BATCH]
        raw = RUN/'capture'/f"batch-{batch[0]['id']}"
        cases = RUN/'capture'/f"batch-{batch[0]['id']}-cases.json"
        cases.parent.mkdir(parents=True, exist_ok=True)
        cases.write_text(json.dumps([dict(id=m['id'], tokens=wide[m['id']]['tokens'], max_tokens=wide[m['id']]['max_tokens']) for m in batch]))
        observe(ORIGINAL_MODEL, cases, raw, 0, 'natural', 'capture', 'projection', RUN/'declaration.json')
        for m in batch:
            result = json.loads((raw/m['id']/'result.json').read_text())
            if [s['selected'] for s in result['steps']] != wide[m['id']]['recorded_tokens']:
                raise ValueError(f"{m['id']}: the answer differs from the recorded one")
            steps = [s['input_position'] for s in result['steps']]  # every generated token, in order
            before = list(range(steps[0]))                          # the question, up to the position the answer starts from
            stream = StreamRows(raw/m['id'], set(steps) | set(before))
            for (what, layer, kind), file in files.items():
                matrix_input, _, _, incoming, *_rest = KINDS[kind]
                if what.endswith('x'):
                    label = f'{matrix_input}-{layer}'
                else:  # the state the matrix's contribution is added to
                    label = ('inp_scaled' if layer == 0 else f'l_out-{layer - 1}') if incoming is None else f'{incoming}-{layer}'
                np.stack([stream.values(label, p) for p in (before if what.startswith('question') else steps)]).astype('<f4').tofile(file)
            stream.close()
            for name in ('stream.f32', 'stream.jsonl'):
                (raw/m['id']/name).unlink()
            index.append(dict(id=m['id'], correct=m['correct'], steps=len(steps), question_positions=len(before)))
            print(json.dumps(index[-1]), flush=True)
    for file in files.values():
        file.close()
    save(folder/'index.json', dict(layers=layers, answers=index, width={f'{kind}-{layer}': tensors[f'blk.{layer}.{KINDS[kind][1]}.weight']['shape'][0] for _, layer, kind in matrices(layers)},
         meaning='x-<kind>-<layer>.f32 holds the input of that matrix at every generated token of every answer, answers in the listed order, rows one after '
                 'another, float32, as wide as width says. state-<kind>-<layer>.f32 holds the state the matrix\'s contribution is added to, 1536 wide, same rows. '
                 'question-x and question-state hold the same at the positions of the question.'))


def rows(what, layer, kind):
    """The stored rows of one matrix as [row, width], and the first row of each answer."""
    index = json.loads((RUN/'inputs'/'index.json').read_text())
    width = index['width'][f'{kind}-{layer}'] if what.endswith('x') else 1536
    count = [a['question_positions' if what.startswith('question') else 'steps'] for a in index['answers']]
    data = np.memmap(RUN/'inputs'/f'{what}-{kind}-{layer}.f32', dtype='<f4', mode='r').reshape(-1, width)
    if data.shape[0] != sum(count):
        raise ValueError(f'{what}-{kind}-{layer}: {data.shape[0]} rows stored, {sum(count)} listed')
    return data, np.concatenate([[0], np.cumsum(count)[:-1]])


def overlap():
    group = json.loads((OVERLAP/'members.json').read_text())['members']
    aligned = np.load(OVERLAP/'aligned.npy')
    layers = json.loads((RUN/'inputs'/'index.json').read_text())['layers']
    correct = np.array([m['correct'] for m in group])
    even = np.zeros(len(group), dtype=bool)
    for flag in (True, False):  # alternating within each group, as in trace-overlap
        even[np.flatnonzero(correct == flag)[::2]] = True
    names = matrices(layers)
    gap, chance, halves = (np.zeros((aligned.shape[1], len(names))) for _ in range(3))
    for j, layer, kind in names:
        x, first = rows('x', layer, kind)
        for p in range(aligned.shape[1]):
            at = np.asarray(x[first + aligned[:, p]], dtype='f8')  # each answer's step laid at this place of trace-overlap's axis
            a, b = at[correct], at[~correct]
            d = a.mean(0) - b.mean(0)
            gap[p, j] = d @ d
            chance[p, j] = sum(((g - g.mean(0)) ** 2).sum() / (len(g) - 1) / len(g) for g in (a, b))
            d1 = at[correct & even].mean(0) - at[~correct & even].mean(0)
            d2 = at[correct & ~even].mean(0) - at[~correct & ~even].mean(0)
            halves[p, j] = d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2))
        print(f'{kind} {layer}', flush=True)
    np.savez(RUN/'recurrence.npz', gap=gap, chance=chance, halves=halves)
    sent = np.load(SENT/'recurrence.npz')
    diverged = sent['gap'] > 2 * sent['chance']  # where the two groups are further apart, in what the matrix sends, than two halves of one group
    ratio = gap / chance
    for kind, offset in (('attention', 0), ('feedforward', 1)):
        cols = slice(offset, len(names), 2)
        print(f"{kind:11s} where the sending diverges: input gap over chance {np.median(ratio[:, cols][diverged[:, cols]]):5.1f}, halves agree {np.median(halves[:, cols][diverged[:, cols]]):5.2f} | "
              f"where the groups follow each other: {np.median(ratio[:, cols][~diverged[:, cols]]):5.1f}, {np.median(halves[:, cols][~diverged[:, cols]]):5.2f}")
    for name, sel in (('before 142', slice(10, 142)), ('places 142-304', slice(142, 305)), ('places 305-456', slice(305, 457)), ('places 457-479', slice(457, 480))):
        print(f"{name:15s} input gap over chance {np.median(ratio[sel]):5.1f} | halves agree {np.median(halves[sel]):5.2f}")


if __name__ == '__main__':
    dict(record=record, overlap=overlap)[sys.argv[1]]()
