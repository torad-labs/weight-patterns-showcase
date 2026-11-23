"""sent-overlap: the overlap of trace-overlap's 60 answers, read where each matrix writes.

From code/signal-trace:
  python3 -m pipeline.overlap.sent_overlap record|overlap

record   The same 60 answers again on the original model, with the inputs of the 70 attention-output and
         feed-forward-down matrices. At every place of trace-overlap's axis, what each matrix sends: its output with
         its size taken out, as the model's norm does before the learned scale. The wrong group's average inputs are
         kept with it.
overlap  The shared pattern of what each matrix sends in the correct group, in the wrong group, the gap between them
         and how much of it is shared. The calculation is trace-overlap's.
"""
import json
import sys

import numpy as np

from pipeline.acquisition.export_stream_traces import StreamRows
from pipeline.native_records.native_trace import checkpoint_values
from pipeline.overlap.trace_overlap import overlap_of
from pipeline.paths import ORIGINAL_MODEL, ROUND_TWO_RUNS
from pipeline.weight_edit.a2 import NORM_EPSILON, save
from pipeline.weight_edit.a2_pattern import BATCH, KINDS

OVERLAP = ROUND_TWO_RUNS/'trace-overlap'
WIDE = ROUND_TWO_RUNS/'wide-overlap'
RUN = ROUND_TWO_RUNS/'sent-overlap'


def matrices(layers):
    """The 70 matrices in path order: (index, layer, kind)."""
    return [(2 * n + offset, layer, kind) for n, layer in enumerate(layers) for kind, (*_rest, offset) in KINDS.items()]


def matrix_weights(model, tensors, layers):
    """The 70 matrices of a model, [input, output], by (layer, kind)."""
    weights = {}
    for _, layer, kind in matrices(layers):
        b = tensors[f'blk.{layer}.{KINDS[kind][1]}.weight']
        weights[layer, kind] = checkpoint_values(model, b).reshape(b['shape'][:2], order='F').astype('f8')
    return weights


def sending(stream, positions, layers, weights, inputs=None):
    """What each matrix sends at each of the positions, and the size taken out. inputs(layer, kind, x) receives the matrix inputs."""
    sent = np.zeros((len(positions), 2 * len(layers), 1536), dtype='f4')
    size = np.zeros((len(positions), 2 * len(layers)))
    for index, layer, kind in matrices(layers):
        x = np.stack([stream.values(f'{KINDS[kind][0]}-{layer}', p) for p in positions]).astype('f8')
        y = x @ weights[layer, kind]
        size[:, index] = np.sqrt((y ** 2).mean(-1) + NORM_EPSILON)  # the model's norm divides by this, then multiplies by the learned scale
        sent[:, index] = y / size[:, index, None]
        if inputs is not None:
            inputs(layer, kind, x)
    return sent, size


def record():
    from pipeline.acquisition.run_a_observation import run
    group = json.loads((OVERLAP/'members.json').read_text())['members']
    aligned = np.load(OVERLAP/'aligned.npy')
    wide = {m['id']: m for m in json.loads((WIDE/'members.json').read_text())}
    places = aligned.shape[1]
    (RUN/'sent').mkdir(exist_ok=True)
    todo = [(i, m) for i, m in enumerate(group) if not (RUN/'sent'/f"{m['id']}.npy").exists()]
    if len(todo) != len(group):
        raise ValueError('A partial recording exists; the wrong group\'s average inputs need every answer in one pass')
    tensors, weights, layers, keys, half, wrong = None, None, None, None, None, 0
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        raw = RUN/'capture'/f"batch-{batch[0][1]['id']}"
        cases = RUN/'capture'/f"batch-{batch[0][1]['id']}-cases.json"
        cases.parent.mkdir(parents=True, exist_ok=True)
        cases.write_text(json.dumps([dict(id=m['id'], tokens=wide[m['id']]['tokens'], max_tokens=wide[m['id']]['max_tokens']) for _, m in batch]))
        run(ORIGINAL_MODEL, cases, raw, 0, 'natural', 'capture', 'projection', RUN/'declaration.json')
        if tensors is None:
            tensors = json.loads((raw/'weight-bindings.json').read_text())['tensors']
            layers = sorted(int(n.split('.')[1]) for n in tensors if n.endswith('.attn_output.weight'))
            weights = matrix_weights(ORIGINAL_MODEL, tensors, layers)
            keys = {name: np.zeros((places, w.shape[0])) for name, w in weights.items()}
            half = {name: np.zeros((places, w.shape[0])) for name, w in weights.items()}
        for i, m in batch:
            result = json.loads((raw/m['id']/'result.json').read_text())
            if [s['selected'] for s in result['steps']] != wide[m['id']]['recorded_tokens']:
                raise ValueError(f"{m['id']}: the answer differs from the recorded one")
            positions = [result['steps'][s]['input_position'] for s in aligned[i]]
            stream = StreamRows(raw/m['id'], set(positions))
            def keep(layer, kind, x):
                keys[layer, kind] += x
                half[layer, kind] += x if wrong % 2 == 0 else -x
            sent, size = sending(stream, positions, layers, weights, None if m['correct'] else keep)
            wrong += not m['correct']
            stream.close()
            np.save(RUN/'sent'/f"{m['id']}-size.npy", size)
            np.save(RUN/'sent'/f"{m['id']}.npy", sent)
            for file in ('stream.f32', 'stream.jsonl'):
                (raw/m['id']/file).unlink()
            print(json.dumps(dict(recorded=m['id'], correct=m['correct'])), flush=True)
    folder = RUN/'wrong-inputs'
    folder.mkdir()
    for (layer, kind), value in keys.items():
        np.save(folder/f'key-{kind}-{layer}.npy', (value / wrong).astype('f4'))
        # (mean of one half - mean of the other half) / 2: it holds no shared input, only what differs between answers
        np.save(folder/f'key-halves-{kind}-{layer}.npy', (half[layer, kind] / wrong).astype('f4'))
    save(folder/'index.json', dict(answers=wrong, places=places, layers=layers,
         meaning='key-<kind>-<layer>[place] is the wrong answers\' average input of that matrix at that place. sent/<id>.npy[place, matrix] is what the '
                 'matrix sends there, matrices in path order (attention then feed-forward of each layer); sent/<id>-size.npy the size taken out.'))
    save(folder/'tensors.json', tensors)


def overlap():
    group = json.loads((OVERLAP/'members.json').read_text())['members']
    traces = [np.load(RUN/'sent'/f"{m['id']}.npy", mmap_mode='r') for m in group]
    places = traces[0].shape[0]
    overlap_of(traces, np.tile(np.arange(places), (len(group), 1)), np.array([m['correct'] for m in group]), RUN)
    z = np.load(RUN/'recurrence.npz')
    ratio, share = z['gap'] / z['chance'], np.clip(1 - z['chance'] / z['gap'], 0, None)
    move = share * np.sqrt(z['gap'] / traces[0].shape[2])  # the shared move as a part of what the matrix sends, whose size is the square root of the width
    for name, sel in (('before 150', slice(10, 150)), ('places 150-265', slice(150, 266)), ('after 265', slice(266, places - 10))):
        for kind, offset in (('attention', 0), ('feedforward', 1)):
            print(f"{name:15s} {kind:11s} gap over chance {np.median(ratio[sel, offset::2]):5.1f} | halves agree {np.median(z['halves'][sel, offset::2]):5.2f} | "
                  f"shared move over what is sent {np.median(move[sel, offset::2]):5.3f}")
    print('gap over chance by layer, places 150-265:', ' '.join(f'{np.median(ratio[150:266, k:k + 2]):.1f}' for k in range(0, ratio.shape[1], 2)))


if __name__ == '__main__':
    dict(record=record, overlap=overlap)[sys.argv[1]]()
