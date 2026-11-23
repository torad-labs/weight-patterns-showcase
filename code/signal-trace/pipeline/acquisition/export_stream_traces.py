"""Export all recorded residual turns while retaining native matrix-input provenance.

Goal: let overlap analysis follow every actual recorded residual merge at every
natural prediction. Missing inputs or disconnected source names must fail, and
the native stream remains the authority for projection inputs and components.
"""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from pipeline.native_records.run_declaration import check_native_run, digest


class StreamRows:
    """Index native rows by recorded token position without copying the tensor file."""

    def __init__(self, source, positions):
        self.source = Path(source)
        self.data = np.memmap(self.source/'stream.f32', dtype='<f4', mode='r')
        self.rows = {}
        self.metadata = {}
        wanted = set(positions)
        with (self.source/'stream.jsonl').open() as records:
            for line in records:
                r = json.loads(line)
                shape, axis = tuple(r['shape']), r['token_axis']
                offset, count = r['offset_bytes'], r['values']
                if (axis < 0 or axis >= len(shape) or len(r['positions']) != shape[axis]
                        or math.prod(shape) != count or offset % 4
                        or offset < 0 or offset//4 + count > self.data.size):
                    raise ValueError('Invalid or truncated native stream tensor')
                # Prefill microbatches without requested logits legitimately
                # expose an empty result_norm. They supply no token rows.
                if count == 0:
                    continue
                label = r['label']
                width = count // shape[axis]
                old = self.metadata.setdefault(label, dict(width=width,
                    native_type=r['native_type'], operation=r['operation']))
                if old['width'] != width or old['native_type'] != r['native_type']:
                    raise ValueError(f'Changed native shape/type for {label}')
                for row, position in enumerate(r['positions']):
                    if position not in wanted:
                        continue
                    entry = (offset//4, count, shape, axis, row, r['operation'], r['source_names'])
                    key = (label, position)
                    if key in self.rows:
                        if not np.array_equal(self._values(self.rows[key]).view('u4'), self._values(entry).view('u4')):
                            raise ValueError(f'Conflicting native rows: {key}')
                    self.rows[key] = entry

    def _values(self, entry):
        offset, count, shape, axis, row = entry[:5]
        tensor = self.data[offset:offset+count].reshape(shape, order='F')
        return np.take(tensor, row, axis=axis).ravel(order='F')

    def values(self, label, position):
        try:
            return self._values(self.rows[label, position])
        except KeyError as error:
            raise ValueError(f'Missing native row {label} at {position}') from error

    def close(self):
        self.data._mmap.close()


def connections(stream):
    layers = sorted(int(m.group(1)) for label in stream.metadata
                    if (m := re.fullmatch(r'l_out-(\d+)', label)))
    if not layers or layers != list(range(layers[-1]+1)):
        raise ValueError('Missing contiguous layer exits in native stream')
    result = []
    for layer in layers:
        incoming = 'inp_scaled' if layer == 0 else f'l_out-{layer-1}'
        specs = [
            ('attention', incoming, f'attn_post_norm-{layer}', f'attn_out-{layer}'),
            ('feedforward', f'attn_out-{layer}', f'ffn_post_norm-{layer}', f'pe_in-{layer}'),
            ('per_layer_embedding', f'pe_in-{layer}', f'per_layer_embd_out-{layer}', f'pe_out-{layer}'),
        ]
        for kind, incoming, branch, merged in specs:
            for name in [incoming, branch, merged]:
                if name not in stream.metadata:
                    raise ValueError(f'Missing connected tensor {name}')
            result.append(dict(layer=layer, kind=kind, incoming=incoming,
                               branch=branch, merged=merged, layer_exit=f'l_out-{layer}'))
    return result


def export_case(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Never overwrite an exported trace')
    result = json.loads((source/'result.json').read_text())
    steps = result['steps']
    positions = [r['input_position'] for r in steps]
    if not positions or len(set(positions)) != len(positions):
        raise ValueError('Natural decisions need distinct recorded input positions')
    stream = StreamRows(source, positions)
    try:
        path = connections(stream)
        width = stream.metadata[path[0]['incoming']]['width']
        shape = (len(positions), len(path), width)
        output.mkdir(parents=True)
        turns = np.lib.format.open_memmap(output/'turns.npy', mode='w+', dtype='f4', shape=shape)
        norms = np.lib.format.open_memmap(output/'norms.npy', mode='w+', dtype='f8', shape=shape[:2]+(2,))
        verified = 0
        for decision, position in enumerate(positions):
            for merge, edge in enumerate(path):
                x, branch, y = [stream.values(edge[key], position) for key in ['incoming', 'branch', 'merged']]
                record = stream.rows[edge['merged'], position]
                if record[5] != 'ADD' or sorted(record[6]) != sorted([edge['incoming'], edge['branch']]):
                    raise ValueError(f"Disconnected source names for {edge['merged']} at {position}")
                if x.size != width or branch.size != width or y.size != width:
                    raise ValueError('Residual width changed along the path')
                if not np.array_equal(np.add(x, branch).view('u4'), y.view('u4')):
                    raise ValueError(f"Native ADD does not reconstruct at {edge['merged']}, {position}")
                x, y = x.astype('f8'), y.astype('f8')
                nx, ny = np.linalg.norm(x), np.linalg.norm(y)
                if not (nx > 0 and ny > 0 and np.isfinite(nx+ny)):
                    raise ValueError('Undefined residual direction or size')
                turns[decision, merge] = y/ny - x/nx
                norms[decision, merge] = nx, ny
                verified += 1
        turns.flush()
        norms.flush()
        native_inputs = {label: meta for label, meta in stream.metadata.items()
                         if label.startswith(('attn_norm-', 'ffn_norm-', 'ffn_geglu-', 'kqv_out-'))}
        for label in native_inputs:
            if any((label, p) not in stream.rows for p in positions):
                raise ValueError(f'Missing matrix input {label}')
        exported = dict(case=result['id'], predictions=len(positions), input_positions=positions,
            shape=list(shape), connections=path, verified_add_rows=verified,
            native_inputs=native_inputs,
            definition='unit(merged) - unit(incoming), separately at every recorded merge and natural prediction; norms are native-vector L2 norms',
            retention='Reduced projection stream. Native matrix inputs remain in the source stream, not reconstructed from the turns.',
            source=str(source), source_result_sha256=digest(source/'result.json'),
            source_events_sha256=digest(source/'stream.jsonl'), source_values_sha256=digest(source/'stream.f32'),
            vector_sha256=digest(output/'turns.npy'), norms_sha256=digest(output/'norms.npy'))
        (output/'index.json').write_text(json.dumps(exported, indent=2)+'\n')
        return exported
    finally:
        stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--declaration', type=Path, required=True)
    args = parser.parse_args()
    check_native_run(args.output, args.declaration)
    record = export_case(args.source, args.output)
    print(json.dumps({key: record[key] for key in ['case', 'shape', 'verified_add_rows']}))
