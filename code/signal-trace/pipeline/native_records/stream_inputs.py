"""Read complete native prefix inputs without inventing absent cache writers."""
import json
from pathlib import Path

import numpy as np


def prefix_rows(directory, labels, last_position):
    directory = Path(directory)
    retained = {label: {} for label in labels}
    with (directory / 'stream.f32').open('rb') as binary, (directory / 'stream.jsonl').open() as records:
        for line in records:
            record = json.loads(line)
            label = record['label']
            if label not in retained:
                continue
            shape, axis = record['shape'], record['token_axis']
            if len(record['positions']) != shape[axis]:
                raise ValueError('Stream tensor has unresolved token positions')
            binary.seek(record['offset_bytes'])
            raw = binary.read(4 * record['values'])
            if len(raw) != 4 * record['values']:
                raise ValueError('Truncated native stream tensor')
            tensor = np.frombuffer(raw, dtype='<f4').reshape(shape, order='F')
            for row, position in enumerate(record['positions']):
                if position > last_position:
                    continue
                value = np.take(tensor, row, axis=axis).ravel(order='F').copy()
                previous = retained[label].get(position)
                if previous is not None and not np.array_equal(previous.view('u4'), value.view('u4')):
                    raise ValueError('Conflicting native values for one prefix position')
                retained[label][position] = value
    result = {}
    for label, rows in retained.items():
        missing = set(range(last_position + 1)) - rows.keys()
        if missing:
            raise ValueError(f'Missing {label} prefix position {min(missing)}')
        result[label] = np.stack([rows[p] for p in range(last_position + 1)])
    return result
