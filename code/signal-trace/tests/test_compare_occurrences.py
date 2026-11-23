"""Known connected transformations must stay connected in the comparison."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.native_records.compare_occurrences import compare_frames


class ConnectedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='a-connected-compare-', dir='/tmp')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self, name, rows, broken_source=False):
        path = self.root / name
        path.mkdir()
        offset = 0
        events = []
        with (path / 'signal-data.bin').open('wb') as data:
            def snap(values, tensor_name):
                nonlocal offset
                raw = np.asarray(values, dtype='<f4').tobytes()
                chunk = {'offset': offset, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                data.write(raw)
                offset += len(raw)
                return {'name': tensor_name, 'type': 'f32', 'shape': [len(values), 1, 1, 1],
                        'strides': [4, len(raw), len(raw), len(raw)], 'bytes': len(raw), 'chunks': [chunk]}

            states = [snap(row, f's{i}') for i, row in enumerate(rows)]
            for i, op in enumerate(('SCALE', 'SCALE', 'ADD')):
                ports = ([0] if i == 0 else [0] if i == 1 else [1, 2])
                inputs = []
                for j in ports:
                    inputs.append({'layout': states[j], 'snapshot': states[j],
                                   'producer_event': None if j == 0 else j - 1,
                                   'source_status': 'runtime input' if j == 0 else 'recorded'})
                if broken_source and i == 2:
                    inputs[1]['producer_event'] = None
                    inputs[1]['source_status'] = 'unresolved backend copy'
                events.append({'event': i, 'graph_execution': 1, 'phase': 'decode',
                               'input_position': 10, 'name': f'operation-{i}', 'operation': op,
                               'inputs': inputs, 'output': states[i + 1]})
        (path / 'signal-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        return path

    def mapping(self):
        return {'relation': 'known fixture; not research evidence', 'frames': [
            {'left_execution': 1, 'right_execution': 1, 'reason': 'same supplied step'}]}

    def test_preserves_each_transformation_and_both_merge_branches(self):
        # x -> 2x and 3x -> sum. Different x, same transformations.
        left = self.fixture('left', [[1, 2], [2, 4], [3, 6], [5, 10]])
        right = self.fixture('right', [[2, 1], [4, 2], [6, 3], [10, 5]])
        out = self.root / 'comparison'
        compare_frames(left, right, self.mapping(), out)
        events = [json.loads(line) for line in (out / 'contrasts.jsonl').read_text().splitlines()]
        self.assertEqual(len(events), 3)
        values = np.fromfile(out / 'contrasts.bin', dtype='<f4')
        def vector(record):
            return values[record['offset_bytes'] // 4: (record['offset_bytes'] + record['bytes']) // 4]
        np.testing.assert_array_equal(vector(events[0]['output']), [2, -2])
        np.testing.assert_array_equal(vector(events[1]['output']), [3, -3])
        np.testing.assert_array_equal(vector(events[2]['output']), [5, -5])
        self.assertEqual([x['left']['producer_event'] for x in events[2]['inputs']], [0, 1])
        np.testing.assert_array_equal(vector(events[2]['inputs'][0]['contrast']), [2, -2])
        np.testing.assert_array_equal(vector(events[2]['inputs'][1]['contrast']), [3, -3])
        report = json.loads((out / 'comparison.json').read_text())
        self.assertEqual(report['left_unpaired_executions'], [])
        self.assertEqual(report['right_unpaired_executions'], [])

    def test_missing_producer_is_not_repaired_by_matching_names(self):
        left = self.fixture('left', [[1, 2], [2, 4], [3, 6], [5, 10]])
        right = self.fixture('right', [[2, 1], [4, 2], [6, 3], [10, 5]], broken_source=True)
        with self.assertRaisesRegex(ValueError, 'source-port structure'):
            compare_frames(left, right, self.mapping(), self.root / 'comparison')

    def test_a_pair_requires_an_explicit_correspondence_reason(self):
        left = self.fixture('left', [[1, 2], [2, 4], [3, 6], [5, 10]])
        with self.assertRaisesRegex(ValueError, 'reason'):
            compare_frames(left, left, {'relation': 'fixture', 'frames': [
                {'left_execution': 1, 'right_execution': 1}]}, self.root / 'comparison')


if __name__ == '__main__':
    unittest.main()
