"""CPU qualification of typed state, native byte transactions and explicit recovery."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import unittest.mock

import numpy as np

from pipeline.weight_edit import full_state as state
from pipeline.weight_edit.a2 import to_bf16


ROOT = Path(__file__).resolve().parents[3]


class FullStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.fd = os.memfd_create('typed-native-cpu-fixture')
        os.ftruncate(self.fd, 96)
        self.specs = (
            state.TensorSpec('matrix', (2, 3), 'bf16', 0),
            state.TensorSpec('norm', (3,), 'f32', 16),
            state.TensorSpec('table', (4, 3), 'bf16', 32, True, ('tied_head',)),
        )
        self.seed = {'model_sha256': 'fixture-base'}
        self.identity = {'run': 'fixture', 'generation': 21}
        os.pwrite(self.fd, to_bf16(np.ones(6, '<f4')).tobytes(), 0)
        os.pwrite(self.fd, np.ones(3, '<f4').tobytes(), 16)
        os.pwrite(self.fd, to_bf16(np.ones(12, '<f4')).tobytes(), 32)
        self.synced = []
        self.s = state.FullState(self.fd, self.specs, model_identity=self.identity,
                                base_identity=self.seed, iteration=21, chunk_elements=2,
                                journal=self.folder / 'working.json')

    def tearDown(self):
        os.close(self.fd)
        self.temporary.cleanup()

    def sync(self, name, start, values):
        self.synced.append((name, start, values.copy()))

    def directions(self, rows=(1,)):
        return {'matrix': np.full((2, 3), .001, '<f4'),
                'norm': np.full(3, 2 ** -25, '<f4'),
                'table': np.full((len(rows), 3), .001, '<f4')}

    def update(self, rows=(1,), directions=None):
        directions = self.directions(rows) if directions is None else directions
        selected = {'matrix': None, 'norm': None, 'table': rows}
        next_id = dict(self.identity, generation=self.s.iteration + 1)
        # Candidate pass for the installed JVP is read-only, including remainders.
        before = os.pread(self.fd, 96, 0)
        candidates = {n: list(self.s.preview(n, d, 1., rows=selected[n])) for n, d in directions.items()}
        self.assertEqual(os.pread(self.fd, 96, 0), before)
        self.s.begin(selected, expected_identity=self.s.model_identity, next_identity=next_id, sync=self.sync)
        for name, direction in directions.items():
            for index, chunk in enumerate(self.s.preview(name, direction, 1., rows=selected[name])):
                self.assertEqual(chunk.raw, candidates[name][index].raw)
                self.s.install(chunk)
        return self.s.finish()

    def test_f32_remainder_accumulates_sub_ulp_requests(self):
        base = np.ones(1, '<f4')
        residual = np.zeros(1, '<f4')
        direction = np.full(1, 2 ** -25, '<f4')
        endpoints = []
        for _ in range(4):
            _, base, residual, _, _ = state.accumulate(base, residual, direction, 1., 'f32')
            endpoints.append(float(base[0]))
        self.assertEqual(endpoints[:2], [1., 1.])
        self.assertEqual(endpoints[-1], float(np.nextafter(np.float32(1), np.float32(2))))
        self.assertEqual(float(residual[0]), 0.)

    def test_bf16_matches_historical_arithmetic_byte_for_byte(self):
        path = ROOT / 'studies/04-round-2-overlap/runs/vector-chance-loss-rise-diagnosis/empirical-eta-continuation/bf16_accumulation.py'
        spec = importlib.util.spec_from_file_location('historical_accumulation_cpu_check', path)
        historical = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(historical)
        rng = np.random.default_rng(317)
        base = state.decode(to_bf16(rng.normal(size=(7, 9)).astype('<f4')).tobytes(), 'bf16').reshape(7, 9)
        remainder = (rng.normal(size=base.shape) * 1e-4).astype('<f4')
        direction = rng.normal(size=base.shape).astype('<f4')
        for length in (0., .26989262219171917, 1.):
            actual = state.accumulate(base, remainder, direction, length, 'bf16')
            wanted, retained = historical.step(base, remainder, direction, length)
            self.assertEqual(actual[1].tobytes(), wanted.tobytes())
            self.assertEqual(actual[2].tobytes(), retained.tobytes())

    def test_bf16_encoder_matches_all_high_words_at_rounding_boundaries(self):
        # Every sign/exponent/high mantissa, including subnormal/NaN payload bits,
        # at exact, below-half, half-even/odd, above-half and carry low words.
        high = np.arange(65536, dtype='<u4') << 16
        for low in (0, 0x7fff, 0x8000, 0x8001, 0xffff):
            values = (high | np.uint32(low)).view('<f4')
            before = values.tobytes()
            self.assertEqual(state._bf16_bytes(values), to_bf16(values).tobytes())
            self.assertEqual(values.tobytes(), before)

    def test_optimized_accumulation_matches_retained_expression_bytes_and_receipts(self):
        # Frozen expression order from pre-optimization full_state (1083e687),
        # deliberately retaining its explicit casts and out-of-place operations.
        rng = np.random.default_rng(317)
        for dtype in ('bf16', 'f32'):
            for scale in (1e-30, 1., 1e20):
                base = (rng.normal(size=(9, 7)) * scale).astype('<f4').T
                if dtype == 'bf16':
                    base = state.decode(to_bf16(base).tobytes(), dtype).reshape(base.shape)
                remainder = (rng.normal(size=base.shape) * scale * 1e-4).astype('<f4')
                direction = (rng.normal(size=base.shape) * scale).astype('<f4')
                for length in (0., .26989262219171917, 1.):
                    with self.subTest(dtype=dtype, scale=scale, length=length):
                        wanted = base.astype(np.float64)
                        wanted += remainder
                        scaled = direction.astype(np.float64)
                        scaled *= length
                        wanted += scaled
                        wanted32 = wanted.astype('<f4')
                        raw = to_bf16(wanted32).tobytes() if dtype == 'bf16' else wanted32.tobytes()
                        stored = state.decode(raw, dtype).reshape(base.shape)
                        residual = (wanted32 - stored if dtype == 'bf16' else
                                    (wanted - stored.astype(np.float64)).astype('<f4'))
                        delta = stored.astype(np.float64) - base.astype(np.float64)
                        exact = np.array_equal(delta.astype('<f4').astype(np.float64), delta)
                        expected_receipt = dict(requested_nonzero_entries=int(np.count_nonzero(wanted != base)),
                            fresh_direction_nonzero_entries=int(np.count_nonzero(direction)),
                            remainder_before_squared=float(np.einsum('i,i->', remainder.reshape(-1),
                                remainder.reshape(-1), dtype=np.float64)))
                        before = [v.tobytes() for v in (base, remainder, direction)]
                        receipt = {}
                        actual = state.accumulate(base, remainder, direction, length, dtype, receipt=receipt)
                        self.assertEqual(actual[0], raw)
                        for observed, expected in zip(actual[1:4], (stored, residual, delta)):
                            self.assertEqual(observed.tobytes(), expected.tobytes())
                        self.assertEqual(actual[4], exact)
                        self.assertEqual(receipt, expected_receipt)
                        self.assertEqual([v.tobytes() for v in (base, remainder, direction)], before)

    def test_preview_does_not_modify_borrowed_dense_or_sparse_remainders(self):
        self.s.remainders['matrix'] = np.full((2, 3), .001, '<f4')
        self.s.remainders['table'] = {1: np.full(3, .002, '<f4')}
        self.assertTrue(np.shares_memory(self.s._remainder_span(self.specs[0], 0, 2), self.s.remainders['matrix']))
        self.assertTrue(np.shares_memory(self.s._remainder_span(self.specs[2], 3, 2), self.s.remainders['table'][1]))
        old_dense, old_sparse = self.s.remainders['matrix'].tobytes(), self.s.remainders['table'][1].tobytes()
        list(self.s.preview('matrix', self.directions()['matrix'], 1.))
        list(self.s.preview('table', self.directions()['table'], 1., rows=(1,)))
        self.assertEqual(self.s.remainders['matrix'].tobytes(), old_dense)
        self.assertEqual(self.s.remainders['table'][1].tobytes(), old_sparse)

    def test_rounding_ties_signed_zero_and_native_subnormal_conversion(self):
        values = np.array([1 + 1 / 256, 1 + 3 / 256, -0., np.nextafter(np.float32(0), np.float32(1))], '<f4')
        result = state.accumulate(values, np.zeros(4, '<f4'), np.zeros(4, '<f4'), 0., 'bf16')
        np.testing.assert_array_equal(np.frombuffer(result[0], '<u2'), [0x3f80, 0x3f82, 0, 0])
        # Ordered +0 additions, just as historical arithmetic, may clear -0.
        tiny = np.array([np.nextafter(np.float32(0), np.float32(1))], '<f4')
        self.assertEqual(state.accumulate(tiny, np.zeros(1, '<f4'), np.zeros(1, '<f4'), 0., 'f32')[0], tiny.tobytes())

    def test_nonfinite_and_native_overflow_fail(self):
        base = np.ones(1, '<f4')
        for length, direction in [(float('nan'), base), (-1., base), (1., np.array([np.inf], '<f4')),
                                  (float(np.finfo(np.float64).max), base)]:
            for dtype in ('bf16', 'f32'):
                with self.subTest(length=length, dtype=dtype), self.assertRaises(ValueError):
                    state.accumulate(base, np.zeros(1, '<f4'), direction, length, dtype)

    def test_actual_displacement_reports_f32_cast_loss(self):
        base = np.array([1e20], '<f4')
        raw, stored, remainder, delta, exact = state.accumulate(base, np.zeros(1, '<f4'), -base, 1., 'f32')
        self.assertTrue(exact)
        # Widely separated finite F32 endpoints need not have an exact F32 difference.
        result = state.accumulate(np.array([1.], '<f4'), np.zeros(1, '<f4'), np.array([2 ** -30], '<f4'), 2 ** 60, 'f32')
        self.assertFalse(result[-1])
        self.assertEqual(float(result[3][0]), float(result[1][0]) - 1.)

    def test_mixed_transaction_covers_all_bytes_and_callback(self):
        receipt = self.update()
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(receipt['iteration'], 22)
        self.assertEqual({v[0] for v in self.synced}, {'matrix', 'norm', 'table'})
        self.assertEqual(os.pread(self.fd, 4, 12), bytes(4))
        self.assertTrue(np.any(self.s.remainder('norm')))
        self.assertEqual(set(self.s.remainders['table']), {1})
        self.assertEqual(self.s.remainder('tied_head', [1]).tobytes(), self.s.remainder('table', [1]).tobytes())
        inventory = receipt['transition']['committed']
        self.assertEqual(set(inventory), {'matrix', 'norm', 'table'})
        self.assertTrue(all(v['complete'] for v in inventory.values()))
        self.assertEqual(inventory['matrix']['elements'], 6)
        self.assertEqual(inventory['matrix']['requested_nonzero_entries'], 6)
        self.assertEqual(inventory['matrix']['changed_entries'], 0)

    def test_an_install_writes_only_the_pages_whose_bytes_change(self):
        page, offset = state.mmap.PAGESIZE, 100
        before = bytes(range(256)) * (4 * page // 256)
        written = []

        def write(fd, start, raw):
            written.append((start, len(raw)))
            os.pwrite(fd, raw, start)

        fd = os.open(self.folder / 'native', os.O_CREAT | os.O_RDWR)
        try:
            for changed, spans in (
                    ((), []),                                                    # nothing differs: nothing is written
                    ((0,), [(offset, page - offset)]),                           # the span's first, partial page
                    ((page, 2 * page - offset), [(page, 2 * page)]),             # neighbouring pages go as one write
                    ((5, 3 * page), [(offset, page - offset), (3 * page, page)]),
                    ((4 * page - 1,), [(4 * page, offset)])):                    # the span's last, partial page
                raw = bytearray(before)
                for index in changed:
                    raw[index] ^= 0xff
                os.pwrite(fd, before, offset)
                written.clear()
                with unittest.mock.patch.object(state, '_write', write):
                    state._write_changed(fd, offset, before, bytes(raw))
                self.assertEqual(written, spans)
                self.assertEqual(os.pread(fd, len(raw), offset), bytes(raw))
        finally:
            os.close(fd)

    def test_an_installed_span_that_keeps_its_bytes_is_not_written_again(self):
        writes = []
        with unittest.mock.patch.object(state, '_write', lambda fd, start, raw: writes.append(start)):
            # Requests under half a stored step change no stored value: only the remainders move. The BF16
            # tensors' request is one F32 holds beside 1.0, since their remainder is formed from the F32 sum.
            receipt = self.update(directions={'matrix': np.full((2, 3), 2 ** -20, '<f4'),
                                              'norm': np.full(3, 2 ** -25, '<f4'),
                                              'table': np.full((1, 3), 2 ** -20, '<f4')})
        self.assertEqual(receipt['status'], 'complete')
        self.assertEqual(writes, [])
        self.assertTrue(np.all(self.s.remainder('matrix')))
        self.assertTrue(np.all(self.s.remainder('norm')))

    def test_sparse_rows_survive_change_and_include_dormant_recurrence(self):
        self.update((1,))
        before = self.s.remainder('table', [1]).copy()
        compact = np.full((1, 3), .001, '<f4')
        expanded, ids = self.s.expand_rows('table', compact, (3,))
        self.assertEqual(ids, (1, 3))
        np.testing.assert_array_equal(expanded[0], np.zeros(3, '<f4'))
        directions = self.directions(ids)
        directions['table'] = expanded
        self.update(ids, directions)
        self.assertEqual(set(self.s.remainders['table']), {1, 3})
        # Dormant E is applied with S=0; this particular sub-grid E remains carried.
        np.testing.assert_array_equal(self.s.remainder('table', [1]), before)
        with self.assertRaises(ValueError):
            self.s.begin({'matrix': None, 'norm': None, 'table': (3,)},
                         expected_identity=self.s.model_identity, next_identity={'generation': 23}, sync=self.sync)

    def test_requested_counts_include_remainder_and_actual_cancellation(self):
        self.s.remainders['matrix'] = np.full((2, 3), .001, '<f4')
        dormant = list(self.s.preview('matrix', np.zeros((2, 3), '<f4'), 1.))
        self.assertEqual(sum(c.requested_nonzero_entries for c in dormant), 6)
        self.assertEqual(sum(c.fresh_direction_nonzero_entries for c in dormant), 0)
        self.assertTrue(all(not np.count_nonzero(c.displacement) for c in dormant))
        canceled = list(self.s.preview('matrix', -self.s.remainders['matrix'], 1.))
        self.assertEqual(sum(c.requested_nonzero_entries for c in canceled), 0)
        self.assertEqual(sum(c.fresh_direction_nonzero_entries for c in canceled), 6)
        self.assertTrue(all(not np.count_nonzero(c.remainder) for c in canceled))

    def test_missing_duplicate_alias_or_stale_chunks_refuse(self):
        candidate = next(self.s.preview('matrix', self.directions()['matrix'], 1.))
        with self.assertRaises(ValueError):
            self.s.begin({'matrix': None}, expected_identity=self.identity, next_identity={'generation': 22}, sync=self.sync)
        self.s.begin({'matrix': None, 'norm': None, 'table': []}, expected_identity=self.identity,
                     next_identity={'generation': 22}, sync=self.sync)
        self.s.install(candidate)
        with self.assertRaises(ValueError): self.s.install(candidate)
        with self.assertRaises(ValueError): self.s.finish()
        self.assertFalse(self.s.checkpoint(self.folder / 'partial')['complete'])

    def test_callback_failure_records_partial_state_and_rejects_recovery(self):
        candidate = next(self.s.preview('matrix', np.ones((2, 3), '<f4'), 1.))
        def failing(*args): raise RuntimeError('derivative model copy failed')
        self.s.begin({'matrix': None, 'norm': None, 'table': []}, expected_identity=self.identity,
                     next_identity={'generation': 22}, sync=failing)
        with self.assertRaises(RuntimeError): self.s.install(candidate)
        self.assertEqual(self.s.status, 'incomplete')
        self.assertEqual(state.decode(os.pread(self.fd, 4, 0), 'bf16').tolist(), [2., 2.])
        saved = self.s.checkpoint(self.folder / 'failed')
        self.assertFalse(saved['complete'])
        self.assertEqual(saved['transition']['in_progress']['name'], 'matrix')
        with self.assertRaises(ValueError):
            state.FullState.restore(self.fd, self.folder / 'failed', specs=self.specs,
                expected_base_identity=self.seed, expected_model_identity=self.identity,
                chunk_elements=2, sync=self.sync)

    def test_checkpoint_restore_exact_native_and_dense_sparse_remainders(self):
        self.update((1, 3))
        expected = os.pread(self.fd, 96, 0)
        saved = self.s.checkpoint(self.folder / 'exit')
        self.assertEqual(saved['native']['bytes'], sum(v.nbytes for v in self.specs))
        os.pwrite(self.fd, bytes(96), 0)
        restored = state.FullState.restore(self.fd, self.folder / 'exit', specs=self.specs,
            expected_base_identity=self.seed, expected_model_identity=self.s.model_identity,
            chunk_elements=3, sync=self.sync, journal=self.folder / 'restored.json')
        self.assertEqual(os.pread(self.fd, 96, 0), expected)
        self.assertEqual(restored.iteration, 22)
        for name, rows in [('matrix', None), ('norm', None), ('table', [1, 3])]:
            self.assertEqual(restored.remainder(name, rows).tobytes(), self.s.remainder(name, rows).tobytes())
        with self.assertRaises(FileExistsError): self.s.checkpoint(self.folder / 'exit')

    def test_restore_rejects_corruption_before_any_native_write(self):
        self.update()
        self.s.checkpoint(self.folder / 'exit')
        path = self.folder / 'exit/native.bin'
        raw = path.read_bytes(); path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
        before = os.pread(self.fd, 96, 0)
        with self.assertRaises(ValueError):
            state.FullState.restore(self.fd, self.folder / 'exit', specs=self.specs,
                expected_base_identity=self.seed, expected_model_identity=self.s.model_identity,
                chunk_elements=2, sync=self.sync)
        self.assertEqual(os.pread(self.fd, 96, 0), before)

    def test_existing_journal_cannot_be_overwritten_by_constructor(self):
        before = (self.folder / 'working.json').read_bytes()
        with self.assertRaises(ValueError):
            state.FullState(self.fd, self.specs, model_identity=self.identity, base_identity=self.seed,
                            chunk_elements=2, journal=self.folder / 'working.json')
        self.assertEqual((self.folder / 'working.json').read_bytes(), before)

    def test_identity_handoff_changes_no_native_or_remainder_bytes(self):
        self.update()
        before = os.pread(self.fd, 96, 0)
        remainder = self.s.remainder('matrix').tobytes()
        old_identity = self.s.model_identity
        provenance = {'record': 'old/final-state.json', 'sha256': 'fixture-checkpoint-binding'}
        receipt = self.s.rebind_complete_identity({'run': 'successor', 'generation': 0},
                                                 iteration=0, provenance=provenance)
        self.assertEqual(receipt['iteration'], 0)
        self.assertEqual(receipt['transition']['source_identity'], old_identity)
        self.assertEqual(receipt['transition']['source_iteration'], 22)
        self.assertEqual(receipt['transition']['provenance'], provenance)
        self.assertEqual(os.pread(self.fd, 96, 0), before)
        self.assertEqual(self.s.remainder('matrix').tobytes(), remainder)

    def test_restore_callback_failure_keeps_new_journal_incomplete(self):
        self.update()
        self.s.checkpoint(self.folder / 'exit')
        def failing(*args): raise RuntimeError('restored derivative model copy failed')
        with self.assertRaises(RuntimeError):
            state.FullState.restore(self.fd, self.folder / 'exit', specs=iter(self.specs),
                expected_base_identity=self.seed, expected_model_identity=self.s.model_identity,
                chunk_elements=2, sync=failing, journal=self.folder / 'failed-restore.json')
        record = json.loads((self.folder / 'failed-restore.json').read_text())
        self.assertEqual(record['status'], 'incomplete')
        self.assertEqual(record['transition']['in_progress'], {'name': 'matrix', 'start': 0})

    def test_retained317_metadata_matches_physical_native_and_remainder_layout(self):
        bindings = json.loads((ROOT / 'studies/04-round-2-overlap/runs/sent-overlap/wrong-inputs/tensors.json').read_text())
        saved = json.loads((ROOT / 'studies/04-round-2-overlap/runs/vector-chance-full-weight-training/final-state.json').read_text())
        self.assertTrue(saved['accumulator']['complete'])
        self.assertEqual(saved['weight_state'], 'complete')
        self.assertIsNone(saved['pending_install'])
        self.assertEqual(saved['update'], saved['accumulator']['iteration'])
        self.assertEqual(saved['model_identity'], saved['accumulator']['model_identity'])
        at = 0
        for row in saved['accumulator']['layout']:
            spec = state.TensorSpec.from_binding(bindings[row['name']])
            self.assertEqual(spec.native_type, 'bf16')
            self.assertEqual(list(spec.shape), row['shape'])
            self.assertEqual(row['offset_bytes'], at)
            self.assertEqual(row['bytes'], spec.size * 4)
            at += row['bytes']
        self.assertEqual(at, saved['accumulator']['bytes'])
        self.assertEqual({r['name'] for r in saved['inventory']}, {r['name'] for r in saved['accumulator']['layout']})
        for name in ('token_embd.weight', 'per_layer_token_embd.weight'):
            spec = state.TensorSpec.from_binding(bindings[name], sparse_rows=True)
            self.assertEqual(spec.shape[0], 262144)

    def legacy(self):
        raw = np.array([[.001, -.002, .003], [0., 0., .004]], '<f4').tobytes()
        (self.folder / 'old.f32').write_bytes(raw)
        np.savez(self.folder / 'old.npz', index_0=np.array([0, 5], '<i8'),
                 value_0=to_bf16(np.array([2., 3.], '<f4')))
        return dict(update=21, base_seed=self.seed, weight_state='complete', pending_install=None,
            model_identity=self.identity, inventory=[dict(name='matrix', entries=2)],
            overrides=dict(file='old.npz', sha256=state._digest(self.folder / 'old.npz')),
            accumulator=dict(format='bf16-fp32-remainder/1', file='old.f32',
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), dtype='<f4', complete=True,
                model_identity=self.identity, iteration=21,
                layout=[dict(name='matrix', shape=[2, 3], offset_bytes=0, bytes=len(raw))]))

    def test_legacy_native_and_remainder_restore_preserves_old_new_zero(self):
        record = self.legacy()
        # A seed loaded before its saved overrides is not yet the317 model.
        self.s = state.FullState(self.fd, self.specs, model_identity=self.identity,
            base_identity=self.seed, iteration=21, chunk_elements=2, complete=False)
        self.assertEqual(self.s.status, 'uninitialized')
        with self.assertRaises(ValueError):
            list(self.s.preview('matrix', self.directions()['matrix'], 1.))
        self.s.restore_legacy(record, self.folder, expected_base_seed=self.seed, sync=self.sync)
        np.testing.assert_array_equal(state.decode(os.pread(self.fd, 12, 0), 'bf16'), [2, 1, 1, 1, 1, 3])
        self.assertEqual(self.s.remainder('matrix').tobytes(), (self.folder / 'old.f32').read_bytes())
        self.assertFalse(self.s.remainder('norm').any())
        self.assertFalse(self.s.remainder('table', [0, 3]).any())
        self.assertEqual(self.s.status, 'complete')

    def test_legacy_mismatched_remainder_fails_without_native_write(self):
        record = self.legacy(); record['accumulator']['model_identity'] = {'wrong': True}
        before = os.pread(self.fd, 96, 0)
        with self.assertRaises(ValueError):
            self.s.restore_legacy(record, self.folder, expected_base_seed=self.seed, sync=self.sync)
        self.assertEqual(os.pread(self.fd, 96, 0), before)
        self.assertEqual(self.s.status, 'incomplete')

    def test_shapes_aliases_sparse_rows_and_overlapping_storage(self):
        binding = dict(name='scale', shape=[1, 1, 1, 1], native_type='f32', offset_bytes=60, bytes=4)
        self.assertEqual(state.TensorSpec.from_binding(binding).shape, (1,))
        binding.update(name='projection', shape=[3, 2, 1, 1], native_type='bf16', offset_bytes=0, bytes=12)
        self.assertEqual(state.TensorSpec.from_binding(binding).shape, (2, 3))
        with self.assertRaises(ValueError):
            state.FullState(self.fd, (*self.specs, state.TensorSpec('overlap', (1,), 'f32', 2)),
                            model_identity=self.identity, base_identity=self.seed, chunk_elements=2)
        for rows in ([1, 1], [2, 1], [-1], [4], [1.5]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                list(self.s.preview('table', np.zeros((len(rows), 3), '<f4'), 1., rows=rows))


if __name__ == '__main__':
    unittest.main()
