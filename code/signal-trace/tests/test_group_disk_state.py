import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory, TemporaryFile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipeline.weight_edit import final_experiment as baseline
from pipeline.weight_edit import full_parameters, group_disk_state, host_pages, native_key_driver, turn_pair
from tests import test_full_state


def full_state_fixture():
    """A small typed native state whose descriptor works on Python builds without Linux's memfd_create."""
    fixture = test_full_state.FullStateTests()
    with (TemporaryFile() as fd,
          patch.object(os, 'memfd_create', lambda name: os.dup(fd.fileno()), create=True)):
        fixture.setUp()
    return fixture


class DiskStateTests(unittest.TestCase):
    def test_next_install_and_checkpoint_match_memory_backing(self):
        checkpoints = []
        for disk in (False, True):
            fixture = full_state_fixture()
            store = host_pages.Store(fixture.folder, 'remainder')
            try:
                fixture.update((1, 3))
                if disk:
                    group_disk_state.page_state(fixture.s, store)
                fixture.update((0, 1, 3))
                checkpoint = fixture.s.checkpoint(fixture.folder/'final')
                checkpoints.append((checkpoint['native'], checkpoint['remainder'],
                                    (fixture.folder/'final'/'remainder.f32').read_bytes()))
            finally:
                store.close()
                fixture.tearDown()
        self.assertEqual(checkpoints[0], checkpoints[1])

    def test_native_reader_sees_installed_bytes_without_modifying_seed(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            seed, working = root/'stock', root/'working'
            seed.write_bytes(b'original')
            digest = hashlib.sha256(seed.read_bytes()).hexdigest()
            fd = group_disk_state.disk_model(working, seed, digest)
            try:
                os.pwrite(fd, b'new', 0)
                self.assertEqual(Path(f'/proc/self/fd/{fd}').read_bytes(), b'newginal')
                self.assertEqual(seed.read_bytes(), b'original')
                with self.assertRaises(FileExistsError):
                    group_disk_state.disk_model(working, seed, digest)
            finally:
                os.close(fd)
            with self.assertRaises(ValueError):
                group_disk_state.disk_model(root/'invalid', seed, 'wrong')

    def test_native_storage_receipt_binds_readonly_alias_and_disk_size(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            location = root/'run'/'working-files'
            location.mkdir(parents=True)
            model = location/'model.gguf'
            model.write_bytes(b'fixture')
            model.chmod(0o600)
            declaration = dict(outputs=['run'], disk_working_state=dict(model_bytes=7))
            with (model.open('r+b') as writer, model.open('rb') as reader,
                  patch.object(native_key_driver.baseline, 'ROOT', root)):
                receipt = native_key_driver.storage_receipt(None, declaration, writer.fileno(), reader.fileno())
                self.assertEqual(receipt['file_bytes'], 7)
                with self.assertRaises(ValueError):
                    native_key_driver.storage_receipt(None, declaration, writer.fileno(), writer.fileno())
                declaration['disk_working_state']['model_bytes'] = 6
                with self.assertRaises(ValueError):
                    native_key_driver.storage_receipt(None, declaration, writer.fileno(), reader.fileno())

    def test_remainders_keep_dense_arrays_in_the_store_and_rows_in_memory(self):
        with TemporaryDirectory() as folder:
            store, changes = host_pages.Store(folder, 'remainder'), []
            first = np.array([[-0., .5], [.25, 0.]], dtype='<f4')
            second = np.array([1., 2., 3.], dtype='<f4')
            rows = {2: np.array([5.], dtype='<f4')}
            state = SimpleNamespace(remainders={'a': first, 'table': rows}, preview=None, checkpoint=None)
            try:
                group_disk_state.page_state(state, store, changed=lambda: changes.append(len(state.remainders)))
                kept = state.remainders
                self.assertIs(kept['table'], rows)
                self.assertIs(kept.setdefault('table', {}), rows)
                self.assertIsNone(kept.get('absent'))
                self.assertEqual(kept.get('absent', {}), {})
                self.assertNotIn('absent', kept)
                self.assertEqual(kept['a'].tobytes(), first.tobytes())   # the signed zero is kept
                self.assertIsNot(kept['a'], first)
                kept['b'] = second   # a new dense remainder moves into the file as it is created
                kept['b'][0] = 4.
                kept['a'][1, 1] = -1.   # asking for one array lets the last one go; its values stay
                self.assertEqual(kept['b'].tolist(), [4., 2., 3.])
                self.assertEqual(kept['a'].tolist(), [[-0., .5], [.25, -1.]])
                self.assertEqual(second.tolist(), [1., 2., 3.])
                kept.settle()
                layout = kept.layout()
                self.assertEqual(set(layout), {'a', 'b'})
                self.assertEqual(layout['a'], dict(file=store.path.name, offset_bytes=0, shape=[2, 2], dtype='<f4'))
                self.assertEqual(layout['b']['offset_bytes'], host_pages.PAGE)
                with store.path.open('rb') as stream:
                    stream.seek(layout['b']['offset_bytes'])
                    self.assertEqual(stream.read(12), kept['b'].tobytes())
                self.assertEqual(changes, [2, 2, 3])   # moved in, then one new dense remainder
            finally:
                store.close()

    def test_restore_reads_dense_remainders_straight_into_the_working_file(self):
        fixture = full_state_fixture()
        try:
            fixture.update((1, 3))
            fixture.s.checkpoint(fixture.folder/'saved'/'checkpoint')
            expected = {name: value if isinstance(value, dict) else np.array(value)
                        for name, value in fixture.s.remainders.items()}
            run = fixture.folder/'run'
            run.mkdir()
            working = group_disk_state.Working(SimpleNamespace(), run)

            def original(root, candidate):
                return type(fixture.s).restore(fixture.fd, (root/candidate['checkpoint']['record']).parent,
                    specs=tuple(fixture.s.specs.values()), expected_base_identity=fixture.s.base_identity,
                    expected_model_identity=fixture.s.model_identity, chunk_elements=fixture.s.chunk_elements,
                    sync=lambda name, start, values: None)

            try:
                restored = working.restore(original)(fixture.folder, dict(checkpoint=dict(record='saved/checkpoint/state.json')))
                self.assertIs(working.remainders, restored.remainders)
                for name, value in expected.items():
                    if isinstance(value, dict):
                        self.assertEqual(set(restored.remainders[name]), set(value))
                        continue
                    self.assertEqual(host_pages.location(restored.remainders[name])[0], working.remainder.path.name)
                    self.assertEqual(restored.remainders[name].tobytes(), value.tobytes())
                layout = json.loads((working.folder/'remainder-layout.json').read_text())
                self.assertEqual(set(layout), {name for name, value in expected.items() if not isinstance(value, dict)})
                # The restored state goes on exactly as the one it was saved from.
                fixture.s, before = restored, fixture.s
                fixture.identity = dict(fixture.identity, generation=before.iteration - 1)
                self.assertEqual(restored.checkpoint(fixture.folder/'again')['remainder'],
                                 before.checkpoint(fixture.folder/'before')['remainder'])
            finally:
                working.close_scratch()
                working.remainder.close()
        finally:
            fixture.tearDown()

    def test_host_model_weights_are_the_stored_values_in_the_leaves_file(self):
        stored = {'a': torch.arange(6.).view(2, 3).to(torch.bfloat16), 'b': torch.tensor([-0., 1e-42, 3.5])}

        class Model:
            def __init__(self, file, tensors, device='cuda'):
                self.w = {name: module.stored(file, t).to(device) for name, t in tensors.items()}

        module = SimpleNamespace(Model=Model, stored=lambda file, t: stored[t].clone())
        with TemporaryDirectory() as folder:
            working = group_disk_state.Working(SimpleNamespace(), Path(folder))
            try:
                load = working.load(lambda name, path: module)
                self.assertIs(load('another_module', None).Model, Model)
                paged = load(group_disk_state.MODEL, None).Model('file', dict(a='a', b='b'), device='cpu')
                self.assertEqual(working.leaves.size, host_pages.file_bytes(v.numel() * 4 for v in stored.values()))
                for name, value in stored.items():
                    self.assertEqual(paged.w[name].dtype, torch.float32)
                    self.assertEqual(host_pages.location(paged.w[name])[0], working.leaves.path.name)
                    self.assertEqual(paged.w[name].numpy().tobytes(), value.float().numpy().tobytes())
                working.settle()
                self.assertEqual(paged.w['a'].flatten().tolist(), list(range(6)))
            finally:
                working.close_scratch()
                working.remainder.close()

    def test_an_installed_span_reaches_the_weights_file_as_the_elements_it_changes(self):
        calls = []

        def frozen(model, state, name, start, values):
            calls.append(name)

        state = SimpleNamespace(specs=dict(a=SimpleNamespace(sparse_rows=False), table=SimpleNamespace(sparse_rows=True)))
        with TemporaryDirectory() as folder:
            working = group_disk_state.Working(SimpleNamespace(), Path(folder))
            try:
                leaf = working.leaves.tensor((2, 3))
                leaf.copy_(torch.tensor([[0., 1., 2.], [3., 0., 5.]]))
                model = SimpleNamespace(weights=dict(a=leaf.requires_grad_(), table=torch.zeros(1, 3)))
                sync = working.sync(frozen)
                installed = np.array([1., 2.5, 3., -0.], '<f4')   # two keep their bits, one changes, one changes sign only
                sync(model, state, 'a', 1, installed)
                self.assertEqual(calls, [])
                self.assertEqual(leaf.detach().numpy().tobytes(),
                                 np.array([[0., 1., 2.5], [3., -0., 5.]], '<f4').tobytes())
                sync(model, state, 'a', 1, installed)   # an install of the values already held stores nothing
                self.assertEqual(leaf.detach().numpy().tobytes(),
                                 np.array([[0., 1., 2.5], [3., -0., 5.]], '<f4').tobytes())
                sync(model, state, 'table', 0, np.zeros(3, '<f4'))
                self.assertEqual(calls, ['table'])   # a table's retained rows keep the frozen copy
            finally:
                working.close_scratch()
                working.remainder.close()

    def limits_fixture(self):
        """Three blocks, the last reusing the keys before it; each dense tensor fits one page."""
        architecture = dict(layers=3, layers_sharing_keys=1, windowed=[False] * 3, width=4)
        names = [*full_parameters.GLOBAL, *(f'blk.{layer}.{part}.weight' for layer in range(3)
                                           for part in full_parameters.PARTS)]
        shapes = {'token_embd.weight': [512, 4000, 1, 1], 'per_layer_token_embd.weight': [1024, 4000, 1, 1]}
        tensors = {name: dict(native_type='bf16', shape=shapes.get(name, [5, 3, 1, 1]), offset_bytes=0,
                              bytes=2 * np.prod(shapes.get(name, [5, 3])).item()) for name in names}
        return group_disk_state.file_limits(tensors, architecture, 123)

    def test_working_file_limits_follow_the_stored_shapes_in_whole_pages(self):
        page, found = host_pages.PAGE, self.limits_fixture()
        # 55 dense tensors; the forward never reads the last block's own three key tensors; six more have
        # no path to the loss or are not eligible.
        self.assertEqual(found, dict(model=123, leaves=52 * page, remainder=49 * page))
        # No file for the direction, the displacement or a native trace.
        self.assertEqual(group_disk_state.total(found), 123 + (52 + 49) * page)

    def test_pending_bytes_are_what_the_working_files_can_still_grow(self):
        page, limits = host_pages.PAGE, self.limits_fixture()
        with TemporaryDirectory() as folder:
            working = group_disk_state.Working(SimpleNamespace(), Path(folder), limits)
            try:
                self.assertEqual(working.pending(), group_disk_state.total(limits))
                working.model.write_bytes(b'x' * 100)
                working.remainder.array((page // 4 + 1,))
                working.leaves.array((60 * page // 4,))   # past its limit: nothing more is counted for it
                self.assertEqual(working.pending(), group_disk_state.total(limits) - 100 - 2 * page - limits['leaves'])
            finally:
                working.close_scratch()
                working.remainder.close()

    def test_turns_are_formed_one_merge_at_a_time_with_the_same_values(self):
        generator = torch.Generator().manual_seed(3)
        forward = {key: [torch.randn(5, 4, generator=generator) for _ in range(3)] for key in ('incoming', 'merged')}
        whole = turn_pair.turns(forward, expected_merges=3)
        lazy = turn_pair.Turns(forward, 3)
        self.assertEqual(len(lazy), 3)
        for index in (0, 1, 2, -1):
            self.assertTrue(torch.equal(lazy[index], whole[index]))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(lazy, whole, strict=True)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(lazy[1:], whole[1:], strict=True)))
        with self.assertRaises(IndexError):
            lazy[3]
        with self.assertRaises(ValueError):
            turn_pair.Turns(forward, 2)
        forward['incoming'][2] = torch.randn(4, 4)
        with self.assertRaises(ValueError):
            turn_pair.Turns(forward, 3)[2]
        requires = {key: [value.clone().requires_grad_() for value in values] for key, values in whole and forward.items()}
        requires['incoming'][2] = requires['incoming'][0].detach().clone().requires_grad_()
        turn_pair.Turns(requires, 3)[2].sum().backward()   # a lazy turn keeps its gradient
        self.assertIsNotNone(requires['incoming'][2].grad)

    def test_a_fresh_native_trace_is_read_from_its_capture_stream_one_merge_at_a_time(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            stream = root/'capture'/'raw'/'member-a'/'stream.f32'
            stream.parent.mkdir(parents=True)
            np.arange(48, dtype='<f4').tofile(stream)
            takes = []

            def taken(values, entry, positions):
                takes.append(entry)
                return np.array(values[entry][positions])

            def raw_forward(raw, merges, stray=False):   # the frozen reader's shape: what it returns is what it took
                values = np.memmap(Path(raw)/'stream.f32', dtype='<f4', mode='r', shape=(6, 4, 2))
                incoming, merged = [], []
                for merge in range(merges):
                    before, branch, after = (held.states_reading.taken(values, 3 * merge + part, [0, 2, 3]) for part in range(3))
                    self.assertEqual(np.add(before, branch).shape, after.shape)
                    incoming.append(torch.from_numpy(before))
                    merged.append(torch.from_numpy(after + branch if stray else after))
                return dict(incoming=incoming, merged=merged, kept='as it was')

            held = SimpleNamespace(raw_forward=raw_forward, states_reading=SimpleNamespace(taken=taken))
            expected = raw_forward(stream.parent, 2)
            (root/'run-folder').mkdir()
            working = group_disk_state.Working(SimpleNamespace(held=held), root/'run-folder')
            try:
                with working.native_traces():
                    native = held.raw_forward(stream.parent, 2)
                    self.assertEqual(native['kept'], 'as it was')
                    for key in ('incoming', 'merged'):
                        self.assertIsInstance(native[key], group_disk_state.Stream)
                        self.assertEqual(len(native[key]), 2)
                    del takes[:]
                    first = native['merged'][1]
                    self.assertEqual(takes, [5])   # one merge asked for, one merge read
                    self.assertEqual((first.dtype, first.numpy().tobytes()),
                                     (torch.float32, expected['merged'][1].numpy().tobytes()))
                    self.assertIsNone(host_pages.location(first))   # no file of its own
                    for key in ('incoming', 'merged'):
                        for value, wanted in zip(native[key], expected[key], strict=True):
                            self.assertEqual(value.numpy().tobytes(), wanted.numpy().tobytes())
                        self.assertTrue(torch.equal(native[key][-1], expected[key][-1]))
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(native[key][1:], expected[key][1:], strict=True)))
                        with self.assertRaises(IndexError):
                            native[key][2]
                    turns = turn_pair.Turns(native, 2)
                    whole = turn_pair.turns(expected, expected_merges=2)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(turns, whole, strict=True)))
                    with self.assertRaises(ValueError):   # a merge the reader did not take from the stream has no cell
                        held.raw_forward(stream.parent, 2, stray=True)
                    self.assertIs(held.states_reading.taken, taken)
                self.assertIs(held.raw_forward, raw_forward)
            finally:
                working.close_scratch()
                working.remainder.close()

    def test_the_streamed_trace_is_what_the_frozen_reader_returns_through_the_key_filter(self):
        app = baseline.controller(baseline.MEASURED)
        count, width = 3, 4
        generator = np.random.default_rng(5)
        rows = {name: generator.standard_normal((count, width)).astype('<f4') for name in ('a0', 'b0', 'b1')}
        rows['c0'] = np.add(rows['a0'], rows['b0'])
        rows['c1'] = np.add(rows['c0'], rows['b1'])
        sources = dict(c0=['a0', 'b0'], c1=['c0', 'b1'])
        nodes = [dict(merge=0, incoming='a0', branch='b0', connection='c0'),
                 dict(merge=1, incoming='c0', branch='b1', connection='c1')]
        with TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root/'capture'/'raw'/'member-a'
            raw.mkdir(parents=True)
            records, offset = [], 0
            with (raw/'stream.f32').open('wb') as values:
                def record(label, positions, block, shape):
                    nonlocal offset
                    values.write(block.tobytes())
                    records.append(dict(label=label, values=block.size, shape=shape, positions=positions, token_axis=1,
                                        offset_bytes=offset, native_type='f32', operation='ADD' if label in sources else 'NONE',
                                        source_names=sources.get(label, [])))
                    offset += block.nbytes
                for positions in ([0, 2], [1]):   # one label's rows sit in more than one record of the stream
                    for label, value in rows.items():
                        record(label, positions, value[positions], [width, len(positions)])
                    record('Kcur_pos-0', positions, np.ones((2, len(positions), 5), '<f4'), [5, len(positions), 2])
            (raw/'stream.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
            source = SimpleNamespace(states_reading=app.held.states_reading, raw_forward=app.held.raw_forward)

            def read_raw(raw, nodes, count, width):
                return native_key_driver.residual_forward(source, raw, nodes, count, width)

            expected = read_raw(raw, nodes, count, width)
            self.assertEqual(expected['merged'][1].numpy().tobytes(), rows['c1'].tobytes())
            (root/'run-folder').mkdir()
            working = group_disk_state.Working(app, root/'run-folder')
            try:
                with patch.object(app.held, 'raw_forward', read_raw):
                    with working.native_traces():
                        native = app.held.raw_forward(raw, nodes, count, width)
                        for key in ('incoming', 'merged'):
                            self.assertIsInstance(native[key], group_disk_state.Stream)
                            for value, wanted in zip(native[key], expected[key], strict=True):
                                self.assertEqual((value.dtype, value.shape, value.numpy().tobytes()),
                                                 (wanted.dtype, wanted.shape, wanted.numpy().tobytes()))
                    self.assertIs(app.held.raw_forward, read_raw)
            finally:
                working.close_scratch()
                working.remainder.close()


if __name__ == '__main__':
    unittest.main()
