"""Exercise the documented learner on tiny native-format CPU weights only."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import torch

from pipeline.weight_edit import final_experiment as baseline
from tests.test_full_graph import made, tf


def signature(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def sample(app, fixture, state):
    # Reopen the actual native bytes for each decoding pass. Neither the token
    # scores nor token identities enter the loss or determine correspondence.
    model = tf.Model(fixture.file, fixture.tensors, fixture.a, device='cpu',
                     leaves=lambda _: False, cache='float16')
    prompt = (3, 7)
    with torch.no_grad():
        prefill = model.forward(prompt, again=False)
        scores = model.scores(prefill['result_norm'][-1])
        history = (*prompt, int(scores.argmax()))
        rows = {name: state.required_rows(name, sorted(set(history))) for name in app.full_parameters.TABLES}
        parameters = app.full_parameters.ParameterModel(model, history, row_ids=rows)
        forward = parameters.forward(history, again=False)
    return history, scores, parameters, forward


class InstalledLearnerTests(unittest.TestCase):
    def test_two_native_edits_across_fresh_decode_and_checkpoint_restore(self):
        torch.set_num_threads(1)
        app = baseline.controller(baseline.RECOVERY)
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            folder = Path(temp)
            fixture = made(folder)
            fixture.a['final_logit_softcapping'] = 30.0
            seed_file = Path(fixture.file)
            seed = seed_file.read_bytes()
            base = {'fixture_sha256': hashlib.sha256(seed).hexdigest()}
            _, specs = app.state_specs(fixture.tensors, fixture.a)
            chunk = max(s.size for s in specs if not s.sparse_rows)
            with os.fdopen(app.factory.memory_model(seed_file, base['fixture_sha256']), 'r+b') as native_file:
                fd = native_file.fileno()
                model_file = Path(f'/proc/self/fd/{fd}')
                fixture.file = model_file
                state = app.full_state.FullState(fd, specs, model_identity={'revision': 0},
                                                 base_identity=base, chunk_elements=chunk)
                _, _, _, first = sample(app, fixture, state)
                turns = app.turn_pair.turns(first, expected_merges=105)
                target = app.vector_chance.Reference([t * .9 for t in turns], [
                    torch.full((len(t),), 1e-10, dtype=torch.float64) for t in turns], expected_merges=105)
                fixed = signature([*target._reference, *target._chance])
                recorder = app.Recorder(folder / 'events.jsonl')
                try:
                    measurements = []
                    for revision in (1, 2):
                        history, scores, parameters, forward = sample(app, fixture, state)
                        pairs = app.held.turn_correction.correspondence(target, forward)
                        before = app.held.turn_correction.measurement(target, forward, pairs)
                        if measurements:
                            self.assertEqual(before, measurements[-1])
                        graph = app.full_graph.Graph(parameters, history, forward)
                        proposal, direction = app.propose(graph, history, target, pairs, before, state, .2, recorder)
                        self.assertEqual(proposal['status'], 'ready')
                        self.assertEqual(set(direction), set(state.specs))
                        self.assertEqual(len(proposal['continuous_moments']), 105)
                        self.assertGreater(proposal['changed_values'], 0)
                        previous_bytes = model_file.read_bytes()
                        receipt = app.install(state, parameters, graph, direction, proposal['applied_length'],
                                              proposal['candidate_chunk_hashes'], {'revision': revision})
                        self.assertEqual(receipt['model_identity'], {'revision': revision})
                        self.assertNotEqual(model_file.read_bytes(), previous_bytes)
                        _, fresh_scores, _, fresh = sample(app, fixture, state)
                        self.assertFalse(torch.equal(scores, fresh_scores))
                        fresh_pairs = app.held.turn_correction.correspondence(target, fresh)
                        fresh_loss = app.held.turn_correction.measurement(target, fresh, fresh_pairs)
                        self.assertNotEqual(before['objective'], fresh_loss['objective'])
                        measurements.append(fresh_loss)
                        self.assertEqual(signature([*target._reference, *target._chance]), fixed)
                        if revision == 1:
                            checkpoint = folder / 'checkpoint'
                            state.checkpoint(checkpoint)
                            installed = model_file.read_bytes()
                            os.pwrite(fd, seed, 0)
                            manifest = baseline.read(checkpoint / 'state.json')
                            remainder = checkpoint / manifest['remainder']['file']
                            kept = remainder.read_bytes()
                            self.assertTrue(any(kept))
                            remainder.write_bytes(bytes([kept[0] ^ 1]) + kept[1:])
                            restore = dict(specs=specs, expected_base_identity=base, chunk_elements=chunk,
                                           expected_model_identity={'revision': 1}, sync=lambda *unused: None)
                            with self.assertRaisesRegex(ValueError, 'payload hash/size'):
                                app.full_state.FullState.restore(fd, checkpoint, **restore)
                            self.assertEqual(model_file.read_bytes(), seed)
                            remainder.write_bytes(kept)
                            state = app.full_state.FullState.restore(fd, checkpoint, **restore)
                            self.assertEqual(model_file.read_bytes(), installed)
                            # Restore includes every sub-native FP32 remainder.
                            restored = folder / 'restored'
                            state.checkpoint(restored)
                            self.assertEqual((restored / manifest['remainder']['file']).read_bytes(), kept)
                finally:
                    recorder.close()
            self.assertEqual(seed_file.read_bytes(), seed)
