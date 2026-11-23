"""CPU qualification of complete, observational local loss records."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pipeline.observability.recorder import Recorder
from pipeline.weight_edit import full_loss_maps as maps
from pipeline.weight_edit.vector_chance import outside


def labels(n):
    return [dict(block=m // 3, branch=('attention', 'feed_forward', 'per_layer_input')[m % 3])
            for m in range(n)]


IDENTITY = maps.LossIdentity('fixture-explicit-radius-v1', 'a' * 64, 'b' * 64, 'fixture-supplied-radius')


def measure(original, current, radius, pairs):
    r, m = len(original[0]), len(original)
    i, j = torch.tensor(pairs).T
    mass = 1 / (r * m * torch.bincount(i, minlength=r)[i].double())
    return math.fsum(float(outside(b[j] - a[i], t[i].square() / 2, mass).residual.square().sum())
                     for a, b, t in zip(original, current, radius))


def fixture(n=3, *, pairs=None, history='history0', changed=False):
    pairs = np.array([[0, 0], [0, 1], [1, 2]] if pairs is None else pairs, dtype=np.int64)
    original = [torch.zeros((2, 1), dtype=torch.float64) for _ in range(n)]
    current = [torch.tensor([[0.5 if changed else 1.], [3.], [2.]], dtype=torch.float64) for _ in range(n)]
    radius = [torch.tensor([1., 0.], dtype=torch.float64) for _ in range(n)]
    bundle = maps.build_loss_maps(original, current, radius, pairs, expected_merges=n,
        merge_labels=labels(n), identity=IDENTITY,
        observation=dict(kind='fresh_native', model_revision=0, acquisition_id='capture0', history_id=history, complete=True),
        declared_objective=measure(original, current, radius, pairs))
    return bundle, original, current, radius, pairs


class FullLossMapsTests(unittest.TestCase):
    def test_pair_mass_distances_boundary_zero_radius_and_both_position_maps(self):
        b, *_ = fixture()
        np.testing.assert_array_equal(b.arrays['outside'], [[False] * 3, [True] * 3, [True] * 3])
        np.testing.assert_array_equal(b.arrays['distance'][:, 0], [1., 3., 2.])
        np.testing.assert_array_equal(b.arrays['excess'][:, 0], [0., 2., 2.])
        np.testing.assert_array_equal(b.arrays['radius'][:, 0], [1., 1., 0.])
        np.testing.assert_allclose(b.arrays['pair_mass'], [1 / 12, 1 / 12, 1 / 6])
        self.assertEqual(b.summary['objective'], b.summary['canonical_objective'])
        self.assertEqual(b.arrays['reference_position_block_loss'].shape, (2, 1))
        self.assertEqual(b.arrays['current_position_block_loss'].shape, (3, 1))
        for key, total in b.summary['regrouped_sums'].items():
            self.assertEqual(total - b.summary['objective'], b.summary['regrouping_roundoff'][key])
        np.testing.assert_array_equal(b.arrays['reference_multiplicity'], [2, 1])
        np.testing.assert_array_equal(b.arrays['current_multiplicity'], [1, 1, 1])

    def test_all105_merges_match_existing_native_scalar_without_grad_mutation(self):
        generator = torch.Generator().manual_seed(17)
        original = [torch.randn((3, 7), generator=generator, dtype=torch.float64).requires_grad_() for _ in range(105)]
        current = [torch.randn((4, 7), generator=generator, dtype=torch.float64).requires_grad_() for _ in range(105)]
        chance = [torch.rand(3, generator=generator, dtype=torch.float64) for _ in range(105)]
        radius = [(2 * c).sqrt() for c in chance]
        pairs = np.array([[0, 0], [0, 1], [1, 2], [2, 3]])
        i, j = torch.tensor(pairs).T
        mass = 1 / (3 * 105 * torch.bincount(i)[i].double())
        terms = [float(outside(b[j] - a[i], c[i], mass).residual.detach().square().sum())
                 for a, b, c in zip(original, current, chance)]
        b = maps.build_loss_maps(original, current, radius, pairs, expected_merges=105,
            merge_labels=labels(105), identity=IDENTITY,
            observation=dict(kind='fresh_native', model_revision=1, acquisition_id='fresh', history_id='h', complete=True),
            declared_objective=math.fsum(terms))
        self.assertEqual(len(b.per_merge), 105)
        self.assertEqual(b.summary['blocks'], 35)
        self.assertEqual([r['objective'] for r in b.per_merge], terms)
        self.assertEqual(b.summary['canonical_objective'], math.fsum(terms))
        self.assertTrue(all(x.grad is None and x.requires_grad for x in original + current))
        self.assertTrue(all(a.ndim <= 2 for a in b.arrays.values()))

    def test_invalid_or_partial_inputs_and_exact_scalar_mismatch_fail(self):
        b, a, c, radius, pairs = fixture()
        args = dict(expected_merges=3, merge_labels=labels(3), identity=IDENTITY,
                    observation=b.observation, declared_objective=b.summary['objective'])
        with self.assertRaisesRegex(ValueError, 'differs from declared'):
            maps.build_loss_maps(a, c, radius, pairs, **(args | {'declared_objective': 0.}))
        for bad in (radius[:2], [torch.tensor([-1., 0.])] * 3, [torch.tensor([float('nan'), 0.])] * 3):
            with self.assertRaises(ValueError):
                maps.build_loss_maps(a, c, bad, pairs, **args)
        with self.assertRaises(ValueError):
            maps.build_loss_maps(a, c, radius, pairs[:-1], **args)
        with self.assertRaises(ValueError):
            maps.build_loss_maps(a, c, radius, pairs.astype(float), **args)
        with self.assertRaises(ValueError):
            maps.build_loss_maps(a, c, radius, pairs, **(args | {'merge_labels': labels(3)[:2]}))

    def test_correspondence_dilution_and_inside_movement_remain_visible(self):
        old, *_ = fixture()
        new, *_ = fixture(pairs=[[0, 0], [1, 1], [1, 2]], history='changed-history', changed=True)
        summary, data = maps.correspondence_changes(old, new)
        self.assertEqual((summary['pairs_added'], summary['pairs_removed']), (1, 1))
        self.assertEqual(summary['common_pairs_changed_mass'], 2)
        self.assertFalse(summary['same_history'])
        self.assertEqual(summary['inside_both_moved'], 3)
        np.testing.assert_array_equal(data['distance_before'][1], data['distance_after'][1])
        self.assertTrue((data['weighted_loss_after'][1] < data['weighted_loss_before'][1]).all())
        with self.assertRaisesRegex(ValueError, 'different loss'):
            maps.correspondence_changes(old, replace(new, identity=replace(IDENTITY, tolerance_sha256='c' * 64)))

    def test_inventory_counts_all_eligible_not_only_received_records(self):
        eligible = [dict(name=f'w{i}', dtype='BF16' if i % 2 else 'F32', shape=[2, 2]) for i in range(538)]
        grads = {'w0': dict(status='connected', finite=True, nonzero_entries=0, norm_squared=0.),
                 'w1': dict(status='disconnected'),
                 'w2': dict(status='connected', finite=True, nonzero_entries=1, norm_squared=2.)}
        updates = {'w0': dict(complete=True, changed_entries=0, requested_nonzero_entries=0),
                   'w2': dict(complete=True, changed_entries=0, requested_nonzero_entries=1, displacement_squared=0., remainder_after_squared=2.)}
        inventory = maps.gradient_update_inventory(eligible, gradients=grads, updates=updates)
        self.assertEqual(len(inventory['rows']), 538)
        self.assertEqual(inventory['gradient_states'], dict(zero=1, disconnected=1, nonzero=1, absent=535))
        self.assertEqual(inventory['rows'][2]['update_state'], 'rounded_away')
        self.assertFalse(inventory['complete_gradient_inventory'])
        self.assertFalse(inventory['all_gradients_observed'])
        bad = maps.gradient_update_inventory(eligible, gradients={
            'w0': dict(status='connected', finite=False, nonzero_entries=1, norm_squared=None)})
        self.assertEqual(bad['rows'][0]['gradient_state'], 'nonfinite')
        with self.assertRaises(ValueError):
            maps.gradient_update_inventory(eligible, gradients={'foreign': dict(status='disconnected')})

    def test_partial_update_receipt_never_certifies_installation(self):
        eligible = [dict(name='w', dtype='F32', shape=[2, 2])]
        for complete, expected in ((False, 'incomplete'), (None, 'completion_unobserved')):
            update = dict(changed_entries=0, requested_nonzero_entries=1)
            if complete is not None:
                update['complete'] = complete
            result = maps.gradient_update_inventory(eligible, gradients={}, updates={'w': update})
            self.assertEqual(result['rows'][0]['update_state'], expected)
            self.assertTrue(result['complete_update_inventory'])
            self.assertFalse(result['all_updates_observed'])

    def test_response_types_coefficient_one_and_fitted_optimum_are_distinct(self):
        old, *_ = fixture()
        fresh, *_ = fixture(history='new-generated-history', changed=True)
        continuous = [dict(merge=m, baseline=1., loss_cross=-2., response_norm_squared=1., predicted_objective=999.) for m in range(3)]
        installed = [dict(merge=m, baseline=1., loss_cross=-0.5, response_norm_squared=0.25, predicted_objective=888.) for m in range(3)]
        result = maps.response_comparison(old, fresh, continuous=continuous, installed=installed, applied_length=0.25)
        self.assertEqual(result['per_merge'][0]['continuous_prediction'], 0.5625)
        self.assertEqual(result['per_merge'][0]['installed_displacement_surrogate'], 0.75)
        self.assertEqual(result['installed_displacement_coefficient'], 1)
        self.assertTrue(result['history_changed'])
        self.assertEqual(result['per_merge'][0]['fresh_native_loss'], fresh.per_merge[0]['objective'])
        with self.assertRaises(ValueError):
            maps.response_comparison(old, replace(fresh, observation=fresh.observation | {'kind': 'held_history'}), continuous=continuous, installed=installed, applied_length=1.)

    def test_local_artifact_bindings_recorder_compatibility_and_rendering(self):
        import matplotlib
        matplotlib.use('Agg')
        b, original, current, *_ = fixture()
        with tempfile.TemporaryDirectory(dir='/tmp') as temporary:
            path = Path(temporary)
            record = maps.write_local(b, path / 'local')
            data = path / 'local/loss-maps.npz'
            self.assertEqual(hashlib.sha256(data.read_bytes()).hexdigest(), record['arrays']['sha256'])
            with np.load(data, allow_pickle=False) as archive:
                self.assertEqual(archive['distance'].shape, (3, 3))
                self.assertFalse(any('turn' in name or 'weight_tensor' in name for name in archive.files))
            with self.assertRaises(FileExistsError):
                maps.write_local(b, path / 'local')
            recorder = Recorder(path / 'events.jsonl')
            with recorder.span('native.measure') as outcome:
                outcome.update(maps.recorder_attributes(b))
            recorder.close()
            events = [json.loads(x) for x in (path / 'events.jsonl').read_text().splitlines()]
            self.assertFalse(any(e.get('name') == 'native.objective' for e in events))
            end = next(e for e in events if e['event'] == 'span_end')
            self.assertEqual(end['attributes']['output.tolerance_sha256'], IDENTITY.tolerance_sha256)
            maps.render_loss_views(b, path / 'maps.png')
            maps.render_cell_overlay(b, original, current, pair_index=1, merge=0, path=path / 'cell.png')
            self.assertGreater((path / 'maps.png').stat().st_size, 0)
            self.assertGreater((path / 'cell.png').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
