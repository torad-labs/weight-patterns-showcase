"""Exercise prefix residuals through the documented native-weight learner."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipeline.weight_edit import common_prefix_driver as driver, common_prefix_objective as prefix
from pipeline.weight_edit import final_experiment as baseline, validation_stopping
from tests.test_final_experiment import sample
from tests.test_full_graph import made


class CommonPrefixTests(unittest.TestCase):
    def test_satisfied_or_inside_cells_have_zero_direct_pullback(self):
        before = torch.tensor([[1., 0.], [1., 0.], [1., 0.]], dtype=torch.float64)
        amplitudes = torch.tensor([0., 1., 1.], dtype=torch.float64, requires_grad=True)
        after = torch.stack((torch.ones_like(amplitudes), amplitudes), dim=1)
        targets = torch.tensor([.1, 1 / 2**.5, .1], dtype=torch.float64)
        rows = prefix.Cells(torch.arange(3), torch.zeros_like(before),
                           torch.tensor([[0., 1.]]).expand(3, -1).double(), targets, torch.ones(3))
        residual = rows.residual({'incoming': [before], 'merged': [after]}, 0)
        self.assertTrue(torch.allclose(residual, torch.tensor([.1, 0., 0.], dtype=torch.float64)))
        self.assertEqual(torch.autograd.grad(residual.square().sum(), amplitudes)[0].tolist(), [-.2, 0., 0.])

    def test_evaluation_boundary_waits_for_exact_measured_archive_and_verifies_exit(self):
        with TemporaryDirectory(dir='/tmp') as temp, patch.object(baseline, 'ROOT', Path(temp)):
            run = Path(temp)
            app = baseline.controller(baseline.MEASURED)
            app.ROOT = run
            loss = {'loss_version': 'fixture'}
            history = []
            for i, objective in enumerate([3., *([4.]*10), 2.]):
                record = run/f'measurement-{i}.json'
                app.save(record, dict(step=i, objective=objective, complete=True, ended=True, loss_identity=loss))
                history.append(baseline.binding(record))
            calibration = run/'calibration.json'
            app.save(calibration, dict(fit_history=history))
            actual_source = Path(validation_stopping.__file__)
            source = run/'validation_stopping.py'
            source.write_bytes(actual_source.read_bytes())
            with patch.object(validation_stopping, '__file__', str(source)):
                config = driver.evaluation_boundary(baseline.binding(calibration), loss)
            self.assertEqual(config['completed_updates'], 11)
            config['continuation'] = dict(completed_updates_before=7, remaining_updates=4, historical_endpoint=401)
            training = driver.Training(app, run, {}, None)
            events = []

            def archive(run, previous_step, current_step, declaration, *, stop_requested):
                self.assertFalse(stop_requested())
                events.append(('archive', current_step))
                return {'complete': True}

            def execute(run, declaration, recorder, capture, completion):
                for i in range(1, 5):
                    training.generation = i
                    training.current = {}, None, history[7+i], True
                    app.archive_source.archive_previous(run, i-1, i, declaration,
                                                       stop_requested=lambda: ('stop', 4) in events)
                self.assertEqual(events, [('archive', 1), ('archive', 2), ('archive', 3), ('archive', 4), ('stop', 4)])
                app.save(run/'final-state.json', dict(status='complete', iteration=4, last_measurement=history[11]))
                app.save(run/'launcher-exit.json', dict(exit_code=0, last_measurement=history[11]))

            app.archive_source = SimpleNamespace(archive_previous=archive)
            with patch.object(app, 'execute', execute), training.evaluation(config), \
                 patch.object(driver.signal, 'raise_signal', side_effect=lambda sig: events.append(('stop', 4))):
                app.execute(run, {}, None, None, {})
            receipt = baseline.read(run/'evaluation-boundary.json')
            self.assertEqual(receipt['completed_updates'], 11)
            self.assertEqual(receipt['local_completed_updates'], 4)
            self.assertEqual(receipt['historical_step'], 401)
            self.assertEqual(receipt['measurement'], history[11])
            self.assertEqual(receipt['final_state'], baseline.binding(run/'final-state.json'))

    def test_continued_boundary_charges_previously_measured_updates_to_original_interval(self):
        with TemporaryDirectory(dir='/tmp') as temp, patch.object(baseline, 'ROOT', Path(temp)):
            prior = Path(temp)/'prior'
            (prior/'loop/7').mkdir(parents=True)

            def save(path, value):
                path.write_text(json.dumps(value))
                return baseline.binding(path)

            origin_boundary = dict(completed_updates=11, calibration={'bound': 'original'})
            policy = dict(policy={'bound': 'common-prefix-axis'})
            reason = 'RuntimeError: archive readback failed; no local stream released: ["test"]'
            save(prior/'declaration.json', dict(prefix_evaluation_boundary=origin_boundary, prefix_objective=policy,
                 loss_identity='reference-and-tolerance', counter_semantics={'historical_predecessor': 390}))
            identity = {'run': 'prior', 'generation': 7}
            observation = dict(step=7, model_identity=identity, complete=True, ended=True, prefix_objective={})
            measurement = save(prior/'loop/7/measurement.json', observation)
            state = dict(status='complete', checkpoint={'native-and-remainder': 'bound'}, reason=reason,
                         iteration=7, model_identity=identity, last_measurement=measurement)
            checkpoint = save(prior/'final-state.json', state)
            terminal = dict(exit_code=1, state_status='complete', installed_updates=7, reason=reason,
                            last_measurement=measurement)
            end = save(prior/'launcher-exit.json', terminal)
            kind = driver.archived_exit_kind(terminal, state, observation)
            declaration = dict(resume_checkpoint=checkpoint, start_fresh=measurement, prefix_objective=policy,
                loss_identity='reference-and-tolerance', counter_semantics={'historical_predecessor': 397},
                predecessor_exit=dict(record=end, kind=kind))
            configured = driver.continued_boundary(declaration, origin_boundary)
            continued = configured['continuation']
            self.assertEqual(configured['completed_updates'], 11)
            self.assertEqual(continued['completed_updates_before'], 7)
            self.assertEqual(continued['remaining_updates'], 4)
            self.assertEqual(continued['historical_endpoint'], 401)
            self.assertEqual(continued['source_measurement'], measurement)
            self.assertEqual(continued['source_checkpoint'], checkpoint)
            with self.assertRaisesRegex(ValueError, 'changed the frozen evaluation interval'):
                driver.continued_boundary(dict(declaration, prefix_objective=dict(policy={'bound': 'different'})),
                                          origin_boundary)
            with self.assertRaisesRegex(ValueError, 'must account for completed updates'):
                driver.continued_boundary(dict(declaration, counter_semantics={'historical_predecessor': 390}),
                                          origin_boundary)
            with self.assertRaisesRegex(ValueError, 'complete measured prefix checkpoint'):
                driver.archived_exit_kind(dict(terminal, reason='unrelated failure'), state, observation)

    def test_public_mode_installs_prefix_gradient_even_when_ordinary_loss_starts_at_zero(self):
        torch.set_num_threads(1)
        app = baseline.controller(baseline.MEASURED)
        source = baseline.ROOT / 'studies/04-round-2-overlap/runs/whole-prefix-frozen-stages/register.py'
        with TemporaryDirectory(dir='/tmp') as temp:
            root = Path(temp)
            run = root/'studies/round/runs/run'
            run.mkdir(parents=True)
            registrar = run/'register.py'
            registrar.write_bytes(source.read_bytes())
            fixture = made(root)
            fixture.a['final_logit_softcapping'] = 30.
            seed = Path(fixture.file)
            identity = {'fixture_sha256': sha256(seed.read_bytes()).hexdigest()}
            _, specs = app.state_specs(fixture.tensors, fixture.a)
            chunk = max(s.size for s in specs if not s.sparse_rows)
            with os.fdopen(app.factory.memory_model(seed, identity['fixture_sha256']), 'r+b') as file, \
                 patch.object(app, 'ROOT', root), patch.object(baseline, 'ROOT', root):
                fixture.file = Path(f'/proc/self/fd/{file.fileno()}')
                state = app.full_state.FullState(file.fileno(), specs, model_identity={'revision': 0},
                                                base_identity=identity, chunk_elements=chunk)
                _, _, _, initial = sample(app, fixture, state)
                turns = app.turn_pair.turns(initial, expected_merges=105)
                reference = app.vector_chance.Reference(turns, [torch.zeros(len(t)) for t in turns],
                                                        expected_merges=105)
                reference_turns = np.stack([t[1:].numpy() for t in turns], axis=1)
                early = reference_turns[:1]
                norms = np.linalg.norm(early, axis=2, keepdims=True)
                axes = np.divide(early, norms, out=np.zeros_like(early), where=norms > 0)
                axes[..., 0] += (norms == 0).squeeze(-1)
                targets = np.full(early.shape[:2], .01)
                # A later query may pass the supported interval without being
                # clamped to its last positive source event.
                arrays = run/'frozen.npz'
                np.savez(arrays, reference_turns=reference_turns, early_turns=early, axes=axes, targets=targets)
                policy = run/'objective.json'
                policy.write_text(json.dumps(dict(format=prefix.FORMAT, arrays=baseline.binding(arrays),
                    registrar=baseline.binding(registrar), first_source_event=1)))
                prior = {'record': 'prior/loop/7/measurement.json', 'sha256': 'f'*64}
                continuation = dict(source_measurement=prior, completed_updates_before=7, remaining_updates=4,
                                    historical_start=390, historical_endpoint=401)
                declaration = dict(prefix_objective=dict(policy=baseline.binding(policy), driver={'record': 'prefix-driver'}),
                    prefix_evaluation_boundary=dict(completed_updates=11, continuation=continuation),
                    program={'record': 'historical-controller'}, operation='fixture', original_model={'sha256': 'a'*64},
                    reference={'sha256': 'b'*64}, tolerance={'sha256': 'c'*64}, resume_checkpoint={'sha256': 'd'*64})
                app.save(run/'declaration.json', declaration)

                def acquire(run, number, case, model_identity, recorder, previous=None):
                    history, scores, model, native = sample(app, fixture, state)
                    pairs = app.held.turn_correction.correspondence(reference, native)
                    measured = app.held.turn_correction.measurement(reference, native, pairs)
                    path = run/'loop'/str(number)/'measurement.json'
                    path.parent.mkdir(parents=True)
                    app.save(path, dict(measured, previous_measurement=previous))
                    return native, pairs, history, measured, model, app.binding(path), True, scores

                observed = {}

                def execute(run, dec, recorder, capture, completion):
                    case = dict(tokens=[3, 7])
                    source = app.acquire(run, 0, case, {'revision': 0}, recorder)
                    panel = app.acquire(run/'validation', 0, case, {'revision': 0}, recorder)
                    self.assertNotIn('prefix_objective', panel[3])
                    graph = app.full_graph.Graph(source[4], source[2], source[0])
                    proposal, direction = app.propose(graph, source[2], reference, source[1], source[3], state, .2, recorder)
                    self.assertEqual(proposal['status'], 'ready')
                    self.assertGreater(proposal['changed_values'], 0)
                    self.assertEqual(set(direction), set(state.specs))
                    self.assertEqual(len(proposal['continuous_moments']), 105)
                    self.assertEqual(len(proposal['installed_moments']), 105)
                    app.install(state, source[4], graph, direction, proposal['applied_length'],
                                proposal['candidate_chunk_hashes'], {'revision': 1})
                    fresh = app.acquire(run, 1, case, {'revision': 1}, recorder, previous=source[5])
                    observed.update(source=source, fresh=fresh, proposal=proposal)
                    completion.update(status='FINISHED', reason='bounded_cpu_fixture')

                def dispatch(python, argv, env):
                    self.assertEqual(Path(argv[1]).name, 'prefix-driver')
                    driver.execute(run)

                with patch.object(baseline, 'run_path', return_value=run), \
                     patch.object(baseline, 'inspect_run', return_value={'storage_ready': True}), \
                     patch.object(baseline, 'controller', return_value=app), \
                     patch.object(app.held, 'check_native_run', return_value=declaration), \
                     patch.object(app, 'acquire', acquire), patch.object(app, 'execute', execute), \
                     patch.object(sys, 'argv', ['final_experiment', 'run', str(run)]), \
                     patch.object(baseline.os, 'chdir'), patch.object(baseline.os, 'execve', side_effect=dispatch):
                    baseline.main()
                source, fresh, proposal = (observed[k] for k in ('source', 'fresh', 'proposal'))
                self.assertEqual(source[3]['objective'], 0.)
                self.assertAlmostEqual(source[3]['combined_objective'], .01**2)
                self.assertEqual(source[3]['evaluation_position'], dict(local_generation=0, completed_updates=7, historical_step=397))
                self.assertEqual(fresh[3]['evaluation_position'], dict(local_generation=1, completed_updates=8, historical_step=398))
                self.assertEqual(baseline.read(run/'loop/0/measurement.json')['previous_measurement'], prior)
                self.assertEqual(baseline.read(run/'loop/1/measurement.json')['previous_measurement'], source[5])
                self.assertEqual(source[3]['prefix_objective']['supported_cells'], 105)
                self.assertEqual(source[3]['prefix_objective']['matched_cells'], 105)
                self.assertEqual(proposal['before']['objective'], source[3]['combined_objective'])
                self.assertEqual(proposal['before']['ordinary_objective'], 0.)
                groups = proposal['prefix_objective']['response_groups']
                self.assertEqual(set(groups), {'continuous_direction', 'actual_native_displacement'})
                cont = groups['continuous_direction']
                self.assertAlmostEqual(proposal['continuous_prediction'], cont['ordinary']['prediction'] + cont['prefix']['prediction'])
                self.assertLess(cont['prefix']['prediction'], source[3]['prefix_objective']['objective'])
                self.assertNotEqual(fresh[3]['prefix_objective']['objective'], source[3]['prefix_objective']['objective'])
                self.assertEqual(fresh[3]['combined_objective'],
                                 fresh[3]['objective'] + fresh[3]['prefix_objective']['objective'])
                self.assertFalse(torch.equal(fresh[-1], source[-1]))
                receipt = baseline.read(run/'loop/0/prefix-correspondence.json')
                self.assertEqual(receipt['pairs'], [dict(source_event=1, query_event=1)])
