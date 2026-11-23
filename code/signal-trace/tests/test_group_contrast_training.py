"""CPU qualification for signed cohort requests, coverage and fresh feedback."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipeline.observability.recorder import Recorder
from pipeline.weight_edit import final_experiment as baseline, group_contrast_driver as driver
from pipeline.weight_edit import group_contrast_objective as objective
from pipeline.weight_edit import host_pages


def group(centre=None, variation=None):
    return objective.Pieces(0, torch.tensor([0]), torch.tensor([0]),
        torch.tensor([[0., 1., 0.]], dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
        torch.zeros((1,3), dtype=torch.float64), torch.ones(1, dtype=torch.float64), centre, variation)


def forward(y, z):
    one = z*0.+1.
    before = torch.stack((one, one*0., one*0.)).reshape(1,3)
    after = torch.stack((one, y, z)).reshape(1,3)
    return dict(incoming=[before], merged=[after])


class GroupContrastTests(unittest.TestCase):
    def test_calibrated_projection_follows_new_native_mate_into_shared_gradient(self):
        with TemporaryDirectory(dir='/tmp') as directory, patch.object(baseline, 'ROOT', Path(directory)):
            root = Path(directory)
            registrar = root/'registrar.py'
            registrar.write_text("def endpoints(scores):\n    return [[] for _ in scores.T]\n")
            matcher = root/'matcher.py'
            matcher.write_text(
                "class Matcher:\n"
                "    @classmethod\n"
                "    def from_spec(cls, spec, root): return cls()\n"
                "    def register(self, q, spec):\n"
                "        return {'accepted':True,'pairs':[[int(q[:,0,1].argmax()), len(q)-1]]}\n")
            policy = root/'matcher.json'
            policy.write_text('{"stage_order":[{"phase":"symbolic","offset":0}]}')
            base, extra = root/'base.npz', root/'signal.npz'
            np.savez(base, reference_turns=np.array([[[0.,1.,0.]]]), stable_cells=np.array([[0,0]]),
                     stable_axes=np.array([[0.,0.,1.]]), stable_offsets=np.zeros(1))
            np.savez(extra, signal_cells=np.array([[0,0]]), signal_axes=np.array([[0.,1.,0.]]),
                     signal_offsets=np.array([.1]))
            member = dict(id='a', arrays=baseline.binding(base), first_source_event=9,
                          operations=[dict(name='early', field='signal', arrays=baseline.binding(extra), role='failing',
                                           matcher='phase', confirm_full_trace=True),
                                      dict(name='stable', field='stable', role='preservation')])
            rule = dict(program=baseline.binding(matcher), policy=baseline.binding(policy),
                        reference_rows=[0])
            field = objective.Objective(dict(registrar=baseline.binding(registrar), members=[member],
                                             operation_matchers={'phase':rule}, support_policy=objective.LIVE_SUPPORT))
            w = torch.tensor([.01, .2, .4], dtype=torch.float64, requires_grad=True)
            x = torch.zeros((3,3), dtype=w.dtype)
            x[:,0] = 1.
            def registered(values, mate, baseline_support=None):
                f = dict(incoming=[x], merged=[torch.stack((x[:,0], values, x[:,2]), dim=1)])
                ops = field.register('a', f, prompt_events=2, baseline_support=baseline_support,
                                     full_trace_pairs=np.array([[9, mate]]))
                return f, ops
            f, ops = registered(w, 2)
            initial = {op.name: op.required for op in ops}
            local = next(ops[0].residuals(f))
            grad, = torch.autograd.grad(local.square().sum(), w)
            self.assertEqual(torch.nonzero(grad).flatten().tolist(), [2])
            with torch.no_grad():
                self.assertEqual(ops[0].measure(f, 9)['cells'][0]['event'], 2)
            fresh, ops = registered(w.flip(0), 1, initial)
            with torch.no_grad():
                self.assertEqual(ops[0].measure(fresh, 9)['cells'][0]['event'], 1)
            _, (missing, _) = registered(w.flip(0), 2, initial)
            self.assertEqual(missing.measure(fresh, 9)['status'], 'unmeasured')
            self.assertEqual(list(missing.residuals(fresh)), [])
            self.assertEqual(missing.registration['target_projection'], .1)

    def test_signed_projection_near_zero_and_off_axis_preservation_have_pullback(self):
        y = torch.tensor(-1.e-7, dtype=torch.float64, requires_grad=True)
        zero = torch.zeros((), dtype=torch.float64)
        _, signed, residual = group().values(forward(y, zero))
        self.assertLess(float(signed.detach()), 0.)
        self.assertLess(float(torch.autograd.grad(residual.square().sum(), y)[0]), 0.)
        z = torch.tensor(.02, dtype=torch.float64, requires_grad=True)
        centres = torch.zeros((1,3), dtype=torch.float64)
        variation = torch.tensor([[[1.,0.,0.],[0.,0.,0.]]], dtype=torch.float64)
        _, signed, vector = group(centres, variation).values(forward(zero, z))
        self.assertEqual(float(signed.detach()), 0.)
        self.assertGreater(float(vector.square().sum().detach()), 0.)
        self.assertGreater(float(torch.autograd.grad(vector.square().sum(), z)[0]), 0.)

    def test_shared_registration_freezes_initial_support_and_detects_lost_cells(self):
        existing = baseline.ROOT/'studies/04-round-2-overlap/runs/whole-prefix-frozen-stages/register.py'
        with TemporaryDirectory(dir='/tmp') as temp:
            root = Path(temp)
            registrar = root/'studies/r/runs/register/endpoints.py'
            registrar.parent.mkdir(parents=True)
            registrar.write_bytes(existing.read_bytes())
            file = root/'field.npz'
            reference = np.array([[[.1,.2,.3]], [[.2,.3,.1]], [[.3,.1,.2]]])
            geometry = dict(reference_turns=reference, down_cells=np.array([[0,0],[1,0],[2,0]]),
                            down_axes=np.array([[0.,1.,0.]]*3),down_offsets=np.array([0.,-.2,0.]),
                            friction_cells=np.array([[0,0]]), friction_centres=np.zeros((1,3)),
                            friction_variation=np.array([[[1.,0.,0.]]]))
            np.savez(file, **geometry)
            with patch.object(baseline, 'ROOT', root):
                spec = dict(registrar=baseline.binding(registrar), members=[dict(id='a', arrays=baseline.binding(file),
                    operations=[dict(name='descent',role='failing',field='down'),
                                dict(name='friction',role='preservation',field='friction')])])
                field = objective.Objective(spec)
            a = torch.tensor([[1.,0.,0.],[1.,0.,0.]], dtype=torch.float64)
            before = dict(incoming=[a], merged=[torch.tensor([[1.,.1,.2], [1.,.5,0.]], dtype=torch.float64)])
            field.endpoints = lambda _: [[0],[1]]
            old = field.register('a', before, prompt_events=1)
            required = {op.name:op.required for op in old}
            field.endpoints = lambda _: [[0],[2]]
            fresh = field.register('a', before, prompt_events=1, baseline_support=required)
            saved, new = (driver.aggregate([dict(id='a', complete=True, operations=[op.measure(before, 50) for op in ops])])
                          for ops in (old, fresh))
            self.assertTrue(saved['complete'])
            self.assertIsNone(saved['members'][0]['operations'][1]['cells'][0]['signed_error'])
            self.assertFalse(new['complete'])
            self.assertIsNone(new['objective'])
            self.assertLess(new['measured_contribution'], saved['objective'])
            self.assertEqual(new['members'][0]['operations'][0]['missing_baseline_support'], [[51,0]])
            self.assertEqual(new['members'][0]['operations'][0]['new_source_support'], [[52,0]])
            comparison = driver.paired_comparison(saved, new)
            self.assertFalse(comparison['full_comparison_valid'])
            self.assertEqual(comparison['matched']['failing']['change'], 0.)

    def test_fresh_preservation_error_changes_the_next_full_weight_derivative(self):
        with TemporaryDirectory(dir='/tmp') as directory:
            folder = Path(directory)
            recorder = Recorder(folder/'events.jsonl')
            zero = torch.zeros((), dtype=torch.float64)
            w = torch.tensor(0., dtype=torch.float64)
            rows = {'embedding': (1, 2, 3)}
            fixture = SimpleNamespace(coeff=1.)

            class Graph:
                leaves = ('w',)
                parameter_rows = rows
                def __init__(self, member, weight):
                    self.member = member
                    self.w = {'w': weight}
                    captured = self.forward(())
                    self.native_in, self.native_out = captured['incoming'], captured['merged']
                def forward(self, history):
                    value = self.w['w']
                    return forward(value, zero) if self.member == 'a' else forward(zero, fixture.coeff*value)

            centres = torch.zeros((1,3), dtype=torch.float64)
            variation = torch.tensor([[[1.,0.,0.]]], dtype=torch.float64)
            failing = group()
            failing.offsets[0] = .5
            climb = group()
            climb.offsets[0] = .25
            protected = group(centres, variation)
            def operation(name, role, piece):
                support = frozenset({(0,0)})
                return objective.Operation(name, role, 1, 1, (piece,), support, support, role=='preservation')

            app = SimpleNamespace(graph_for=lambda model,state,history,native,raw,dec: Graph('b', fixture.weight),
                                  residuals=lambda reference,pairs,rows: lambda forward: iter(()))
            training = driver.Training(app, folder, {}, {'primary':'a'}, None)
            training.declaration = {}
            training.current = dict(a={'history':(1,2),'registrations':[operation('descent','failing',failing),
                                                                      operation('climb','failing',climb)]},
                                    b={'history':(2,3),'native':None,'raw':None,
                                       'registrations':[operation('friction','preservation',protected)]})
            calls = []
            state = SimpleNamespace(required_rows=lambda name, ids: calls.append(ids) or tuple(ids))
            with training.graph_rows(state):
                self.assertEqual(state.required_rows('embedding',[1,2]), (1,2,3))
            self.assertEqual(calls, [[1,2,3]])
            try:
                fixture.weight = w
                initial, _, squared_norm = training.gradients(Graph('a', w), state, recorder)
                # At w=0 both signed projections have derivative 1 with respect to the same physical weight.
                self.assertAlmostEqual(float(initial['w']), 2*(.5+.25))
                continuous, _, parts = training.responses(Graph('a', w), state, initial, None, None,
                                                            recorder, 'continuous_direction')
                self.assertAlmostEqual(continuous.slope, -squared_norm)
                self.assertEqual({part['operation'] for part in parts}, {'descent','climb','friction'})
                installed = (w + .1*continuous.length*initial['w']).bfloat16().double()
                rounded, _, _ = training.responses(Graph('a', w), state, {'w': installed-w}, None, None,
                                                    recorder, 'actual_native_displacement')
                self.assertLess(rounded.slope, 0.)
                fixture.weight = installed
                with_bleed, _, _ = training.gradients(Graph('a', installed), state, recorder)
                fixture.coeff = 0.
                without_bleed, _, _ = training.gradients(Graph('a', installed), state, recorder)
                self.assertLess(float(with_bleed['w']), float(without_bleed['w']))
            finally:
                recorder.close()

    def test_a_disk_backed_run_keeps_its_direction_in_memory(self):
        with TemporaryDirectory(dir='/tmp') as directory:
            folder = Path(directory)
            recorder = Recorder(folder/'events.jsonl')
            zero = torch.zeros((), dtype=torch.float64)

            class Graph:
                leaves = ('w', 'idle')
                parameter_rows = {}
                def __init__(self, member):
                    self.member = member
                    self.w = {'w': torch.tensor([.125, -.5], dtype=torch.float32), 'idle': torch.ones((2, 3), dtype=torch.float32)}
                    captured = self.forward(())
                    self.native_in, self.native_out = captured['incoming'], captured['merged']
                def forward(self, history):
                    first, second = self.w['w'].double()
                    return forward(first, second*second) if self.member == 'a' else forward(second, 3.*first)

            def operation(name, offset):
                piece = group()
                piece.offsets[0] = offset
                support = frozenset({(0,0)})
                return objective.Operation(name, 'failing', 1, 1, (piece,), support, support, False)

            app = SimpleNamespace(graph_for=lambda model,state,history,native,raw,dec: Graph('b'),
                                  residuals=lambda reference,pairs,rows: lambda forward: iter(()))
            current = dict(a={'history':(1,2),'registrations':[operation('descent', .5)]},
                           b={'history':(2,3),'native':None,'raw':None,'registrations':[operation('climb', .25)]})
            settled = []
            working = SimpleNamespace(settle=lambda: settled.append(True))
            try:
                results = []
                for backing in (None, working):
                    training = driver.Training(app, folder, {}, {'primary':'a'}, None, backing)
                    training.declaration, training.current = {}, current
                    results.append(training.gradients(Graph('a'), None, recorder))
                memory, backed = results
                self.assertTrue(bool(memory[0]['w'].abs().sum()))
                self.assertEqual(settled, [True])   # the derivative's end still lets the working files' pages go
                self.assertEqual(tuple(backed[0]), tuple(memory[0]))
                for name, value in memory[0].items():
                    self.assertIsNone(host_pages.location(backed[0][name]))   # no file holds the direction
                    self.assertEqual(backed[0][name].dtype, torch.float32)
                    self.assertEqual(backed[0][name].numpy().tobytes(), value.numpy().tobytes())
                self.assertEqual(backed[1:], memory[1:])
                self.assertTrue(bool(torch.signbit(backed[0]['idle']).all()))   # a never-reached leaf is negative zero
            finally:
                recorder.close()

    def test_each_step_reserves_the_whole_cohort_and_the_working_files_growth(self):
        def budget(specs, state, context, inherited):
            return dict(free_required_bytes=100, context=context)

        step = lambda *args, **kwargs: None
        app = SimpleNamespace(acquire=step, graph_for=step, execute=step, propose=step, storage_budget=budget)
        config = {'primary': 'a', 'maximum_events_per_member': {'a': 3, 'b': 4}}
        for working, expected in ((None, dict(free_required_bytes=100, context=7)),
                                  (SimpleNamespace(limits=None), dict(free_required_bytes=100, context=7)),
                                  (SimpleNamespace(limits={}, pending=lambda: 7),
                                   dict(free_required_bytes=107, context=7, working_files_bytes=7))):
            training = driver.Training(app, None, {}, config, None, working)
            with training.connected({}):
                self.assertEqual(app.storage_budget('specs', 'state', 1, 'inherited'), expected)
            self.assertIs(app.storage_budget, budget)

    def test_live_support_weights_new_matches_and_leaves_missing_donor_cells_unmeasured(self):
        def trace(angles, weight):
            angles = torch.tensor(angles, dtype=torch.float64)
            incoming = torch.zeros((len(angles), 3), dtype=torch.float64)
            incoming[:,0] = 1.
            merged = torch.stack((incoming[:,0], .15*angles.cos()+weight, .15*angles.sin()), dim=1)
            return dict(incoming=[incoming], merged=[merged])

        registrar = baseline.ROOT/'studies/04-round-2-overlap/runs/whole-prefix-frozen-stages/register.py'
        origin = torch.tensor(0., dtype=torch.float64)
        complete = trace([0., .17, 1.5], origin)
        [turn] = objective.turns(complete, expected_merges=1)
        with TemporaryDirectory(dir='/tmp') as directory:
            root = Path(directory)
            # The historical registrar resolves its repository root from its file depth.
            local_registrar = root/'studies/r/runs/register/endpoints.py'
            local_registrar.parent.mkdir(parents=True)
            local_registrar.write_bytes(registrar.read_bytes())
            path = root/'field.npz'
            np.savez(path, reference_turns=turn.numpy()[:,None,:], down_cells=np.array([[0,0],[1,0],[2,0]]),
                     down_axes=np.array([[0.,1.,0.]]*3), down_offsets=np.array([0.,0.,.5]),
                     stable_cells=np.array([[0,0]]), stable_axes=np.array([[0.,0.,1.]]), stable_offsets=np.zeros(1))
            with patch.object(baseline, 'ROOT', root):
                member=dict(id='a', arrays=baseline.binding(path), operations=[dict(name='descent',role='failing',field='down'),
                    dict(name='friction',role='preservation',field='stable')])
                spec=dict(registrar=baseline.binding(local_registrar), support_policy=objective.LIVE_SUPPORT,
                          members=[member])
                field=objective.Objective(spec)
        initial = trace([0., .17], origin)
        old, old_stable = field.register('a', initial, prompt_events=1)
        initial_support = {op.name:op.required for op in (old, old_stable)}
        weight = origin.detach().requires_grad_()
        current = trace([0., 1.5, 1.5], weight)
        new, new_stable = field.register('a', current, prompt_events=1, baseline_support=initial_support)
        with torch.no_grad():
            baseline_read, current_read = old.measure(initial, 50), new.measure(current, 50)
            old_summary, new_summary = (driver.aggregate([dict(id='a', complete=True, operations=[read, stable.measure(fwd,50)])])
                for read,stable,fwd in [(baseline_read,old_stable,initial),(current_read,new_stable,current)])
        self.assertEqual(baseline_read['missing_donor_support'], [[52,0]])
        self.assertEqual(current_read['missing_donor_support'], [[51,0]])
        self.assertEqual(current_read['missing_baseline_support'], [[51,0]])
        self.assertEqual(current_read['new_source_support'], [[52,0]])
        self.assertIsNone(current_read['objective'])
        self.assertEqual([c['source_event'] for c in current_read['cells']], [50,52,52])
        change = driver.paired_comparison(old_summary, new_summary)
        self.assertEqual(change['matched']['failing'], dict(common_cells=1,source=baseline_read['cells'][0]['weighted_loss'],
            successor=current_read['cells'][0]['weighted_loss'],change=0.))
        self.assertFalse(change['full_comparison_valid'])
        self.assertIsNone(new_summary['objective'])
        # Half the total mass belongs to the failing role, split across the three frozen source cells.
        # Two visits of cell 52 share its mass and its gradient.
        measured = sum(r.square().sum() for r in new.residuals(current))
        projection = torch.nn.functional.normalize(current['merged'][0],dim=1)[:,1]
        expected = (projection[0].square() + (projection[1:]-.5).square().mean()) / 6
        self.assertAlmostEqual(new_summary['measured_contribution'], expected.item())
        actual_gradient = torch.autograd.grad(measured, weight, retain_graph=True)[0]
        self.assertAlmostEqual(actual_gradient.item(), torch.autograd.grad(expected,weight)[0].item())
        whole, _ = field.register('a', complete, prompt_events=1, baseline_support=initial_support)
        whole_read = whole.measure(complete, 50)
        self.assertEqual(whole_read['status'],'complete')
        self.assertEqual(whole_read['missing_donor_support'],[])
        self.assertEqual(whole_read['objective'],whole_read['measured_contribution'])

        # Operation-aware matching must override the generic prefix visits on
        # its own field only, including when the ordered operation is absent.
        match = dict(accepted=True, score=.9, pairs=[[0, 2]])
        matcher = SimpleNamespace(register=lambda query, policy: match)
        field.operation_matchers['stages'] = (matcher, dict(stage_order=[
            dict(phase='symbolic', offset=0), dict(phase='numeric', offset=-1),
            dict(phase='numeric', offset=0)]), [0,1,2])
        member['operations'][0]['matcher'] = 'stages'
        staged, stable = field.register('a', current, prompt_events=1)
        self.assertEqual([c['event'] for c in staged.measure(current,50)['cells']], [0,1,2])
        self.assertIsNone(stable.registration)
        match['accepted'] = False
        missing, stable = field.register('a', current, prompt_events=1)
        read = missing.measure(current,50)
        self.assertEqual(read['status'],'unmeasured')
        self.assertIsNone(read['objective'])
        self.assertFalse(read['operation_registration']['accepted'])
        self.assertGreater(stable.represented,0)
