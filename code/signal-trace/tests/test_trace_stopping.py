"""Whole-trajectory stopping on bounded CPU captures. No scientific run is launched."""
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipeline.weight_edit import final_experiment, trace_stopping as stop, validation_driver as driver


HISTORY = [10.,11.,9.,10.,8.,7.,6.,5.,4.]


def observation(obj, retention=0., complete=True, replay=.05):
    return dict(objective=obj,newly_outside_loss=retention,complete=complete,
                repeat_range={'objective':replay,'newly_outside_loss':replay})


class TraceSelectionTests(unittest.TestCase):
    def test_resolution_retention_and_a_single_rebound(self):
        rule = stop.TracePolicy(['fit','other'],HISTORY)
        for i in range(5):
            fit = 10.-i
            panel = {'fit':observation(fit),'other':observation(4.-i,.10 if i else 0.)}
            d = rule.observe(i,fit,panel if rule.due(i) else None)
            if i == 2:
                self.assertFalse(d['promote'])
                self.assertFalse(d['terminal'])
        self.assertEqual(d['decision'],'persistent_loss_on_previously_in_tolerance_cells')
        self.assertEqual(d['selected_generation'],0)
        self.assertEqual(d['persistent_retention_departures'],['other'])
        self.assertEqual(d['capability_success'],'unmeasured')

    def test_resolution_plateau_incomplete_and_monotone_calibration(self):
        for improving,reason in [(True,'fit_gain_without_resolved_development_gain'),
                                 (False,'no_resolved_fit_or_development_gain')]:
            policy=stop.TracePolicy(['fit','other'],HISTORY)
            for i in range(5):
                fit = 10.-i if improving else 10.
                panel = {'fit':observation(fit),'other':observation(4.-i*.01)}
                decision=policy.observe(i,fit,panel if policy.due(i) else None)
            self.assertEqual(decision['decision'],reason)
        rule=stop.TracePolicy(['fit','other'],[5.,4.,3.])
        p={'fit':observation(3.),'other':observation(2.)}
        for i in range(3):
            d=rule.observe(i,3.,p)
        self.assertEqual(d['decision'],'compute_allowance_exhausted')
        rule=stop.TracePolicy(['fit','other'],HISTORY)
        d=rule.observe(0,3.,{'fit':observation(3.),'other':observation(0.,complete=False)})
        self.assertEqual((d['decision'],d['selected_generation']),('incomplete_trace_evaluation',None))

    def test_all_dtw_matches_define_retention_even_if_the_trace_total_falls(self):
        def maps(outside,loss):
            return SimpleNamespace(arrays={'outside':np.array(outside,bool),'pairs':np.array([[0,0],[0,1],[1,2]]),
                'reference_position_merge_loss':np.array(loss)},summary={'objective':float(np.sum(loss)),
                'observation_complete':True,'max_excess':1.,'pairs':3})
        initial=maps([[False,False],[True,False],[False,False]],[[3.,0.],[0.,0.]])
        changed=maps([[True,False],[False,True],[False,False]],[[1.,.3],[0.,0.]])
        protected,a=stop.local_retention(initial)
        _,b=stop.local_retention(changed,protected)
        self.assertEqual(a['baseline_inside_cells'],3)
        self.assertTrue(b['objective'] < a['objective'])
        self.assertEqual((b['newly_outside_cells'],b['newly_outside_loss']),(1,.3))

    def test_documented_native_measurement_boundary_and_continuation(self):
        app=final_experiment.controller(final_experiment.MEASURED)
        with TemporaryDirectory(dir='/tmp') as td, patch.object(driver.baseline,'ROOT',Path(td)), patch.object(app,'ROOT',Path(td)):
            root=Path(td)
            def put(path, data):
                p=root/path
                p.parent.mkdir(parents=True,exist_ok=True)
                app.save(p,data)
                return app.binding(p)
            cases={name:dict(id=name,tokens=t,max_tokens=4) for name,t in [('x',[3,4]),('y',[8,7])]}
            cb={k:put('case-'+k,[v]) for k,v in cases.items()}
            initial=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
            incoming=[initial.clone() for _ in range(3)]
            merged=[initial+torch.tensor([.25,-.4])*(m+1) for m in range(3)]
            original={'incoming':incoming,'merged':merged}
            ref=app.vector_chance.Reference(app.turn_pair.turns(original,expected_merges=3),
                [torch.zeros(3) for _ in range(3)],expected_merges=3)
            identity=app.full_loss_maps.LossIdentity('fixture','a'*64,'b'*64,'fixture')
            identity_dict=vars(identity)
            history=[]
            for i,v in enumerate(HISTORY):
                model=dict(run='history',generation=i)
                capture=put('history/c'+str(i),dict(model_identity=model,cases=cb['x']))
                history.append(put('history/m'+str(i),dict(member='x',objective=v,complete=True,ended=True,
                    loss_identity=identity_dict,model_identity=model,capture=capture,generation=i)))
            plan=dict(format=stop.FORMAT,fit=dict(id='x',tokens=[3,4],family='f'),
                held_families=['sealed'],panel=[dict(id='y',family='d',cases=cb['y'])],
                calibration=put('history.json',{'fit_history':history}))
            policy_binding=put('plan.json',plan)
            dec=dict(functional_loop={'training_members':['x']},native_context=8,loss_identity=identity_dict)
            _,rule=driver.load_policy(policy_binding,dec)
            self.assertEqual((rule.cadence,rule.budget),(2,8))
            invalid=dict(plan,panel=[dict(plan['panel'][0],operations=['label'])])
            with self.assertRaisesRegex(ValueError,'optional frozen target'):
                driver.load_policy(put('bad-plan',invalid),dec)
            invalid=dict(plan,panel=[dict(plan['panel'][0],family='sealed')])
            with self.assertRaisesRegex(ValueError,'overlap'):
                driver.load_policy(put('leak-plan',invalid),dec)
            calls=[]
            def capture(fd,folder,model,case_path,engine,inputs):
                case=json.loads(case_path.read_text())[0]
                self.assertEqual(inputs['member'],case['id'])
                raw=folder/'raw'/case['id']
                raw.mkdir(parents=True)
                result=dict(teacher_token_count=0,conditioned_continuation=False,prompt_tokens=len(case['tokens']),
                    steps=[{'selected':5},{'selected':1}],ended=True,stopped_limit=False)
                app.save(raw/'result.json',result)
                out=dict(model_identity=model,cases=app.binding(case_path))
                app.save(folder/'capture.json',out)
                calls.append((case['id'],model['generation'],fd))
                return out
            def forward(raw,merges,events,width):
                self.assertEqual((len(merges),events,width),(3,3,2))
                return {'incoming':incoming,'merged':[x+torch.tensor([.35,0.]) for x in merged]}
            metrics=[]
            recorder=SimpleNamespace(context=lambda **kw:nullcontext(),span=lambda *a,**kw:nullcontext({}),
                metric=lambda name,v,**kw:metrics.append(name))
            run=root/'run'
            run.mkdir()
            boundary=driver.Boundary(app,run,plan,rule,None)
            declaration=dict(resume_checkpoint=put('base.json',{}),validation_stopping=dict(policy=policy_binding))
            with patch.object(app.factory,'capture',side_effect=capture), patch.object(app.held,'raw_forward',side_effect=forward), \
                 patch.object(app,'archive_source',SimpleNamespace(archive_previous=lambda:None),create=True), \
                 patch.object(driver.signal,'raise_signal') as request, boundary.connected(declaration):
                for i in range(5):
                    app.acquire(run,i,37,ref,identity,cases['x'],cb['x'],None,{'member':'x'},
                        {'merges':['m1','m2','m3']},{'run':'run','generation':i},
                        None,recorder,{'free_required_bytes':0})
            request.assert_called_once()
            receipt=json.loads((run/'validation-decision.json').read_text())
            self.assertEqual(receipt['decision'],'no_resolved_fit_or_development_gain')
            self.assertEqual(receipt['selected']['generation'],0)
            self.assertEqual([g for k,g,fd in calls if k=='y'],[0,0,2,4])
            self.assertEqual([g for k,g,fd in calls if k=='x'],[0,0,1,2,3,4])
            self.assertEqual({fd for k,g,fd in calls},{37})
            self.assertEqual(metrics.count('native.objective'),5)
            self.assertIn('development.trace.y.objective',metrics)
            self.assertNotIn('operations',receipt['traces']['y'])
            self.assertTrue((run/'validation/2/y/local/loss-maps.npz').is_file())
            records=[app.binding(run/'loop'/str(i)/'selection-observation.json') for i in range(3)]
            _,resumed=driver.load_policy(policy_binding,dec)
            recovered=driver.replay(resumed,dict(policy=policy_binding,history=records))
            self.assertEqual(recovered['generation'],0)
            self.assertEqual(len(resumed.fit),3)
            self.assertFalse(resumed.due(3))
