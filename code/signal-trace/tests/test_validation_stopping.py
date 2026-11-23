"""Finite-development policy checks on synthetic data, without scientific execution."""
from contextlib import nullcontext
import json
from pathlib import Path
import signal
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipeline.weight_edit import validation_driver as driver, validation_stopping as stop, turn_pair


NAMES = ('early', 'down', 'drag')
OPS = {name: {'role': ('repair' if name=='down' else 'retain')} for name in NAMES}
CAL = dict(fit_history=[10.,11.,9.,10.,8.,7.,6.,5.,4.],fit_repeat=[10.,10.],
           operation={k:dict(qualified_correct=[.1,.2],repeat=[.12,.12]) for k in NAMES})


def observed(down=.8, early=.1, drag=.1, down_status='wrong'):
    return dict(early=dict(distance=early,status='right'),down=dict(distance=down,status=down_status),
                drag=dict(distance=drag,status='right'))


def step(policy,n,loss,panel):
    return policy.observe(n,loss,panel if policy.due(n) else None)


class StoppingTests(unittest.TestCase):
    def test_two_windows_and_lineage_of_protected_operations(self):
        for decreasing, reason in ((True,'fitted_gain_without_validation_gain'),
                                   (False,'unresolved_fit_and_validation_gain')):
            policy=stop.Policy(CAL,OPS)
            self.assertEqual((policy.cadence,policy.budget),(2,8))
            for i in range(5):
                d=step(policy,i,10-i if decreasing else 10.,observed())
                if d and i<4:
                    self.assertFalse(d['terminal'])
            self.assertEqual((d['decision'],d['selected_generation']),(reason,0))

    def test_transient_control_loss_and_incomplete_repair_dont_qualify(self):
        policy=stop.Policy(CAL,OPS)
        for i in range(7):
            p=observed(early=.6) if i==2 else observed(down=max(.1,.9-i*.12),
                                                       down_status='right' if i==6 else 'wrong')
            d=step(policy,i,10.-i,p)
            if d and i<6:
                self.assertFalse(d['terminal'])
        self.assertEqual(d['decision'],'first_observed_joint_panel_closure')
        self.assertEqual(d['qualified'],['down','drag','early'])
        self.assertEqual(d['selected_generation'],6)

    def test_persistent_regression_and_unscorable_and_budget_are_distinct(self):
        policy=stop.Policy(CAL,OPS)
        for i in range(5):
            d=step(policy,i,10.,observed(drag=.6 if i else .1))
        self.assertEqual(d['decision'],'persistent_trace_departure')
        self.assertEqual(d['right_outside_geometry'], ['drag'])
        policy=stop.Policy(CAL,OPS)
        for i in range(5):
            p=observed()
            if i: p['drag']['status']='wrong'
            d=step(policy,i,10.,p)
        self.assertEqual(d['decision'],'persistent_arithmetic_regression')
        policy=stop.Policy(CAL,OPS)
        bad=observed();bad['down']['status']='absent'
        self.assertEqual(step(policy,0,10.,bad)['decision'],'incomplete_validation')
        policy=stop.Policy(CAL,OPS)
        for i in range(9):
            d=step(policy,i,10.,observed(down=2-i*.18))
        self.assertEqual(d['decision'],'compute_allowance_exhausted')

    def test_physics_panel_reads_native_add_stream_at_the_registered_width(self):
        with TemporaryDirectory() as directory, patch.object(driver.baseline,'ROOT',Path('/tmp')):
            folder=Path(directory)
            initial=torch.tensor([[1.,1.],[1.,1.],[1.,1.]])
            a=torch.tensor([[1.,2.],[1.,2.],[1.,3.]])
            b=torch.tensor([[3.,1.],[3.,1.],[2.,1.]])
            original=dict(incoming=[initial, initial],merged=[a,b])
            ref=np.stack([x.numpy() for x in turn_pair.turns(original,expected_merges=2)],axis=1)
            trace=folder/'turns.npy';np.save(trace,ref)
            ops=[]
            for i,name in enumerate(('down','drag')):
                region=folder/(name+'.json')
                region.write_text(json.dumps(dict(turns=driver.baseline.binding(trace),cells=[[2,i]])))
                ops.append(dict(name=name,reference=driver.baseline.binding(region)))
            changed=a.clone(); changed[-1,:]=torch.tensor([2.,1.])
            app=driver.baseline.controller(driver.baseline.MEASURED)
            nodes=[dict(merge=m,incoming='x',branch=f'branch{m}',connection=f'y{m}') for m in range(2)]
            case=dict(id='example',tokens=[2,3],max_tokens=3)
            cases=folder/'cases.json'
            cases.write_text(json.dumps([case]))
            panel=[dict(id='example',cases=driver.baseline.binding(cases),operations=ops)]
            boundary=driver.Boundary(app,folder,dict(fit=dict(tokens=[1]),panel=panel,scorer={'record':'fixture'}),None,
                SimpleNamespace(score=lambda *args:{key:'right' for key in ('down','drag')}))
            # The real recorder adapter consumes these typed rows and checks each declared ADD.
            def capture(fd,destination,identity,*unused):
                raw=destination/'raw/example'
                raw.mkdir(parents=True)
                result=dict(teacher_token_count=0,conditioned_continuation=False,prompt_tokens=2,
                            steps=[{'selected':4},{'selected':1}],ended=True,stopped_limit=False)
                app.save(raw/'result.json',result)
                outputs=[('x',initial,'NONE',[])]
                for m,out in enumerate((changed,b)):
                    outputs.extend([(f'branch{m}',out-initial,'NONE',[]),(f'y{m}',out,'ADD',['x',f'branch{m}'])])
                with (raw/'stream.f32').open('wb') as data,(raw/'stream.jsonl').open('w') as index:
                    for label,tensor,op,sources in outputs:
                        array=tensor.numpy().astype('<f4')
                        row=dict(label=label,shape=[2,3],positions=[0,1,2],values=6,token_axis=1,
                                 native_type='f32',offset_bytes=data.tell(),operation=op,source_names=sources)
                        data.write(array.tobytes())
                        index.write(json.dumps(row)+'\n')
                app.save(destination/'capture.json',dict(model_identity=identity))
                return dict(model_identity=identity)
            args=dict(case={'tokens':[1]},budget={'free_required_bytes':0},fd=0,inputs={},
                recorder=SimpleNamespace(context=lambda **kw:nullcontext(),span=lambda *a,**kw:nullcontext()),
                reading={'merges':nodes},reference=SimpleNamespace(width=2),engine=None,model_identity={'generation':0})
            with patch.object(app,'ROOT',Path('/tmp')),patch.object(app.factory,'capture',side_effect=capture):
                d=boundary.panel(0,args)
                self.assertGreater(d['example/down']['distance'],0.)
                self.assertEqual(d['example/drag']['distance'],0.)
                args['reference']=SimpleNamespace(width=3)
                with self.assertRaisesRegex(ValueError,'finite width differs'):
                    boundary.panel(1,args)

    def test_synchronous_boundary_finishes_the_measured_revision(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            def acquire(number):
                (root/'loop'/str(number)).mkdir(parents=True)
                return (None,None,None,{'objective':10.-number},None,{'record':str(number)},True,None)
            app=SimpleNamespace(acquire=acquire,continuation=SimpleNamespace(restore=lambda:object()),
                archive_source=SimpleNamespace(archive_previous=lambda:{'archives':['prior']}),
                save=lambda path,value:path.write_text(json.dumps(value)))
            boundary=driver.Boundary(app,root,{},stop.Policy(CAL,OPS),None)
            def panel(n,args):
                (root/'validation'/str(n)).mkdir(parents=True)
                return observed(down=.1,down_status='right') if n==2 else observed()
            with patch.object(boundary,'panel',side_effect=panel),patch.object(signal,'raise_signal') as request:
                with boundary.connected(dict(resume_checkpoint={'record':'base'},
                                              validation_stopping={'policy':{'record':'policy'}})):
                    for n in range(9):
                        result=app.acquire(n)
                        if boundary.terminal: break
                    self.assertEqual(n,2)
                    self.assertEqual(result[5]['record'],'2')
                    self.assertEqual(app.archive_source.archive_previous()['archives'],[])
                request.assert_called_once_with(signal.SIGUSR1)
            receipt=json.loads((root/'validation-decision.json').read_text())
            self.assertEqual(receipt['selected'],{'generation':2,'exit_state':'final-state.json',
                                                 'exit_checkpoint':'final-checkpoint/state.json'})
            self.assertIs(app.acquire,acquire)

    def test_replay_preserves_spent_updates_and_incumbent(self):
        with TemporaryDirectory() as tmp, patch.object(driver.baseline,'ROOT',Path(tmp)):
            root=Path(tmp)
            def store(name,data):
                p=root/name
                p.write_text(json.dumps(data))
                return driver.baseline.binding(p)
            source=store('policy',{})
            incumbent=dict(generation=0,state=store('base',{}))
            history=[]
            for i in range(3):
                measure=store('m'+str(i),dict(ended=True,complete=True,objective=10.-i))
                history.append(store('h'+str(i),dict(policy=source,generation=i,measurement=measure,
                    panel=observed() if i%2==0 else None,selected=incumbent)))
            policy=stop.Policy(CAL,OPS)
            self.assertEqual(driver.replay(policy,dict(policy=source,history=history)),incumbent)
            self.assertEqual(len(policy.fit),3)
            self.assertIsNone(policy.observe(3,7.))
            self.assertEqual(policy.observe(4,6.,observed())['decision'],'fitted_gain_without_validation_gain')

    def test_bound_panel_admission_and_blind_split(self):
        with TemporaryDirectory() as temp, patch.object(driver.baseline,'ROOT',Path(temp)):
            root=Path(temp)
            def put(path,data):
                p=root/path
                p.write_text(json.dumps(data))
                return driver.baseline.binding(p)
            marker=put('scorer.py','synthetic registration only')
            trace=put('fixture-trace','synthetic trace binding')
            region=put('region',dict(turns=trace,cells=[[0,0]]))
            def measure(index,mode,key,prompt,label,value):
                ident={'run':'fixed','generation':0}
                case=put(str(index)+'-prompt',[{'id':'example','tokens':prompt}])
                cap=put(str(index)+'-capture',dict(acquisition=index,model_identity=ident,cases=case))
                weights=put('fixed-weights',dict(model_identity=ident))
                return put(str(index)+'-calibration',dict(capture=cap,complete=True,weights=weights,
                    family='dev',prompt_sha256=driver.prompt_digest(prompt),operation=key,instrument=label,
                    value=value,status='wrong' if mode=='qualified_wrong' else 'right'))
            history=[]
            for i,x in enumerate(CAL['fit_history']):
                history.append(put('fit-'+str(i),dict(generation=i,complete=True,ended=True,loss_identity={},
                    member='fit',model_identity=dict(run='calibration',generation=i),
                    capture=put('native-'+str(i),{'model_identity':dict(run='calibration',generation=i)}),objective=x)))
            case_tokens=[2,3]
            fitted_tokens=[1,2]
            op=[dict(name=n,role=OPS[n]['role'],capability=g,reference=region) for n,g in
                zip(NAMES,('earlier_repair','descent','friction'),strict=True)]
            calibration=dict(fit_history=history,fit_repeat=[
                measure('f'+str(i),'repeat',None,fitted_tokens,{},10.) for i in (1,2)],operation={})
            for o in op:
                k='example/'+o['name']
                modes={'repeat': [.1,.1], 'qualified_correct':[.1,.2]}
                if o['role']=='repair': modes['qualified_wrong']=[.8]
                calibration['operation'][k]={m:[measure(o['name']+m+str(i),m,k,case_tokens,
                    {'metric':driver.METRIC,'scorer':marker,'reference':region},x) for i,x in enumerate(xs)] for m,xs in modes.items()}
            item=dict(id='example',family='dev',cases=put('cases',[dict(id='example',tokens=case_tokens,max_tokens=5)]),
                      operations=op)
            spec=dict(format='development-operation-stopping/1',fit=dict(id='fit',family='fit-family',tokens=fitted_tokens),
                      held_families=['sealed-family-label'],panel=[item],calibration=put('cal',calibration),
                      scorer=marker,scorer_dependencies=[])
            dec=dict(functional_loop={'training_members':['fit']},loss_identity={},native_context=8)
            _,policy=driver.load_policy(put('plan',spec),dec)
            self.assertEqual((policy.cadence,policy.budget),(2,8))
            item['family']='sealed-family-label'
            with self.assertRaisesRegex(ValueError,'overlapping development'):
                driver.load_policy(put('overlap',spec),dec)
