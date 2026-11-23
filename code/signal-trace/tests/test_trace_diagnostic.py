"""Observe two different weight directions during a real tiny installed-weight update on CPU."""
from hashlib import sha256
import json
import os
from contextlib import ExitStack
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pipeline.weight_edit import final_experiment as baseline, paired_turn_signal, prefix_turn_signal, native_key_driver, trace_diagnostic as diagnostic, validation_driver
from tests.test_final_experiment import sample
from tests.test_full_graph import made


class TraceDiagnosticTests(unittest.TestCase):
    def test_public_controller_picks_up_both_existing_jvps_and_regenerates(self):
        self.check_controller_round_trip(paired_turn_signal)

    def test_whole_prefix_reader_uses_both_jvps_and_regenerated_state(self):
        self.check_controller_round_trip(prefix_turn_signal)

    def test_registered_finite_and_both_jvps_survive_install(self):
        self.check_controller_round_trip(prefix_turn_signal, finite=True)

    def check_controller_round_trip(self, reader, finite=False):
        registrar = baseline.ROOT / 'studies/04-round-2-overlap/runs/whole-prefix-frozen-stages/register.py'
        torch.set_num_threads(1)
        app = baseline.controller(baseline.MEASURED)
        with TemporaryDirectory(dir='/tmp') as temp, patch.object(app,'ROOT',Path(temp)), patch.object(baseline,'ROOT',Path(temp)):
            root=Path(temp)
            fixture=made(root,qualified_shapes=finite)
            fixture.a['final_logit_softcapping']=30.
            seed=Path(fixture.file)
            base={'fixture_sha256':sha256(seed.read_bytes()).hexdigest()}
            _,specs=app.state_specs(fixture.tensors,fixture.a)
            chunk=max(s.size for s in specs if not s.sparse_rows)
            with os.fdopen(app.factory.memory_model(seed,base['fixture_sha256']),'r+b') as f:
                fixture.file=Path(f'/proc/self/fd/{f.fileno()}')
                state=app.full_state.FullState(f.fileno(),specs,model_identity={'revision':0},
                    base_identity=base,chunk_elements=chunk)
                _,_,_,forward=sample(app,fixture,state)
                ts=app.turn_pair.turns(forward,expected_merges=105)
                ref=app.vector_chance.Reference([t*.9 for t in ts],[torch.full((len(t),),1e-10,
                    dtype=torch.float64) for t in ts],expected_merges=105)
                identity=app.full_loss_maps.LossIdentity('tiny','a'*64,'b'*64,'tiny')
                def acquire(number,case,model_identity,previous):
                    history,_,parameters,native=sample(app,fixture,state)
                    pairs=app.held.turn_correction.correspondence(ref,native)
                    measured=app.held.turn_correction.measurement(ref,native,pairs)
                    turns=app.turn_pair.turns(native,expected_merges=105)
                    maps=app.full_loss_maps.build_loss_maps(ref._reference,turns,
                        [torch.sqrt(2*c) for c in ref._chance],pairs,expected_merges=105,
                        merge_labels=[dict(block=m//3,branch=('a','b','c')[m%3]) for m in range(105)],
                        identity=identity,observation=dict(kind='fresh_native',model_revision=model_identity,
                            complete=True,acquisition_id=str(number),history_id=str(number)),
                        declared_objective=measured['objective'])
                    folder=root/'loop'/str(number)
                    folder.mkdir(parents=True)
                    app.save(folder/'measurement.json',dict(revision=number,**measured))
                    return native,pairs,history,measured,parameters,app.binding(folder/'measurement.json'),True,maps
                strengths=[float(torch.linalg.vector_norm(t[-1])) for t in ts]
                merge = strengths.index(max(strengths))
                query=ts[merge][-1].detach().double()
                artifact=dict(format='paired-turn-stage/1',key='frozen-pattern',merge=merge,merges=105,
                    width=len(query),offset=0.,axis=(query/torch.linalg.vector_norm(query)).tolist(),
                    anchor_pairs=[ts[merge][-2:].detach().reshape(-1).tolist()],stage_radius_squared=32.)
                inputs,dependencies=[],[]
                if reader is prefix_turn_signal:
                    reference = np.stack([t[1:].detach().float().numpy() for t in ts],axis=1)
                    np.save(root/'turns.npy',reference)
                    method = root/'studies/round/runs/test/register.py'
                    method.parent.mkdir(parents=True)
                    method.write_bytes(registrar.read_bytes())
                    inputs=[app.binding(root/'turns.npy')]
                    dependencies=[app.binding(method)]
                    artifact=dict(format='whole-prefix-fixed-turn-log-gain/1',key='frozen-pattern',
                        reference_turns=inputs[0],registrar=dependencies[0],first_reference_event=1,
                        reference_event=2,merge=merge,reference_turn=(.9*query).tolist(),
                        reference_log_gain=0.,sqrt_gain_weight=1.,cutoff=.01)
                app.save(root/'artifact.json',artifact)
                (root/'reader.py').write_bytes(Path(reader.__file__).read_bytes())
                spec=dict(format=diagnostic.FORMAT,artifact=app.binding(root/'artifact.json'),
                          reader=app.binding(root/'reader.py'),dependencies=dependencies,inputs=inputs)
                app.save(root/'policy.json',spec)
                cfg=dict(trace_diagnostic=dict(policy=app.binding(root/'policy.json'),
                        driver={'record':'validation_driver.py'}),program={'record':'historical-controller.py'},
                    operation='fixture', original_model={'sha256':'a'*64},reference={'sha256':'b'*64},
                    tolerance={'sha256':'c'*64},resume_checkpoint={'sha256':'d'*64})
                app.save(root/'declaration.json',cfg)
                observed={}
                def execute(run,dec,recorder,capture,completion):
                    a=app.acquire(0,dict(tokens=[3,7]),{'revision':0},None)
                    if finite:
                        raw=root/'tiny-native-projection'
                        raw.mkdir()
                        n=len(a[2])
                        calls={m:[dict(call=i,microbatch=i,positions=(i,)) for i in range(n)]
                               for m in range(a[4].layers)}
                        w=app.legacy.projection_adapter.Model(a[4],calls,calls)
                        with (raw/'stream.f32').open('wb') as data,(raw/'stream.jsonl').open('w') as index:
                            for layer in range(w.keys_from):
                                keys,_=w.keys(layer,a[0]['incoming'][3*layer],torch.arange(n))
                                for i,row in enumerate(keys.detach().numpy()):
                                    name=f'Kcur_pos-{layer}'
                                    record=dict(label=name,name=name,operation='ROPE',native_type='f32',token_axis=2,
                                        call=i,microbatch=i,microbatch_start=i,microbatch_count=1,positions=[i],
                                        position_basis='materialized microbatch order',shape=[len(row),1,1,1],
                                        values=len(row),offset_bytes=data.tell())
                                    index.write(json.dumps(record)+'\n')
                                    data.write(row.tobytes())
                        graph=app.graph_for(w,state,a[2],a[0],raw,cfg)
                        self.assertEqual(len(graph.model.origin),w.keys_from)
                    else:
                        graph=app.full_graph.Graph(a[4],a[2],a[0])
                    proposal,direction=app.propose(graph,a[2],ref,a[1],a[3],state,.2,recorder)
                    self.assertEqual(proposal['status'],'ready')
                    app.install(state,a[4],graph,direction,proposal['applied_length'],
                                proposal['candidate_chunk_hashes'],{'revision':1})
                    b=app.acquire(1,dict(tokens=[3,7]),{'revision':1},a[5])
                    observed.update(a=a,b=b,proposal=proposal)
                    completion.update(status='FINISHED',reason='bounded_cpu_fixture')
                def dispatch(python,argv,env):
                    self.assertEqual(Path(argv[1]).name,'validation_driver.py')
                    with patch.object(sys,'argv',argv[1:]):
                        validation_driver.main()
                finite_reader=ExitStack()
                if finite:
                    def cpu_graph(wrapped,state,history,native,raw,declaration):
                        return app.full_graph.Graph(wrapped,history,native)
                    finite_reader.enter_context(patch.object(app,'graph_for',cpu_graph))
                    finite_reader.enter_context(native_key_driver.connected(app,{'card':0}))
                with finite_reader, patch.object(baseline,'run_path',return_value=root), \
                     patch.object(baseline,'inspect_run',return_value={'storage_ready':True}), \
                     patch.object(baseline,'controller',return_value=app), \
                     patch.object(app.held,'check_native_run',return_value=cfg), \
                     patch.object(app,'acquire',acquire),patch.object(app,'execute',execute), \
                     patch.object(sys,'argv',['final_experiment','run',str(root)]), \
                     patch.object(baseline.os,'chdir'),patch.object(baseline.os,'execve',side_effect=dispatch):
                    baseline.main()
                a,b,proposal=observed['a'],observed['b'],observed['proposal']
                p=json.loads((root/proposal['trace_diagnostic']['record']).read_text())
                first,after=(root/'loop'/str(n)/'trace-diagnostic' for n in (0,1))
                source=json.loads((first/'fresh.json').read_text())
                target=json.loads((after/'fresh.json').read_text())
                change=json.loads((after/'transition.json').read_text())
                key='frozen-pattern'
                self.assertTrue(source['cells'], source['match_status'])
                cont=p['phases'][diagnostic.CONTINUOUS][key]
                real=p['phases'][diagnostic.ROUNDED][key]
                if finite:
                    receipt=json.loads((root/proposal['trace_diagnostic_finite']['record']).read_text())
                    self.assertEqual(receipt['status'],'measured')
                    self.assertEqual(receipt['source'],a[5])
                    self.assertEqual(receipt['jvp'],proposal['trace_diagnostic'])
                    [site]=receipt['sites']
                    self.assertEqual(site['cell'],source['cells'][0]['cell'])
                    self.assertEqual(site['finite_change'],site['endpoint']-source['cells'][0]['value'])
                    self.assertTrue(np.isfinite(site['finite_change']))
                    self.assertNotEqual(site['finite_change'],real['predicted_change'])
                    self.assertGreater(receipt['changed_coordinates'],0)
                self.assertEqual(cont['cell'],real['cell'])
                self.assertEqual(cont['cell'],source['cells'][0]['cell'])
                self.assertEqual(cont['origin_difference'],0.)
                self.assertEqual(cont['predicted_change'],cont['tangent']*proposal['applied_length'])
                self.assertEqual(real['coefficient'],1.)
                self.assertEqual(real['predicted_change'],real['tangent'])
                self.assertNotEqual(real['predicted_change'],cont['predicted_change'])
                self.assertEqual(change['source'],a[5])
                self.assertEqual(change['target'],b[5])
                self.assertEqual(change['matches'][key]['observed_change'],
                                 target['cells'][0]['value']-source['cells'][0]['value'])
                self.assertEqual(change['independent_conditioned_replay'],'unavailable')
                self.assertEqual(change['matched_consumed_history'],a[2]==b[2])
                if a[2]==b[2]:
                    fixed=change['fixed_source_cells_native']['matches'][key]
                    self.assertEqual(fixed['source_cell'],fixed['target_cell'])
                else:
                    self.assertEqual(change['fixed_source_cells_native']['scope'],'unavailable')

    def test_ambiguity_partial_and_missing_matches_are_not_pairings(self):
        x=dict(cell=dict(key='x',event=2,merge=4),value=.2)
        y=dict(cell=dict(key='y',event=3,merge=4),value=.3)
        matches=diagnostic.comparison([x,x],[x,y],{})
        self.assertEqual(matches['x']['status'],'ambiguous_source_or_target')
        self.assertEqual(matches['y']['status'],'missing_source_or_target')

    def test_public_registration_rechecks_all_bound_inputs(self):
        with TemporaryDirectory(dir='/tmp') as temp, patch.object(baseline,'ROOT',Path(temp)):
            root=Path(temp)
            def store(name,data):
                path=root/name
                path.write_text(json.dumps(data))
                return baseline.binding(path)
            artifact=store('frozen',{'source':'synthetic'})
            source=store('reader.py',{})
            operand=store('frozen-array',[])
            spec=dict(format=diagnostic.FORMAT,artifact=artifact,reader=source,dependencies=[],inputs=[operand])
            bound=store('policy',spec)
            self.assertEqual(diagnostic.registration(bound),spec)
            (root/'frozen-array').write_text('changed')
            with self.assertRaisesRegex(ValueError,'invalid diagnostic binding'):
                diagnostic.registration(bound)
