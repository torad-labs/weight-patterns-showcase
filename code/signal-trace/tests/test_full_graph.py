"""Full538/105 CPU graph plumbing on a synthetic Gemma4-shaped model.

No research/native acquisition or answer targets. Synthetic dimensions are small;
actual retained metadata independently establish the physical inventory/memory.
"""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.autograd import forward_ad

from pipeline.weight_edit import full_parameters as parameters, full_graph

ROOT = Path(__file__).resolve().parents[3]
location = ROOT/'records/development/2026-11-12-path-forward/torch_forward.py'
spec = importlib.util.spec_from_file_location('full_scope_forward_fixture', location)
tf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tf)
HISTORY = (3, 7, 1, 4, 4)


def made(folder, qualified_shapes=False):
    rng, tensors, chunks, offset = np.random.default_rng(71), {}, [], 0
    width,per = (64,64) if qualified_shapes else (16,4)
    full_head,local_head = (64,32) if qualified_shapes else (8,4)
    a = dict(layers=35, width=width, heads=2, key_heads=1, per_layer_width=per,
             head_width=full_head, head_width_windowed=local_head, window=3,
             windowed=[layer % 5 != 4 for layer in range(35)], layers_sharing_keys=20,
             norm_eps=1e-6, rope_base=1_000_000., rope_base_windowed=10_000.)
    def put(name, shape, dtype, values=None):
        nonlocal offset
        x = (rng.normal(size=tuple(reversed(shape))).astype('<f4') / np.sqrt(shape[0])
             if values is None else np.asarray(values,dtype='<f4').reshape(tuple(reversed(shape))))
        x=x.astype('<f4')
        raw = ((x.view('<u4') >> 16).astype('<u2').tobytes() if dtype=='bf16' else x.tobytes())
        tensors[name] = dict(name=name,native_type=dtype,shape=list(shape)+[1]*(4-len(shape)),offset_bytes=offset,bytes=len(raw))
        offset += len(raw);chunks.append(raw)
    put('rope_freqs.weight',(full_head//2,),'f32',np.ones(full_head//2))
    put('token_embd.weight',(width,12),'bf16')
    put('per_layer_token_embd.weight',(35*per,12),'bf16')
    put('per_layer_model_proj.weight',(width,35*per),'bf16')
    put('per_layer_proj_norm.weight',(per,),'f32',rng.uniform(.8,1.2,per))
    put('output_norm.weight',(width,),'f32',np.ones(width))
    for layer in range(35):
        head = local_head if a['windowed'][layer] else full_head
        feed = (128 if qualified_shapes else 24)*(1 if layer<15 else 2)
        shapes = dict(ffn_down=(feed,width),ffn_gate=(width,feed),ffn_up=(width,feed),
                      inp_gate=(width,per),proj=(per,width),attn_k=(width,head),attn_v=(width,head),
                      attn_q=(width,2*head),attn_output=(2*head,width))
        for part in parameters.PARTS:
            name=f'blk.{layer}.{part}.weight'
            if part in shapes:put(name,shapes[part],'bf16')
            else:
                size=1 if part=='layer_output_scale' else head if part in ('attn_k_norm','attn_q_norm') else width
                put(name,(size,),'f32',rng.uniform(.8,1.2,size))
    file=Path(folder)/'synthetic.gguf';file.write_bytes(b''.join(chunks))
    return tf.Model(file,tensors,a,device='cpu',leaves=lambda _:False,cache='float16')


class Reference:
    merges=105
    def __init__(self,forward):
        self.targets=[value.detach()*.8 for value in tf.turns(forward)]
    def measure(self,forward,pairs):
        for value,target in zip(tf.turns(forward),self.targets):
            yield SimpleNamespace(residual=value-target,outside=torch.ones(len(value),dtype=torch.bool))


class FullGraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(dir='/tmp')
        self.source=made(self.temporary.name)
        # Include a dormant previously touched row to qualify cumulative state.
        rows=tuple(sorted(set(HISTORY)|{11}))
        self.model=parameters.ParameterModel(self.source,HISTORY,row_ids={name:rows for name in parameters.TABLES})
        with torch.no_grad():self.native=self.model.forward(HISTORY,again=False)
        self.graph=full_graph.Graph(self.model,HISTORY,self.native)
        self.reference=Reference(self.native)
        self.pairs=np.column_stack((np.arange(len(HISTORY)),np.arange(len(HISTORY))))
    def tearDown(self):self.temporary.cleanup()

    def test_cpu_shared_operand_policy_and_both_derivatives(self):
        # Native CPU conversion keeps subnormal codes and ties to even.
        bits=torch.tensor([0x3f808000,0x3f818000,0x00010000,-2147483648],dtype=torch.int32)
        torch.testing.assert_close(parameters.cpu_bf16_round(bits.view(torch.float32)).view(torch.int32),
                                   torch.tensor([0x3f800000,0x3f820000,0x00010000,-2147483648],dtype=torch.int32))
        x=torch.tensor([[1.003,2.009],[3.007,-4.019],[2.001,.013]],requires_grad=True)
        w=torch.tensor([[.333,1.008],[-.509,.711]],requires_grad=True)
        groups=((2,),(0,1))
        result=parameters.cpu_shared_linear(x,w,groups=groups)
        xq,wq=parameters.cpu_bf16_round(x),parameters.cpu_bf16_round(w)
        torch.testing.assert_close(result,xq@wq.T)
        gx,gw=torch.autograd.grad(result.sum(),(x,w))
        torch.testing.assert_close(gx,torch.ones_like(result)@wq)
        torch.testing.assert_close(gw,torch.ones_like(result).T@xq)
        dx,dw=torch.full_like(x,.2),torch.full_like(w,-.1)
        with forward_ad.dual_level():
            out=parameters.cpu_shared_linear(forward_ad.make_dual(x.detach(),dx),forward_ad.make_dual(w.detach(),dw),groups=groups)
            _,q=forward_ad.unpack_dual(out)
            torch.testing.assert_close(q,dx@wq.T+xq@dw.T)
        path=Path(self.temporary.name)/'input-index.jsonl'
        groups=((4,),(0,1),(2,3))
        path.write_text(''.join(json.dumps(dict(label='inp_scaled',native_type='f32',
            shape=[16,len(group),1,1],microbatch_count=len(group),positions=list(group)))+'\n' for group in groups))
        self.assertEqual(parameters.shared_calls_from_index(path,events=5,width=16),groups)
        with self.assertRaisesRegex(ValueError,'each event once'):
            parameters.shared_calls_from_index(path,events=6,width=16)
        dense='blk.0.attn_q.weight'
        self.assertEqual(self.model.weights[dense].data_ptr(),self.source.w[dense].data_ptr())

    def test_real_metadata_inventory_dtype_exclusions_and_memory(self):
        root=ROOT/'studies/04-round-2-overlap/runs'
        tensors=json.loads((root/'sent-overlap/wrong-inputs/tensors.json').read_text())
        architecture=json.loads((root/'functional-layout-original/source-records/architecture.json').read_text())
        specs=parameters.inventory(tensors,architecture,HISTORY)
        eligible=[row for row in specs if row.eligible]
        self.assertEqual((len(specs),len(eligible)),(601,538))
        self.assertEqual(sum(row.native_type=='bf16' for row in eligible),278)
        self.assertEqual(sum(row.native_type=='f32' for row in eligible),260)
        self.assertEqual(sum(row.reason.startswith('unused stored') for row in specs),60)
        self.assertEqual(next(row for row in specs if row.name=='rope_freqs.weight').eligible,False)
        plan=parameters.memory_plan(specs,events=612,width=1536,layers=35)
        self.assertEqual(plan['dense_f32_leaf_bytes'],7_508_417_672)
        self.assertLess(plan['compact_embedding_leaf_bytes'],sum(tensors[n]['bytes'] for n in parameters.TABLES))
        self.assertEqual(next(row for row in specs if row.name==parameters.TABLES[0]).aliases,('output.weight',))

    def test_shared_embedding_candidates_duals_and_dormant_rows(self):
        name=parameters.TABLES[0]; original=self.graph.w
        values=original[name].detach().clone().requires_grad_(True)
        self.graph.w=original|{name:values}
        Path(self.source.file).unlink()  # all future gathers must use the bound leaf
        selected=self.model.rows(name,(4,4))
        gradient=torch.autograd.grad(selected.sum(),values)[0]
        index=self.model.parameter_rows[name].index(4)
        torch.testing.assert_close(gradient[index],torch.full_like(gradient[index],2))
        dormant=self.model.parameter_rows[name].index(11)
        self.assertEqual(float(gradient[dormant].abs().sum()),0.)
        with forward_ad.dual_level():
            delta=torch.arange(values.numel(),dtype=torch.float32).reshape(values.shape)
            self.graph.w=original|{name:forward_ad.make_dual(values.detach(),delta)}
            primal,tangent=forward_ad.unpack_dual(self.model.rows(name,(4,4)))
            torch.testing.assert_close(tangent,delta[[index,index]])
            torch.testing.assert_close(primal,values.detach()[[index,index]])
        self.graph.w=original
        self.assertEqual(len(self.graph.leaves),538)
        specs=parameters.inventory(self.source.tensors,self.source.a,HISTORY,
            row_ids={name:np.array(self.graph.parameter_rows[name],dtype=np.int64)})
        selected=next(row for row in specs if row.name==name)
        self.assertTrue(all(type(value) is int for value in selected.row_ids))
        with self.assertRaisesRegex(ValueError,'must be integers'):
            parameters.inventory(self.source.tensors,self.source.a,HISTORY,row_ids={name:(1.5,)})
        with self.assertRaisesRegex(ValueError,'include the full current history'):
            parameters.inventory(self.source.tensors,self.source.a,HISTORY,row_ids={name:(1,)})

    def test_full105_vjp_matches_independent_unanchored_graph_and_joint_jvp(self):
        actual=full_graph.weight_gradient(self.graph,HISTORY,self.reference,pairs=self.pairs)
        self.assertEqual((len(actual.direction),actual.before['pair_merge_cells']),(538,105*len(HISTORY)))
        self.assertTrue(all(not row['gradient_was_none'] for row in actual.diagnostics.values()))
        # Independent ordinary forward has no native anchor construction/checkpoint.
        forward=self.model.forward(HISTORY,again=False)
        objective=torch.stack([o.residual.double().square().sum() for o in self.reference.measure(forward,self.pairs)]).sum()
        expected=torch.autograd.grad(objective,tuple(self.model.leaves.values()))
        for (name,_),gradient in zip(self.model.leaves.items(),expected):
            torch.testing.assert_close(actual.direction[name],-gradient,rtol=2e-4,atol=2e-6)
        for name in (*parameters.TABLES,'per_layer_model_proj.weight','per_layer_proj_norm.weight',
                     'blk.13.attn_k.weight','blk.14.attn_v.weight','blk.0.ffn_norm.weight',
                     'blk.0.layer_output_scale.weight','blk.34.inp_gate.weight'):
            self.assertEqual(actual.diagnostics[name]['status'],'connected_nonzero',name)
        response=full_graph.directional_response(self.graph,HISTORY,actual.direction,
            lambda f:(o.residual for o in self.reference.measure(f,self.pairs)))
        self.assertAlmostEqual(response.slope/-actual.norm_squared,1.,places=4)
        self.assertEqual(response.status,'proposal')
        for name in parameters.TABLES:
            dormant=self.model.parameter_rows[name].index(11)
            self.assertEqual(float(actual.direction[name][dormant].abs().sum()),0.)

    def test_existing_precision_wrappers_accept_full_scope_cpu_single_calls(self):
        def load(name,path):
            specification=importlib.util.spec_from_file_location(name,path)
            module=importlib.util.module_from_spec(specification)
            specification.loader.exec_module(module)
            return module
        ffn=load('full_ffn_fixture',ROOT/'studies/04-round-2-overlap/runs/vector-chance-loss-rise-diagnosis/per-merge-response/ffn-native-precision-adapter/adapter.py')
        projection=load('full_projection_fixture',ROOT/'records/development/2026-11-16-projection-gradient-adapter/adapter.py')
        with tempfile.TemporaryDirectory(dir='/tmp') as directory:
            source=made(directory,qualified_shapes=True)
            model=parameters.ParameterModel(source,HISTORY)
            with torch.no_grad():native=model.forward(HISTORY,again=False)
            calls={layer:[dict(positions=(i,)) for i in range(len(HISTORY))] for layer in range(35)}
            wrapped=ffn.Model(model,calls,rms=tf.rms,gelu=tf.gelu,batch_single_column=True)
            wrapped=projection.Model(wrapped,calls,calls)
            graph=full_graph.Graph(wrapped,HISTORY,native)
            reference=Reference(native)
            derivative=full_graph.weight_gradient(graph,HISTORY,reference,pairs=self.pairs)
            self.assertEqual(len(derivative.direction),538)
            self.assertTrue(all(not d['gradient_was_none'] and d['finite'] for d in derivative.diagnostics.values()))
            self.assertTrue(all(d['nonzero_entries']>=0 for d in derivative.diagnostics.values()))
            response=full_graph.directional_response(graph,HISTORY,derivative.direction,
                lambda f:(o.residual for o in reference.measure(f,self.pairs)))
            self.assertAlmostEqual(response.slope/-derivative.norm_squared,1.,places=4)

    def test_connected_zero_is_distinct_from_detached_and_mapping_restores(self):
        original=self.graph.w
        _,_,_,diagnostics=full_graph._derivative(self.graph,HISTORY,
            lambda f:(f['merged'][0].sum()*0,{}))
        self.assertEqual(diagnostics[parameters.TABLES[0]]['status'],'connected_zero')
        self.assertEqual(diagnostics['blk.34.ffn_down.weight']['status'],'disconnected_or_detached')
        self.assertIs(self.graph.w,original)
        def broken(_):raise ValueError('synthetic objective failed')
        with self.assertRaisesRegex(ValueError,'synthetic objective failed'):
            full_graph._derivative(self.graph,HISTORY,broken)
        self.assertIs(self.graph.w,original)


if __name__=='__main__':unittest.main()
