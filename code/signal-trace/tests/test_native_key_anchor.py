import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from pipeline.weight_edit import final_experiment as baseline, full_parameters, full_graph, native_key_anchor as anchor, native_key_finite as finite
from tests.test_full_graph import made, HISTORY


class NativeKeyAnchorTest(unittest.TestCase):
    def test_complete_producer_registration_current_key_derivatives_and_cache_reuse(self):
        torch.set_num_threads(1)
        with TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder)
            app=baseline.controller(baseline.MEASURED)
            source=made(root,qualified_shapes=True)
            params=full_parameters.ParameterModel(source,HISTORY)
            with torch.no_grad():
                native=params.forward(HISTORY,again=False)
            calls={m:[dict(call=i,microbatch=i,positions=(i,)) for i in range(len(HISTORY))]
                   for m in range(params.layers)}
            wrapped=app.legacy.projection_adapter.Model(params,calls,calls)
            ref={}
            rows=[]
            with (root/'stream.f32').open('wb') as out:
                for layer in range(params.keys_from):
                    width=params.head(layer)
                    # Distinct position and producer at both the rounding and positional boundaries.
                    ref[layer]=np.linspace(.00123,.67123,len(HISTORY)*width,dtype='f4').reshape(-1,width)+layer
                    for i,a in enumerate(ref[layer]):
                        label=f'Kcur_pos-{layer}'
                        rows.append(dict(call=i,microbatch=i,microbatch_start=i,microbatch_count=1,
                            label=label,name=label,position_basis='materialized microbatch order',positions=[i],
                            native_type='f32',operation='ROPE',token_axis=2,shape=[width,1,1,1],
                            values=width,offset_bytes=out.tell()))
                        out.write(a.tobytes())
            def index(records):
                (root/'stream.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            index(rows)
            k=anchor.wrap(wrapped,HISTORY,native,root)
            graph=full_graph.Graph(k,HISTORY,native)
            f=graph.forward(HISTORY,again=False)
            for layer,(held,v) in f['keys'].items():
                torch.testing.assert_close(held,torch.from_numpy(ref[layer]).half().float(),atol=0,rtol=0)
                self.assertEqual(v.dtype,torch.float32)
            self.assertEqual(set(f['keys']),set(range(params.keys_from)))
            self.assertTrue(all(params.keys_of(i) in f['keys'] for i in range(params.layers)))
            x=native['incoming'][0].detach().requires_grad_()
            pos=torch.arange(len(HISTORY))
            keys,values=k.keys(0,x,pos)
            grad,=torch.autograd.grad(keys.sum(),x)
            self.assertGreater(float(grad.abs().max()),0.)
            with torch.autograd.forward_ad.dual_level():
                query=torch.autograd.forward_ad.make_dual(x,torch.ones_like(x))
                dual,_=k.keys(0,query,pos)
                _,tangent=torch.autograd.forward_ad.unpack_dual(dual)
                torch.testing.assert_close(grad.sum(),tangent.sum(),atol=1e-4,rtol=1e-4)
            for actual,wanted in zip(f['merged'],native['merged'],strict=True):
                self.assertTrue(torch.equal(actual,wanted))
            # The reader rejects a missing producer call and a shifted coordinate.
            index(rows[:-1])
            with self.assertRaisesRegex(ValueError,'Missing K-producing calls'):
                anchor.load_native_keys(root,wrapped,len(HISTORY))
            index([dict(rows[0],positions=[1]),*rows[1:]])
            with self.assertRaisesRegex(ValueError,'mismatched native K call'):
                anchor.load_native_keys(root,wrapped,len(HISTORY))

    def test_absent_ambiguous_and_failed_finite_read_restore_source_mapping(self):
        original={'w':torch.tensor(1.)}
        graph=SimpleNamespace(w=original)
        candidate={'w':torch.tensor(2.)}
        artifact={}
        calls=[]
        def forward(history,again=False):
            calls.append(history)
            return graph.w
        reader=SimpleNamespace(score=lambda f,c,a: f['w'])
        def cell(i):
            return dict(cell=dict(key='stage',event=i,merge=0),value=1.)
        self.assertEqual(finite.read_at_cells(graph,forward,(1,),candidate,reader,artifact,[])['status'],'no_match')
        self.assertFalse(calls)
        out=finite.read_at_cells(graph,forward,(1,),candidate,reader,artifact,[cell(0),cell(1)])
        self.assertEqual(out['status'],'ambiguous_registration')
        self.assertEqual([r['finite_change'] for r in out['sites']],[1.,1.])
        self.assertIs(graph.w,original)
        def broken(*args,**kwargs):
            raise RuntimeError('missing finite operand')
        with self.assertRaisesRegex(RuntimeError,'missing finite operand'):
            finite.read_at_cells(graph,broken,(1,),candidate,reader,artifact,[cell(0)])
        self.assertIs(graph.w,original)
