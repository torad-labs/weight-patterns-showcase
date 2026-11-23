"""Finite held-history observations at the exact registered sites of an upcoming install."""
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
import inspect

import numpy as np
import torch

from pipeline.weight_edit import trace_diagnostic

METRIC='proposal.trace_signal.rounded_finite_change'


class Endpoint(Mapping):
    """Overlay exact previewed endpoint entries; materialize only a requested changed tensor."""
    def __init__(self, original):
        self.original=original
        self.changed=defaultdict(list)

    def __iter__(self):
        return iter(self.original)

    def __len__(self):
        return len(self.original)

    def __getitem__(self,key):
        original=self.original[key]
        if key not in self.changed:
            return original
        result=original.detach().clone()
        flat=result.view(-1)
        for offset, indices, stored in self.changed[key]:
            flat[indices+offset]=stored
        return result

    def observe(self,name,cursor,chunk):
        # Compare the actual encoded bytes, which also preserves signed-zero changes.
        count=len(chunk.stored)
        itemsize=len(chunk.raw)//count
        kind={2:'<u2',4:'<u4'}[itemsize]
        changed=np.flatnonzero(np.frombuffer(chunk.before,dtype=kind)!=np.frombuffer(chunk.raw,dtype=kind))
        if len(changed):
            self.changed[name].append((cursor,torch.from_numpy(changed),torch.from_numpy(chunk.stored[changed].copy())))

    @contextmanager
    def during_preview(self,state):
        prior=state.preview
        def collecting(name,*args,**kwargs):
            cursor=0
            for chunk in prior(name,*args,**kwargs):
                # Sparse physical rows appear in compact leaf order throughout preview.
                self.observe(name,cursor,chunk)
                cursor+=len(chunk.stored)
                yield chunk
            if cursor!=self.original[name].numel():
                raise ValueError(f'Rounded preview does not cover the active leaf: {name}')
        state.preview=collecting
        try:
            yield
        finally:
            state.preview=prior


def read_at_cells(graph,forward,history,endpoint,reader,artifact,source):
    rows=[]
    if not source:
        return dict(status='no_match',sites=rows)
    previous=graph.w
    try:
        with torch.no_grad():
            graph.w=endpoint
            finite=forward(history,again=False)
            for row in source:
                value=trace_diagnostic.scalar(reader.score(finite,row['cell'],artifact))
                rows.append(dict(cell=row['cell'],native_origin=row['value'],endpoint=value,
                                 finite_change=value-row['value']))
    finally:
        graph.w=previous
    keys=[r['cell']['key'] for r in rows]
    duplicates=sorted({key for key in keys if keys.count(key)>1})
    return dict(status='ambiguous_registration' if duplicates else 'measured',ambiguous_keys=duplicates,sites=rows)


class Diagnostic(trace_diagnostic.Diagnostic):
    @contextmanager
    def connected(self):
        with super().connected():
            original=self.app.propose
            @wraps(original)
            def observed(*args,**kwargs):
                b=inspect.signature(original).bind(*args,**kwargs).arguments
                graph=b['graph']
                forward=graph.forward  # before the gradient routine adds its origin-only check
                endpoint=Endpoint(graph.w)
                source=self.current
                # No candidate side effects, full-model clone, or second native decode.
                with endpoint.during_preview(b['state']):
                    proposal,direction=original(*args,**kwargs)
                if direction is None:
                    return proposal,direction
                with b['recorder'].span('proposal.finite_trace_signal',detached=True):
                    finite=read_at_cells(graph,forward,b['history'],endpoint,self.reader,self.artifact,source['cells'])
                finite.update(scope='finite anchored graph at exact rounded weights on frozen original history and sites',
                              policy=self.binding,source=source['measurement'],jvp=proposal['trace_diagnostic'],
                              changed_coordinates=sum(len(ids) for blocks in endpoint.changed.values() for _,ids,_ in blocks))
                path=self.run/'loop'/str(source['generation'])/'trace-diagnostic'/'finite.json'
                self.app.save(path,finite)
                if finite['status']=='measured' and len(finite['sites'])==1:
                    r=finite['sites'][0]
                    b['recorder'].metric(METRIC,r['finite_change'],step=source['generation']+1,
                                         attributes={'diagnostic_key':r['cell']['key']})
                return dict(proposal,trace_diagnostic_finite=self.app.binding(path)),direction
            self.app.propose=observed
            try:
                yield
            finally:
                self.app.propose=original
