"""Independent native prechoice registration using two complete transforms per witness."""
from hashlib import file_digest
from pathlib import Path

import numpy as np

from pipeline.overlap.complete_overlap import unit_steps


class Matcher:
    def __init__(self, vectors, groups, chunk=64):
        self.vectors=vectors
        self.groups=[tuple(g) for g in groups]
        self.chunk=chunk  # Byte-management only. Each dot covers the full native field.

    @classmethod
    def from_spec(cls,spec,root):
        if spec['format']!='whole-field-family-prechoice/1':
            raise ValueError('Unknown whole-field phase policy')
        [binding]=spec['inputs']
        path=Path(root)/binding['record']
        with path.open('rb') as handle:
            if file_digest(handle,'sha256').hexdigest()!=binding['sha256']:
                raise ValueError('Changed phase donor bank')
        with np.load(path,allow_pickle=False) as d:
            return cls(d['vectors'], d['groups'].tolist())

    def candidates(self,field,group):
        """Each physical family casts one equally weighted vote for a query event."""
        if len(field)<2 or field.ndim!=3:
            raise ValueError('Expected decode-only consecutive native turn fields')
        selected=[i for i,g in enumerate(self.groups) if g!=group]
        if not selected: raise ValueError('No independent witness family')
        # vectors [source, downhill/friction, previous/current, native width across merges].
        templates=self.vectors[selected]
        if templates.shape[1:3]!=(2,2) or templates.shape[-1]!=int(np.prod(field.shape[1:])):
            raise ValueError('Native read and frozen phase representations differ')
        n,k=len(field),len(selected)
        scores=np.empty((n,k,2,2),dtype=np.float32)
        matrix=np.ascontiguousarray(templates.reshape(-1,templates.shape[-1]).T)
        for left in range(0,n,self.chunk):
            part=field[left:left+self.chunk]
            if hasattr(part,'detach'):part=part.detach().cpu().float().numpy()
            scores[left:left+len(part)]=(unit_steps(part)@matrix).reshape(len(part),k,2,2)
        pair=scores[:-1,:,:,0]+scores[1:,:,:,1]
        groups=[self.groups[i] for i in selected]
        tallies=np.zeros(n-1,dtype=np.float64)
        evidence=[]
        for family in sorted(set(groups)):
            inside=[i for i,g in enumerate(groups) if g==family]
            avg=pair[:,inside,:].mean(axis=1,dtype=np.float64)
            best=np.flatnonzero(avg[:,0]==avg[:,0].max())
            vote=None
            if len(best)==1 and avg[best[0],0]>avg[best[0],1]:
                vote=int(best[0])+1
                tallies[vote-1]+=1
            evidence.append(dict(family=list(family),event=vote))
        # Strict majority counts the abstaining families in the denominator.
        rows=(np.flatnonzero(2*tallies>len(evidence))+1).tolist()
        return dict(candidates=rows, family_votes=evidence)

    def register(self,query_turns,spec):
        result=self.candidates(query_turns,tuple(spec['query_group']))
        mates=spec.get('mate_events')
        if mates is None:
            return dict(**result,accepted=False,events=[],status='whole_trace_mate_unavailable')
        mates=list(map(int,mates))
        ok=len(mates)==1 and result['candidates']==mates
        return dict(**result,accepted=ok,events=mates if ok else [],mate_events=mates,
                    status='agreed' if ok else 'unresolved')
