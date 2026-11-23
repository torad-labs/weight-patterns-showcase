"""Intersect the archived two-turn role predictions with a unique whole-trace mate."""
from pipeline.weight_edit.operand_phase import Matcher as FrozenMatcher


class Matcher(FrozenMatcher):
    def register(self,query_turns,spec):
        reading=super().register(query_turns,spec)
        supplied=spec.get('mate_events')
        mates=[] if supplied is None else list(map(int,supplied))
        ok=len(mates)==1 and mates[0] in reading['candidates']
        return dict(reading,accepted=ok,events=mates if ok else [],mate_events=mates,
                    status='agreed' if ok else 'unresolved')
