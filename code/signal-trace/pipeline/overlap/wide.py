"""wide-overlap: the block question on both sides, recorded with only what the trace needs.

From code/signal-trace:
  python3 -m pipeline.overlap.wide capture

The cohort is frozen in the run's members.json (committed November 9, 2026, a06609c9): the 30 recorded answers of the
channel wording that subtract the descent, and for each the unused correct answer that shares the most of its numbers, so
both groups vary in the same way. Those labels were made once, by the selection this program held then, which read each
recorded answer with the answer judge (pipeline/decisions/answers.py). The selection and the judge left the tree on
November 15, 2026, so that no label is made again or read differently: members.json is the record, and the bytes of both
are in git up to efdfc802 (records/development/2026-11-12-path-forward/retired-progress-readers.json).

capture  Each answer of members.json again on GPU0 with the residual stream, held to its recorded tokens. The turns at all
         105 merges are exported and the raw stream is deleted, batch by batch.
"""
import json
import sys

from pipeline.acquisition.export_stream_traces import export_case
from pipeline.acquisition.run_a_observation import run
from pipeline.paths import ORIGINAL_MODEL, ROUND_TWO_RUNS

RUN = ROUND_TWO_RUNS/'wide-overlap'
BATCH = 10  # answers recorded before their raw streams are exported and deleted; only bounds the disk in use


def capture():
    members = json.loads((RUN/'members.json').read_text())
    done = {p.name for p in (RUN/'exported').glob('*')} if (RUN/'exported').exists() else set()
    todo = [m for m in members if m['id'] not in done]
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        name = f"batch-{batch[0]['id']}"
        cases = RUN/'capture'/f'{name}-cases.json'
        cases.parent.mkdir(parents=True, exist_ok=True)
        # The first generated token is decided inside the pass that reads the question, so the recorder cannot
        # be limited to the generated positions without losing it. The whole stream is recorded and deleted after export.
        cases.write_text(json.dumps([dict(id=m['id'], tokens=m['tokens'], max_tokens=m['max_tokens']) for m in batch]))
        raw = RUN/'capture'/name
        run(ORIGINAL_MODEL, cases, raw, 0, 'natural', 'capture', 'residual', RUN/'declaration.json')
        for m in batch:
            result = json.loads((raw/m['id']/'result.json').read_text())
            if [s['selected'] for s in result['steps']] != m['recorded_tokens']:
                raise ValueError(f"{m['id']}: the answer differs from the recorded one")
            export_case(raw/m['id'], RUN/'exported'/m['id'])
            for file in ('stream.f32', 'stream.jsonl'):
                (raw/m['id']/file).unlink()
            print(json.dumps(dict(exported=m['id'], tokens=len(result['steps']))), flush=True)


if __name__ == '__main__':
    dict(capture=capture)[sys.argv[1]]()
