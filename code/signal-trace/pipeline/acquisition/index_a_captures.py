"""Index A's completed natural-response captures as acquisition proceeds.

This is a bounded task preprocessor, not a persistent service. It never changes
source captures. If the acquisition already released an archived event file,
the existing CLI restores and verifies that file for this read.
"""
import argparse
import hashlib
import json
import subprocess
import time

from pipeline.native_records.context_index import index_context
from pipeline.acquisition.run_a_baseline import ROOT, RUN


def run(wait):
    cases = json.loads((ROOT/'research/experiments/signal-trace-repair/A-cohort.json').read_text())['cases']
    output = RUN/'context-index'
    output.mkdir(exist_ok=True)
    while True:
        unfinished = []
        for case in cases:
            destination = output/(case['id']+'.json')
            if destination.exists():
                continue
            unfinished.append(case['id'])
            audit_path = ROOT/case['audit']
            if not audit_path.exists():
                continue
            source = ROOT/case['input']
            if hashlib.sha256(source.read_bytes()).hexdigest() != case['input_sha256']:
                raise ValueError('A source question changed after cohort selection')
            capture = ROOT/case['capture']
            events = capture/'signal-events.jsonl'
            restored = not events.exists()
            if restored:
                archive = RUN/'cohort-archives'/(case['id']+'.json')
                if not archive.exists():
                    raise RuntimeError('An audited capture is absent without its verified archive record')
                subprocess.run(['torad-research','traces','download',str(events.relative_to(ROOT/'runs'))], check=True)
            audit = json.loads(audit_path.read_text())
            prompt = json.loads(source.read_text())[0]
            result = json.loads((capture/'result.json').read_text())
            report = index_context(events, audit, prompt['tokens'], result['tokens'])
            report.update(case=case['id'], capture=case['capture'], audit=case['audit'],
                          gpu=case.get('gpu', 0), baseline=case['baseline'],
                          input_sha256=case['input_sha256'], event_file_restored_for_read=restored)
            temporary = destination.with_suffix('.writing')
            temporary.write_text(json.dumps(report,indent=2)+'\n')
            temporary.replace(destination)
            # A restored file is left available for analysis; this process does
            # not decide when another reader has finished with it.
            print(json.dumps({'indexed':case['id'],'frames':len(report['frames']),
                              'predictions':len(report['prediction_executions'])}),flush=True)
        pending = [x['id'] for x in cases if not (output/(x['id']+'.json')).exists()]
        if not pending or not wait:
            print(json.dumps({'expected_cases':len(cases),'indexed':len(cases)-len(pending),'pending':pending}),flush=True)
            return
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    run(parser.parse_args().wait)
