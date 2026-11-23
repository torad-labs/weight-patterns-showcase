"""Run A's baseline and screening on the original GPU under its shared lock.

No edits or trace fitting. Reproduce all historical full-score hashes before
screening new questions. Never overwrite an existing output directory.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from pipeline.paths import ROOT
RUN=ROOT/'runs/legs/gemma4-e2b/signal-trace-A-2026-11-07'
OBSERVER=ROOT/'code/signal-trace/recorder/observe'
MODEL=ROOT/'models/gemma4/gemma4-e2b/gguf-bf16-bartowski/google_gemma-4-E2B-it-bf16.gguf'
BACKENDS=ROOT/'dependencies/llama-cpp-build'
OLD=ROOT/'runs/legs/gemma4-e2b/physics-weight-edit-2026-11-07/original-plain'


def status(state, **kw):
    row={'at_ct':datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(), 'state':state,**kw}
    with (RUN/'acquisition-progress.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def run(label,cases):
    out=RUN/label
    if out.exists():
        raise RuntimeError(f'Output exists, inspect before continuing: {out}')
    cmd=[str(OBSERVER),str(MODEL),str(out),str(cases),str(BACKENDS),'plain','full','trace']
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0',GGML_CUDA_DISABLE_FUSION='1')
    status('running',label=label,command=cmd)
    with (RUN/f'{label}.stdout.jsonl').open('x') as stdout,(RUN/f'{label}.stderr.txt').open('x') as stderr:
        subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,check=True)
    status('completed',label=label)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    lockpath=Path(f'/run/user/{os.getuid()}/gpu0.lock')
    status('waiting_for_existing_gpu0_lock',lock=str(lockpath))
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        status('gpu0_acquired')
        free=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if int(free.strip())<11000:
            raise RuntimeError('GPU0 has insufficient free memory even after acquiring its lock; leave other processes untouched')
        run('historical-plain',RUN/'historical-cases.json')
        checks=[]
        for c in json.loads((RUN/'historical-cases.json').read_text()):
            cid=c['id'];old=json.loads((OLD/cid/'result.json').read_text());new=json.loads((RUN/'historical-plain'/cid/'result.json').read_text())
            tokens_equal=old['tokens']==new['tokens']
            scores_equal=[s['logits_sha256'] for s in old['steps']]==[s['logits_sha256'] for s in new['steps']]
            checks.append({'id':cid,'tokens_equal':tokens_equal,'all_score_hashes_equal':scores_equal,'decisions':len(new['steps'])})
        (RUN/'historical-reproduction.json').write_text(json.dumps(checks,indent=2)+'\n')
        if not all(c['tokens_equal'] and c['all_score_hashes_equal'] for c in checks):
            raise RuntimeError('Historical reproduction differs; investigate before screening or editing')
        status('historical_reproduction_verified',cases=len(checks),decisions=sum(c['decisions'] for c in checks))
        for i in range(4):
            run(f'discovery-{i:02}',RUN/f'discovery-{i:02}-cases.json')
    status('baseline_and_screening_complete',new_weight_edits=0,fresh_evaluation_run=False)


if __name__=='__main__':
    try:
        main()
    except BaseException as e:
        status('stopped',error=str(e))
        raise
