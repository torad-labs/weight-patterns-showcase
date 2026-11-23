"""Run declared A observations with unchanged native inference and an explicit role."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from zoneinfo import ZoneInfo

from pipeline.native_records.group_observation import publish
from pipeline.native_records.run_declaration import digest
from pipeline.native_records.run_declaration import check_native_run
from pipeline.acquisition.run_a_baseline import ROOT, BACKENDS


def execution_record(declaration, **observed):
    """Keep declared provenance while recording the actual native execution."""
    return {**declaration, **observed}


RECORDERS = ROOT/'code/signal-trace/recorder'
# The stream recorders installed side by side. observe-stream is the one every pass takes unless a caller names another; observe-stream-late
# is the same program with the late start added (STREAM-RECORDER.md), for a declared run that names it and gives its cases the late_start field.
STREAM_RECORDERS = ('observe-stream', 'observe-stream-late')


def command_of(model, inputs, output, mode, retention, program=None):
    """The recorder a pass runs and its command line. `program` names another installed stream recorder than observe-stream, for stream
    retention only; without it every retention takes the program it always took."""
    if program is not None and (retention in ('selected', 'none') or program not in STREAM_RECORDERS):
        raise ValueError(f'{program!r}: another recorder is named for stream retention only, and it is one of {STREAM_RECORDERS}')
    if retention=='selected':
        observer = RECORDERS/'observe-selected'
        return observer, [str(observer), str(model), str(output), str(inputs), str(BACKENDS), mode, 'full', 'trace']
    if retention=='none':
        if mode!='plain':raise ValueError('No retention means plain inference')
        observer = RECORDERS/'observe-stream'
        return observer, [str(observer), str(model), str(output), str(inputs), str(BACKENDS), 'plain']
    if mode!='capture':raise ValueError('Stream retention is an explicit capture mode')
    observer = RECORDERS/(program or 'observe-stream')
    return observer, [str(observer), str(model), str(output), str(inputs), str(BACKENDS), mode, retention, 'stream']


def beside(output, ending):
    """The recorder's own stdout and stderr files of a pass: the pass's whole name with the ending after it. `with_suffix` cut the name at its
    last dot, so two passes named after a tensor (`...-blk.0.attn_output.weight-<state>`) asked for one file and the second was refused; a
    name without a dot gets the file it always got."""
    return output.with_name(output.name + ending)


def read_through(model, told=None, block=1 << 24):
    """A file read once from its first byte to its last, nothing kept, so its pages are in memory when the recorder maps it. Returns the
    bytes read and says them with the seconds in `told`. Why: on November 14, 2026 the working model of a sweep (9.31 GB) lost its pages
    between passes, and a late pass's recorder took 25 to 122 s where it takes 12.4 s with the model in memory (16 of 31 passes that
    started under 0.95 in memory, none of 19 that started whole: speed review, "Both sweeps at 67 of 70"). A recorder that maps a file
    whose pages are gone brings them back fault by fault; one read from end to end brings them back at the disk's own rate. The block
    is a buffer's size and no setting of the result: the same bytes are read at any size. A file that cannot be read is left to the
    recorder to report."""
    began, done = time.monotonic(), 0
    try:
        with open(model, 'rb', buffering=0) as source:
            held = bytearray(block)
            while True:
                count = source.readinto(held)
                if not count:
                    break
                done += count
    except OSError:
        pass
    if told is not None:
        told.update(bytes=done, seconds=round(time.monotonic() - began, 3))
    return done


def run(model, inputs, output, gpu, kind, mode, retention='selected', declaration_path=None, program=None):
    before, beside_it = {}, {}
    early = threading.Thread(target=read_through, args=(model, before), daemon=True)    # beside the verifier's scan, which takes longer
    early.start()
    declaration=check_native_run(output,declaration_path)
    cases = json.loads(inputs.read_text())
    if kind == 'natural' and any(any(key.startswith('teacher_') for key in c) for c in cases):
        raise ValueError('Natural behavior cannot use supplied response tokens')
    if program is None and any('late_start' in c for c in cases):
        raise ValueError('A case asks for a late start and the pass does not name the recorder that has one (program=\'observe-stream-late\'): observe-stream would read it whole and say nothing')
    if output.exists():
        raise ValueError('Existing observation must be inspected, not overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    observer, command = command_of(model, inputs, output, mode, retention, program)
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu), GGML_CUDA_DISABLE_FUSION='1')
    while True:
        with Path(f'/run/user/{os.getuid()}/gpu{gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            free = int(subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],text=True).strip())
            if free >= 11000:
                start=datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat()
                print(json.dumps(dict(started_ct=start, output=str(output), gpu=gpu, cases=len(cases), kind=kind)),flush=True)
                identity=model.stat()
                hardware=subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=index,name,uuid,pci.bus_id,driver_version','--format=csv,noheader'],text=True).strip()
                early.join()
                again = threading.Thread(target=read_through, args=(model, beside_it), daemon=True)   # the wait for the card may have cost the pages again
                again.start()
                with beside(output, '.stdout.jsonl').open('x') as stdout, beside(output, '.stderr.log').open('x') as stderr:
                    subprocess.run(command,env=env,stdout=stdout,stderr=stderr,check=True)
                again.join()
                after=model.stat()
                if (after.st_ino,after.st_size,after.st_mtime_ns)!=(identity.st_ino,identity.st_size,identity.st_mtime_ns):
                    raise ValueError('Checkpoint changed during native observation')
                for c in cases:
                    result=json.loads((output/c['id']/'result.json').read_text())
                    if kind=='natural' and (result['teacher_token_count'] or result['conditioned_continuation']):
                        raise ValueError('Unexpected conditioned response in natural observation')
                publish(output/'execution.json',execution_record(declaration,status='complete',kind=kind,gpu=gpu,hardware=hardware,command=command,
                    retention=retention,scope=('No internal capture' if mode=='plain' else
                        'Full selected-position records' if retention=='selected' else
                        'Reduced-retention residual/projection stream at every decision; unchanged full records only at declared positions'),
                    started_ct=start,completed_ct=datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(),
                    checkpoint=str(model),checkpoint_bytes=identity.st_size,checkpoint_mtime_ns=identity.st_mtime_ns,
                    inputs_sha256=digest(inputs),observer_sha256=digest(observer),cases=[c['id'] for c in cases],
                    model_read_through=dict(before_the_recorder=before, beside_the_recorder=beside_it),
                    environment={k:env[k] for k in ['CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','GGML_CUDA_DISABLE_FUSION']}))
                print(json.dumps(dict(complete=str(output),gpu=gpu,kind=kind)),flush=True)
                return
        time.sleep(15)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ['model','inputs','output']:
        p.add_argument(field,type=Path)
    p.add_argument('--gpu',type=int,choices=[0,1],required=True)
    p.add_argument('--kind',choices=['conditional','natural'],required=True)
    p.add_argument('--mode',choices=['plain','capture'],default='plain')
    p.add_argument('--retention',choices=['selected','residual','projection','none'],default='selected')
    p.add_argument('--declaration',type=Path)
    p.add_argument('--program',choices=STREAM_RECORDERS,help='another installed stream recorder than observe-stream; stream retention only')
    a=p.parse_args()
    run(a.model.resolve(),a.inputs.resolve(),a.output.resolve(),a.gpu,a.kind,a.mode,a.retention,a.declaration,a.program)
