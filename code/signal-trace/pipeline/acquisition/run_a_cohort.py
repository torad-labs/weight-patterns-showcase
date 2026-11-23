"""Acquire the fixed A cohort, qualify each full recording, and archive through torad-research.

This is acquisition only. Every materialized operation is retained. Release of
the two large local files is optional and follows a successful full remote-byte
verification; the manifest, response, settings, audit and restoration paths stay.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

from pipeline.qualification.audit_saved_capture import audit
from pipeline.acquisition.run_a_baseline import ROOT, RUN, OBSERVER, MODEL, BACKENDS


def publish(path, value):
    tmp = path.with_suffix(path.suffix + '.writing')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def note(state, **fields):
    row = dict(at_ct=datetime.datetime.now(ZoneInfo('America/Chicago')).isoformat(),
               state=state, **fields)
    with (RUN/'cohort-progress.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def capture_environment(job):
    gpu = int(job.get('gpu', 0))
    if gpu not in (0, 1):
        raise ValueError('A uses physical GPU0 or GPU1')
    execution_path = job.get('baseline_execution')
    if execution_path:
        execution = json.loads((ROOT/execution_path).read_text())
        command = execution.get('command', [])
        if (execution.get('gpu') != gpu or execution.get('status') != 'complete'
                or len(command) < 3 or Path(command[1]) != MODEL
                or Path(command[2])/job['id'] != ROOT/job['baseline']):
            raise ValueError('Capture must use its own completed same-card baseline')
    elif gpu != 0:
        raise ValueError('GPU1 needs a recorded same-card baseline')
    return dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                GGML_CUDA_DISABLE_FUSION='1', XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',
                DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')


@contextmanager
def reservations():
    with (RUN/'capture-storage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = RUN/'capture-reservations.json'
        state = json.loads(path.read_text()) if path.exists() else {}
        for key, entry in list(state.items()):
            try:
                os.kill(entry['pid'], 0)
            except ProcessLookupError:
                del state[key]
        yield state
        publish(path, state)


def allocated_bytes(output):
    """Space already charged to the filesystem, including only existing files."""
    output = Path(output)
    paths = [output] if output.is_file() else output.rglob('*')
    total = 0
    for path in paths:
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_blocks*512
        except FileNotFoundError:
            # Another reservation can finish and publish a download while read.
            continue
    return total


def reserve_space(key, output, required=20*1024**3):
    """Reserve unwritten bytes so concurrent cards cannot spend them twice."""
    with reservations() as state:
        if key in state:
            raise RuntimeError(f'Capture reservation already in use: {key}')
        pending = 0
        for entry in state.values():
            allocated = allocated_bytes(entry['output']) if entry.get('output') else 0
            written = max(0, allocated-entry.get('initial_allocated_bytes', 0))
            pending += max(0, entry['required_bytes']-written)
        if shutil.disk_usage(RUN).free-pending < required:
            return False
        state[key] = dict(pid=os.getpid(), output=str(output), required_bytes=required,
                          initial_allocated_bytes=allocated_bytes(output))
        return True


def release_space(key):
    with reservations() as state:
        state.pop(key, None)


def acquire(job, release):
    capture, baseline = ROOT/job['capture'], ROOT/job['baseline']
    gpu = int(job.get('gpu', 0))
    env = capture_environment(job)
    out, audit_path = capture.parent, ROOT/job['audit']
    completed = RUN/'cohort-archives'/(job['id']+'.json')
    if completed.exists():
        record = json.loads(completed.read_text())
        if (record['input_sha256'] != job['input_sha256'] or record['verify_exit_code'] != 0
                or record.get('gpu', 0) != gpu):
            raise ValueError('Existing archive belongs to a different or unverified input')
        note('already_archived', case=job['id'])
        return
    source = ROOT/job['input']
    if hashlib.sha256(source.read_bytes()).hexdigest() != job['input_sha256']:
        raise ValueError('Cohort input changed after selection')
    logdir = RUN/'cohort-logs'/job['id']
    logdir.mkdir(parents=True, exist_ok=True)
    if not (capture/'result.json').exists():
        if out.exists():
            raise RuntimeError(f'Partial recording needs inspection; not overwriting {out}')
        note(f'waiting_for_gpu{gpu}', case=job['id'], gpu=gpu)
        waiting_for_space = False
        while True:
            with Path(f'/run/user/{os.getuid()}/gpu{gpu}.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                reservation_key = f'gpu{gpu}'
                if reserve_space(reservation_key, out):
                    try:
                        free = subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free',
                                                       '--format=csv,noheader,nounits'], text=True)
                        if int(free.strip()) < 11000:
                            raise RuntimeError(f'GPU{gpu} is occupied despite lock; leave other processes untouched')
                        hardware = subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                            '--query-gpu=index,name,uuid,pci.bus_id,driver_version',
                            '--format=csv,noheader'], text=True).strip()
                        if job.get('baseline_execution'):
                            original = json.loads((ROOT/job['baseline_execution']).read_text())
                            if original.get('hardware') != hardware:
                                raise ValueError('Physical card differs from the recorded baseline')
                        cmd = [str(OBSERVER),str(MODEL),str(out),str(source),str(BACKENDS),'capture','full','trace']
                        note('capturing', case=job['id'], gpu=gpu, hardware=hardware, command=cmd)
                        with (logdir/'capture.stdout').open('x') as stdout, (logdir/'capture.stderr').open('x') as stderr:
                            subprocess.run(cmd, env=env, stdout=stdout, stderr=stderr, check=True)
                        publish(out/'execution.json', dict(gpu=gpu, hardware=hardware,
                            baseline=job['baseline'], baseline_execution=job.get('baseline_execution'),
                            input_sha256=job['input_sha256'], command=cmd))
                    finally:
                        release_space(reservation_key)
                    break
                if not waiting_for_space:
                    note('waiting_for_disk', case=job['id'], gpu=gpu, free_bytes=shutil.disk_usage(RUN).free,
                         required_bytes=20*1024**3)
                    waiting_for_space = True
            # Archival can release space while this case waits. Leave the GPU
            # unlocked and retry the same unchanged case rather than aborting A.
            time.sleep(10)
    if not (out/'settings.json').exists() or not (out/'weight-bindings.json').exists():
        raise RuntimeError('Observer did not close the complete recording')
    note('qualifying', case=job['id'])
    report = audit(capture, baseline)
    publish(audit_path, report)
    files = sorted(p for p in out.rglob('*') if p.is_file())
    large = [capture/'signal-data.bin', capture/'signal-events.jsonl']
    if not all(p in files and not p.is_symlink() for p in large):
        raise RuntimeError('Full native data and event files must exist before archiving')
    entries, states = [], {}
    for p in files:
        before = p.stat()
        with p.open('rb') as f:
            digest = hashlib.file_digest(f, 'sha256').hexdigest()
        states[p] = before
        entries.append(dict(path=str(p.relative_to(ROOT)), bytes=before.st_size, sha256=digest,
                            object_key='traces/'+p.relative_to(ROOT/'runs').as_posix()))
    args = [str(p) for p in files]
    for action in ['upload', 'verify']:
        note('archiving_'+action, case=job['id'], bytes=sum(e['bytes'] for e in entries))
        with (logdir/(action+'.log')).open('a') as log:
            subprocess.run(['torad-research','traces',action,*args], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    # No analysis reads the active acquisition directory. A researcher can restore
    # either large file by its recorded path; never release a changed local file.
    for p in large:
        now, old = p.stat(), states[p]
        if (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns) != (
                old.st_dev,old.st_ino,old.st_size,old.st_mtime_ns,old.st_ctime_ns):
            raise RuntimeError('Local capture changed during archiving; retain it')
    record = dict(case=job['id'], input_sha256=job['input_sha256'], files=entries, gpu=gpu,
                  baseline=job['baseline'], baseline_execution=job.get('baseline_execution'),
                  verify_exit_code=0, archived_by='torad-research', bucket='torad-research',
                  released_local=[], logs=str(logdir.relative_to(ROOT)),
                  restore_command='torad-research traces download '+out.relative_to(ROOT/'runs').as_posix())
    completed.parent.mkdir(exist_ok=True)
    publish(completed, record)
    if release:
        for p in large:
            p.unlink()
            record['released_local'].append(str(p.relative_to(ROOT)))
            publish(completed, record)
    note('capture_archived', case=job['id'], decisions=report['decisions'],
         events=report['events'], local_large_files_released=len(record['released_local']))


def run_jobs(jobs, release, workers=1):
    """Keep independent capture/archive cases progressing under the GPU lock."""
    if workers < 1:
        raise ValueError('At least one case worker is required')
    if len({job['id'] for job in jobs}) != len(jobs):
        raise ValueError('A job list may not capture the same case twice')
    if workers == 1:
        for job in jobs:
            acquire(job, release)
        return
    remaining = iter(jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for _ in range(min(workers, len(jobs))):
            pending.add(pool.submit(acquire, next(remaining), release))
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Inspect every finished case before starting another. On an error,
            # already active cases finish preserving their data; no new ones start.
            for future in done:
                future.result()
            for _ in done:
                job = next(remaining, None)
                if job is not None:
                    pending.add(pool.submit(acquire, job, release))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs', type=Path)
    parser.add_argument('--release-verified', action='store_true')
    parser.add_argument('--workers', type=int, default=1,
                        help='Independent cases; one capture per assigned GPU while archives overlap')
    a = parser.parse_args()
    try:
        jobs = json.loads(a.jobs.read_text())
        run_jobs(jobs, a.release_verified, a.workers)
        note('cohort_acquisition_complete', jobs=len(jobs), new_weight_edits=0)
    except BaseException as e:
        note('cohort_acquisition_stopped', error=str(e))
        raise
