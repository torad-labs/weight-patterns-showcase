"""Successor publication boundary around the inherited guarded archive workflow.

Only the publication checker is replaced while this synchronous call executes.
Upload/full-byte verification/active-input release remain the existing operators.
No old source or receipt is edited. No MLflow network call belongs to the learner.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading

ROOT=Path(__file__).resolve().parents[3]
FILES=('loss-maps.json','computation-overlap.png','local-turn-overlay.png','overlay-selection.json')
FORMAT='local-loss-diagnostics-publication/1'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


legacy=load('final_experiment_consumed_archive',ROOT/'studies/04-round-2-overlap/runs/vector-chance-loss-rise-diagnosis/empirical-eta-continuation/consumed_trace_archive.py')
# This separately loaded operator carries the already-qualified worker ContextVars.
legacy.archive=load('final_experiment_archive_operator',ROOT/'records/development/2026-11-17-capture-context/archive_inputs.py')
_lock=threading.RLock()


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def publication_key(record):
    payload={k:record[k] for k in ('format','step','measurement_sha256','loss_identity','artifacts','parent_run','step_run')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def published(root,run,number,measured,state_path):
    try:
        state=json.loads(Path(state_path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None
    observed=state.get(run.name,{})
    view=observed.get('training_view',{})
    receipt=view.get('local_loss_diagnostics',{}).get(str(number),{})
    if not receipt.get('published'):
        return None
    step=run/'loop'/str(number);directory=step/'diagnostics'
    if (receipt.get('format')!=FORMAT or receipt.get('step')!=str(number)
            or receipt.get('measurement_sha256')!=digest(step/'measurement.json')
            or receipt.get('loss_identity')!=measured.get('loss_identity')
            or receipt.get('parent_run')!=observed.get('run')
            or receipt.get('step_run')!=view.get('steps',{}).get(str(number),{}).get('run')
            or set(receipt.get('artifacts',{}))!=set(FILES)
            or receipt['published']!=publication_key(receipt)):
        raise ValueError('local diagnostic publication does not bind this exact measurement/loss/run')
    if directory.is_symlink():
        raise ValueError('diagnostic publication must remain local regular files')
    for name in FILES:
        path=directory/name;binding=receipt['artifacts'][name]
        if (not path.is_file() or path.is_symlink() or path.resolve()!=path.absolute()
                or path.stat().st_size!=binding['bytes'] or digest(path)!=binding['sha256']):
            raise ValueError('published local diagnostic changed or is absent: '+name)
    maps=json.loads((directory/'loss-maps.json').read_text())
    if (maps.get('format')!='complete-local-computational-loss/1'
            or maps.get('identity')!=measured.get('loss_identity')
            or any(maps.get('summary',{}).get(k)!=measured.get(k) for k in ('objective','outside','pair_merge_cells'))
            or maps.get('summary',{}).get('canonical_matches_declared') is not True
            or maps.get('observation',{}).get('model_revision')!=measured.get('model_identity')
            or maps.get('observation',{}).get('complete') is not True
            or measured.get('complete') is not True or measured.get('ended') is not True):
        raise ValueError('published maps do not reproduce the exact complete fresh native measurement')
    return receipt


def archive_previous(run,previous_step,current_step,declaration,*,stop_requested=None):
    with _lock:
        original=legacy.published
        legacy.published=published
        try:
            return legacy.archive_previous(run,previous_step,current_step,declaration,stop_requested=stop_requested)
        finally:
            legacy.published=original


def source_files():
    return sorted({Path(__file__).resolve(),Path(legacy.__file__).resolve(),
                   Path(legacy.archive.__file__).resolve(),*legacy.source_files()})
