"""Continue at a completed native update, preserving the original experiment.

The execution owner first stops the source controller at its next acquisition
boundary and binds a handoff record. This module never signals that controller.
It copies the last complete checkpoint into the new run, then requires a fresh
native observation at the next global step. No inverse vectors are recovered.
"""
from pathlib import Path
import os
import shutil

import torch

from pipeline.research_guard.verify_round2 import ROOT

from . import functional_install, gap_state, state_tensors, vector_loop
from .functional_reading import digest
from .functional_reference import checked
from .vector_chance_run import binding, now, save
from .vector_recovery import UNCHANGED


def _inside(root, source, expected):
    path = Path(root) / source['record']
    if path.resolve() != expected.resolve():
        raise ValueError('the handoff record belongs to another run or step')
    # Source checkpoint paths must be ordinary files, including their parents.
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('continuation checkpoints cannot use symlinks')
    return checked(root, source)


def _copy(source, target, expected):
    pending = target.with_name(target.name + '.copying')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or pending.exists():
        raise ValueError('a continuation checkpoint is copied once, never overwritten')
    with source.open('rb') as incoming, pending.open('xb') as outgoing:
        offset = 0
        while chunk := incoming.read(gap_state.BLOCK):
            outgoing.write(chunk)
            os.posix_fadvise(incoming.fileno(), offset, len(chunk), os.POSIX_FADV_DONTNEED)
            offset += len(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
        os.posix_fadvise(outgoing.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if digest(pending) != expected:
        raise ValueError('the copied checkpoint differs from its source binding')
    os.replace(pending, target)


def _source(root, run, declaration):
    continuation = declaration['continuation']
    source = checked(root, continuation['declaration'])
    source_run = (root / continuation['declaration']['record']).resolve().parent
    if not source_run.is_relative_to(root):
        raise ValueError('the source run must remain inside its declared root')
    if source_run.is_relative_to(run) or run.is_relative_to(source_run):
        raise ValueError('source and continuation run directories must not overlap')
    for key in UNCHANGED:
        if declaration.get(key) != source.get(key):
            raise ValueError(f'continuation changes the original experiment: {key}')
    review = source.get('whole_build_review', {})
    if review.get('accepted') is not True:
        raise ValueError('the source experiment has no accepted whole-build review')
    checked(root, review['source'], parse=False)
    current = {item['record']: item['sha256'] for item in declaration['implementation']}
    if len(current) != len(declaration['implementation']):
        raise ValueError('implementation bindings must not be duplicated')
    if any(current.get(item['record']) != item['sha256'] for item in source['implementation']):
        raise ValueError('continuation must retain every original implementation binding')
    handoff = _inside(root, continuation['handoff'], source_run / 'handoff.json')
    completed = handoff.get('completed_updates')
    if (handoff.get('format') != 'vector-loop-handoff/1'
            or handoff.get('declaration') != continuation['declaration']
            or handoff.get('controller_stopped') is not True
            or type(completed) is not int or completed < 1):
        raise ValueError('continuation requires a stopped controller and completed update count')
    step = source_run / 'loop' / str(completed)
    next_step = source_run / 'loop' / str(completed + 1)
    if any((source_run / name).exists() for name in ('result.json', 'closure.json')):
        raise ValueError('a completed source result is not an acquisition-boundary handoff')
    if any((next_step / name).exists() for name in ('measurement.json', 'proposal.json', 'reached.json')):
        raise ValueError('the source has advanced beyond the declared acquisition boundary')
    if any(path.name.isdigit() and int(path.name) > completed + 1
           for path in step.parent.iterdir()):
        raise ValueError('the source has later steps than its handoff count')
    reached = _inside(root, handoff['reached'], step / 'reached.json')
    compact_binding = _inside(root, handoff['compact'], step / 'compact.json')
    capture = _inside(root, handoff['capture_plan'], next_step / 'capture-plan.json')
    file = step / 'model.gguf'
    if any(part.is_symlink() for part in (file, *file.parents)):
        raise ValueError('continuation checkpoints cannot use symlinks')
    if (reached.get('format') != 'functional-reached-state/1'
            or reached.get('step') != completed
            or (root / reached['file']).resolve() != file.resolve()):
        raise ValueError('the reached checkpoint does not bind the completed global step')
    expected_plan = vector_loop.capture_commands(source_run, source, completed + 1, file, root=root)
    if 'vector_solver' in source:
        graph = torch.device(source['device'])
        placement = source['vector_solver']
        visible = placement.get('visible_devices', [])
        if (placement.get('format') != 'joint-inverse-device/1' or graph.type != 'cuda'
                or graph.index is None or graph.index >= len(visible)):
            raise ValueError('the source device mapping does not identify the graph GPU')
        expected_plan.update(physical_graph_uuid=visible[graph.index], graph_card=source['card'])
    if capture != expected_plan:
        raise ValueError('the next acquisition plan differs from the saved checkpoint or protocol')
    compact = _inside(root, compact_binding, step / 'state-tensors/record.json')
    joint_file = functional_install.manifest(file)
    state_file = gap_state.Working.beside(file)
    tensors_file = step / 'state-tensors/tensors.bin'
    files = [(file, reached['model_sha256']), (joint_file, reached['joint_sha256']),
             (state_file, digest(state_file)),
             (step / 'state-tensors/record.json', compact_binding['sha256']),
             (tensors_file, compact['tensors_sha256'])]
    for path, expected in files:
        if any(part.is_symlink() for part in (path, *path.parents)) or digest(path) != expected:
            raise ValueError('the source checkpoint differs from its binding or uses a symlink')
    if (compact.get('format') != state_tensors.FORMAT
            or compact.get('model_sha256') != reached['model_sha256']
            or compact.get('original_sha256') != declaration['original_model']['sha256']):
        raise ValueError('the compact checkpoint belongs to another original or reached model')
    return source_run, handoff, reached, compact, files


class Backend(vector_loop.Backend):
    def __init__(self, run, declaration, *, root=ROOT, acquire=None):
        root, run = Path(root).resolve(), Path(run).resolve()
        if not run.is_relative_to(root):
            raise ValueError('the continuation must remain inside its declared root')
        if (run / 'loop').exists() or (run / 'continuation.json').exists():
            raise ValueError('a continuation is staged once, never silently resumed')
        source_run, handoff, reached, compact, files = _source(root, run, declaration)
        self.completed_updates = handoff['completed_updates']
        super().__init__(run, declaration, root=root, acquire=acquire)
        source_step = source_run / 'loop' / str(self.completed_updates)
        source_model = source_step / 'model.gguf'
        installed = functional_install.load(source_model, reached['model_sha256'], self.names)
        working = gap_state.Working.taken_up(source_model, self.original, self.tensors)
        kept = working.kept()
        if (kept != installed['kept'] or compact['changed'] != kept['changed']
                or compact['identity'] != kept['identity']
                or state_tensors.made_again(source_step / 'state-tensors', compact,
                                           self.original, self.tensors) != reached['model_sha256']):
            raise ValueError('the full and compact source checkpoints do not identify the same state')
        # This is the existing operational disk floor, not a research parameter.
        if shutil.disk_usage(run).free - sum(path.stat().st_size for path, _ in files) < 40 * 10**9:
            raise OSError('insufficient disk for the continuation above the project 40 GB floor')
        destination = run / 'loop' / str(self.completed_updates)
        for source, expected in files:
            _copy(source, destination / source.relative_to(source_step), expected)
        model = destination / 'model.gguf'
        functional_install.load(model, reached['model_sha256'], self.names)
        local = gap_state.Working.taken_up(model, self.original, self.tensors).kept()
        if local != kept:
            raise ValueError('the staged checkpoint no longer matches its source state')
        # Recheck small records and source bytes before publishing the new pointer.
        if _source(root, run, declaration)[-1] != files:
            raise ValueError('the source checkpoint changed while it was being staged')
        save(destination / 'compact.json', binding(destination / 'state-tensors/record.json', root))
        save(destination / 'reached.json', reached | dict(file=str(model.relative_to(root)),
             continued_from=declaration['continuation']['handoff']))
        save(run / 'continuation.json', dict(format='vector-loop-continuation/1', at=now(),
             source=declaration['continuation'], completed_updates=self.completed_updates,
             next_step=self.completed_updates + 1,
             reached=binding(destination / 'reached.json', root),
             source_checkpoint_read_only=True, inverse_vectors_recovered=False))

    def observe(self, number):
        if number <= self.completed_updates:
            raise ValueError('continuation must acquire after its completed global updates')
        return super().observe(number)


def iterate_from(backend, completed_updates):
    """The ordinary pair loop with an explicit, already-completed update count."""
    if type(completed_updates) is not int or completed_updates < 1:
        raise ValueError('continuation requires a positive completed update count')
    if backend.completed_updates != completed_updates:
        raise ValueError('the controller count differs from the staged checkpoint')
    updates = completed_updates
    while True:
        observation = backend.observe(updates + 1)
        if observation['complete'] is not True or observation['ended'] is not True:
            result = dict(status='incomplete_acquisition', closure=False, updates=updates)
            break
        if observation['outside'] == 0:
            result = dict(status='closed', closure=True, updates=updates,
                          post_closure=backend.post_closure(), repair_claim=False)
            break
        correction = backend.correct(updates + 1, observation)
        if correction['status'] != 'accepted':
            result = dict(status=correction['status'], closure=False, updates=updates,
                          correction=correction)
            break
        updates += 1
    result.update(continued_from_updates=completed_updates, new_updates=updates - completed_updates)
    result.setdefault('repair_claim', False)
    backend.finish(result)
    return result
