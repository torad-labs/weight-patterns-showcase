"""One native execution, joint BF16 correction, then a fresh native execution.

Only the engine's fresh complete trace can close this loop. The anchored Torch
graph proposes a joint update on held history; its trial value is not a native
measurement. Post-closure acquisition is a separately declared operation.
"""
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.research_guard.verify_round2 import ROOT

from . import functional_install, gap_state, native_graph, pair_loop, state_tensors, turn_artifact, turn_correction
from .functional_reading import Reading, digest
from .functional_reference import checked
from .vector_chance_run import binding, now, save

READERS = Path('records/development/2026-11-12-path-forward')


def capture_commands(run, declaration, number, model, *, root=ROOT) -> dict[str, Any]:
    run, root, model = Path(run).resolve(), Path(root).resolve(), Path(model).resolve()
    spec = declaration['functional_loop']
    step = run / 'loop' / str(number)
    capture, raw = step / 'capture', step / 'raw'
    cohort = Path(spec['cohort']).name
    env = dict(PASS=str(number), KEEP_RAW=str(raw), ROLLOUTS=str(raw),
               CASES=str(root / spec['cases']['record']),
               PYTHONPATH=str(root / 'code/signal-trace'), PYTHONDONTWRITEBYTECODE='1')
    return dict(env=env, commands=[
        [sys.executable, str(root / READERS / 'native_events.py'), str(run), cohort,
         str(model), spec['history_before'], str(capture / 'own')],
        [sys.executable, str(root / READERS / 'states_reading_roles.py'), str(run), cohort,
         str(model), str(capture)],
        [sys.executable, str(root / READERS / 'native_events_protocol.py'), 'held',
         str(root / spec['reader_protocol']['record']), str(capture), str(step / 'reading')],
    ])


@contextmanager
def card_lock(device, card):
    if torch.device(device).type != 'cuda':
        yield
        return
    with Path(f'/run/user/{os.getuid()}/gpu{card}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def compact_state(file, original, tensors, kept, previous, previous_kept, *, root=ROOT):
    file, original, previous = Path(file), Path(original), Path(previous)
    if previous.resolve() in (original.resolve(), file.resolve()):
        raise ValueError('only a superseded working copy can be released')
    folder = file.parent / 'state-tensors'
    state_tensors.keep(file, original, tensors, folder, kept=kept)
    if previous_kept['changed']:
        previous_folder = previous.parent / 'state-tensors'
        pointer = previous.parent / 'compact.json'
        if not pointer.is_file():
            raise ValueError('the superseded state has no bound compact reconstruction')
        source = json.loads(pointer.read_text())
        if (Path(root) / source['record']).resolve() != (previous_folder / 'record.json').resolve():
            raise ValueError('the prior compact binding belongs to another state')
        record = checked(Path(root), source)
        if (record['model_sha256'] != previous_kept['model_sha256']
                or digest(previous_folder / 'tensors.bin') != record['tensors_sha256']
                or state_tensors.made_again(previous_folder, record, original, tensors)
                   != previous_kept['model_sha256']):
            raise ValueError('the superseded state has no intact compact reconstruction')
    elif previous_kept['model_sha256'] != digest(original):
        raise ValueError('the unchanged superseded copy must be the original bytes')
    if digest(previous) != previous_kept['model_sha256']:
        raise ValueError('the superseded model changed before release')
    previous.unlink()
    save(file.parent / 'released-copy.json', dict(at=now(), file=str(previous.relative_to(root)),
        model_sha256=previous_kept['model_sha256'], compact_or_original='verified before release'))
    return binding(folder / 'record.json', root)


def case_protocol(root, spec, members, cohort):
    cases = checked(root, spec['cases'])
    protocol = checked(root, spec['reader_protocol'])
    if cases.get('cohort') != cohort or protocol.get('cohort') != cohort:
        raise ValueError('case manifest, protocol and declaration must name the same acquisition cohort')
    if (cases.get('format') != 'whole-gap-cases/1' or cases.get('role') != 'fitting-named'
            or cases.get('ids') != members or protocol.get('format') != 'whole-gap-reader-protocol/4'
            or protocol.get('acquisition') != 'native-events-only'):
        raise ValueError('acquisition requires its named fitting manifest and native-only protocol')
    if (protocol.get('cases', {}).get('role') != 'fitting-named'
            or protocol['cases'].get('manifest_sha256') != spec['cases']['sha256']):
        raise ValueError('the acquisition protocol must bind the same fitting manifest')
    for path, sha in protocol['instruments'].items():
        checked(root, dict(record=path, sha256=sha), parse=False)


class Backend:
    def __init__(self, run, declaration, *, root=ROOT, acquire=None):
        self.run, self.root = Path(run).resolve(), Path(root).resolve()
        self.declaration = declaration
        if not self.run.is_relative_to(self.root):
            raise ValueError('the native run must remain inside its declared root')
        if declaration.get('operation') != 'single-vector-chance-loop':
            raise ValueError('the native backend requires the single vector chance loop')
        for source in declaration['implementation']:
            checked(self.root, source, parse=False)
        self.spec = declaration['functional_loop']
        if self.spec.get('acquisition') != 'native-events-only':
            raise ValueError('the loop must acquire native events without registration')
        self.definition = checked(self.root, declaration['source_definition'])
        self.original_binding = declaration['original_model']
        if self.original_binding != self.definition['sources']['model']:
            raise ValueError('the original model must be the reference population original')
        checked(self.root, self.original_binding, parse=False)
        self.original = self.root / self.original_binding['record']
        self.architecture = checked(self.root, declaration['architecture'])
        self.tensors = checked(self.root, declaration['tensors'])
        self.names = tuple(name for name in self.tensors
                           if name.endswith(('.attn_output.weight', '.ffn_down.weight')))
        if (not self.names or self.spec['matrices'] != list(self.names)
                or len(self.names) != 2 * self.architecture['layers']):
            raise ValueError('every editable matrix in the model inventory must be declared')
        for key in ('architecture', 'tensors'):
            if declaration[key] != self.definition['sources'][key]:
                raise ValueError('the model layout must be the original layout')
        members = self.spec['training_members']
        if (len(members) != 1 or members[0] not in self.definition['members']
                or self.definition['members'][members[0]]['group'] != 'wrong'):
            raise ValueError('training uses exactly one originally wrong fitting prompt')
        self.member = members[0]
        case_protocol(self.root, self.spec, members, Path(self.spec['cohort']).name)
        post = declaration['post_closure']
        if not all(key in post for key in ('members', 'cases', 'reader_protocol')):
            raise ValueError('post-closure cases and reader protocol must be fixed before training')
        if (len(post['members']) != len(set(post['members']))
                or set(post['members']) != set(self.definition['members']) - {self.member}):
            raise ValueError('post-closure membership must be every other fitting prompt')
        case_protocol(self.root, post, post['members'], Path(self.spec['cohort']).name)
        record = checked(self.root, declaration['reference'])
        if record.get('format') != 'single-original-vector-chance/1':
            raise ValueError('the loop requires the original vector chance artifact')
        if record['source_definition'] != declaration['source_definition']:
            raise ValueError('reference and current acquisition must share original provenance')
        self.reference = turn_artifact.load(self.root, declaration['reference'],
            original_model_sha256=self.original_binding['sha256'],
            expected_merges=3 * self.architecture['layers'])
        checked(self.root, declaration['forward'], parse=False)
        source = importlib.util.spec_from_file_location('bound_vector_loop_forward',
                                                       self.root / declaration['forward']['record'])
        if source is None or source.loader is None:
            raise ValueError('the hash-bound model forward is unavailable')
        module = importlib.util.module_from_spec(source)
        source.loader.exec_module(module)
        self.Model = module.Model
        self.device = declaration['device']
        self.card = declaration.get('card', 0)
        self.acquire = acquire if acquire is not None else self._acquire
        self.last = None
        self.case = None
        self.pairs = None
        self.model_file = self.original
        self.model_sha256 = self.original_binding['sha256']
        self._state(1)

    def _state(self, number):
        if number == 1:
            source = self.spec['first_state']
            if (source['file'] != self.original_binding['record']
                    or source['model_sha256'] != self.original_binding['sha256']):
                raise ValueError('the one-prompt experiment starts from the original model')
        else:
            source = json.loads((self.run / 'loop' / str(number - 1) / 'reached.json').read_text())
            if source.get('format') != 'functional-reached-state/1' or source.get('step') != number - 1:
                raise ValueError('the preceding reached state belongs to another step')
            file = (self.root / source['file']).resolve()
            if not file.is_relative_to(self.run):
                raise ValueError('a reached model must remain inside its own run')
            if digest(functional_install.manifest(file)) != source['joint_sha256']:
                raise ValueError('the reached joint manifest changed')
            functional_install.load(file, source['model_sha256'], self.names)
        file = self.root / source['file']
        if digest(file) != source['model_sha256']:
            raise ValueError('the model differs from its declared reached state')
        self.model_file, self.model_sha256 = file, source['model_sha256']

    def _acquire(self, number, model):
        plan = capture_commands(self.run, self.declaration, number, model, root=self.root)
        env = os.environ | plan['env'] | dict(
            CUDA_VISIBLE_DEVICES=str(self.card),
            RECORDER_UNDER=f'flock /run/user/{os.getuid()}/gpu{self.card}.lock')
        step = self.run / 'loop' / str(number)
        save(step / 'capture-plan.json', plan)
        for index, command in enumerate(plan['commands']):
            with (step / f'acquisition-{index}.log').open('w') as log:
                subprocess.run(command, cwd=self.root / 'code/signal-trace', env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        return step / 'reading'

    def _model(self, device, *, editable):
        return self.Model(self.model_file, self.tensors, self.architecture, device=device,
                          cache='float16', leaves=lambda name: editable and name in self.names)

    def observe(self, number):
        step = self.run / 'loop' / str(number)
        if step.exists():
            raise ValueError('a native observation is never overwritten or silently resumed')
        self._state(number)
        step.mkdir(parents=True)
        folder = Path(self.acquire(number, self.model_file)).resolve()
        if folder != step / 'reading':
            raise ValueError('the fresh reading must belong to this numbered acquisition')
        model = self._model('cpu', editable=False)
        reader = Reading(folder, model, {self.member: self.definition['prompts'][self.member]},
                         expected_reading_sha256=digest(folder / 'reading.json'),
                         expected_model_sha256=self.model_sha256)
        with torch.no_grad():
            case = reader.read(self.member)
            pairs = turn_correction.correspondence(self.reference, case.native)
            measured = turn_correction.measurement(self.reference, case.native, pairs)
        np.save(step / 'pairs.npy', pairs, allow_pickle=False)
        measured.update(member=self.member, complete=case.complete is True, ended=case.ended,
                        step=number, measured_at=now(), events=len(case.history),
                        merges=self.reference.merges, model_file=str(self.model_file.relative_to(self.root)),
                        model_sha256=self.model_sha256,
                        reference_sha256=self.declaration['reference']['sha256'],
                        reading=binding(folder / 'reading.json', self.root),
                        correspondence=binding(step / 'pairs.npy', self.root),
                        measure='fresh native input-output turns; no output-token judge')
        save(step / 'measurement.json', measured)
        self.last, self.case, self.pairs = measured, case, pairs
        return measured

    def correct(self, number, observation):
        if observation is not self.last or self.case is None or self.pairs is None or observation['step'] != number:
            raise ValueError('a correction needs its immediately preceding native observation')
        if not observation['complete'] or not observation['ended'] or observation['outside'] == 0:
            raise ValueError('only a complete outside native trace supplies a correction')
        self._state(number)
        step = self.run / 'loop' / str(number)
        with card_lock(self.device, self.card):
            model = self._model(self.device, editable=True)
            if tuple(model.leaves) != self.names:
                raise ValueError('the joint graph must expose every declared matrix')
            graph = native_graph.Graph(model, self.case.history, self.case.native)
            def progress(iteration, objective):
                save(step / 'inverse-progress.json', dict(at=now(), iteration=iteration,
                                                        linear_objective=objective))
            proposal = turn_correction.propose(graph, self.case.history, self.reference,
                member=self.member, pairs=self.pairs, progress=progress)
            record = dict(status=proposal.native.status, before=proposal.before,
                          value=asdict(proposal.native.value), length=proposal.native.length,
                          trials=proposal.native.trials,
                          value_is='held-history native-anchored Torch surrogate, not fresh native measurement',
                          closure=False, at=now())
            save(step / 'proposal.json', record)
            weights = proposal.native.weights
            del graph, model, proposal
            gc.collect()
            if torch.device(self.device).type == 'cuda':
                torch.cuda.empty_cache()
        if record['status'] != 'accepted':
            return record
        if weights is None or set(weights) != set(self.names):
            raise ValueError('an accepted proposal must supply every native matrix')
        # The floor is the existing project storage rule, not a research parameter.
        required = self.original.stat().st_size * (2 if number == 1 else 1)
        if self.declaration.get('compact_states'):
            required += sum(self.tensors[name]['bytes'] for name in self.names)
        if shutil.disk_usage(self.run).free - required < 40 * 10**9:
            raise OSError('insufficient disk for complete model copies above the project 40 GB floor')
        if number == 1:
            start = gap_state.Working.make(self.run / 'seed.gguf', self.original,
                                          self.original_binding['sha256'], self.tensors)
        else:
            start = gap_state.Working.taken_up(self.model_file, self.original, self.tensors)
        output = step / 'model.gguf'
        installed = functional_install.install(start, self.model_sha256, self.names, weights, output)
        functional_install.load(output, installed['model_sha256'], self.names)
        reached = dict(format='functional-reached-state/1', step=number,
                       file=str(output.relative_to(self.root)), model_sha256=installed['model_sha256'],
                       joint_sha256=digest(functional_install.manifest(output)))
        save(step / 'reached.json', reached)
        if self.declaration.get('compact_states'):
            compact = compact_state(output, self.original, self.tensors, installed['kept'],
                                    start.file, start.kept(), root=self.root)
            save(step / 'compact.json', compact)
        self.case = None
        return record | dict(reached=reached)

    def post_closure(self):
        if self.last is None or self.last['step'] < 2 or not self.last['complete'] or not self.last['ended'] or self.last['outside']:
            raise ValueError('post-closure work requires a fresh complete zero-excess trace after an update')
        closure = dict(closure=True, updates=self.last['step'] - 1,
                       model_file=self.last['model_file'], model_sha256=self.last['model_sha256'],
                       reference_sha256=self.last['reference_sha256'],
                       measurement=binding(self.run / 'loop' / str(self.last['step']) / 'measurement.json', self.root),
                       protocol=self.declaration['post_closure'], recorded_at=now(),
                       repair_claim=False)
        save(self.run / 'closure.json', closure)
        return dict(status='awaiting_separately_declared_post_closure_acquisition',
                    closure=binding(self.run / 'closure.json', self.root), generated_answers=0)

    def finish(self, result):
        save(self.run / 'result.json', result | dict(completed_at=now(),
             prediction_vs_result=result['status'],
             failures_only='Closure is only this computational rule; no capability or independent transfer claim.',
             what_this_changes='Every accepted update was installed completely before fresh native remeasurement.'))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    run = Path(parser.parse_args().run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('all outputs belong inside the declared run')
    if (run / 'loop').exists() or (run / 'result.json').exists():
        raise ValueError('a native loop is not overwritten or silently resumed')
    if declaration.get('whole_build_review', {}).get('accepted') is not True:
        raise ValueError('the requested independent whole-build review must precede native correction')
    checked(ROOT, declaration['whole_build_review']['source'], parse=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save(run / 'execution.json', dict(started_at=now(), operation=declaration['operation']))
    try:
        result = pair_loop.iterate(Backend(run, declaration))
    except BaseException as error:
        save(run / 'interrupted.json', dict(at=now(), complete=False, error=type(error).__name__,
             message=str(error), partials='retained'))
        raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
