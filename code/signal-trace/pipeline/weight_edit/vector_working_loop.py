"""Resident trainable weights; one GGUF handoff; fresh native measurement.

The existing recorder loads a file for each capture. Write that one working
file after each direct update. Checkpoint saving is a separate operation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.research_guard.verify_round2 import ROOT

from . import pair_loop, turn_correction, vector_gradient_loop, vector_loop, working_identity, working_model
from .functional_reading import Reading, digest
from .functional_reference import checked
from .vector_chance_run import binding, now, save


class Backend(vector_gradient_loop.Backend):
    def __init__(self, run, declaration, *, root=ROOT, acquire=None):
        if declaration.get('working_model', {}).get('format') != working_identity.FORMAT:
            raise ValueError('the single working model must be declared explicitly')
        if declaration.get('compact_states'):
            raise ValueError('checkpoint saving is separate from the optimization step')
        self.store = None
        self.models = {}
        self.start = declaration['working_model'].get('start')
        super().__init__(run, declaration, root=root, acquire=acquire)
        for spec in (self.spec, declaration['post_closure']):
            protocol = checked(self.root, spec['reader_protocol'])
            if working_identity.KIND not in protocol.get('model_identity_formats', []):
                raise ValueError('native protocols must distinguish revisions from content hashes')
        if shutil.disk_usage(self.run).free - self.original.stat().st_size < 40 * 10**9:
            raise OSError('insufficient disk for the one working copy above the existing project floor')
        self.store = working_model.Working.create(self.run / 'working', self.original,
            self.original_binding['sha256'], self.tensors, self.names, run=str(self.run.relative_to(self.root)),
            start=self.root / self.start['file'] if self.start else None,
            start_sha256=self.start['model_sha256'] if self.start else None)
        self._state(1)

    def close(self):
        self.models.clear()
        if self.store is not None:
            self.store.close()

    def _state(self, number):
        if self.store is None:
            if self.start is not None:
                reached = checked(self.root, self.start['reached'])
                first = dict(file=self.start['file'], model_sha256=self.start['model_sha256'])
                if (reached.get('format') != 'functional-reached-state/1' or reached.get('step', 0) < 1
                        or any(reached[key] != value for key, value in first.items())
                        or self.spec['first_state'] != first):
                    raise ValueError('the retained start must match its complete reached record and first-state binding')
                self.model_file, self.model_sha256 = self.root / first['file'], first['model_sha256']
                return
            return vector_loop.Backend._state(self, number)  # startup source check only
        state = self.store.current()
        if state['generation'] != number - 1:
            raise ValueError('the observation must follow the current complete update')
        self.model_file, self.model_sha256 = self.store.file, None
        self.model_identity = state['model_identity']

    def _model(self, device, *, editable):
        key = (str(device), editable)
        if key not in self.models:
            self.models[key] = super()._model(device, editable=editable)
        return self.models[key]

    def _acquire(self, number, model):
        assert self.store is not None and self.store.owner is not None
        self.store.current()
        step = self.run / 'loop' / str(number)
        plan = vector_loop.capture_commands(self.run, self.declaration, number, model, root=self.root)
        env = os.environ | plan['env'] | dict(CUDA_VISIBLE_DEVICES=str(self.card),
            WP_WORKING_MODEL_OWNER_FD=str(self.store.owner),
            RECORDER_UNDER=f'flock /run/user/{os.getuid()}/gpu{self.card}.lock')
        save(step / 'capture-plan.json', plan)
        for index, command in enumerate(plan['commands']):
            with (step / f'acquisition-{index}.log').open('w') as log:
                subprocess.run(command, cwd=self.root / 'code/signal-trace', env=env,
                    pass_fds=(self.store.owner,), stdout=log, stderr=subprocess.STDOUT, check=True)
        self.store.current()
        return step / 'reading'

    def observe(self, number):
        assert self.store is not None
        step = self.run / 'loop' / str(number)
        if step.exists():
            raise ValueError('a native observation is never overwritten or silently resumed')
        self._state(number)
        step.mkdir(parents=True)
        folder = Path(self.acquire(number, self.model_file)).resolve()
        if folder != step / 'reading':
            raise ValueError('the fresh reading must belong to this numbered acquisition')
        self.store.current()
        reader = Reading(folder, self._model('cpu', editable=False),
            {self.member: self.definition['prompts'][self.member]},
            expected_reading_sha256=digest(folder / 'reading.json'), expected_model_identity=self.model_identity)
        with torch.no_grad():
            case = reader.read(self.member)
            pairs = turn_correction.correspondence(self.reference, case.native)
            measured = turn_correction.measurement(self.reference, case.native, pairs)
        self.store.current()
        np.save(step / 'pairs.npy', pairs, allow_pickle=False)
        measured.update(member=self.member, complete=case.complete is True, ended=case.ended,
            step=number, measured_at=now(), events=len(case.history), merges=self.reference.merges,
            model_file=str(self.model_file.relative_to(self.root)), **working_identity.fields(self.model_identity),
            generation=number - 1, reference_sha256=self.declaration['reference']['sha256'],
            reading=binding(folder / 'reading.json', self.root), correspondence=binding(step / 'pairs.npy', self.root),
            measure='fresh native input-output turns; no output-token judge')
        save(step / 'measurement.json', measured)
        self.last, self.case, self.pairs = measured, case, pairs
        return measured

    def install(self, number, weights):
        assert self.store is not None
        state = self.store.install(self.model_identity, weights, number)
        # Keep the already loaded trainable state and measurement model resident.
        # Embedding tables reopened by rows() are outside these editable matrices.
        with torch.no_grad():
            for model in self.models.values():
                for name in self.names:
                    model.w[name].copy_(torch.from_numpy(weights[name]).to(model.w[name]))
        reached = dict(format='native-working-reached-state/1', step=number, generation=number,
            file=str(self.model_file.relative_to(self.root)), **working_identity.fields(state['model_identity']),
            retention='current working bytes only; historical revisions are metadata, not checkpoints')
        save(self.run / 'loop' / str(number) / 'reached.json', reached)
        return reached

    def post_closure(self):
        result = super().post_closure()
        assert self.last is not None
        path = self.run / 'closure.json'
        record = json.loads(path.read_text())
        record.update(**working_identity.fields(self.last['model_identity']), generation=self.last['generation'])
        save(path, record)
        result['closure'] = binding(path, self.root)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    run = Path(parser.parse_args().run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('all outputs belong inside the declared run')
    if any((run / name).exists() for name in ('authorized-stop.json', 'working', 'loop', 'execution.json', 'result.json')):
        raise ValueError('existing or explicitly stopped research is preserved; never auto-restarted')
    if declaration.get('whole_build_review', {}).get('accepted') is not True:
        raise ValueError('independent review must cover this native handoff')
    checked(ROOT, declaration['whole_build_review']['source'], parse=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backend = None
    save(run / 'execution.json', dict(started_at=now(), operation=declaration['operation'],
        gradient_step=declaration['gradient_step'], working_model=declaration['working_model']))
    try:
        backend = Backend(run, declaration)
        result = pair_loop.iterate(backend)
    except BaseException as error:
        save(run / 'interrupted.json', dict(at=now(), complete=False, error=type(error).__name__,
            message=str(error), partials='preserved; no automatic restart or checkpoint recovery'))
        raise
    finally:
        if backend is not None:
            backend.close()
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
