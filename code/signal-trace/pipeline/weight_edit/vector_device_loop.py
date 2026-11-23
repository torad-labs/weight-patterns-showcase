"""Prospective two-device inverse through the unchanged native loop.

This entrypoint never attaches to an existing process or resumes a partial
inverse. The declared graph card remains the backend's responsibility; vectors
acquire the second card only within a complete inverse. Scientific observations,
BF16 installation, fresh generation and closure use the existing controller.
"""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import subprocess
import threading
from types import ModuleType

import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.research_guard.verify_round2 import ROOT

from . import joint_functional_step, joint_inverse, joint_inverse_device, pair_loop, vector_loop
from .functional_reading import digest
from .functional_reference import checked
from .vector_chance_run import now, save

FORMAT = 'joint-inverse-device/1'
_route_lock = threading.Lock()


class Backend(vector_loop.Backend):
    def _acquire(self, number, model):
        # CUDA child ordinals are local to the visible list. Keep the graph UUID
        # selected by validation while the recorder lock keeps its physical ID.
        plan = vector_loop.capture_commands(self.run, self.declaration, number, model, root=self.root)
        graph = torch.device(self.device)
        assert graph.index is not None
        uuid = self.declaration['vector_solver']['visible_devices'][graph.index]
        env = os.environ | plan['env'] | dict(CUDA_VISIBLE_DEVICES=uuid,
            RECORDER_UNDER=f'flock /run/user/{os.getuid()}/gpu{self.card}.lock')
        step = self.run / 'loop' / str(number)
        save(step / 'capture-plan.json', plan | dict(physical_graph_uuid=uuid, graph_card=self.card))
        for index, command in enumerate(plan['commands']):
            with (step / f'acquisition-{index}.log').open('w') as log:
                subprocess.run(command, cwd=self.root / 'code/signal-trace', env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        return step / 'reading'


@contextmanager
def solver_route(graph_device, vector_device, vector_card):
    """Route only the joint adapter, restoring its module even on failure.

The native backend already holds the graph lock when it calls this solver.
The original inverse module (including any observer wrappers) stays intact.
"""
    graph_device, vector_device = torch.device(graph_device), torch.device(vector_device)
    if not _route_lock.acquire(blocking=False):
        raise RuntimeError('only one declared inverse route may be active in this process')
    original = joint_functional_step.joint_inverse

    def solve(residual, parameters, *, progress=None):
        if not parameters or any(weight.device != graph_device for weight in parameters):
            raise ValueError('every inverse parameter must remain on the declared graph device')
        with vector_loop.card_lock(vector_device, vector_card):
            try:
                return joint_inverse_device.solve(residual, parameters,
                    vector_device=vector_device, progress=progress)
            finally:
                if vector_device.type == 'cuda':
                    gc.collect()
                    with torch.cuda.device(vector_device):
                        torch.cuda.empty_cache()

    facade = ModuleType('prospectively_bound_joint_inverse')
    setattr(facade, 'solve', solve)
    try:
        joint_functional_step.joint_inverse = facade
        yield
    finally:
        joint_functional_step.joint_inverse = original
        _route_lock.release()


def validate_placement(declaration, *, environment=None, inventory=None):
    """Bind logical CUDA devices and lock numbers to physical GPU UUIDs."""
    spec = declaration['vector_solver']
    if spec.get('format') != FORMAT:
        raise ValueError('the explicit device-resident inverse format is required')
    graph, vector = torch.device(declaration['device']), torch.device(spec['device'])
    cards = (declaration['card'], spec['card'])
    if (graph.type != 'cuda' or vector.type != 'cuda' or graph.index is None
            or vector.index is None or graph == vector
            or any(type(card) is not int or card < 0 for card in cards)
            or cards[0] >= cards[1]):
        raise ValueError('distinct explicit CUDA devices must acquire physical cards in increasing order')
    visible = spec['visible_devices']
    if (not isinstance(visible, list) or len(visible) != 2 or len(set(visible)) != 2
            or not all(isinstance(value, str) and value.startswith('GPU-') for value in visible)
            or {graph.index, vector.index} != {0, 1}):
        raise ValueError('both logical devices must have explicit distinct visible GPU UUIDs')
    environment = os.environ if environment is None else environment
    if environment.get('CUDA_VISIBLE_DEVICES') != ','.join(visible):
        raise ValueError('CUDA_VISIBLE_DEVICES differs from the declared UUID mapping')
    if inventory is None:
        output = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                                check=True, capture_output=True, text=True).stdout
        inventory = {int(index.strip()): uuid.strip() for index, uuid in
                     (line.split(',') for line in output.splitlines() if line.strip())}
    if (inventory.get(cards[0]) != visible[graph.index]
            or inventory.get(cards[1]) != visible[vector.index]):
        raise ValueError('GPU lock cards differ from the declared physical devices')
    return dict(graph_device=str(graph), vector_device=str(vector), graph_card=cards[0],
                vector_card=cards[1], visible_devices=visible)


def validate_sources(declaration, *, root=ROOT):
    """Require prospective source and complete graph qualification bindings."""
    root = Path(root)
    spec = declaration['vector_solver']
    sources = declaration['implementation']
    declared = {item['record']: item['sha256'] for item in sources}
    if len(declared) != len(sources):
        raise ValueError('implementation bindings must not be duplicated')
    for module in (joint_inverse_device, joint_inverse, joint_functional_step, pair_loop, vector_loop):
        if module.__file__ is None:
            raise ValueError('the device inverse requires file-backed dependencies')
        file = Path(module.__file__).resolve()
        path = str(file.relative_to(root))
        if declared.get(path) != digest(file):
            raise ValueError(f'the device inverse dependency is not bound: {path}')
    runner = Path(__file__).resolve()
    if declared.get(str(runner.relative_to(root))) != digest(runner):
        raise ValueError('the device loop entrypoint must be bound')
    candidate = Path(joint_inverse_device.__file__).resolve()
    expected = dict(record=str(candidate.relative_to(root)), sha256=digest(candidate))
    if spec.get('source') != expected:
        raise ValueError('the declared inverse source differs from the loaded candidate')
    qualified = checked(root, spec['qualification'])
    if (qualified.get('format') != 'joint-inverse-two-device-qualification/1'
            or qualified.get('complete') is not True
            or qualified.get('source') != expected
            or qualified.get('implementation') != declaration['implementation']
            or qualified.get('forward') != declaration['forward']
            or qualified.get('source_definition') != declaration['source_definition']
            or qualified.get('architecture') != declaration['architecture']
            or qualified.get('tensors') != declaration['tensors']
            or qualified.get('model_sha256') != declaration['model_sha256']
            or qualified.get('reference') != declaration['reference']
            or qualified.get('matrices') != declaration['functional_loop']['matrices']
            or qualified.get('training_members') != declaration['functional_loop']['training_members']
            or qualified.get('graph_device') != declaration['device']
            or qualified.get('vector_device') != spec['device']
            or qualified.get('visible_devices') != spec['visible_devices']
            or qualified.get('complete_graph_products') is not True
            or qualified.get('noninitial_candidate_error') is not True):
        raise ValueError('complete two-device graph qualification must match the declared experiment')
    return qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    run = Path(parser.parse_args().run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('all outputs belong inside the declared run')
    if any((run / item).exists() for item in ('loop', 'result.json', 'execution.json')):
        raise ValueError('a device loop is never overwritten or silently resumed')
    if 'recovery' in declaration or 'continuation' in declaration:
        raise ValueError('this entrypoint starts at original weights; it cannot resume a prior inverse or update')
    if declaration.get('whole_build_review', {}).get('accepted') is not True:
        raise ValueError('the independent whole-build review must cover this device loop')
    checked(ROOT, declaration['whole_build_review']['source'], parse=False)
    validate_sources(declaration)
    placement = validate_placement(declaration)
    backend = Backend(run, declaration)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save(run / 'execution.json', dict(started_at=now(), operation=declaration['operation'],
         inverse=declaration['vector_solver'], placement=placement,
         inverse_restart='zero; no partial inverse is resumed'))
    try:
        with solver_route(placement['graph_device'], placement['vector_device'], placement['vector_card']):
            result = pair_loop.iterate(backend)
    except BaseException as error:
        save(run / 'interrupted.json', dict(at=now(), complete=False, error=type(error).__name__,
             message=str(error), partials='retained; no prior run changed'))
        raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
