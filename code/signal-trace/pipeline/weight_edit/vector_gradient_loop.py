"""Direct computational gradient, complete native installation, fresh trace.

This separately declared controller replaces the uncompleted CPU-vector solve.
It never uses an inverse iterate, candidate-search loop or response-text judge.
"""
import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
from typing import Any

import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.research_guard.verify_round2 import ROOT

from . import functional_install, gap_state, native_graph, pair_loop, trace_gradient, vector_loop
from .functional_reading import digest
from .functional_reference import checked
from .vector_chance_run import now, save

FORMAT = 'response-scaled-weight-gradient/1'


class Backend(vector_loop.Backend):
    def __init__(self, run, declaration, *, root=ROOT, acquire=None):
        if declaration.get('gradient_step', {}).get('format') != FORMAT:
            raise ValueError('the direct gradient rule must be declared explicitly')
        if 'recovery' in declaration or 'continuation' in declaration:
            raise ValueError('this direct-gradient run starts freshly at the original model')
        super().__init__(run, declaration, root=root, acquire=acquire)

    def correct(self, number, observation) -> dict[str, Any]:
        if observation is not self.last or self.case is None or self.pairs is None or observation['step'] != number:
            raise ValueError('a correction requires its immediately preceding native observation')
        if not observation['complete'] or not observation['ended'] or observation['outside'] == 0:
            raise ValueError('only a complete outside native path supplies a weight gradient')
        self._state(number)
        step = self.run / 'loop' / str(number)
        with vector_loop.card_lock(self.device, self.card):
            is_cuda = torch.device(self.device).type == 'cuda'
            if is_cuda:
                # Native acquisition is a subprocess; this controller can still
                # have an uninitialized allocator on its first correction.
                torch.cuda.init()
                torch.cuda.reset_peak_memory_stats(self.device)
            model = self._model(self.device, editable=True)
            if tuple(model.leaves) != self.names:
                raise ValueError('the differentiable graph must expose every declared matrix')
            graph = native_graph.Graph(model, self.case.history, self.case.native)

            def progress(phase, details):
                memory = (dict(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                               gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
                          if is_cuda else {})
                save(step / 'gradient-progress.json', dict(at=now(), phase=phase, **details, **memory))

            proposal = trace_gradient.propose(graph, self.case.history, self.reference,
                                             pairs=self.pairs, progress=progress)
            if any(proposal.before[key] != observation[key] for key in ('objective', 'outside', 'pair_merge_cells')):
                raise ValueError('the differentiable origin must reproduce its complete native measurement')
            record = dict(status=proposal.status, before=proposal.before,
                gradient_norm_squared=proposal.gradient_norm_squared,
                response=asdict(proposal.response) if proposal.response is not None else None,
                changed_values=proposal.changed_values, timings=proposal.timings,
                rule=FORMAT, closure=False, at=now(),
                value_is='one local linear response prediction; fresh native effect not yet measured',
                acceptance='complete prescribed native installation only; no candidate search or loss acceptance gate')
            if is_cuda:
                record['gpu'] = dict(device=self.device,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            save(step / 'proposal.json', record)
            weights = proposal.weights
            del proposal, graph, model
            gc.collect()
            if torch.device(self.device).type == 'cuda':
                torch.cuda.empty_cache()
        if record['status'] != 'ready':
            return record
        if weights is None or set(weights) != set(self.names):
            raise ValueError('the prescribed update must contain every native matrix')
        reached = self.install(number, weights)
        self.case = None
        record.update(status='accepted', installed_at=now(), reached=reached)
        save(step / 'proposal.json', record)
        return record

    def install(self, number, weights):
        """Legacy retained-copy publication; the working-file backend overrides it."""
        step = self.run / 'loop' / str(number)
        required = self.original.stat().st_size * (2 if number == 1 else 1)
        if self.declaration.get('compact_states'):
            required += sum(self.tensors[name]['bytes'] for name in self.names)
        if shutil.disk_usage(self.run).free - required < 40 * 10**9:
            raise OSError('insufficient disk for complete native copies above the existing project floor')
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
            compact = vector_loop.compact_state(output, self.original, self.tensors, installed['kept'],
                                               start.file, start.kept(), root=self.root)
            save(step / 'compact.json', compact)
        return reached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    run = Path(parser.parse_args().run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('all outputs belong inside the declared run')
    if any((run / name).exists() for name in ('loop', 'execution.json', 'result.json')):
        raise ValueError('a direct-gradient run is never overwritten or silently resumed')
    if declaration.get('whole_build_review', {}).get('accepted') is not True:
        raise ValueError('the independent whole-build review must cover this direct-gradient route')
    checked(ROOT, declaration['whole_build_review']['source'], parse=False)
    replacement = declaration.get('replaces', {})
    if replacement.get('controller_pid') and Path('/proc', str(replacement['controller_pid'])).exists():
        raise ValueError('the previous controller must be stopped before this controller starts')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save(run / 'execution.json', dict(started_at=now(), operation=declaration['operation'],
         gradient_step=declaration['gradient_step'], replaces=replacement,
         partial_inverse='not used; previous unsaved inverse is abandoned by explicit direction'))
    try:
        result = pair_loop.iterate(Backend(run, declaration))
    except BaseException as error:
        save(run / 'interrupted.json', dict(at=now(), complete=False, error=type(error).__name__,
             message=str(error), partials='retained; previous experiment records unchanged'))
        raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
