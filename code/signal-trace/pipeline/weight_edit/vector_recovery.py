"""Explicitly restart an interrupted first inverse from its retained native trace.

The source run remains read-only. Only the original-weight first observation is
reused; after a complete installed update, the existing backend acquires a new
native trace. No inverse iterate is inferred from progress scalars.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.research_guard.verify_round2 import ROOT

from . import pair_loop, turn_correction, vector_loop
from .functional_reading import Reading
from .functional_reference import checked
from .vector_chance_run import binding, now, save


# These fields specify the computation and execution being continued. Recovery
# adds provenance and an entrypoint; it does not select another experiment.
UNCHANGED = ('operation', 'model_sha256', 'original_model', 'source_definition',
             'forward', 'architecture', 'tensors', 'reference', 'device', 'card',
             'compact_states', 'functional_loop', 'post_closure', 'parameters', 'rule')


class Backend(vector_loop.Backend):
    def __init__(self, run, declaration, *, root=ROOT, acquire=None):
        root = Path(root).resolve()
        recovery = declaration['recovery']
        source = checked(root, recovery['declaration'])
        source_run = (root / recovery['declaration']['record']).resolve().parent
        recovery_run = Path(run).resolve()
        if source_run.is_relative_to(recovery_run) or recovery_run.is_relative_to(source_run):
            raise ValueError('source and recovery run directories must not overlap')
        for key in UNCHANGED:
            if declaration.get(key) != source.get(key):
                raise ValueError(f'recovery changes the original experiment: {key}')
        if source.get('whole_build_review', {}).get('accepted') is not True:
            raise ValueError('the source experiment has no accepted whole-build review')
        checked(root, source['whole_build_review']['source'], parse=False)
        current = {item['record']: item['sha256'] for item in declaration['implementation']}
        if len(current) != len(declaration['implementation']):
            raise ValueError('implementation bindings must not be duplicated')
        if any(current.get(item['record']) != item['sha256'] for item in source['implementation']):
            raise ValueError('recovery must retain every original implementation binding')
        step = source_run / 'loop/1'
        if (root / recovery['measurement']['record']).resolve() != step / 'measurement.json':
            raise ValueError('recovery must bind the original first observation')
        if (any((step / name).exists() for name in ('proposal.json', 'reached.json', 'model.gguf'))
                or any((source_run / name).exists() for name in ('result.json', 'closure.json'))
                or any(path.name != '1' for path in step.parent.iterdir())):
            raise ValueError('recovery is only for an interrupted first inverse without an update')
        observed = checked(root, recovery['measurement'])
        for key, expected in (('step', 1), ('model_file', source['original_model']['record']),
                              ('model_sha256', source['original_model']['sha256']),
                              ('reference_sha256', source['reference']['sha256']),
                              ('member', source['functional_loop']['training_members'][0])):
            if observed.get(key) != expected:
                raise ValueError(f'the retained observation has another {key}')
        if (observed.get('complete') is not True or observed.get('ended') is not True
                or not observed['outside']):
            raise ValueError('recovery needs the complete outside observation that supplied the inverse')
        for key, path in (('reading', step / 'reading/reading.json'),
                          ('correspondence', step / 'pairs.npy')):
            if (root / observed[key]['record']).resolve() != path:
                raise ValueError(f'the retained {key} belongs to another observation')
            if recovery[key] != observed[key]:
                raise ValueError(f'recovery must use the previously recorded {key} hash')
        record = checked(root, observed['reading'])
        if (record.get('declaration_sha256') != recovery['declaration']['sha256']
                or record.get('acquisition') != 'native-events-only'):
            raise ValueError('retained native acquisition must keep its original declaration identity')
        checked(root, observed['correspondence'], parse=False)
        super().__init__(run, declaration, root=root, acquire=acquire)
        self.retained = observed

    def observe(self, number):
        if number != 1:
            return super().observe(number)
        step = self.run / 'loop/1'
        if step.exists():
            raise ValueError('a recovered observation is never overwritten')
        self._state(1)
        observed = self.retained
        folder = (self.root / observed['reading']['record']).parent
        model = self._model('cpu', editable=False)
        reader = Reading(folder, model, {self.member: self.definition['prompts'][self.member]},
                         expected_reading_sha256=observed['reading']['sha256'],
                         expected_model_sha256=self.model_sha256)
        checked(self.root, observed['correspondence'], parse=False)
        pairs = np.load(self.root / observed['correspondence']['record'], allow_pickle=False)
        with torch.no_grad():
            case = reader.read(self.member)
            measured = turn_correction.measurement(self.reference, case.native, pairs)
        for key, value in measured.items():
            if observed.get(key) != value:
                raise ValueError(f'retained computational measurement does not reproduce: {key}')
        if case.complete is not True or case.ended is not True:
            raise ValueError('the retained native observation is incomplete')
        if len(case.history) != observed['events'] or self.reference.merges != observed['merges']:
            raise ValueError('the retained computational scope changed')
        step.mkdir(parents=True)
        np.save(step / 'pairs.npy', pairs, allow_pickle=False)
        measured.update({key: observed[key] for key in
                         ('member', 'complete', 'ended', 'step', 'events', 'merges',
                          'model_file', 'model_sha256', 'reference_sha256', 'reading')})
        measured.update(measured_at=now(),
                        correspondence=binding(step / 'pairs.npy', self.root),
                        recovered_from=self.declaration['recovery']['measurement'],
                        original_measured_at=observed['measured_at'],
                        acquisition_declaration=self.declaration['recovery']['declaration'],
                        measure='retained native input-output turns, remeasured; no fresh acquisition')
        save(step / 'measurement.json', measured)
        self.last, self.case, self.pairs = measured, case, pairs
        return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    run = Path(parser.parse_args().run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('all outputs belong inside the declared run')
    if (run / 'loop').exists() or (run / 'result.json').exists():
        raise ValueError('a recovery run is never overwritten or silently resumed')
    if declaration.get('whole_build_review', {}).get('accepted') is not True:
        raise ValueError('the independent review must cover this recovery before execution')
    checked(ROOT, declaration['whole_build_review']['source'], parse=False)
    backend = Backend(run, declaration)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save(run / 'execution.json', dict(started_at=now(), operation=declaration['operation'],
         recovery=declaration['recovery'], inverse_restart='zero; no saved inverse vectors'))
    try:
        result = pair_loop.iterate(backend)
    except BaseException as error:
        save(run / 'interrupted.json', dict(at=now(), complete=False, error=type(error).__name__,
             message=str(error), partials='retained; original source run untouched'))
        raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
