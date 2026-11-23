"""Compile original predictive vector chance, without a model execution."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np

from pipeline.native_records.run_declaration import check_native_run
from pipeline.research_guard.verify_round2 import ROOT

from .functional_reference import checked
from . import vector_chance_reference


def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def binding(path, root):
    with Path(path).open('rb') as file:
        return dict(record=str(Path(path).relative_to(root)),
                    sha256=hashlib.file_digest(file, 'sha256').hexdigest())


def compile_original(run: Path, declaration, *, root=ROOT):
    started = time.perf_counter()
    for source in declaration['implementation']:
        checked(root, source, parse=False)
    basis = checked(root, declaration['basis'])
    if basis.get('format') != 'single-original-turn-reference/1' or basis.get('complete') is not True:
        raise ValueError('the original reference selection must already be complete and fixed')
    definition = checked(root, basis['source_definition'])
    names = tuple(name for name, item in definition['members'].items() if item['group'] == 'correct')
    if (list(names) != basis['original_successes'] or set(names) != set(basis['original_turns'])
            or basis['reference'] not in names
            or basis['source_model_sha256'] != definition['sources']['model']['sha256']):
        raise ValueError('the complete original population and model must stay fixed')
    paths = {name: np.asarray(path) for name, path in checked(root, basis['correspondence']).items()}
    arrays = {}
    for name in names:
        source = basis['original_turns'][name]
        checked(root, source, parse=False)
        array = np.load(root / source['record'], mmap_mode='r', allow_pickle=False)
        if array.dtype != np.dtype('float64') or list(array.shape) != source['shape']:
            raise ValueError('an original turn array differs from its declared shape or dtype')
        arrays[name] = array
    def progress(completed, required):
        save(run / 'progress.json', dict(at=now(), completed=completed, required=required,
             complete=completed == required, seconds=time.perf_counter() - started))
    calibration = vector_chance_reference.compile(
        basis['reference'], names, arrays.__getitem__, paths, progress=progress)
    fields = {}
    for name in ('chance', 'within', 'between', 'offset'):
        path = run / (name + '.npy')
        value = getattr(calibration, name)
        np.save(path, value, allow_pickle=False)
        fields[name] = binding(path, root) | dict(shape=list(value.shape), dtype=str(value.dtype))
    coverage = run / 'original-coverage.json'
    save(coverage, calibration.coverage)
    inherited = ('reference', 'source_definition', 'source_model_sha256', 'original_successes',
                 'original_turns', 'trace', 'correspondence', 'scores')
    reference = {key: basis[key] for key in inherited}
    reference.update(format='single-original-vector-chance/1', complete=True, compiled_at=now(),
        basis=declaration['basis'], **fields, coverage=binding(coverage, root),
        normalization='canonical-cpu-float64',
        chance_definition='mean nonself witness mean mapped-event squared whole-turn distance to the fixed reference; not a population-mean standard error',
        boundary='norm(current_turn - reference_turn) <= sqrt(2 * original_chance)',
        native_loop_admitted=False)
    save(run / 'reference.json', reference)
    result = dict(status='compiled_original_vector_chance', complete=True, completed_at=now(),
        reference=basis['reference'], original_successes=len(names), witnesses=len(paths),
        reference_shape=basis['trace']['shape'], chance_shape=list(calibration.chance.shape),
        reference_sha256=binding(run / 'reference.json', root)['sha256'],
        seconds=time.perf_counter() - started,
        complete_witnesses_inside=calibration.coverage['complete_witnesses_inside'],
        zero_chance_cells=calibration.coverage['zero_chance_cells'],
        decomposition_max_absolute_error=calibration.coverage['decomposition_max_absolute_error'],
        weight_updates=0, generated_answers=0, wrong_trace_evaluations=0,
        closure=False, repair_claim=False, native_loop_admitted=False,
        prediction_vs_result='Complete predictive whole-vector chance, its within/between/offset decomposition and descriptive original coverage compiled from the fixed originals.',
        failures_only='Predictive empirical variation is not a confidence bound or a proof of computational equivalence. Reference selection and correspondence saw all originals; whole-vector variance can mask a stable-direction departure.',
        what_this_changes='Supplies a fixed original vector chance field for the separately reviewed one-prompt native loop. No weight changed and no correction loop is admitted by compilation.')
    save(run / 'result.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    run = Path(parser.parse_args().run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if declaration.get('operation') != 'single-original-vector-chance':
        raise ValueError('this compiler requires its prospective original-only declaration')
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('all outputs belong inside the declared run')
    if (run / 'chance.npy').exists() or (run / 'result.json').exists():
        raise ValueError('an original vector compilation is not overwritten or silently resumed')
    save(run / 'execution.json', dict(started_at=now(), operation=declaration['operation'],
                                     generated_answers=0, weight_updates=0))
    try:
        result = compile_original(run, declaration)
    except BaseException as error:
        save(run / 'interrupted.json', dict(at=now(), complete=False, error=type(error).__name__,
             message=str(error), partials='retained', weight_updates=0))
        raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
