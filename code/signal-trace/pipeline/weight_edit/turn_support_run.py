"""Original-only signed support with fixed reference and fixed correspondence."""
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
from . import turn_support


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
        raise ValueError('the complete original reference selection must be fixed before this diagnostic')
    definition = checked(root, basis['source_definition'])
    names = tuple(name for name, item in definition['members'].items() if item['group'] == 'correct')
    if (list(names) != basis['original_successes'] or set(names) != set(basis['original_turns'])
            or basis['reference'] not in names
            or basis['source_model_sha256'] != definition['sources']['model']['sha256']):
        raise ValueError('the complete original fitting-success inventory and model must stay fixed')
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
    support = turn_support.compile(basis['reference'], names, arrays.__getitem__, paths, progress=progress)
    low, high, coverage = run / 'lower.npy', run / 'upper.npy', run / 'original-coverage.json'
    np.save(low, support.lower, allow_pickle=False)
    np.save(high, support.upper, allow_pickle=False)
    save(coverage, support.coverage)
    reference = dict(format='single-original-turn-support/1', complete=True, compiled_at=now(),
        basis=declaration['basis'], reference=basis['reference'], source_definition=basis['source_definition'],
        source_model_sha256=basis['source_model_sha256'], original_successes=list(names),
        original_turns=basis['original_turns'], trace=basis['trace'], correspondence=basis['correspondence'],
        lower=binding(low, root), upper=binding(high, root), coverage=binding(coverage, root),
        closure_boundary_admitted=False,
        formula='exact signed extrema of every mapped original departure, including zero as the fixed reference origin')
    save(run / 'reference.json', reference)
    result = dict(status='compiled_original_support', complete=True, completed_at=now(),
        reference=basis['reference'], original_successes=len(names), witnesses=len(paths),
        reference_shape=list(support.lower.shape), reference_sha256=binding(run / 'reference.json', root)['sha256'],
        seconds=time.perf_counter() - started,
        complete_witnesses_inside=support.coverage['complete_witnesses_inside'],
        prospective_weakening_condition_met=support.coverage['prospective_weakening_condition_met'],
        weight_updates=0, generated_answers=0, wrong_trace_evaluations=0,
        closure=False, repair_claim=False, closure_boundary_admitted=False,
        prediction_vs_result='Complete signed original support and both coverage readings were compiled without changing the selected reference or paths.',
        failures_only='Calibration containment is construction, not validation; finite-sample ranges and independent coordinate combinations can misrepresent new correct computation. The declared leave-one-witness-out structural weakening condition is recorded separately.',
        what_this_changes='Original-only evidence for assessing the interval form. No native correction loop or post-closure evaluation is admitted.')
    save(run / 'result.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    run = Path(parser.parse_args().run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if declaration.get('operation') != 'single-original-turn-support':
        raise ValueError('this compiler requires its prospective original-only declaration')
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('all outputs belong inside the declared run')
    if (run / 'lower.npy').exists() or (run / 'result.json').exists():
        raise ValueError('an original support compilation is not overwritten or silently resumed')
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
