"""Compile one original turn reference and its nonself component RMS boundary.

File-only original observations: no generation, weight update, wrong-answer trace
evaluation or output-text judge. The declared implementation and source hashes
are checked before reading arrays; all products remain inside the declared run.
"""
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.research_guard.verify_round2 import ROOT

from .functional_reading import Reading, digest
from .functional_reference import checked
from . import turn_pair, turn_reference


def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def compile_original(run: Path, declaration, *, root=ROOT):
    for binding in declaration['implementation']:
        checked(root, binding, parse=False)
    definition = checked(root, declaration['source_definition'])
    sources = definition['sources']
    names = tuple(name for name, entry in definition['members'].items() if entry['group'] == 'correct')
    if list(names) != declaration['original_successes']:
        raise ValueError('the complete original fitting-success inventory must be declared unchanged')
    forward = declaration['forward']
    checked(root, forward, parse=False)
    spec = importlib.util.spec_from_file_location('bound_turn_reference_forward', root / forward['record'])
    if spec is None or spec.loader is None:
        raise ValueError('the bound original model reader is unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model(root / sources['model']['record'], checked(root, sources['tensors']),
                         checked(root, sources['architecture']), device=declaration['device'],
                         cache='float16', leaves=lambda name: False)
    reader = Reading((root / sources['reading']['record']).parent, model, definition['prompts'],
                     expected_reading_sha256=sources['reading']['sha256'],
                     expected_model_sha256=sources['model']['sha256'])
    started = time.perf_counter()
    traces = {}
    folder = run / 'original-turns'
    folder.mkdir()
    with torch.no_grad():
        for name in names:
            case = reader.read(name)
            if case.complete is not True or not case.ended:
                raise ValueError('every original witness must have a complete naturally ended acquisition')
            values = turn_pair.turns(case.native, expected_merges=3 * model.layers)
            array = np.stack([value.cpu().numpy() for value in values], axis=1)
            path = folder / (name + '.npy')
            np.save(path, array, allow_pickle=False)
            traces[name] = dict(record=str(path.relative_to(root)), sha256=digest(path),
                                shape=list(array.shape), dtype=str(array.dtype))
            save(run / 'extraction.json', dict(at=now(), completed=len(traces), required=len(names),
                 traces=traces, complete=False, weight_updates=0))
            del case, values, array
    del reader, model
    if str(declaration['device']).startswith('cuda'):
        torch.cuda.empty_cache()
    scores = {}
    def read(name):
        # The arrays were made here from verified originals, and remain hash-bound.
        return np.load(root / traces[name]['record'], mmap_mode='r', allow_pickle=False)
    def progress(name, score):
        scores[name] = score
        save(run / 'selection-progress.json', dict(at=now(), completed=len(scores),
             required=len(names), scores=scores, complete=False))
    selected = turn_reference.select(names, read, progress=progress)
    tolerance_path = run / 'tolerance.npy'
    np.save(tolerance_path, selected.tolerance, allow_pickle=False)
    paths_file = run / 'correspondence.json'
    save(paths_file, {name: path.tolist() for name, path in selected.paths.items()})
    coverage_file = run / 'original-coverage.json'
    def coverage_progress(completed, required):
        save(run / 'coverage-progress.json', dict(at=now(), completed=completed,
             required=required, complete=completed == required))
    save(coverage_file, turn_reference.coverage(selected, read, progress=coverage_progress))
    reference = dict(format='single-original-turn-reference/1', reference=selected.reference,
        source_definition=declaration['source_definition'], original_successes=list(names),
        source_model_sha256=sources['model']['sha256'], original_turns=traces,
        trace=traces[selected.reference],
        tolerance=dict(record=str(tolerance_path.relative_to(root)), sha256=digest(tolerance_path)),
        correspondence=dict(record=str(paths_file.relative_to(root)), sha256=digest(paths_file)),
        coverage=dict(record=str(coverage_file.relative_to(root)), sha256=digest(coverage_file)),
        formula='per-component nonself original RMS departure, each witness and each reference event equally weighted',
        scores=selected.scores, complete=True, compiled_at=now())
    save(run / 'reference.json', reference)
    result = dict(status='compiled_original_reference', complete=True, completed_at=now(),
        reference=selected.reference, witnesses=len(names) - 1, original_successes=len(names),
        reference_shape=traces[selected.reference]['shape'], seconds=time.perf_counter() - started,
        reference_sha256=digest(run / 'reference.json'), weight_updates=0, generated_answers=0,
        wrong_trace_evaluations=0, repair_claim=False, closure=False,
        prediction_vs_result='One observed original reference and its complete component RMS boundary compiled from every declared fitting success.',
        failures_only='RMS is a measured variation scale, not a confidence bound; some original successes may stand outside it. Coordinate bounds can admit unobserved combinations, and chronological correspondence is not proof of shared reasoning.',
        what_this_changes='Supplies original-only reference data for a separately declared one-prompt experiment; no training pair, native step, closure or post-closure evaluation is admitted.')
    save(run / 'result.json', result)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    run = Path(parser.parse_args().run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if declaration.get('operation') != 'single-original-turn-reference':
        raise ValueError('this compiler requires its own prospective original-only declaration')
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('all compilation products belong inside their declared run')
    if (run / 'original-turns').exists() or (run / 'result.json').exists():
        raise ValueError('an original compilation is not silently resumed or overwritten')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save(run / 'execution.json', dict(started_at=now(), operation=declaration['operation'],
                                     weight_updates=0, generated_answers=0))
    try:
        result = compile_original(run, declaration)
    except BaseException as error:
        save(run / 'interrupted.json', dict(at=now(), complete=False, error=type(error).__name__,
             message=str(error), partials='retained', weight_updates=0))
        raise
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
