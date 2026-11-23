"""Prospectively bound tracing for a future research invocation.

python -m pipeline.observability bindings vector-loop
python -m pipeline.observability run vector-loop /path/to/declared-run
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

from .recorder import FORMAT, Recorder

ENTRYPOINTS = {
    'vector-loop': 'pipeline.weight_edit.vector_loop',
    'vector-recovery': 'pipeline.weight_edit.vector_recovery',
    'vector-device-loop': 'pipeline.weight_edit.vector_device_loop',
    'vector-gradient-loop': 'pipeline.weight_edit.vector_gradient_loop',
    'vector-working-loop': 'pipeline.weight_edit.vector_working_loop',
    'vector-reference': 'pipeline.weight_edit.vector_chance_run',
    'turn-reference': 'pipeline.weight_edit.turn_reference_run',
}


def source_bindings(root):
    from .instrumentation import TARGETS
    paths = set(Path(__file__).parent.glob('*.py'))
    paths.update(root / 'code/signal-trace/pipeline/weight_edit' / (name + '.py')
                 for name, _, _, _ in TARGETS)
    paths.update(root / 'code/signal-trace' / (name.replace('.', '/') + '.py')
                 for name in ENTRYPOINTS.values())
    return [dict(record=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted(paths)]


def validate_spec(declaration, entrypoint, sources):
    spec = declaration.get('observability', {})
    if (spec.get('format') != FORMAT or spec.get('entrypoint') != entrypoint
            or spec.get('events') != 'observability/events.jsonl' or spec.get('sources') != sources):
        raise ValueError('tracing must be declared prospectively with its exact entrypoint and sources')
    declared = {item['record']: item['sha256'] for item in declaration['implementation']}
    if any(declared.get(item['record']) != item['sha256'] for item in sources):
        raise ValueError('tracing sources must also belong to the run implementation inventory')


def main():
    from pipeline.research_guard.verify_round2 import ROOT
    from pipeline.native_records.run_declaration import check_native_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('bindings', 'run'))
    parser.add_argument('target', choices=ENTRYPOINTS)
    parser.add_argument('run', nargs='?')
    args = parser.parse_args()
    module_name = ENTRYPOINTS[args.target]
    sources = source_bindings(ROOT)
    if args.command == 'bindings':
        print(json.dumps(dict(observability=dict(format=FORMAT, entrypoint=module_name,
              events='observability/events.jsonl', sources=sources), implementation_additions=sources), indent=2))
        return
    if not args.run:
        parser.error('run requires a declared run folder')
    run = Path(args.run).resolve()
    if not run.is_relative_to(ROOT / 'studies/04-round-2-overlap/runs'):
        raise ValueError('tracing output belongs inside the declared research run')
    path = run / 'declaration.json'
    declaration = check_native_run(run, path)
    validate_spec(declaration, module_name, sources)
    output = run / 'observability/events.jsonl'
    if output.exists():
        raise ValueError('an existing traced invocation is never overwritten or silently restarted')
    attrs = dict(project='weight-patterns', folder=run.name, operation=declaration.get('operation', ''),
                 declaration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 original_model_sha256=declaration.get('model_sha256', ''), entrypoint=module_name,
                 reference_sha256=declaration.get('reference', {}).get('sha256', ''))
    recorder = Recorder(output, attrs)
    original_argv = sys.argv
    status = 'FINISHED'
    try:
        from .instrumentation import instrument
        with instrument(recorder), recorder.span('research.invocation', attributes=attrs, detached=True):
            sys.argv = [module_name, str(run)]
            importlib.import_module(module_name).main()
    except BaseException as error:
        status = 'KILLED' if isinstance(error, KeyboardInterrupt) else 'FAILED'
        if isinstance(error, SystemExit) and error.code in (None, 0):
            status = 'FINISHED'
        raise
    finally:
        sys.argv = original_argv
        recorder.close(status)


if __name__ == '__main__':
    main()
