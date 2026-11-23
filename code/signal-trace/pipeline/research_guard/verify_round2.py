"""Round-two drift verifier. Exit 1 with every violation if the work leaves the method.

Run: python3 code/signal-trace/pipeline/research_guard/verify_round2.py
It runs in the pre-commit hook and before any native run. The invariants live in
studies/04-round-2-overlap/round-2-method.json.
"""
import json
import re
import sys
from pathlib import Path

try:                                # a member of the pipeline package, or a script beside its sibling
    from pipeline.research_guard import forbidden_bytes, input_archives, recorded_losses
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import forbidden_bytes
    import input_archives
    import recorded_losses

ROOT = Path(__file__).resolve().parents[4]
METHOD = ROOT / 'studies/04-round-2-overlap/round-2-method.json'
SELF = {Path(__file__).resolve(), METHOD.resolve()}
PARAMETER_FIELDS = ('measured_from', 'derivation', 'scales_with')


def steps_of(declaration):
    value = (declaration or {}).get('method_step')
    values = value if isinstance(value, list) else [value]
    return {int(v) for v in values if str(v).isdigit()}


def load_runs(root, method):
    runs = {}
    base = root / method['runs_root']
    if not base.exists():
        return runs
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        declaration, result = folder / 'declaration.json', folder / 'result.json'
        runs[folder.name] = dict(
            folder=folder,
            declaration=json.loads(declaration.read_text()) if declaration.exists() else None,
            result=json.loads(result.read_text()) if result.exists() else None)
    return runs




def check_run(root, name, run, runs, method, problems, archives=(), lost=None):
    d, folder = run['declaration'], run['folder']
    if d is None:
        problems.append(f'{name}: no declaration.json (the run record is missing)')
        return
    for field in method['declaration_fields']:
        if not d.get(field):
            problems.append(f'{name}: declaration lacks {field}')
    steps = steps_of(d)
    if not steps or not steps <= set(range(1, 7)):
        problems.append(f'{name}: method_step must be among 1-6 (see round-2-method.json)')
    shared = (root / method['read_only_inputs']).resolve()
    base = (root / method['runs_root']).resolve()
    model = method.get('read_only_model')
    for item in d.get('inputs', []):
        if not isinstance(item, str) or not item:
            problems.append(f'{name}: input {item!r} is not a path')
            continue
        path = (root / item).resolve()
        # A verified archive record may stand for a file that is not here (input_archives); where the input comes from is checked all the same.
        archived = input_archives.historical(root, path, archives)
        problems.extend(f'{name}: input {item}: {problem}' for problem in archived or [])
        # A recorded loss (recorded_losses) excuses the absence of that very path, to the declaration the list names for it as the bytes it
        # hashed, and nothing else: where the input comes from is checked below.
        if archived is None and not path.exists() and not recorded_losses.declared(lost, item, name, folder / 'declaration.json'):
            problems.append(f'{name}: input {item} does not exist' + (recorded_losses.NOT_THIS_DECLARATION if item in (lost or {}) else ''))
        elif path == base:
            problems.append(f'{name}: input {item} is the whole runs folder, not a run or a file of one')
        elif path.is_relative_to(base):
            source = path.relative_to(base).parts[0]
            if source != name and source not in runs:
                problems.append(f'{name}: input {item} comes from {source}, which is not a run')
        elif model and path == (root / model['path']).resolve():
            if d.get('model_sha256') != model['sha256']:
                problems.append(f'{name}: reading the original model requires model_sha256 = {model["sha256"]}')
        elif not path.is_relative_to(shared):
            problems.append(f'{name}: input {item} is outside the read-only inputs, the original model and round-2 runs')
    for item in d.get('outputs', []):
        if not (root / item).resolve().is_relative_to(folder.resolve()):
            problems.append(f'{name}: output {item} is outside its own folder')
    if steps & {5, 6}:
        target = d.get('target_source')
        path = (root / target).resolve() if isinstance(target, str) and target else None      # a target that is no path is no target
        source = path.relative_to(base).parts[0] if path and path != base and path.is_relative_to(base) else None
        archived = input_archives.historical(root, path, archives) if path else None
        problems.extend(f'{name}: target_source {target}: {problem}' for problem in archived or [])
        if path is None or (archived is None and not path.exists()) or source not in runs \
                or not ({3, 4} & steps_of(runs[source]['declaration'])):
            problems.append(f'{name}: edit steps need target_source inside a step-3/4 overlap run')
    for key, value in (d.get('parameters') or {}).items():
        if not isinstance(value, dict):
            problems.append(f'{name}: parameter {key} is a bare number; give {", ".join(PARAMETER_FIELDS)}')
            continue
        for field in PARAMETER_FIELDS:
            if not value.get(field):
                problems.append(f'{name}: parameter {key} lacks {field}')
        cited = value.get('measured_from')
        if cited and not (root / cited).exists():
            problems.append(f'{name}: parameter {key} cites a missing record {cited}')
    if run['result'] is not None:
        for field in method['result_fields']:
            if field not in run['result']:
                problems.append(f'{name}: result.json lacks {field}')


def check_forbidden(root, method, problems):
    pattern = re.compile('|'.join(re.escape(s) for s in method['forbidden_references']))
    table = forbidden_bytes.needles(method['forbidden_references'])
    for place in (root / 'code/signal-trace/pipeline', root / method['runs_root']):
        if not place.exists():
            continue
        for path in place.rglob('*'):
            if not path.is_file() or path.suffix not in {'.py', '.json', '.md'} or path.resolve() in SELF:
                continue
            if path.stat().st_size > 2_000_000:
                continue
            if table is not None and forbidden_bytes.without_a_reference(path.read_bytes(), table):
                continue
            hit = pattern.search(path.read_text(errors='ignore'))
            if hit:
                problems.append(f'{path.relative_to(root)}: references deleted first-attempt work ({hit.group()})')


def check_status_page(root, method, runs, problems):
    page = (root / method['status_page']).read_text()
    for name in runs:
        if name not in page:
            problems.append(f'{method["status_page"]}: does not mention run {name}; update the status page')


def verify(root=ROOT):
    method = json.loads((root / 'studies/04-round-2-overlap/round-2-method.json').read_text())
    problems = []
    runs = load_runs(root, method)
    lost, unlisted = recorded_losses.load(root, method)
    problems.extend(unlisted)
    archives, unsound = input_archives.load(root, method, runs, lost)
    problems.extend(unsound)
    for name, run in runs.items():
        check_run(root, name, run, runs, method, problems, archives, lost)
    check_forbidden(root, method, problems)
    check_status_page(root, method, runs, problems)
    return problems


def main():
    problems = verify()
    for p in problems:
        print('round-2 drift:', p, file=sys.stderr)
    if problems:
        print(f'round-2 drift: {len(problems)} violation(s). Read ROUND-2.md and round-2-method.json.', file=sys.stderr)
        sys.exit(1)
    print('round-2 verifier: no drift')


if __name__ == '__main__':
    main()
