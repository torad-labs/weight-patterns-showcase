"""Freeze original coordinates and sources, not a numerical function bank."""
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from .functional_layout import Plan
from .functional_reading import digest, identity, tokens_hash


def checked(root: Path, binding: Mapping[str, str], *, parse=True) -> Any:
    relative = Path(binding['record'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('a source binding must remain inside its declared root')
    file = root
    for part in relative.parts:
        file = file / part
        if file.is_symlink():
            raise ValueError('a source binding cannot follow a link')
    before = identity(file)
    if digest(file) != binding['sha256'] or identity(file) != before:
        raise ValueError(f"source sha256 differs: {relative}")
    if not parse:
        return None
    value = json.loads(file.read_text())
    if identity(file) != before:
        raise ValueError(f"source changed after sha256 verification: {relative}")
    return value


def compile_definition(root, sources: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    root = Path(root)
    checked(root, sources['model'], parse=False)
    reading = checked(root, sources['reading'])
    rows = checked(root, sources['row_map'])
    membership = checked(root, sources['original_membership'])
    paths = checked(root, sources['paths'])
    architecture = checked(root, sources['architecture'])
    checked(root, sources['tensors'])
    if (reading.get('format') != 'whole-gap-reading/1'
            or reading['model_sha256'] != sources['model']['sha256']
            or membership['original_model_sha256'] != sources['model']['sha256']
            or membership['original_reading']['sha256'] != sources['reading']['sha256']
            or membership['paths']['sha256'] != sources['paths']['sha256']
            or reading['row_map']['sha256'] != sources['row_map']['sha256']):
        raise ValueError('the original model, reading, membership and paths must share their bound sources')
    table = {item['id']: item for item in reading['table']}
    entries = {item['id']: item for item in rows['answers']}
    if (len(table) != len(reading['table']) or len(entries) != len(rows['answers'])
            or set(table) != set(entries) or len(reading['answers_read']) != len(table)
            or set(reading['answers_read']) != set(table)):
        raise ValueError('the complete original fitting inventory is missing or duplicated')
    if any(item['group'] not in ('correct', 'wrong') for item in table.values()):
        raise ValueError('original group labels are not the bound correct/wrong inventory')
    successes = list(membership['donors'])
    if (len(set(successes)) != len(successes)
            or set(successes) != {name for name, item in table.items() if item['group'] == 'correct'}):
        raise ValueError('the probes must contain every original fitting success exactly once')
    members, prompts = {}, {}
    for name, item in entries.items():
        members[name] = dict(question=item['question'], steps=item['steps'], group=table[name]['group'])
        prompt = checked(root, sources['prompt:' + name])['tokens']
        if len(prompt) != item['question'] + 1 or tokens_hash(prompt) != item['prompt_sha256']:
            raise ValueError('an original prompt differs from its bound acquisition')
        prompts[name] = prompt
    merges = len(reading['merges'])
    if ([item['merge'] for item in reading['merges']] != list(range(merges))
            or merges != 3 * architecture['layers'] or reading['width'] != architecture['width']
            or sum(item['question'] + item['steps'] for item in members.values()) != rows['rows']
            or rows['rows'] != reading['rows']):
        raise ValueError('original source dimensions do not cover every event and merge')
    # The saved paths are directional: recipient -> successful donor. The
    # success-by-success portion defines probes. No reverse-path union is made.
    original = {name: members[name] for name in successes}
    plan = Plan(original, successes, paths, merges=merges)
    return dict(format='conditional-function-layout/1',
                role='fixed-original-coordinate-definition-not-a-numerical-bank',
                numerical_functions_materialized=False,
                sources=copy.deepcopy(dict(sources)), members=members, successes=successes,
                prompts=prompts, merges=merges, width=reading['width'],
                counts=plan.statistics(),
                domain='native-sized query vectors, conditioned on actual untouched context',
                probe_mass='1/(original successes * distinct mapped events from that success)',
                reference='joint convex hull of raw functions; exclude axis and recipient identities',
                correspondence='saved directional original computational paths; no symmetric union')


def load_definition(root, definition: Mapping[str, Any]) -> Plan:
    rebuilt = compile_definition(root, definition['sources'])
    if rebuilt != definition:
        raise ValueError('the definition differs from its complete bound original sources')
    paths = checked(Path(root), definition['sources']['paths'])
    members = {name: definition['members'][name] for name in definition['successes']}
    return Plan(members, definition['successes'], paths, merges=definition['merges'])


def main():
    import argparse
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from pipeline.native_records.run_declaration import check_native_run
    from pipeline.research_guard.verify_round2 import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    run = Path(args.run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if declaration.get('operation') != 'original-conditional-function-layout':
        raise ValueError('this command requires its own original-coordinate declaration')
    for binding in declaration['implementation']:
        checked(ROOT, binding, parse=False)
    output = ROOT / declaration['definition_output']
    if output.parent.resolve() != run or output.exists():
        raise ValueError('the original definition is written once inside its own run')
    made = compile_definition(ROOT, declaration['sources'])
    output.write_text(json.dumps(made, indent=2) + '\n')
    result = dict(
        format='conditional-function-layout-result/1',
        completed_at=datetime.now(ZoneInfo('America/Chicago')).isoformat(),
        declaration_sha256=digest(run / 'declaration.json'),
        definition=dict(file=output.name, sha256=digest(output)),
        fitting_members=len(made['members']), successes=len(made['successes']),
        merges=made['merges'], counts=made['counts'],
        numerical_functions_materialized=False,
        prediction_vs_result='All original successful axes, raw mapped query/witness events and acquisition sources are bound; no numerical function has been evaluated.',
        failures_only=[],
        what_this_changes='The conditional-function construction has an immutable original coordinate definition. This is neither a numerical bank nor native qualification, discrimination, repair or closure.')
    (run / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
