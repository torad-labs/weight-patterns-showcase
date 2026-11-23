"""Declared original-only conditional-function measurement; no weights are written."""
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import time
from zoneinfo import ZoneInfo
from typing import Any

import numpy as np
import torch

from .functional_baseline import measure_domain
from .functional_measure import OriginalFunctions
from .functional_reading import Reading, digest
from .functional_reference import checked, load_definition


def now():
    return datetime.now(ZoneInfo('America/Chicago')).isoformat()


def save(file, value):
    file.write_text(json.dumps(value, indent=2) + '\n')


def scan(source, plan, members, paths, output: Path) -> dict[str, Any]:
    totals = {name: {key: [] for key in ('distance', 'projection_gap', 'projection_rounding_bound')}
              for name in members}
    blocks = []
    began = time.perf_counter()
    evaluated = 0
    for merge in range(plan.merges):
        source.prepare(merge)
        for axis in plan.axes:
            for domain in ('question', 'answer'):
                start = time.perf_counter()
                measured = measure_domain(source, plan, members, paths, axis, domain)
                local = output / 'numerical' / str(merge) / f'{axis}-{domain}.npz'
                local.parent.mkdir(parents=True, exist_ok=True)
                if local.exists():
                    raise ValueError('a numerical measurement is never overwritten')
                np.savez_compressed(local, names=np.asarray(list(members)), **measured.pop('cells'))
                measured.update(merge=merge, seconds=time.perf_counter() - start,
                                local=dict(file=str(local.relative_to(output)), sha256=digest(local)))
                record = local.with_suffix('.json')
                save(record, measured)
                for name, values in measured['members'].items():
                    for key in totals[name]:
                        totals[name][key].append(values[key])
                evaluated += measured['evaluated_probe_rows']
                blocks.append(dict(file=str(record.relative_to(output)), sha256=digest(record)))
                save(output / 'progress.json', dict(status='measuring', at=now(), merge=merge,
                     axis=axis, domain=domain, blocks_complete=len(blocks),
                     blocks_required=plan.merges * len(plan.axes) * 2,
                     seconds=time.perf_counter() - began, evaluated_probe_rows=evaluated,
                     partial_sums_are_not_a_whole_trace_measure=True))
        print(json.dumps(dict(merge_complete=merge, blocks=len(blocks),
                              seconds=time.perf_counter() - began)), flush=True)
    result = {name: {key: math.fsum(values) for key, values in metrics.items()}
              for name, metrics in totals.items()}
    return dict(members=result, mean_distance=math.fsum(value['distance'] for value in result.values()) / len(result),
                seconds=time.perf_counter() - began, blocks=blocks,
                evaluated_probe_rows=evaluated, complete_represented_scope=True,
                original_axis_event_merges=plan.statistics()['axis_merge_events'],
                recipient_axis_event_merges=plan.statistics()['axis_merge_events'] * len(members))


def main():
    import argparse
    from pipeline.native_records.run_declaration import check_native_run
    from pipeline.research_guard.verify_round2 import ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    args = parser.parse_args()
    run = Path(args.run).resolve()
    declaration = check_native_run(run, run / 'declaration.json')
    if declaration.get('operation') != 'original-conditional-function-baseline':
        raise ValueError('a complete original functional-baseline declaration is required')
    if (run / 'result.json').exists() or (run / 'progress.json').exists():
        raise ValueError('an original measurement runs once; a partial is not silently resumed')
    for binding in declaration['implementation']:
        checked(ROOT, binding, parse=False)
    definition = checked(ROOT, declaration['definition'])
    plan = load_definition(ROOT, definition)
    sources = definition['sources']
    spec = importlib.util.spec_from_file_location('bound_functional_torch_forward', ROOT / declaration['forward']['record'])
    if spec is None or spec.loader is None:
        raise ValueError('the declared Torch model cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = module.Model(ROOT / sources['model']['record'],
                         checked(ROOT, sources['tensors']), checked(ROOT, sources['architecture']),
                         device='cuda', cache=declaration['cache_precision'], leaves=lambda name: False)
    reader = Reading((ROOT / sources['reading']['record']).parent, model, definition['prompts'],
                     expected_reading_sha256=sources['reading']['sha256'],
                     expected_model_sha256=sources['model']['sha256'])
    source = OriginalFunctions(reader)
    paths = checked(ROOT, sources['paths'])
    identity = dict(started_at=now(), model_sha256=sources['model']['sha256'],
                    definition=declaration['definition'], declaration_sha256=digest(run / 'declaration.json'),
                    torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                    device=torch.cuda.get_device_name(model.device), cache=model.cache,
                    original_native_card=reader.record.get('card'),
                    acquisitions_at_limit=source.acquisition_limits,
                    whole_generation_scope=not source.acquisition_limits,
                    arithmetic='float32 conditional query actions; float64 joint projection; TF32 disabled',
                    context='reconstructed keys/values with bound float16 cache rounding, not captured cache bytes',
                    scope='native-observed intercepts and declared Torch conditional input derivatives; not native Jacobians')
    save(run / 'execution.json', identity)
    try:
        measured = scan(source, plan, definition['members'], paths, run)
    except BaseException as error:
        save(run / 'interrupted.json', dict(at=now(), error=type(error).__name__, message=str(error),
             complete=False, closure=False, partials='retained; no full-trace result'))
        raise
    groups = {}
    for group in sorted({item['group'] for item in definition['members'].values()}):
        selected = [measured['members'][name] for name, item in definition['members'].items() if item['group'] == group]
        groups[group] = dict(members=len(selected), mean_distance=math.fsum(item['distance'] for item in selected) / len(selected))
    measured.update(format='conditional-function-original-baseline/1', completed_at=now(), groups=groups,
                    execution=identity, closure=False, native_jacobian_qualified=False,
                    peak_device_bytes=torch.cuda.max_memory_allocated(model.device),
                    prediction_vs_result='Complete original fitting conditional-function measurement; inspect per-member, per-event and projection-uncertainty records. No edited model was evaluated.',
                    failures_only={name: value for name, value in measured['members'].items()
                                   if definition['members'][name]['group'] == 'wrong'},
                    what_this_changes='The original first-order actual-context function reference and baseline have been numerically evaluated. This measures the explicit construction, not global function equality, capability repair or closure.')
    save(run / 'result.json', measured)
    print(json.dumps({key: measured[key] for key in ('completed_at', 'mean_distance', 'groups', 'seconds', 'peak_device_bytes')}, indent=2))


if __name__ == '__main__':
    main()
