"""Read all original fitting functions against the fixed original agreement.

Only numerical evaluation is batched. Reference and recipient correspondences
remain their distinct, saved directional paths; their union is never an
alignment or support rule. A repeated function evaluation may be reused.
"""
import math
import time
from typing import Any

import numpy as np
import torch

from . import functional_projection


def current_events(members, paths, axis, recipient, domain):
    field = 'question' if domain == 'question' else 'steps'
    n, m = members[axis][field], members[recipient][field]
    if axis == recipient:
        pairs = [(row, row) for row in range(n)]
    else:
        raw = np.asarray(paths[recipient][axis][domain])
        if raw.ndim != 2 or raw.shape[1] != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError('current original correspondence needs complete integer pairs')
        # Saved pair: [successful donor axis row, recipient row].
        pairs = sorted(set((int(i), int(j)) for i, j in raw))
    if {i for i, _ in pairs} != set(range(n)) or {j for _, j in pairs} != set(range(m)):
        raise ValueError('both original and recipient domains must be complete')
    offset = members[recipient]['question'] if domain == 'answer' else 0
    grouped = [[] for _ in range(n)]
    for i, j in pairs:
        grouped[i].append(offset + j)
    return grouped


@torch.no_grad()
def measure_domain(source, plan, members, paths, axis, domain) -> dict[str, Any]:
    began = time.perf_counter()
    device = source.model.device
    field = 'question' if domain == 'question' else 'steps'
    count = members[axis][field]
    probes = [plan.probes(axis, domain, row) for row in range(count)]
    reference = [plan.witnesses(axis, domain, row, recipient=axis) for row in range(count)]
    current = {name: current_events(members, paths, axis, name, domain) for name in members}
    common = [torch.stack([source.query(p.case)[p.position] for p in bank]) for bank in probes]
    weight = [torch.tensor([p.denominator for p in bank], dtype=torch.float64).rsqrt()[:, None]
              for bank in probes]
    # Hold evaluated values on CPU while the actual-context local derivative
    # is computed one source at a time. No numerical value selects a row.
    values = {}
    function_rows = 0
    for name in members:
        rows, inputs, slices = [], [], {}
        start = 0
        for row in range(count):
            needed = set(current[name][row])
            needed.update(point.position for point in reference[row] if point.case == name)
            for position in sorted(needed):
                size = len(probes[row])
                rows.extend([position] * size)
                inputs.append(common[row])
                slices[row, position] = slice(start, start + size)
                start += size
        evaluated = source.evaluate(name, torch.tensor(rows, dtype=torch.long, device=device),
                                    torch.cat(inputs)).cpu()
        if not bool(torch.isfinite(evaluated).all()):
            raise ValueError('nonfinite original function makes the whole measurement unavailable')
        values[name] = {
            key: (evaluated[section].double() * weight[key[0]]).flatten()
            for key, section in slices.items()
        }
        function_rows += len(rows)
        del evaluated
    evaluated_at = time.perf_counter()
    normalization = plan.merges * len(plan.axes) * (members[axis]['question'] + members[axis]['steps'])
    cells = {key: np.zeros((len(members), count), dtype=np.float64) for key in ('distance', 'gap', 'bound', 'pairs')}
    event_functions: list[list[tuple[str, int]]] = [[] for _ in range(count)]
    for name, table in values.items():
        for row, position in table:
            event_functions[row].append((name, position))
    grouped = {}
    for row, functions in enumerate(event_functions):
        coordinate_count = len(values[functions[0][0]][row, functions[0][1]])
        grouped.setdefault((len(functions), coordinate_count, len(reference[row])), []).append(row)
    for selected in grouped.values():
        packed = torch.stack([torch.stack([values[name][row, position]
                              for name, position in event_functions[row]]) for row in selected]).to(device)
        indices = [{point: index for index, point in enumerate(event_functions[row])} for row in selected]
        original_refs = [[indices[index][p.case, p.position] for p in reference[row]]
                         for index, row in enumerate(selected)]
        shared = functional_projection.geometry(packed, original_refs)
        del packed
        for member_index, name in enumerate(members):
            buckets = {}
            for index, row in enumerate(selected):
                witnesses = [indices[index][p.case, p.position] for p in reference[row] if p.case != name]
                if not witnesses:
                    raise ValueError('an original event has no nonself reference functions')
                for position in current[name][row]:
                    buckets.setdefault(len(witnesses), []).append(
                        (index, row, witnesses, indices[index][name, position]))
            for batch in buckets.values():
                local = [item[0] for item in batch]
                rows = np.asarray([item[1] for item in batch], dtype=np.int64)
                refs, standing = [item[2] for item in batch], [item[3] for item in batch]
                found = functional_projection.nearest(shared, refs, standing, cells=local)
                mass = torch.tensor([len(current[name][row]) * normalization for row in rows],
                                    device=device, dtype=torch.float64)
                if not all(bool(torch.isfinite(found[key]).all()) for key in ('squared', 'gap', 'bound')):
                    raise ValueError('nonfinite projection makes the whole measurement unavailable')
                for key, field in (('distance', 'squared'), ('gap', 'gap'), ('bound', 'bound')):
                    contributions = found[field].clamp_min(0) / mass
                    np.add.at(cells[key][member_index], rows, contributions.cpu().numpy())
                np.add.at(cells['pairs'][member_index], rows, 1)
        del shared
    result = {name: dict(distance=math.fsum(cells['distance'][index]),
                        projection_gap=math.fsum(cells['gap'][index]),
                        projection_rounding_bound=math.fsum(cells['bound'][index]),
                        axis_events=count, pairs=int(cells['pairs'][index].sum()))
              for index, name in enumerate(members)}
    return dict(axis=axis, domain=domain, members=result, cells=cells, evaluated_probe_rows=function_rows,
                function_seconds=evaluated_at - began, projection_seconds=time.perf_counter() - evaluated_at,
                geometry='original-reference Gram and original/current cross products; no eigendecomposition')
