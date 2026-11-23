"""Views of the saved computational comparison; no model run or token judgment.

The worker reads each bound native merge once and the one fixed reference field.
Only scalar distance, radius and loss-contribution fields are kept. The exact
saved correspondence, canonical turns and existing residual implementation hold.
"""
import datetime
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from wp_training_models import record_generation, record_identity


SCHEMA = 'computational-overlap-view/1'
NUMERICAL_PACKAGES = ('torch==2.15.0', 'numpy==2.5.5')
GRADIENT_PHASES = {
    'weight_derivative_started': 'computing complete weight derivative',
    'weight_derivative_complete': 'weight derivative complete; measuring response',
    'directional_response_complete': 'directional response complete; rounding BF16 weights',
    'native_rounding_complete': 'BF16 rounding complete; native write pending',
}
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
AXES = '''How to read these plots

OVERLAP HEATMAP
X: saved computational correspondence pair index. Each pair names an original
reference event and a current native event; repeated matches are preserved.
Y: merge index in model order. Each layer has attention, feedforward, and
per-layer-input merges. This is not an alignment of words or token positions.
Color: max(distance - original tolerance radius, 0), dimensionless. White means
inside tolerance; positive excess uses a logarithmic color scale so small and
large divergences remain visible. Every pair/merge is shown. A turn is
unit(output) - unit(input) at a merge, with all vector coordinates retained.

ACTUAL COMPUTATION OVERLAY
X: every coordinate of the actual turn vector, in its recorded model basis.
Y: unit(output) - unit(input), dimensionless. The blue and orange curves are the
fixed correct and current computations, respectively. Shading marks coordinate
differences; tolerance applies to the whole vector distance, not each coordinate.
The selected merge has the largest total native loss contribution; within it the
selected correspondence pair has the largest contribution. Its location is marked
on the full divergence map. Both event indices come from saved computational
correspondence, never words. This is one exact-coordinate detail, NOT the whole
trace, PCA or a scalar distance posing as two traces. Raster display resolution may
hide small differences; overlap.json retains every plotted value. Selection and
rendering do not define or change the full-vector computational loss.

NATIVE PROGRESS
X: the recorded completed-update generation; legacy observation numbering uses
native observation step minus one. RAM continuations begin at local generation 0.
Zero is this run's starting state, which may retain updates from a predecessor.
Y: Computational loss (canonical metric: native.objective), the full native sum
of squared weighted excess residuals, dimensionless; zero
means no excess under this computational rule. The separate outside-cell plot
counts correspondence-pair/merge cells. Only recorded native observations appear.
A retained initial observation is not fresh inference by this attempt.

INNER SOLVER PROGRESS
X: observed accepted inner inverse iteration, within this one outer step.
Y: the solver's linearized sum of squared residuals, dimensionless. Lower predicts
less discrepancy in the linear approximation. It is not a fresh native result,
an installed weight update, or computational closure. Overwritten/missed samples
are unknown. Refresh the artifact to see the latest published snapshot.

DIRECT GRADIENT STEP
For response-scaled weight-gradient runs the third panel shows measured operation
times instead of inverse iterations. X: derivative, directional response, BF16
rounding. Y: wall time in seconds. The displayed gradient norm, response scale and
linear loss prediction describe one proposal, not its fresh native effect.
gradient/* metric charts use the native observation number as Step. The next
native observation supplies the measured effect after installation/regeneration.

In Model metrics, search native.objective for Computational loss. It is the
canonical metric; existing training/native_discrepancy remains compatible.
For the ordinary MLflow metric charts, choose Step on the horizontal axis:
training/native_discrepancy and training/outside_cells use native observation
number (one before any update); solver/linear_objective uses accepted inner
iteration. The labeled plots translate observation number into completed updates.
Elapsed-time metrics use seconds. Resource charts describe the host and both GPUs.

BEFORE / AFTER
The first complete recorded computation is the baseline. The after panel remains
empty until a later native observation exists. Each panel retains that
observation's own complete correspondence; equal pair indices across panels need
not identify identical event pairs. Both panels use the same color scale.

RAM / RAW-STREAM OBSERVATIONS
The full map shows the saved weighted residual-squared contribution, aggregated
by original reference event and merge. It is not a distance or tolerance radius.
Every mapped current event contributes, and repeated correspondence is retained.
White means zero contribution. The selected vector pair is one detail inside an
aggregate cell: it is not the entire cell or an alternate learning criterion.
The canonical native objective, map sum and recorded aggregation roundoff stay
separate. Only the selected cell is numerically replayed. Small-map and index
hashes are checked; full raw/reference digests are producer provenance, not fresh
whole-file audits. Selected bytes have stat continuity and their own hashes.
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path, expected):
    """One sequential read; release file-cache pages after retaining these bytes."""
    path = Path(path)
    before = path.stat()
    with path.open('rb') as source:
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
        data = source.read()
        if hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(source.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('comparison source changed while being read')
    if sha(data) != expected:
        raise ValueError(f'comparison binding changed: {path.name}')
    return data


def binding_path(root, binding):
    path = (root / binding['record']).resolve()
    if not path.is_relative_to(root / 'studies/04-round-2-overlap/runs'):
        raise ValueError('comparison metadata must belong to a named run')
    return path


def bound_json(root, binding):
    return json.loads(checked_bytes(binding_path(root, binding), binding['sha256']))


def array(path, expected):
    import numpy as np
    data = checked_bytes(path, expected)
    header = io.BytesIO(data)
    version = np.lib.format.read_magic(header)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(header)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(header)
    else:
        raise ValueError('unsupported comparison array format')
    if dtype.hasobject:
        raise ValueError('object arrays are not computational records')
    return np.frombuffer(data, dtype=dtype, offset=header.tell()).reshape(shape, order='F' if fortran else 'C')


def causal_rows(row_map, member, rows):
    import numpy as np
    if row_map['order'] != 'all_answers_then_all_questions' or row_map['rows'] != rows:
        raise ValueError('native reading row order changed')
    entries = [x for x in row_map['answers'] if x['id'] == member]
    if len(entries) != 1:
        raise ValueError('native comparison needs its one declared member')
    entry = entries[0]
    q, a = entry['question_rows_from'], entry['answer_rows_from']
    order = np.r_[np.arange(q, q + entry['question']), np.arange(a, a + entry['steps'])]
    if len(order) != rows or not np.array_equal(np.sort(order), np.arange(rows)):
        raise ValueError('comparison must retain every native row exactly once')
    return order


def numerical_inputs(root, declaration, reading, folder, member, rows, width):
    """Recorded IDs are embedding addresses only; never decoded, scored or shown."""
    import numpy as np
    import torch
    definition = bound_json(root, declaration['source_definition'])
    tensors = bound_json(root, declaration['tensors'])
    table = next(x for x in reading['table'] if x['id'] == member)
    path = folder / 'answers' / member / 'answer.json'
    acquisition = json.loads(checked_bytes(path, table['answer_json_sha256']))
    history = definition['prompts'][member] + acquisition['tokens'][:-1]
    if len(history) != rows:
        raise ValueError('numerical input addresses differ from the recorded event inventory')
    edits = declaration['functional_loop']['matrices']
    if not edits or any(not name.endswith(('.attn_output.weight', '.ffn_down.weight')) for name in edits):
        raise ValueError('original embedding/scales cannot represent this edit inventory')
    model = root / declaration['original_model']['record']
    before = model.stat()
    embedding = tensors['token_embd.weight']
    if embedding['native_type'] != 'bf16' or embedding['shape'][0] != width:
        raise ValueError('numerical input table has another format')
    with model.open('rb') as source:
        def read_at(offset, size):
            raw = os.pread(source.fileno(), size, offset)
            if len(raw) != size:
                raise ValueError('numerical input component is truncated')
            return raw
        unique = {}
        for token in sorted(set(history)):
            if not 0 <= token < embedding['shape'][1]:
                raise ValueError('numerical input address leaves its bound table')
            unique[token] = read_at(embedding['offset_bytes'] + token * width * 2, width * 2)
        bits = np.frombuffer(b''.join(unique[token] for token in history), dtype='<u2').reshape(rows, width)
        incoming = torch.from_numpy((bits.astype(np.uint32) << 16).view(np.float32).copy()) * math.sqrt(width)
        scales = []
        for layer in range(len(reading['merges']) // 3 - 1):
            field = tensors[f'blk.{layer}.layer_output_scale.weight']
            if field['native_type'] != 'f32' or field['bytes'] != 4:
                raise ValueError('native layer scale is not a scalar float32')
            raw = read_at(field['offset_bytes'], 4)
            scales.append(torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy()))
    after = model.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError('original numerical inputs changed while being read')
    return incoming, scales, dict(
        model_identity='declaration-bound original; no fresh full-file hash',
        embedding_bytes_read=len(unique) * width * 2, scalar_bytes_read=len(scales) * 4,
        embedding_rows_sha256=sha(b''.join(unique[k] for k in sorted(unique))))


def projection_sources(root):
    """Numerical identity is independent of changes to charts and publication."""
    paths = ('weight_edit/turn_pair.py', 'weight_edit/vector_chance.py',
             'weight_edit/reference_rule.py', 'overlap/complete_overlap.py')
    return dict(numerical_functions_sha256=NUMERICAL_SOURCE_SHA256,
                numerical_packages=list(NUMERICAL_PACKAGES),
                research_sources={f'code/signal-trace/pipeline/{p}': sha((Path(root) / 'code/signal-trace/pipeline' / p).read_bytes())
                                  for p in paths})


def file_identity(status):
    return [status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns]


def partial_vector(path, shape, event, merge):
    """Read one NPY vector; its full digest remains producer provenance."""
    import numpy as np
    with Path(path).open('rb') as source:
        before = file_identity(os.fstat(source.fileno()))
        version = np.lib.format.read_magic(source)
        if version != (1, 0):
            raise ValueError('selected reference vector needs the recorded NPY format')
        stored_shape, fortran, dtype = np.lib.format.read_array_header_1_0(source)
        header = source.tell()
        if (tuple(shape) != stored_shape or fortran or dtype != np.dtype('<f8')
                or before[2] != header + math.prod(shape) * dtype.itemsize):
            raise ValueError('selected reference vector has another shape or storage')
        offset = header + (event * shape[1] + merge) * shape[2] * dtype.itemsize
        raw = os.pread(source.fileno(), shape[2] * dtype.itemsize, offset)
        if len(raw) != shape[2] * dtype.itemsize or before != file_identity(os.fstat(source.fileno())):
            raise ValueError('selected reference vector changed while being read')
    if before != file_identity(Path(path).stat()):
        raise ValueError('selected reference vector path changed')
    return np.frombuffer(raw, dtype=dtype).copy(), dict(file_identity=before, offset_bytes=offset,
        bytes_read=len(raw), selected_bytes_sha256=sha(raw), whole_file_hash_checked=False)


def raw_edges(root, capture, merge, positions, rows, width, merges):
    """Validate the bound full index, then read only selected native ADD rows."""
    import numpy as np
    nodes = []
    for layer in range(merges // 3):
        nodes.extend([
            (f'inp_scaled' if layer == 0 else f'l_out-{layer-1}', f'attn_post_norm-{layer}', f'attn_out-{layer}'),
            (f'attn_out-{layer}', f'ffn_post_norm-{layer}', f'pe_in-{layer}'),
            (f'pe_in-{layer}', f'per_layer_embd_out-{layer}', f'pe_out-{layer}'),
        ])
    if len(nodes) != merges:
        raise ValueError('native stream must retain three merges per layer')
    labels = {name for node in nodes for name in node}
    coverage = {name: np.zeros(rows, dtype=bool) for name in labels}
    expected_add = {after: sorted([before, branch]) for before, branch, after in nodes}
    selected = nodes[merge]
    locations = {name: {} for name in selected}
    index_binding, stream_binding = capture['raw']['stream.jsonl'], capture['raw']['stream.f32']
    stream = binding_path(root, stream_binding)
    index = checked_bytes(binding_path(root, index_binding), index_binding['sha256'])
    with stream.open('rb') as source:
        before = file_identity(os.fstat(source.fileno()))
        for line in index.splitlines():
            record = json.loads(line)
            label = record['label']
            if label not in coverage or not record['values']:
                continue
            shape, at = record['shape'], record['positions']
            offset, values = record['offset_bytes'], record['values']
            if (record['native_type'] != 'f32' or record['token_axis'] != 1
                    or len(shape) < 2 or shape[0] != width or math.prod(shape[2:]) != 1
                    or len(at) != shape[1] or values != width * len(at) or offset < 0 or offset % 4
                    or offset + values * 4 > before[2] or len(set(at)) != len(at)
                    or any(type(p) is not int or p < 0 or p >= rows for p in at)
                    or bool(coverage[label][at].any())):
                raise ValueError('raw computational row inventory differs')
            if label in expected_add and (record['operation'] != 'ADD'
                    or sorted(record['source_names']) != expected_add[label]):
                raise ValueError('raw computational connection is not its declared ADD')
            coverage[label][at] = True
            if label in locations:
                for i, event in enumerate(at):
                    if event in positions:
                        locations[label][event] = offset + i * width * 4
        if not all(bool(found.all()) for found in coverage.values()):
            raise ValueError('raw index does not cover every event at every merge')
        arrays, pieces = [], []
        for label in selected:
            values = []
            for event in positions:
                offset = locations[label][event]
                raw = os.pread(source.fileno(), width * 4, offset)
                if len(raw) != width * 4:
                    raise ValueError('selected native edge is truncated')
                values.append(np.frombuffer(raw, dtype='<f4').copy())
                pieces.append(dict(label=label, event=event, offset_bytes=offset,
                                   bytes_read=len(raw), selected_bytes_sha256=sha(raw)))
            arrays.append(np.stack(values))
        if before != file_identity(os.fstat(source.fileno())):
            raise ValueError('selected native edges changed while being read')
    if before != file_identity(stream.stat()):
        raise ValueError('selected native stream path changed')
    incoming, branch, merged = arrays
    if (not all(bool(np.isfinite(x).all()) for x in arrays)
            or not np.array_equal(np.add(incoming, branch).view('u4'), merged.view('u4'))):
        raise ValueError('selected native ADD does not reproduce its recorded output')
    return incoming, merged, dict(stream=stream_binding, index=index_binding,
        file_identity=before, whole_file_hash_checked=False, selected_chunks=pieces,
        bytes_read=sum(x['bytes_read'] for x in pieces), full_index_hash_checked=True)


def raw_project(root, declaration, measured, reference):
    """Bound loss map plus one actual vector detail, without a bulk trace replay."""
    import numpy as np
    import torch
    from pipeline.overlap.complete_overlap import validate_pairs
    from pipeline.weight_edit.turn_pair import turns
    from pipeline.weight_edit.vector_chance import outside
    revision = record_identity(measured)
    if (revision['kind'] != 'native-working-revision/1'
            or revision['original_sha256'] != declaration['original_model']['sha256']):
        raise ValueError('raw observation must identify its declared working revision')
    generation = record_generation(measured)
    capture = bound_json(root, measured['capture'])
    recovered = None
    if generation == 0 and measured.get('recovered_from'):
        if measured['recovered_from'] != declaration.get('start_fresh'):
            raise ValueError('raw baseline differs from the declared retained acquisition')
        recovered = bound_json(root, measured['recovered_from'])
        start = bound_json(root, declaration['start_experiment'])
        if (capture['model_identity'] != recovered['model_identity']
                or any(recovered[k] != measured[k] for k in ('capture', 'correspondence', 'reference'))):
            raise ValueError('retained baseline capture identity changed')
        if capture['model_identity'].get('kind') == 'native-working-revision/1':
            candidate = bound_json(root, start['candidate'])
            if (candidate['pending_install'] is not None or candidate['weight_state'] != 'complete'
                    or candidate['last_measurement'] != measured['recovered_from']
                    or candidate['model_identity'] != recovered['model_identity']
                    or candidate['update'] != recovered['generation']
                    or not recovered['complete'] or not recovered['ended']):
                raise ValueError('retained checkpoint and native observation are not the same completed state')
        elif capture['model_identity'].get('candidate') != start['candidate']:
            raise ValueError('retained baseline candidate identity changed')
        saved = recovered.get('measurement', recovered)
        if (measured['saved_native_objective'] != saved['objective']
                or measured['scalar_reduction_difference'] != measured['objective'] - saved['objective']
                or any(saved[k] != measured[k] for k in ('outside', 'pair_merge_cells'))
                or recovered['residual_squared']['sha256'] != measured['residual_squared']['sha256']):
            raise ValueError('retained baseline measurement or contribution map changed')
    elif capture['model_identity'] != revision:
        raise ValueError('raw capture and measurement identify different revisions')
    shape = reference['trace']['shape']
    rows, merges, width = shape
    if merges != measured['merges'] or merges <= 0:
        raise ValueError('raw comparison merge inventory differs')
    contribution = array(binding_path(root, measured['residual_squared']), measured['residual_squared']['sha256'])
    chance = array(binding_path(root, reference['chance']), reference['chance']['sha256'])
    pairs = array(binding_path(root, measured['correspondence']), measured['correspondence']['sha256'])
    if (contribution.dtype != np.dtype('<f8') or contribution.shape != (rows, merges)
            or list(contribution.shape) != measured['residual_squared']['shape']
            or chance.dtype != np.dtype('<f8') or chance.shape != contribution.shape
            or not bool(np.isfinite(contribution).all()) or bool((contribution < 0).any())):
        raise ValueError('bound computational contribution map has another shape or value domain')
    validate_pairs(pairs, rows, measured['events'])
    if len(pairs) * merges != measured['pair_merge_cells']:
        raise ValueError('raw map correspondence inventory differs')
    total = float(contribution.sum())
    if total != measured['residual_squared']['sum']:
        raise ValueError('bound computational contribution sum differs')
    roundoff = total - measured['objective']
    if recovered is None and roundoff != measured['map_aggregation_roundoff']:
        raise ValueError('map aggregation differs from the recorded reduction roundoff')
    merge_totals = contribution.sum(axis=0)
    selected = int(merge_totals.argmax())
    event = int(contribution[:, selected].argmax())
    indices = np.flatnonzero(pairs[:, 0] == event)
    positions = pairs[indices, 1].tolist()
    correct, reference_proof = partial_vector(binding_path(root, reference['trace']), shape, event, selected)
    incoming, merged, raw_proof = raw_edges(root, capture, selected, positions, measured['events'], width, merges)
    with torch.no_grad():
        turn = turns({'incoming': [torch.from_numpy(incoming)], 'merged': [torch.from_numpy(merged)]}, expected_merges=1)[0]
        delta = turn - torch.from_numpy(correct)
        mass = 1. / (rows * merges * torch.full((len(indices),), len(indices), dtype=torch.float64))
        bound = torch.full((len(indices),), float(chance[event, selected]), dtype=torch.float64)
        observed = outside(delta, bound, mass)
        losses = observed.residual.double().square().sum(dim=1).numpy()
        cell = np.zeros(1, dtype='<f8')
        np.add.at(cell, np.zeros(len(indices), dtype=int), losses)
        if float(cell[0]) != float(contribution[event, selected]):
            raise ValueError('selected raw turn vectors do not reproduce their bound aggregate loss cell')
        chosen = int(losses.argmax())
        detail = dict(merge=selected, pair=int(indices[chosen]), reference_event=event,
            current_event=int(positions[chosen]), merge_loss=float(merge_totals[selected]),
            aggregate_reference_loss=float(contribution[event, selected]), weighted_loss=float(losses[chosen]),
            distance=float(torch.linalg.vector_norm(delta[chosen])), radius=float((2 * bound[chosen]).sqrt()),
            reference_turn=correct.tolist(), current_turn=turn[chosen].numpy().tolist(),
            selection='Largest total-loss merge, then largest aggregated reference cell, then largest contributing pair in that cell; display detail only')
    return dict(model_sha256=None, model_identity=revision, generation=generation,
        source_capture=measured['capture'], source_contribution=measured['residual_squared'],
        recovered_from=measured.get('recovered_from'), pairs=pairs.tolist(),
        map_kind='reference-weighted-residual-squared', contribution=contribution.T.tolist(),
        merge_labels=[f'Layer {i//3} · {("attention", "feedforward", "layer")[i%3]} · merge {i}' for i in range(merges)],
        merge_loss=merge_totals.tolist(), coordinate_detail=detail,
        map_sum=total, map_aggregation_roundoff=roundoff, objective_reproduced_exactly=roundoff == 0,
        bound_map_reconciled=True, selected_cell_reproduced_exactly=True,
        input_components=dict(reference_vector=reference_proof, native_edges=raw_proof,
            verification='Full small-map/chance/pairs/index hashes checked. Full raw/reference digests remain producer provenance; selected reads have stat continuity and chunk hashes. Only the selected aggregate cell is numerically replayed; whole-path counts remain the producer measurement.'))


def legacy_project(root, measurement_path):
    import numpy as np
    import torch
    root, measurement_path = Path(root).resolve(), Path(measurement_path).resolve()
    sys.path.insert(0, str(root / 'code/signal-trace'))
    from pipeline.overlap.complete_overlap import validate_pairs
    from pipeline.weight_edit.turn_pair import turns
    from pipeline.weight_edit.vector_chance import outside
    # Retain the numerical runtime's reduction configuration. Forcing a different
    # thread count can change the native squared-sum's final float64 bits.
    measured = json.loads(measurement_path.read_text())
    declaration_path = measurement_path.parents[2] / 'declaration.json'
    declaration = json.loads(declaration_path.read_text())
    producing_sources = projection_sources(root)
    declared_sources = {item['record']: item['sha256'] for item in declaration['implementation']}
    if any(declared_sources.get(path) != digest for path, digest in producing_sources['research_sources'].items()):
        raise ValueError('comparison mathematics differs from the declared implementation')
    if not measured['complete'] or not measured['ended']:
        raise ValueError('a partial acquisition cannot supply a complete overlap view')
    if measured['reference_sha256'] != declaration['reference']['sha256']:
        raise ValueError('the measured fixed reference changed')
    reference = bound_json(root, declaration['reference'])
    if reference['format'] != 'single-original-vector-chance/1':
        raise ValueError('this view requires the recorded whole-turn vector boundary')
    reading_path = binding_path(root, measured['reading'])
    reading = bound_json(root, measured['reading'])
    model_identity = record_identity(measured)
    if record_identity(reading) != model_identity or reading['acquisition'] != 'native-events-only':
        raise ValueError('comparison is not the measured native model')
    model_fields = {'model_sha256': measured['model_sha256']}
    if model_identity['kind'] == 'native-working-revision/1':
        generation = record_generation(measured)
        # reading.generation describes acquisition in prose. Its weight revision
        # is model_identity.generation, already compared in full above.
        if model_identity['original_sha256'] != declaration['original_model']['sha256']:
            raise ValueError('native reading and measurement have different update/original identities')
        # Bound historical reading arrays remain usable after the mutable file
        # advances. Their run/update label does not assert retained weight bytes.
        model_fields.update(model_identity=model_identity, generation=generation)
    folder = reading_path.parent
    if reading['row_map']['file'] != 'row-map.json' or reading['rows'] != measured['events']:
        raise ValueError('native row-map identity changed')
    row_map = json.loads(checked_bytes(folder / reading['row_map']['file'], reading['row_map']['sha256']))
    order = causal_rows(row_map, measured['member'], measured['events'])
    original = array(binding_path(root, reference['trace']), reference['trace']['sha256'])
    chance = array(binding_path(root, reference['chance']), reference['chance']['sha256'])
    pairs = array(binding_path(root, measured['correspondence']), measured['correspondence']['sha256'])
    if (original.dtype != np.float64 or list(original.shape) != reference['trace']['shape']
            or chance.dtype != np.float64 or chance.shape != original.shape[:2]
            or original.shape[1:] != (measured['merges'], reading['width'])):
        raise ValueError('bound reference dimensions changed')
    validate_pairs(pairs, len(original), measured['events'])
    oi, ci = torch.from_numpy(pairs.copy()).long().T
    mass = 1. / (len(original) * measured['merges'] * torch.bincount(oi, minlength=len(original))[oi].double())
    incoming, scales, input_proof = numerical_inputs(root, declaration, reading, folder,
        measured['member'], measured['events'], reading['width'])
    distances, radii, losses, merge_totals, outside_count, labels = [], [], [], [], 0, []
    detail = None
    with torch.no_grad():
        for index, node in enumerate(reading['merges']):
            if node['merge'] != index or node['file'] != f'nodes/{index}/state.npy':
                raise ValueError('native merge inventory changed')
            stored = array(folder / node['file'], node['sha256'])
            if (stored.dtype != np.dtype('<f4') or list(stored.shape) != node['shape']
                    or stored.shape != (measured['events'], reading['width'])):
                raise ValueError('native state dimensions changed')
            merged = torch.from_numpy(stored[order].copy())
            turn = turns({'incoming': [incoming], 'merged': [merged]}, expected_merges=1)[0]
            delta = turn[ci] - torch.from_numpy(original[:, index].copy())[oi]
            bound = torch.from_numpy(chance[:, index].copy())[oi]
            result = outside(delta, bound, mass)
            distance = torch.linalg.vector_norm(delta, dim=-1)
            radius = (2 * bound).sqrt()
            squared = result.residual.double().square()
            distances.append(distance.numpy().tolist())
            radii.append(radius.numpy().tolist())
            losses.append(squared.sum(dim=-1).numpy().tolist())
            merge_totals.append(float(squared.sum()))
            if detail is None or merge_totals[-1] > detail['merge_loss']:
                pair_index = int(squared.sum(dim=-1).argmax())
                reference_event, current_event = map(int, pairs[pair_index])
                detail = dict(merge=index, pair=pair_index, reference_event=reference_event,
                    current_event=current_event, merge_loss=merge_totals[-1],
                    weighted_loss=losses[-1][pair_index], distance=distances[-1][pair_index],
                    radius=radii[-1][pair_index],
                    reference_turn=original[reference_event, index].tolist(),
                    current_turn=turn[current_event].numpy().tolist(),
                    selection='Largest total-loss merge, then largest contributing pair within it; display detail only')
            outside_count += int(result.outside.sum())
            labels.append(f'Layer {node["layer"]} · {node["kind"]} · merge {index}')
            incoming = merged * scales[index // 3] if (index + 1) % 3 == 0 and index + 1 < measured['merges'] else merged
    objective = math.fsum(merge_totals)
    if outside_count != measured['outside'] or len(pairs) * len(labels) != measured['pair_merge_cells']:
        raise ValueError('displayed comparison does not reproduce the complete measured inventory')
    if objective != measured['objective']:
        raise ValueError(f'displayed objective {objective!r} differs from native record {measured["objective"]!r}')
    return dict(format=SCHEMA, measured_at=measured['measured_at'], step=measured['step'],
        member=measured['member'], reference_member=reference['reference'],
        acquisition_origin='retained_input' if 'recovered_from' in measured else 'fresh_native',
        objective=objective, outside=outside_count, pair_merge_cells=measured['pair_merge_cells'],
        measurement=dict(record=str(measurement_path.relative_to(root)), sha256=sha(measurement_path.read_bytes())),
        declaration_sha256=sha(declaration_path.read_bytes()), reference=declaration['reference'],
        source_reading=measured['reading'], correspondence=measured['correspondence'],
        **model_fields, input_components=input_proof,
        torch_threads=torch.get_num_threads(),
        producer=dict(module_sha256=SOURCE_SHA256, **producing_sources),
        pairs=pairs.tolist(), merge_labels=labels, distance=distances, radius=radii,
        weighted_loss=losses, merge_loss=merge_totals, coordinate_detail=detail,
        objective_reproduced_exactly=True,
        interpretation='fixed original correct reference versus originally wrong training member; no token judgment')


def project(root, measurement_path):
    if 'reading' in json.loads(Path(measurement_path).read_text()):
        return legacy_project(root, measurement_path)
    import numpy as np
    import torch
    root, measurement_path = Path(root).resolve(), Path(measurement_path).resolve()
    sys.path.insert(0, str(root / 'code/signal-trace'))
    from pipeline.overlap.complete_overlap import validate_pairs
    from pipeline.weight_edit.turn_pair import turns
    from pipeline.weight_edit.vector_chance import outside
    # Retain the numerical runtime's reduction configuration. Forcing a different
    # thread count can change the native squared-sum's final float64 bits.
    measured = json.loads(measurement_path.read_text())
    declaration_path = measurement_path.parents[2] / 'declaration.json'
    declaration = json.loads(declaration_path.read_text())
    producing_sources = raw_projection_sources(root)
    declared_sources = {item['record']: item['sha256'] for item in declaration['implementation']}
    if any(declared_sources.get(path) != digest for path, digest in producing_sources['research_sources'].items()):
        raise ValueError('comparison mathematics differs from the declared implementation')
    if not measured['complete'] or not measured['ended']:
        raise ValueError('a partial acquisition cannot supply a complete overlap view')
    if measured['reference_sha256'] != declaration['reference']['sha256']:
        raise ValueError('the measured fixed reference changed')
    reference = bound_json(root, declaration['reference'])
    if reference['format'] != 'single-original-vector-chance/1':
        raise ValueError('this view requires the recorded whole-turn vector boundary')
    if 'reading' not in measured and 'residual_squared' in measured:
        record = raw_project(root, declaration, measured, reference)
        if record['model_identity']['run'] != str(declaration_path.parent.relative_to(root)):
            raise ValueError('raw observation names another controller run')
        return dict(record, format=SCHEMA, measured_at=measured['measured_at'], step=measured['step'],
            member=measured['member'], reference_member=reference['reference'],
            acquisition_origin='retained_input' if 'recovered_from' in measured else 'fresh_native',
            objective=measured['objective'], outside=measured['outside'], pair_merge_cells=measured['pair_merge_cells'],
            measurement=dict(record=str(measurement_path.relative_to(root)), sha256=sha(measurement_path.read_bytes())),
            declaration_sha256=sha(declaration_path.read_bytes()), reference=declaration['reference'],
            correspondence=measured['correspondence'], torch_threads=torch.get_num_threads(),
            producer=dict(module_sha256=SOURCE_SHA256, **producing_sources),
            interpretation='Checksum-bound weighted residual contributions over every reference event/merge; mapped current events aggregated. The selected pair overlay is an actual computational detail, not the whole aggregate cell or a token judgment.')
    raise ValueError('observation has no supported native trace representation')


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'figure.facecolor': 'white'})
    return plt


def color_scale(fields):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    positive = np.concatenate([x[x > 0] for x in fields])
    cmap = LinearSegmentedColormap.from_list('excess', ['#ffe0a3', '#f08832', '#7f2704'])
    cmap.set_bad('white')
    norm = (LogNorm(vmin=positive.min(), vmax=positive.max()) if len(positive) and positive.min() < positive.max()
            else Normalize(vmin=0, vmax=positive.max() if len(positive) else None))
    return cmap, norm


def color_label(norm, *, contribution=False):
    from matplotlib.colors import LogNorm
    what = 'Weighted residual-squared contribution' if contribution else 'Excess turn distance'
    return what + ('\n' if contribution else ' ') + '(dimensionless; ' + ('log' if isinstance(norm, LogNorm) else 'linear') + ' color scale)'


def heatmap(record):
    import numpy as np
    if record.get('map_kind') == 'reference-weighted-residual-squared':
        return np.array(record['contribution']), 'Original reference event index (mapped pairs aggregated)', True
    return np.maximum(np.array(record['distance']) - np.array(record['radius']), 0), 'Saved computational correspondence pair index', False


def completed_updates(record):
    value = record.get('completed_updates', record.get('generation'))
    return value if type(value) is int else record['step'] - 1


def projection_verified(record):
    if record.get('map_kind') == 'reference-weighted-residual-squared':
        return (record.get('bound_map_reconciled') is True
                and record.get('selected_cell_reproduced_exactly') is True
                and record.get('map_aggregation_roundoff') == record.get('map_sum', math.nan) - record['objective'])
    return record.get('objective_reproduced_exactly') is True


def render_overlap(record, destination, *, dpi=80):
    import numpy as np
    plt = plots()
    excess, x_label, aggregated = heatmap(record)
    detail = record['coordinate_detail']
    selected = detail['merge']
    fig, (heat, profile) = plt.subplots(2, 1, figsize=(14, 9), layout='constrained')
    fig.suptitle(f'Computational overlap · observation {record["step"]}\n'
                 f'{record["member"]} ↔ fixed correct reference {record["reference_member"]}', fontsize=16)
    cmap, norm = color_scale([excess])
    shown = heat.imshow(np.ma.masked_equal(excess, 0), aspect='auto', interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
    heat.set(title=f'{record["outside"]:,} of {record["pair_merge_cells"]:,} pair/merge cells outside original tolerance',
             xlabel=x_label, ylabel='Merge index in model order (3 per layer)')
    heat.scatter([detail['reference_event'] if aggregated else detail['pair']], [selected], marker='s', s=75, facecolors='none', edgecolors='#1555b5',
                 linewidths=1.5, label='Exact-coordinate detail below')
    heat.legend(loc='upper right', fontsize=8)
    fig.colorbar(shown, ax=heat, label=color_label(norm, contribution=aggregated))
    correct, current = np.asarray(detail['reference_turn']), np.asarray(detail['current_turn'])
    x = np.arange(len(correct))
    profile.fill_between(x, correct, current, color='#efab82', alpha=.65, label='Coordinate differences')
    profile.plot(x, correct, color='#1555b5', linewidth=1.1, label='Fixed correct computation')
    profile.plot(x, current, color='#c46528', linewidth=1.0, linestyle='--',
                 label='Current computation (initially wrong case)')
    profile.set(title=f'Actual turn vectors · {record["merge_labels"][selected]} · pair {detail["pair"]}\n'
                f'Reference event {detail["reference_event"]} ↔ current event {detail["current_event"]} · '
                f'vector distance {detail["distance"]:.5g} / tolerance radius {detail["radius"]:.5g}',
                xlabel=f'Turn-vector coordinate index (all {len(correct):,} components)',
                ylabel='Turn value: unit(output) − unit(input) (dimensionless)')
    profile.legend(loc='upper left', fontsize=8)
    selection = ('One pair detail within the selected aggregated reference cell; the map sums every matched pair. '
                 if aggregated else 'One pair/merge detail: largest total-loss merge, then its largest contributing pair. ')
    profile.text(0, -0.27, selection +
                 'The full trace is represented by the map above.\n'
                 'Both curves are actual recorded coordinates; no PCA or word alignment. Shading is coordinate difference; '
                 'tolerance applies to the whole vector.\n'
                 f'Full Computational loss (native.objective) = {record["objective"]:.10g}; acquisition = {record["acquisition_origin"]}. '
                 'Exact values: overlap.json. No repair claim.', transform=profile.transAxes, fontsize=8)
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def render_before_after(before, after, destination, *, start_label=None, dpi=80):
    import numpy as np
    plt = plots()
    maps = [heatmap(r) for r in [before, after] if r]
    if any(item[2] != maps[0][2] for item in maps):
        raise ValueError('before/after maps must display the same computational quantity')
    fields = [item[0] for item in maps]
    aggregated = maps[0][2]
    cmap, norm = color_scale(fields)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout='constrained')
    fig.suptitle('Computational divergence · before and after weight updates', fontsize=16)
    for index, (ax, record) in enumerate(zip(axes, [before, after])):
        if record is None:
            ax.set_axis_off()
            ax.text(.5, .5, 'No later native computation recorded\nAwaiting a complete native measurement\nafter another installed update',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14, color='#526270')
            continue
        shown = ax.imshow(np.ma.masked_equal(fields[index], 0), origin='lower', aspect='auto',
                          interpolation='nearest', cmap=cmap, norm=norm)
        baseline = f'{start_label}\n' if index == 0 and start_label else ''
        ax.set(title=f'{"Before" if index == 0 else "After"}: {completed_updates(record)} completed updates in this run\n'
               f'{baseline}Computational loss {record["objective"]:.7g} · {record["outside"]:,}/{record["pair_merge_cells"]:,} outside',
               xlabel=maps[index][1], ylabel='Merge index (3 per layer)')
    fig.colorbar(shown, ax=list(axes), label=color_label(norm, contribution=aggregated) + ' · shared', shrink=.8)
    fig.text(.5, -.02, ('White: zero weighted residual contribution. Each reference cell sums its mapped pairs. ' if aggregated
             else 'White: inside original tolerance. Orange: outside. ') + 'Each acquisition retains its own full '
             'computational correspondence; matching is never based on words.', ha='center', fontsize=10)
    fig.savefig(destination, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def render_progress(native, solver, destination, *, step, origin, gradient=None, start_label=None, dpi=80):
    from matplotlib.ticker import MaxNLocator
    plt = plots()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    fig.suptitle('Computational loss · ' + datetime.datetime.now().astimezone().strftime('%b %d, %H:%M %Z'), fontsize=16)
    for ax, key, title, ylabel in ((axes[0], 'objective', 'Computational loss · native.objective', 'Computational loss (dimensionless)'),
                                  (axes[1], 'outside', 'Native divergence remaining', 'Outside correspondence-pair / merge cells (count)')):
        ax.plot([completed_updates(x) for x in native], [x[key] for x in native], 'o-', color='#27496b')
        ax.set(title=title, xlabel='Completed weight updates in this run', ylabel=ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim(bottom=0)
        if len(native) == 1:
            ax.set_xticks([completed_updates(native[0])])
            ax.text(0.02, 0.96, 'One recorded observation\nAwaiting the next native measurement', transform=ax.transAxes,
                    va='top', fontsize=9)
    if gradient is not None:
        timings = gradient.get('timings', {})
        phases = [('weight_derivative_seconds', 'Derivative'), ('directional_response_seconds', 'Response'),
                  ('native_rounding_seconds', 'BF16 rounding')]
        measured = [(label, timings[key]) for key, label in phases if key in timings]
        axes[2].bar([label for label, value in measured], [value for label, value in measured], color='#c46528')
        axes[2].set(title=f'Direct weight step · observation {step}', xlabel='Measured operation', ylabel='Wall time (seconds)')
        details = [gradient.get('phase_label', 'Awaiting weight derivative')]
        for key, label in (('gradient_norm_squared', 'Squared gradient norm'),
                           ('response_length', 'Response-derived alpha'), ('eta', 'Empirical eta'),
                           ('length', 'Applied scale (eta × alpha)' if 'eta' in gradient else 'Response scale'),
                           ('linear_prediction', 'Predicted linear loss at applied scale')):
            if gradient.get(key) is not None:
                details.append(f'{label}: {gradient[key]:.6g}')
        axes[2].text(.02, .98, '\n'.join(details), transform=axes[2].transAxes, va='top', fontsize=8,
                     bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
        footer = 'One derivative and one response determine the proposal. Its linear prediction is not regenerated native loss.'
    else:
        axes[2].plot([x['iteration'] for x in solver], [x['objective'] for x in solver], '.-', color='#c46528')
        axes[2].set(title=f'Inner inverse solve · observation {step}', xlabel='Observed accepted inner iteration',
                    ylabel='Linearized squared residual loss (dimensionless)')
        axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        if not solver:
            axes[2].text(.5, .5, 'No recorded inner-solver samples yet', transform=axes[2].transAxes,
                         ha='center', va='center', fontsize=10)
        footer = 'Inner values are sampled linear predictions, not regenerated native loss. Missing iterations are unknown.'
    axes[2].set_ylim(bottom=0)
    baseline = f'{start_label}. ' if start_label else ''
    fig.text(0.5, -0.10, f'{baseline}Initial acquisition: {origin}.\n{footer}', ha='center', va='top', fontsize=10)
    fig.savefig(destination, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def publish_overlap(client, root, folder, step, parent, child, state, checkpoint, cache_root):
    """Called after a complete native measurement; computation is cached per hash."""
    root, folder, step = Path(root), Path(folder), Path(step)
    measured_path = step / 'measurement.json'
    measured_sha = sha(measured_path.read_bytes())
    cache = Path(cache_root) / folder.name / f'step-{int(step.name):04d}'
    cache.mkdir(parents=True, exist_ok=True)
    receipt = state.setdefault('overlap', {}).setdefault(step.name, {})
    key = sha((measured_sha + SOURCE_SHA256).encode())
    if receipt.get('published') == key:
        return receipt
    target = cache / 'overlap.json'
    record = json.loads(target.read_text()) if target.is_file() else None
    measurement = json.loads(measured_path.read_text())
    current_sources = (projection_sources(root) if 'reading' in measurement else raw_projection_sources(root))
    def matches(record):
        return (isinstance(record, dict) and record.get('measurement', {}).get('sha256') == measured_sha
                and record.get('format') == SCHEMA and projection_verified(record)
                and all(record.get(key) == measurement[key] for key in ('objective', 'outside', 'pair_merge_cells'))
                and all(record.get('producer', {}).get(key) == value for key, value in current_sources.items()))
    if not matches(record):
        uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
        command = ['nice', '-n', '19', 'ionice', '-c', '3', uv, 'run', '--offline',
                   *[part for package in NUMERICAL_PACKAGES for part in ('--with', package)], '--with', 'matplotlib==3.11.2',
                   'python', str(Path(__file__).resolve()), str(root), str(measured_path), str(cache)]
        worker = subprocess.run(command, cwd=root, env=os.environ | {'CUDA_VISIBLE_DEVICES': ''},
                                capture_output=True, text=True)
        if worker.returncode:
            raise RuntimeError('computational view worker failed: ' + worker.stderr[-1800:])
        record = json.loads(target.read_text())
        if not matches(record):
            raise ValueError('projection worker output does not identify the verified measurement and numerical sources')
    if receipt.get('published') != key:
        render_overlap(record, cache / 'overlap.png')
        (cache / 'axes.txt').write_text(AXES)
        for name in ('overlap.json', 'overlap.png', 'axes.txt'):
            client.log_artifact(parent, str(cache / name), artifact_path=f'computation/step-{int(step.name):04d}')
            client.log_artifact(child, str(cache / name), artifact_path='computation')
        receipt.update(published=key, cache=str(cache), measurement_sha256=measured_sha,
                       projection_source=record['producer'], renderer_sha256=SOURCE_SHA256,
                       scalar_sha256=sha(target.read_bytes()))
        checkpoint()
    return receipt


def publish_progress(client, folder, parent, state, checkpoint, cache_root):
    """Small-record refresh: no native arrays or model components are reopened."""
    folder = Path(folder)
    observations = []
    for step in sorted((p for p in (folder / 'loop').iterdir() if p.is_dir() and p.name.isdecimal()), key=lambda p: int(p.name)):
        path = step / 'measurement.json'
        if path.is_file():
            record = json.loads(path.read_text())
            if record.get('complete') and record.get('ended'):
                observations.append(record)
    if not observations:
        return
    latest = observations[-1]
    child = state.get('steps', {}).get(str(latest['step']), {}).get('run')
    if not child:
        return
    snapshot = dict(native=[{k: m[k] for k in ('step', 'objective', 'outside', 'measured_at')} |
                           {'completed_updates': completed_updates(m)} for m in observations],
        step=latest['step'], origin='retained_input' if 'recovered_from' in observations[0] else 'fresh_native',
        inverse_progress_sha256=sha((folder / 'loop' / str(latest['step']) / 'inverse-progress.json').read_bytes())
        if (folder / 'loop' / str(latest['step']) / 'inverse-progress.json').is_file() else None,
        plot_source_sha256=SOURCE_SHA256,
        available_overlap={k: dict(publication=v.get('published'), scalar_sha256=v.get('scalar_sha256'))
                           for k, v in state.get('overlap', {}).items()})
    step_folder = folder / 'loop' / str(latest['step'])
    declaration = json.loads((folder / 'declaration.json').read_text())
    start = declaration.get('working_model', {}).get('start')
    if start:
        reached = bound_json(folder.resolve().parents[3], start['reached'])
        snapshot['start_label'] = f'Retained predecessor update {reached["step"]} (run start)'
    elif declaration.get('start_fresh') and observations[0]['step'] == 0:
        snapshot['start_label'] = 'Retained declared starting state (local baseline 0)'
    if declaration.get('gradient_step'):
        progress_path, proposal_path = step_folder / 'gradient-progress.json', step_folder / 'proposal.json'
        progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
        proposal = json.loads(proposal_path.read_text()) if proposal_path.is_file() else {}
        response = proposal.get('response') or {}
        timings = proposal.get('timings') or {k: v for k, v in progress.items() if k.endswith('_seconds')}
        status = proposal.get('status')
        phase = (('Update installed; awaiting fresh native measurement' if status == 'accepted'
            else 'BF16 proposal ready; native write pending' if status == 'ready' else status.replace('_', ' '))
            if status else GRADIENT_PHASES.get(progress.get('phase'), 'Awaiting weight derivative'))
        if (step_folder / 'reached.json').is_file():
            phase = ('Fresh native measurement complete after installed update'
                     if 'residual_squared' in latest and 'reading' not in latest else
                     'Update installed; awaiting fresh native measurement')
        snapshot['gradient'] = dict(phase_label=phase,
            gradient_norm_squared=proposal.get('gradient_norm_squared', progress.get('gradient_norm_squared')),
            length=proposal.get('applied_length', response.get('length', progress.get('length'))),
            linear_prediction=proposal.get('linear_prediction_at_applied_length',
                response.get('predicted_objective', progress.get('linear_prediction'))), timings=timings)
        if 'applied_length' in proposal:
            snapshot['gradient'].update(response_length=response.get('length'), eta=proposal['eta'])
    key = sha(json.dumps(snapshot, sort_keys=True).encode())
    if state.get('progress_view') == key:
        return
    state.pop('comparison_view', None)
    checkpoint()
    samples = [] if 'gradient' in snapshot else client.get_metric_history(child, 'solver/linear_objective')
    snapshot['solver'] = [dict(iteration=m.step, objective=m.value, timestamp_ms=m.timestamp)
                          for m in sorted(samples, key=lambda m: (m.timestamp, m.step))]
    cache = Path(cache_root) / folder.name
    cache.mkdir(parents=True, exist_ok=True)
    render_progress(snapshot['native'], snapshot['solver'], cache / 'progress.png', step=snapshot['step'],
                    origin=snapshot['origin'], gradient=snapshot.get('gradient'), start_label=snapshot.get('start_label'))
    (cache / 'progress.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (cache / 'axes.txt').write_text(AXES)
    for name in ('progress.png', 'progress.json', 'axes.txt'):
        client.log_artifact(parent, str(cache / name), artifact_path='computation')
    first = cache / f'step-{observations[0]["step"]:04d}/overlap.json'
    last = cache / f'step-{latest["step"]:04d}/overlap.json'
    def published_record(path, observation):
        receipt = state.get('overlap', {}).get(str(observation['step']), {})
        if not receipt.get('published') or not receipt.get('scalar_sha256') or not path.is_file():
            return None
        payload = path.read_bytes()
        measurement_sha = sha((folder / 'loop' / str(observation['step']) / 'measurement.json').read_bytes())
        if sha(payload) != receipt['scalar_sha256'] or measurement_sha != receipt.get('measurement_sha256'):
            return None
        record = json.loads(payload)
        if (record.get('measurement', {}).get('sha256') != measurement_sha
                or not projection_verified(record)):
            return None
        return record
    before, last_record = published_record(first, observations[0]), published_record(last, latest)
    if before is not None and last_record is not None:
        after = last_record if len(observations) > 1 else None
        render_before_after(before, after, cache / 'before-after.png', start_label=snapshot.get('start_label'))
        client.log_artifact(parent, str(cache / 'before-after.png'), artifact_path='computation')
        state['comparison_view'] = dict(published=key, artifact='computation/before-after.png',
                                       first_step=observations[0]['step'], latest_step=latest['step'])
    state['progress_view'] = key
    checkpoint()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('measurement', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    record = project(args.root, args.measurement)
    (args.destination / 'overlap.json').write_text(json.dumps(record, separators=(',', ':')) + '\n')
    (args.destination / 'axes.txt').write_text(AXES)
    render_overlap(record, args.destination / 'overlap.png')
    print(json.dumps({k: record[k] for k in ('objective', 'outside', 'pair_merge_cells', 'objective_reproduced_exactly')}))


# Keep the unchanged reading-array projection cache valid. The raw-map route
# has its own numerical fingerprint and cannot reuse that legacy cache.
NUMERICAL_SOURCE_SHA256 = sha(('\n\n'.join(inspect.getsource(f) for f in (
    record_identity, record_generation,
    sha, checked_bytes, binding_path, bound_json, array, causal_rows, numerical_inputs,
    projection_sources)) + '\n\n' + inspect.getsource(legacy_project).replace(
        'def legacy_project(', 'def project(', 1)).encode())


def raw_projection_sources(root):
    return projection_sources(root) | {'numerical_functions_sha256': RAW_NUMERICAL_SOURCE_SHA256}


RAW_NUMERICAL_SOURCE_SHA256 = sha('\n\n'.join(inspect.getsource(f) for f in (
    record_identity, record_generation, sha, checked_bytes, binding_path, bound_json, array,
    file_identity, partial_vector, raw_edges, raw_project, projection_sources,
    raw_projection_sources, project)).encode())


if __name__ == '__main__':
    main()
