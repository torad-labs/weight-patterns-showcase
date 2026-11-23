"""Keep all original connected turns; release only reproduced working streams.

The authoritative full native recordings remain in verified R2 archives. This
exports every prediction and all 105 connected merges, not a selected anchor.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np

from pipeline.acquisition.run_a_baseline import ROOT, RUN
from pipeline.native_records.group_observation import publish
from pipeline.observation.compare_a_earlier_passages import stream_frames
from pipeline.observation.inspect_a_connected_progressions import View, branches
from pipeline.qualification.qualify_a_coverage import check_case, qualified_labels


OUT = RUN / 'shared/observation/original-model-traces'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def export(member, release=False):
    cid = member['case']
    directory = OUT / f'original-gpu{member["gpu"]}' / cid
    destination = OUT / 'original-turns' / cid
    report_path = destination / 'index.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        return {k: report[k] for k in ['case', 'predictions', 'verified_native_vectors', 'bytes']}
    qualification = check_case(member, directory, qualified_labels(member['gpu']))
    (OUT / 'qualification').mkdir(exist_ok=True)
    publish(OUT / 'qualification' / (cid + '.json'), qualification)
    annotation = json.loads((RUN / 'response-correspondence/cases' / (cid + '.json')).read_text())
    original = json.loads(Path(member['baseline']).read_text())
    view = View(cid)
    template = view.template(min(view.frames))
    connections = branches(template)
    labels = {}
    for c in connections:
        labels[c['incoming']] = template['schema'][c['incoming']]['name']
        labels[c['merged']] = (f'pe_out-{c["layer"]}' if c['kind'] == 'per_layer_embedding'
                               else template['schema'][c['merged']]['name'])
    width = view.vector(min(view.frames), connections[0]['incoming']).size
    predictions = {s['input_position']: s['step'] for s in original['steps']}
    if sorted(predictions.values()) != list(range(len(predictions))):
        raise ValueError('Natural prediction order is incomplete')
    destination.mkdir(parents=True, exist_ok=True)
    writing = destination / 'turns.npy.writing'
    data = np.lib.format.open_memmap(writing, mode='w+', dtype='<f4', shape=(len(predictions), len(connections), width))
    norms = np.empty((len(predictions), len(connections), 2), dtype='<f4')
    seen, verified = set(), 0
    for frame in stream_frames(directory, set(labels.values())):
        for position, fields in frame.items():
            if position not in predictions:
                continue
            prediction = predictions[position]
            if prediction in seen or set(labels.values()) - fields.keys():
                raise ValueError('Repeated or incomplete original connected prediction')
            incoming = np.stack([fields[labels[c['incoming']]] for c in connections]).astype('f8')
            merged = np.stack([fields[labels[c['merged']]] for c in connections]).astype('f8')
            if prediction in view.frames:
                old_in = np.stack([view.vector(prediction, c['incoming']) for c in connections])
                old_out = np.stack([view.vector(prediction, c['merged']) for c in connections])
                if not np.array_equal(old_in, incoming) or not np.array_equal(old_out, merged):
                    raise ValueError('Working stream differs from the original retained native branch vectors')
                verified += len(connections) * 2
            ni = np.linalg.norm(incoming, axis=1, keepdims=True)
            no = np.linalg.norm(merged, axis=1, keepdims=True)
            data[prediction] = merged / np.maximum(no, 1e-30) - incoming / np.maximum(ni, 1e-30)
            norms[prediction, :, 0], norms[prediction, :, 1] = ni[:, 0], no[:, 0]
            seen.add(prediction)
    if seen != set(predictions.values()):
        raise ValueError('Export omitted natural prediction positions')
    data.flush()
    del data
    writing.replace(destination / 'turns.npy')
    np.save(destination / 'norms.npy', norms)
    archive = (ROOT / 'records/A-pilot-raw-archive-2026-11-08.json' if cid in ['block-base', 'block-changed']
               else RUN / 'cohort-archives' / (cid + '.json'))
    if not archive.exists():
        raise ValueError('Full original native archive record is missing')
    source_size = (directory / 'stream.f32').stat().st_size
    report = dict(case=cid, gpu=member['gpu'], role=annotation['role'], predictions=len(predictions),
        connections=connections, annotation=str(RUN / 'response-correspondence/cases' / (cid + '.json')),
        vectors=str(destination / 'turns.npy'), norms=str(destination / 'norms.npy'),
        shape=[len(predictions), len(connections), width],
        vector_sha256=sha256(destination / 'turns.npy'), verified_native_vectors=verified,
        bytes=(destination / 'turns.npy').stat().st_size + (destination / 'norms.npy').stat().st_size,
        qualification=qualification, full_original_archive_record=str(archive),
        source_stream_metadata=str(directory / 'stream.jsonl'), source_working_bytes=source_size,
        definition='At every natural prediction, ordered unit(merged) minus unit(incoming) at all 105 producer-bound merges; float32 derived vectors with native input/output norms. Not a substitute for the full native source.',
        source_release='Only the deterministic reproduced working stream can be removed; original full native data remain in verified R2 archives.')
    publish(report_path, report)
    if release:
        # The native original is already archived; this repeat's full stream is
        # reproducible from the same model, prompt, engine and card. No new
        # edited capture or authoritative original archive is deleted here.
        (directory / 'stream.f32').unlink()
        publish(destination / 'working-copy-released.json', dict(path=str(directory / 'stream.f32'),
            bytes=source_size, retained=str(report_path), source_archive=str(archive),
            reason='Reproduced working copy compacted after same-token/score qualification and complete turn export.'))
    return {k: report[k] for k in ['case', 'predictions', 'verified_native_vectors', 'bytes']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, choices=[1, 2, 3, 4], default=2)
    p.add_argument('--release-working', action='store_true')
    args = p.parse_args()
    declaration = json.loads((OUT / 'declaration.json').read_text())
    complete = []
    pending = []
    for member in declaration['population']:
        if (OUT / f'original-gpu{member["gpu"]}' / member['case'] / 'result.json').exists():
            complete.append(member)
        else:
            pending.append(member['case'])
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(export, member, args.release_working) for member in complete]):
            row = future.result(); rows.append(row); print(json.dumps(row), flush=True)
    publish(OUT / 'original-turns' / ('index.json' if not pending else 'partial.json'),
            dict(complete=not pending, cases=rows, pending=pending,
                 predictions=sum(r['predictions'] for r in rows), bytes=sum(r['bytes'] for r in rows)))


if __name__ == '__main__':
    main()
