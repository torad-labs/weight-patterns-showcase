"""Keep lossless local event records for A's group analysis as raw captures archive.

The complete source event stream is preserved, not a selection of operations or
predictions. Native tensor bytes remain in each verified capture's R2 archive.
This bounded preprocessing job exits after the natural-response cohort is represented.
"""
import argparse
import hashlib
import json
import subprocess
import time

from pipeline.acquisition.run_a_baseline import ROOT, RUN


def release_verified_events(root, record, context, archive):
    """Release only unchanged, remotely verified events after context indexing."""
    if context is None or archive is None or archive.get('verify_exit_code') != 0:
        return None
    expected = record['source_sha256']
    entry = next((x for x in archive['files'] if x['path'] == record['source']), None)
    if (entry is None or entry['sha256'] != expected
            or entry['bytes'] != record['source_bytes']
            or context['events_sha256'] != expected):
        raise ValueError('Archive, compressed source and context index disagree; retain events')
    path = root/record['source']
    if not path.exists():
        return None
    if path.is_symlink():
        raise ValueError('Refusing to release a source symlink')
    before = path.stat()
    with path.open('rb') as source:
        actual = hashlib.file_digest(source, 'sha256').hexdigest()
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if (actual != expected or after.st_size != record['source_bytes']
            or identity(before) != identity(after)):
        raise ValueError('Local event source changed; retain events')
    path.unlink()
    return record['source']


def compress_events(case, destination):
    audit = json.loads((ROOT/case['audit']).read_text())
    source = ROOT/case['capture']/'signal-events.jsonl'
    if not source.exists():
        subprocess.run(['torad-research', 'traces', 'download',
                        str(source.relative_to(ROOT/'runs'))], check=True)
    target = destination/(case['id']+'.jsonl.zst')
    temporary = target.with_suffix('.writing')
    digest, total = hashlib.sha256(), 0
    with source.open('rb') as original, temporary.open('xb') as output:
        process = subprocess.Popen(['zstd', '-q', '-1', '-T1', '-c'],
                                   stdin=subprocess.PIPE, stdout=output)
        try:
            while data := original.read(4*1024**2):
                process.stdin.write(data)
                digest.update(data)
                total += len(data)
        finally:
            process.stdin.close()
            code = process.wait()
    if code != 0 or digest.hexdigest() != audit['events_sha256'] or total != audit['event_bytes']:
        raise ValueError('Compressed source does not cover the complete audited event file')
    temporary.replace(target)
    with target.open('rb') as f:
        compressed_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    record = dict(case=case['id'], source=str(source.relative_to(ROOT)),
                  source_bytes=total, source_sha256=digest.hexdigest(),
                  compressed=str(target.relative_to(ROOT)),
                  compressed_bytes=target.stat().st_size, compressed_sha256=compressed_hash,
                  events=audit['events'], native_tensor_data=case['capture']+'/signal-data.bin',
                  scope='Lossless complete event metadata for group analysis; no vector or operation selection')
    (destination/(case['id']+'.json')).write_text(json.dumps(record, indent=2)+'\n')
    return record


def main(wait, release):
    cases = json.loads((ROOT/'research/experiments/signal-trace-repair/A-cohort.json').read_text())['cases']
    destination = RUN/'context-index/compressed-events'
    destination.mkdir(parents=True, exist_ok=True)
    while True:
        for case in cases:
            record_path = destination/(case['id']+'.json')
            if record_path.exists():
                if release:
                    context = RUN/'context-index'/(case['id']+'.json')
                    archive = RUN/'cohort-archives'/(case['id']+'.json')
                    if context.exists() and archive.exists():
                        record = json.loads(record_path.read_text())
                        archived = json.loads(archive.read_text())
                        # The capture worker publishes its archive before unlinking
                        # large files. Only release a subsequently restored copy;
                        # never race that worker's own release transaction.
                        if record['source'] not in archived.get('released_local', []):
                            continue
                        released = release_verified_events(
                            ROOT, record, json.loads(context.read_text()), archived)
                        if released:
                            record['released_verified_local_events'] = released
                            temporary = record_path.with_suffix('.writing')
                            temporary.write_text(json.dumps(record, indent=2)+'\n')
                            temporary.replace(record_path)
                            print(json.dumps(dict(released_verified_events=released)), flush=True)
                continue
            if not (ROOT/case['audit']).exists():
                continue
            record = compress_events(case, destination)
            print(json.dumps({k: record[k] for k in ['case','events','source_bytes','compressed_bytes']}), flush=True)
        pending = [x['id'] for x in cases if not (destination/(x['id']+'.json')).exists()]
        if not pending or not wait:
            print(json.dumps(dict(expected=len(cases), complete=len(cases)-len(pending), pending=pending)), flush=True)
            return
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--release-verified', action='store_true',
                        help='Release event JSON only after the recorded full remote-byte verification and context indexing')
    args = parser.parse_args()
    main(args.wait, args.release_verified)
