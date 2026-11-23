"""Admit the byte-verified restoration of the retired read-only provenance file."""
import sys
from pathlib import Path

from pipeline.research_guard import input_archives
from pipeline.weight_edit import final_experiment as baseline

ARCHIVE = baseline.ROOT/'studies/04-round-2-overlap/input-archives/42646d7012c5-model.gguf.json'


def bind(app, dec):
    binding = dec['retired_input_admission']['archive']
    archive_path = baseline.ROOT / binding['record']
    if archive_path != ARCHIVE or baseline.binding(archive_path) != binding:
        raise ValueError('unexpected retired predecessor archive')
    record, problems = input_archives.one(baseline.ROOT, archive_path.name)
    if record is None or problems:
        raise ValueError(f'archive registry admission: {problems}')
    model = app.held.SOURCE/'working/model.gguf'
    state_path = model.with_name('state.json')
    if record['input'] != str(model.relative_to(baseline.ROOT)) or set(record['files']) != {'.'}:
        raise ValueError('archive covers another input')
    evidence = record['provenance']['sources']
    if evidence[str(state_path.relative_to(baseline.ROOT))] != baseline.digest(state_path):
        raise ValueError('archived state has different provenance')
    history = baseline.ROOT/'records/maintenance-runs/2026-11-17-evening-retention/retired-model.sha256'
    historical = history.read_text().strip().split(maxsplit=1)
    entry = record['files']['.']
    remote = record.get('remote_bytes_verified') or {}
    transcript = archive_path.parent/'verifications'/(archive_path.stem+'.txt')
    if (remote.get('inventory_sha256') != record['inventory_sha256']
            or remote.get('output_sha256') != baseline.digest(transcript)
            or remote.get('files') != 1 or remote.get('bytes') != entry['bytes']):
        raise ValueError('retired archive lacks its exact successful full remote readback')
    if (historical != [entry['sha256'], record['input']]
            or baseline.digest(history) != evidence[str(history.relative_to(baseline.ROOT))]):
        raise ValueError('archived payload differs from the historical model digest')
    original_gate = app.held.frozen
    identity = app.held.working_identity
    verified = None

    def frozen(inputs):
        nonlocal verified
        # Keep the original admission in force whenever the historical inode is still present.
        if identity.fingerprint(model) == inputs['state']['file_identity']:
            return original_gate(inputs)
        events_file = app.held.CGROUP/'cgroup.events'
        try:
            events = dict(line.split() for line in events_file.read_text().splitlines())
        except FileNotFoundError:
            events = {'populated':'0'}
        state = baseline.read(state_path)
        if (events.get('populated') != '0' or state_path.is_symlink() or model.is_symlink()
                or state != inputs['state'] or state['format'] != identity.FORMAT or state['status'] != 'ready'
                or state['generation'] != identity.from_record(state)['generation']
                or Path(state['model_file']) != model or state['file_identity'][2] != entry['bytes']
                or identity.fingerprint(state['original_file']) != state['original_file_identity']):
            raise ValueError('retired producer, state, original model or restored identity disagrees')
        receipt = input_archives.receipt_of(baseline.ROOT,record).get('.')
        if not receipt or not input_archives.same_file(baseline.ROOT,record['input'],receipt):
            raise ValueError('the archive restore lacks a current verified file receipt')
        if verified != receipt:
            actual, seen = input_archives.read_held(model)
            if actual != entry['sha256'] or seen != receipt:
                raise ValueError('restored model failed full native-byte or stable-file identity verification')
            verified = dict(receipt) if isinstance(receipt,dict) else receipt
        return state
    app.held.frozen = frozen
    return frozen


def main():
    run = baseline.run_path(Path(sys.argv[1]))
    dec = baseline.read(run/'declaration.json')
    app = baseline.controller(baseline.MEASURED)
    bind(app,dec)
    app.main()


if __name__ == '__main__':
    main()
