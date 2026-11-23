"""Run-local revision labels for the current working file, not content hashes."""
import json
from pathlib import Path

FORMAT = 'single-working-model/1'
KIND = 'native-working-revision/1'


def fingerprint(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def from_record(record):
    value = record.get('model_identity')
    if value is not None:
        if (record.get('model_sha256') is not None or not isinstance(value, dict)
                or set(value) != {'kind', 'run', 'generation', 'original_sha256'}
                or value['kind'] != KIND or not isinstance(value['run'], str) or not value['run']
                or type(value['generation']) is not int or value['generation'] < 0
                or not _sha(value['original_sha256'])):
            raise ValueError('a working revision needs its run, update count and original binding; it is not a file hash')
        return value
    sha = record.get('model_sha256')
    if not _sha(sha):
        raise ValueError('a model needs an explicit revision or legacy file SHA256')
    return dict(kind='file-sha256', sha256=sha)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def fields(value):
    if value.get('kind') == 'file-sha256':
        return dict(model_sha256=value['sha256'])
    result = dict(model_identity=value, model_sha256=None)
    from_record(result)
    return result


def load_state(path, *, model=None, expected_identity=None):
    path = Path(path)
    state = json.loads(path.read_text())
    file = Path(state['model_file'])
    revision = from_record(state)
    if (path.is_symlink() or path != file.parent / 'state.json' or file.name != 'model.gguf'
            or state.get('format') != FORMAT or state.get('status') != 'ready'
            or state['generation'] != revision['generation'] or file.is_symlink()
            or fingerprint(file) != state['file_identity']
            or fingerprint(state['original_file']) != state['original_file_identity']
            or (model is not None and Path(model).absolute() != file)
            or (expected_identity is not None and revision != expected_identity)):
        raise ValueError('the working file is incomplete, changed, or not the expected revision')
    return state
