"""Hash an explicitly inventoried inactive file without retaining each consumed chunk.

Invoked only by the opt-in archive CLI mode. This never changes or deletes a file.
The caller supplies the existing R2 range size as the local buffer bound.
"""
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path


def identity(st):
    return dict(dev=st.st_dev, ino=st.st_ino, bytes=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def restored_digest(path, expected, name, root):
    """Reuse only the digest whose exact inventory the existing local restore receipt binds."""
    if Path(name).name != name or not name.endswith('.json'):
        raise ValueError('restored archive must be one registry record name')
    registry = root / 'studies/04-round-2-overlap/input-archives'
    record = json.loads((registry / name).read_text())
    receipt = json.loads((registry / '.restored' / name).read_text())
    files = record['files']
    inventory = ''.join(f'{sub}\t{entry["bytes"]}\t{entry["sha256"]}\n' for sub, entry in sorted(files.items()))
    bound = hashlib.sha256(inventory.encode()).hexdigest()
    if record['inventory_sha256'] != bound or receipt['inventory_sha256'] != bound:
        raise ValueError('restored receipt does not bind this exact inventory')
    base = root / record['input']
    if not base.is_relative_to(root / 'studies') or base.resolve() != base:
        raise ValueError('archive input is not canonical under studies')
    sub = '.' if Path(path) == base else Path(path).relative_to(base).as_posix()
    entry = files[sub]
    if receipt['files'][sub] != expected or entry['bytes'] != expected['bytes']:
        raise ValueError('allowlist differs from the restored file identity')
    if not re.fullmatch('[0-9a-f]{64}', entry['sha256']):
        raise ValueError('invalid restored file digest')
    return entry['sha256']


def digest(path, allowlist, chunk_bytes, restored_archive=None, root=None):
    if chunk_bytes <= 0:
        raise ValueError('chunk size must be positive')
    path = os.path.abspath(path)
    expected = allowlist.get(path)
    if expected is None:
        raise ValueError('file is not in the inactive archive read allowlist')
    previous = restored_digest(path, expected, restored_archive, root or Path(__file__).resolve().parents[3]) if restored_archive else None
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or identity(before) != expected:
            raise ValueError('file identity differs from the inactive archive allowlist')
        offset = before.st_size if previous else 0
        result = previous
        if previous is None:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_NOREUSE)
            hasher = hashlib.sha256()
            while True:
                chunk = os.read(fd, chunk_bytes)
                if not chunk:
                    break
                hasher.update(chunk)
                os.posix_fadvise(fd, offset, len(chunk), os.POSIX_FADV_DONTNEED)
                offset += len(chunk)
                del chunk
            result = hasher.hexdigest()
        if offset != before.st_size or identity(os.fstat(fd)) != expected or identity(os.stat(path, follow_symlinks=False)) != expected:
            raise ValueError('file changed during the archive hash')
        return result
    finally:
        os.close(fd)


if __name__ == '__main__':
    try:
        if len(sys.argv) not in (4, 5):
            raise ValueError('cache-hash.py <file> <allowlist.json> <chunk-bytes> [restored-archive.json]')
        print(digest(sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), int(sys.argv[3]), sys.argv[4] if len(sys.argv) == 5 else None))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f'cache-conscious hash refused: {error}', file=sys.stderr)
        raise SystemExit(1)
