"""One working GGUF handoff to the existing file-only native generator.

No checkpoint, redo journal, per-step model/tensor hashes, or automatic recovery.
An interrupted write leaves status=writing and cannot supply a fresh observation.
Restore is a separate explicit operation from the original or a saved checkpoint.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from . import gap_state, working_identity as identity


def save(path, record):
    path = Path(path)
    temporary = path.with_name(path.name + '.next')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    os.replace(temporary, path)


class Working:
    folder: Path
    original: Path
    file: Path
    names: tuple[str, ...]
    bindings: dict
    owner: int | None
    fd: int | None
    state: dict[str, Any]

    @classmethod
    def create(cls, folder, original, original_sha256, bindings, names, *, run,
               start=None, start_sha256=None):
        self = cls()
        self.folder, self.original = Path(folder).resolve(), Path(original).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = self.folder / 'model.gguf'
        self.names, self.bindings = tuple(names), bindings
        self.owner = os.open(self.folder / 'owner.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.fd = None
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if self.file.exists() or (self.folder / 'state.json').exists():
                raise ValueError('an existing working attempt is preserved; restore separately, never auto-resume')
            spans = sorted((bindings[n]['offset_bytes'], bindings[n]['offset_bytes'] + bindings[n]['bytes']) for n in names)
            original_stat = identity.fingerprint(self.original)
            if (not self.names or len(set(self.names)) != len(self.names)
                    or any(a < 0 or b <= a or b > original_stat[2] for a, b in spans)
                    or any(a[1] > b[0] for a, b in zip(spans, spans[1:]))):
                raise ValueError('declared native matrix ranges must be distinct and inside the model')
            with self.original.open('rb') as source:
                if hashlib.file_digest(source, 'sha256').hexdigest() != original_sha256:
                    raise ValueError('the original differs from its declared source')
            source_file = self.original if start is None else Path(start).resolve()
            source_sha = original_sha256 if start is None else start_sha256
            source_stat = identity.fingerprint(source_file)
            if source_file == self.original and source_sha != original_sha256:
                raise ValueError('the original starting state must use its verified original SHA256')
            if source_file != self.original:
                with source_file.open('rb') as source:
                    if hashlib.file_digest(source, 'sha256').hexdigest() != source_sha:
                        raise ValueError('the retained starting state differs from its declared SHA256')
            if source_stat[2] != original_stat[2]:
                raise ValueError('the retained starting state has another model layout')
            revision = dict(kind=identity.KIND, run=run, generation=0, original_sha256=original_sha256)
            self.state = dict(format=identity.FORMAT, status='writing', generation=0,
                **identity.fields(revision), model_file=str(self.file), original_file=str(self.original),
                original_file_identity=original_stat,
                starts_from=dict(file=str(source_file), model_sha256=source_sha))
            save(self.folder / 'state.json', self.state)
            self.fd = os.open(self.file, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with source_file.open('rb') as source, os.fdopen(os.dup(self.fd), 'wb') as output:
                shutil.copyfileobj(source, output)
            opened = os.fstat(self.fd)
            if ([opened.st_dev, opened.st_ino] == original_stat[:2]
                    or [opened.st_dev, opened.st_ino, opened.st_size] != identity.fingerprint(self.file)[:3]
                    or opened.st_size != original_stat[2] or identity.fingerprint(self.original) != original_stat
                    or identity.fingerprint(source_file) != source_stat):
                raise ValueError('the working copy must be distinct from its unchanged original')
            self.state.update(status='ready', file_identity=identity.fingerprint(self.file))
            save(self.folder / 'state.json', self.state)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        for name in ('fd', 'owner'):
            fd = getattr(self, name, None)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def current(self):
        if self.owner is None or self.fd is None:
            raise ValueError('the controller no longer owns this working model')
        state = identity.load_state(self.folder / 'state.json', expected_identity=self.state['model_identity'])
        opened = os.fstat(self.fd)
        if [opened.st_dev, opened.st_ino, opened.st_size] != state['file_identity'][:3]:
            raise ValueError('the owned descriptor is not the current working model')
        return state

    def install(self, expected_identity, weights, step):
        current = self.current()
        assert self.fd is not None
        if (expected_identity != current['model_identity'] or set(weights) != set(self.names)
                or type(step) is not int or step != current['generation'] + 1):
            raise ValueError('every prescribed matrix must follow the current complete revision')
        if any(not bool(np.isfinite(weights[n]).all()) for n in self.names):
            raise ValueError('native updates must be finite')
        self.state['status'] = 'writing'
        save(self.folder / 'state.json', self.state)
        # pwrite is unbuffered. Each complete call is visible to the subsequently
        # launched reader; checkpoint durability is deliberately a separate job.
        for name in self.names:
            raw = gap_state.encoded(weights[name].T, self.bindings[name])
            self._write(self.fd, self.bindings[name]['offset_bytes'], raw)
        opened = os.fstat(self.fd)
        actual = identity.fingerprint(self.file)
        if (actual[:3] != [opened.st_dev, opened.st_ino, opened.st_size]
                or identity.fingerprint(self.original) != self.state['original_file_identity']):
            raise ValueError('the native working path or original changed during installation')
        self.state.update(status='ready', generation=step, file_identity=actual,
                          model_identity=dict(expected_identity, generation=step))
        save(self.folder / 'state.json', self.state)
        return self.current()

    @staticmethod
    def _write(fd, offset, raw):
        view = memoryview(raw)
        while view:
            count = os.pwrite(fd, view, offset)
            if count <= 0:
                raise OSError('native matrix write made no progress')
            offset += count
            view = view[count:]
