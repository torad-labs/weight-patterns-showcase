"""A model state of a sweep over many matrices: named by what it holds, and kept in one file the sweep writes into.

A sweep writes one matrix after another and reads the model between writes. Naming every state by its file's sha256 reads the whole file
again for each command (9.8 s a time on this model). A state is named here by its content instead:

  identity   the original file's sha256 and, for every tensor whose bytes are no longer the original's, the tensor's name, its native type,
             shape and byte count, and the sha256 of its bytes as the file holds them. The tensors are listed in name order, so two
             histories of writes that end in the same bytes are one state, and a tensor written back to the original's bytes is no
             longer listed: the original always has the one name `identity(original_sha256, {})`. The name is typed (KIND) and is the
             sha256 of that list's canonical encoding. It is not a file's sha256 and is never written where one is expected.
  Working    one file a sweep writes into. It is made as a copy of the original, which is only read. A matrix is put where its tensor lies,
             as the file's BF16 holds it, and read back value for value; the manifest beside the file names the tensor as being written
             before the first byte moves, so a write that was cut is seen and never taken for a state. Taken up by another process, the
             file is held to its manifest before anything else: a path or a time never says what a file holds. A stop is undone in
             the one way that needs no guess: the copy is made under another name and takes the file's name only when it is whole,
             and the tensor a cut write names is given the original's bytes again (`mended`).
  kept       binds a state's name to its whole file's sha256, with every byte outside the changed tensors compared with the original's.
             A state that is kept, read by a reader that takes a file's sha256, or sent anywhere is bound this way first.
  a start    a working file may be made from a state another sweep reached instead of from the original: a pass of a correction that
             goes on from where the pass before it ended. The start is that state's file and its `kept` record. The new file is a copy
             of it, its name is still the original's sha256 and the tensors that differ from the original (the start's among them), and a
             cut write is undone to the start's bytes, which are the bytes the tensor held when this pass began. The original stays the
             one file every state is named from and compared with outside its changed tensors.

Nothing here chooses a matrix, a change or an order (records/development/2026-11-14-whole-gap-solver/README.md)."""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from pipeline.native_records.native_trace import checkpoint_values
from pipeline.weight_edit.a2 import to_bf16

KIND = 'original+tensors/1'
BLOCK = 1 << 24   # bytes hashed at once


def digest(path, spans=()):
    """A file's sha256, or with `spans` (sorted, apart) the sha256 of every byte outside them."""
    path, found, begin = Path(path), hashlib.sha256(), 0
    size = path.stat().st_size
    with path.open('rb') as file:
        for start, end in list(spans) + [(size, size)]:
            file.seek(begin)
            while begin < start:
                block = file.read(min(BLOCK, start - begin))
                if not block:
                    raise ValueError(f'{path.name} ends before byte {start}')
                found.update(block)
                begin += len(block)
            begin = end
    return found.hexdigest()


def identity(original_sha256, changed):
    """A state's name from the original's sha256 and its changed tensors ({name: native_type, shape, bytes, sha256})."""
    listed = [[name, each['native_type'], [int(n) for n in each['shape']], int(each['bytes']), each['sha256']] for name, each in sorted(changed.items())]
    encoding = json.dumps([KIND, original_sha256, listed], separators=(',', ':'), ensure_ascii=True)
    return dict(kind=KIND, sha256=hashlib.sha256(encoding.encode('ascii')).hexdigest(), original_sha256=original_sha256, tensors=[each[0] for each in listed])


def encoded(values, binding):
    """A matrix (input width x width, float32 values the BF16 holds) as its tensor's bytes in the file's own layout."""
    values = np.ascontiguousarray(values, dtype='<f4')
    if binding['native_type'] != 'bf16':
        raise ValueError(f"{binding['name']} is stored as {binding['native_type']}: only BF16 tensors are written")
    if list(values.shape) != [int(n) for n in binding['shape'][:2]]:
        raise ValueError(f"{binding['name']} is {binding['shape'][:2]} in the file, and the matrix given is {list(values.shape)}")
    bits = to_bf16(values.ravel(order='F'))
    back = (bits.astype('<u4') << 16).view('<f4')
    if not np.array_equal(back, values.ravel(order='F'), equal_nan=True):
        raise ValueError(f"{binding['name']}: the matrix given holds a value the file's BF16 does not hold")
    if bits.nbytes != binding['bytes']:
        raise ValueError(f"{binding['name']} takes {binding['bytes']} bytes in the file, and the matrix given {bits.nbytes}")
    return bits.tobytes()


def _span(path, binding):
    with Path(path).open('rb') as file:
        file.seek(binding['offset_bytes'])
        raw = file.read(binding['bytes'])
    if len(raw) != binding['bytes']:
        raise ValueError(f"{Path(path).name} ends inside {binding['name']}")
    return raw


class Working:
    """One file a sweep writes into, with its manifest (<file>.state.json): the original it copies, the changed tensors and the name of
    the state it holds. `bindings` is each tensor's place in the file ({name: name, native_type, shape, offset_bytes, bytes}), the same
    in the original and in every state of it."""

    def __init__(self, file, original, bindings, manifest):
        self.file, self.original, self.bindings, self.manifest = Path(file), Path(original), bindings, manifest
        self._originals = {}

    @staticmethod
    def beside(file):
        return Path(str(file) + '.state.json')

    @classmethod
    def make(cls, file, original, original_sha256, bindings, start=None):
        """A new working file: the original copied, after the original itself is held to the sha256 it is declared by. The copy is made
        under another name and takes the file's name once it is whole, so a file of that name with no manifest is a whole copy an earlier
        process was stopped before naming: it is held to the original's sha256 and taken, and any other file there is refused.
        With `start` (`file`: a state's file; `kept`: what Working.kept bound it to) the copy is of that state instead: the start's
        record is held to this original and to these tensors, its file to the record's sha256, and the new file begins with the
        start's changed tensors as its own."""
        file, original = Path(file), Path(original)
        if cls.beside(file).exists():
            raise ValueError(f'{file.name} exists: a working file is made once, and taken up after that')
        if file.resolve() == original.resolve():
            raise ValueError('the original is only read: a working file is another file')
        if digest(original) != original_sha256:
            raise ValueError('the original is not the file its sha256 declares')
        source, source_sha256, changed, begun = original, original_sha256, {}, None
        if start is not None:
            source, kept = Path(start['file']), start['kept']
            if file.resolve() == source.resolve():
                raise ValueError('the start is only read: a working file is another file')
            if kept.get('identity') != identity(original_sha256, kept.get('changed', {})) or not kept.get('every_other_byte_is_the_originals'):
                raise ValueError('the start\'s record does not bind a state of this original')
            if set(kept['changed']) - set(bindings):
                raise ValueError(f"the start changed {sorted(set(kept['changed']) - set(bindings))}, which this file's tensors do not name")
            if any(kept['changed'][name][key] != ([int(n) for n in bindings[name]['shape']] if key == 'shape' else bindings[name][key])
                   for name in kept['changed'] for key in ('native_type', 'shape', 'bytes')):
                raise ValueError('the start\'s changed tensors are not laid as this file\'s are')
            source_sha256, changed = kept['model_sha256'], {name: dict(each) for name, each in kept['changed'].items()}
            if digest(source) != source_sha256:
                raise ValueError('the start is not the file its record binds')
            begun = dict(file=str(source), model_sha256=source_sha256, identity=kept['identity'], changed={name: dict(each) for name, each in changed.items()})
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.exists():
            if digest(file) != source_sha256:
                raise ValueError(f'{file.name} exists with no manifest and is not the bytes it would be a copy of: nothing says what it holds')
        else:
            shutil.copyfile(source, str(file) + '.copying')
            os.replace(str(file) + '.copying', file)
        manifest = dict(kind=KIND, original=str(original), original_sha256=original_sha256, changed=changed, writing=None)
        if begun is not None:
            manifest['start'] = begun
        made = cls(file, original, bindings, manifest)
        made._save()
        return made

    @classmethod
    def taken_up(cls, file, original, bindings):
        """A working file another process left, held to its manifest: every changed tensor's bytes are the manifest's, and every other byte
        is the original's. A write that was cut is refused: the tensor it names holds bytes nobody recorded."""
        file, original = Path(file), Path(original)
        manifest = json.loads(cls.beside(file).read_text())
        if manifest['kind'] != KIND:
            raise ValueError(f"{file.name}'s manifest is of another kind: {manifest['kind']}")
        if manifest['writing'] is not None:
            raise ValueError(f"{file.name}: a write of {manifest['writing']} was cut, and the tensor holds bytes no record names")
        taken = cls(file, original, bindings, manifest)
        for name, each in manifest['changed'].items():
            if hashlib.sha256(_span(file, bindings[name])).hexdigest() != each['sha256']:
                raise ValueError(f'{file.name} does not hold the {name} its manifest names')
        taken._others_are_the_originals()
        return taken

    @classmethod
    def mended(cls, file, original, bindings):
        """`taken_up`, after a write that was cut is undone. The tensor such a write names holds bytes nobody recorded; it is given the
        bytes it held when the file was made (the original's, or its start's) and the state's name is the manifest's with that tensor as
        it began: for a tensor written once since the file was made, the state before the write. A file with no cut write is taken up as
        it is."""
        file, original = Path(file), Path(original)
        manifest = json.loads(cls.beside(file).read_text())
        name = manifest.get('writing')
        if manifest.get('kind') == KIND and name is not None:
            start = manifest.get('start')
            began = start['changed'].get(name) if start else None   # None: the tensor was the original's when the file was made
            raw = _span(Path(start['file']) if began is not None else original, bindings[name])
            if began is not None and hashlib.sha256(raw).hexdigest() != began['sha256']:
                raise ValueError(f'the start no longer holds the {name} its record names: the cut write cannot be undone from it')
            with file.open('r+b') as out:
                out.seek(bindings[name]['offset_bytes'])
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            if _span(file, bindings[name]) != raw:
                raise ValueError(f'{file.name} does not hold the {name} just written back')
            if began is not None:
                manifest['changed'][name] = dict(began)
            else:
                manifest['changed'].pop(name, None)
            manifest['writing'] = None
            cls(file, original, bindings, manifest)._save()
        return cls.taken_up(file, original, bindings)

    def _save(self):
        beside = self.beside(self.file)
        with open(str(beside) + '.next', 'w') as out:
            json.dump(dict(self.manifest, identity=self.identity()), out, indent=1)
            out.flush()
            os.fsync(out.fileno())
        os.replace(str(beside) + '.next', beside)

    def _spans(self):
        return sorted((self.bindings[name]['offset_bytes'], self.bindings[name]['offset_bytes'] + self.bindings[name]['bytes']) for name in self.manifest['changed'])

    def _others_are_the_originals(self):
        if self.file.stat().st_size != self.original.stat().st_size:
            raise ValueError(f'{self.file.name} is not the size of its original')
        spans = self._spans()
        if digest(self.file, spans) != digest(self.original, spans):
            raise ValueError(f'{self.file.name} differs from its original outside the tensors its manifest names')

    def identity(self):
        return identity(self.manifest['original_sha256'], self.manifest['changed'])

    def started(self):
        """The name of the state the file was made as: the original's, or its start's."""
        start = self.manifest.get('start')
        return identity(self.manifest['original_sha256'], start['changed'] if start else {})

    def moved(self, name):
        """Whether a tensor's bytes are no longer the ones the file was made with: the original's, or its start's."""
        start = self.manifest.get('start')
        return self.manifest['changed'].get(name) != (start['changed'].get(name) if start else None)

    def held(self, name):
        """A tensor of the file as the solver holds a matrix: input width x width, float32."""
        binding = self.bindings[name]
        return checkpoint_values(self.file, binding).reshape(binding['shape'][:2], order='F').astype('f4')

    def first(self, name):
        """A tensor of the original, as `held` gives one of the file."""
        binding = self.bindings[name]
        return checkpoint_values(self.original, binding).reshape(binding['shape'][:2], order='F').astype('f4')

    def begun(self, name):
        """A tensor as the file held it when it was made: the original's, or its start's where the start had changed it (held to the
        start's record, since the start's file is another file that only its own record binds)."""
        binding, start = self.bindings[name], self.manifest.get('start')
        began = start['changed'].get(name) if start else None
        if began is None:
            return self.first(name)
        if hashlib.sha256(_span(start['file'], binding)).hexdigest() != began['sha256']:
            raise ValueError(f'the start no longer holds the {name} its record names')
        return checkpoint_values(Path(start['file']), binding).reshape(binding['shape'][:2], order='F').astype('f4')

    def put(self, name, values):
        """A matrix written where its tensor lies and read back value for value. Returns the state's name after it. A matrix that is the
        original's again leaves the tensor out of the name."""
        binding = self.bindings[name]
        raw = encoded(values, binding)
        if name not in self._originals:
            self._originals[name] = hashlib.sha256(_span(self.original, binding)).hexdigest()
        self.manifest['writing'] = name
        self._save()
        with self.file.open('r+b') as file:
            file.seek(binding['offset_bytes'])
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        if _span(self.file, binding) != raw:
            raise ValueError(f'{self.file.name} does not hold the {name} just written')
        found = hashlib.sha256(raw).hexdigest()
        if found == self._originals[name]:
            self.manifest['changed'].pop(name, None)
        else:
            self.manifest['changed'][name] = dict(native_type=binding['native_type'], shape=[int(n) for n in binding['shape']], bytes=int(binding['bytes']), sha256=found)
        self.manifest['writing'] = None
        self._save()
        return self.identity()

    def kept(self):
        """The state's name bound to its whole file's sha256, every byte outside the changed tensors being the original's."""
        self._others_are_the_originals()
        for name, each in self.manifest['changed'].items():
            if hashlib.sha256(_span(self.file, self.bindings[name])).hexdigest() != each['sha256']:
                raise ValueError(f'{self.file.name} does not hold the {name} its manifest names')
        return dict(identity=self.identity(), model_sha256=digest(self.file), changed=self.manifest['changed'], every_other_byte_is_the_originals=True)
