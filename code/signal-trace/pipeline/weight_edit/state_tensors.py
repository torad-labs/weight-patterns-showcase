"""A reached state kept without its whole file: its changed tensors' bytes, and the whole file made again from the original.

A state of the model is the original file with some tensors' bytes replaced (gap_state). A loop that publishes a state for every
update writes the whole file each time, and a few of those fill the disk. What a state holds that the original does not is its
changed tensors. `keep` writes those bytes under a folder with a record that binds them to the state (its name, its whole file's
sha256 and each tensor's sha256). Before it returns it reads the original and the kept bytes in the file's own order and holds
their sha256 to the state's: the file made again from this folder is the state's file byte for byte, shown without writing it.
`restore` makes the whole file again from the original and holds it to that record: it leaves the file that was kept, or no file.

Nothing here removes a state's file or says when one may leave: that is its loop's to decide, after `keep` has returned."""
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from pipeline.weight_edit import gap_state

FORMAT = 'state-tensors/1'
LAYOUT = ('native_type', 'shape', 'bytes')


def _save(path, record):
    with open(str(path) + '.next', 'w') as out:
        json.dump(record, out, indent=1)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(str(path) + '.next', path)


def _laid_as(changed, bindings):
    """Whether every changed tensor is laid in the file as its record says: the same type, shape and byte count."""
    return all(name in bindings and all(changed[name][key] == ([int(n) for n in bindings[name]['shape']] if key == 'shape' else bindings[name][key])
                                        for key in LAYOUT) for name in changed)


def keep(file, original, bindings, folder, kept=None) -> dict[str, Any]:
    """The changed tensors of the state in `file`, written under `folder` and read back. The folder is made here and is never
    written twice. `kept` is the state's Working.kept record where its maker already holds it; without one the file is held to
    its manifest here, which reads the whole file. Returns the record written as <folder>/record.json."""
    file, original, folder = Path(file), Path(original), Path(folder)
    state: dict[str, Any] = dict(gap_state.Working.taken_up(file, original, bindings).kept() if kept is None else kept)
    named: dict[str, Any] = dict(state.get('identity') or {})
    changed: dict[str, dict[str, Any]] = state['changed']
    if (not state.get('every_other_byte_is_the_originals') or not _laid_as(changed, bindings)
            or named != gap_state.identity(named.get('original_sha256'), changed)):
        raise ValueError('the record given does not bind a state of an original with these tensors')
    folder.mkdir(parents=True)
    try:
        tensors: dict[str, dict[str, int]] = {}
        at = 0
        with open(folder/'tensors.bin', 'wb') as out:
            for name in sorted(changed):
                raw = gap_state._span(file, bindings[name])
                if hashlib.sha256(raw).hexdigest() != changed[name]['sha256']:
                    raise ValueError(f'{file.name} does not hold the {name} its record names')
                out.write(raw)
                tensors[name] = dict(at=at, bytes=len(raw))
                at += len(raw)
            out.flush()
            os.fsync(out.fileno())
        record: dict[str, Any] = dict(format=FORMAT, original_sha256=named['original_sha256'], identity=named, model_sha256=state['model_sha256'],
                                      changed=changed, tensors=tensors, bytes=at, tensors_sha256=gap_state.digest(folder/'tensors.bin'))
        if made_again(folder, record, original, bindings) != state['model_sha256']:
            raise ValueError(f'made again from the original and these bytes, {file.name} would not be the state its record names')
        _save(folder/'record.json', record)
    except BaseException:
        shutil.rmtree(folder)       # made here a moment ago and bound by no record: a part of a state is not a state
        raise
    return record


def made_again(folder, record, original, bindings) -> str:
    """The sha256 the whole file has when it is made again: the original's bytes in the file's order, with every kept tensor's
    bytes where its tensor lies. Nothing is written."""
    found, begin, size = hashlib.sha256(), 0, Path(original).stat().st_size
    places = sorted((int(bindings[name]['offset_bytes']), name) for name in record['tensors'])
    with open(original, 'rb') as first, open(Path(folder)/'tensors.bin', 'rb') as kept:
        for start, name in [*places, (size, None)]:
            if start < begin:
                raise ValueError('two kept tensors lie over one another in the file')
            first.seek(begin)
            while begin < start:
                block = first.read(min(gap_state.BLOCK, start - begin))
                if not block:
                    raise ValueError(f'{Path(original).name} ends before byte {start}')
                found.update(block)
                begin += len(block)
            if name is not None:
                place = record['tensors'][name]
                kept.seek(place['at'])
                raw = kept.read(place['bytes'])
                if len(raw) != place['bytes'] or place['bytes'] != bindings[name]['bytes']:
                    raise ValueError(f'the kept bytes of {name} are not the size of its place in the file')
                found.update(raw)
                begin = start + place['bytes']
    return found.hexdigest()


def restore(folder, original, bindings, file) -> dict[str, Any]:
    """The whole file of a kept state, made again at `file` from the original and held to the state's record. Returns its
    Working.kept record. A file that turns out not to be the state's is removed and refused."""
    folder, original, file = Path(folder), Path(original), Path(file)
    record = json.loads((folder/'record.json').read_text())
    if record.get('format') != FORMAT or set(record['tensors']) != set(record['changed']) or not _laid_as(record['changed'], bindings):
        raise ValueError('not a kept state of a file with these tensors')
    if any(record['tensors'][name]['bytes'] != bindings[name]['bytes'] for name in record['tensors']):
        raise ValueError('a kept tensor is not the size of its place in the file')
    beside = gap_state.Working.beside(file)
    if file.exists() or beside.exists():
        raise ValueError(f'{file.name} exists: a state\'s file is made once')
    if gap_state.digest(folder/'tensors.bin') != record['tensors_sha256']:
        raise ValueError('the kept bytes are not the ones their record binds')
    if gap_state.digest(original) != record['original_sha256']:
        raise ValueError('the original is not the file the state was made from')
    file.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(str(file) + '.copying')
    shutil.copyfile(original, pending)
    with open(folder/'tensors.bin', 'rb') as source, pending.open('r+b') as out:
        for name, place in record['tensors'].items():
            source.seek(place['at'])
            out.seek(bindings[name]['offset_bytes'])
            out.write(source.read(place['bytes']))
        out.flush()
        os.fsync(out.fileno())
    os.replace(pending, file)
    manifest = dict(kind=gap_state.KIND, original=str(original), original_sha256=record['original_sha256'],
                    changed={name: dict(each) for name, each in record['changed'].items()}, writing=None)
    working = gap_state.Working(file, original, bindings, manifest)
    working._save()
    try:
        kept: dict[str, Any] = dict(working.kept())
        if kept['model_sha256'] != record['model_sha256'] or kept['identity'] != record['identity']:
            raise ValueError(f'{file.name} made again is not the state that was kept')
    except BaseException:
        file.unlink()
        beside.unlink()
        raise
    return kept
