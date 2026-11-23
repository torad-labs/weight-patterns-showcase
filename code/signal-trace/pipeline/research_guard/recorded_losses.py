"""Declared inputs and record sources that no store holds any more, named one by one with a written reason.

Why it exists: on November 19, 2026 the round-two tree left this disk with no record. Everything git, the bucket, the November 18
publication archive and the local MLflow store held was put back (records/maintenance-runs/2026-11-19-evening-studies-deletion/).
A few dozen files were in none of them. The declarations that name them as inputs, and the archive records bound to them as
sources, are historical records that must not be rewritten, so the verifier is told, path by path, which absences are known and
which historical record each one is known to.

The method file names one list and binds it by sha256:

  "recorded_losses": {"record": "<repository-relative JSON file>", "sha256": "<sha256 of that file's bytes>", "why": "<one sentence>"}

and the list is

  {"form": "recorded-losses/1",
   "paths": {"<a path below the round-two runs>": {
       "sha256": "<the lost file's recorded sha256>" | null,
       "why": "<what it was>",
       "declared_by": {"<run>": "<sha256 of that run's declaration.json>", ...},
       "bound_by": {"<archive record's file name>": "<sha256 of that record file>", ...}}, ...}}

What holds:
 - With no `recorded_losses` in the method file nothing here reads a file and every check is exactly what it was.
 - The list is the bytes the method file hashed. A changed list stands for nothing until the method file is changed with it.
 - Every entry is a clean path below the runs root, names nothing sealed, says why, and names at least one declaration or archive
   record that needs it. A path with no recorded hash says null.
 - A listed path that is on this disk is a problem: what came back is taken off the list. So is one that cannot be looked at.
 - A loss is known only to the records its entry names, as the bytes it names: a run's declaration.json with the sha256 listed
   for that run (`declared`, asked by verify_round2.check_run), and an archive record file with the sha256 listed for it (`bound`,
   asked by input_archives.load). A new declaration, a new archive record, or an old one whose bytes changed gets nothing from the
   list: for it the path is simply missing.
 - To those records a loss excuses that path's absence and nothing else. Where an input comes from, a run's declaration and result,
   parameters and their citations are checked as before. For a model archive record whose `written` source is lost, the one thing
   that source alone could show (that it states the inventoried file's sha256) is not asked; the record is the bytes the list
   hashed, so its inventory is the one it had (storage_archives.check_record).
 - No native run may read a lost path: run_declaration.check_native_run refuses a run whose declaration names, as an input, as its
   target or as a path in any other field, a listed path, anything below one, or a folder that holds one, under any spelling or
   link that resolves to it, or names another run whose folder holds one (`named_in`, `touching`). Paths are compared as this file
   system resolves them: on one that ignores letter case a differently cased spelling would not be caught. What this cannot see is
   a path a driver builds from something the declaration does not state (a run name held in another record, a name fixed in code).
"""
import hashlib
import json
import os
import re
from pathlib import Path

FORM = 'recorded-losses/1'
METHOD = 'studies/04-round-2-overlap/round-2-method.json'
EXCLUDED = 'studies/04-round-2-overlap/input-archives/excluded.json'
HEX = re.compile(r'[0-9a-f]{64}\Z')
# What the verifier adds when a missing input is listed, but for another run or for other bytes of this run's declaration: the list was
# not made for the declaration as it reads now. The answer is to restore the declaration's listed bytes, never to rebuild the list.
NOT_THIS_DECLARATION = '; the recorded losses list it, but not for this declaration as it reads now'
NOT_THIS_RECORD = 'the recorded losses list a source of this record, but not for this record file as it reads now'


def clean(text):
    """A relative path in its one spelling, or None."""
    if not isinstance(text, str) or not text or '\\' in text or text.startswith('/'):
        return None
    return None if any(part in ('', '.', '..') for part in text.split('/')) else text


def sealed_prefixes(root):
    """The sealed name prefixes of the registry's exclusion list; 'reserved' when the list cannot be read."""
    try:
        listed = json.loads((Path(root) / EXCLUDED).read_text()).get('sealed_name_prefixes')
    except (OSError, ValueError, AttributeError):
        listed = None
    return [p.lower() for p in listed] if isinstance(listed, list) and listed and all(isinstance(p, str) and p for p in listed) else ['reserved']


def seen(path):
    """'here', 'gone', or 'unknown' when the path cannot be looked at (a loss is listed only when its absence is seen)."""
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return 'gone'
    except OSError:
        return 'unknown'
    return 'here'


def names_to_hashes(value):
    """Whether a value is a table of plain names to sha256 (it may be empty)."""
    return isinstance(value, dict) and all(isinstance(name, str) and name and '/' not in name and isinstance(digest, str) and HEX.match(digest)
                                           for name, digest in value.items())


def load(root, method):
    """({lost path: its entry}, problems). A list with any problem stands for nothing."""
    stated = method.get('recorded_losses')
    if stated is None:
        return {}, []
    say = lambda text: f'recorded losses: {text}'
    if not (isinstance(stated, dict) and clean(stated.get('record')) and isinstance(stated.get('sha256'), str)
            and HEX.match(stated['sha256']) and isinstance(stated.get('why'), str) and stated['why'].strip()):
        return {}, [say('the method file must give the list as record, sha256 and why')]
    try:
        raw = (Path(root) / stated['record']).read_bytes()
    except OSError:
        return {}, [say(f'{stated["record"]} cannot be read')]
    if hashlib.sha256(raw).hexdigest() != stated['sha256']:
        return {}, [say(f'{stated["record"]} is not the file the method file hashed')]
    try:
        listed = json.loads(raw)
    except ValueError:
        return {}, [say(f'{stated["record"]} is not JSON')]
    paths = listed.get('paths') if isinstance(listed, dict) else None
    if not isinstance(listed, dict) or listed.get('form') != FORM or not isinstance(paths, dict) or not paths:
        return {}, [say(f'{stated["record"]} is not a {FORM} list of paths')]
    base, sealed, problems, lost = method['runs_root'] + '/', sealed_prefixes(root), [], {}
    for path, entry in paths.items():
        if clean(path) is None or not path.startswith(base):
            problems.append(say(f'{path!r} is not a clean path below {base}'))
        elif any(prefix in part.lower() for part in path.split('/') for prefix in sealed):
            problems.append(say('an entry names sealed material, which is never listed'))
        elif not (isinstance(entry, dict) and isinstance(entry.get('why'), str) and entry['why'].strip()
                  and 'sha256' in entry and (entry['sha256'] is None or (isinstance(entry['sha256'], str) and HEX.match(entry['sha256'])))):
            problems.append(say(f'{path}: needs why, and sha256 as 64 hex digits or null'))
        elif not (names_to_hashes(entry.get('declared_by')) and names_to_hashes(entry.get('bound_by'))
                  and (entry['declared_by'] or entry['bound_by'])):
            problems.append(say(f'{path}: needs declared_by and bound_by, each a table of names to sha256, and at least one record that names it'))
        elif (state := seen(Path(root) / path)) == 'here':
            problems.append(say(f'{path} is on this disk; take it off the list'))
        elif state == 'unknown':
            problems.append(say(f'{path} cannot be looked at; only an absence that is seen is listed'))
        else:
            lost[path] = dict(sha256=entry['sha256'], declared_by=dict(entry['declared_by']), bound_by=dict(entry['bound_by']))
    return ({}, problems) if problems else (lost, [])


def named(root):
    """The lost paths of a repository as its own method file gives them (none when the list has a problem)."""
    return load(root, json.loads((Path(root) / METHOD).read_text()))[0]


def declared(lost, item, run, declaration):
    """Whether a missing input is a loss known to this very declaration: listed for this run with the sha256 of its declaration.json."""
    listed = ((lost or {}).get(item) or {}).get('declared_by', {}).get(run)
    if not listed:
        return False
    try:
        return hashlib.sha256(Path(declaration).read_bytes()).hexdigest() == listed
    except OSError:
        return False


def bound(lost, name, raw):
    """{lost source: its recorded sha256} for one archive record: the losses the list binds to a record file of this name and these bytes."""
    if not lost:
        return {}
    digest = hashlib.sha256(raw).hexdigest()
    return {path: entry['sha256'] for path, entry in lost.items() if entry['bound_by'].get(name) == digest}


def resolved(root, item):
    """A declared path as it resolves below the repository ('' for the repository itself), or None when it leaves it or cannot be resolved."""
    try:
        root, path = Path(root).resolve(), (Path(root) / item).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if path == root:
        return ''
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else None


def touching(lost, root, item, as_resolved=None):
    """The lost paths a declared path cannot be read without: the path itself, any listed path below it, or a listed folder above it.
    Both sides are compared as they are spelled and as they resolve, so another spelling of a path, or a link to it, is that path.
    `as_resolved` is {lost path: how it resolves}, for a caller that asks of many paths."""
    if not lost or not isinstance(item, str) or not item:
        return []
    spelled = {item.rstrip('/') or None, resolved(root, item)} - {None}
    if '' in spelled:
        return sorted(lost)                                     # the repository itself holds every loss
    def same(path):
        return any(path == s or path.startswith(s + '/') or s.startswith(path + '/') for s in spelled)
    if as_resolved is None:
        as_resolved = {path: resolved(root, path) or path for path in lost}
    return sorted(path for path in lost if same(path) or same(as_resolved[path]))


def strings(value, where=''):
    """Every string in a declaration, with the place it stands (`inputs[2]`, `parameters.step.measured_from`)."""
    if isinstance(value, str):
        yield where, value
    elif isinstance(value, dict):
        for key, inner in value.items():
            yield from strings(inner, f'{where}.{key}' if where else str(key))
    elif isinstance(value, list):
        for index, inner in enumerate(value):
            yield from strings(inner, f'{where}[{index}]')


def named_in(lost, root, declaration, runs_root=None, own=None):
    """What a declaration names that cannot be read without a loss, one line each: the place, the path and the losses it touches.
    Inputs and the target are held whole, in whatever shape they are written. Any other string is held when it is written as a path
    (it has a '/'): a driver may read a checkpoint, a reference or a definition from a field the verifier does not know. A string
    with no '/' is held when it is the name of a run whose folder holds a loss (`runs_root` given): drivers join a run's name with a
    file's. The run's own name (`own`) is not held, so a run may write again what its own folder lost. Outputs are written, not
    read, and are left out."""
    if not lost:
        return []
    as_resolved = {path: resolved(root, path) or path for path in lost}
    holding, base = {}, (runs_root.rstrip('/') + '/') if runs_root else None
    for path in lost if base else ():
        if path.startswith(base):
            holding.setdefault(path[len(base):].split('/')[0], []).append(path)
    found = []
    for where, text in strings(declaration):
        field = where.split('.')[0].split('[')[0]
        if field == 'outputs':
            continue
        if field in ('inputs', 'target_source') or '/' in text:
            held = touching(lost, root, text, as_resolved)
        else:
            held = sorted(holding.get(text, [])) if text != own else []
        if held:
            found.append(f'{where}: {text}: {", ".join(held)}')
    return found
