"""Two more kinds of archive record beside the answer-row record of input_archives.py: a model file, and a storage-only backup of
previously acquired, unsealed data that holds answers no fit uses.

The contract is Astra's (records/retrospectives/2026-11-14-storage-only-archives-astra.md), built at its request by
weight-research-perfomance. It keeps apart three things the first form held in one membership field: what the archived thing is
(`artifact`), what the copy is for (`storage_role`), and whether a run still needs the bytes here, which no record states: it is
settled when a file is released, from the runs named active at that moment (archive_inputs.py).

A record of this form has every field the checks of history, readiness, restore and release read in a declared-input-archive/2
record (input, kind, bucket, files, inventory_sha256, remote_bytes_verified, restore), so each of those checks holds for it
unchanged: a file that is gone needs the read-back bound to this inventory, and a native run needs its own declared inputs back and
held to their inventory before it computes. It adds

  "form": "declared-input-archive/3", "artifact": ..., "storage_role": ...

and is one of two shapes, any other being refused.

A MODEL ("artifact": "model", "storage_role": "declared-input", "kind": "file"). A model file holds weights, not captured answer
rows, so it carries no membership at all: no answer is invented for it, no fitting ids are assigned to it, and an empty list is not
read as "none". It carries its provenance instead:

  "provenance": {"answer_rows": "<NO_ROWS, word for word>",
                 "sources": {"<a record that stays on this disk, repository-relative>": "<its sha256>", ...},
                 "produced_by": "<the declaration.json of the run the file lies in; one of the sources>",
                 "written": "<the source that states the written file's sha256>",
                 "original": {"sha256": "<the model it was made from>", "stated_by": "<the source that states it>"}}

The sources are bound by sha256 and stay here, as a membership's do. The written file's identity is held to them: the source named
`written` must state the inventoried file's sha256, and the source named by `original` must state the original's. What the check
cannot see is whether the sources tell the truth; and an archive says nothing of what the model is good for: its run's standing,
and any later use of it, are what they were.

A STORAGE-ONLY BACKUP ("artifact": "answer-rows", "storage_role": "storage-only"): a second copy of bytes that had one.

  "membership": {"sources": ..., "answers": ..., "pooled": ...,     as in form 2, except that an answer may be one no fit uses
                 "cohort": "no-fit" | "mixed"},                     what the exclusion list says of those answers
  "eligibility": "<NO_GRANT, word for word>"

Its input lies at or below a source the registry's storage-only.json names: the role does not spread to data acquired later. Every
inventoried file still resolves to stated answers: a file whose answers no unsealed index or declaration states is not a backup's
either, and nothing sealed is looked at to resolve one. A sealed name is refused exactly as before, from the path's spelling, before
anything is listed or read. A backup whose answers hold none no fit uses is an ordinary archive and is written in form 2. The record
grants nothing: no input eligibility, no fit, calibration, selection or evaluation. The exclusion list is not changed by it, and a
form-2 record keeps its meaning: it still refuses every no-fit name.

This file is imported by input_archives.py, so it travels where that file does: it names no tool of ours and nothing private. It
takes input_archives as an argument (`archives`) and imports nothing of the pipeline.
"""
import json
import os
import re
import stat
from pathlib import Path

FORM = 'declared-input-archive/3'
NAMED = 'storage-only.json'         # in the registry: the sources a storage-only backup may hold, and the decision that names them
MODEL = ('model', 'declared-input')
BACKUP = ('answer-rows', 'storage-only')
NO_ROWS = 'not applicable: a model file holds weights, not captured answer rows'
NO_GRANT = 'none: a backup of bytes; it admits nothing as an input, a fit, a calibration, a selection or an evaluation'
NEVER = re.compile(r'(?!)')         # matches nothing


def sealed_only(names):
    """The exclusions with the no-fit names taken out: what a storage-only backup is held to. The sealed names stay."""
    return dict(names, no_fit=set(), no_fit_named=NEVER)


def cohort(answers, names):
    """What the exclusion list says of a set of answers: 'fitting' when it names none of them, 'no-fit' when it names all, else 'mixed'."""
    held = set(answers) & names['no_fit']
    return 'fitting' if not held else 'no-fit' if held == set(answers) else 'mixed'


def named_sources(archives, root):
    """The sources a storage-only backup may hold, or (None, why not)."""
    try:
        listed = json.loads((Path(root) / archives.REGISTRY / NAMED).read_text())
    except (OSError, ValueError) as error:
        return None, f'{archives.REGISTRY}/{NAMED} cannot be read ({type(error).__name__}); no storage-only backup is accepted without it'
    if not isinstance(listed, dict) or not archives.strings(listed.get('sources')) or not all(archives.clean(s) for s in listed['sources']):
        return None, f'{archives.REGISTRY}/{NAMED} lacks its sources, as clean repository paths; no storage-only backup is accepted without them'
    return listed['sources'], None


def sources_of(record):
    """The records a record is bound to and that stay on this disk: a membership's sources, or a model's provenance sources."""
    stated = record.get('membership') or record.get('provenance') or {}
    return stated.get('sources') if isinstance(stated.get('sources'), dict) else {}


def lost_as_bound(lost, source, recorded):
    """Whether a source is a recorded loss known to this record (recorded_losses.bound), listed with the very sha256 the record bound for it."""
    return bool(lost) and isinstance(recorded, str) and bool(recorded) and lost.get(source) == recorded


def held_sources(archives, root, sources, names, what, lost=None):
    """The problems of a set of bound sources: each a clean, unsealed repository file that resolves to itself, here, with its sha256.
    A source that is gone is no problem only when the recorded losses list it with the sha256 this record bound."""
    problems = []
    for source, recorded in sources.items():
        if archives.clean(source) is None or archives.excluded_part(source, names) or not archives.resolves_to_itself(root, source):
            problems.append(f'its {what} source {source!r} is not a clean, unsealed repository path that resolves to itself')
            continue
        try:
            plain = stat.S_ISREG(os.lstat(Path(root) / source).st_mode)
        except FileNotFoundError:
            if lost_as_bound(lost, source, recorded):
                continue
            plain = False
        except OSError:
            plain = False
        if not plain:
            problems.append(f'its {what} source {source} is not on this disk as a plain file; a record\'s sources stay here')
        elif not (isinstance(recorded, str) and archives.HEX.match(recorded)) or archives.digest(Path(root) / source) != recorded:
            problems.append(f'its {what} source {source} is not the file the record hashed')
    return problems


def states(root, source, value):
    """Whether a small source record holds a sha256 in its text."""
    try:
        return isinstance(value, str) and bool(value) and value in (Path(root) / source).read_text()
    except (OSError, ValueError):
        return False


def model(archives, root, record, item, method, names, lost=None):
    """A model archive's own problems, and its bound sources."""
    problems = []
    if 'membership' in record:
        problems.append('a model archive states no membership: answer rows are not applicable to a model file, and no list stands for that')
    if record.get('kind') != 'file':
        problems.append('a model archive holds one file')
    stated = record.get('provenance')
    if not (isinstance(stated, dict) and isinstance(stated.get('sources'), dict) and stated['sources']):
        return problems + ['a model archive needs its provenance: the records that say what produced the file, bound by sha256'], {}
    sources = stated['sources']
    if stated.get('answer_rows') != NO_ROWS:
        problems.append(f'its provenance must say of answer rows, word for word: {NO_ROWS}')
    problems += held_sources(archives, root, sources, names, 'provenance', lost)
    base = method['runs_root'] + '/'
    run = item[len(base):].split('/')[0] if item.startswith(base) else None
    if not run or stated.get('produced_by') != f'{base}{run}/declaration.json' or stated['produced_by'] not in sources:
        problems.append('its provenance\'s produced_by is not, among its sources, the declaration of the run the file lies in')
    if stated.get('written') not in sources:
        problems.append('its provenance\'s written is not one of its sources')
    original = stated.get('original')
    if not (isinstance(original, dict) and isinstance(original.get('sha256'), str) and archives.HEX.match(original['sha256'])
            and original.get('stated_by') in sources):
        problems.append('its provenance needs the original it was made from: its sha256 and the source that states it')
    elif not problems and not states(root, original['stated_by'], original['sha256']):
        problems.append(f'its provenance source {original["stated_by"]} does not state the original\'s sha256')
    return problems, sources


def backup(archives, root, record, item, seen, names):
    """A storage-only backup's own problems, its checked membership and its bound sources."""
    problems = []
    if record.get('eligibility') != NO_GRANT:
        problems.append(f'a storage-only backup must say of its eligibility, word for word: {NO_GRANT}')
    named, problem = named_sources(archives, root)
    if named is None:
        problems.append(problem)
    elif not any(item == source or item.startswith(source + '/') for source in named):
        problems.append(f'input {item} is not at or below a source {archives.REGISTRY}/{NAMED} names for storage-only backup')
    stated = record.get('membership')
    membership, found = archives.membership_of(root, stated, seen)
    problems += found
    if membership is None:
        return problems, None, {}
    said = cohort(stated['answers'], names)
    if said == 'fitting':
        problems.append('its membership names no answer the exclusion list holds: that is an ordinary archive, written in form 2')
    elif stated.get('cohort') != said:
        problems.append(f'its membership\'s cohort is stated as {stated.get("cohort")!r} and the exclusion list makes it {said!r}')
    return problems, membership, stated['sources']


def check_record(archives, root, name, record, method, keep, names, lost=None):
    """Every way a form-3 record fails to be an exact inventory bound to its input, its bucket location, what it is and its verification.
    `lost` is {source: sha256} of the recorded losses known to this very record file (recorded_losses.bound; empty for a new or changed
    record). A model archive's provenance source that is gone is accepted only when it is one of them with the sha256 the record bound,
    and what its `written` source alone could show (that it states the inventoried file's sha256) is then not asked: the record is the
    bytes the list hashed, so its inventory is the one it had. The source that states the original is never excused."""
    say = lambda text: f'input archive {name}: {text}'
    item, kind, files = archives.clean(record.get('input')), record.get('kind'), record.get('files')
    if item is None:
        return [say('names no clean repository-relative input')]
    shape = (record.get('artifact'), record.get('storage_role'))
    if shape == MODEL:
        seen, membership = names, None
        found, sources = model(archives, root, record, item, method, names, lost)
    elif shape == BACKUP:
        seen = sealed_only(names)
        found, membership, sources = backup(archives, root, record, item, seen, names)
    else:
        return [say('is neither a model archive (artifact model, storage_role declared-input) nor a storage-only backup '
                    '(artifact answer-rows, storage_role storage-only)')]
    problems = [say(problem) for problem in archives.origin(root, item, method, keep, seen)] + [say(problem) for problem in found]
    if kind not in ('file', 'folder'):
        problems.append(say('kind is neither file nor folder'))
    bucket = record.get('bucket')
    if not (isinstance(bucket, dict) and bucket.get('traces_root') == archives.TRACES_ROOT and item.startswith(archives.TRACES_ROOT + '/')
            and bucket.get('location') == item[len(archives.TRACES_ROOT) + 1:]):
        problems.append(say(f'its bucket location is not the input below {archives.TRACES_ROOT}/'))
    if not isinstance(files, dict) or not files:
        return problems + [say('holds no inventory')]
    if problems:
        return problems
    if kind == 'file' and list(files) != ['.']:
        problems.append(say("a file's inventory is the one entry '.'"))
    unresolved = []
    for sub, entry in files.items():
        if kind != 'file' and archives.clean(sub) is None:
            problems.append(say(f'file path {sub!r} is not a clean path below the input'))
            continue
        size = entry.get('bytes') if isinstance(entry, dict) else None
        if not (isinstance(size, int) and not isinstance(size, bool) and size >= 0
                and isinstance(entry.get('sha256'), str) and archives.HEX.match(entry['sha256'])):
            problems.append(say(f'{sub}: needs its bytes and its sha256'))
            continue
        whole = archives.below(item, sub)
        part = archives.excluded_part(whole, seen)
        if part:
            problems.append(say(f'{sub}: {part!r} names sealed{"" if shape == BACKUP else " or no-fit"} material, which is never archived'))
        elif archives.stays(whole, keep) or whole in sources:
            problems.append(say(f'{sub}: a declaration, a result, a parameter record, a bound source or archive metadata stays on this disk'))
        elif membership is not None and archives.answers_of(whole, sub, membership) is None:
            unresolved.append(sub)
    if unresolved:
        problems.append(say(f'{len(unresolved)} file(s) hold answers its membership does not resolve (first {unresolved[0]}); '
                            'unresolved membership stays on this disk'))
    if problems:
        return problems
    written = record['provenance']['written'] if shape == MODEL else None
    if (shape == MODEL and not lost_as_bound(lost, written, sources.get(written))
            and not states(root, written, files['.']['sha256'])):
        problems.append(say(f'its provenance source {record["provenance"]["written"]} does not state the sha256 of the file inventoried'))
    if record.get('inventory_sha256') != archives.inventory_sha256(files):
        problems.append(say('inventory_sha256 is not the hash of its files'))
    verified = record.get('remote_bytes_verified')
    if verified is not None and not (
            isinstance(verified, dict) and verified.get('passed_ct') and verified.get('command')
            and verified.get('files') == len(files) and verified.get('bytes') == sum(e['bytes'] for e in files.values())
            and verified.get('inventory_sha256') == record.get('inventory_sha256')):
        problems.append(say('its remote verification is not of this inventory (its files, bytes and inventory_sha256 must be the record\'s)'))
    return problems
