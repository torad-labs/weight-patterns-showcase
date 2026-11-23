"""A verified bucket copy standing for a declared input: the record, its checks, readiness before a native run, and restore.

The contract is Astra's (records/retrospectives/2026-11-13-declared-input-archives-astra.md, 59e34c96, and its four boundary repairs of
November 13, 2026, 9:25 PM CT), built at its request by weight-research-perfomance. With no record in the registry every check of the
verifier and of a native run is exactly what it was. Nothing here uploads, deletes or releases a file, and nothing here contacts the
bucket.

The verifier and every native run import this file, so it travels to a rented box with a solve: it names no tool of ours and nothing
private. The operator's commands (make a record, send, read back, fetch) are in archive_inputs.py beside it, which nothing imports.

The registry is studies/04-round-2-overlap/input-archives: one <name>.json per archived input, and excluded.json. A record is

  {"form": "declared-input-archive/2",
   "input": "<the archived file or folder, repository-relative, as it resolves: no link, no '..'>",
   "kind": "file" | "folder",
   "bucket": {"traces_root": "studies", "location": "<the input below studies/, where the bucket tool keeps it>"},
   "membership": {"sources": {"<an unsealed index or declaration, repository-relative>": "<its sha256>", ...},
                  "answers": ["<every answer the inventoried files hold, as the sources name them>", ...],
                  "pooled": {"<a file or folder below the input that pools answers; '.' for the input>": ["<answer>", ...], ...}},
   "files": {"<path below the input; '.' for a file>": {"bytes": n, "sha256": "<64 hex>"}, ...},
   "inventory_sha256": "<sha256 of the sorted path, bytes, sha256 lines>",
   "remote_bytes_verified": null | {"passed_ct": ..., "command": ..., "files": n, "bytes": n, "inventory_sha256": ...},
   "restore": "<the command that brings the bytes back>"}

Astra's five conditions, and where each is kept:
 1. Exact inventory and binding. `origin` and `check_record`: the input in its one spelling, inside the round-two runs or the read-only
    inputs, no link on the way to it; the bucket location is that input and no other; every file has its path, bytes and sha256; the
    remote verification names the files, the bytes and the hash of THIS inventory, so a receipt of another inventory, or a bare `true`,
    is refused. Two records over one path are refused. Sending, reading back and fetching go file by file from this inventory and from
    nothing else (archive_inputs.py), so an object of the bucket the inventory does not name is never asked for.
 2. A folder that is still here is checked all the same. `historical` looks at every inventoried file at or below a declared path,
    whether or not the path exists: a file that is gone needs the record's remote verification, a file that is here must have its
    inventory's byte count (its content is held by sha256 at restore, before a native run reads it: the record check hashes nothing
    of an input). A declared child missing from this disk is covered only when the inventory holds that very file.
 3. Nothing else is skipped. The verifier's own checks of where an input comes from, of the source run, of the target and of the
    original model run for an archived path as for a present one (verify_round2.check_run). A run's declaration.json and result.json,
    every record a declared parameter cites, a membership's sources and the registry itself may not be in an inventory: they stay.
 4. A record is not readiness. `unrestored`, called by run_declaration.check_native_run for the run that is about to compute: every
    inventoried file of that run's own declared inputs and target must be on this disk and be, by device, inode, size, modification
    time and change time, the file restore held to its inventory (`identity`; the receipt). A replacement of the same size with its
    time put back is another inode or another change time, so it ends the receipt without any file being hashed at a run's start. So
    does a new hard link to the file, a change of its mode, or a device renumbered by a restart: restore again. No commit contacts
    anything.
 5. Restore. `restore` holds every file to its inventory by sha256 (`read_held`: opened without following a link, the same identity
    before and after the read), overwrites nothing (a file in the way that is not its inventory's is reported and left), and writes
    its receipt only when every file is right.

Never archived, and never listed, opened or read by `make`:
 - anything whose path has a part that begins with a sealed name (`reserved`), or holds one after a character that is no letter or
   digit, and anything whose path names an answer no fit uses as a whole word (the id itself, `<id>.own`, `<id>-output-norms.npy`,
   `000-<id>--<other id>.json`). The names are in excluded.json; without it no record is accepted.
 - a file whose answers the record's membership does not resolve. A file's answers are those its path names, of the membership's
   `answers`; or, when it names none, those of the nearest `pooled` entry at or above it; an exact `pooled` entry for the file itself
   comes first. No answer of a membership may be one no fit uses, so a file that pools a no-fit answer, or whose answers no unsealed
   index or declaration states, has no place in an inventory: mixed and unresolved membership stays on this disk. The sources are
   bound by sha256 and stay here. What the check cannot see: whether a source tells the truth about an array. Whoever writes the
   membership answers for that; the check holds that every archived file has a stated membership and that it names no no-fit answer.
A second name of a sealed file under another name (a hard link) is not seen: nothing here looks inside a sealed folder to compare.

Two more kinds of record are form 3, in storage_archives.py beside this file (Astra's contract of November 14, 2026): a model file
with its provenance in place of a membership, and a storage-only backup of named, previously acquired, unsealed data that holds
answers no fit uses. `check_record` hands a form-3 record to that file; every other check here reads the fields the two forms share,
and a form-2 record means what it meant.
"""
import hashlib
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

try:                                # a member of the pipeline package, or a script's sibling
    from pipeline.research_guard import parameter_sources, recorded_losses, storage_archives
except ImportError:
    import parameter_sources
    import recorded_losses
    import storage_archives

FORM = 'declared-input-archive/2'
REGISTRY = 'studies/04-round-2-overlap/input-archives'
EXCLUDED = 'excluded.json'
RECEIPTS = '.restored'     # what restore verified on this disk: local, never committed
TRACES_ROOT = 'studies'
METHOD = 'studies/04-round-2-overlap/round-2-method.json'
HEX = re.compile(r'[0-9a-f]{64}\Z')
IDENTITY = ('dev', 'ino', 'bytes', 'mtime_ns', 'ctime_ns')


def clean(text):
    """A relative path in its one spelling, or None: not absolute, no empty part, no '.' and no '..'."""
    if not isinstance(text, str) or not text or '\\' in text:
        return None
    return None if any(part in ('', '.', '..') for part in text.split('/')) else text


def relative(root, path):
    """A path as it resolves, relative to the repository, or None when it leaves it."""
    root, path = Path(root).resolve(), Path(path).resolve()
    return path.relative_to(root).as_posix() if path.is_relative_to(root) and path != root else None


def below(item, sub):
    return item if sub == '.' else f'{item}/{sub}'


def inventory_sha256(files):
    lines = ''.join(f'{path}\t{entry["bytes"]}\t{entry["sha256"]}\n' for path, entry in sorted(files.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def digest(path):
    """A small file's sha256 (a membership's source). An input's own files are read by `read_held`."""
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def strings(value):
    return isinstance(value, list) and bool(value) and all(isinstance(v, str) and v for v in value)


def words(names, whole=True):
    """A pattern that finds a name where no letter or digit stands before it and, for a whole word, none after it."""
    return re.compile(r'(?<![0-9A-Za-z])(?:' + '|'.join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + ')'
                      + (r'(?![0-9A-Za-z])' if whole else ''), 0 if whole else re.IGNORECASE)


def exclusions(root):
    """The names no archive may hold, or (None, why not)."""
    try:
        listed = json.loads((Path(root) / REGISTRY / EXCLUDED).read_text())
    except (OSError, ValueError) as error:
        return None, f'{REGISTRY}/{EXCLUDED} cannot be read ({type(error).__name__}); no archive record is accepted without it'
    if not isinstance(listed, dict) or not strings(listed.get('sealed_name_prefixes')) or not strings(listed.get('no_fit_answers')):
        return None, f'{REGISTRY}/{EXCLUDED} lacks sealed_name_prefixes or no_fit_answers; no archive record is accepted without them'
    return dict(sealed=words(listed['sealed_name_prefixes'], whole=False), no_fit=set(listed['no_fit_answers']),
                no_fit_named=words(listed['no_fit_answers'])), None


def excluded_part(path, names):
    """The part of a path that names sealed or no-fit material, or None."""
    for part in path.split('/'):
        if names['sealed'].search(part) or names['no_fit_named'].search(part):
            return part
    return None


def kept_local(root, method, runs):
    """What stays on this disk whatever is archived: each run's declaration.json and result.json, every record a declared parameter
    cites, the method file and the registry. A citation that a whole refinement narrows (parameter_sources.py) is kept as that
    refinement's sources and evidence; every other citation is kept whole, so a path stays while any one parameter cites it whole."""
    base = method['runs_root']
    keep = {METHOD, REGISTRY}
    narrowed, _ = parameter_sources.narrowed(sys.modules[__name__], root, method, runs)
    for name, run in runs.items():
        keep |= {f'{base}/{name}/declaration.json', f'{base}/{name}/result.json'}
        parameters = (run.get('declaration') or {}).get('parameters')
        for key, value in parameters.items() if isinstance(parameters, dict) else ():
            cited = value.get('measured_from') if isinstance(value, dict) else None
            if isinstance(cited, str) and cited:
                if (name, key) in narrowed:
                    keep |= narrowed[(name, key)]
                    continue
                try:
                    keep.add(relative(root, Path(root) / cited) or cited)
                except (OSError, ValueError):
                    keep.add(cited)
    return keep


def stays(whole, keep):
    """Whether a path is one of the records that stay, or lies inside one."""
    parts = whole.split('/')
    return any('/'.join(parts[:n]) in keep for n in range(1, len(parts) + 1))


def resolves_to_itself(root, item):
    try:
        return relative(root, Path(root) / item) == item
    except (OSError, RuntimeError, ValueError):
        return False


def origin(root, item, method, keep, names):
    """Why a clean path may not be an archive's input, from its spelling and the way it resolves alone: nothing at or below it is
    listed, opened or read. `make` asks this before it walks or hashes anything; the verifier asks it of every record."""
    problems = []
    part = excluded_part(item, names)
    if part:
        problems.append(f'{part!r} names sealed or no-fit material, which is never archived')
    if not any(item.startswith(allowed + '/') for allowed in (method['runs_root'], method['read_only_inputs'])):
        problems.append(f'input {item} is outside the round-two runs and the read-only inputs')
    if not item.startswith(TRACES_ROOT + '/'):
        problems.append(f'input {item} is not below {TRACES_ROOT}/, where the bucket keeps it')
    if stays(item, keep):
        problems.append(f'input {item} is, or lies inside, a declaration, a result, a parameter record or archive metadata, which stay on this disk')
    if not resolves_to_itself(root, item):
        problems.append(f'input {item} is not the path it resolves to (a link on the way to it, or a path that leaves the repository)')
    return problems


def membership_of(root, stated, names):
    """A record's membership, checked: (what `answers_of` reads, problems). Its sources are read (small index files, each bound by its
    sha256); nothing of the input is."""
    if not (isinstance(stated, dict) and isinstance(stated.get('sources'), dict) and stated['sources']
            and strings(stated.get('answers')) and isinstance(stated.get('pooled'), dict)):
        return None, ['its membership needs sources, answers and pooled: which answers its files hold, and the unsealed index or declaration that says so']
    problems = []
    answers = set(stated['answers'])
    no_fit = sorted(answers & names['no_fit'])
    if no_fit:
        problems.append(f'its membership names {len(no_fit)} answer(s) no fit uses (first {no_fit[0]}); they are never archived')
    for key, held in stated['pooled'].items():
        if not (key == '.' or clean(key)) or not strings(held) or not set(held) <= answers:
            problems.append(f'its membership\'s pooled entry {key!r} is not a clean path below the input with answers of the membership')
    for source, recorded in stated['sources'].items():
        if clean(source) is None or excluded_part(source, names) or not resolves_to_itself(root, source):
            problems.append(f'its membership source {source!r} is not a clean, unsealed repository path that resolves to itself')
            continue
        try:
            plain = stat.S_ISREG(os.lstat(Path(root) / source).st_mode)
        except OSError:
            plain = False
        if not plain:
            problems.append(f'its membership source {source} is not on this disk as a plain file; a membership\'s sources stay here')
        elif not (isinstance(recorded, str) and HEX.match(recorded)) or digest(Path(root) / source) != recorded:
            problems.append(f'its membership source {source} is not the file the record hashed')
    return dict(named=words(answers), pooled=stated['pooled'], sources=set(stated['sources'])), problems


def answers_of(whole, sub, membership):
    """The answers a file holds, as a checked membership has them, or None when it does not resolve them: the file's own pooled entry,
    else the answers its path names, else the nearest pooled entry above it."""
    if sub in membership['pooled']:
        return membership['pooled'][sub]
    named = sorted(set(membership['named'].findall(whole)))
    if named:
        return named
    parts = sub.split('/')
    for n in range(len(parts) - 1, 0, -1):
        if '/'.join(parts[:n]) in membership['pooled']:
            return membership['pooled']['/'.join(parts[:n])]
    return membership['pooled'].get('.')


def check_record(root, name, record, method, keep, names, lost=None):
    """Every way a record fails to be an exact inventory bound to its input, its bucket location, its membership and its verification.
    `lost` is {source: sha256} of the recorded losses known to this very record file (recorded_losses.bound), nothing when the record is
    new or was changed: only a model archive's provenance source may be one of them."""
    say = lambda text: f'input archive {name}: {text}'
    if isinstance(record, dict) and record.get('form') == storage_archives.FORM:
        return storage_archives.check_record(sys.modules[__name__], root, name, record, method, keep, names, lost)
    if not isinstance(record, dict) or record.get('form') != FORM:
        return [say(f'is not a {FORM} record')]
    item, kind, files = clean(record.get('input')), record.get('kind'), record.get('files')
    if item is None:
        return [say('names no clean repository-relative input')]
    problems = [say(problem) for problem in origin(root, item, method, keep, names)]
    if kind not in ('file', 'folder'):
        problems.append(say('kind is neither file nor folder'))
    bucket = record.get('bucket')
    if not (isinstance(bucket, dict) and bucket.get('traces_root') == TRACES_ROOT and item.startswith(TRACES_ROOT + '/')
            and bucket.get('location') == item[len(TRACES_ROOT) + 1:]):
        problems.append(say(f'its bucket location is not the input below {TRACES_ROOT}/'))
    membership, found = membership_of(root, record.get('membership'), names)
    problems += [say(problem) for problem in found]
    if not isinstance(files, dict) or not files:
        return problems + [say('holds no inventory')]
    if problems:
        return problems
    if kind == 'file' and list(files) != ['.']:
        problems.append(say("a file's inventory is the one entry '.'"))
    unresolved = []
    for sub, entry in files.items():
        if kind != 'file' and clean(sub) is None:
            problems.append(say(f'file path {sub!r} is not a clean path below the input'))
            continue
        size = entry.get('bytes') if isinstance(entry, dict) else None
        if not (isinstance(size, int) and not isinstance(size, bool) and size >= 0
                and isinstance(entry.get('sha256'), str) and HEX.match(entry['sha256'])):
            problems.append(say(f'{sub}: needs its bytes and its sha256'))
            continue
        whole = below(item, sub)
        part = excluded_part(whole, names)
        if part:
            problems.append(say(f'{sub}: {part!r} names sealed or no-fit material, which is never archived'))
        elif stays(whole, keep) or whole in membership['sources']:
            problems.append(say(f'{sub}: a declaration, a result, a parameter record, a membership source or archive metadata stays on this disk'))
        elif answers_of(whole, sub, membership) is None:
            unresolved.append(sub)
    if unresolved:
        problems.append(say(f'{len(unresolved)} file(s) hold answers its membership does not resolve (first {unresolved[0]}); mixed or unresolved membership stays on this disk'))
    if problems:
        return problems
    if record.get('inventory_sha256') != inventory_sha256(files):
        problems.append(say('inventory_sha256 is not the hash of its files'))
    verified = record.get('remote_bytes_verified')
    if verified is not None and not (
            isinstance(verified, dict) and verified.get('passed_ct') and verified.get('command')
            and verified.get('files') == len(files) and verified.get('bytes') == sum(e['bytes'] for e in files.values())
            and verified.get('inventory_sha256') == record.get('inventory_sha256')):
        problems.append(say('its remote verification is not of this inventory (its files, bytes and inventory_sha256 must be the record\'s)'))
    return problems


def stated_sources(record):
    """The source paths a record states, whatever else is wrong with it (none when it states them in no readable form)."""
    stated = (record.get('membership') or record.get('provenance')) if isinstance(record, dict) else None
    sources = stated.get('sources') if isinstance(stated, dict) else None
    return [source for source in sources if isinstance(source, str)] if isinstance(sources, dict) else []


def load(root, method, runs, lost=None):
    """The registry's whole records, and the problems of the others. A record with a problem stands for nothing.
    `lost` is the recorded losses as recorded_losses.load gives them."""
    folder = Path(root) / REGISTRY
    files = sorted(p for p in folder.glob('*.json') if p.name not in (EXCLUDED, storage_archives.NAMED)) if folder.is_dir() else []
    if not files:
        return [], []
    names, problem = exclusions(root)
    if names is None:
        return [], [f'input archives: {problem}']
    keep, records, problems = kept_local(root, method, runs), [], []
    for file in files:
        try:
            raw = file.read_bytes()
            record = json.loads(raw)
        except (OSError, ValueError):
            problems.append(f'input archive {file.name}: cannot be read as JSON')
            continue
        # A recorded loss is known to a record only when the list names this record file as these very bytes (recorded_losses.bound).
        known = recorded_losses.bound(lost, file.name, raw)
        found = check_record(root, file.name, record, method, keep, names, known)
        if found and lost and any(source in lost and source not in known for source in stated_sources(record)):
            found.append(f'input archive {file.name}: {recorded_losses.NOT_THIS_RECORD}')     # said beside the problem, never instead of it
        problems.extend(found)
        if not found:
            records.append(dict(record, name=file.name))
    sources = set().union(*(storage_archives.sources_of(record) for record in records)) if records else set()
    for i, a in enumerate(records):
        taken = sorted(sources & {below(a['input'], sub) for sub in a['files']})
        if taken:
            problems.append(f'input archive {a["name"]}: {taken[0]} is a bound source of an archive and stays on this disk')
        for b in records[i + 1:]:
            if a['input'] == b['input'] or a['input'].startswith(b['input'] + '/') or b['input'].startswith(a['input'] + '/'):
                problems.append(f'input archives {a["name"]} and {b["name"]}: two records over one path ({a["input"]}, {b["input"]})')
    return records, problems


def entries(records, rel):
    """The inventoried files at or below a repository-relative path, each with its record, and whether a record holds the whole path
    (its input is the path or lies above it)."""
    found, whole_path = [], False
    for record in records:
        item = record['input']
        above = item == rel or rel.startswith(item + '/')
        if not above and not item.startswith(rel + '/'):
            continue
        mine = [(record, below(item, sub), entry) for sub, entry in record['files'].items()
                if below(item, sub) == rel or below(item, sub).startswith(rel + '/')]
        found += mine
        whole_path = whole_path or (above and bool(mine))
    return found, whole_path


def on_disk(root, whole, entry):
    """'present' when the file is here with its inventory's bytes, 'absent' when it is not here, else what is wrong with it."""
    try:
        status = os.lstat(Path(root) / whole)
    except (FileNotFoundError, NotADirectoryError):
        return 'absent'
    except OSError as error:
        return f'cannot be looked at here ({type(error).__name__})'
    if not stat.S_ISREG(status.st_mode):
        return 'is not a plain file here, as its inventory has it'
    return 'present' if status.st_size == entry['bytes'] else f'has {status.st_size} bytes here and {entry["bytes"]} in its inventory'


def historical(root, path, records):
    """For the record verifier. None when no archive record concerns the path: the caller's own existence check then holds, as it
    always did. Otherwise the path's problems (none: every inventoried file is here with its bytes, or gone with a verified record)."""
    if not records:
        return None
    rel = relative(root, path)
    found, whole_path = entries(records, rel) if rel else ([], False)
    if not found:
        return None
    problems = ['is not on this disk and no archive record holds the whole of it'] if not whole_path and not Path(path).exists() else []
    counts, first = {}, {}
    for record, whole, entry in found:
        state = on_disk(root, whole, entry)
        if state == 'present' or (state == 'absent' and record.get('remote_bytes_verified') is not None):
            continue
        key = (record['name'], 'not on this disk, and the archive carries no remote verification' if state == 'absent' else state)
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, whole)
    return problems + [f'{count} file(s) {state} (archive {name}), first {first[(name, state)]}' for (name, state), count in counts.items()]


def identity(status):
    """What makes a file on this disk the same file: its device, inode, size, modification time and change time."""
    return dict(dev=status.st_dev, ino=status.st_ino, bytes=status.st_size, mtime_ns=status.st_mtime_ns, ctime_ns=status.st_ctime_ns)


def same_file(root, whole, known):
    """Whether the file here is, by its whole identity, the one a receipt holds. Nothing is hashed."""
    try:
        return isinstance(known, dict) and identity(os.lstat(Path(root) / whole)) == {key: known.get(key) for key in IDENTITY}
    except OSError:
        return False


def read_held(file):
    """A plain file's sha256 with the identity it kept through the read: (sha256, identity), or (None, why not). The file is opened
    without following a link, and what was opened is what was looked at before and what stands there after."""
    try:
        before = os.lstat(file)
        if not stat.S_ISREG(before.st_mode):
            return None, 'is not a plain file'
        handle = os.open(file, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        return None, f'cannot be read ({type(error).__name__})'
    with os.fdopen(handle, 'rb') as source:
        opened = os.fstat(source.fileno())
        value = hashlib.file_digest(source, 'sha256').hexdigest()
        closing = os.fstat(source.fileno())
    try:
        after = os.lstat(file)
    except OSError:
        return None, 'changed while it was read'
    if not identity(before) == identity(opened) == identity(closing) == identity(after):
        return None, 'changed while it was read'
    return value, identity(before)


def receipt_of(root, record):
    try:
        receipt = json.loads((Path(root) / REGISTRY / RECEIPTS / record['name']).read_text())
    except (OSError, ValueError):
        return {}
    return receipt.get('files', {}) if isinstance(receipt, dict) and receipt.get('inventory_sha256') == record['inventory_sha256'] else {}


def not_ready(root, declaration, records):
    """For a native run: what keeps this run's own declared inputs and target from being read now. One line per input and record."""
    target = declaration.get('target_source')
    problems = []
    for item in [i for i in declaration.get('inputs') or [] if isinstance(i, str)] + ([target] if isinstance(target, str) else []):
        rel = relative(root, Path(root) / item)
        found = entries(records, rel)[0] if rel else []
        counts, first, receipts = {}, {}, {}
        for record, whole, entry in found:
            state = on_disk(root, whole, entry)
            if state == 'present':
                if record['name'] not in receipts:
                    receipts[record['name']] = receipt_of(root, record)
                sub = '.' if whole == record['input'] else whole[len(record['input']) + 1:]
                if same_file(root, whole, receipts[record['name']].get(sub)):
                    continue
                state = 'not verified against its inventory since it came back'
            elif state == 'absent':
                state = 'in the bucket, not on this disk'
            key = (record['name'], state)
            counts[key] = counts.get(key, 0) + 1
            first.setdefault(key, whole)
        for (name, state), count in counts.items():
            problems.append(f'input {item}: {count} file(s) {state}, first {first[(name, state)]}; restore it first: '
                            f'python3 code/signal-trace/pipeline/research_guard/archive_inputs.py restore {name} --fetch')
    return problems


def guard():
    """The verifier's module, as a package member or as this script's sibling."""
    try:
        from pipeline.research_guard import verify_round2
    except ImportError:
        import verify_round2
    return verify_round2


def unrestored(root, declaration):
    """What run_declaration.check_native_run asks before a run computes: nothing when the registry holds no record."""
    folder = Path(root) / REGISTRY
    if not folder.is_dir() or not any(p.name not in (EXCLUDED, storage_archives.NAMED) for p in folder.glob('*.json')):
        return []
    return not_ready(root, declaration, registry(root)[0])


def registry(root):
    """A repository's whole records, the registry's problems, the method and what stays on this disk, as the verifier sees them."""
    method = json.loads((Path(root) / METHOD).read_text())
    runs = guard().load_runs(Path(root), method)
    records, problems = load(root, method, runs, recorded_losses.load(root, method)[0])
    return records, problems, method, kept_local(root, method, runs)


def one(root, name):
    """A whole record by its file's name, or (None, why not)."""
    records, problems, _, _ = registry(root)
    for record in records:
        if record['name'] == name:
            return record, []
    return None, [p for p in problems if name in p] or [f'input archive {name}: no whole record of that name']


def write(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    scratch = file.with_name(file.name + '.writing')
    scratch.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(scratch, file)


def held(root, record):
    """Every inventoried file against its inventory, by sha256. Returns (problems, each right file's identity)."""
    problems, stamps = [], {}
    for sub, entry in sorted(record['files'].items()):
        whole = below(record['input'], sub)
        state = on_disk(root, whole, entry)
        if state == 'absent':
            problems.append(f'{whole} is not on this disk')
        elif state != 'present':
            problems.append(f'{whole} {state}; it is left as it is')
        else:
            value, seen = read_held(Path(root) / whole)
            if value is None:
                problems.append(f'{whole} {seen}')
            elif value != entry['sha256']:
                problems.append(f'{whole} has its inventory\'s bytes and another sha256; it is left as it is')
            else:
                stamps[sub] = seen
    return problems, stamps


def stamp(root, record, stamps):
    write(Path(root) / REGISTRY / RECEIPTS / record['name'],
          dict(inventory_sha256=record['inventory_sha256'], verified_ct=time.strftime('%Y-%m-%d %H:%M:%S %Z'), files=stamps))


def unchanged(root, record):
    """The inventoried files that are not, by their whole identity, the ones the receipt holds (every file when there is no receipt)."""
    seen = receipt_of(root, record)
    return [below(record['input'], sub) for sub in record['files'] if not same_file(root, below(record['input'], sub), seen.get(sub))]


def restore(root, name, fetch=None):
    """Hold a record's files to its inventory. A fetch, when given, is asked for the inventory's missing files and for nothing else.
    Nothing here is overwritten. With no problem the receipt is written and the input is ready; with any, no receipt stands."""
    record, problems = one(root, name)
    if record is None:
        return problems
    (Path(root) / REGISTRY / RECEIPTS / name).unlink(missing_ok=True)
    missing = sorted(sub for sub, entry in record['files'].items() if on_disk(root, below(record['input'], sub), entry) == 'absent')
    if fetch and missing:
        fetch(root, record, missing)
    problems, stamps = held(root, record)
    if not problems:
        stamp(root, record, stamps)
    return problems
