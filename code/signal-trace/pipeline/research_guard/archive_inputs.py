"""The operator's side of input archives: make a record of a local input, send its files, bind the bucket's read-back to the record,
take the bound files off this disk, bring an archived input back. The checks are in input_archives.py, which the verifier and every native run import; nothing imports
this file, so it does not travel to a rented box (it names the bucket tool).

  python3 archive_inputs.py check                          the registry's problems
  python3 archive_inputs.py make <input> <membership.json> an unverified record of a local input (hashes each archivable file once)
  python3 archive_inputs.py make-storage <input> <membership.json>   the same for a storage-only backup: its answers may be ones no fit uses
  python3 archive_inputs.py make-model <file> <provenance.json>      an unverified record of a model file, with its provenance
  python3 archive_inputs.py files <name>                   the inventory's files below studies/, one per line
  python3 archive_inputs.py upload <name>                  send the inventory's files, and nothing else, to the bucket
  python3 archive_inputs.py verify <name>                  read the bucket's bytes of the inventory's files back and bind the result
  python3 archive_inputs.py release <name> [--active [<run> ...]]   take a bound inventory's files off this disk (what it removes: below)
  python3 archive_inputs.py restore <name> [--fetch]       hold the files to the inventory; --fetch downloads the missing ones first
  python3 archive_inputs.py citations <path>               the declared parameters whose citation holds <path>, and which a refinement narrows
  python3 archive_inputs.py refine <declaration> <parameter> <stated.json>   a parameter's citation narrowed to its sources (parameter_sources.py)

One inventory serves all of it: `upload`, `verify` and `restore --fetch` name the record's files to the bucket tool one by one and
never a folder, so nothing the inventory leaves out (a sealed or no-fit name, a file of mixed or unresolved membership, an object
someone else put under the same folder) is sent, read or pulled.

The order, for an input that only finished runs declare (speed review, declared-inputs/declared-inputs.txt, made again that day) and
whose scope Astra has named: write its membership (which answers its files hold, from an unsealed index or declaration), then make,
upload, verify. Only after verify has bound may a file of the inventory leave this disk (release), by the rule in STATE.md and only
for what Astra has named. Before any run reads the input again: restore --fetch.

<membership.json> is {"sources": ["<an unsealed index or declaration>", ...], "answers": ["<id>", ...], "pooled": {"<a file or
folder below the input, or '.'>": ["<id>", ...]}}: input_archives.py says how a file's answers are read from it. A file it does not
resolve is not inventoried and stays here.

`release` is the one command here that removes anything, and it removes only this: files of a bound inventory, each by its own path,
each still the file the inventory hashed and the bucket's bytes were compared with (device, inode, size and both times, as the receipt
holds them). It removes no folder and follows no link. It removes nothing at all when the record does not load whole (a membership
source is then not the file the record hashed), when no read-back is bound, when the registry does not keep the text of the read-back
the record is bound to, or when any inventoried file still here is another file. What it did is kept in the registry
(releases/<name>.json). No other command deletes a file.
"""
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:                                # a member of the pipeline package, or a script beside its sibling
    from pipeline.research_guard import input_archives as archives
    from pipeline.research_guard import parameter_sources as sources
    from pipeline.research_guard import storage_archives as storage
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import input_archives as archives
    import parameter_sources as sources
    import storage_archives as storage

TOOL = 'torad-research'
POINTER = 8                        # bytes of one argv or envp pointer on this 64-bit machine (struct.calcsize('P'))
ONE_ARGUMENT = 32 * 4096           # Linux's limit on one argument string: MAX_ARG_STRLEN, 32 pages (execve(2))
VERIFICATIONS = 'verifications'    # the bucket tool's read-back output of each bound record, kept in the registry
RELEASES = 'releases'              # what each release took off this disk, kept in the registry


def room(fixed, env):
    """The bytes one call's file arguments may take on this machine: what the system allows a new program's arguments and environment
    together (sysconf SC_ARG_MAX), less the environment this call passes, less its fixed arguments, less the 2048 bytes POSIX has
    xargs leave free. Every string costs its bytes, its end byte and its pointer."""
    cost = lambda text: len(os.fsencode(text)) + 1 + POINTER
    return os.sysconf('SC_ARG_MAX') - 2048 - sum(cost(f'{key}={value}') for key, value in env.items()) - sum(map(cost, fixed)) - POINTER * 2


def calls(root, record, subs, space, alone=False):
    """The inventory's files in calls whose arguments fit in `space` bytes (`room`), never two names of one file (hard links) in a
    call: the bucket tool stores a second name sent in the same call as a link, and each file of an inventory is a file of its own.
    `alone` gives every file a call of its own."""
    batch, seen, used = [], set(), 0
    for sub in subs:
        try:
            status = os.lstat(Path(root) / archives.below(record['input'], sub))
            key = (status.st_dev, status.st_ino)
        except OSError:
            key = None
        cost = len(os.fsencode(archives.below(record['bucket']['location'], sub))) + 1 + POINTER
        if cost > min(space, ONE_ARGUMENT):
            raise ValueError(f'{sub}: its path does not fit in one argument of a call on this machine')
        if batch and (alone or used + cost > space or (key is not None and key in seen)):
            yield batch
            batch, seen, used = [], set(), 0
        batch.append(sub)
        seen.add(key)
        used += cost
    if batch:
        yield batch


def streams():
    """How many streams of calls run at once: one for each processor thread this process may run on (sched_getaffinity), so whoever
    starts a command sets it with the threads they give it (taskset). The bucket tool takes one file at a time. Measured November
    13, 2026 on this machine's link: one stream sent an inventory of 9,405 files of 4.4 MB at 6.4 MB/s and six streams sent it at
    31 MB/s, each stream's process at about 3% of a thread: a small file's round trips set the pace, neither the processor nor the
    link (two streams of large files together sent 78 MB/s). The count is not a measured optimum: it is the threads the command
    was given."""
    return len(os.sched_getaffinity(0))


def shares(record, subs, count):
    """The files in at most `count` shares of near-equal bytes, each in the order given: every file goes, in its turn, to the share
    that holds the fewest bytes so far."""
    parts = [[] for _ in range(max(1, min(count, len(subs))))]
    held = [0] * len(parts)
    for sub in subs:
        least = held.index(min(held))
        parts[least].append(sub)
        held[least] += record['files'][sub]['bytes']
    return [part for part in parts if part]


def given_back(root, record, subs):
    """Advise the kernel to discard cached pages of inventory files after a completed transfer call. The speed review's November 13
    cache record (README.md:1885) measured a capture's model at 0.77 to 0.81 resident during a transfer, with 3.87 GB reread in 24 s;
    during the separate give-back loop, then with no transfer, it was whole in every sample. CPU load also affected that capture.
    This advice runs after a whole call: it cannot prevent cache pressure during a large in-flight batch, and the kernel may retain
    pages despite DONTNEED. A read-only descriptor changes no file content or identity, follows no link, and skips missing files.
    """
    for sub in subs:
        try:
            handle = os.open(Path(root) / archives.below(record['input'], sub), os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        finally:
            os.close(handle)


def bucket(root, verb, record, subs, alone=False):
    """The bucket tool on files of the inventory and on nothing else: each is named to it by its own path, never its folder. A call
    takes as many files as its arguments have room for on this machine (`room`); `alone` names one file a call. The files go in
    `streams` shares at once, each share's calls one after another, and after each call its files' cached pages are given back
    (`given_back`). Returns (the tool's output, share by share, and how many calls failed)."""
    strangers = [sub for sub in subs if sub not in record['files']]
    if strangers:
        raise ValueError(f'{strangers[0]} is not in the inventory of {record["name"]}')
    env = dict(os.environ, TORAD_RESEARCH_TRACES=str(Path(root) / archives.TRACES_ROOT))
    fixed = [TOOL, 'traces', verb]
    space = room(fixed, env)

    def stream(part):
        said, failed = [], 0
        for batch in calls(root, record, part, space, alone):
            done = subprocess.run(fixed + [archives.below(record['bucket']['location'], sub) for sub in batch],
                                  env=env, capture_output=True, text=True)
            said += [done.stdout, done.stderr]
            failed += done.returncode != 0
            given_back(root, record, batch)
        return ''.join(said), failed

    parts = shares(record, subs, streams())
    if not parts:
        return '', 0
    with ThreadPoolExecutor(len(parts)) as pool:
        told = list(pool.map(stream, parts))
    return ''.join(said for said, _ in told), sum(failed for _, failed in told)


def download(root, record, missing):
    """restore's fetch: the inventory's missing files, one a call, so each comes back as a plain file (the bucket tool brings a
    stored link back as a symlink when its target comes in the same call). The bucket tool keeps a file
    only when it matches its sha256 and overwrites nothing; restore then holds each to the inventory."""
    said, failed = bucket(root, 'download', record, missing, alone=True)
    if failed:
        print(said, file=sys.stderr)


def stated(root, membership, names):
    """An operator's membership as a record holds it, its sources hashed, and as `answers_of` reads it: ((the two), problems)."""
    if not isinstance(membership, dict) or not archives.strings(membership.get('sources')):
        return None, ['a membership needs its sources: the unsealed index or declaration that says which answers the files hold']
    sources = {}
    for source in membership['sources']:
        if archives.clean(source) is None or archives.excluded_part(source, names) or not archives.resolves_to_itself(root, source) \
                or not (Path(root) / source).is_file():
            return None, [f'membership source {source!r} is not a clean, unsealed repository file that resolves to itself']
        sources[source] = archives.digest(Path(root) / source)
    made = dict(sources=sources, answers=membership.get('answers'), pooled=membership.get('pooled', {}))
    checked, problems = archives.membership_of(root, made, names)
    return (made, checked), problems


def make(root, item, membership, backup=False):
    """An unverified record of a local input: the bytes and sha256 of every file its membership resolves. It stands for nothing that
    is missing until `verify` binds. Whether the path may be an archive's input at all is settled from its spelling and the way it
    resolves, before anything at or below it is listed or read. Below a folder, a sealed or no-fit name, a link, a special file, a
    record that stays and a file of mixed or unresolved membership are not inventoried and none of them is read: they stay here, and
    the record counts them. Returns (the record's name or None, problems).

    `backup` makes a storage-only backup (storage_archives.py): the input must lie at or below a source the registry names for it,
    its membership may name answers no fit uses and a no-fit name is inventoried; a sealed name is refused and left as before."""
    records, _, method, keep = archives.registry(root)
    names, problem = archives.exclusions(root)
    item = archives.clean(item)
    if names is None or item is None:
        return None, [problem or 'the input is not a clean repository-relative path (nothing was listed or read)']
    listed_names, names = names, storage.sealed_only(names) if backup else names
    refused = archives.origin(root, item, method, keep, names)
    if backup:
        named, problem = storage.named_sources(archives, root)
        if named is None:
            refused.append(problem)
        elif not any(item == source or item.startswith(source + '/') for source in named):
            refused.append(f'input {item} is not at or below a source {archives.REGISTRY}/{storage.NAMED} names for storage-only backup')
    refused += [f'{r["name"]} already holds {r["input"]}' for r in records
                if r['input'] == item or r['input'].startswith(item + '/') or item.startswith(r['input'] + '/')]
    if refused:
        return None, [f'{problem} (nothing was listed or read)' for problem in refused]
    both, refused = stated(root, membership, names)
    if refused:
        return None, [f'{problem} (of the input nothing was listed or read)' for problem in refused]
    members, checked = both
    path = Path(root) / item
    try:
        mode = os.lstat(path).st_mode
    except OSError:
        mode = 0
    if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
        return None, [f'{item} is not a file or folder on this disk']
    left, listed = {}, []
    leave = lambda why: left.__setitem__(why, left.get(why, 0) + 1)

    def place(sub, whole):
        if archives.stays(whole, keep) or whole in checked['sources']:
            leave('a record that stays on this disk')
        elif archives.answers_of(whole, sub, checked) is None:
            leave('membership mixed or unresolved')
        else:
            listed.append(sub)

    if stat.S_ISREG(mode):
        place('.', item)
    else:
        for folder, folders, here in os.walk(path):                      # a link to a folder is listed by its name and never entered
            inside = Path(folder).relative_to(path).as_posix()
            inside = '' if inside == '.' else inside + '/'
            entered = []
            for name in sorted(folders):
                if archives.excluded_part(name, names):
                    leave('a sealed or no-fit name')
                elif os.path.islink(os.path.join(folder, name)):
                    leave('a link or a special file')
                else:
                    entered.append(name)
            folders[:] = entered
            for name in sorted(here):
                if archives.excluded_part(name, names):
                    leave('a sealed or no-fit name')
                else:
                    place(inside + name, f'{item}/{inside}{name}')
    files, stamps, problems = {}, {}, []
    for sub in listed:
        value, seen = archives.read_held(Path(root) / archives.below(item, sub))
        if value is None and seen == 'is not a plain file':
            leave('a link or a special file')
        elif value is None:
            problems.append(f'{archives.below(item, sub)} {seen}')
        else:
            files[sub] = dict(bytes=seen['bytes'], sha256=value)
            stamps[sub] = seen
    if not files and not problems:
        problems.append(f'{item}: nothing at or below it may be archived ({left or "it is empty"})')
    name = f'{hashlib.sha256(item.encode()).hexdigest()[:12]}-{Path(item).name}.json'
    location = item[len(archives.TRACES_ROOT) + 1:]
    record = dict(form=archives.FORM, input=item, kind='folder' if stat.S_ISDIR(mode) else 'file',
                  bucket=dict(traces_root=archives.TRACES_ROOT, location=location), membership=members, files=files,
                  inventory_sha256=archives.inventory_sha256(files), left_on_this_disk=left, remote_bytes_verified=None,
                  restore=f'python3 code/signal-trace/pipeline/research_guard/archive_inputs.py restore {name} --fetch')
    if backup:
        record.update(form=storage.FORM, artifact=storage.BACKUP[0], storage_role=storage.BACKUP[1], eligibility=storage.NO_GRANT,
                      membership=dict(members, cohort=storage.cohort(members['answers'], listed_names)))
    problems = problems or archives.check_record(root, name, record, method, keep, listed_names)
    if problems:
        return None, problems
    archives.write(Path(root) / archives.REGISTRY / name, record)
    archives.stamp(root, dict(record, name=name), stamps)
    return name, []


def make_model(root, item, provenance):
    """An unverified record of a model file (storage_archives.py): its bytes and sha256, and in place of a membership the records
    that say what produced it, each bound by its sha256 and staying here. `provenance` is {"sources": [<the producing run's
    declaration.json, its result, the record that states the written file's sha256>, ...], "produced_by": <that declaration>,
    "written": <the source stating the file's sha256>, "original": {"sha256": <the model it was made from>, "stated_by": <a source>}}.
    Whether the path may be archived is settled from its spelling before the file is read. Returns (the record's name or None, problems)."""
    records, _, method, keep = archives.registry(root)
    names, problem = archives.exclusions(root)
    item = archives.clean(item)
    if names is None or item is None:
        return None, [problem or 'the input is not a clean repository-relative path (nothing was listed or read)']
    refused = archives.origin(root, item, method, keep, names)
    refused += [f'{r["name"]} already holds {r["input"]}' for r in records
                if r['input'] == item or r['input'].startswith(item + '/') or item.startswith(r['input'] + '/')]
    if not (isinstance(provenance, dict) and archives.strings(provenance.get('sources'))):
        refused.append('a model archive needs its provenance sources: the records that say what produced the file')
    if refused:
        return None, [f'{problem} (nothing was listed or read)' for problem in refused]
    sources = {}
    for source in provenance['sources']:
        if archives.clean(source) is None or archives.excluded_part(source, names) or not archives.resolves_to_itself(root, source) \
                or not (Path(root) / source).is_file():
            return None, [f'provenance source {source!r} is not a clean, unsealed repository file that resolves to itself (of the input nothing was read)']
        sources[source] = archives.digest(Path(root) / source)
    value, seen = archives.read_held(Path(root) / item)
    if value is None:
        return None, [f'{item} {seen}']
    files = {'.': dict(bytes=seen['bytes'], sha256=value)}
    name = f'{hashlib.sha256(item.encode()).hexdigest()[:12]}-{Path(item).name}.json'
    record = dict(form=storage.FORM, artifact=storage.MODEL[0], storage_role=storage.MODEL[1], input=item, kind='file',
                  bucket=dict(traces_root=archives.TRACES_ROOT, location=item[len(archives.TRACES_ROOT) + 1:]),
                  provenance=dict(answer_rows=storage.NO_ROWS, sources=sources, produced_by=provenance.get('produced_by'),
                                  written=provenance.get('written'), original=provenance.get('original')),
                  files=files, inventory_sha256=archives.inventory_sha256(files), left_on_this_disk={}, remote_bytes_verified=None,
                  restore=f'python3 code/signal-trace/pipeline/research_guard/archive_inputs.py restore {name} --fetch')
    problems = archives.check_record(root, name, record, method, keep, names)
    if problems:
        return None, problems
    archives.write(Path(root) / archives.REGISTRY / name, record)
    archives.stamp(root, dict(record, name=name), {'.': seen})
    return name, []


def needed(root, active):
    """The declared inputs and targets of the runs named active, repository-relative as they resolve: (paths, problems)."""
    runs = archives.guard().load_runs(Path(root), json.loads((Path(root) / archives.METHOD).read_text()))
    unknown = [name for name in active if not (runs.get(name) or {}).get('declaration')]
    if unknown:
        return [], [f'{unknown[0]} is not a declared run of round two; nothing is released']
    paths = set()
    for name in active:
        declaration = runs[name]['declaration']
        target = declaration.get('target_source')
        for item in [i for i in declaration.get('inputs') or [] if isinstance(i, str)] + ([target] if isinstance(target, str) else []):
            rel = archives.relative(root, Path(root) / item)
            if rel:
                paths.add(rel)
    return sorted(paths), []


def attach(root, name, verify_output, command, passed_ct):
    """Bind a passed read-back to a record: the bucket tool said `verified` for every file of the inventory and failed none, and the
    local files it compared the bucket's bytes with are still, by their whole identity, the ones the inventory hashed."""
    record, problems = archives.one(root, name)
    if record is None:
        return problems
    lines = [line.strip() for line in verify_output.splitlines()]
    if any('FAILED' in line for line in lines):
        return [f'input archive {name}: the verification reports a failure']
    said = {re.split(r'\s{2,}| -> ', line[len('verified'):].strip())[0] for line in lines if line.startswith('verified ')}
    missed = [sub for sub in record['files'] if archives.below(record['bucket']['location'], sub) not in said]
    if missed:
        return [f'input archive {name}: the verification does not say verified for {len(missed)} inventoried file(s), first {missed[0]}']
    moved = archives.unchanged(root, record)
    if moved:
        return [f'input archive {name}: {len(moved)} file(s) are not the ones its inventory hashed (first {moved[0]}); run restore, then verify again']
    stored = json.loads((Path(root) / archives.REGISTRY / name).read_text())
    stored['remote_bytes_verified'] = dict(passed_ct=passed_ct, command=command, files=len(record['files']),
                                           bytes=sum(e['bytes'] for e in record['files'].values()),
                                           inventory_sha256=record['inventory_sha256'],
                                           output_sha256=hashlib.sha256(verify_output.encode()).hexdigest())
    archives.write(Path(root) / archives.REGISTRY / name, stored)
    return []


def verify(root, name):
    """Read the bucket's bytes of every inventoried file back (the bucket tool's own verify, on those files only), keep what it said in
    the registry, and bind it to the record when it passed for all of them."""
    record, problems = archives.one(root, name)
    if record is None:
        return problems
    said, failed = bucket(root, 'verify', record, sorted(record['files']))
    kept = Path(root) / archives.REGISTRY / VERIFICATIONS / (name[:-len('.json')] + '.txt')
    kept.parent.mkdir(parents=True, exist_ok=True)
    kept.write_text(said)
    if failed:
        return [f'input archive {name}: {failed} verify call(s) failed; what the tool said is in {kept.relative_to(root)}']
    return attach(root, name, said, f'the bucket tool\'s traces verify on each file of {name}; its output: {kept.relative_to(root).as_posix()}',
                  time.strftime('%Y-%m-%d %H:%M:%S %Z'))


def release(root, name, active=None):
    """Take a bound inventory's files off this disk, each by its own path, and keep what was done in the registry. Returns (what was
    done, problems). With any problem found before the first removal nothing is removed: the record must load whole (its membership
    sources are then the files it hashed), carry a read-back bound to its inventory whose text the registry keeps, and every
    inventoried file still here must be, by its whole identity, the one the receipt holds: the file the inventory hashed and the
    bucket's bytes were compared with. A file already gone is passed over; no folder is removed and no link followed.

    `active` names the runs that are executing or still to read: a file at or below a declared input or target of one of them stays,
    whole, whatever copy the bucket holds (a copy and a release are two decisions). A storage-only backup is released only with it
    given, an empty list when no run is active; the receipt keeps the names."""
    record, problems = archives.one(root, name)
    if record is None:
        return None, problems
    if active is None and (record.get('artifact'), record.get('storage_role')) == storage.BACKUP:
        return None, [f'input archive {name}: a storage-only backup is released only with the runs active now named '
                      '(--active <run> ..., or --active alone when none is); nothing is released']
    needs, problems = needed(root, active) if active is not None else ([], [])
    if problems:
        return None, [f'input archive {name}: {problem}' for problem in problems]
    bound = record.get('remote_bytes_verified')
    if bound is None:
        return None, [f'input archive {name}: no read-back of the bucket\'s bytes is bound to it; nothing is released']
    kept = Path(root) / archives.REGISTRY / VERIFICATIONS / (name[:-len('.json')] + '.txt')
    try:
        said = hashlib.sha256(kept.read_bytes()).hexdigest()
    except OSError:
        said = None
    if said != bound.get('output_sha256'):
        return None, [f'input archive {name}: the registry does not keep the verification its record is bound to '
                      f'({kept.relative_to(root).as_posix()}); nothing is released']
    receipt = archives.receipt_of(root, record)
    here, gone, moved, kept_for_a_run = [], 0, [], 0
    for sub, entry in sorted(record['files'].items()):
        whole = archives.below(record['input'], sub)
        if archives.on_disk(root, whole, entry) == 'absent':
            gone += 1
        elif any(whole == path or whole.startswith(path + '/') for path in needs):
            kept_for_a_run += 1
        elif archives.same_file(root, whole, receipt.get(sub)):
            here.append((sub, whole, entry['bytes']))
        else:
            moved.append(whole)
    if moved:
        return None, [f'input archive {name}: {len(moved)} file(s) here are not the ones its inventory hashed (first {moved[0]}); '
                      f'nothing is released. Run restore, then verify again']
    files, size, problems = 0, 0, []
    for sub, whole, count in here:
        if not archives.same_file(root, whole, receipt.get(sub)):                # looked at again at the moment it goes
            problems.append(f'input archive {name}: {whole} changed while its inventory was being released; it and what follows it stay')
            break
        try:
            os.unlink(Path(root) / whole)
        except OSError as error:
            problems.append(f'input archive {name}: cannot remove {whole} ({type(error).__name__}: {error}); '
                            'it and what follows it stay')
            break
        files, size = files + 1, size + count
    done: dict[str, object] = dict(released_ct=time.strftime('%Y-%m-%d %H:%M:%S %Z'), files_released=files, bytes_released=size, already_gone=gone,
                still_here=len(here) - files, read_back_passed_ct=bound['passed_ct'])
    if active is not None:
        done.update(active_runs=sorted(active), kept_for_an_active_run=kept_for_a_run)
    if problems:
        done['problems'] = problems
    note = Path(root) / archives.REGISTRY / RELEASES / name
    try:
        earlier = json.loads(note.read_text())['releases']
    except (OSError, ValueError, KeyError):
        earlier = []
    if files:
        archives.write(note, dict(record=name, input=record['input'], inventory_sha256=record['inventory_sha256'],
                                  restore=record['restore'], releases=earlier + [done]))
    if files + gone == len(record['files']):                                     # nothing of it is here: its receipt holds nothing
        (Path(root) / archives.REGISTRY / archives.RECEIPTS / name).unlink(missing_ok=True)
    return done, problems


def refinements(root, deep=False):
    """(the citations whole refinements narrow, the problems of the others); `deep` hashes every named source file."""
    method = json.loads((Path(root) / archives.METHOD).read_text())
    return sources.narrowed(archives, root, method, archives.guard().load_runs(Path(root), method), deep)


def citing(root, path):
    """Print every declared parameter whose citation is, or holds, a path, counted from the declarations, and whether a whole
    refinement narrows it. The path stays on this disk while one of them is whole. Returns the refinements' problems."""
    method = json.loads((Path(root) / archives.METHOD).read_text())
    runs = archives.guard().load_runs(Path(root), method)
    narrowed, problems = sources.narrowed(archives, root, method, runs, deep=True)
    whole = 0
    for run, key, cited in sources.citations(archives, root, method, runs):
        if path == cited or path.startswith(cited + '/'):
            kept = narrowed.get((run, key))
            whole += kept is None
            print(f'{run} / {key}: cites {cited}; ' + ('whole' if kept is None else f'narrowed to {len(kept)} sources and evidence records'))
    print(f'{path}: {"stays whole, " + str(whole) + " citation(s) are not narrowed" if whole else "no whole citation holds it"}')
    return problems


def refine(root, declaration, parameter, stated):
    """Write a parameter's refinement, made and checked whole by parameter_sources.made (every named file hashed after its name has
    passed the spelling, link and seal checks). An existing refinement of that parameter is never overwritten. Returns (the file's
    name or None, problems)."""
    record, problems = sources.made(archives, root, declaration, parameter, stated)
    if record is None:
        return None, problems
    run = declaration.split('/')[-2]
    name = f'{run}--{parameter}.json'
    folder = Path(root) / archives.REGISTRY / sources.FOLDER
    folder.mkdir(exist_ok=True)
    try:
        with open(folder / name, 'x') as out:
            json.dump(record, out, indent=1)
            out.write('\n')
    except FileExistsError:
        return None, [f'{archives.REGISTRY}/{sources.FOLDER}/{name} exists: a refinement is not overwritten']
    return name, []


def main(arguments):
    root = Path(__file__).resolve().parents[4]
    if arguments == ['check']:
        problems = archives.registry(root)[1] + refinements(root, deep=True)[1]
    elif arguments[:1] == ['citations'] and len(arguments) == 2:
        problems = citing(root, arguments[1])
    elif arguments[:1] == ['refine'] and len(arguments) == 4:
        name, problems = refine(root, arguments[1], arguments[2], json.loads(Path(arguments[3]).read_text()))
        print(f'written: {archives.REGISTRY}/{sources.FOLDER}/{name}' if name else 'no refinement written')
    elif arguments[:1] in (['make'], ['make-storage'], ['make-model']) and len(arguments) == 3:
        stated_by = json.loads(Path(arguments[2]).read_text())
        name, problems = make_model(root, arguments[1], stated_by) if arguments[0] == 'make-model' \
            else make(root, arguments[1], stated_by, backup=arguments[0] == 'make-storage')
        print(f'written, unverified: {archives.REGISTRY}/{name}' if name else 'no record written')
        if name:
            print('left on this disk:', json.loads((root / archives.REGISTRY / name).read_text())['left_on_this_disk'])
    elif arguments[:1] in (['files'], ['upload'], ['verify']) and len(arguments) == 2:
        record, problems = archives.one(root, arguments[1])
        if record and arguments[0] == 'files':
            for sub in sorted(record['files']):
                print(archives.below(record['bucket']['location'], sub))
        elif record and arguments[0] == 'upload':
            said, failed = bucket(root, 'upload', record, sorted(record['files']))
            print(said)
            problems = [f'{failed} upload call(s) failed'] if failed else []
        elif record:
            problems = verify(root, arguments[1])
    elif arguments[:1] == ['release'] and (len(arguments) == 2 or arguments[2] == '--active'):
        done, problems = release(root, arguments[1], arguments[3:] if len(arguments) > 2 else None)
        if done is not None:
            print(json.dumps(done))
    elif arguments[:1] == ['restore'] and arguments[2:] in ([], ['--fetch']) and len(arguments) >= 2:
        problems = archives.restore(root, arguments[1], download if arguments[2:] else None)
    else:
        raise SystemExit(__doc__)
    for problem in problems:
        print(problem, file=sys.stderr)
    raise SystemExit(1 if problems else 0)


if __name__ == '__main__':
    main(sys.argv[1:])
