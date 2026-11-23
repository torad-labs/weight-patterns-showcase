"""A declared parameter's citation narrowed to the records its derivation needs, by a refinement bound to the declaration's bytes.

The contract is Astra's (records/retrospectives/2026-11-14-parameter-source-refinement-astra.md), built at its request by
weight-research-perfomance. Why it exists: `input_archives.kept_local` keeps on this disk every record a declared parameter cites as its
`measured_from`, and seven parameters of five declarations cite a whole acquisition folder, so everything beneath that folder stays
whatever its files are. A refinement says which records under such a citation the parameter's derivation actually rests on. It changes
only what must stay on this disk as that parameter's evidence. It archives nothing, admits no answer, opens no sealed or no-fit name and
changes no declaration, value, formula or result.

A refinement is one file, <registry>/parameter-sources/<name>.json, which stays on this disk with the registry:

  {"form": "parameter-source-refinement/1",
   "declaration": "<runs root>/<run>/declaration.json",
   "declaration_sha256": "<sha256 of that file's bytes when the refinement was written>",
   "parameter": "<the parameter's key in that declaration>",
   "measured_from": "<the citation, exactly as the declaration states it>",
   "sources": {"<a file at or below the citation>": {"bytes": n, "sha256": "<64 hex>"},
               "<a folder at or below the citation, kept whole>": {"subtree": "<why its files are not named one by one>"}, ...},
   "evidence": {"<the producer's index or record that ties the sources to the parameter>": "<its sha256>", ...},
   "why": "<what the derivation needs and how the sources were taken from the evidence>"}

What holds, and where:
 - The binding. A refinement narrows a citation only while the declaration's file is the bytes it was bound to, the parameter is in it
   and its `measured_from` is the string the refinement states. A changed declaration, a renamed or removed parameter, another citation:
   the refinement stands for nothing and the citation is kept whole, as before it existed. So is a refinement with any other problem.
 - The sources are named, one path each: a file with its bytes and sha256, or a folder kept whole with the reason its files are not
   named. No wildcard, no empty set. Each is at or below the citation (a refinement narrows a citation; it does not move it), in its one
   spelling, with no link on the way, and names nothing sealed and no answer no fit uses: such a name is refused from its spelling,
   before the path is looked at. A file must be on this disk with its byte count; `deep` (the operator's check, never the verifier's)
   hashes it too. A named folder is never listed or entered here.
 - The evidence is bound by sha256 and read whole each time, like a membership's sources: small index files. It stays here too.
 - A refinement takes nothing back from an archive: a source or a piece of evidence that an archive record of the registry inventories
   refuses the refinement, never the record. So no refinement can turn a whole record into one that stands for nothing.
 - The union. `input_archives.kept_local` asks `narrowed` once and replaces, parameter by parameter, a cited path by its refinement's
   sources and evidence. A parameter with no whole refinement keeps its citation whole, so a folder cited by seven parameters stays
   whole until all seven are refined, and a new declaration that cites it keeps it whole again.

With no <registry>/parameter-sources folder nothing here reads a file and every check is exactly what it was."""
import json
import os
import stat
from pathlib import Path

FORM = 'parameter-source-refinement/1'
FOLDER = 'parameter-sources'
WILD = frozenset('*?[]{}')


def cited_path(archives, root, cited):
    """A citation as `kept_local` keeps it: the path it resolves to, or the string itself."""
    try:
        return archives.relative(root, Path(root) / cited) or cited
    except (OSError, ValueError):
        return cited


def citations(archives, root, method, runs):
    """Every (run, parameter, citation as kept) a declaration states, counted from the declarations."""
    found = []
    for name, run in sorted(runs.items()):
        parameters = (run.get('declaration') or {}).get('parameters')
        for key, value in sorted(parameters.items()) if isinstance(parameters, dict) else ():
            cited = value.get('measured_from') if isinstance(value, dict) else None
            if isinstance(cited, str) and cited:
                found.append((name, key, cited_path(archives, root, cited)))
    return found


def inventoried(archives, root):
    """Every path an archive record of the registry inventories, whole or not: a refinement gives way to all of them."""
    paths = set()
    for file in sorted((Path(root) / archives.REGISTRY).glob('*.json')):
        try:
            record = json.loads(file.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get('input'), str) and isinstance(record.get('files'), dict):
            paths.update(archives.below(record['input'], sub) for sub in record['files'] if isinstance(sub, str))
    return paths


def mode_of(root, path):
    try:
        return os.lstat(Path(root) / path)
    except OSError:
        return None


def named(archives, root, path, names):
    """Why a path may not be named at all, from its spelling and the way it resolves: nothing at it is opened."""
    if archives.clean(path) is None or WILD & set(path):
        return 'is not one path in its one spelling (a wildcard names no record)'
    part = archives.excluded_part(path, names)
    if part:
        return f'{part!r} names sealed or no-fit material, which a refinement neither names nor looks at'
    if not archives.resolves_to_itself(root, path):
        return 'is not the path it resolves to (a link on the way to it, or a path that leaves the repository)'
    return None


def check(archives, root, record, method, runs, names, archived, deep=False):
    """((run, parameter), the paths that stay for it) for a whole refinement, or (None, why it stands for nothing)."""
    if not isinstance(record, dict) or record.get('form') != FORM:
        return None, f'is not a {FORM} record'
    base, declaration, key = method['runs_root'], record.get('declaration'), record.get('parameter')
    tail = '/declaration.json'
    run = declaration[len(base) + 1:-len(tail)] if isinstance(declaration, str) and declaration.startswith(base + '/') and declaration.endswith(tail) else None
    if not run or '/' in run or run not in runs:
        return None, 'names no declaration of a run of this round'
    try:
        now = archives.digest(Path(root) / declaration)
    except OSError:
        now = None
    if now is None or now != record.get('declaration_sha256'):
        return None, f'{declaration} is not the bytes the refinement was bound to'
    parameters = (runs[run].get('declaration') or {}).get('parameters')
    value = parameters.get(key) if isinstance(parameters, dict) and isinstance(key, str) else None
    cited = value.get('measured_from') if isinstance(value, dict) else None
    if not isinstance(cited, str) or not cited or cited != record.get('measured_from'):
        return None, f'parameter {key!r} of {run} does not cite what the refinement states'
    whole = cited_path(archives, root, cited)
    sources, evidence = record.get('sources'), record.get('evidence')
    if not isinstance(sources, dict) or not sources:
        return None, 'names no source: an empty set establishes nothing'
    if not isinstance(evidence, dict) or not evidence:
        return None, 'names no producer or index record that ties its sources to the parameter'
    if not isinstance(record.get('why'), str) or not record['why'].strip():
        return None, 'does not say what the derivation needs'
    for path, entry in sorted(sources.items()):
        problem = named(archives, root, path, names)
        if problem:
            return None, f'source {path!r} {problem}'
        if not (path == whole or path.startswith(whole + '/')):
            return None, f'source {path} is not at or below the citation {whole}: a refinement narrows a citation, it does not move it'
        if not isinstance(entry, dict):
            return None, f'source {path} is neither a file with its bytes and sha256 nor a named subtree'
        status = mode_of(root, path)
        if set(entry) == {'subtree'} and isinstance(entry['subtree'], str) and entry['subtree'].strip():
            if status is None or not stat.S_ISDIR(status.st_mode):
                return None, f'source {path} is not a folder on this disk'
            if any(taken == path or taken.startswith(path + '/') for taken in archived):
                return None, f'source {path} holds files an archive record inventories: a refinement takes nothing back from an archive'
        elif (set(entry) == {'bytes', 'sha256'} and isinstance(entry['bytes'], int) and not isinstance(entry['bytes'], bool)
              and isinstance(entry['sha256'], str) and archives.HEX.match(entry['sha256'])):
            if status is None or not stat.S_ISREG(status.st_mode) or status.st_size != entry['bytes']:
                return None, f'source {path} is not on this disk as the file the refinement names'
            if path in archived:
                return None, f'source {path} is in an archive record\'s inventory: a refinement takes nothing back from an archive'
            if deep and archives.digest(Path(root) / path) != entry['sha256']:
                return None, f'source {path} is not the bytes the refinement names'
        else:
            return None, f'source {path} is neither a file with its bytes and sha256 nor a named subtree'
    for path, said in sorted(evidence.items()):
        problem = named(archives, root, path, names)
        if problem:
            return None, f'evidence {path!r} {problem}'
        status = mode_of(root, path)
        if status is None or not stat.S_ISREG(status.st_mode):
            return None, f'evidence {path} is not a file on this disk'
        if path in archived:
            return None, f'evidence {path} is in an archive record\'s inventory: a refinement takes nothing back from an archive'
        if not (isinstance(said, str) and archives.HEX.match(said)) or archives.digest(Path(root) / path) != said:
            return None, f'evidence {path} is not the bytes the refinement was bound to'
    return (run, key), set(sources) | set(evidence)


def narrowed(archives, root, method, runs, deep=False):
    """({(run, parameter): the paths that stay in its citation's place}, problems). A refinement with a problem narrows nothing."""
    folder = Path(root) / archives.REGISTRY / FOLDER
    files = sorted(folder.glob('*.json')) if folder.is_dir() else []
    if not files:
        return {}, []
    names, problem = archives.exclusions(root)
    if names is None:
        return {}, [f'parameter sources: {problem}; every citation stays whole']
    archived, found, problems = inventoried(archives, root), {}, []
    for file in files:
        try:
            record = json.loads(file.read_text())
        except (OSError, ValueError):
            record = None
        key, result = check(archives, root, record, method, runs, names, archived, deep)
        if key is None:
            problems.append(f'parameter sources {file.name}: {result}; the citation stays whole')
        else:
            found.setdefault(key, []).append((file.name, result))
    kept = {}
    for key, whole in sorted(found.items()):
        if len(whole) > 1:
            problems.append(f'parameter sources {", ".join(name for name, _ in whole)}: more than one refinement of parameter {key[1]!r} '
                            f'of {key[0]}; the citation stays whole')
        else:
            kept[key] = whole[0][1]
    return kept, problems


def made(archives, root, declaration, parameter, stated):
    """A refinement's record from what the operator states: {"sources": [<file>, ...], "subtrees": {<folder>: <why>}, "evidence":
    [<record>, ...], "why": <text>}. Each named file is hashed here, after its name has passed the spelling, link and seal checks,
    so a sealed or no-fit name is never opened. Returns (record or None, problems): the record is whole by `check`, hashes included."""
    method = json.loads((Path(root) / archives.METHOD).read_text())
    runs = archives.guard().load_runs(Path(root), method)
    names, problem = archives.exclusions(root)
    if names is None:
        return None, [problem]
    if not isinstance(stated, dict):
        return None, ['what is stated is not an object with sources, subtrees, evidence and why']
    files, subtrees, evidence = stated.get('sources') or [], stated.get('subtrees') or {}, stated.get('evidence') or []
    if not (isinstance(files, list) and isinstance(subtrees, dict) and isinstance(evidence, list)):
        return None, ['sources and evidence are lists of paths and subtrees is an object of folder: reason']
    base, tail = method['runs_root'], '/declaration.json'
    run = declaration[len(base) + 1:-len(tail)] if declaration.startswith(base + '/') and declaration.endswith(tail) else None
    parameters = (runs.get(run, {}).get('declaration') or {}).get('parameters') if run else None
    value = parameters.get(parameter) if isinstance(parameters, dict) else None
    if not isinstance(value, dict) or not isinstance(value.get('measured_from'), str):
        return None, [f'{declaration} states no parameter {parameter!r} with a measured_from']
    problems = [f'{path!r} {why}' for path in list(files) + list(subtrees) + list(evidence) for why in [named(archives, root, path, names)] if why]
    if problems:
        return None, [f'{problem} (nothing was opened)' for problem in problems]
    sources = {}
    try:
        for path in files:
            sources[path] = dict(bytes=os.lstat(Path(root) / path).st_size, sha256=archives.digest(Path(root) / path))
        bound = {path: archives.digest(Path(root) / path) for path in evidence}
        record = dict(form=FORM, declaration=declaration, declaration_sha256=archives.digest(Path(root) / declaration), parameter=parameter,
                      measured_from=value['measured_from'], sources=dict(sources, **{path: dict(subtree=why) for path, why in subtrees.items()}),
                      evidence=bound, why=stated.get('why'))
    except OSError as error:
        return None, [f'{error.filename} cannot be read ({type(error).__name__})']
    key, result = check(archives, root, record, method, runs, names, inventoried(archives, root), deep=True)
    return (record, []) if key else (None, [result])
