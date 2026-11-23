"""Development: what goes to a rented card with a solve, and a search of it before it goes. Files only; nothing of the project is imported.

  python3 -I solve_files.py closure <code/signal-trace> <module> [<module> ...]   the pipeline files those modules import, one per line
  python3 -I solve_files.py search <folder>                                      exit 1, with each hit, when a file there names what must not leave

Why. Until November 13, 2026, 5:25 AM CT box-start.sh sent the whole pipeline/ folder to a rented box. A solve imports two modules of it and
what they import; the rest was sent for nothing, and some of it named what does not belong on a machine that is not ours: the solver's own
files quoted the founder (taken out at their source in 3640f366), and five files of acquisition/ and observation/ name the lab through its
archive tool. A removal on a rented disk is not an erasure, so the rule is in the sending: a box gets the files the solve imports, staged in
a folder here first, and the staged folder is searched; a hit refuses the send and names the file and the line.

closure reads imports by syntax (ast), inside functions too, since the solver imports some modules where it uses them: `import pipeline.x`,
`from pipeline.x import y` (y a module or a name of x), and relative forms. Every package's __init__.py on the way is included. A module loaded
by a computed name or path is not seen by syntax: closure prints such calls to the standard error so that they are read, not guessed.

search looks, in every file that decodes as text, for a person's name, a home folder, the lab's name and a mail address. The patterns are the
ones below and nothing cleverer: a new private word is added here."""
import ast, re, sys
from pathlib import Path

PRIVATE = (('the founder\'s name', re.compile(r'marcos|damasceno', re.I)),
           ('a home folder', re.compile(r'/home/[A-Za-z]|/Users/[A-Za-z]')),
           ('the lab\'s name', re.compile(r'torad', re.I)),
           ('a mail address', re.compile(r'[A-Za-z0-9._-]+@[A-Za-z0-9-]+\.[a-z]{2,}')))
BY_NAME = re.compile(r'spec_from_file_location|import_module|__import__|runpy\.|SourceFileLoader')


def file_of(root, module):
    path = root.joinpath(*module.split('.'))
    if path.with_suffix('.py').is_file():
        return path.with_suffix('.py')
    if (path/'__init__.py').is_file():
        return path/'__init__.py'
    return None


def closure(root, seeds):
    found, todo, by_name = {}, list(seeds), []
    while todo:
        module = todo.pop()
        if module in found or not module.startswith('pipeline'):
            continue
        path = file_of(root, module)
        if path is None:
            continue
        found[module] = path
        parts = module.split('.')
        todo += ['.'.join(parts[:i]) for i in range(1, len(parts))]
        text = path.read_text()
        by_name += [f'{path.relative_to(root)}:{n}: {line.strip()[:160]}' for n, line in enumerate(text.splitlines(), 1) if BY_NAME.search(line)]
        package = parts if path.name == '__init__.py' else parts[:-1]
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                todo += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = package[:len(package) - node.level + 1] if node.level else []
                source = '.'.join(base + ([node.module] if node.module else []))
                todo += [source] + [f'{source}.{a.name}' for a in node.names]
    return found, by_name


def search(folder):
    hits = []
    for path in sorted(p for p in folder.rglob('*') if p.is_file()):
        try:
            text = path.read_bytes().decode()
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            for what, pattern in PRIVATE:
                if pattern.search(line):
                    hits.append(f'{path.relative_to(folder)}:{n}: {what}')
    return hits


if __name__ == '__main__':
    if sys.argv[1] == 'closure':
        root = Path(sys.argv[2]).resolve()
        found, by_name = closure(root, sys.argv[3:])
        missing = [m for m in sys.argv[3:] if m not in found]
        if missing:
            sys.exit(f'not found under {root}: {missing}')
        for line in by_name:
            print('loaded by a computed name, read it:', line, file=sys.stderr)
        print('\n'.join(sorted(str(p.relative_to(root)) for p in found.values())))
    elif sys.argv[1] == 'search':
        hits = search(Path(sys.argv[2]).resolve())
        print('\n'.join(hits[:40]) if hits else 'nothing private found')
        if len(hits) > 40:
            print(f'... and {len(hits) - 40} more')
        sys.exit(1 if hits else 0)
    else:
        sys.exit('closure or search')
