"""What a release did not give back: released files a virtual machine's file daemon still holds open, and their blocks.

  python3 -I held_deleted.py look                         every virtiofsd of this user: how many deleted files it holds, and how many GiB, by place
  python3 -I held_deleted.py free <record>.json [more]    give back the blocks of the files those records released (see below)

Why it exists. On November 14, 2026 at 10:24 PM CT four stores left this disk through their archive records (1,118 files, 71.91 GiB,
each read back from the bucket first) and free disk did not move: 73,978 MiB before, 74,945 MiB after. Claude Desktop's cowork machine
(qemu, `claude-cowork-vm`, started 5:15:48 PM CT) shares `/` with its guest through `virtiofsd --shared-dir / --cache auto`, and that
daemon keeps a handle on every file its guest has looked up: 22,942 files of this repository, and the 1,118 deleted ones. A deleted file
whose handle is held keeps its blocks until the guest forgets it or the machine stops. Nothing in `df`, `du` or the release's receipt
says so: `lsof -nP -p <the daemon>` does, with "(deleted)" at the line's end.

`free` gives the blocks back without touching the daemon: for each file a named record released, when a daemon holds a deleted file of
exactly that path, with no name left on this disk (link count 0) and the inventory's size, the file is cut to nothing through the
daemon's own handle (/proc/<pid>/fd/<n>). It refuses the whole record when the record has no bound read-back, when the registry holds no
release receipt for it, or when any of its files is on this disk. It touches no file that has a name, no file a record does not list,
and no process. The bytes are the bucket's from then on, as the release already decided; `archive_inputs.py restore <record> --fetch`
brings them back.

The way that needs no tool is the guest's: dropping its dentries and inodes (or stopping the machine) makes the daemon let go. That is
the machine's owner's, not this repository's."""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REGISTRY = ROOT/'studies/04-round-2-overlap/input-archives'
GONE = ' (deleted)'


def daemons():
    mine = os.getuid()
    for entry in os.listdir('/proc'):
        if entry.isdigit():
            try:
                if os.stat(f'/proc/{entry}').st_uid == mine and Path(f'/proc/{entry}/comm').read_text().strip() == 'virtiofsd':
                    yield int(entry)
            except OSError:
                continue


def held(pid):
    """A daemon's deleted files: (the handle's path under /proc, the file's path when it had one)."""
    try:
        names = os.listdir(f'/proc/{pid}/fd')
    except OSError:
        return
    for fd in names:
        link = f'/proc/{pid}/fd/{fd}'
        try:
            target = os.readlink(link)
        except OSError:
            continue
        if target.endswith(GONE) and target.startswith('/'):
            yield link, target[:-len(GONE)]


def look():
    for pid in daemons():
        places = {}
        for link, path in held(pid):
            try:
                size = os.stat(link).st_size
            except OSError:
                continue
            parts = Path(path).parts
            place = str(Path(*parts[:parts.index('runs') + 2])) if 'runs' in parts and str(ROOT) in path else str(Path(*parts[:5]))
            count, total = places.get(place, (0, 0))
            places[place] = (count + 1, total + size)
        print(f'virtiofsd {pid}: {sum(n for n, _ in places.values())} deleted files held, {sum(b for _, b in places.values()) / 2 ** 30:.2f} GiB')
        for place, (count, total) in sorted(places.items(), key=lambda each: -each[1][1])[:20]:
            print(f'  {count:>6} files {total / 2 ** 30:>8.2f} GiB  {place}')


def released(name):
    """The files a record released, {path: bytes}, or the reason it is refused."""
    record = json.loads((REGISTRY/name).read_text())
    bound = record.get('remote_bytes_verified') or {}
    if bound.get('files') != len(record['files']) or bound.get('inventory_sha256') != record.get('inventory_sha256'):
        return None, 'no read-back is bound to its inventory'
    if not (REGISTRY/'releases'/name).is_file():
        return None, 'the registry holds no release receipt for it'
    files = {}
    for inner, told in record['files'].items():
        path = ROOT/record['input'] if record['kind'] == 'file' else ROOT/record['input']/inner
        if os.path.lexists(path):
            return None, f'{path.relative_to(ROOT)} is on this disk'
        files[str(path)] = told['bytes']
    return files, None


def free(names):
    for name in names:
        files, refused = released(name)
        if refused:
            print(json.dumps(dict(record=name, refused=refused)), flush=True)
            continue
        cut, total, passed = 0, 0, 0
        for pid in daemons():
            for link, path in held(pid):
                if path not in files:
                    continue
                try:
                    seen = os.stat(link)
                except OSError:
                    continue
                if seen.st_nlink != 0 or seen.st_size != files[path]:   # another name still holds it, or it is not the inventoried file (or was cut before)
                    passed += 1
                    continue
                os.truncate(link, 0)
                cut, total = cut + 1, total + seen.st_size
        print(json.dumps(dict(record=name, files_released=len(files), held_and_given_back=cut, bytes=total, gib=round(total / 2 ** 30, 2), held_and_passed_over=passed,
                              at_ct=time.strftime('%Y-%m-%d %H:%M:%S %Z'))), flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'look':
        look()
    elif len(sys.argv) >= 3 and sys.argv[1] == 'free':
        free(sys.argv[2:])
    else:
        raise SystemExit(__doc__)
