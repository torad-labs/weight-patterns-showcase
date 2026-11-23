"""The engine a reader's recorder pass runs since November 15, 2026, and what each pass's own process says it ran.

  python3 -I recorder_engine.py     the identity as it stands, from files and the loader's own trace: no pass, no card, no model
  import recorder_engine            own_text.py, states_reading.py, whole_reading.py and reader_protocol.py

What changed. Through `controlled-bf16-7` a reader started the recorder on the installed engine alone (dependencies/llama-cpp-build,
the recorder's RUNPATH). recorder-kept-original (studies/04-round-2-overlap/runs/recorder-kept-original; admitted, and its result
accepted, by Astra: records/retrospectives/2026-11-15-recorder-kept-admission-astra.md) ran the same recorder with one folder ahead
of it, dependencies/engine-f089c9c-sm120-kept, and GGML_SCHED_EVAL_KEPT=1: a scheduler that computes a split as one graph and reads
the recorder's outputs back after it. All thirty fitting answers' files were the installed side's bytes, in 0.756 of its time on a
loaded machine. Astra adopted it for the readers through a protocol made again (`controlled-bf16-8`), on a condition this file is
for: a pass's record says which engine it RAN, the installed engine apart from the effective one, with the search order and both
places. A folder named beside the old identity is not an execution identity.

What the engine is (`Engine.record`). The loader decides, not a list of folders: the recorder is asked which file it takes for each
library it needs, by the loader's own trace (LD_TRACE_LOADED_OBJECTS=1: the program does not run), once in the caller's environment
as a reader set it until now and once in the environment a pass is started with (`environment`). Bound are the libraries the loader
takes from either engine folder in either trace, and every library the two traces take from different files:

  libggml-base.so.0   the overlay's, the bytes recorder-kept-original qualified (KEPT_SHA256); refused when it is any other file
  libgomp.so.1        ALSO another file, which the qualification's own record did not say: the overlay's library was built on this
                      machine without the installed libraries' $ORIGIN runpath, so its need of libgomp is met by the system's
                      library and not by the copy in the installed folder. The qualified comparison ran this way (the same
                      environment is built here, argument for argument); it is bound here by path and sha256, so a system update
                      of that library changes this identity and a protocol made before it refuses the capture.
  the rest            the installed folder's (libllama, libggml, the processor and card backends), as before

`sha256` is over each bound library's name and its file's sha256, and the scheduler's setting. `installed.libraries_sha256` is the
installed folder's identity as reading_identity.libraries gives it: what the original control and every reading through
`controlled-bf16-7` carry as `libraries_sha256`. A capture or reading made through this file carries `sha256` there instead: it ran
other libraries, and says so in the key every reader of an execution already compares.

What a pass says for itself (`Engine.watch`, `Engine.ended`). While a pass runs, its process is found (the started process or a
descendant, when the pass is started under a lock's holder) and its own map is read once it shows the scheduler's library. When the
pass has ended it is refused unless: every bound library is mapped from the bound file, the scheduler's library by its inode too;
no bound file changed (inode, size, time of change) since it was hashed, and the scheduler's library is hashed again; the
scheduler's own line and its own counts are in the pass's log; no split was computed unmarked and every split was one range, which
is what was qualified (a split the scheduler falls back on is correct by its own snapshot rule and was never compared on the card).

It makes no reference, target, loss or score, reads no answer and starts no recorder by itself."""
import hashlib, json, os, re, subprocess, sys, threading, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reading_identity

ROOT = Path(__file__).resolve().parents[3]
RECORDER = ROOT/'code/signal-trace/recorder/observe-stream'
INSTALLED = reading_identity.ENGINE                                   # dependencies/llama-cpp-build: the recorder's RUNPATH
OVERLAY = ROOT/'dependencies'/'engine-f089c9c-sm120-kept'             # ahead of everything else a pass's loader searches
BASE = 'libggml-base.so.0'                                            # the scheduler's library, by the name the recorder asks for
KEPT_SHA256 = 'de163ac072154fd89c4a19ec8db00c6df711570c09ba442a8ca2d9c19dd6b47d'   # its bytes as recorder-kept-original qualified them
QUALIFIED = 'studies/04-round-2-overlap/runs/recorder-kept-original'
VARIABLE = 'GGML_SCHED_EVAL_KEPT'
NOT_CARRIED = (VARIABLE, 'GGML_SCHED_EVAL_DEFERRED', 'GGML_CUDA_TIME_LAUNCH', 'LLAMA_TIME_DECODE')   # another scheduler's switch and the engine's timers: no reader's pass carries them
SAID = "GGML_SCHED_EVAL_KEPT=1, a split is computed whole and the eval callback's outputs are read back after it"   # the scheduler's own line
TOTALS = re.compile(r'GGML_SCHED_EVAL_KEPT totals: (\d+) splits, (\d+) ranges, (\d+) outputs read back, (\d+) splits not marked')
STAGED = re.compile(r'GGML_SCHED_EVAL_KEPT: outputs are staged in (\S+) memory')
TRACED = re.compile(r'^\s*(\S+) => (\S+) \(0x[0-9a-f]+\)\s*$')
FORM = 'recorder-engine/1'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def without(given):
    """The caller's environment as a reader set it through controlled-bf16-7: fusion off, nothing ahead of the installed engine."""
    env = {key: value for key, value in given.items() if key not in NOT_CARRIED}
    env['GGML_CUDA_DISABLE_FUSION'] = '1'
    return env


def environment(given, overlay=OVERLAY):
    """What a recorder pass is started with: the caller's environment, fusion off, the overlay ahead of the caller's own library
    path and the scheduler's setting. It is recorder_kept.environment('kept', ...) of the qualification, argument for argument."""
    env = without(given)
    env[VARIABLE] = '1'
    env['LD_LIBRARY_PATH'] = ':'.join(part for part in (str(overlay), given.get('LD_LIBRARY_PATH', '')) if part)
    return env


def traced(recorder, env):
    """{the name a library is asked for by: the file the loader takes for it}, for everything `recorder` needs in `env`. The loader's
    own trace: the program is not run."""
    said = subprocess.run([str(recorder)], env=dict(env, LD_TRACE_LOADED_OBJECTS='1'), capture_output=True, text=True)
    if said.returncode:
        raise ValueError(f'the loader could not trace {recorder}: {said.stderr[-300:]}')
    found = {}
    for line in said.stdout.splitlines():
        if 'not found' in line:
            raise ValueError(f'the loader finds no file for a library {recorder} needs: {line.strip()}')
        match = TRACED.match(line)
        if match:
            found[match[1]] = str(Path(match[2]).resolve())
    return found


def stamp(path):
    state = os.stat(path)
    return state.st_ino, state.st_size, state.st_mtime_ns


def named(path, root=ROOT):
    path = Path(path)
    return str(path.relative_to(root)) if root in path.parents else str(path)


def mapped(pid):
    """{file: its inode} of everything mapped from a file in a running process, or None when it cannot be read (it has ended)."""
    try:
        lines = Path(f'/proc/{pid}/maps').read_text().splitlines()
    except OSError:
        return None
    seen = {}
    for line in lines:
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[5].startswith('/'):
            seen[parts[5]] = int(parts[4])
    return seen


def descendants(pid):
    """A process and everything under it, itself first."""
    found, at = [pid], 0
    while at < len(found):
        try:
            for task in os.listdir(f'/proc/{found[at]}/task'):
                found += [int(child) for child in Path(f'/proc/{found[at]}/task/{task}/children').read_text().split()]
        except OSError:
            pass
        at += 1
    return found


class Watch:
    """A started pass's own map, read while it runs: `seen` is {file: inode} of the first process of the pass (the started one or one
    under it) that shows the scheduler's library mapped, None while none has."""

    def __init__(self, process, library, every=0.05):
        self.process, self.library, self.every, self.seen, self.pid = process, Path(library).name, every, None, None
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        while self.seen is None:
            for pid in descendants(self.process.pid):
                maps = mapped(pid)
                if maps and any(Path(path).name == self.library for path in maps):
                    self.seen, self.pid = maps, pid
                    return
            if self.process.poll() is not None:
                return
            time.sleep(self.every)

    def settled(self):
        self.thread.join(timeout=5)
        return self.seen


class Engine:
    """The engine of this process's recorder passes: its identity (`record`), the environment a pass is started with (`env`), and
    each ended pass's own account (`passes`)."""

    def __init__(self, recorder=RECORDER, installed=INSTALLED, overlay=OVERLAY, given=None, trace=traced, kept_sha256=KEPT_SHA256, root=ROOT):
        given = dict(os.environ if given is None else given)
        self.env = environment(given, overlay)
        was, now = trace(recorder, without(given)), trace(recorder, self.env)
        here, ahead = Path(installed).resolve(), Path(overlay).resolve()
        inside = lambda path: path is not None and Path(path).parent in (here, ahead)
        bound = sorted(name for name in now if inside(now[name]) or inside(was.get(name)) or now[name] != was.get(name))
        if BASE not in now or Path(now[BASE]).parent != ahead:
            raise ValueError(f"the loader does not take {BASE} from {overlay}: {now.get(BASE)}")
        self.files = {name: Path(now[name]) for name in bound}
        self.stamps = {name: stamp(path) for name, path in self.files.items()}
        sums = {name: digest(path) for name, path in self.files.items()}
        if sums[BASE] != kept_sha256:
            raise ValueError(f"{now[BASE]} is not the bytes recorder-kept-original qualified ({kept_sha256[:12]}…): {sums[BASE][:12]}…")
        elsewhere = sorted(name for name in bound if name != BASE and Path(now[name]).parent == ahead)
        if elsewhere:
            raise ValueError(f'the overlay supplies more than the scheduler\'s library: {elsewhere}')
        place = lambda path: 'the overlay' if Path(path).parent == ahead else 'the installed folder' if Path(path).parent == here else 'outside both folders'
        libraries = {name: dict(file=named(self.files[name], root), sha256=sums[name], taken_from=place(self.files[name])) for name in bound}
        apart = {name: dict(installed=dict(file=named(was[name], root), sha256=digest(was[name])) if was.get(name) else None, effective=libraries[name])
                 for name in bound if now[name] != was.get(name)}
        listed = ''.join(f'{name} {sums[name]}\n' for name in bound) + f'{VARIABLE}=1\n'
        self.record = dict(
            form=FORM, sha256=hashlib.sha256(listed.encode()).hexdigest(),
            sha256_is='over each bound library\'s name and the sha256 of the file the loader takes for it, then the scheduler\'s setting',
            recorder=named(recorder, root),
            search_order=[f'{named(overlay, root)}: first in LD_LIBRARY_PATH, which the loader reads before a RUNPATH',
                          'the caller\'s own LD_LIBRARY_PATH, after it',
                          f'{named(installed, root)}: the recorder\'s RUNPATH, and $ORIGIN for what the installed libraries need themselves',
                          'the system\'s folders: what the overlay\'s library needs itself is found here, since it carries no RUNPATH'],
            settings={VARIABLE: '1', 'GGML_CUDA_DISABLE_FUSION': '1'},
            installed=dict(folder=named(installed, root), is_folder=named(here, root), libraries_sha256=reading_identity.libraries(installed)['sha256'],
                           libraries_sha256_is='reading_identity.libraries of the installed folder: the libraries_sha256 of the original control and of every reading through controlled-bf16-7'),
            overlay=dict(folder=named(overlay, root), library=named(self.files[BASE], root), sha256=sums[BASE], qualified_by=QUALIFIED),
            libraries=libraries, apart_from_the_installed_engine=apart,
            bound='every library the loader takes from the installed folder or the overlay, with or without the overlay ahead, and every library the two traces take from different files; the loader\'s own trace of the recorder, the program not run')
        self.passes, self.file = [], None

    def kept_in(self, path):
        """Keep the passes' accounts in a file beside an acquisition that can be taken up again, and take up what an earlier process
        of it left there: accounts of this engine, or the acquisition is refused."""
        self.file = Path(path)
        if self.file.exists():
            earlier = json.loads(self.file.read_text())
            if any(account.get('engine_sha256') != self.record['sha256'] for account in earlier):
                raise ValueError(f'{self.file}: the unfinished acquisition ran a pass on another engine than this one')
            self.passes = earlier + self.passes
        return self

    def watch(self, process, wait=2.0):
        """Begin reading a started pass's own map. The caller stands here until the map has shown the scheduler's library, a few
        hundredths of a second for a recorder started directly, and no longer than `wait` (a pass started under a lock's holder
        shows it when the lock is its own): so the reading thread has ended before the caller makes its next processes."""
        watch = Watch(process, self.files[BASE])
        watch.thread.join(timeout=wait)
        return watch

    def ended(self, watch, log, begun=None):
        """One ended pass's own account, kept in `passes` and returned; ValueError where the pass did not run this engine. `log` is
        the pass's error stream."""
        seen = watch.settled()
        if seen is None:
            raise ValueError(f'the pass never showed {self.files[BASE].name} mapped: what it ran is not known')
        for name, path in self.files.items():
            if str(path) not in seen:
                other = sorted(each for each in seen if Path(each).name == path.name)
                raise ValueError(f'the pass did not map {name} from {path}' + (f': it mapped {other}' if other else ''))
            if stamp(path) != self.stamps[name]:
                raise ValueError(f'{path} changed while this process was reading: the pass may have run other bytes than the ones named')
        if seen[str(self.files[BASE])] != self.stamps[BASE][0]:
            raise ValueError(f'the pass mapped {self.files[BASE]} from another inode than the file hashed before it')
        if digest(self.files[BASE]) != self.record['libraries'][BASE]['sha256']:
            raise ValueError(f'{self.files[BASE]} was not the qualified bytes any more when the pass ended')
        text = Path(log).read_text(errors='replace')
        totals = [tuple(int(x) for x in found) for found in TOTALS.findall(text)]
        if SAID not in text or not totals:
            raise ValueError('the scheduler did not say it kept its outputs, or left no count of its splits: the pass is not known to have run it')
        counted = dict(schedulers=len(totals), splits=sum(t[0] for t in totals), ranges=sum(t[1] for t in totals),
                       outputs_read_back=sum(t[2] for t in totals), splits_not_marked=sum(t[3] for t in totals), staged_in=sorted(set(STAGED.findall(text))))
        if counted['splits_not_marked'] or counted['ranges'] != counted['splits'] or not counted['splits']:
            raise ValueError(f'the scheduler left the path that was qualified (every split marked and computed as one range): {counted}')
        account = dict(engine_sha256=self.record['sha256'], mapped={name: named(path) for name, path in self.files.items()},
                       scheduler_library_inode=seen[str(self.files[BASE])], scheduler=counted, **({} if begun is None else dict(began=begun)))
        self.passes.append(account)
        if self.file is not None:
            self.file.write_text(json.dumps(self.passes))
        return account

    def described(self, passes=None):
        """The identity with the passes' own accounts: what a capture's or a reading's record carries as `engine`."""
        return dict(self.record, passes=list(self.passes if passes is None else passes))


def held(said, wanted, name):
    """A record's `engine` (a capture's or a reading's) held to the protocol's: the same identity, and every pass's own account of
    it. ValueError where it is not."""
    if not isinstance(said, dict) or any(said.get(key) != wanted[key] for key in ('form', 'sha256', 'libraries', 'settings')):
        raise ValueError(f"the {name} does not say it ran the protocol's engine")
    passes = said.get('passes') or []
    if not passes:
        raise ValueError(f'the {name} holds no pass\'s own account of the engine it ran')
    for account in passes:
        counted = account.get('scheduler', {})
        if account.get('engine_sha256') != wanted['sha256'] or counted.get('splits_not_marked') != 0 or not counted.get('splits') or counted.get('ranges') != counted.get('splits'):
            raise ValueError(f"a pass of the {name} did not run the protocol's engine on the path that was qualified: {account}")
    return dict(engine_sha256=wanted['sha256'], passes=len(passes))


def qualification(identity, root=ROOT):
    """What the protocol binds of the qualification the overlay rests on, read from the run's own files; ValueError where the run
    does not qualify this engine's scheduler library on this recorder and this installed library."""
    run = Path(root)/QUALIFIED
    declaration, result, compared = (json.loads((run/name).read_text()) for name in ('declaration.json', 'result.json', 'compare/kept.json'))
    made = declaration['implementation']
    installed_base = identity['apart_from_the_installed_engine'][BASE]['installed']
    if made['kept_library_sha256'] != identity['overlay']['sha256'] or made['installed_library_sha256'] != installed_base['sha256'] or made['recorder_sha256'] != digest(Path(root)/identity['recorder']):
        raise ValueError('recorder-kept-original compared another scheduler library, installed library or recorder than this engine\'s')
    if declaration.get('admission', {}).get('by') != 'weight-research-astra' or not str(result.get('status', '')).startswith('finished'):   # the result's status line begins with the word and goes on to say when
        raise ValueError('recorder-kept-original is not an admitted run that finished')
    if not (compared['answers_read'] == compared['same_bytes'] == 30) or compared['apart']:
        raise ValueError('recorder-kept-original does not hold thirty answers of the same bytes on both sides')
    return dict(run=QUALIFIED, declaration_sha256=digest(run/'declaration.json'), result_sha256=digest(run/'result.json'), comparison_sha256=digest(run/'compare/kept.json'),
                answers_of_the_same_bytes=compared['same_bytes'], answers_read=compared['answers_read'], model_sha256=declaration['model_sha256'],
                accepted='records/retrospectives/2026-11-15-recorder-kept-admission-astra.md',
                not_in_its_record='that the kept side\'s libgomp.so.1 was the system\'s and not the installed folder\'s: inferred from the loader\'s rule on the environment the comparison built, which is the one built here; the comparison read only the scheduler library\'s map')


if __name__ == '__main__':
    print(json.dumps(Engine().record, indent=1))
