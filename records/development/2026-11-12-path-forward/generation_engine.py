"""The engine of a capture's GENERATION when several recorder contexts write the answers at once, and what each call says it ran.

  python3 -I generation_engine.py     the identity as it stands, from files and the loader's own trace: no pass, no card, no model
  import generation_engine            own_text_workers.py and reader_protocol_roles.py

PREPARATION, source only. Nothing here is adopted: the readers and `controlled-bf16-8` start one context of the installed recorder
and no file they bind is edited. Astra approved preparing this path, and nothing more, on recorder-workers-original's result
(records/retrospectives/2026-11-15-recorder-workers-result-astra.md): a candidate capture uses it only after one integrated
original-only qualification of all thirty fitting answers, declared and assessed first.

What it is. recorder-workers-original (studies/04-round-2-overlap/runs/recorder-workers-original, 9:34:38 to 9:43:38 AM CT November
15, 2026) ran the workers recorder, the installed program with its one loop over the cases made a function that
OBSERVE_STREAM_WORKERS=<k> starts k times as threads of one process, each with a context of its own. With three workers on the kept
scheduler (the arrangement `workers-3-kept`) ten answers' files were the installed recorder's bytes, 10 of 10. This is that
arrangement as an engine: recorder_engine.Engine, traced for the WORKERS recorder (the loader is asked which file that program takes
for each library, not the installed one), with the worker count set.

Why it is another identity and not another library hash. A generation made this way ran another PROGRAM than the state reading that
follows it (the installed recorder, one context): `generation.recorder_sha256` is the workers recorder's, bound to the qualification's
declaration with the source it was built from (the installed source, the edit, the patch, the built source). `sha256` here covers the
libraries the loader gives that program, the scheduler's setting, the recorder's bytes and the worker count, so no record of a
generation can be read as one of the installed recorder, and no state reading relabelled as this.

What a call says for itself (`ended`), beside what recorder_engine.Engine already holds a pass to (every bound library mapped from
the bound file, the scheduler's library by inode and bytes, the scheduler's own line): the program's own count of its workers and of
the call's cases; every case taken once, by a worker it names, in the order given; and EVERY scheduler's own totals, kept one by one
(`each_scheduler`) and each on the path that was qualified (no split unmarked, every split one range). A context makes three
schedulers (one context left 3 totals lines and three contexts 9 in the qualification's own logs, the same splits in sum), and the
count of lines is held to that: a context that left no account ends the capture. The lines carry no worker's name; they are kept
in the log's order and not assigned. The contexts counted are the ones that took a case (since November 15, 2026, for a capture of
fewer answers than workers): every worker makes its context first, a scheduler says its totals only after a split, so a worker
that took no case leaves none and computed nothing. Totals from more contexts than took a case end the capture as well.

It makes no reference, target, loss or score, reads no answer and starts no recorder by itself."""
import hashlib, json, os, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recorder_engine
from recorder_engine import BASE, TOTALS, digest, named

ROOT = recorder_engine.ROOT
RECORDER = ROOT/'dependencies'/'recorder-workers'/'observe-stream-workers'
WORKERS = 3
SETTING = 'OBSERVE_STREAM_WORKERS'
SET_HERE = (SETTING, 'OBSERVE_STREAM_CPU_THREADS')   # the worker count is this file's; the other is a setting of the workers recorder the qualification ran without
QUALIFIED = 'studies/04-round-2-overlap/runs/recorder-workers-original'
ARRANGEMENT = 'workers-3-kept'
ACCEPTED = 'records/retrospectives/2026-11-15-recorder-workers-result-astra.md'
SOURCE = (('installed_source', 'installed_source_sha256'), ('make_source', 'make_source_py_sha256'), ('patch', 'patch_sha256'), ('workers_source', 'workers_source_sha256'))
CASE = re.compile(r'OBSERVE_STREAM_WORKERS: worker (\d+) case (\d+) (\S+) seconds ([0-9.eE+-]+)')
WHOLE = re.compile(r'OBSERVE_STREAM_WORKERS: (\d+) workers, (\d+) cases, ([0-9.eE+-]+) seconds from')
FORM = 'generation-engine/1'


def qualification(root=ROOT, workers=WORKERS):
    """What the generation rests on, read from recorder-workers-original's own files; ValueError where that run does not qualify
    this many workers on the kept scheduler. It is the component qualification (ten answers, one call): the integrated one, of the
    capture's thirty answers in its three calls, is bound by the protocol when it exists."""
    run = Path(root)/QUALIFIED
    declaration, result, whole, arrangement = (json.loads((run/name).read_text()) for name in ('declaration.json', 'result.json', 'compare/workers.json', f'compare/{ARRANGEMENT}.json'))
    if declaration.get('admission', {}).get('by') != 'weight-research-astra' or not str(result.get('status', '')).startswith('completed'):
        raise ValueError('recorder-workers-original is not an admitted run that completed')
    if arrangement.get('workers') != workers or arrangement.get('library') != 'kept':
        raise ValueError(f'{ARRANGEMENT} is not {workers} workers on the kept scheduler')
    if not (arrangement['answers_read'] == arrangement['same_bytes'] > 0) or whole.get('apart'):
        raise ValueError('recorder-workers-original does not hold every answer of the arrangement at the installed recorder\'s bytes')
    counted = arrangement['scheduler']
    if counted['schedulers'] % workers or counted['splits_not_marked'] or counted['ranges'] != counted['splits']:
        raise ValueError(f'the qualified arrangement\'s schedulers are not a whole number a context on the qualified path: {counted}')
    made = declaration['implementation']
    return dict(run=QUALIFIED, arrangement=ARRANGEMENT, declaration_sha256=digest(run/'declaration.json'), result_sha256=digest(run/'result.json'),
                comparison_sha256=digest(run/'compare/workers.json'), arrangement_sha256=digest(run/f'compare/{ARRANGEMENT}.json'),
                answers_of_the_same_bytes=arrangement['same_bytes'], answers_read=arrangement['answers_read'], workers=workers,
                schedulers_a_context=counted['schedulers'] // workers, card_mib_most=arrangement['card_mib_most'], model_sha256=declaration['model_sha256'],
                workers_recorder_sha256=made['workers_recorder_sha256'], installed_recorder_sha256=made['recorder_sha256'],
                source={key: dict(file=made[key], sha256=made[sha]) for key, sha in SOURCE}, accepted=ACCEPTED,
                is_not='the integrated capture\'s qualification: ten answers in one call, compared in one process run beside the installed recorder')


class Generation(recorder_engine.Engine):
    """The engine of this process's generation calls: recorder_engine.Engine of the workers recorder with the worker count set.
    `record` is the identity, `env` what a call is started with, `passes` each ended call's own account."""

    def __init__(self, recorder=RECORDER, workers=WORKERS, given=None, root=ROOT, rests_on=None, **engine):
        given = {key: value for key, value in dict(os.environ if given is None else given).items() if key not in SET_HERE}
        super().__init__(recorder=recorder, given=given, root=root, **engine)
        self.workers, self.rests_on = int(workers), qualification(root, workers) if rests_on is None else rests_on
        self.env[SETTING] = str(self.workers)
        recorder_sha256 = digest(recorder)
        if recorder_sha256 != self.rests_on['workers_recorder_sha256']:
            raise ValueError(f"{recorder} is not the workers recorder {QUALIFIED} qualified ({self.rests_on['workers_recorder_sha256'][:12]}…): {recorder_sha256[:12]}…")
        source = {}
        for key, each in self.rests_on['source'].items():   # the source the qualified program was built from, as it stands now
            source[key] = dict(each, sha256_now=digest(Path(root)/each['file']))
            if source[key]['sha256_now'] != each['sha256']:
                raise ValueError(f"{each['file']} is not the source the qualified workers recorder was built from")
        of_engine = self.record
        listed = f"{of_engine['sha256']}\n{recorder_sha256}\n{SETTING}={self.workers}\n"
        self.record = dict(
            of_engine, form=FORM, sha256=hashlib.sha256(listed.encode()).hexdigest(),
            sha256_is='over the libraries\' identity below (engine_sha256: each bound library\'s name and file sha256 as the loader gives them to THIS recorder, and the scheduler\'s setting), the recorder\'s own sha256 and the worker count',
            engine_form=of_engine['form'], engine_sha256=of_engine['sha256'], settings=dict(of_engine['settings'], **{SETTING: str(self.workers)}),
            generation=dict(recorder=named(recorder, root), recorder_sha256=recorder_sha256, workers=self.workers, schedulers_a_context=self.rests_on['schedulers_a_context'],
                            source=source, qualified_by=self.rests_on,
                            is_another_program_than='the installed recorder a state reading runs: a generation\'s identity is never a reading\'s'))

    def ended(self, watch, log, begun=None, cases=None):
        """One ended call's own account: recorder_engine.Engine.ended's, with the program's own count of workers and cases, which
        worker took each case, and every scheduler's own totals. `cases`: the ids the call was given, in order. ValueError where the
        call did not run this many workers, did not take each case once, or a scheduler context left the qualified path or no account."""
        text = Path(log).read_text(errors='replace')
        whole, taken = WHOLE.findall(text), sorted(((int(w), int(i), name, float(s)) for w, i, name, s in CASE.findall(text)), key=lambda one: one[1])
        if len(whole) != 1 or int(whole[0][0]) != self.workers:
            raise ValueError(f'the call does not say it ran {self.workers} workers: {whole}')
        if int(whole[0][1]) != len(taken) or [i for _, i, _, _ in taken] != list(range(len(taken))):
            raise ValueError(f'the call does not say each of its {whole[0][1]} cases was taken once: {[(i, name) for _, i, name, _ in taken]}')
        if cases is not None and [name for _, _, name, _ in taken] != list(cases):
            raise ValueError(f'the call did not take the cases it was given, in their order: {[name for _, _, name, _ in taken]}')
        if any(not 0 <= w < self.workers for w, _, _, _ in taken):
            raise ValueError('a case was taken by a worker the call did not start')
        each = [dict(zip(('splits', 'ranges', 'outputs_read_back', 'splits_not_marked'), (int(x) for x in found))) for found in TOTALS.findall(text)]
        # A scheduler says its totals only after it computed a split (ggml-backend-eval-kept.patch: eval_kept_splits > 0), and every worker
        # makes its context before it asks for a case (observe_stream_workers.cpp). So a worker that took no case, as two of three do in a
        # call of one answer, computed nothing and leaves no line: the contexts held to an account are the ones that took a case.
        took = len({w for w, _, _, _ in taken})
        expected = self.record['generation']['schedulers_a_context'] * took
        if len(each) != expected:
            raise ValueError(f'{len(each)} schedulers left their totals where the {took} contexts that took a case leave {expected}: a context of the call left no account')
        off = [one for one in each if one['splits_not_marked'] or one['ranges'] != one['splits']]
        if off:
            raise ValueError(f'a scheduler of the call left the path that was qualified (every split marked and computed as one range): {off}')
        account = super().ended(watch, log, begun)
        account.update(workers=self.workers, each_scheduler=each, each_scheduler_is='every scheduler\'s own totals line, in the log\'s order; a context makes '
                       f"{self.record['generation']['schedulers_a_context']} and the lines carry no worker's name",
                       cases=[dict(case=i, id=name, worker=w, seconds=s) for w, i, name, s in taken], seconds_from_the_workers_start=float(whole[0][2]),
                       contexts_that_took_a_case=took)
        if self.file is not None:
            self.file.write_text(json.dumps(self.passes))
        return account


def held(said, wanted, name):
    """A capture's `engine` held to the protocol's generation: the same identity, and every call's own account of it with every
    scheduler context's. ValueError where it is not."""
    if not isinstance(said, dict) or any(said.get(key) != wanted[key] for key in ('form', 'sha256', 'engine_sha256', 'libraries', 'settings')):
        raise ValueError(f"the {name} does not say it ran the protocol's generation engine")
    if {key: said.get('generation', {}).get(key) for key in ('recorder_sha256', 'workers', 'schedulers_a_context')} != {key: wanted['generation'][key] for key in ('recorder_sha256', 'workers', 'schedulers_a_context')}:
        raise ValueError(f"the {name} was not generated by the protocol's recorder with its count of workers")
    passes = said.get('passes') or []
    if not passes:
        raise ValueError(f'the {name} holds no call\'s own account of the engine it ran')
    workers, a_context = wanted['generation']['workers'], wanted['generation']['schedulers_a_context']
    for account in passes:
        each = account.get('each_scheduler') or []
        took = len({case.get('worker') for case in account.get('cases') or []})   # a worker that took no case computed nothing and leaves no totals (`ended`)
        if account.get('engine_sha256') != wanted['sha256'] or account.get('workers') != workers or len(each) != took * a_context:
            raise ValueError(f"a call of the {name} did not run the protocol's generation with every context's account: {account}")
        if any(one.get('splits_not_marked') != 0 or one.get('ranges') != one.get('splits') for one in each) or not sum(one.get('splits', 0) for one in each):
            raise ValueError(f'a scheduler of a call of the {name} left the path that was qualified: {each}')
        if sorted(case.get('case') for case in account.get('cases') or []) != list(range(len(account.get('cases') or []))) or not account.get('cases'):
            raise ValueError(f'a call of the {name} does not account for each of its cases once')
    return dict(engine_sha256=wanted['sha256'], passes=len(passes), workers=workers, schedulers=sum(len(account['each_scheduler']) for account in passes),
                cases=sum(len(account['cases']) for account in passes))


if __name__ == '__main__':
    print(json.dumps(Generation().record, indent=1))
