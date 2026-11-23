"""A state's own answers ACQUIRED and nothing else: the capture of a functional loop (`functional_loop.acquisition: native-events-only`).

  PASS=<n> KEEP_RAW=<folder> CASES=<manifest> python native_events.py <declared run> <cohort run> <path of the model> <history the declaration names> <out: the own folder>
      from code/signal-trace, on a card under its lock (CUDA_VISIBLE_DEVICES names it)

For each answer the selection names it runs the model on the answer's prompt (the prompt alone, greedy, to the answer's limit) with the
three-worker recorder on the kept scheduler (generation_engine.py), and keeps three things:

  its own history     <out>/<id>/answer.json: the token ids the model itself chose, one a step. They are what the model ran on at
                      every later step, so they say which computation the kept states are; they are compared with nothing.
  its ending          in the same file: whether the model ended the answer itself or the limit ended it.
  its native states   KEEP_RAW/<id>: the recorder's own stream of that generation, untouched. states_reading_roles.py
                      (ROLLOUTS=<that folder>) keeps the reading from it with no second pass.

What it does not do, and own_text.py and own_text_workers.py did (Astra's request of the evening of November 15, 2026, and her record
records/retrospectives/2026-11-15-turn-acquisition-scope-astra.md: the events of a run are paired with the events of the fixed correct
reference, over complete turns, by the loop and after the capture, and never with the same answer's old wrong recording. Until later
that evening this head cited, as Marcos's direction relayed at 6:22 PM CT, that nothing made of a turn vector may pair; the session
that relayed it withdrew that at 7:37 PM CT as its own reading and not his). It does not lay the answer on its own
original recording: no pairs-original.npy, no registration, no original computation named. It exports no turns. It does not open the
original's recorded answer or any array of the original, and it does not ask whether the answer's tokens are the recording's: there is
no reference history here, and the fourth argument is only held to the declaration (functional_capture.admitted) and written down.
Pairing events is the loop's own work, by its own relation, after this capture.

What is read of the cohort: its membership (each answer's id and outcome group, cases.cohort), to make the selection (cases.py,
tube.picks), then each NAMED answer's prompt and the recorder's limit for it (cases.prompts). It does not go through ladder.cohort,
which hands every member its recorded answer and original length: neither is loaded into this program.

It belongs to a declared run: the declaration is checked before a folder is made or the recorder started
(run_declaration.check_native_run), it must declare a functional loop whose acquisition is native-events-only
(capture_bound.native_events), and capture_bound.admitted holds the capture to that loop's state, cohort, manifest and reader
protocol (pipeline.weight_edit.functional_capture). A capture is made whole: one that was interrupted is made again from its start.

<out>/text.json is the record: what the capture belongs to, `acquisition`, the selection, each prompt's sha256, each answer's steps
and ending, the sha256 of each answer.json, the generation's execution (recorder, libraries, generation identity, worker count, card,
driver) with every call's own account, and the programs it was made with. It reads a model; it selects, pairs and judges nothing."""
import hashlib, json, os, shlex, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture_bound

FILES = ('answer.json',)   # what a finished answer of this capture holds
MADE_WITH = ('native_events.py', 'capture_bound.py', 'generation_engine.py', 'recorder_engine.py', 'cases.py')
WHAT = ("each named answer generated from its prompt alone, greedy, to its limit: its own token history, its ending and the recorder's own "
        'stream of that generation. Nothing is paired with anything: no registration, no turns, no original answer or array opened')


def of_tokens(tokens):   # as reading_identity.of_tokens and states_reading_roles.of_tokens hash a token list
    return hashlib.sha256(json.dumps([int(t) for t in tokens], separators=(',', ':')).encode()).hexdigest()


def only(member):
    """What this capture holds of a named answer: its id, its prompt and the prompt's question length, and nothing else a member may
    carry (its outcome group stays with the selection)."""
    return dict(id=member['id'], tokens=list(member['tokens']), question=member['question'])


def answer_of(member, result, limit):
    """One generated answer from the recorder's own result: its history, its ending. ValueError where the answer is not the model's own."""
    if result['teacher_token_count'] or result['conditioned_continuation']:
        raise ValueError(f"{member['id']}: the answer was not the model's own")
    if result['prompt_tokens'] != len(member['tokens']):
        raise ValueError(f"{member['id']}: the recorder was not given this answer's prompt")
    return dict(id=member['id'], steps=len(result['steps']), tokens=[step['selected'] for step in result['steps']],
                ended_by_itself=bool(result['ended']) and not result['stopped_limit'], limit=limit, prompt_sha256=of_tokens(member['tokens']))


def acquire(out, chosen, limits, engine, call, keep_raw, size=10):
    """Every chosen answer generated and kept, `size` answers a recorder call (which only bounds the raw stream on disk).

    chosen: what `only` gives, in the cohort's order. limits: {id: the answer's limit}. engine: generation_engine.Generation.
    call(batch, raw, cases): runs one recorder call on the cases file and returns (the engine's watch of it, its error log, when it
    began); the recorder leaves raw/<id>/ with result.json and the stream. keep_raw: where each answer's raw folder is moved.
    Returns ({id: what text.json says of the answer}, the calls' seconds)."""
    out, keep_raw, answers, seconds = Path(out), Path(keep_raw), {}, 0.0
    keep_raw.mkdir(parents=True, exist_ok=True)
    for begin in range(0, len(chosen), size):
        batch = chosen[begin:begin + size]
        raw, cases = out/f"raw-{batch[0]['id']}", out/f"raw-{batch[0]['id']}-cases.json"
        shutil.rmtree(raw, ignore_errors=True)   # a capture this tool began and did not finish
        for m in batch:
            shutil.rmtree(out/m['id'], ignore_errors=True)
            shutil.rmtree(keep_raw/m['id'], ignore_errors=True)
        cases.write_text(json.dumps([dict(id=m['id'], tokens=m['tokens'], max_tokens=limits[m['id']]) for m in batch]))
        started = time.time()
        watch, log, began = call(batch, raw, cases)
        seconds += time.time() - started
        engine.ended(watch, log, time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(began)), cases=[m['id'] for m in batch])   # the call ran this capture's engine, by its own map, every scheduler's counts and its own account of each case, or the capture ends here
        for m in batch:
            mine = answer_of(m, json.loads((raw/m['id']/'result.json').read_text()), limits[m['id']])
            shutil.move(str(raw/m['id']), str(keep_raw/m['id']))
            (out/m['id']).mkdir()
            (out/m['id']/'answer.json').write_text(json.dumps(mine))
            answers[m['id']] = dict(steps=mine['steps'], ended_by_itself=mine['ended_by_itself'], limit=mine['limit'])
        shutil.rmtree(raw, ignore_errors=True)
        cases.unlink()
    return answers, seconds


def record(belongs, acquired, cohort, selection, chosen, model, model_sha256, engine, recorder, execution, history_named, started, seconds, out,
           *, model_identity=None):
    """What text.json says. execution: (card, driver, the card's number as CUDA_VISIBLE_DEVICES named it)."""
    out = Path(out)
    for m in chosen:
        mine = json.loads((out/m['id']/'answer.json').read_text())
        if mine['id'] != m['id'] or len(mine['tokens']) != mine['steps'] or mine['prompt_sha256'] != of_tokens(m['tokens']):
            raise ValueError(f"{m['id']}: what was kept is not this answer")
    prompts = {m['id']: of_tokens(m['tokens']) for m in chosen}
    card, driver, visible = execution
    from pipeline.weight_edit import working_identity
    identity = working_identity.from_record(dict(model_sha256=model_sha256, model_identity=model_identity))
    return dict(**belongs, acquisition=capture_bound.NATIVE_EVENTS, what=WHAT, cohort=cohort, cases=selection, model=model, **working_identity.fields(identity),
                engine_sha256=engine.record['sha256'], prompts=prompts,
                prompts_sha256=hashlib.sha256(json.dumps(dict(cohort=cohort, prompts=prompts), separators=(',', ':')).encode()).hexdigest(),
                files={m['id']: {name: capture_bound.digest(out/m['id']/name) for name in FILES} for m in chosen},
                execution_role='generation', recorder_sha256=capture_bound.digest(recorder), libraries_sha256=engine.record['engine_sha256'],
                generation_sha256=engine.record['sha256'], workers=engine.workers, engine=engine.described(), card=card, driver=driver, visible=visible,
                made_with={name: capture_bound.digest(HERE/name) for name in MADE_WITH},
                history_named=history_named, history_named_is='what the declaration names as its acquisition reference: held to the declaration and not opened',
                started=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(started)), answers_read=len(chosen), raw_streams_kept=True,
                seconds=dict(rollouts=round(seconds, 1), all=round(time.time() - started, 1)), answers=acquired)


def main():
    from pipeline.acquisition.run_a_baseline import BACKENDS, ROOT
    from pipeline.native_records.run_declaration import check_native_run
    from pipeline.paths import ROUND_TWO_RUNS
    from pipeline.weight_edit import functional_capture, working_identity
    import generation_engine   # the engine of a generation by several contexts: the workers recorder on the kept scheduler, the worker count set
    import cases as cohort
    from cases import chosen as selected_cases

    declared, run, before, out = ROUND_TWO_RUNS/sys.argv[1], ROUND_TWO_RUNS/sys.argv[2], sys.argv[4], Path(sys.argv[5]).resolve()
    model = Path(sys.argv[3]).resolve()
    if (out/'text.json').exists():
        sys.exit(f"{out}/text.json exists: a state's own answers are acquired once")
    if not os.environ.get('KEEP_RAW'):
        sys.exit('KEEP_RAW names where each answer\'s native stream is kept: without it this capture would keep no state')
    keep_raw, under = Path(os.environ['KEEP_RAW']).resolve(), shlex.split(os.environ.get('RECORDER_UNDER', ''))   # RECORDER_UNDER: a command each recorder call is run under, a card's lock taken per call
    # the declared run first: the research verifier and the run's own archived inputs, then that it declares an acquisition of native
    # events only, then whether this capture is that loop's (its state by sha256, its cohort, its manifest, its reader protocol)
    declaration = check_native_run(out, declared/'declaration.json')
    if not capture_bound.native_events(declaration):
        sys.exit(f'{declared.name} declares no functional loop: native_events.py is the capture of a loop that only acquires')
    number = capture_bound.pass_number(os.environ.get('PASS'))
    state = functional_capture.working_state(ROOT, declared, declaration, number, model=model)
    model_identity = None if state is None else state['model_identity']
    model_sha256 = capture_bound.digest(model) if state is None else None
    inherited = capture_bound.lease_fds(state)
    belongs = capture_bound.admitted(ROOT, declared, declaration, run, model_sha256, capture_bound.history(before, ROOT), out,
                                     number, keep_raw, model_identity=model_identity, model=model)
    recorder = generation_engine.RECORDER
    engine = generation_engine.Generation(recorder=recorder)   # which file the loader gives THIS recorder for each library, hashed, before any call
    started = time.time()
    picked, selection = selected_cases(cohort.cohort())   # the cohort's membership alone (ids and outcome groups): the fitting answers, or exactly the answers of the manifest CASES names, fixed before anything is generated
    chosen, limits = cohort.prompts(picked)               # each named answer's prompt and the recorder's limit for it: no recorded answer and no original length is loaded
    chosen = [only(m) for m in chosen]
    out.mkdir(parents=True, exist_ok=True)

    def call(batch, raw, cases):
        logs, began = [Path(str(raw) + '.stdout.jsonl'), Path(str(raw) + '.stderr.log')], time.time()
        with open(logs[0], 'w') as so, open(logs[1], 'w') as se:
            process = subprocess.Popen(under + [str(recorder), str(model), str(raw), str(cases), str(BACKENDS), 'capture', 'residual', 'stream'], env=engine.env, stdout=so, stderr=se, pass_fds=inherited)
            watch = engine.watch(process)   # the call's own map, read while it runs
            code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, process.args)
        return watch, logs[1], began

    acquired, seconds = acquire(out, chosen, limits, engine, call, keep_raw)
    for path in out.glob('raw-*.std*'):
        path.unlink()
    if state is not None:
        working_identity.load_state(model.parent / 'state.json', model=model, expected_identity=model_identity)
    elif capture_bound.digest(model) != model_sha256:
        raise ValueError('the model changed during the capture')
    seen = os.environ.get('CUDA_VISIBLE_DEVICES')
    card, driver = (subprocess.run(['nvidia-smi', f'--query-gpu={what}', '--format=csv,noheader', '-i', seen], capture_output=True, text=True, check=True).stdout.strip()
                    for what in ('name', 'driver_version')) if seen else (None, None)
    said = record(belongs, acquired, run.name, selection, chosen, str(model.relative_to(ROOT)) if ROOT in model.parents else model.name, model_sha256,
                  engine, recorder, (card, driver, seen), capture_bound.history(before, ROOT), started, seconds, out,
                  model_identity=model_identity)
    (out/'text.json').write_text(json.dumps(said, indent=1))
    print(json.dumps({key: said[key] for key in ('acquisition', 'model_sha256', 'visible', 'answers_read', 'seconds')}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 6:
        sys.exit('usage: PASS=<n> KEEP_RAW=<folder> CASES=<manifest> python native_events.py <declared run> <cohort run> <path of the model> <history the declaration names> <out: the own folder>')
    main()
