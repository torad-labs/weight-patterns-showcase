"""A model read on its answers' own text with the engine's direct state kept at every merge, at every answer step AND every question
position: the features the whole-history scorer compares (Astra, records/development/2026-11-14-whole-gap-targets/SCORING.md, "Remaining
native reader dependency, located in current sources").

  python states_reading.py <declared run> <cohort run> <original | path of a model> <recordings | capture folder whose text is given>
      from code/signal-trace, as whole_reading.py is run: on a card under its lock, or with RECORDER_UNDER=<a command that takes the card's
      lock and runs what follows it> (after-the-others.sh <card>). KEEPERS=<n> and KEEP_ON=<processor threads> as whole_reading.py takes
      them. It writes <declared run>/reading/ and nothing else, after the research verifier has passed on the run's declaration
      (pipeline.native_records.run_declaration.check_native_run).
      PASS=<n>            the reading of pass n of a run's repeated correction: it is written to <declared run>/loop/<n>/reading/,
                          beside that pass's capture and correspondence record (gap_run.of_pass), where the pass's measurement reads
                          it; a run has one reading a pass, and each is made whole in a folder of its own.
      ROLLOUTS=<folder>   no pass and no card: <folder>/<id>/ holds the raw stream the model's own generation of each answer wrote
                          (own_text.py with KEEP_RAW=<folder>); the text is the capture folder's, the execution is the one
                          that capture bound when it closed (model, recorder and engine libraries held equal to it here), and each
                          stream is kept as a fed pass's is and removed as it is kept. On card 0 a generation's states are the bits a
                          pass fed that text gives, at every position and merge (speed review, own-export/rollout-is-the-reading.json).

What is computed is whole_reading.py's pass, command for command: one recorder process, `capture residual stream`, every fitting answer's
prompt and then its text supplied one token at a time, no filter and no early stop, so every decision has the engine's own scores and
every step says whether the supplied token is the model's own choice.

What is KEPT is another thing. whole_reading.py keeps, at the answer steps only, the float32 difference of two unit vectors a merge
(export_case's turns) and their two sizes: a state rebuilt from those is a sum of rounded turns, not the engine's output. Here, for every
one of the 105 merges and every input position of an answer (its question positions and its answer steps):

  nodes/<merge>/state.npy    float32 [N, width]: the engine's own output of that merge's ADD, the bytes the recorder read off the card.
                             Nothing is derived, normalized or rounded. Before a row is kept, the engine's ADD is made again from the two
                             operands the same pass recorded (the state it adds to and the block's normalized output) and must give the
                             kept row bit for bit, at every row; the record's operation and source names must be that ADD's.
  terminal/l_out.npy         float32 [N, width]: the last layer's last merge times that layer's output scale, as the engine put it out
                             (l_out-<last>): what the output norm reads.
  terminal/result_norm.npy   float32 [answer rows, width]: the engine's own output norm of it (result_norm), which the head reads. The
                             engine computes it only where it computes scores: at every answer step, and in the prompt only at the last
                             position of each call of the prefill (position 127 of a 173-token prompt), which is no field over the
                             question positions and is not kept (answer.json names those positions).
  row-map.json               order all_answers_then_all_questions, as rows_reading.py and the whole-gap target file order their rows:
                             every answer's answer rows in cohort order, then every answer's question rows. Answer row s of an answer
                             is input position question + s (the decision for its token s); question row q is input position q.
  answers/<id>/answer.json   the text read, token by token whether it is the model's own choice, and how it ends (whole_reading.ending).
  reading.json               the manifest: model, recorder, engine libraries, card and driver; every array's file, shape, dtype and
                             sha256; the merge inventory with each merge's two operands; the table of answers. The engine is
                             recorder_engine.py's (since `controlled-bf16-8` the installed engine with the kept scheduler's library
                             ahead of it): `libraries_sha256` is its identity and `engine` holds it whole with the pass's own account
                             of what it mapped and what its scheduler counted, or the generation's calls' accounts when the reading
                             made no pass. A pass that did not run that engine ends the reading.

Every merge here is in the residual stream as the engine holds it (space `residual`), the last one included: merge 104 is the RAW last
merge, before the last layer's output scale and before the output norm. The target file's merge 104 is in output-norm space; whoever
compares composes that normalization once, from this raw state, and has the engine's own `result_norm` at the answer rows to hold the
composition to. A direction and a log size are not stored: they are x / |x| and ln |x| of a stored row, in whatever precision the
consumer declares.

It generates nothing, selects nothing, writes no model and never reads the answers no fit uses."""
import hashlib, json, multiprocessing, os, shutil, signal, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

FORMAT = 'whole-gap-reading/1'


def digest(path):   # as pipeline.native_records.run_declaration.digest reads a file
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def of_tokens(tokens):   # as reading_identity.of_tokens and the target file's row map hash a token list
    return hashlib.sha256(json.dumps([int(t) for t in tokens], separators=(',', ':')).encode()).hexdigest()


def threads_of(named):   # '4,5,12-13' as taskset reads it
    return {n for part in named.split(',') for n in (range(int(part.split('-')[0]), int(part.split('-')[-1]) + 1))}


def merges_of(layers):
    """The model's merges in order, each with the two operands its ADD takes (export_stream_traces.connections names the same three a layer)."""
    out = []
    for layer in range(layers):
        out += [dict(merge=3 * layer, connection=f'attn_out-{layer}', layer=layer, kind='attention', incoming='inp_scaled' if layer == 0 else f'l_out-{layer - 1}', branch=f'attn_post_norm-{layer}'),
                dict(merge=3 * layer + 1, connection=f'pe_in-{layer}', layer=layer, kind='feedforward', incoming=f'attn_out-{layer}', branch=f'ffn_post_norm-{layer}'),
                dict(merge=3 * layer + 2, connection=f'pe_out-{layer}', layer=layer, kind='layer', incoming=f'pe_in-{layer}', branch=f'per_layer_embd_out-{layer}')]
    return out


def rows_in(folder, count):
    """Where each recorded output's row at each input position sits in a case's stream: {label: width, at (the row's first float, -1 where
    the output has no row at that position), operation, sources}."""
    found, floats = {}, (folder/'stream.f32').stat().st_size // 4
    with open(folder/'stream.jsonl') as records:
        for line in records:
            r = json.loads(line)
            if not r['values']:   # a prefill microbatch without a requested output has an empty result_norm
                continue
            shape, positions = r['shape'], np.asarray(r['positions'], np.int64)
            # x-fast values with the tokens on the second axis (export_stream_traces.StreamRows reads the same layout): a token's row is contiguous
            if (r['token_axis'] != 1 or len(shape) < 2 or int(np.prod(shape[2:])) != 1 or len(positions) != shape[1] or r['values'] != shape[0] * shape[1]
                    or r['offset_bytes'] % 4 or r['offset_bytes'] < 0 or r['offset_bytes'] // 4 + r['values'] > floats or r['native_type'] != 'f32'):
                raise ValueError(f"{folder.name}: {r['label']} is not whole rows of float32 along the tokens")
            entry = found.setdefault(r['label'], dict(width=shape[0], at=np.full(count, -1, np.int64), operation=r['operation'], sources=sorted(r['source_names'])))
            if entry['width'] != shape[0] or entry['operation'] != r['operation'] or entry['sources'] != sorted(r['source_names']):
                raise ValueError(f"{folder.name}: {r['label']} changes its width, operation or sources inside one answer")
            if (positions < 0).any() or (positions >= count).any() or (entry['at'][positions] >= 0).any():
                raise ValueError(f"{folder.name}: {r['label']} has a position outside the answer or recorded twice")
            entry['at'][positions] = r['offset_bytes'] // 4 + np.arange(len(positions)) * shape[0]
    return found


def taken(values, entry, positions):
    at = entry['at'][positions]
    if (at < 0).any():
        raise ValueError('an output has no row at a position it is asked at')
    return np.asarray(values[at[:, None] + np.arange(entry['width'])])


def keep(raw, out, supplied_text, prompt, question, answer_from, question_from, merges, given_from, rollout=False):
    """One answer, from its raw stream to its rows of every array. Runs in a process of its own; nothing here asks for a card.
    rollout: the stream is the one the model's own generation of this text wrote (no token fed to it), kept by own_text.py."""
    result = json.loads((raw/'result.json').read_text())
    supplied = [step['selected'] for step in result['steps']]
    if rollout:
        if supplied != supplied_text or result['teacher_token_count'] or result['conditioned_continuation']:
            raise ValueError(f"{raw.name}: the kept stream is not the model's own generation of the given text")
    elif supplied != supplied_text or result['teacher_token_count'] != len(supplied):
        raise ValueError(f'{raw.name}: the pass did not read the given text')
    steps, count = len(supplied), question + len(supplied)
    if result['prompt_tokens'] != question + 1 or [step['input_position'] for step in result['steps']] != list(range(question, count)):
        raise ValueError(f'{raw.name}: the decisions are not at the positions the row map gives them')
    found, values, every = rows_in(raw, count), np.memmap(raw/'stream.f32', dtype='<f4', mode='r'), np.arange(count)
    held = {}

    def state(label):   # a merge's output is the next merge's first operand: read once
        if label not in held:
            held[label] = taken(values, found[label], every)
        return held[label]

    def put(path, rows):
        array = np.load(path, mmap_mode='r+')
        if array.shape[1] != rows.shape[1]:
            raise ValueError(f'{path}: another width')
        array[answer_from:answer_from + steps] = rows[question:]
        array[question_from:question_from + question] = rows[:question]
        del array   # no flush here: one an array made every keeper wait for the disk (rows_reading.py, November 14); the arrays are flushed once, at the end

    for m in merges:
        entry = found[m['connection']]
        if entry['operation'] != 'ADD' or entry['sources'] != sorted([m['incoming'], m['branch']]):
            raise ValueError(f"{raw.name}: {m['connection']} is not the ADD of {m['incoming']} and {m['branch']}")
        x, branch, y = state(m['incoming']), taken(values, found[m['branch']], every), state(m['connection'])
        if not np.array_equal(np.add(x, branch).view('u4'), y.view('u4')):
            raise ValueError(f"{raw.name}: the engine's ADD does not give {m['connection']} from its two recorded operands")
        if not np.isfinite(y).all():
            raise ValueError(f"{raw.name}: {m['connection']} holds a value that is not finite")
        put(out/'nodes'/str(m['merge'])/'state.npy', y)
        del held[m['incoming']]
        if m['kind'] == 'layer':
            del held[m['connection']]
    last = max(m['layer'] for m in merges)
    put(out/'terminal'/'l_out.npy', taken(values, found[f'l_out-{last}'], every))
    normed = found['result_norm']
    if (normed['at'][question:] < 0).any():
        raise ValueError(f'{raw.name}: result_norm is not at every answer step')
    also = [int(p) for p in np.flatnonzero(normed['at'][:question] >= 0)]   # the engine asks for scores at the end of each call of the prefill too; not a row of every question position, and not kept
    array = np.load(out/'terminal'/'result_norm.npy', mmap_mode='r+')
    array[answer_from:answer_from + steps] = taken(values, normed, np.arange(question, count))
    del array
    other =[n for n, step in enumerate(result['steps']) if step['greedy_token'] != step['selected']]
    mine = dict(id=raw.name, steps=steps, question=question, tokens=supplied, prompt_sha256=of_tokens(prompt), text_sha256=of_tokens(supplied), given_from=given_from,
                first_step_not_its_own=other[0] if other else None, steps_not_its_own=len(other),   # whether a supplied token is the engine's own choice identifies the pass; no score of a token is kept
                ended_with_an_end_token=bool(result['ended']),
                rows_whose_add_was_made_again=count * len(merges), question_positions_with_a_result_norm_not_kept=also, result_json_sha256=digest(raw/'result.json'))
    (out/'answers'/raw.name).mkdir(parents=True)
    json.dump(mine, open(out/'answers'/raw.name/'answer.json', 'w'))
    del values
    shutil.rmtree(raw)
    return dict(kept=raw.name, steps=steps, question=question, first_step_not_its_own=mine['first_step_not_its_own'])


def reading_of(declared, number=None):
    """Where a run's states reading lies: <run>/reading, or for pass `number` of its repeated correction <run>/loop/<number>/reading,
    beside the pass's capture (gap_run.of_pass names the capture, the correspondence record and the sweep there)."""
    if number is None or number == '':
        return Path(declared)/'reading'
    if not str(number).isdigit() or int(number) < 1:
        raise ValueError(f'a pass is numbered from 1, and PASS is {number!r}')
    return Path(declared)/'loop'/str(int(number))/'reading'


def main():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    from pipeline.acquisition.run_a_baseline import BACKENDS, ROOT
    from pipeline.native_records.run_declaration import check_native_run
    from pipeline.overlap.fed_replay import MARGIN_GIB
    from pipeline.paths import ROUND_TWO_RUNS
    from pipeline.weight_edit import ladder, tube
    import recorder_engine
    from reading_identity import ending

    declared, cohort_run, given = ROUND_TWO_RUNS/sys.argv[1], ROUND_TWO_RUNS/sys.argv[2], Path(sys.argv[4]).resolve()
    recordings = sys.argv[4] == 'recordings'
    rollouts = Path(os.environ['ROLLOUTS']).resolve() if os.environ.get('ROLLOUTS') else None   # each answer's own raw capture, kept by the generation that wrote the text (own_text.py, KEEP_RAW)
    if rollouts is not None and recordings:
        raise ValueError("ROLLOUTS holds the capture of a model's own answers: the text is given by that capture's folder, not by the recordings")
    model = tube.ORIGINAL_MODEL if sys.argv[3] == 'original' else Path(sys.argv[3]).resolve()
    out = reading_of(declared, os.environ.get('PASS'))
    declaration = check_native_run(out, declared/'declaration.json')   # the research verifier, and the run's own archived inputs back on this disk
    if out.exists():
        raise ValueError(f'{out} exists: a states reading is made whole, in a folder of its own')
    if os.environ.get('PASS'):   # a pass's reading is of that pass's own capture, which own_text.py admitted for this run and this pass (capture_bound.admitted)
        said = {} if recordings or not (given/'own'/'text.json').exists() else json.loads((given/'own'/'text.json').read_text())
        if given != out.parent/'capture' or said.get('run') != declared.name or said.get('pass_number') != int(os.environ['PASS']):
            raise ValueError(f"pass {os.environ['PASS']} of {declared.name} is read on its own capture ({out.parent/'capture'}), closed for this run and this pass")
    recorder = ROOT/'code/signal-trace/recorder/observe-stream'
    engine = recorder_engine.Engine(recorder=recorder)   # the engine a pass runs since controlled-bf16-8: the installed one with the kept scheduler's library ahead, fusion off (the card is the launcher's)
    env = engine.env
    under = os.environ.get('RECORDER_UNDER', '').split()
    started = time.time()
    _, members, index = ladder.cohort(cohort_run)
    from cases import chosen as selected_cases
    chosen, selection = selected_cases(members)   # the fitting answers, or exactly the answers of the manifest CASES names: fixed before anything is read
    fitted = selection['role'] == 'fitting'
    if recordings:
        text, source = {m['id']: m['recorded'] for m in chosen}, dict(model='the original, as recorded on card 0')
    else:
        text = {m['id']: json.loads((given/'own'/m['id']/'answer.json').read_text())['tokens'] for m in chosen}
        source = json.loads((given/'own.json').read_text()) if (given/'own.json').exists() else {}
    given_from = 'recordings' if recordings else str(given)
    steps, questions = [len(text[m['id']]) for m in chosen], [m['question'] for m in chosen]
    if any(len(m['tokens']) != m['question'] + 1 for m in chosen):
        raise ValueError('a prompt is not its question positions and the first decision\'s input')
    answer_from, question_from = np.cumsum([0] + steps[:-1]), sum(steps) + np.cumsum([0] + questions[:-1])
    rows, merges = int(sum(steps) + sum(questions)), merges_of(len(index['layers']))
    place = {m['id']: n for n, m in enumerate(chosen)}
    # Room, before anything is written: the arrays, and the raw streams of the answers the pass may have written and no keeper has yet
    # removed (one in each keeper's hands, one waiting behind them and the one the recorder is writing), beside the margin every capture
    # of this repository leaves free (fed_replay.MARGIN_GIB). A raw stream holds seven outputs a layer, the scaled embedding and the
    # output norm at every position, 1,536 wide.
    keepers, wide = int(os.environ.get('KEEPERS', 4)), 1536   # ladder.stored reads a state 1,536 wide; the arrays take the width the stream itself gives
    largest = max(q + s for q, s in zip(questions, steps)) * (7 * len(index['layers']) + 2) * wide * 4
    need = (rows * (len(merges) + 1) + int(sum(steps))) * wide * 4 + (0 if rollouts is not None else (keepers + 2) * largest)   # kept rollouts are here already and leave as they are kept
    free = shutil.disk_usage(declared).free
    if (free - need) / 2**30 < MARGIN_GIB:
        raise ValueError(f'{free / 2**30:.0f} GiB free: the reading needs up to {need / 2**30:.0f} beside the {MARGIN_GIB} GiB margin')

    capture = None
    if rollouts is not None:   # no card here: the execution is the generation's, as its capture bound it when it closed. Held to it before any kept stream is used up
        capture = json.loads((given/'own'/'text.json').read_text())
        if (capture.get('model_sha256'), capture.get('recorder_sha256'), capture.get('libraries_sha256')) != (digest(model), digest(recorder), engine.record['sha256']):
            raise ValueError("the kept rollouts' capture is of another model, recorder or engine libraries than the ones this reading names")
        recorder_engine.held(capture.get('engine'), engine.record, "kept rollouts' capture")   # every call of the generation left its own account of that engine
        if any(capture.get(key) in (None, '') for key in ('card', 'driver', 'visible')):
            raise ValueError("the kept rollouts' capture does not say which card it ran on")
        if any(capture.get('files', {}).get(m['id'], {}).get('answer.json') != digest(given/'own'/m['id']/'answer.json') for m in chosen):
            raise ValueError("an answer's text is not the one the capture bound when it closed")

    out.mkdir(parents=True)
    raw, cases = out/'raw', out/'raw-cases.json'
    keeping, width, told = [], None, []

    def arrays_of(stream):   # every array, empty, as wide as an answer's stream says a state is
        with open(stream/'stream.jsonl') as records:
            wide_as = json.loads(records.readline())['shape'][0]
        for m in merges:
            (out/'nodes'/str(m['merge'])).mkdir(parents=True)
            np.lib.format.open_memmap(out/'nodes'/str(m['merge'])/'state.npy', mode='w+', dtype='<f4', shape=(rows, wide_as)).flush()
        (out/'terminal').mkdir()
        np.lib.format.open_memmap(out/'terminal'/'l_out.npy', mode='w+', dtype='<f4', shape=(rows, wide_as)).flush()
        np.lib.format.open_memmap(out/'terminal'/'result_norm.npy', mode='w+', dtype='<f4', shape=(int(sum(steps)), wide_as)).flush()
        return wide_as

    apart = dict(initializer=os.sched_setaffinity, initargs=(0, threads_of(os.environ['KEEP_ON']))) if os.environ.get('KEEP_ON') else {}
    with ProcessPoolExecutor(keepers, mp_context=multiprocessing.get_context('fork'), **apart) as pool:
        hashed = pool.submit(digest, model), pool.submit(digest, recorder)   # beside the pass: no answer is ready to keep while the model loads
        if rollouts is None:
            cases.write_text(json.dumps([dict(id=m['id'], tokens=m['tokens'], teacher_tokens=text[m['id']]) for m in chosen]))
            with open(str(raw) + '.stdout.jsonl', 'w') as so, open(str(raw) + '.stderr.log', 'w') as se:
                pass_ = subprocess.Popen(under + [str(recorder), str(model), str(raw), str(cases), str(BACKENDS), 'capture', 'residual', 'stream'], env=env, stdout=subprocess.PIPE, stderr=se, text=True, start_new_session=True)
                watch = engine.watch(pass_)   # the pass's own map, read while it runs
                before = time.time()
                for line in pass_.stdout:   # a case's line comes after its stream is closed and its result.json and case.json are written
                    so.write(line)
                    so.flush()
                    try:
                        name = json.loads(line).get('id')
                    except ValueError:
                        continue
                    if name not in place:
                        continue
                    now = time.time()
                    told.append(dict(id=name, seconds=round(now - before, 2)))   # the first holds the wait for the card and the model's load
                    before = now
                    if width is None:   # the arrays are made once the first answer says how wide a state is
                        width = arrays_of(raw/name)
                    n = place[name]
                    keeping.append(pool.submit(keep, raw/name, out, text[name], chosen[n]['tokens'], questions[n], int(answer_from[n]), int(question_from[n]), merges, given_from))
                    failed = [each for each in keeping if each.done() and each.exception()]
                    if failed:   # an answer that cannot be kept ends the pass now: its raw streams would only fill the disk (11:03 AM CT November 14: eleven answers, 16 GB, before it was seen)
                        os.killpg(pass_.pid, signal.SIGTERM)
                        pass_.wait()
                        raise failed[0].exception()
                code = pass_.wait()
            recorder_seconds = time.time() - started   # with RECORDER_UNDER, the wait for the card's lock is in it
            if code:
                raise subprocess.CalledProcessError(code, pass_.args)
            engine.ended(watch, str(raw) + '.stderr.log', time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(started)))   # the pass ran this reading's engine, by its own map and its scheduler's own counts, or the reading ends here
        else:   # no pass: every answer's stream is here, written by the generation that wrote its text; each is kept as a fed pass's is, and removed as it is kept
            absent = [m['id'] for m in chosen if not (rollouts/m['id']/'stream.f32').is_file() or not (rollouts/m['id']/'result.json').is_file()]
            if absent:
                raise ValueError(f'ROLLOUTS holds no kept stream of {absent}: a reading is of every answer or of none')
            width = arrays_of(rollouts/chosen[0]['id'])
            keeping = [pool.submit(keep, rollouts/m['id'], out, text[m['id']], m['tokens'], questions[n], int(answer_from[n]), int(question_from[n]), merges, given_from, True)
                       for n, m in enumerate(chosen)]
            recorder_seconds = 0.0
        if len(keeping) != len(chosen):
            raise ValueError(f'the pass told {len(keeping)} of its {len(chosen)} answers')
        for each in keeping:
            print(json.dumps(each.result()), flush=True)
        hashed = hashed[0].result(), hashed[1].result()
        files = [out/'nodes'/str(m['merge'])/'state.npy' for m in merges] + [out/'terminal'/'l_out.npy', out/'terminal'/'result_norm.npy']
        for path in files:   # the one flush
            np.load(path, mmap_mode='r+').flush()
        sums = dict(zip(files, pool.map(digest, files)))
    if rollouts is None:
        shutil.rmtree(raw, ignore_errors=True)
        cases.unlink()

    limit = {m['id']: m['max_tokens'] for m in json.loads((tube.WIDE/'members.json').read_text())}   # the recordings' own limit, which an answer of the model's own is given (own_text.py)
    table, mapped = [], []
    for n, m in enumerate(chosen):
        mine = json.loads((out/'answers'/m['id']/'answer.json').read_text())
        whole, why = ending(mine['first_step_not_its_own'], mine['ended_with_an_end_token'], mine['steps'], limit[m['id']])
        table.append(dict(id=m['id'], group='correct' if m['correct'] else 'wrong', steps=mine['steps'], question=mine['question'], first_step_not_its_own=mine['first_step_not_its_own'],
                          steps_not_its_own=mine['steps_not_its_own'], ended_with_an_end_token=mine['ended_with_an_end_token'], limit=limit[m['id']],
                          the_whole_answer_is_this_models_own=whole, its_ending=why, answer_json_sha256=digest(out/'answers'/m['id']/'answer.json')))
        mapped.append(dict(id=m['id'], steps=mine['steps'], question=mine['question'], answer_rows_from=int(answer_from[n]), question_rows_from=int(question_from[n]),
                           prompt_sha256=mine['prompt_sha256'], text_sha256=mine['text_sha256']))
    json.dump(dict(order='all_answers_then_all_questions', rows=rows, answer_rows=int(sum(steps)), question_rows=int(sum(questions)), answers=mapped,
                   row_of='answer row s of an answer is input position question + s, where the decision for its token s is computed; question row q is input position q',
                   membership='the fitting answers (tube.picks) of the cohort run, in its order; the answers no fit uses are not read' if fitted
                   else f"{selection['of']}; of the cohort run, in its order (case manifest {selection['manifest_sha256']})"), open(out/'row-map.json', 'w'), indent=1)
    if capture is None:
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        card = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader', '-i', os.environ['CUDA_VISIBLE_DEVICES']], capture_output=True, text=True, check=True).stdout.strip()
        driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader', '-i', os.environ['CUDA_VISIBLE_DEVICES']], capture_output=True, text=True, check=True).stdout.strip()
    else:
        card, driver, visible = capture['card'], capture['driver'], capture['visible']
    described = lambda path, shape: dict(file=str(path.relative_to(out)), sha256=sums[path], shape=list(shape), dtype='<f4')
    flagged = [r for r in table if r['first_step_not_its_own'] is not None]
    record = dict(format=FORMAT,
                  what='a model read on a GIVEN text: the engine\'s own output of every merge, at every answer step and every question position of every ' + ('fitting answer; ' if fitted else 'answer of its case manifest; ') +
                       'rows after an answer\'s first step not its own are a conditional replay, not the model\'s own trace',
                  made_by='states_reading.py: one recorder pass for every answer (whole_reading.py\'s pass), each answer kept beside it' if capture is None else
                          'states_reading.py from the streams the model\'s own generation of these answers wrote (own_text.py, KEEP_RAW): no second pass; each answer kept as a fed pass\'s is',
                  made_with={'states_reading.py': digest(__file__), 'whole_reading.py': digest(here/'whole_reading.py'), 'reading_identity.py': digest(here/'reading_identity.py')},
                  run=declared.name, **({'pass_number': int(os.environ['PASS'])} if os.environ.get('PASS') else {}), declaration_sha256=digest(declared/'declaration.json'), declared_step=declaration['method_step'], cohort=cohort_run.name,
                  model=str(model.relative_to(ROOT)) if ROOT in model.parents else str(model), model_sha256=hashed[0], recorder_sha256=hashed[1],
                  recorder_arguments=['capture', 'residual', 'stream'], fusion_disabled='1',
                  libraries_sha256=engine.record['sha256'], libraries='the engine the pass ran, by the file the loader takes for each library and the scheduler\'s setting, hashed together (recorder_engine.py; the installed folder\'s own identity is engine.installed.libraries_sha256)',
                  engine=engine.described() if capture is None else dict(capture['engine'], passes_are='the generation\'s calls: this reading made no pass'),
                  card=card, driver=driver, visible=visible, started=time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime(started)),
                  text_given_from=given_from, text_written_by=source.get('model'), text_written_by_sha256=source.get('model_sha256'),
                  text='each answer\'s prompt, then its text supplied one token at a time; every decision has the engine\'s scores, so every supplied token is held to the model\'s own choice' if capture is None else
                       'each answer\'s prompt, then the model\'s own greedy choice at every step: the generation itself, no token supplied',
                  **({} if capture is None else dict(kept_from_the_generation=dict(capture='own/text.json of the folder the text is given from', text_json_sha256=digest(given/'own'/'text.json'),
                                                                                 capture_started=capture.get('started'), execution='the card, driver and card number are the capture\'s; the recorder, the engine libraries and the model are held equal to it'))),
                  rows=rows, width=width, row_map=dict(file='row-map.json', sha256=digest(out/'row-map.json')),
                  representation=dict(state='float32, the engine\'s own output of the merge\'s ADD as the recorder read it off the card: nothing derived, normalized or rounded',
                                      direction_and_log_size='not stored: x / |x| and ln |x| of a stored row, in the precision the consumer declares',
                                      held_by='at every row of every merge the engine\'s ADD made again from its two recorded operands (float32, as the engine adds) gives the kept row bit for bit'),
                  merges=[dict(m, width=width, space='residual', rows_whose_add_was_made_again=rows, **described(out/'nodes'/str(m['merge'])/'state.npy', (rows, width))) for m in merges],
                  terminal=dict(merge=merges[-1]['merge'],
                                kept_raw='nodes/%d/state.npy is the RAW last merge, in the residual stream: before the last layer\'s output scale and before the output norm' % merges[-1]['merge'],
                                l_out=dict(described(out/'terminal'/'l_out.npy', (rows, width)), label=f"l_out-{merges[-1]['layer']}", rows='every row, as the row map orders them',
                                           meaning='the last merge times the last layer\'s output scale, as the engine put it out: what the output norm reads'),
                                result_norm=dict(described(out/'terminal'/'result_norm.npy', (int(sum(steps)), width)), label='result_norm', rows='the answer rows only, in the row map\'s order (its first answer_rows rows)',
                                                 meaning='the engine\'s own output norm of l_out, which the head reads; the engine computes it only where it computes scores: every answer step, and in the prompt only the last position of each call of the prefill, which is not kept'),
                                composing='a comparison in output-norm space composes the model\'s output normalization once from the raw last merge; result_norm holds that composition to the engine at the answer rows'),
                  answers_read=[m['id'] for m in chosen], of=selection['of'], cases=selection,
                  answers_whose_text_is_not_this_models_own=len(flagged), answers_whose_ending_is_unresolved=[r['id'] for r in table if r['the_whole_answer_is_this_models_own'] is None],
                  generation='the prompt alone, greedy, each answer to its limit (table: limit); a given text is this model\'s own whole answer only where the table says so',
                  table=table, room=dict(free_gib_before=round(free / 2**30, 1), needed_gib_at_most=round(need / 2**30, 1), margin_gib=MARGIN_GIB), seconds_an_answer=told, recorder_seconds=round(recorder_seconds, 1), seconds=round(time.time() - started, 1), load=Path('/proc/loadavg').read_text().split()[:3])
    json.dump(record, open(out/'reading.json', 'w'), indent=1)
    print(json.dumps({key: record[key] for key in ('format', 'model', 'card', 'rows', 'width', 'text_given_from', 'answers_whose_text_is_not_this_models_own', 'answers_whose_ending_is_unresolved', 'recorder_seconds', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()
