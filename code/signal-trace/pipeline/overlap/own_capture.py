"""A model's own answers to the thirty fitting questions, read whole on card 1: every position of the prompt and of the answer the model writes
by itself, all 105 merges and the output norm, through each answer's own end.

From code/signal-trace:
  python -m pipeline.overlap.own_capture prepare <run> <model file>
  python -m pipeline.overlap.own_capture capture <run> <folder for the recorder's raw stream>
  python -m pipeline.overlap.own_capture record  <run>
  python -m pipeline.overlap.own_capture control-prepare <run>
  python -m pipeline.overlap.own_capture control-capture <run> <folder for the recorder's raw stream>

The reading a written candidate needs before anything is said of it (Astra, November 12, 2026: "capture all thirty fitting questions, every
prompt/answer position, all 105 merges plus output norm, through each own answer's end; neither the sign nor final-state paragraph is a
stop"). The original model's own answers on the same card are kept by fed-continuations-replay-original; a candidate is read beside them and
beside nothing else (two cards write other tokens: records/development/2026-11-12-path-forward/README.md).

prepare  One case a question: the question's recorded prompt tokens and the recordings' own limit of tokens, nothing supplied. The thirty
         are the fitting questions whose own answers fed-continuations-replay-original keeps. Kept in cases/ with index.json, which names
         the model file and its sha256.
capture  A question at a time: the recorder's lossless residual stream of the natural pass (the model writes every token itself), then
         kept/<question>.own in the form of fed_replay.py: first.npy (the state entering the first merge), added.npy (what each merge adds),
         result_norm.npy and steps.json, every state checked bit for bit against the raw stream before the stream is removed.
         What is the original's is kept once. The positions whose whole input is the original's (the prompt, and the answer through the row
         that writes the first token differing from the original's own answer on this card) are compared with the original's kept arrays,
         merge by merge, bit for bit. Where the embedding and every merge before some merge m are equal at all of them, those positions
         keep only the merges from m on (added-shared.npy) and `parts` takes the rest from the original's arrays. So for a model that is the
         original with one written matrix, m is that matrix's merge, and the count is itself a reading: nothing before it moved, to the bit.
record   Per question, beside the original's own answer on this card, what the capture holds: the step from which the two histories
         differ; whether the answer ended by itself; the positions, the shared positions and m, and how many of the shared positions are
         the prompt's and how many write the answer. Nothing is measured. Until November 15, 2026 the record also carried, at the shared
         positions and for every merge from m on, the distance between what the candidate's merge adds and what the original's adds over
         the original's size, and their cosine: a reading of a model by its state values, which left with `paired` (below). Until
         November 14, 2026 it carried the judge's fields of each answer; no instrument reads an answer since that evening (Marcos's
         direction). A record made before keeps what it held.

control-prepare, control-capture   The same-history control (the contract's section C): for each answer whose tokens are not the original's
         own answer on this card, the original model reads this model's text, every token but the last supplied and the last left to it
         (the recorder's terminal-token handling, as in fed_replay.py), with the same lossless trace under control/kept. Nothing is fed
         that this model did not write and no alternative answer is generated; the steps at which the original would have written another
         token are recorded and are expected. The positions through the row that writes the first differing token are the original's own
         and are kept once, as fed_replay.capture keeps a continuation's shared prefix.
paired   Left the tree on November 15, 2026 (Astra's decision: the state-value objective is withdrawn, and its measurement programs do
         not stay to run that history again). It read this model on its text against the original on the same text, at every position
         and merge, by the difference of the two unit states and of the two log sizes under the frozen reference's weights. The paired/
         folders it wrote stay as records, and its bytes are in git before that day
         (records/development/2026-11-12-path-forward/retired-state-value-readers.json). `outgoing` and `control_states` stay: they give
         a kept capture's and a control's states back, and measure nothing.

Data roles. A diagnostic reading of one model on the fitting questions. It supplies no target, mask, threshold or score, selects nothing
and judges no run: what a candidate's own path means is the assessment of the seat that owns the objective. The answers no fit uses are
not generated and reserved-40b is not opened."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from pipeline.acquisition.export_stream_traces import StreamRows, connections
from pipeline.acquisition.run_a_observation import run as observe
from pipeline.native_records.run_declaration import digest
from pipeline.overlap import fed_replay as base
from pipeline.paths import ORIGINAL_MODEL, ROOT, ROUND_TWO_RUNS

FIRST = 'fed-continuations-replay-original'       # the run whose kept/ holds the original's thirty own answers on card 1
MERGES = 105


def prepare(run, model):
    first = json.loads((ROUND_TWO_RUNS/FIRST/'cases'/'index.json').read_text())
    own = [each for each in first['cases'] if each['id'].endswith('.own')]
    index, at_most = [], 0
    folder = run/'cases'
    folder.mkdir(exist_ok=True)
    for each in own:
        question = each['question']
        asked = json.loads(base.asked(question).read_text())
        limit = json.loads((base.OWN/'own'/question/'answer.json').read_text())['limit']
        if asked['max_tokens'] != limit or len(asked['tokens']) != each['prompt_tokens']:
            raise ValueError(f"{question}: the recorded question's limit or prompt is not the one the original's own answer was read under")
        wanted = json.dumps([dict(id=f'{question}.own', tokens=asked['tokens'], max_tokens=limit)])
        file = folder/f'{question}.json'
        if file.exists() and file.read_text() != wanted:
            raise ValueError(f'{file.name}: an earlier list differs from this one')
        file.write_text(wanted)
        positions = len(asked['tokens']) - 1 + limit
        at_most += positions
        index.append(dict(id=f'{question}.own', question=question, group=each['group'], prompt_tokens=len(asked['tokens']), limit=limit, positions_at_most=positions,
                          the_originals_own_answer=f'{FIRST}/kept/{question}.own', the_originals_answer_tokens=each['answer_tokens']))
    written = dict(what='the cases of the reading: each fitting question by itself, the prompt alone, the recordings\' own limit of tokens, nothing supplied',
                   model=str(model.relative_to(ROOT)) if ROOT in model.parents else str(model), model_sha256=digest(model), model_bytes=model.stat().st_size,
                   the_originals_own_answers_kept_by=FIRST, built_from={str((ROUND_TWO_RUNS/FIRST/'cases'/'index.json').relative_to(ROOT)): digest(ROUND_TWO_RUNS/FIRST/'cases'/'index.json')},
                   questions=[each['question'] for each in index], cases=index, positions_at_most=at_most,
                   kept_gib_at_most=round(at_most * base.KEPT_ROWS * base.ROW / 2**30, 2),
                   kept_gib_if_the_answers_have_the_originals_lengths=round(sum((e['prompt_tokens'] - 1 + e['the_originals_answer_tokens']) * base.KEPT_ROWS for e in index) * base.ROW / 2**30, 2),
                   largest_raw_question_gib=round(max(e['positions_at_most'] for e in index) * base.RAW_ROWS * base.ROW / 2**30, 2))
    wanted = json.dumps(written, indent=1) + '\n'
    if (folder/'index.json').exists() and (folder/'index.json').read_text() != wanted:
        raise ValueError('index.json: an earlier index differs from this one')
    (folder/'index.json').write_text(wanted)
    free = shutil.disk_usage(run).free / 2**30
    print(json.dumps(dict(questions=len(index), by_group={g: sum(e['group'] == g for e in index) for g in ('wrong', 'correct')}, positions_at_most=at_most, kept_gib_at_most=written['kept_gib_at_most'],
                          kept_gib_if_the_answers_have_the_originals_lengths=written['kept_gib_if_the_answers_have_the_originals_lengths'], largest_raw_question_gib=written['largest_raw_question_gib'],
                          free_gib=round(free, 1), model_sha256=written['model_sha256'])))


def keep(raw, kept, scale, theirs):
    """One natural pass from the recorder's raw stream into kept/, every state checked bit for bit against the stream; what is the original's
    (`theirs`: the original's kept own answer of the question) is kept once."""
    result = json.loads((raw/'result.json').read_text())
    steps, prompt = result['steps'], result['prompt_tokens']
    if result['teacher_token_count'] or any(s['selection'] != 'greedy' for s in steps) or [s['input_position'] for s in steps] != [prompt - 1 + k for k in range(len(steps))]:
        raise ValueError(f'{raw.name}: the pass did not write every token itself at the positions after the prompt')
    positions = list(range(prompt - 1 + len(steps)))
    stream = StreamRows(raw, positions)
    try:
        path = connections(stream)
        width = stream.metadata[path[0]['incoming']]['width']
        if len(path) != MERGES:
            raise ValueError(f'{raw.name}: {len(path)} merges')
        first, added = np.empty((len(positions), width), '<f4'), np.empty((len(positions), len(path), width), '<f4')
        for p in positions:
            state = stream.values(path[0]['incoming'], p)
            first[p] = state
            for n, edge in enumerate(path):
                x, a, y = (stream.values(edge[key], p) for key in ('incoming', 'branch', 'merged'))
                made = stream.rows[edge['merged'], p]
                if made[5] != 'ADD' or sorted(made[6]) != sorted([edge['incoming'], edge['branch']]) or not np.array_equal(state.view('u4'), x.view('u4')) \
                        or not np.array_equal(np.add(x, a).view('u4'), y.view('u4')):
                    raise ValueError(f"{raw.name}: {edge['merged']} at position {p} is not the sum of the state before it and what the stream says was added")
                added[p, n] = a
                state = y
                if n % 3 == 2:
                    state = y * scale[edge['layer']]
                    if not np.array_equal(state.view('u4'), stream.values(edge['layer_exit'], p).view('u4')):
                        raise ValueError(f"{raw.name}: {edge['layer_exit']} at position {p} is not the last sum times the layer's output scale")
        norm = np.stack([stream.values('result_norm', prompt - 1 + k) for k in range(len(steps))]).astype('<f4')
        merges = [(edge['layer'], base.KINDS[k % 3]) for k, edge in enumerate(path)]
    finally:
        stream.close()
    record = json.loads((theirs/'steps.json').read_text())
    if record['kept']['from_position'] or record['prompt_tokens'] != prompt:
        raise ValueError(f"{raw.name}: the original's kept own answer is not whole, or not under a prompt of this length")      # that the prompt is the same one is the bit-for-bit comparison below
    mine, original = [s['selected'] for s in steps], [s['selected'] for s in record['steps']]
    differs = next((k for k in range(max(len(mine), len(original))) if k >= len(mine) or k >= len(original) or mine[k] != original[k]), None)
    # the positions whose whole input is the original's: the prompt, and the answer through the row that writes the first differing token
    shared = min(len(positions), record['kept']['positions'], prompt - 1 + (differs + 1 if differs is not None else len(mine)))
    their_first, their_added = np.load(theirs/'first.npy', mmap_mode='r'), np.load(theirs/'added.npy', mmap_mode='r')
    same_first = bool(np.array_equal(first[:shared].view('u4'), their_first[:shared].view('u4')))
    equal = [bool(np.array_equal(added[:shared, n].view('u4'), their_added[:shared, n].view('u4'))) for n in range(MERGES)]
    before = next((n for n, e in enumerate(equal) if not e), MERGES) if same_first else 0       # the first merge whose added part is not the original's at some shared position
    if not before:
        shared = 0
    kept.mkdir(parents=True)
    np.save(kept/'first.npy', first[shared:])
    np.save(kept/'added.npy', added[shared:])
    if shared:
        np.save(kept/'added-shared.npy', added[:shared, before:])
    np.save(kept/'result_norm.npy', norm)
    slim = {key: result[key] for key in ('id', 'mode', 'scope', 'fusion_disabled', 'tokens', 'content', 'ended', 'stopped_limit', 'prompt_tokens', 'selection', 'teacher_completed', 'teacher_token_count',
                                           'conditioned_continuation', 'capture_events', 'tensor_bytes') if key in result}
    slim['steps'] = [{key: s[key] for key in ('step', 'input_position', 'selected', 'piece', 'logits_sha256', 'top', 'greedy_token', 'selection', 'selected_logit')} for s in steps]
    slim['merges'] = merges
    slim['raw'] = {name: dict(sha256=digest(raw/name), bytes=(raw/name).stat().st_size) for name in ('result.json', 'stream.jsonl', 'stream.f32')}
    slim['kept'] = dict(positions=len(positions), from_position=shared, merges=MERGES, width=width, first_token_differing_from_the_originals_own_answer=differs,
                        positions_whose_whole_input_is_the_originals=prompt - 1 + (differs + 1 if differs is not None else len(mine)),
                        shared_positions=shared, shared_positions_keep_the_merges_from=before if shared else None,
                        merges_from_there_equal_to_the_originals_at_every_shared_position=[n for n in range(before, MERGES) if equal[n]] if shared else None,
                        earlier_positions_and_merges_are_in=theirs.name if shared else None, earlier_positions_and_merges_are_in_run=theirs.parent.parent.name if shared else None,
                        checked='every incoming, added and merged state and every layer exit, bit for bit, against the raw stream; the shared positions\' embedding and earlier merges bit for bit against the original\'s kept arrays')
    (kept/'steps.json').write_text(json.dumps(slim) + '\n')
    return slim['kept']


def parts(run, case):
    """A kept own answer whole: first (positions x width), added (positions x merges x width), and its record; what is the original's is read
    from the original's kept arrays."""
    folder = run/'kept'/case
    record = json.loads((folder/'steps.json').read_text())
    kept = record['kept']
    first, added = np.load(folder/'first.npy'), np.load(folder/'added.npy')
    if kept['shared_positions']:
        theirs = ROUND_TWO_RUNS/kept['earlier_positions_and_merges_are_in_run']/'kept'/kept['earlier_positions_and_merges_are_in']
        n, m = kept['shared_positions'], kept['shared_positions_keep_the_merges_from']
        head = np.concatenate([np.load(theirs/'added.npy', mmap_mode='r')[:n, :m], np.load(folder/'added-shared.npy')], axis=1)
        first, added = np.concatenate([np.load(theirs/'first.npy', mmap_mode='r')[:n], first]), np.concatenate([head, added])
    if len(first) != kept['positions'] or added.shape[:2] != (kept['positions'], MERGES):
        raise ValueError(f'{case}: the kept arrays do not hold its {kept["positions"]} positions')
    return first, added, record


def declaration(run):
    """The run's declaration; the declaring run's when this reading is a folder of that run."""
    return run/'declaration.json' if (run/'declaration.json').exists() else run.parent/'declaration.json'


def capture(run, raw_root):
    index = json.loads((run/'cases'/'index.json').read_text())
    model = ROOT/index['model'] if not Path(index['model']).is_absolute() else Path(index['model'])
    if model.stat().st_size != index['model_bytes'] or digest(model) != index['model_sha256']:
        raise ValueError('The model file is not the one the cases were prepared for')
    by_id = {each['id']: each for each in index['cases']}
    kept = run/'kept'
    kept.mkdir(exist_ok=True)
    scale = base.scales()
    np.save(kept/'scales.npy', scale)
    for n, question in enumerate(index['questions']):
        name = f'{question}.own'
        done, listed_file, raw = kept/f'{question}.done', run/'cases'/f'{question}.json', raw_root/question
        if done.exists():
            continue
        need = by_id[name]['positions_at_most'] * (base.RAW_ROWS + base.KEPT_ROWS) * base.ROW / 2**30
        free = shutil.disk_usage(run).free / 2**30
        if free - need < base.MARGIN_GIB or shutil.disk_usage(raw_root).free / 2**30 - need < base.MARGIN_GIB:
            raise ValueError(f'{question}: {free:.0f} GiB free, it needs up to {need:.0f} beside the {base.MARGIN_GIB} GiB margin')
        if (kept/name).exists() or raw.exists():
            raise ValueError(f'{question}: it is partly kept or its raw stream exists; inspect it')
        began = time.time()
        observe(model, listed_file, raw, base.CARD, 'natural', 'capture', 'residual', declaration(run))   # takes the card's lock itself; refuses an output that exists
        engine = time.time() - began
        told = keep(raw/name, kept/name, scale, ROUND_TWO_RUNS/FIRST/'kept'/name)
        shutil.copyfile(raw/'execution.json', kept/f'{question}.execution.json')
        for suffix in ('.stdout.jsonl', '.stderr.log'):
            shutil.copyfile(raw.with_suffix(suffix), kept/f'{question}{suffix}')
        told = dict(question=question, recorder_seconds=round(engine, 1), kept_seconds=round(time.time() - began - engine, 1), free_gib_before=round(free, 1), **told)
        done.write_text(json.dumps(told) + '\n')
        shutil.rmtree(raw)                      # this question's temporary raw stream, now held losslessly in kept/
        for suffix in ('.stdout.jsonl', '.stderr.log'):
            raw.with_suffix(suffix).unlink()
        print(json.dumps(dict(told, n=n + 1, of=len(index['questions']))), flush=True)


def record(run):
    index = json.loads((run/'cases'/'index.json').read_text())
    out = run/'record'
    out.mkdir(exist_ok=True)
    told = []
    for each in index['cases']:
        folder = run/'kept'/each['id']
        mine = json.loads((folder/'steps.json').read_text())
        kept, prompt = mine['kept'], mine['prompt_tokens']
        theirs_folder = ROUND_TWO_RUNS/FIRST/'kept'/each['id']
        theirs = json.loads((theirs_folder/'steps.json').read_text())
        differs = kept['first_token_differing_from_the_originals_own_answer']
        row = dict(id=each['id'], question=each['question'], group=each['group'], steps=len(mine['steps']), the_originals_steps=len(theirs['steps']), ended_by_itself=bool(mine['ended']),
                   stopped_at_the_limit=bool(mine.get('stopped_limit')), the_text_is_the_originals_on_this_card=differs is None, first_differing_token=differs,
                   positions=kept['positions'], shared_positions=kept['shared_positions'], shared_positions_keep_the_merges_from=kept['shared_positions_keep_the_merges_from'],
                   merges_from_there_equal_to_the_originals_at_every_shared_position=kept['merges_from_there_equal_to_the_originals_at_every_shared_position'])
        if kept['shared_positions']:
            writes = np.arange(kept['shared_positions']) >= prompt - 1
            row['at_the_shared_positions'] = dict(of_the_prompt=int((~writes).sum()), that_write_the_answer=int(writes.sum()))
        told.append(row)
        print(json.dumps(row), flush=True)
    questions = [json.loads(file.read_text()) for file in sorted((run/'kept').glob('*.done'))]
    (out/'own.json').write_text(json.dumps(dict(
        what='per question, this model\'s own answer on card 1 beside the original\'s own answer on card 1: the positions whose whole input is the original\'s and the first merge that differs there; '
             'nothing is measured', model=index['model'], model_sha256=index['model_sha256'], the_originals_own_answers_kept_by=FIRST, questions=questions, cases=told), indent=1) + '\n')
    print(json.dumps(dict(cases=len(told),
                          first_merge_that_differs=sorted({r['shared_positions_keep_the_merges_from'] for r in told if r['shared_positions']}), ended_by_itself=sum(r['ended_by_itself'] for r in told))))


def control_prepare(run):
    index = json.loads((run/'cases'/'index.json').read_text())
    folder = run/'control'/'cases'
    folder.mkdir(parents=True, exist_ok=True)
    listed, same = [], []
    for each in index['cases']:
        mine = json.loads((run/'kept'/each['id']/'steps.json').read_text())
        differs = mine['kept']['first_token_differing_from_the_originals_own_answer']
        if differs is None:
            same.append(each['id'])
            continue
        tokens, prompt, name = [s['selected'] for s in mine['steps']], json.loads((run/'cases'/f"{each['question']}.json").read_text())[0]['tokens'], f"{each['question']}.original-on-this-text"
        if len(prompt) != mine['prompt_tokens']:
            raise ValueError(f"{each['id']}: its case's prompt is not the one the kept answer was read under")
        wanted = json.dumps([dict(id=name, tokens=prompt, teacher_prefix_tokens=tokens[:-1], max_tokens=len(tokens))])
        file = folder/f"{each['question']}.json"
        if file.exists() and file.read_text() != wanted:
            raise ValueError(f'{file.name}: an earlier list differs from this one')
        file.write_text(wanted)
        listed.append(dict(id=name, question=each['question'], group=each['group'], this_models_case=each['id'], prompt_tokens=len(prompt), answer_tokens=len(tokens), supplied_tokens=len(tokens) - 1,
                           the_texts_last_token_left_to_the_model=tokens[-1], the_text_ended_by_itself=bool(mine['ended']), tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                           positions=len(prompt) - 1 + len(tokens), first_token_differing_from_the_originals_own_answer=differs, positions_through_the_first_differing_token=len(prompt) - 1 + differs + 1))
    written = dict(what="the same-history control: the original model reads this model's own text, every token but the last supplied", this_models_answers=index['model'], this_models_sha256=index['model_sha256'],
                   the_originals_own_answers_kept_by=FIRST, questions=[e['question'] for e in listed], cases=listed, answers_with_the_originals_text_not_read_again=same,
                   positions=sum(e['positions'] for e in listed), positions_kept_if_every_shared_prefix_is_equal=sum(e['positions'] - e['positions_through_the_first_differing_token'] for e in listed))
    (folder/'index.json').write_text(json.dumps(written, indent=1) + '\n')
    print(json.dumps(dict(read=len(listed), the_originals_text=len(same), positions=written['positions'], positions_kept=written['positions_kept_if_every_shared_prefix_is_equal'],
                          kept_gib=round(written['positions_kept_if_every_shared_prefix_is_equal'] * base.KEPT_ROWS * base.ROW / 2**30, 2), free_gib=round(shutil.disk_usage(run).free / 2**30, 1))))


def control_capture(run, raw_root):
    index = json.loads((run/'control'/'cases'/'index.json').read_text())
    by_id = {each['id']: each for each in index['cases']}
    kept = run/'control'/'kept'
    kept.mkdir(exist_ok=True)
    scale = base.scales()
    np.save(kept/'scales.npy', scale)
    for n, question in enumerate(index['questions']):
        name = f'{question}.original-on-this-text'
        done, listed_file, raw = kept/f'{question}.done', run/'control'/'cases'/f'{question}.json', raw_root/question
        if done.exists():
            continue
        need = by_id[name]['positions'] * (base.RAW_ROWS + base.KEPT_ROWS) * base.ROW / 2**30
        free = shutil.disk_usage(run).free / 2**30
        if free - need < base.MARGIN_GIB or shutil.disk_usage(raw_root).free / 2**30 - need < base.MARGIN_GIB:
            raise ValueError(f'{question}: {free:.0f} GiB free, it needs {need:.0f} beside the {base.MARGIN_GIB} GiB margin')
        if (kept/name).exists() or raw.exists():
            raise ValueError(f'{question}: it is partly kept or its raw stream exists; inspect it')
        began = time.time()
        observe(ORIGINAL_MODEL, listed_file, raw, base.CARD, 'conditional', 'capture', 'residual', declaration(run))
        engine = time.time() - began
        same = base.keep(raw/name, kept/name, scale, by_id[name]['positions_through_the_first_differing_token'], ROUND_TWO_RUNS/FIRST/'kept'/f'{question}.own')
        shutil.copyfile(raw/'execution.json', kept/f'{question}.execution.json')
        for suffix in ('.stdout.jsonl', '.stderr.log'):
            shutil.copyfile(raw.with_suffix(suffix), kept/f'{question}{suffix}')
        made = json.loads((kept/name/'steps.json').read_text())
        told = dict(question=question, recorder_seconds=round(engine, 1), kept_seconds=round(time.time() - began - engine, 1), free_gib_before=round(free, 1), shared_positions_equal_the_originals_own_answers=same,
                    steps=len(made['steps']), steps_where_the_original_would_have_written_another_token=[s['step'] for s in made['steps'] if s['greedy_token'] != s['selected']])
        done.write_text(json.dumps(told) + '\n')
        shutil.rmtree(raw)
        for suffix in ('.stdout.jsonl', '.stderr.log'):
            raw.with_suffix(suffix).unlink()
        print(json.dumps(dict(told, n=n + 1, of=len(index['questions']))), flush=True)


def outgoing(first, added, scale):
    """The state each merge puts out, by the engine's own float32 sums: positions x merges x width."""
    merged, state = np.empty(added.shape, '<f4'), np.array(first, dtype='<f4')
    for n in range(added.shape[1]):
        merged[:, n] = np.add(state, added[:, n])
        state = merged[:, n] * scale[n // 3] if n % 3 == 2 else merged[:, n]
    return merged


def control_states(run, case):
    """fed_replay.states for a control case: its kept arrays, the positions it shares with the original's own answer from that answer's arrays."""
    folder = run/'control'/'kept'/case
    record = json.loads((folder/'steps.json').read_text())
    first, added = np.load(folder/'first.npy'), np.load(folder/'added.npy')
    start = record['kept']['from_position']
    if start:
        own = ROUND_TWO_RUNS/record['kept']['earlier_positions_are_in_run']/'kept'/record['kept']['earlier_positions_are_in']
        first, added = np.concatenate([np.load(own/'first.npy', mmap_mode='r')[:start], first]), np.concatenate([np.load(own/'added.npy', mmap_mode='r')[:start], added])
    return dict(added=added, merged=outgoing(first, added, np.load(run/'control'/'kept'/'scales.npy'))), None, record


if __name__ == '__main__':
    command, run = sys.argv[1], ROUND_TWO_RUNS/sys.argv[2]
    if command == 'prepare':
        prepare(run, Path(sys.argv[3]).resolve())
    elif command == 'capture':
        capture(run, Path(sys.argv[3]).resolve())
    elif command == 'record':
        record(run)
    elif command == 'control-prepare':
        control_prepare(run)
    elif command == 'control-capture':
        control_capture(run, Path(sys.argv[3]).resolve())
    else:
        sys.exit(__doc__)
