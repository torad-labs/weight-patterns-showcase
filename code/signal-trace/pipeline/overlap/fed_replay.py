"""fed-continuations-replay-original: the whole trace of the answers the original model wrote after a fed word, beside its own answers.
Astra's request of November 12, 2026 (the brainstorm on the correction request): "an original-model trace replay of the existing
label/heading/climb intervention continuations, including failed continuations and correct-case controls ... the question/prefill and every
token/merge through both energy computations; actual incoming/contribution/outgoing data, direction/size, token IDs, forced-position flags,
and decision scores".

From code/signal-trace:
  python -m pipeline.overlap.fed_replay prepare|capture|record <run> [<folder for the recorder's raw stream>]
      (studies/04-round-2-overlap/runs/<run>/run.sh runs them)

prepare  The cases, from texts that exist: for each of the thirty fitting answers, the original's own answer on card 1 (the capture the feeds
         started from); the label swap's continuations; the heading's, the heading-and-label's and the climb's continuations. Every token of
         a text but its last is supplied to the recorder, which reads it in single-token steps as the feeds did. The last token the model
         writes itself, because the recorder refuses an end-of-generation token in supplied text (74 texts end with one) and refuses
         supplied text that leaves the model no token of its own (the 4 texts that stopped at the limit); the record says whether the
         model wrote the text's last token. Kept in
         cases/: one list per question (its own answer first, then its continuations), and index.json with each case's question, group,
         source, its forced steps (the steps at which its source pass supplied a token that was not the model's own first choice) and the
         first of them. It refuses a list that differs from one already written.
capture  Question by question on card 1: run_a_observation.run(original model, the question's list, card 1, 'conditional', 'capture',
         'residual'), the protocol of same-prefix-135-original with supplied tokens. The runner takes the card's lock itself, once per
         question. The recorder's raw stream is 1.5 MB a position (247 tensors of 1536 float32), so the raw stream of a question is written
         to the temporary folder given, read into kept/ and then removed from the temporary folder. What kept/ holds is lossless:
           first.npy        the first incoming state (the scaled embedding) of each kept position
           added.npy        what each of the 105 merges adds at each kept position
           result_norm.npy  the output norm's result at every step
           steps.json       the tokens read, each step's scores record, the merges, the raw files' sha256 and what was checked
         While a case is read, every incoming, added and merged state and every layer's exit is checked, bit for bit, against the recorder's
         own tensors: a merged state is the float32 sum of its incoming state and what the merge adds, the next merge's incoming state is that
         sum, and a layer's exit is its last sum times the layer's recorded output scale. So the three arrays are rebuilt from first.npy and
         added.npy by the same sums (`states`).
         A continuation shares its question and its text before the first forced step with the own answer. Its positions through the row
         that writes the first forced token are compared with the own answer's, bit for bit; where they are equal (the state before a forced
         choice is unchanged) only the positions after them are kept and `states` takes the earlier ones from the own answer; where they are
         not, every position is kept and the case says so. A question whose marker exists is not captured again.
record   Per case, from kept/: that the tokens read are the text's tokens; the forced steps; at each forced step the eight best tokens
         and the scores of the fed token and of the model's own choice; the steps at which the model's own first choice is not the token of
         the text (expected at forced steps only: then the text is what the original writes by itself under those feeds); how many steps
         have the source pass's full scores to the bit; and whether the positions through the first forced choice hold the own answer's
         states. Until November 15, 2026 the record also carried the paired same-history row (at the position whose input is the first fed
         token, the difference of the two outgoing states over the own state's size and their cosine at each merge, and the same for what
         the merge adds): a movement by state values, which left on Astra's decision of that day. A record made before keeps it.

Rows and steps, as in same_prefix.py: step n is the row at position (prompt tokens - 1) + n; it writes answer token n.

Data roles. Diagnostic and reference-construction observations of the unchanged original model on the thirty fitting answers. Nothing joins
the frozen reference, the cohort, the fitting folds or a score; an answer after a fed token is no model's own result; no weight is solved
and no target, mask or threshold is made. The answers no fit uses are not read and reserved-40b is not opened."""
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
from pipeline.paths import ACQUISITION, ORIGINAL_MODEL, ROOT, ROUND_TWO_RUNS

FEEDS = ROOT/'records/development/2026-11-12-path-forward/label-swap'
OWN = ROUND_TWO_RUNS/'correction-at-19/own/controls/original'
TENSORS = ROUND_TWO_RUNS/'sent-overlap/wrong-inputs/tensors.json'
CARD = 1
KINDS = ('attention', 'feedforward', 'layer')   # the three merges of a layer, in the model's order (as same_prefix.py names them)
ROW = 1536 * 4                                  # bytes of one recorded tensor at one position
RAW_ROWS, KEPT_ROWS = 247, 107                  # tensors a position has in the raw stream; arrays' rows kept for it at most (the declaration derives both)
MARGIN_GIB = 40                                  # the machine's disk rule, as same-prefix-135-original/run.sh applies it


def scales():
    tensors = json.loads(TENSORS.read_text())
    return np.array([np.fromfile(ORIGINAL_MODEL, dtype='<f4', count=1, offset=tensors[f'blk.{layer}.layer_output_scale.weight']['offset_bytes'])[0] for layer in range(35)], dtype='<f4')


def listed():
    """Every case from the texts on record: (name, question, group, source, tokens, forced steps, the source pass's folder or None, the source case)."""
    swap = json.loads((FEEDS/'label-swap.json').read_text())
    feeds = json.loads((FEEDS/'second-feeds/second-feeds.json').read_text())
    if Path(swap['capture']).resolve() != OWN.resolve():
        raise ValueError('the label swap did not start from the capture this run names')
    fed_cases = {}
    for file, passes in ((FEEDS/'cases.json', FEEDS/'plain'), (FEEDS/'second-feeds/plain-1.cases.json', FEEDS/'second-feeds/plain-1'), (FEEDS/'second-feeds/plain-2.cases.json', FEEDS/'second-feeds/plain-2')):
        for case in json.loads(file.read_text()):
            fed_cases[case['id']] = (case, passes/case['id'])
    cases = []
    for i, a in swap['answers'].items():
        own = json.loads((OWN/'own'/i/'answer.json').read_text())['tokens']
        cases.append((f'{i}.own', i, a['group'], 'its own answer', own, [], None, None))
        if 'tokens' in a:
            cases.append((f'{i}.label', i, a['group'], f"the label: {a['fed']!r} fed for {a['wrote']!r}", a['tokens'], [a['step']], *fed_cases[f"{i}.label-{a['step']:04d}"][::-1]))
        h = feeds['heading'].get(i)
        if h:
            cases.append((f'{i}.heading', i, a['group'], f"the heading: {h['fed']!r} fed for {h['wrote']!r}", h['after']['tokens'], [h['step']], *fed_cases[f"{i}.heading-{h['step']:04d}"][::-1]))
            if 'with_the_label' in h:
                at = h['with_the_label']['step']
                cases.append((f'{i}.heading-label', i, a['group'], f"the heading ({h['fed']!r} for {h['wrote']!r}) and the label ('Total' for 'Kin')", h['with_the_label']['after']['tokens'], [h['step'], at],
                              *fed_cases[f'{i}.heading-and-label-{at:04d}'][::-1]))
        c = feeds['climb'].get(i)
        if c:
            cases.append((f'{i}.climb', i, a['group'], f"the label and the climb's word: {c['fed']!r} fed for {c['wrote']!r}", c['after']['tokens'], None, *fed_cases[f"{i}.climb-{c['step']:04d}"][::-1]))
    return cases, swap


def asked(question):
    """The question's case.json among the acquisition's folders; the reserved set's folder is passed over by its name and never looked into."""
    found = [sub/question/'case.json' for sub in sorted(ACQUISITION.iterdir()) if sub.is_dir() and not sub.name.startswith('reserved') and (sub/question/'case.json').exists()]
    if len(found) != 1:
        raise ValueError(f'{question}: {len(found)} recorded questions of that name')
    return found[0]


def prepare(run):
    cases, swap = listed()
    own = {question: tokens for name, question, _, _, tokens, _, _, _ in cases if name.endswith('.own')}
    index, batches = [], {}
    for name, question, group, source, tokens, forced, pass_folder, fed_case in cases:
        prompt = json.loads(asked(question).read_text())['tokens']
        if fed_case is not None:
            made = json.loads((pass_folder/'result.json').read_text())
            prefix = fed_case['teacher_prefix_tokens']
            if fed_case['tokens'] != prompt or made['prompt_tokens'] != len(prompt) or [s['selected'] for s in made['steps']] != tokens or tokens[:len(prefix)] != prefix:
                raise ValueError(f'{name}: the source pass does not hold these tokens under this prompt')
            fed_steps = [k for k, s in enumerate(made['steps'][:len(prefix)]) if s['greedy_token'] != s['selected']]   # supplied in the source pass against the model's own first choice
            if forced is None:
                forced = fed_steps
            if forced != fed_steps or any(s['selection'] != ('teacher' if k < len(prefix) else 'greedy') for k, s in enumerate(made['steps'])):
                raise ValueError(f"{name}: the forced steps on record ({forced}) are not the source pass's supplied tokens that were not the model's own choice ({fed_steps})")
            if tokens[:forced[0]] != own[question][:forced[0]]:
                raise ValueError(f'{name}: its text before the first forced step is not the own answer')
        ended = made['ended'] if fed_case is not None else json.loads((OWN/'own'/question/'answer.json').read_text())['ended_by_itself']
        supplied = tokens[:-1]          # the recorder refuses a supplied end-of-generation token, and supplied text that leaves the model no token: the model writes the last one itself
        batches.setdefault(question, []).append(dict(id=name, tokens=prompt, teacher_prefix_tokens=supplied, max_tokens=len(tokens)))
        positions = len(prompt) - 1 + len(tokens)
        shared = len(prompt) - 1 + forced[0] + 1 if forced else 0       # the positions through the row that writes the first forced token
        index.append(dict(id=name, question=question, group=group, source=source, prompt_tokens=len(prompt), answer_tokens=len(tokens), supplied_tokens=len(supplied),
                          the_texts_last_token_left_to_the_model=tokens[-1], the_text_ended_by_itself=bool(ended), tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(), positions=positions, forced_steps=forced,
                          first_forced_step=forced[0] if forced else None, positions_through_the_first_forced_choice=shared,
                          source_pass=None if pass_folder is None else str((pass_folder/'result.json').relative_to(ROOT)),
                          source_pass_sha256=None if pass_folder is None else digest(pass_folder/'result.json')))
    folder = run/'cases'
    folder.mkdir(exist_ok=True)
    total = sum(each['positions'] for each in index)
    kept = sum(each['positions'] - each['positions_through_the_first_forced_choice'] for each in index)
    steps = sum(each['answer_tokens'] for each in index)
    largest = max(sum(len(c['tokens']) - 1 + c['max_tokens'] for c in batch) for batch in batches.values())
    written = dict(what='the cases of the replay: every token of a text but its last is supplied; the model writes the last token itself',
                   built_from={str(p.relative_to(ROOT)): digest(p) for p in (FEEDS/'label-swap.json', FEEDS/'second-feeds/second-feeds.json', FEEDS/'cases.json', FEEDS/'second-feeds/plain-1.cases.json',
                                                                              FEEDS/'second-feeds/plain-2.cases.json')},
                   own_capture=str(OWN.relative_to(ROOT)), model_sha256=swap['model_sha256'], questions=list(batches), cases=index, positions=total, positions_kept_if_every_shared_prefix_is_equal=kept,
                   kept_gib=round(((kept * (KEPT_ROWS - 1) + steps) * ROW) / 2**30, 2), kept_gib_if_no_shared_prefix_is_equal=round(((total * (KEPT_ROWS - 1) + steps) * ROW) / 2**30, 2),
                   largest_raw_question_gib=round(largest * RAW_ROWS * ROW / 2**30, 2))
    for question, batch in batches.items():
        wanted = json.dumps(batch)
        file = folder/f'{question}.json'
        if file.exists() and file.read_text() != wanted:
            raise ValueError(f'{file.name}: an earlier list differs from this one')
        file.write_text(wanted)
    wanted = json.dumps(written, indent=1) + '\n'
    if (folder/'index.json').exists() and (folder/'index.json').read_text() != wanted:
        raise ValueError('index.json: an earlier index differs from this one')
    (folder/'index.json').write_text(wanted)
    free = shutil.disk_usage(run).free / 2**30
    print(json.dumps(dict(cases=len(index), questions=len(batches), positions=total, positions_kept=kept, kept_gib=written['kept_gib'], kept_gib_at_most=written['kept_gib_if_no_shared_prefix_is_equal'],
                          largest_raw_question_gib=written['largest_raw_question_gib'], free_gib=round(free, 1), fits=free - written['kept_gib_if_no_shared_prefix_is_equal'] - written['largest_raw_question_gib'] >= MARGIN_GIB,
                          by_source={k: sum(e['id'].endswith('.' + k) for e in index) for k in ('own', 'label', 'heading', 'heading-label', 'climb')},
                          texts_that_ended_by_themselves=sum(e['the_text_ended_by_itself'] for e in index),
                          forced={e['id']: e['forced_steps'] for e in index if len(e['forced_steps']) > 1})))


def keep(raw, kept, scale, shared=0, own=None):
    """One case from the recorder's raw stream into kept/, every state checked bit for bit against the stream as it is read. `shared`: its
    positions through the first forced choice, compared with the kept own answer's (`own`) and left out where they are equal."""
    result = json.loads((raw/'result.json').read_text())
    steps, prompt = result['steps'], result['prompt_tokens']
    fed = result['teacher_token_count']
    if len(steps) - fed != 1 or any(s['selection'] != ('teacher' if k < fed else 'greedy') for k, s in enumerate(steps)) or [s['input_position'] for s in steps] != [prompt - 1 + k for k in range(len(steps))]:
        raise ValueError(f'{raw.name}: the pass did not read every supplied token and then write one of its own, at the positions after the prompt')
    positions = list(range(prompt - 1 + len(steps)))
    stream = StreamRows(raw, positions)
    try:
        path = connections(stream)
        width = stream.metadata[path[0]['incoming']]['width']
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
        merges = [(edge['layer'], KINDS[k % 3]) for k, edge in enumerate(path)]
    finally:
        stream.close()
    same = None
    if shared and own is not None:
        theirs = json.loads((own/'steps.json').read_text())
        same = bool(theirs['kept']['from_position'] == 0 and theirs['prompt_tokens'] == prompt
                    and np.array_equal(first[:shared].view('u4'), np.load(own/'first.npy', mmap_mode='r')[:shared].view('u4'))
                    and np.array_equal(added[:shared].view('u4'), np.load(own/'added.npy', mmap_mode='r')[:shared].view('u4')))
    start = shared if same else 0
    kept.mkdir(parents=True)
    np.save(kept/'first.npy', first[start:])
    np.save(kept/'added.npy', added[start:])
    np.save(kept/'result_norm.npy', norm)
    slim = {key: result[key] for key in ('id', 'mode', 'scope', 'fusion_disabled', 'tokens', 'content', 'ended', 'stopped_limit', 'prompt_tokens', 'selection', 'teacher_completed', 'teacher_token_count',
                                           'conditioned_continuation', 'capture_events', 'tensor_bytes') if key in result}
    slim['steps'] = [{key: s[key] for key in ('step', 'input_position', 'selected', 'piece', 'logits_sha256', 'top', 'greedy_token', 'selection', 'selected_logit')} for s in steps]
    slim['merges'] = merges
    slim['raw'] = {name: dict(sha256=digest(raw/name), bytes=(raw/name).stat().st_size) for name in ('result.json', 'stream.jsonl', 'stream.f32')}
    slim['kept'] = dict(positions=len(positions), from_position=start, merges=len(path), width=width, positions_through_the_first_forced_choice=shared,
                        those_positions_hold_the_own_answers_states_bit_for_bit=same, earlier_positions_are_in=own.name if start and own is not None else None,
                        earlier_positions_are_in_run=own.parent.parent.name if start and own is not None else None,
                        checked='every incoming, added and merged state and every layer exit, bit for bit, against the raw stream')
    (kept/'steps.json').write_text(json.dumps(slim) + '\n')
    return same


def states(run, case):
    """A kept case's states, rebuilt by the engine's own float32 sums: positions x merges x width arrays of the state each merge adds to
    (incoming), what it adds (added) and the state it puts out (merged), the three arrays same_prefix.loaded returns; the merges; and the
    case's record (steps.json). Positions a continuation shares with the own answer come from the own answer's arrays."""
    folder = run/'kept'/case
    record = json.loads((folder/'steps.json').read_text())
    first, added = np.load(folder/'first.npy'), np.load(folder/'added.npy')
    start = record['kept']['from_position']
    if start:
        holder = record['kept'].get('earlier_positions_are_in_run')
        own = (ROUND_TWO_RUNS/holder if holder and holder != run.name else run)/'kept'/record['kept']['earlier_positions_are_in']
        first = np.concatenate([np.load(own/'first.npy', mmap_mode='r')[:start], first])
        added = np.concatenate([np.load(own/'added.npy', mmap_mode='r')[:start], added])
    scale = np.load(run/'kept'/'scales.npy')
    incoming, merged = np.empty(added.shape, '<f4'), np.empty(added.shape, '<f4')
    state = first.copy()
    for n in range(added.shape[1]):
        incoming[:, n] = state
        merged[:, n] = np.add(state, added[:, n])
        state = merged[:, n] * scale[n // 3] if n % 3 == 2 else merged[:, n]
    return dict(incoming=incoming, added=added, merged=merged), [tuple(m) for m in record['merges']], record


def capture(run, raw_root, own_run=None):
    """own_run: the run whose kept/ holds the own answers, when this run does not capture them itself (a follow-up capture)."""
    index = json.loads((run/'cases'/'index.json').read_text())
    by_id = {each['id']: each for each in index['cases']}
    kept = run/'kept'
    own_kept = ROUND_TWO_RUNS/own_run/'kept' if own_run else kept
    kept.mkdir(exist_ok=True)
    scale = scales()
    np.save(kept/'scales.npy', scale)
    for n, question in enumerate(index['questions']):
        done, listed_file, raw = kept/f'{question}.done', run/'cases'/f'{question}.json', raw_root/question
        if done.exists():
            continue
        names = [c['id'] for c in json.loads(listed_file.read_text())]
        need = sum(by_id[name]['positions'] for name in names) * (RAW_ROWS + KEPT_ROWS) * ROW / 2**30
        free = shutil.disk_usage(run).free / 2**30
        if free - need < MARGIN_GIB or shutil.disk_usage(raw_root).free / 2**30 - need < MARGIN_GIB:
            raise ValueError(f'{question}: {free:.0f} GiB free, its cases need {need:.0f} beside the {MARGIN_GIB} GiB margin')
        if any((kept/name).exists() for name in names) or raw.exists():
            raise ValueError(f'{question}: a case of it is partly kept or its raw stream exists; inspect it')
        began = time.time()
        observe(ORIGINAL_MODEL, listed_file, raw, CARD, 'conditional', 'capture', 'residual', run/'declaration.json')   # takes the card's lock itself; refuses an output that exists
        engine = time.time() - began
        same = {name: keep(raw/name, kept/name, scale, by_id[name]['positions_through_the_first_forced_choice'], own_kept/f'{question}.own') for name in names}
        shutil.copyfile(raw/'execution.json', kept/f'{question}.execution.json')
        for suffix in ('.stdout.jsonl', '.stderr.log'):
            shutil.copyfile(raw.with_suffix(suffix), kept/f'{question}{suffix}')
        told = dict(question=question, cases=names, recorder_seconds=round(engine, 1), kept_seconds=round(time.time() - began - engine, 1), free_gib_before=round(free, 1),
                    shared_positions_equal_the_own_answers=same)
        done.write_text(json.dumps(told) + '\n')
        shutil.rmtree(raw)                      # this question's temporary raw stream, now held losslessly in kept/
        for suffix in ('.stdout.jsonl', '.stderr.log'):
            raw.with_suffix(suffix).unlink()
        print(json.dumps(dict(told, n=n + 1, of=len(index['questions']))), flush=True)


def record(run, own_run=None):
    """own_run: taken as capture takes it, for the callers that pass it; nothing reads it since the paired same-history row left the record
    (November 15, 2026). A case whose token stream is another case's (same_token_stream_as) was not read again: it is listed with its own
    source and the case that holds its trace."""
    index = json.loads((run/'cases'/'index.json').read_text())
    out = run/'record'
    out.mkdir(exist_ok=True)
    told, kept = [], run/'kept'
    for each in index['cases']:
        if each.get('same_token_stream_as'):
            told.append(dict(each, not_read_again='its token stream is the named case\'s, token for token under the same prompt; that case holds its trace'))
            continue
        now = json.loads((kept/each['id']/'steps.json').read_text())
        steps = now['steps']
        supplied = next(c['teacher_prefix_tokens'] for c in json.loads((run/'cases'/f"{each['question']}.json").read_text()) if c['id'] == each['id'])
        last = each['the_texts_last_token_left_to_the_model']
        text = supplied + [last]                                                  # the text's tokens, as the index's sha256 holds them
        if hashlib.sha256(json.dumps(text).encode()).hexdigest() != each['tokens_sha256'] or [s['selected'] for s in steps[:len(supplied)]] != supplied or len(steps) != len(text):
            raise ValueError(f"{each['id']}: the tokens read are not the tokens supplied")
        differs = [s['step'] for s, token in zip(steps, text) if s['greedy_token'] != token]
        entry = dict(id=each['id'], question=each['question'], group=each['group'], source=each['source'], prompt_tokens=now['prompt_tokens'], steps=len(steps), forced_steps=each['forced_steps'],
                     steps_where_the_models_own_choice_is_not_the_text=differs, the_text_is_the_models_own_under_these_feeds=set(differs) <= set(each['forced_steps']),
                     the_model_wrote_the_texts_last_token_itself=steps[-1]['selected'] == last, the_text_ended_by_itself=each['the_text_ended_by_itself'],
                     at_the_forced_steps=[dict(step=k, fed=steps[k]['piece'], fed_score=steps[k]['selected_logit'], own_choice=steps[k]['top'][0]['piece'], own_choice_score=steps[k]['top'][0]['logit'],
                                               best=[dict(id=t['id'], piece=t['piece'], logit=t['logit']) for t in steps[k]['top']]) for k in each['forced_steps']],
                     kept=now['kept'], raw=now['raw'])
        if each['source_pass']:
            made = json.loads((ROOT/each['source_pass']).read_text())['steps']
            entry['steps_with_the_source_pass_scores_to_the_bit'] = sum(a['logits_sha256'] == b['logits_sha256'] for a, b in zip(made, steps))
            entry['greatest_change_of_a_best_tokens_score_from_the_source_pass'] = max((abs(a['top'][0]['logit'] - b['top'][0]['logit']) for a, b in zip(made, steps) if a['top'][0]['id'] == b['top'][0]['id']), default=None)
        if each['first_forced_step'] is not None:
            entry['positions_through_the_first_forced_choice'] = each['positions_through_the_first_forced_choice']
            entry['those_positions_hold_the_own_answers_states_bit_for_bit'] = now['kept']['those_positions_hold_the_own_answers_states_bit_for_bit']
        told.append(entry)
        print(json.dumps({k: entry.get(k) for k in ('id', 'steps', 'forced_steps', 'steps_with_the_source_pass_scores_to_the_bit', 'those_positions_hold_the_own_answers_states_bit_for_bit',
                                                     'the_text_is_the_models_own_under_these_feeds')} | dict(own_choice_differs_at=differs)), flush=True)
    questions = [json.loads((kept/f'{question}.done').read_text()) | dict(execution_sha256=digest(kept/f'{question}.execution.json')) for question in index['questions']]
    shutil.copyfile(kept/f"{index['questions'][0]}.execution.json", out/'execution-of-the-first-question.json')
    (out/'replay.json').write_text(json.dumps(dict(what_this_is='per case of the replay: the tokens read, the forced steps and their scores, the reproduction of the source pass, and whether the state before the '
                                                                 'first forced choice is the own answer\'s; per question, the recorder call that read it',
                                                    merges=json.loads((kept/next(e['id'] for e in index['cases'] if not e.get('same_token_stream_as'))/'steps.json').read_text())['merges'], questions=questions, cases=told), indent=1) + '\n')


if __name__ == '__main__':
    command, run = sys.argv[1], ROUND_TWO_RUNS/sys.argv[2]
    if command == 'prepare':
        prepare(run)
    elif command == 'capture':
        capture(run, Path(sys.argv[3]).resolve())
    elif command == 'record':
        record(run)
    else:
        sys.exit(__doc__)
