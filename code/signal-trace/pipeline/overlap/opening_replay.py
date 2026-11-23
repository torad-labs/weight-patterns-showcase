"""opening-transplants-replay-original: the whole trace of the answers the original model wrote under another answer's opening.
The follow-up to fed-continuations-replay-original that Astra asked for on November 12, 2026
(records/retrospectives/2026-11-12-fork-request-decision-astra.md, "Capture priority includes the opening transplants": "Add the existing
opening transplants as a separately declared follow-up capture, using their recorded text and source provenance, including failed and
reverse/correct-case controls. Reuse compatible baselines and deduplicate identical input streams where verified, while preserving the
distinct intervention metadata.").

From code/signal-trace:
  python -m pipeline.overlap.opening_replay prepare|capture|record <run> [<folder for the recorder's raw stream>]
      (studies/04-round-2-overlap/runs/<run>/run.sh runs them)

prepare  The cases, from texts that exist: every pass of the prefix transplant (its kept record, records/development/2026-11-12-path-forward/
         label-swap/transplant/prefix-transplant.json; the program that made it and judged its answers was removed on November 14, 2026; 27 passes: a right answer's opening under a wrong fitting question, and the reverse on correct fitting
         answers), the ones that changed nothing included. Per pass, from its own record: the opening's length, the supplied tokens that
         were not the model's own first choice there (the forced steps; the rest of the opening the model would have written itself), and
         the first token that differs from the question's own answer. Two passes with one token stream under one prompt are one case: the
         first is read and the other is listed with its own donor as the same stream. A pass whose stream is the own answer's is listed
         as that and not read. Kept in cases/: one recorder list per question and index.json. As in fed_replay.py every token of a text
         but its last is supplied and the model writes the last one itself.
capture  fed_replay.capture with the own answers of fed-continuations-replay-original: the same recorder call, the same lossless kept/
         arrays and the same bit-for-bit checks. A case's positions through the row that writes its first token differing from the own
         answer are compared with that run's kept own answer, bit for bit, and kept once there where they are equal.
record   fed_replay.record: per case the tokens read, the forced steps and their scores, the reproduction of the source pass, whether the
         state before the first difference is the own answer's, and the paired same-history row.

Data roles, as in fed_replay.py: diagnostic and reference-construction observations of the unchanged original model on fitting answers.
Nothing joins the frozen reference, the cohort, the fitting folds or a score; an answer under a supplied opening is no model's own result;
no weight is solved and no target, mask or threshold is made. A donor is an answer on record whose opening's tokens are read from the
transplant's own case list; the answers no fit uses are not read and reserved-40b is not opened."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

from pipeline.native_records.run_declaration import digest
from pipeline.overlap import fed_replay as base
from pipeline.paths import ROOT, ROUND_TWO_RUNS

FIRST = 'fed-continuations-replay-original'       # the run whose kept/ holds the thirty own answers
MOVED = base.FEEDS/'transplant'
sha = lambda tokens: hashlib.sha256(json.dumps(tokens).encode()).hexdigest()


def prepare(run):
    passes = json.loads((MOVED/'prefix-transplant.json').read_text())
    listed = {case['id']: case for case in json.loads((MOVED/'plain.cases.json').read_text())}
    first = json.loads((ROUND_TWO_RUNS/FIRST/'cases'/'index.json').read_text())
    own_entry = {each['question']: each for each in first['cases'] if each['id'].endswith('.own')}
    index, batches, read = [], {}, {}
    for t in passes['transplant'].values():
        question, donor = t['answer'], t['donor']
        source = MOVED/'plain'/f"{question}.from-{donor}-{t['step']:04d}"
        fed_case, made = listed[source.name], json.loads((source/'result.json').read_text())
        own_case = next(c for c in json.loads((ROUND_TWO_RUNS/FIRST/'cases'/f'{question}.json').read_text()) if c['id'] == f'{question}.own')
        own = own_case['teacher_prefix_tokens'] + [own_entry[question]['the_texts_last_token_left_to_the_model']]
        prompt, opening, tokens = own_case['tokens'], fed_case['teacher_prefix_tokens'], [s['selected'] for s in made['steps']]
        if sha(own) != own_entry[question]['tokens_sha256'] or fed_case['tokens'] != prompt or made['prompt_tokens'] != len(prompt) or tokens[:len(opening)] != opening or len(opening) != t['step'] \
                or any(s['selection'] != ('teacher' if k < len(opening) else 'greedy') for k, s in enumerate(made['steps'])):
            raise ValueError(f'{source.name}: the source pass does not hold this opening under this prompt')
        forced = [k for k, s in enumerate(made['steps'][:len(opening)]) if s['greedy_token'] != s['selected']]      # supplied against the model's own first choice
        differs = next((k for k in range(max(len(own), len(tokens))) if k >= len(own) or k >= len(tokens) or own[k] != tokens[k]), None)
        if (differs is None) != (not forced) or (forced and differs != forced[0]):
            raise ValueError(f'{source.name}: the first token differing from the own answer ({differs}) is not the first forced step ({forced[:1]})')
        name = f'{question}.opening-of-{donor}'
        shared = len(prompt) - 1 + differs + 1 if differs is not None else None       # the positions through the row that writes the first differing token
        same_as = f'{FIRST}/{question}.own' if differs is None else read.get((question, sha(tokens)))
        entry = dict(id=name, question=question, group=t['group'], donor=donor, source=f"the opening of {donor}: its first {len(opening)} tokens supplied", prompt_tokens=len(prompt), answer_tokens=len(tokens),
                     supplied_tokens=len(tokens) - 1, supplied_opening_tokens=len(opening), the_texts_last_token_left_to_the_model=tokens[-1], the_text_ended_by_itself=bool(made['ended']), tokens_sha256=sha(tokens),
                     positions=len(prompt) - 1 + len(tokens), forced_steps=forced, first_forced_step=forced[0] if forced else None, first_token_differing_from_the_own_answer=differs,
                     positions_through_the_first_forced_choice=shared or 0, same_token_stream_as=same_as,   # what the pass then wrote and how it was judged are not carried into a case (no instrument reads an answer since November 14, 2026)
                     source_pass=str((source/'result.json').relative_to(ROOT)), source_pass_sha256=digest(source/'result.json'))
        index.append(entry)
        if same_as is None:
            read[question, sha(tokens)] = name
            batches.setdefault(question, []).append(dict(id=name, tokens=prompt, teacher_prefix_tokens=tokens[:-1], max_tokens=len(tokens)))
    folder = run/'cases'
    folder.mkdir(exist_ok=True)
    taken = [each for each in index if each['same_token_stream_as'] is None]
    total, kept, steps = (sum(each['positions'] for each in taken), sum(each['positions'] - each['positions_through_the_first_forced_choice'] for each in taken), sum(each['answer_tokens'] for each in taken))
    largest = max(sum(len(c['tokens']) - 1 + c['max_tokens'] for c in batch) for batch in batches.values())
    written = dict(what='the cases of the replay: every pass of the prefix transplant; every token of a text but its last is supplied and the model writes the last token itself; one token stream is read once',
                   built_from={str(p.relative_to(ROOT)): digest(p) for p in (MOVED/'prefix-transplant.json', MOVED/'plain.cases.json', ROUND_TWO_RUNS/FIRST/'cases'/'index.json')},
                   own_answers_kept_by=FIRST, model_sha256=passes['model_sha256'], questions=list(batches), cases=index, passes=len(index), read=len(taken), positions=total,
                   positions_kept_if_every_shared_prefix_is_equal=kept, kept_gib=round(((kept * (base.KEPT_ROWS - 1) + steps) * base.ROW) / 2**30, 2),
                   kept_gib_if_no_shared_prefix_is_equal=round(((total * (base.KEPT_ROWS - 1) + steps) * base.ROW) / 2**30, 2), largest_raw_question_gib=round(largest * base.RAW_ROWS * base.ROW / 2**30, 2))
    for question, batch in batches.items():
        wanted, file = json.dumps(batch), folder/f'{question}.json'
        if file.exists() and file.read_text() != wanted:
            raise ValueError(f'{file.name}: an earlier list differs from this one')
        file.write_text(wanted)
    wanted = json.dumps(written, indent=1) + '\n'
    if (folder/'index.json').exists() and (folder/'index.json').read_text() != wanted:
        raise ValueError('index.json: an earlier index differs from this one')
    (folder/'index.json').write_text(wanted)
    free = shutil.disk_usage(run).free / 2**30
    print(json.dumps(dict(passes=len(index), read=len(taken), the_own_answers_stream=[e['id'] for e in index if e['same_token_stream_as'] and e['same_token_stream_as'].endswith('.own')],
                          another_passes_stream={e['id']: e['same_token_stream_as'] for e in index if e['same_token_stream_as'] and not e['same_token_stream_as'].endswith('.own')},
                          questions=len(batches), positions=total, positions_kept=kept, kept_gib=written['kept_gib'], kept_gib_at_most=written['kept_gib_if_no_shared_prefix_is_equal'],
                          largest_raw_question_gib=written['largest_raw_question_gib'], free_gib=round(free, 1), by_group={g: sum(e['group'] == g for e in index) for g in ('wrong', 'correct')},
                          forced_steps_per_pass=sorted(len(e['forced_steps']) for e in index))))


if __name__ == '__main__':
    command, run = sys.argv[1], ROUND_TWO_RUNS/sys.argv[2]
    if command == 'prepare':
        prepare(run)
    elif command == 'capture':
        base.capture(run, Path(sys.argv[3]).resolve(), FIRST)
    elif command == 'record':
        base.record(run, FIRST)
    else:
        sys.exit(__doc__)
