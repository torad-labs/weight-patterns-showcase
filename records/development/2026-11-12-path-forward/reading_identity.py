"""Development: a native reading taken by what it computed, not by the folder it was made in.

Three things are kept apart (Astra, records/retrospectives/2026-11-14-native-reuse-astra.md:13-23 and 39-49):

  the physical reading   what the engine computed and the bytes kept of it. It is named by everything that went into the computation: the
                         model file's sha256, the recorder program, the engine's libraries, the card, how the text was read, which outputs
                         were kept, and for each answer the prompt's tokens, the text's tokens and how the text ends. It stays where it was
                         first made, with the record it was made with. Nothing here writes there or changes what it says of itself.
  the run binding        a later run that needs the same computation writes, inside its own folder, a binding: what it asked for, the
                         reading that holds it (its place, its own record and that record's sha256), and answer by answer the files and
                         their hashes. An answer no reading holds is a miss with its reason, and is read the ordinary way. A binding says
                         of itself that nothing was computed for it.
  the score              is no part of either. Targets, requests, registration and metric are the scorer's own inputs: a binding carries
                         bytes, never a score.

  python reading_identity.py describe <reading folder> [<a recorder pass's results.json, for the engine's end tokens>]
  python reading_identity.py find <asked.json> <root> [<root> ...]
  python reading_identity.py bind <asked.json> <binding.json to write> <root> [<root> ...]
      from code/signal-trace. A reading folder is one whole_reading.py or dose_reading.py made (own.json and own/<answer>/answer.json: the
      turns of every merge at the answer steps), or one states_reading.py made (reading.json, row-map.json, nodes/: the engine's own state
      at every merge at every answer step and question position). The two keep different things of the same pass and are never taken for
      each other: `asked(..., kept='turns' | 'states')`. A root is searched for such folders; asked.json is what `asked` returns.

What an identity holds, and where each part comes from:

  model_sha256, recorder_sha256, card          the reading's own.json.
  libraries                                    the engine's shared libraries the recorder loads (ENGINE: their names and sha256, hashed
                                               together). A reading with no recorded library hash stays unresolved: current file timestamps
                                               do not establish historical bytes (a restore or copy can preserve mtime). A prior source-bound
                                               receipt or a separately bound exact-equivalence record is needed, never an invented historical
                                               hash. This lookup does not yet resolve such additional evidence.
  reading                                      how the text was read and what was kept: the recorder's mode, scope and retention, the
                                               fusion switch, the answer's tokens supplied one at a time, every merge at every answer step.
  per answer: prompt_sha256, text_sha256       the prompt's tokens (the cohort's) and the text's tokens (answer.json), each hashed as JSON.
  per answer: own, ending                      whether the text is the answer this model writes by itself, whole (whole_reading.ending): a
                                               replay whose every token is the model's choice proves the text up to its last token, and its
                                               end only with an end token or the generation's limit. `find` gives a replay to one who asks
                                               for the model's own answer only where this is True.

Nothing here opens an array, runs a model or reads an answer no fit uses: it reads records.
"""
import hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT/'dependencies'/'llama-cpp-build'
READING = dict(recorder_arguments=['capture', 'residual', 'stream'], fusion_disabled='1', text='the answer\'s tokens supplied one at a time after its prompt',
               kept='every merge at every answer step (turns, norms, start), as export_case and ladder.first_direction keep them')
# The same pass with another thing kept (states_reading.py): one who asks for the one is never given the other.
STATES = dict(READING, kept='the engine\'s own float32 state after every merge at every answer step and every question position, and the last layer\'s output and '
                            'its output norm at the answer steps (states_reading.py, whole-gap-reading/1)')
KEPT = dict(turns=READING, states=STATES)
STATES_FORMAT = 'whole-gap-reading/1'


def ending(first_step_not_its_own, ended_with_an_end_token, steps, limit):   # whole_reading.py's until November 15, 2026, the same bytes: here a reading kept from a generation takes it without importing the replay
    """Is a given text the answer the model writes by itself, whole? (True, False or None, and why.) Every token its own choice says so of the
    text up to its last token; of its end only when that last token is an end token (then chosen too) or the text stops at the generation's
    limit. Otherwise what the model does after the last given token was never computed, and the answer is read again by generating it
    (Astra, records/retrospectives/2026-11-14-native-reuse-astra.md:51-57)."""
    if first_step_not_its_own is not None:
        return False, 'a token of the text is not this model\'s choice'
    if ended_with_an_end_token:
        return True, 'an end token, this model\'s own choice'
    if limit is not None and steps == limit:
        return True, 'the generation\'s limit'
    return None, 'unresolved: the text stops before an end token and under the limit'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def of_tokens(tokens):
    return hashlib.sha256(json.dumps([int(t) for t in tokens], separators=(',', ':')).encode()).hexdigest()


def libraries(folder=ENGINE):
    """The engine's shared libraries as they are now: one sha256 over each file's name and sha256, and the latest time one was written."""
    files = sorted(p for p in Path(folder).iterdir() if p.is_file() and not p.is_symlink() and '.so' in p.name)
    if not files:
        raise ValueError(f'no library in {folder}')
    listed = ''.join(f'{p.name} {digest(p)}\n' for p in files)
    return dict(sha256=hashlib.sha256(listed.encode()).hexdigest(), files=len(files), last_written=max(p.stat().st_mtime for p in files))


def began(text):
    """A record's `started` ('2026-11-13 05:08:51 CST') as seconds, in this machine's zone, which is the records'."""
    return time.mktime(time.strptime(' '.join(text.split()[:2]), '%Y-%m-%d %H:%M:%S'))


def end_tokens(results):
    """The tokens the engine itself calls an end of generation, from a recorder pass's results.json: the last token of every answer the pass
    says ended. An answer it says did not end must not close on one of them."""
    told = json.loads(Path(results).read_text())
    found = {int(one['tokens'][-1]) for one in told if one['ended']}
    if any(int(one['tokens'][-1]) in found for one in told if not one['ended']):
        raise ValueError(f'{results}: an answer that did not end closes on a token another answer ended with')
    return found


def describe(folder, prompts, now=None, endings=None):
    """The identities a reading on disk holds, from its own records. `prompts`: each answer's prompt tokens, by id (the cohort's).
    Returns the reading's shared part and, per answer, what is its own; raises where the folder is no reading.
    A reading made before whole_reading.py wrote each answer's ending has none in its record. `endings` then lets it be read from the text
    itself: dict(end_tokens=<the engine's end tokens, `end_tokens`>, limits={answer: the generation's limit}, source=<where they come from>)."""
    folder = Path(folder)
    if (folder/'reading.json').exists():
        return described_states(folder)
    record = json.loads((folder/'own.json').read_text())
    for key in ('model_sha256', 'recorder_sha256', 'card', 'answers_read'):
        if not record.get(key):
            raise ValueError(f'{folder}: its own.json names no {key}')
    shared = dict(model_sha256=record['model_sha256'], recorder_sha256=record['recorder_sha256'], card=record['card'], reading=READING)
    if record.get('libraries_sha256'):
        shared.update(libraries_sha256=record['libraries_sha256'], libraries_from='the reading\'s own record')
    else:
        shared.update(libraries_sha256=None,
                      libraries_from='not recorded; current file timestamps do not establish historical library bytes')
    if 'first_merge' in record or 'from_the_matrix_merge_on' in record:
        raise ValueError(f'{folder}: a reading from a merge on is not a whole reading; it is not described here')
    table = {row['id']: row for row in record.get('table', [])}
    answers = {}
    for name in record['answers_read']:
        path = folder/'own'/name/'answer.json'
        if not path.exists():
            answers[name] = dict(missing='its answer.json is not on this disk')
            continue
        mine, row = json.loads(path.read_text()), table.get(name, {})
        if name not in prompts:
            raise ValueError(f'{name}: no prompt is given for it')
        own, why = row.get('the_whole_answer_is_this_models_own'), row.get('its_ending', 'not recorded')
        if 'the_whole_answer_is_this_models_own' not in row and endings and 'first_step_not_its_own' in mine:
            own, why = ending(mine['first_step_not_its_own'], int(mine['tokens'][-1]) in endings['end_tokens'], mine['steps'], endings['limits'].get(name))
            why += f" (read from the text: the engine's end tokens from {endings['source']})"
        answers[name] = dict(prompt_sha256=of_tokens(prompts[name]), text_sha256=of_tokens(mine['tokens']), steps=mine['steps'], own=own, ending=why,
                             first_step_not_its_own=mine.get('first_step_not_its_own'), answer_json_sha256=digest(path))
    return dict(folder=str(folder.relative_to(ROOT)) if ROOT in folder.parents else str(folder), record_sha256=digest(folder/'own.json'), started=record.get('started'),
                made_by=record.get('made_by', 'dose_reading.py'), text_given_from=record.get('text_given_from'), shared=shared, answers=answers)


def described_states(folder):
    """The identities a states reading holds (states_reading.py: reading.json, row-map.json, answers/<answer>/answer.json), from its own
    records. Every answer's rows lie in the same arrays, so beside each answer's identity it says where its rows are (the row map) and,
    once for the reading, every array with its sha256. The prompt's hash is the reading's own: what it fed, hashed when it fed it."""
    record = json.loads((folder/'reading.json').read_text())
    if record.get('format') != STATES_FORMAT:
        raise ValueError(f"{folder}: its reading.json is of format {record.get('format')!r}, not {STATES_FORMAT}")
    for key in ('model_sha256', 'recorder_sha256', 'libraries_sha256', 'card', 'answers_read'):
        if not record.get(key):
            raise ValueError(f'{folder}: its reading.json names no {key}')
    if record.get('recorder_arguments') != STATES['recorder_arguments'] or record.get('fusion_disabled') != STATES['fusion_disabled']:
        raise ValueError(f'{folder}: read with other recorder arguments or another fusion switch than a states reading is')
    if digest(folder/'row-map.json') != record['row_map']['sha256']:
        raise ValueError(f'{folder}: its row map is not the one its reading.json names')
    shared = dict(model_sha256=record['model_sha256'], recorder_sha256=record['recorder_sha256'], card=record['card'], reading=STATES,
                  libraries_sha256=record['libraries_sha256'], libraries_from='the reading\'s own record')
    table, rows = {row['id']: row for row in record['table']}, {row['id']: row for row in json.loads((folder/'row-map.json').read_text())['answers']}
    answers = {}
    for name in record['answers_read']:
        path = folder/'answers'/name/'answer.json'
        if not path.exists():
            answers[name] = dict(missing='its answer.json is not on this disk')
            continue
        mine, row, lies = json.loads(path.read_text()), table[name], rows[name]
        if (mine['prompt_sha256'], mine['text_sha256'], mine['steps']) != (lies['prompt_sha256'], lies['text_sha256'], lies['steps']) or of_tokens(mine['tokens']) != mine['text_sha256']:
            raise ValueError(f'{folder}: {name} is not the same answer in its answer.json and in the row map')
        answers[name] = dict(prompt_sha256=mine['prompt_sha256'], text_sha256=mine['text_sha256'], steps=mine['steps'], own=row['the_whole_answer_is_this_models_own'], ending=row['its_ending'],
                             first_step_not_its_own=mine['first_step_not_its_own'], answer_json_sha256=digest(path),
                             rows={key: lies[key] for key in ('answer_rows_from', 'question_rows_from', 'steps', 'question')})
    arrays = {m['file']: m['sha256'] for m in record['merges']} | {record['terminal'][key]['file']: record['terminal'][key]['sha256'] for key in ('l_out', 'result_norm')}
    return dict(folder=str(folder.relative_to(ROOT)) if ROOT in folder.parents else str(folder), record_sha256=digest(folder/'reading.json'), started=record.get('started'),
                made_by=record['made_by'], text_given_from=record.get('text_given_from'), shared=shared, answers=answers,
                arrays=arrays, row_map_sha256=record['row_map']['sha256'], rows=record['rows'])


def asked(run, model_sha256, recorder_sha256, libraries_sha256, card, prompts, texts, own, kept='turns'):
    """What a run asks for: one model through one recorder and engine on one card, on each answer's prompt and text, with `kept` what the
    reading keeps of the pass ('turns': READING, whole_reading.py's; 'states': STATES, states_reading.py's).
    `run` is the cohort run the prompts are of. `own` True asks for the answer the model writes by itself (a replay serves only where its
    whole text is proved the model's own); False asks for the reading of that text whoever wrote it."""
    return dict(run=run, shared=dict(model_sha256=model_sha256, recorder_sha256=recorder_sha256, card=card, reading=KEPT[kept], libraries_sha256=libraries_sha256),
                own=bool(own), answers={name: dict(prompt_sha256=of_tokens(prompts[name]), text_sha256=of_tokens(texts[name])) for name in texts})


def holds(described, wanted):
    """Answer by answer, whether a described reading holds what is asked: (the answers it holds, {answer: why not})."""
    mine, theirs = described['shared'], wanted['shared']
    apart = [key for key in ('model_sha256', 'recorder_sha256', 'card', 'reading', 'libraries_sha256') if mine.get(key) != theirs.get(key) or mine.get(key) is None]
    if apart:
        return [], {name: 'another ' + ', '.join(apart) for name in wanted['answers']}
    yes, no = [], {}
    for name, each in wanted['answers'].items():
        have = described['answers'].get(name)
        if have is None or 'missing' in have:
            no[name] = 'the reading does not hold this answer' if have is None else have['missing']
        elif have['prompt_sha256'] != each['prompt_sha256']:
            no[name] = 'another prompt'
        elif have['text_sha256'] != each['text_sha256']:
            no[name] = 'another text'
        elif wanted['own'] and have['own'] is not True:
            no[name] = 'the text is not proved this model\'s own whole answer: ' + str(have['ending'])
        else:
            yes.append(name)
    return yes, no


def readings_under(roots):
    for root in roots:
        for folder, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ('raw', '__pycache__') and not d.startswith('reserved')]
            if ('own.json' in files and 'own' in dirs) or ('reading.json' in files and 'nodes' in dirs):
                yield Path(folder)


def find(wanted, roots, prompts, now=None, endings=None):
    """For each answer asked, the first reading under the roots that holds it; and, for the answers none holds, every reason met."""
    found, reasons = {}, {name: [] for name in wanted['answers']}
    for folder in readings_under(roots):
        try:
            described = describe(folder, prompts, now, endings)
        except (ValueError, KeyError, OSError) as error:
            for name in reasons:
                reasons[name].append(f'{folder}: {error}')
            continue
        yes, no = holds(described, wanted)
        for name in yes:
            found.setdefault(name, described)
        for name, why in no.items():
            reasons[name].append(f"{described['folder']}: {why}")
    return found, {name: why for name, why in reasons.items() if name not in found}


def bind(wanted, roots, prompts, out, inventory_of=None, now=None, endings=None):
    """Write a run's binding: what it asked, the reading each answer is taken from with that reading's own record and files, and the misses.
    `out` lies inside the asking run. `inventory_of(reading folder)` gives an archive inventory's {relative path: sha256} for a reading whose
    arrays are in the bucket under its path (None where there is none); the files on disk are listed by name and size either way."""
    out = Path(out).resolve()
    found, misses = find(wanted, roots, prompts, now, endings)
    for described in {d['folder']: d for d in found.values()}.values():
        if (ROOT/described['folder']) in out.parents:
            raise ValueError('a binding is written in the asking run, never inside the reading it binds')
    answers, arrays = {}, {}
    for name, described in found.items():
        told = dict(reading=described['folder'], reading_record_sha256=described['record_sha256'], reading_started=described['started'], reading_made_by=described['made_by'],
                    libraries_from=described['shared']['libraries_from'], **described['answers'][name])
        if 'arrays' in described:   # a states reading: every answer's rows lie in the reading's own arrays, at the rows its answer names
            here = ROOT/described['folder']
            arrays[described['folder']] = dict(row_map_sha256=described['row_map_sha256'], rows=described['rows'], files_sha256=described['arrays'],
                                               files_here={file: (here/file).stat().st_size for file in described['arrays'] if (here/file).exists()})
            answers[name] = told
            continue
        where = ROOT/described['folder']/'own'/name
        listed = (inventory_of(ROOT/described['folder']) or {}) if inventory_of else {}
        answers[name] = dict(told, files_here={p.name: p.stat().st_size for p in sorted(where.iterdir()) if p.is_file()},
                             files_sha256={Path(key).name: value for key, value in listed.items() if Path(key).parent.name == name})
    binding = dict(what='a binding of readings made earlier: nothing was computed for it. Each answer names the reading that holds the computation asked for, where that reading '
                        'was first made, and its own record; the reading is not copied, moved or renamed. A miss is read the ordinary way. No score is bound: a score is made '
                        'from these bytes with the scorer\'s own targets, requests and registration.',
                   asked=wanted, bound=answers, misses=misses, computed_here=False, made_by='reading_identity.py', made=time.strftime('%Y-%m-%d %H:%M:%S %Z'),
                   **(dict(arrays=arrays) if arrays else {}))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(binding, indent=1))
    return binding


if __name__ == '__main__':
    from pipeline.paths import ROUND_TWO_RUNS
    from pipeline.weight_edit import ladder

    def prompts_of(run):
        _, members, _ = ladder.cohort(ROUND_TWO_RUNS/run)
        return {m['id']: m['tokens'] for m in members}

    if sys.argv[1] == 'describe':
        from pipeline.weight_edit import tube
        folder = Path(sys.argv[2]).resolve()
        endings = None
        if len(sys.argv) > 3:
            endings = dict(end_tokens=end_tokens(sys.argv[3]), limits={m['id']: m['max_tokens'] for m in json.loads((tube.WIDE/'members.json').read_text())}, source=sys.argv[3])
        told = describe(folder, {}, endings=endings) if (folder/'reading.json').exists() else describe(folder, prompts_of(json.loads((folder/'own.json').read_text())['run']), endings=endings)
        print(json.dumps(told | dict(answers=len(told['answers']), own={str(k): sum(a.get('own') is k for a in told['answers'].values()) for k in (True, False, None)}), indent=1))
    else:
        wanted = json.loads(Path(sys.argv[2]).read_text())
        roots = sys.argv[3:] if sys.argv[1] == 'find' else sys.argv[4:]
        prompts = prompts_of(wanted['run'])
        if sys.argv[1] == 'find':
            found, misses = find(wanted, roots, prompts)
            print(json.dumps(dict(found={name: d['folder'] for name, d in found.items()}, misses=misses), indent=1))
        else:
            binding = bind(wanted, roots, prompts, sys.argv[3])
            print(json.dumps(dict(bound=len(binding['bound']), misses=len(binding['misses']), written=sys.argv[3])))
