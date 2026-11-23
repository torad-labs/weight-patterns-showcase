"""A sweep made as a run's command: one working file, one record a matrix, and a sweep that was stopped taken up where it stood.

gap_sweep.sweep makes a sweep in one call. Seventy matrices are an hour or two of passes and solves, so a run needs three things around
it, and nothing more is here. Under the folder a run gives its sweep:

  sweep.json                written before the first matrix: what this sweep is. The original by its sha256, the target artifact by its
                            target.json's sha256, the matrices in their order, the fitting answers and their groups, the rule by its
                            name. A sweep taken up under anything else is refused: it would be another sweep in this one's file
  working.gguf              the one file the sweep writes into (gap_state.Working), with its manifest beside it
  matrices/<n>-<name>.json  one record a matrix, written when the matrix is done: the sweep's own entry (the state before and after, what
                            the rows cover, the search, every ridge's reading on the groups left out, the rule's ridge, what was
                            written), its seconds, and the solver's files as this process read them
  swept.json                written after the last matrix: each matrix's ridge and states in order, and the last state bound to its
                            whole file's sha256 (gap_state.Working.kept)

Taking up. A matrix's record is written after the matrix, so a stop leaves the file at the last record's state or one write past it. One
write past it, the matrix that comes next in the order is given the bytes it held when the sweep began and is solved again; so is a
matrix whose write was cut (gap_state.Working.mended). A sweep visits each matrix once, so those are the bytes the matrix held before
its write. The file is then held to the last record's state, and a file in any other state is refused. No record is written twice.

A sweep that goes on from a state another sweep reached. With `start` (that state's file and the record that binds it:
gap_state.Working.kept) the working file is a copy of that state and not of the original, the first record starts from that state's
name, and a stop is undone to that state's bytes. With a target laid on the model's own answers (gap_requests.Laid) the rows a matrix
is solved on are the rows of answers the model itself wrote, each row's request the reference's laid on it by the capture's pairs,
and each record keeps what its requests carry beside the rows. The laying must be of a capture of the very model the sweep starts
from. Both are part of what the sweep is (sweep.json): taken up under another start, other own answers or other pairs, it is refused.

`main` is the command of a declared run, and it makes one pass of a correction that is repeated: pass 1 starts from the state the
declaration names (the original, or a state bound by its record), and each later pass from the state the pass before it swept to.
Under the run, for pass n:

  loop/<n>/capture/             the capture of the state the pass starts from, on that state's own answers (the readers' own:
                                own/text.json, own/<id>/answer.json, own/<id>/pairs-original.npy)
  loop/<n>/correspondence.json  the record of that capture's pairs against the target (`whole-gap-correspondence/2`), the one the
                                state is measured through
  loop/<n>/sweep/               this pass's sweep, as above
  loop/<n>/late-start/          where the pass's passes are started late, the comparisons that let them (gap_late): the pass's own,
                                since each pass reads another text from another state

What the declaration names under `sweep` is all it takes beside those, and each matrix's rows come from one pass of the working file
on the capture's own answers (gap_rows.native). The solves hold the lock of the card they are on (correction.card_taken); a pass is
made on card 0 under that card's lock, which the recorder takes by itself while this process gives it up (tube.CARD). Where the
declaration names a late start, a pass may be started from what the pass before it recorded (gap_late): each record says which
recorder read its rows and from which seeds, and a sweep taken up begins with an ordinary pass. Nothing here chooses a matrix, an
order, a target, a group, a ridge, a start or how many passes are made, and nothing here measures a state: a pass ends in a file,
and what that file leaves of the target is read by the readers on its own answers
(records/development/2026-11-14-whole-gap-solver/README.md)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pipeline.weight_edit import field, gap_requests, gap_rows, gap_sets, gap_state, gap_sweep, gap_targets, set_points, tube

KIND = 'whole-gap sweep/1'
RULES = {'least held': gap_sweep.least_held}   # a declaration names its rule by one of these
ARITHMETICS = ('double', 'single')   # how a sweep's sums and decompositions are made (field.Matrix.in_single): a declaration that names none asks the first
# the solver's files as this process read them: a record names what ran, not what lies there when the sweep is done
AS_READ = dict({f'{each.__name__.rsplit(".", 1)[1]}_sha256': tube.digest(each.__file__) for each in (field, gap_state, gap_targets, gap_requests, gap_rows, gap_sets, gap_sweep, set_points, tube)},
               gap_run_sha256=tube.digest(__file__), gap_late_sha256=tube.digest(Path(__file__).with_name('gap_late.py')), torch=torch.__version__)


def path_order(tensors):
    """Every layer's attention output and then its feed-forward, from the first layer to the last: the order the model computes in."""
    parts = [held[1] for held in tube.KINDS.values()]
    layers = sorted(int(name.split('.')[1]) for name in tensors if name.startswith('blk.') and name.endswith(f'.{parts[0]}.weight'))
    return [f'blk.{layer}.{part}.weight' for layer in layers for part in parts]


def block_held(bindings):
    """The solver's block of rows (field.BLOCK) held to the matrices a sweep solves: it is the narrowest input among them, so that at no
    matrix a block of rows (block x input width) is larger than the sums it is added into (input width x input width). The block
    orders every sum's additions, so a sweep whose matrices give another number is refused and writes nothing."""
    narrowest = min(int(held['shape'][0]) for held in bindings.values())
    if field.BLOCK != narrowest:
        raise gap_targets.Refused(f'the solver sums {field.BLOCK} rows at once and the narrowest input among this sweep\'s matrices is {narrowest} wide')
    return narrowest


def reference(folder, root):
    """What a declared sweep is solved against, as a sweep takes it. The folder says which by the manifest it holds: a target artifact
    (a point a row: gap_targets.read), or a reference that is a set at every row (success_sets.Reference, presented by
    gap_sets.Sets). Either way its `sha256` is its manifest's, which the declaration names and a capture's record of pairs binds."""
    folder = Path(folder)
    if (folder/'target.json').is_file():
        return gap_targets.read(folder, root)
    if (folder/'manifest.json').is_file():
        from pipeline.weight_edit import success_sets   # that module reads the comparison's whole side: loaded only for a reference of sets
        try:
            return gap_sets.Sets(success_sets.Reference(folder, root))
        except ValueError as why:
            raise gap_targets.Refused(f'{folder} is no reference of sets a sweep can take: {why}') from None
    raise gap_targets.Refused(f'{folder} holds neither a target artifact nor a reference of sets')


def described(original_sha256, targets, names, fitting, groups, rule, start=None, single=False):
    """What a sweep is, as its folder keeps it: nothing of how far it has come. A sweep from the original on the target's own rows is
    described as it always was; the state it starts from, the laying of its target and an arithmetic in single precision are named
    only where there is one."""
    what = dict(kind=KIND, original_sha256=original_sha256, target_sha256=targets.sha256, matrices=list(names), fitting=list(fitting),
                groups=None if groups is None else [sorted(int(each) for each in group) for group in groups], rule=rule)
    if single:
        what['arithmetic'] = 'single'
    if start is not None:
        what['start'] = dict(identity=start['kept']['identity'], model_sha256=start['kept']['model_sha256'])
    if isinstance(targets, gap_requests.Laid):
        what['laid'] = targets.described()
    return what


def kept(path, value):
    """A record written whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path) + '.next', 'w') as out:
        out.write(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(str(path) + '.next', path)


def record_of(folder, position, name):
    return Path(folder)/'matrices'/f'{position:02d}-{name}.json'


def taken_up(folder, original, original_sha256, bindings, what, start=None):
    """The sweep's working file and the records of the matrices already swept, in their order. A folder that holds no sweep gets this
    one's description and its file; a folder that holds this sweep is held to its records, as the module says; a folder that holds
    another sweep, records that do not follow one another, or a file in a state no record leads to, is refused. `start` is the state
    the sweep begins from where that is not the original (its `file` and `kept`)."""
    folder = Path(folder)
    file, names = folder/'working.gguf', what['matrices']
    written = sorted(each.name for each in (folder/'matrices').glob('*.json')) if (folder/'matrices').exists() else []
    if not (folder/'sweep.json').exists():
        if written or file.exists() or gap_state.Working.beside(file).exists():
            raise gap_targets.Refused(f'{folder} holds records or a working file and no description of the sweep they belong to')
        kept(folder/'sweep.json', what)
    elif json.loads((folder/'sweep.json').read_text()) != json.loads(json.dumps(what)):
        raise gap_targets.Refused(f'{folder} holds another sweep: its original, target, matrices, answers, groups, rule, start, laying or arithmetic are not the ones asked for')
    done = []
    for position, name in enumerate(names):
        if not record_of(folder, position, name).exists():
            break
        done.append(json.loads(record_of(folder, position, name).read_text()))
    if written != [record_of(folder, position, name).name for position, name in enumerate(names[:len(done)])]:
        raise gap_targets.Refused(f'{folder}/matrices holds records that are not the first {len(done)} of this sweep\'s order')
    began = gap_state.identity(original_sha256, {}) if start is None else start['kept']['identity']
    state = began
    for record in done:
        if record['state_before'] != state:
            raise gap_targets.Refused(f"{record['matrix']}: its record does not start from the state the record before it ended in")
        state = record['state_after']
    if not gap_state.Working.beside(file).exists():
        if done:
            raise gap_targets.Refused(f'{folder} holds records and no working file')
        return gap_state.Working.make(file, original, original_sha256, bindings, start=start), done
    working = gap_state.Working.mended(file, original, bindings)
    if working.started() != began:
        raise gap_targets.Refused(f'{file.name} was made from another state than the one this sweep starts from')
    if working.identity() != state and len(done) < len(names) and working.moved(names[len(done)]):
        working.put(names[len(done)], working.begun(names[len(done)]))   # written, and stopped before its record: solved again
    if working.identity() != state:
        raise gap_targets.Refused(f'{file.name} holds a state no record of this sweep leads to')
    return working, done


def run(folder, original, original_sha256, bindings, targets, names, rows_of, fitting, groups, rule, until=None, say=None, start=None, single=False):
    """The sweep `rule` (a name of RULES) makes of `names` in that order, made or taken up under `folder`. `until` stops it after that
    many more matrices; the rest is gap_sweep.sweep's. Returns the working file and every matrix's record so far; when the last matrix
    is done the last state is bound to its file's sha256 in swept.json. `targets` is a target artifact, or one laid on a model's own
    answers (gap_requests.Laid), which has to be laid on a capture of the model the sweep starts from: `start`, or the original.
    With `single` every solve's sums and decompositions are in single precision, which is part of what the sweep is."""
    folder, names = Path(folder), list(names)
    laid = targets if isinstance(targets, gap_requests.Laid) else None
    if laid is not None and (laid.bound or {}).get('model_sha256') != (original_sha256 if start is None else start['kept']['model_sha256']):
        raise gap_targets.Refused('the requests are laid on a capture of another model than the one this sweep starts from')
    what = described(original_sha256, targets, names, fitting, groups, rule, start, single)
    working, done = taken_up(folder, original, original_sha256, bindings, what, start)
    ends = len(names) if until is None else min(len(names), len(done) + until)
    for position in range(len(done), ends):
        began = time.time()
        entry, = gap_sweep.sweep(working, targets, [names[position]], rows_of, fitting, groups, RULES[rule], single=single)
        record = dict(position=position, of=len(names), **entry, **({} if laid is None else {'laid': laid.carried_at(names[position])}),
                      seconds=round(time.time() - began, 1), made_with=AS_READ)
        kept(record_of(folder, position, names[position]), record)
        done.append(record)
        if say:
            say(record)
    if len(done) == len(names) and not (folder/'swept.json').exists():
        kept(folder/'swept.json', dict(sweep=what, matrices=[dict(matrix=each['matrix'], power=each['power'], state_after=each['state_after']) for each in done],
                                       last_state=working.kept()))
    return working, done


def of_pass(run_folder, number):
    """Where pass `number` of a run's correction lies under the run: its capture, the capture's correspondence record and its sweep."""
    if not isinstance(number, int) or number < 1:
        raise gap_targets.Refused(f'a pass is numbered from 1, and {number!r} was asked')
    at = Path(run_folder)/'loop'/str(number)
    return dict(capture=at/'capture', correspondence=at/'correspondence.json', sweep=at/'sweep')


def begins(run_folder, asked, number, root):
    """The state pass `number` starts from, as `run` takes a start: None for the original. Pass 1 starts where the declaration says
    (`start`: 'original', or a state's `file`, named from the repository's root, and the `kept` record that binds it); a later pass
    starts from the file the pass before it swept to its last matrix, bound by that sweep's swept.json."""
    if number == 1:
        given = asked.get('start')   # a declaration that names no start is refused below: a pass is never begun from a default
        if given == 'original':
            return None
        if not isinstance(given, dict) or sorted(given) != ['file', 'kept']:
            raise gap_targets.Refused("a declared start is 'original', or a state's file and the record that binds it")
        return dict(file=Path(root)/given['file'], kept=given['kept'])
    before = of_pass(run_folder, number - 1)['sweep']
    if not (before/'swept.json').exists():
        raise gap_targets.Refused(f'pass {number - 1} is not swept to its last matrix: pass {number} has no state to start from')
    return dict(file=before/'working.gguf', kept=json.loads((before/'swept.json').read_text())['last_state'])


def laid_for(targets, at, asked):
    """A pass's requests: the reference laid on the pass's capture by its record of pairs (gap_requests.Laid.read). Refused where that
    capture was not held to the reader protocol the declaration names (`capture_protocol_sha256`): a record of pairs says which
    protocol its capture and reading were held to, and whether that is the declared one is asked here, before a matrix is solved."""
    laid = gap_requests.Laid.read(targets, at['correspondence'], at['capture'])
    if asked.get('capture_protocol_sha256') is None or laid.bound['protocol_sha256'] != asked['capture_protocol_sha256']:
        raise gap_targets.Refused('the capture the requests are laid on was not held to the reader protocol the declaration names (capture_protocol_sha256)')
    return laid


def main(run_folder, number, until=None):
    """Pass `number` of a declared run's correction. Its declaration's `sweep` names: `target` (the folder of the artifact, or of the
    reference of sets, from the repository's root: `reference`) and `target_sha256`; `capture_protocol_sha256` (the reader protocol
    every pass's capture is held to: `laid_for`); `matrices` (a list, or 'path order'); `rule` (a name of RULES); `cohort` (the run folder the
    answers' questions are read from); `read_as` (the names the target's schema gives a merge's state and the model's last merge as the
    output norm passes it on); `start` (`begins`); and, for passes started late, `late_start` (the late recorder's `program` and its
    `sha256`: without it every pass is the ordinary one); and `arithmetic`, where the solves' sums and decompositions are not in double
    precision (a name of ARITHMETICS). The fitting answers are tube.picks' and their groups
    correction.case_folds', as in every solve; each answer's text is the capture's own (`of_pass`), and the cohort's recorded text
    enters nowhere."""
    from pipeline.paths import ROOT
    from pipeline.weight_edit import correction, gap_late, ladder
    run_folder = Path(run_folder).resolve()
    declaration = tube.declared(run_folder)
    asked = declaration['sweep']
    targets = reference(ROOT/asked['target'], ROOT)
    if targets.sha256 != asked['target_sha256']:
        raise gap_targets.Refused('the target artifact, or the reference of sets, is not the one the declaration names by its sha256')
    arithmetic = asked.get('arithmetic', ARITHMETICS[0])
    if arithmetic not in ARITHMETICS:
        raise gap_targets.Refused(f'a declared arithmetic is one of {ARITHMETICS}, and {arithmetic!r} was asked')
    at = of_pass(run_folder, number)
    start = begins(run_folder, asked, number, ROOT)
    laid = laid_for(targets, at, asked)
    _, members, _ = ladder.cohort(ROOT/asked['cohort'])
    fitting, groups = tube.picks(members)
    used = gap_rows.on_own([m for m, take in zip(members, fitting) if take], laid.tokens)
    folds = [np.flatnonzero(each).tolist() for each in correction.case_folds(used, [np.asarray(each[fitting]) for each in groups])]
    tensors = json.loads(ladder.TENSORS.read_text())
    names = path_order(tensors) if asked['matrices'] == 'path order' else list(asked['matrices'])
    block_held({name: tensors[name] for name in names})
    late = gap_late.declared(run_folder, asked['late_start'], used, declaration['model_sha256'], starts_from=None if start is None else start['kept']['model_sha256'],
                             kept_under=at['sweep'].parent/'late-start') if asked.get('late_start') else None
    rows_of = gap_rows.native(run_folder, members, fitting, tensors, asked['read_as']['merge'], asked['read_as']['last_merge_of_the_model'], late, own=laid.tokens)
    held = correction.card_taken('candidate')   # the lock of the card the solves are on: card 0's is given up around every pass (tube.passed)
    say = lambda record: print(json.dumps(dict(position=record['position'], matrix=record['matrix'], power=record['power'], seconds=record['seconds'], read_by=(record.get('passed') or {}).get('recorder'),
                                               written=None if record['written'] is None else {key: record['written'][key] for key in ('weights_moved', 'change_over_matrix') if key in record['written']},
                                               time=time.strftime('%H:%M:%S'))), flush=True)
    try:
        written = sorted({*names, *(start['kept']['changed'] if start else ())})   # every matrix this file may hold changed: this pass's, and its start's
        return run(at['sweep'], tube.ORIGINAL_MODEL, declaration['model_sha256'], {name: tensors[name] for name in written}, laid, names, rows_of,
                   [m['id'] for m in used], folds, asked['rule'], until, say, start, single=arithmetic == 'single')
    finally:
        if late is not None:
            late.close()
        if held is not None:
            held.close()


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        raise SystemExit('usage: python -m pipeline.weight_edit.gap_run <run folder> <pass, from 1> [<matrices to sweep before stopping>]')
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else None)
