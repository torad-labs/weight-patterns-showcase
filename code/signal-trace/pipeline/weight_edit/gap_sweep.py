"""A correction of the whole gap swept over many matrices, one after another in the order given.

For each matrix in turn: the rows as the working model now computes them, from whoever reads the model (`rows_of`); every row's request,
from where the target says its state should stand and where it now stands (gap_targets.matrix); every ridge read on the groups of answers
left out (field.path); the ridge the caller's rule takes from that reading (`rule`); and the matrix that ridge gives, written into the
working file (gap_state.Working.put). The next matrix's rows are then read from the file as it stands, so each matrix is asked what the
ones before it left of the target. A rule that takes no ridge leaves its matrix as it is.

A reading taken inside one sweep is held out at its first matrix only. Once a matrix is written from every fitting answer, each later
matrix reads the group it leaves out through weights that group helped to fit, so what `sweep`'s ridges are read on is no longer answers
the file was not solved on. `crossed` keeps the reading what it says it is: the sweep is made once for each group, in step, in a file of
its own that is solved without that group from its first matrix to its last. At each matrix every file reads its own rows, is solved on
the answers outside its group, and is read at every ridge on its group's rows, which nothing in that file was ever solved on; the rule
takes one ridge from those readings together, and every file writes its own matrix at it. What each of those files then leaves of its
own group's targets, read whole, is a correction's reading on answers no weight in the file was fit to. The file that is solved on
every fitting answer is a sweep of its own, its ridges named from the crossed one's (`sweep` with no groups).

Neither sweep chooses a matrix, an order, a target, a group or a ridge: each is the caller's, and each is recorded beside what it gave.
`least_held` is one rule a caller may pass: the ridge whose reading on the groups left out is least, and none where no ridge reads
under the matrix as it stands. It proposes a matrix; it judges nothing of a whole correction
(records/retrospectives/2026-11-14-whole-gap-native-rule-astra.md). Before anything is solved on them the rows are held to the state they were read from and to the target's own row map: rows of another
state of the file, or read on other tokens, other answers or another order, are refused whatever their shapes
(records/development/2026-11-14-whole-gap-solver/README.md)."""
import torch

from pipeline.weight_edit import field, gap_targets


def least_held(name, read):
    """A rule for `sweep` and `crossed`: of the powers every group's path read, the one at which what the groups left out are left at,
    summed over the groups, is least; the larger ridge where two read the same; and None, which leaves the matrix as it is, where no
    power reads under the matrix as it stands. A reading is a group's `all`: the rows that ask and the rows that ask nothing, in one
    sum. The powers are each least squares' own (field.Shared.powers, and upward to a ridge that moves no weight of the file): no
    other number enters."""
    powers = set.intersection(*({stage['power'] for stage in group['stages']} for group in read))
    if not powers:
        return None
    summed = {power: sum(next(stage['all'] for stage in group['stages'] if stage['power'] == power) for group in read) for power in powers}
    best = min(sorted(summed, reverse=True), key=summed.get)
    return best if summed[best] < sum(group['as_it_stands']['all'] for group in read) else None


def sweep(working, targets, names, rows_of, fitting, groups, rule, solved_on=None, say=None, single=False):
    """One sweep. `working` is the file being written (gap_state.Working), `targets` the target artifact (gap_targets.Targets), `names` the
    matrices in the order they are swept. Where the rows are read on answers the model itself wrote, `targets` is the artifact laid on
    those rows (gap_requests.Laid): it is read here as a target is, and holds the rows to its capture instead of to the reference.
      rows_of(name, working)  the matrix's rows as the working file now computes them: x (rows x input width, float32, as the engine
                 multiplies it), incoming (rows x width: the state the matrix's output is added to), gain (the block's learned scale),
                 space (the observable those states are read in), state (the name of the state the rows were read from:
                 working.identity() at the reading), answers and tokens (the row source's own inventory, for Targets.held_to), for
                 a feed-forward whose layer's last merge has targets, last (field.Final's gate, proj, scale, t and together), and,
                 where the source says it, passed (how the rows were read: the recorder, the pass, its seeds), kept in the entry
      fitting    the declared fitting answers' ids, in their order
      groups     the fitting answers' groups, as numbers in that order: field.path leaves each out in turn. With None no ridge is read
                 here, and the rule names each matrix's ridge from elsewhere (a crossed sweep's)
      rule(name, read)  a power of two of the least squares' own unit, or None: `read` is field.path's reading of every ridge on each
                 group left out, or None when no ridge is read here
      solved_on  the rows the sweep is solved on (all, when None): a row left unmarked is in no sum and no reading
      say        called with each matrix's entry as it is made
      single     every matrix's sums and least squares in single precision (field.Matrix.in_single); each entry then says so. A
                 path read that way ends where single precision stops resolving the rows' sums, far above the double one's end. Where
                 the rule's ridge is that end itself the reading had not turned inside what was read, so the matrix is read again in
                 double precision and written from that reading, and its entry keeps the single reading beside the double one
    Returns one entry a matrix: the state before and after, what the rows cover, the search, the reading, the rule's ridge and what was
    written."""
    told = []
    for name in names:
        before = working.identity()
        given = rows_of(name, working)
        if given['state'] != before:
            raise gap_targets.Refused(f'{name}: its rows were read from another state of the file than the one it holds')
        targets.held_to(given['answers'], given['tokens'], fitting)
        x = given['x']
        w = torch.from_numpy(working.held(name)).to(x.device)
        matrix = gap_targets.matrix(targets, name, x, given['incoming'], w, given['gain'], given['space'], given.get('last'), solved_on)
        if single:
            matrix.in_single()
        read = field.path(matrix, groups) if groups is not None else None
        power = rule(name, read)
        arithmetic = {'arithmetic': 'single'} if single else {}
        if single and read is not None and power is not None and power == max(group['stages'][-1]['power'] for group in read):
            arithmetic = {'arithmetic': 'double', 'read_in_single': read, 'power_in_single': power}
            read = field.path(matrix.in_double(), groups)
            power = rule(name, read)
        entry = dict(matrix=name, state_before=before, **({'passed': given['passed']} if 'passed' in given else {}), **arithmetic, rows=int(matrix.x.shape[0]),
                     asking=int(matrix.asking.sum()), unknown=matrix.unknown, sought=matrix.sought, read=read, power=power, written=None, state_after=before)
        if power is not None:
            written, wrote = field.written_at(matrix, power)
            entry.update(written=wrote, state_after=working.put(name, written.cpu().numpy()))
        told.append(entry)
        if say:
            say(entry)
        del matrix, x, w, given
    return told


def crossed(workings, targets, names, rows_of, fitting, groups, rule, say=None):
    """One sweep for each group of the fitting answers, in step. `workings[n]` is the file solved without `groups[n]`; `targets`,
    `names`, `rows_of`, `fitting` and `say` are `sweep`'s. For each matrix in turn, every file: its rows as it now computes them, every
    row's request, the least squares of the rows outside its group, and every ridge read on its group's rows (field.path with that one
    group). Then `rule(name, read)` takes one power from the readings of all the files, in the groups' order (the form `sweep` gives its
    rule), and each file writes the matrix its own least squares gives at that power: the matrix its stage of that power read. A rule
    that takes no ridge leaves the matrix as it is in every file. No file is written before every file's rows are read and held to
    their state and to the target's row map.
    Returns one entry a matrix: the power, and for each file the group it was solved without, the state before and after, what the rows
    cover, the search, the reading and what was written."""
    if len(workings) != len(groups) or len({each.file.resolve() for each in workings}) != len(workings):
        raise ValueError('a crossed sweep has one working file for each group of the fitting answers, and no file twice')
    told = []
    for name in names:
        folds, kept = [], []
        for working, group in zip(workings, groups):
            before = working.identity()
            given = rows_of(name, working)
            if given['state'] != before:
                raise gap_targets.Refused(f'{name}: its rows were read from another state of the file than the one it holds')
            targets.held_to(given['answers'], given['tokens'], fitting)
            x = given['x']
            w = torch.from_numpy(working.held(name)).to(x.device)
            matrix = gap_targets.matrix(targets, name, x, given['incoming'], w, given['gain'], given['space'], given.get('last'))
            systems = []
            read, = field.path(matrix, [group], systems=systems)
            folds.append(dict(without=read['answers'], state_before=before, rows=int(matrix.x.shape[0]), asking=int(matrix.asking.sum()), unknown=matrix.unknown,
                              sought=matrix.sought, read=read, written=None, state_after=before))
            kept.append((systems[0], x.device))
            del matrix, x, w, given
        power = rule(name, [fold['read'] for fold in folds])
        if power is not None:
            for working, fold, (system, device) in zip(workings, folds, kept):
                w = torch.from_numpy(working.held(name)).to(device)
                written, moved = field.written(w, system.on(device).change(power))
                fold.update(written=dict(power=power, ridge=system.unit * 2.0 ** power, unit=system.unit, rank=system.rank, **moved),
                            state_after=working.put(name, written.cpu().numpy()))
        entry = dict(matrix=name, power=power, folds=folds)
        told.append(entry)
        if say:
            say(entry)
        del kept
    return told
