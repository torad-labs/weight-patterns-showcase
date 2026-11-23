"""A separability diagnostic at one matrix, for a run that declares one (runs/separable-at-17-original). It makes no candidate and no model.

Both declared candidates of the correction objective (correction-at-19, correction-at-17-original) brought their fitting objective down and
applied themselves where no cell asked: each carried the correct group's mean from the centre of its cells to their edge, and moved the prompt's
positions, which no cell reads (records/development/2026-11-12-correction-objective/rows_moved.py). A cell charges nothing inside its radius
and nothing reads a member's own movement or the prompt, so a solve is ALLOWED to move them. Whether the matrix is FORCED to, when it makes
the move the wrong group's cells ask for, is not known from that, and the two call for different next constructions. This reads it.

On the rows a run names, from the matrix it starts from, the declared path of ridges (correction.Descent, nothing of the solve changed) is
walked under each variant the declaration lists, each on the three rotations of the fitting folds: solved on two, read on the third.

   cells     the correction objective as declared: the cells alone. These are the fold paths of the candidate made beside this run.
   correct   the same cells and weights, and in the same sum each CORRECT member's own mean movement from its original computation, at every
             place and at every position of its prompt, charged whole (correction.quiet_of). A correct member's original computation on the
             same text is a known correct reference.
   prompts   as `correct`, and every position of a WRONG member's prompt.
   unasked   as `prompts`, and each wrong member's own mean movement at the places where the wrong group's cell asks for no move.
The last two requests have no known correct reference behind them (an unresolved wrong cell is not established agreement, and a wrong
question's prompt state is not a known correct one: Astra, November 12). They are here to see what each costs, not as an objective.

After every stage, for the answers solved on and for the answers read, apart: the cells' sum and its parts; how much of the gap the wrong
group's mean has made where a move is asked (direction, and size); how the two groups' cells stand against their radii; how far a member's own
mean and a prompt position have moved, per group; and the four requests' sums, each read the same way under every variant. Every movement is
counted from the original, as the cells count.

At the end each charged path is read beside the `cells` path of the same fold at its first stage that makes at least as much of the gap on
the answers read (the path's end where none does): a correct member's own movement, and a prompt position's, per part of the gap made. A
stronger ridge alone also moves everything less; this reading asks whether the request buys more than that.

This is one specified set of simultaneous requests at one matrix with a stated amount of work. It selects no candidate, changes no reference
and no declared setting, reads no model through the engine and writes none. A path whose solve fails is reported as that, with its status;
it does not show that the requests cannot be met.

   python -m pipeline.weight_edit.separable <run>        from code/signal-trace, on card 0 under its lock (the run's run.sh starts it)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROUND_TWO_RUNS
from pipeline.weight_edit import correction, tube
from pipeline.weight_edit.tube import EFFORT, PATH, RESOLVED, TINY

REQUESTS = dict(correct_places=dict(members='correct', prompt=False), correct_prompt=dict(members='correct', places=False),
                wrong_prompt=dict(members='wrong', places=False), wrong_unresolved=dict(members='wrong', prompt=False))   # correction.quiet_of's
VARIANTS = dict(cells=(), correct=('correct_places', 'correct_prompt'), prompts=('correct_places', 'correct_prompt', 'wrong_prompt'),
                unasked=('correct_places', 'correct_prompt', 'wrong_prompt', 'wrong_unresolved'))
PARTS = (0.1, 0.5, 0.9)   # the quantiles a spread is told in


def together(requests, names):
    """The requests named, as one: per merge, the sum of their weights."""
    return [{key: sum(requests[name][k][key] for name in names) for key in requests[names[0]][k]} for k in range(len(requests[names[0]]))]


def described(part, cells, requests, weights, g, asked, asked_size):
    """What the head of this file lists, for one part's answers at one matrix: the figures of a stage, and the same by place."""
    with torch.no_grad():
        stand = tube.Stand(part, weights, g)
        mean, size = correction.seen(part, stand.moved)[0]
        rest = correction.rested(part, stand.moved)
        members, sizes, turned_q, _ = rest[0]
        told = cells.read(mean, size)
        gap, gap_size = cells.target[0], cells.size_target[0]
        over = (mean - cells.target).norm(dim=-1) / cells.radius.clamp(min=TINY)   # how each group's cell stands: |mean - target| over its radius
        along = (mean * gap).sum(-1) / (gap ** 2).sum(-1).clamp(min=TINY)        # each group's mean movement along the gap, as a part of the gap
        grown = size / torch.where(asked_size, gap_size, torch.ones_like(gap_size))   # the same of the log size, where the size cell asks a move
        lengths, at_prompt, of_wrong = members.norm(dim=-1), turned_q.norm(dim=1), part.wrong[part.asker]
        spread = lambda values: [float(values.quantile(each)) for each in PARTS] if values.numel() else None
        figures = dict(cells=told['loss'], direction=told['direction'], size=told['size'], cells_outside=told['cells_outside'],
                       wrong=dict(made=float(along[0][asked].mean()), made_spread=spread(along[0][asked]), made_size=float(grown[0][asked_size].mean()) if bool(asked_size.any()) else None,
                                  over_radius=spread(over[0]), own=float(lengths[part.wrong].mean()), own_size=float(sizes[part.wrong].abs().mean()), prompt=spread(at_prompt[of_wrong])),
                       correct=dict(along_the_gap=float(along[1][asked].mean()), over_radius=spread(over[1]), own=float(lengths[~part.wrong].mean()),
                                    own_size=float(sizes[~part.wrong].abs().mean()), prompt=spread(at_prompt[~of_wrong])),
                       requests={name: float(correction.quiet_sum(each, rest)[0]) for name, each in requests.items()})
        by_place = dict(over_radius=over, along_the_gap=along, own=torch.stack([lengths[part.wrong].mean(0), lengths[~part.wrong].mean(0)]),
                        own_size=torch.stack([sizes[part.wrong].abs().mean(0), sizes[~part.wrong].abs().mean(0)]))
        return figures, {name: value.cpu().numpy() for name, value in by_place.items()}


def line(name, rounds, seconds, told):
    one = lambda each: (f"{each['cells']:.3e} | {each['wrong']['made']:+.3f} {each['wrong']['over_radius'][1]:.2f} {each['wrong']['own']:.4f} {each['wrong']['prompt'][1]:.4f} | "
                        f"{each['correct']['along_the_gap']:+.3f} {each['correct']['over_radius'][1]:.2f} {each['correct']['own']:.4f} {each['correct']['prompt'][1]:.4f}")
    return f"  {name:>5} {rounds:>3} {seconds:>5.0f} || {one(told['solved'])} || {one(told['read'])}"


HEAD = ('  ridge rounds secs || SOLVED ON: cells | wrong: gap made, median cell over its radius, a member\'s own move, median prompt position | '
        'correct: mean along the gap, median cell over its radius, a member\'s own move, median prompt position || READ: the same')


def matched(cells_path, end):
    """The stage of a fold's `cells` path that a charged path's end is read beside: the first at which the answers read make at least as much
    of the gap; the path's end where none does (the second value says which)."""
    for stage in cells_path['stages']:
        if stage['read']['wrong']['made'] >= end['read']['wrong']['made']:
            return stage, True
    return cells_path['stages'][-1], False


def diagnostic(run):
    folder = run/'paths'
    declaration = check_native_run(folder, run/'declaration.json')
    if folder.exists():
        raise ValueError('The run already has its paths')
    variants = list(declaration['diagnostic']['variants'])
    if not variants or variants[0] != 'cells' or any(each not in VARIANTS for each in variants):
        raise ValueError(f'every variant is read beside `cells`, which comes first; the variants known are {list(VARIANTS)}')
    made_with = {file.name: tube.digest(file) for file in (Path(__file__), Path(correction.__file__), Path(tube.__file__))}   # as they are when the command starts
    at = correction.started_from(run)
    beside = json.loads((tube.ROOT/declaration['parameters']['last_ridge']['measured_from']).read_text())   # the candidate made at this matrix on these rows
    if (beside['started_from']['rows'], beside['started_from']['model_sha256'], beside['tensor']) != (at['state']['source_rows'], at['state']['model_sha256'], at['state']['matrix']):
        raise ValueError('the candidate named beside this run was not solved on these rows of this matrix')
    last = beside['selected_ridge_power']
    on = correction.laid(at, np.load(tube.ROOT/at['state']['source_rows']))
    trace, frames, w, g, stands, used, fitting, groups, places = (on[name] for name in ('trace', 'frames', 'w', 'g', 'stands', 'used', 'fitting', 'groups', 'places'))
    if len(frames) != 1:
        raise ValueError('this diagnostic is written for a matrix read at its own merge alone')
    cells = frames[0]
    folds = [np.asarray(each[fitting]) for each in groups]
    names = [[m['id'] for m, take in zip(used, fold) if take] for fold in folds]
    if names != [each['answers'] for each in beside['folds']]:
        raise ValueError('the folds are not the ones the candidate beside this run was read on')
    everyone = np.logical_or.reduce(folds)
    whole = trace.part(everyone, places)
    unit = correction.unit_of(whole, frames, stands, g)
    first = correction.first_power(whole, frames, g, stands, unit)   # one unit and one first ridge for every path, from the cells alone, as correction.chosen takes them
    del whole
    asked = cells.target[0].norm(dim=-1) > cells.radius[0]              # the places where the wrong group's direction cell asks for a move
    asked_size = cells.size_target[0].abs() > cells.size_radius[0]      # and its size cell
    folder.mkdir()
    record = dict(run=run.name, tensor=at['state']['matrix'], rows=at['state']['source_rows'], rows_sha256=tube.digest(tube.ROOT/at['state']['source_rows']), model_sha256=at['state']['model_sha256'],
                  reference=str(at['reference'].relative_to(tube.ROOT)), reference_index_sha256=tube.digest(at['reference']/'index.json'), counted_from=on['counted_from'],
                  unit=unit, first_ridge_power=first, last_ridge_power=last,
                  beside=dict(candidate=declaration['parameters']['last_ridge']['measured_from'], unit=beside['unit'], first_ridge_power=beside['first_ridge_power'], selected_round=beside['selected_round']),
                  work=dict(EFFORT, resolved=RESOLVED, path=PATH), variants={name: list(VARIANTS[name]) for name in variants}, requests=REQUESTS, places=places,
                  places_where_the_wrong_cell_asks_a_move=dict(direction=int(asked.sum()), size=int(asked_size.sum())),
                  cell_weight=dict(direction=[float(each) for each in cells.weight[:, 0]], size=[float(each) for each in cells.size_weight[:, 0]]),
                  request_weight='a member\'s squared own mean movement at a place, and a prompt position\'s, by its group\'s cell weight (the wrong group first, above) over the number of the group\'s members among the answers summed over',
                  fitting_answers=[m['id'] for m in used], folds=names, made_with=made_with, paths=[], beside_cells=[])
    kept = {}
    with open(run/'summary.txt', 'w') as text:
        def say(*told):
            print(*told, file=text, flush=True)
            print(*told, flush=True)
        say(f"{run.name}: {at['state']['matrix']} from the matrix the run starts from, on {at['state']['source_rows']}. Every variant on the three rotations of the folds, from ridge 2^{first} "
            f"of the unit ({unit:.9g}) down to 2^{last}. The candidate beside it had the unit {beside['unit']:.9g} and the first ridge 2^{beside['first_ridge_power']}. "
            f"The wrong group's direction cell asks a move at {int(asked.sum())} of {places} places, its size cell at {int(asked_size.sum())}.")
        for variant in variants:
            for k, fold in enumerate(folds):
                parts = dict(solved=trace.part(everyone & ~fold, places), read=trace.part(fold, places))
                requests = {key: {name: correction.quiet_of(part, frames, **which) for name, which in REQUESTS.items()} for key, part in parts.items()}
                if VARIANTS[variant]:
                    parts['solved'].quiet = together(requests['solved'], VARIANTS[variant])   # in the sum that is solved; the answers read keep the cells alone
                walk = correction.Descent(parts['solved'], frames, w, g, stands, unit, EFFORT, beside=[parts['read']])
                both = lambda weights: {key: described(part, cells, requests[key], weights, g, asked, asked_size) for key, part in parts.items()}
                figures = lambda found: {key: each[0] for key, each in found.items()}
                say(f'{variant}, read fold {k}')
                say(HEAD)
                start = figures(both(stands))
                say(line('start', 0, 0, start))
                stages, power, found = [], first, None
                while walk.ended is None and power >= last:
                    began = time.time()
                    stage = walk.stage(power)
                    found = both(walk.weights)
                    stages.append(dict(ridge_power=power, status=stage['status'], rounds=stage['rounds'], rounds_taken=stage['rounds_taken'], seconds=round(time.time() - began, 1), objective=stage['objective'],
                                       read_objective=stage['beside'][0], change_over_matrix=stage['change_over_matrix'], **figures(found),
                                       each_round=[{name: each.get(name) for name in ('taken', 'objective_before', 'objective', 'objective_of_the_step_to_first_order', 'part_of_the_step_taken', 'passes', 'iterations', 'seconds')}
                                                   | dict(slope_left=[one['slope_left'] for one in each.get('each_pass', [])]) for each in stage['each_round']]))
                    say(line(f'2^{power}', stage['rounds_taken'], stages[-1]['seconds'], figures(found)) + f"   {stage['status']}")
                    power -= PATH
                ended = walk.ended or 'the last ridge declared'
                np.savez(folder/f'{variant}-fold-{k}.npz', matrix=walk.weights.cpu().numpy(), asked=asked.cpu().numpy(), asked_size=asked_size.cpu().numpy(),
                         **({f'{key}_{name}': value for key, each in found.items() for name, value in each[1].items()} if found else {}))
                path = dict(variant=variant, fold_read=k, first_ridge_power=first, ended=ended, weights_changed=int((walk.weights != stands).sum()), start=start, stages=stages)
                tube.save(folder/f'{variant}-fold-{k}.json', path)
                kept[variant, k] = path
                before = next((each['beside'][0] for each in beside['fold_paths'][k]['stages'] if each['ridge_power'] == last), None)
                if variant == 'cells' and stages:
                    say(f"  the answers read stand at {stages[-1]['read_objective']:.9g} of the cells' sum; the same fold of the candidate beside this run stood at {before:.9g} at the end of this ridge")
                record['paths'].append(dict(variant=variant, fold_read=k, ended=ended, stages_made=len(stages), last_stage=stages[-1]['status'] if stages else None,
                                            end={key: (stages[-1][key] if stages else start[key]) for key in ('solved', 'read')}, change_over_matrix=stages[-1]['change_over_matrix'] if stages else 0.0,
                                            read_objective=stages[-1]['read_objective'] if stages else None, the_candidate_beside_read=before if variant == 'cells' else None,
                                            seconds=round(sum(each['seconds'] for each in stages), 1)))
                tube.save(run/'summary.json', record)   # kept as it grows
                del walk, parts, requests, both, found
                with tube.CARD.given():   # between two paths another project's waiting job gets the card
                    time.sleep(5)
        say('each charged path\'s end on the answers READ (never solved on), beside the `cells` path of the same fold at its first stage that makes at least as much of the gap there:')
        say('  variant fold | gap made | a correct member\'s own move per part of the gap made: here, at `cells` (its ridge, its gap made) | a correct member\'s prompt position (median) per part of the gap made: here, at `cells` | '
            'a wrong member\'s prompt position (median): here, at `cells` | the cells\' sum read: here, at `cells`')
        for variant in variants[1:]:
            for k in range(len(folds)):
                path, base = kept[variant, k], kept['cells', k]
                if not path['stages'] or not base['stages']:
                    say(f'  {variant:<8} {k}: no stage was made')
                    continue
                end = path['stages'][-1]
                stage, reached = matched(base, end)
                made, there = end['read']['wrong']['made'], stage['read']['wrong']['made']
                per = lambda value, gap: value / gap if gap > 0 else None
                told = dict(variant=variant, fold_read=k, gap_made=made, cells_stage=dict(ridge_power=stage['ridge_power'], gap_made=there, makes_at_least_as_much=reached),
                            correct_own_per_gap_made=[per(end['read']['correct']['own'], made), per(stage['read']['correct']['own'], there)],
                            correct_prompt_per_gap_made=[per(end['read']['correct']['prompt'][1], made), per(stage['read']['correct']['prompt'][1], there)],
                            wrong_prompt=[end['read']['wrong']['prompt'][1], stage['read']['wrong']['prompt'][1]], cells_read=[end['read']['cells'], stage['read']['cells']],
                            solved_gap_made=end['solved']['wrong']['made'])
                record['beside_cells'].append(told)
                show = lambda pair: ' '.join('none' if each is None else f'{each:.4f}' for each in pair)
                say(f"  {variant:<8} {k} | {made:+.3f} | {show(told['correct_own_per_gap_made'])} (2^{stage['ridge_power']}, {there:+.3f}{'' if reached else ', its end: no stage makes as much'}) | "
                    f"{show(told['correct_prompt_per_gap_made'])} | {show(told['wrong_prompt'])} | {end['read']['cells']:.3e} {stage['read']['cells']:.3e}")
        tube.save(run/'summary.json', record)


if __name__ == '__main__':
    tube.CARD.take()
    diagnostic(ROUND_TWO_RUNS/sys.argv[1])
