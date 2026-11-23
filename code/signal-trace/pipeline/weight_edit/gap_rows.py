"""Where a sweep's rows come from: the `rows_of` that gap_sweep.sweep is given. For one matrix on one state of the file it is the
matrix's input as the engine multiplies it and the state its output is added to, at every answer step of the fitting answers and then
at every question position, and beside them the parts of the model the solver's reading stands on: the block's learned scale and, for
a feed-forward, its layer's last merge.

  stored   the original's rows as they were recorded once (ladder.INPUTS). No pass through the model. A file that holds another state
           than the original is refused: these rows are no other state's
  native   one pass of the working file through the run's declared recorder (tube.passed), for any state. A run's passes are checked
           against its declaration, so this source works inside a declared run and nowhere else. With a run's declared late start
           (gap_late.Late) a pass may be started from the outputs the pass before it recorded; the rows are the same bytes either way,
           and what is given back says which recorder computed them. With `own`, each answer's tokens as the model itself wrote
           them, the pass is read on those answers and not on the ones recorded for the original (`on_own`): the rows of a model
           that no longer answers as the original did

Both give the same form, with the answers the rows lie on and each answer's tokens, so that what the requests were made for is held to
what was read (gap_targets.Targets.held_to; gap_requests.Laid.held_to for a model's own answers). The learned scale and the last merge's parts are no matrix a sweep writes: they are read from the
original file, whose bytes a working file keeps everywhere outside the matrices written (gap_state). Behind the last layer's
feed-forward the merge is the model's last, and a target holds it as what the output norm passes on: with `passed_space`, the name
the target's schema gives that observable, the last merge's parts carry the norm's own (correction.terminal) and that name. Nothing
here chooses a matrix, a row or a request."""
from pathlib import Path

import numpy as np
import torch

from pipeline.weight_edit import gap_state, ladder, tube


def of_name(name):
    """The layer and the kind of the matrix a tensor's name stands for."""
    parts, kinds = name.split('.'), {held[1]: kind for kind, held in tube.KINDS.items()}
    if len(parts) != 4 or parts[0] != 'blk' or parts[3] != 'weight' or parts[2] not in kinds or not parts[1].isdigit():
        raise ValueError(f'{name} is no matrix a sweep solves')
    return int(parts[1]), kinds[parts[2]]


def lying(chosen):
    """The answers as their rows lie (every answer's steps in the answers' order, then every answer's question positions), and each
    answer's tokens: the prompt's, then the answer's own."""
    steps, questions = [m['steps'] for m in chosen], [m['question'] for m in chosen]
    answer_from = np.cumsum([0] + steps)
    question_from = answer_from[-1] + np.cumsum([0] + questions)
    answers = [dict(id=m['id'], steps=m['steps'], question=m['question'], answer_rows_from=int(answer_from[n]), question_rows_from=int(question_from[n]))
               for n, m in enumerate(chosen)]
    return answers, {m['id']: (m['tokens'], m['recorded']) for m in chosen}


def on_own(chosen, own):
    """The answers with the text a model itself wrote in the place of the recorded one: `own` is each answer's own tokens ({id:
    tokens}), for exactly these answers. The question and its positions are the answer's as before; its steps are its own text's."""
    if sorted(own) != sorted(m['id'] for m in chosen):
        raise ValueError('the own answers are not the chosen answers, each once')
    return [dict(m, recorded=[int(token) for token in own[m['id']]], steps=len(own[m['id']])) for m in chosen]


def beside(name, tensors, chosen, reading, passed_space=None):
    """What the solver's reading of one matrix stands on beside its rows: the block's learned scale and, for a feed-forward, its layer's
    last merge as the file holds it (field.Final's own parts, with no request). `reading` is how the rows were read, which says where
    the engine rounds what goes into a matrix (tube.read_together). With `passed_space`, the last layer's last merge also carries
    what the output norm needs to pass it on, and that name as the space it is read in (field.Terminal)."""
    layer, kind = of_name(name)
    device, steps = tube.DEVICE, [m['steps'] for m in chosen]
    values = lambda tensor: tube.checkpoint_values(tube.ORIGINAL_MODEL, tensors[tensor])
    vector = lambda tensor: torch.from_numpy(values(tensor).astype('f4').ravel()).to(device)
    matrix = lambda tensor: torch.from_numpy(values(tensor).reshape(tensors[tensor]['shape'][:2], order='F').astype('f4')).to(device)
    gain, last = vector(f'blk.{layer}.{tube.KINDS[kind][2]}.weight'), None
    if kind == 'feedforward':
        together = tube.read_together(sum(steps) + sum(m['question'] for m in chosen), steps, reading, device)
        last = dict(gate=matrix(f'blk.{layer}.inp_gate.weight'), proj=matrix(f'blk.{layer}.proj.weight'), scale=vector(f'blk.{layer}.post_norm.weight'),
                    t=tube.per_layer_inputs(chosen, layer, tensors, together), together=together)
        if passed_space is not None and f'blk.{layer + 1}.{tube.KINDS[kind][1]}.weight' not in tensors:   # no layer after this one
            from pipeline.weight_edit import correction   # that module reads the whole solver of November 12: loaded only here
            last.update(passes_on=correction.terminal(tensors, layer), space=passed_space)
    return gain, last


def stored(members, index, fitting, tensors, space, original_sha256, passed_space=None):
    """The original's rows of the fitting answers, as they were recorded while the original generated. `space` is the observable these
    rows are read in, by the name the target's schema gives it, and `passed_space` the name it gives the model's last merge as the
    output norm passes it on."""
    chosen = [m for m, take in zip(members, fitting) if take]
    answers, tokens = lying(chosen)
    state = gap_state.identity(original_sha256, {})
    take, every_step, steps = tube.rows_of(members, fitting), sum(m['steps'] for m in members), [m['steps'] for m in chosen]

    def rows_of(name, working):
        if working.identity() != state:
            raise ValueError(f'{name}: the stored rows are the original\'s, and the file holds another state')
        layer, kind = of_name(name)
        gain, last = beside(name, tensors, chosen, 'as generated', passed_space)
        read = []
        for what, wide in (('x', index['width'][f'{kind}-{layer}']), ('state', gain.shape[0])):
            answer, question = (np.memmap(ladder.INPUTS/f'{prefix}{what}-{kind}-{layer}.f32', dtype='<f4', mode='r').reshape(-1, wide) for prefix in ('', 'question-'))
            read.append(torch.from_numpy(np.concatenate([answer[take[take < every_step]], question[take[take >= every_step] - every_step]])).to(tube.DEVICE))
        return dict(x=tube.multiplied(read[0], steps, 'as generated'), incoming=read[1], gain=gain, last=last, space=space, state=state, answers=answers, tokens=tokens)

    return rows_of


def unused(run, name, state, ending=''):
    """A name for a pass of one matrix on one state that no pass of the run has taken: a stopped process's pass is left as it lies and
    never written over. No dot is in it, because the recorder's own output files are named by cutting a pass's name at its last dot."""
    base = f"sweep-{name.replace('.', '_')}-{state['sha256'][:16]}{ending}"
    taken = lambda each: any((Path(run)/'passes'/f'{each}{rest}').exists() for rest in ('', '-cases.json', '.stdout.jsonl', '.stderr.log'))
    found, number = base, 1
    while taken(found):
        number += 1
        found = f'{base}-{number}'
    return found


def native(run, members, fitting, tensors, space, passed_space=None, late=None, own=None):
    """Each matrix's rows from one pass of the working file, read the way the run declares its passes. The pass takes the card's lock by
    itself (tube.passed), and its raw stream is not kept: the rows are returned in memory. `space` and `passed_space` as `stored`. With
    `late` (gap_late.Late, made for the same answers) it decides which recorder makes each pass, and `passed` in what is given back says
    how the rows were read. With `own` ({id: tokens}) the fitting answers are read with the text the model itself wrote (`on_own`)."""
    tube.declared(run)
    chosen = [m for m, take in zip(members, fitting) if take]
    if own is not None:
        chosen = on_own(chosen, own)
    answers, tokens = lying(chosen)
    steps = [m['steps'] for m in chosen]

    def rows_of(name, working):
        layer, kind = of_name(name)
        state = working.identity()
        if late is None:
            passed = unused(run, name, state)
            arrays, how = tube.passed(working.file, run, passed, list(ladder.labels_of(layer, kind)), chosen), dict(recorder='observe-stream', passed=passed)
        else:
            arrays, how = late.read(name, working)
        x, incoming = (torch.from_numpy(each).to(tube.DEVICE) for each in arrays)
        gain, last = beside(name, tensors, chosen, tube.READING, passed_space)
        return dict(x=tube.multiplied(x, steps, tube.READING), incoming=incoming, gain=gain, last=last, space=space, state=state, answers=answers, tokens=tokens, passed=how)

    return rows_of
