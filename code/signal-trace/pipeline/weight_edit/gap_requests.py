"""The requests of a correction at the rows of a model's own answers.

A target artifact (gap_targets) says, for every row of the reference history, where the row's state should stand after a merge. A model
that has been written to answers in its own way: its rows are not the reference's rows, and a record of the computation lays each of its
rows on one or more of the reference's (the pairs). The comparison of a whole own history with the reference (context_score.paired_loss)
is a sum over the pairs: a reference row's coefficient, shared equally among the own rows laid on it, times the squared distance of the
own row's state from that reference row's target. This module gives the solver that same sum as one request a row of the model's own
answer, with nothing of it dropped.

For an own row k laid on the reference rows j, with c_kj the coefficient of j over the number of own rows laid on j, and u the row's unit
state:

    sum_j c_kj |u - t_j|^2  =  C_k |u - m_k|^2 + D_k
    C_k = sum_j c_kj,   m_k = sum_j c_kj t_j / C_k,   D_k = sum_j c_kj |t_j - m_k|^2

and the same for the log of the size, with a_k and E_k. Three things follow, and each is kept:

- m_k is the coefficient-weighted mean of the targets and is not brought back to unit length. Its length is part of the identity: with a
  mean made unit again the row's sum would be another sum.
- C_k keeps the reference's mass. Every reference row's coefficient is shared among the own rows laid on it, so the C_k add up to the
  reference's coefficients, whatever the number of own rows.
- D_k and E_k do not move with the row. They are what the reference rows laid on one own row differ by among themselves, and no state
  of that row makes them smaller. They are returned beside the request: what the solver reads of the rows, with them added, is the
  comparison's sum, and a least of the solver's reading is never read as a sum of nothing.

The targets of a pair may be any point a pair is asked for. Where the reference is a set at every row (what several answers' own
points span there, gap_sets), a pair's point is the set's point nearest the own row's state, found where the row stands when the request
is made (set_points). Two own rows laid on one reference row then have two points, and each pair goes through the same reduction with
its own point in t_j's place (`reduced`, `_Nearest`): no point is found once a reference row, and none from a mean of rows. A point so
found is held while a matrix is solved. The row's distance to its set is never more than its distance to that held point and is equal
to it where the row stood, so what the solver reads of the rows, with the dispersion, is the set's own sum where the matrix stands and
bounds it from above at every other matrix; the sum itself is made again from the rows a model leaves, at their own nearest points.

One weight a row serves both channels where the reference's two channels share their coefficient and status at every row, which
gap_targets.Node.asked already holds an artifact to. No request is made from a token: a pair is two row numbers.

`Laid` is a whole artifact laid this way, in the form a sweep takes a target in (gap_sweep.sweep, gap_targets.matrix): the rows of a
model's own answers, each merge's request from `paired`, and what says a reading's rows are those rows. That last is the capture of the
model's answers and nothing of the reference's text: an answer's own tokens are named by their sha256 as the capture recorded them, so
that rows read on another input than the pairs were laid for are refused. `Laid.read` makes one from the record a capture is scored
through (`whole-gap-correspondence/2`), so the pairs a pass is solved on are the pairs the state it starts from was measured with.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pipeline.weight_edit import gap_targets, set_points

F8 = torch.float64
BLOCK = 4096   # pairs laid at once: it orders the additions and bounds the work's memory, and puts no pair into a sum or out of one
FORM, BASIS = 'whole-gap-correspondence/2', 'original-computation'   # the record of a capture's pairs, and what its pairs were laid on


class Refused(gap_targets.Refused):
    """The pairs do not lay every own row and every reference row, or name a row that is not there; or a record of them is not what
    it says it is."""


def paired(asked, pairs, rows):
    """One request a row of a model's own answer, from the reference's requests and the pairs that lay the own rows on the reference's.

    `asked` is what gap_targets.Node.asked returns for a merge: `unit` (reference rows x width), `log_size`, `weight` (the root of each
    row's coefficient, one for both channels) and `known`. `pairs` is whole numbers (pairs x 2): an own row, then the reference row it is
    laid on, both in the artifact's row order. `rows` is the number of own rows.

    Returned, at the own rows: `unit` and `log_size` (the means m and a, with nothing in the places of rows that ask nothing), `weight`
    (the root of C), `known` (the row is laid on a reference row whose target is known), `dispersion` (D under `direction` and E under
    `size`, a row each) and `mass` (the reference's known coefficients summed, and the C summed: the same number)."""
    unit, log_size, known = asked['unit'], asked['log_size'], asked['known']
    device = unit.device
    pairs = torch.as_tensor(pairs, device=device)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype not in (torch.int32, torch.int64) or len(pairs) == 0:
        raise Refused('pairs are whole numbers, two a pair: an own row and the reference row it is laid on')
    own, reference = pairs[:, 0].long(), pairs[:, 1].long()
    there = unit.shape[0]
    if int(own.min()) < 0 or int(reference.min()) < 0 or int(own.max()) >= rows or int(reference.max()) >= there:
        raise Refused(f'a pair names a row outside the {rows} own rows or the {there} reference rows')
    laid_on = torch.bincount(reference, minlength=there)
    lays = torch.bincount(own, minlength=rows)
    if bool((laid_on == 0).any()) or bool((lays == 0).any()):
        raise Refused(f'{int((lays == 0).sum())} own rows are laid on no reference row and {int((laid_on == 0).sum())} reference rows have no own row laid on them: '
                      'every row of both histories is in the comparison')
    coefficient = asked['weight'].to(F8) ** 2 * known
    share = coefficient[reference] / laid_on[reference]            # c_kj: nothing where the reference row's target is unknown
    request, total = reduced(own, share, lambda some: unit[reference[some]].to(F8), log_size[reference].to(F8), rows, unit.shape[1])
    return dict(request, mass=dict(reference=float(coefficient.sum()), laid=float(total.sum())))


def reduced(own, share, unit_of, log_size, rows, width):
    """The identity above, for pairs that each have a point of their own: `own` is each pair's own row and `share` its coefficient
    c_kj; `unit_of(some)` gives the unit part of the points of a slice of the pairs (pairs x width, double precision) and `log_size`
    every pair's log of the size. Returned: the request at the `rows` own rows (`unit` and `log_size`, the means m and a; `weight`,
    the root of C; `known`; `dispersion`, D and E a row each), and C itself."""
    device = share.device
    total = torch.zeros(rows, dtype=F8, device=device).index_add_(0, own, share)
    asks = total > 0
    safe = torch.where(asks, total, torch.ones_like(total))
    mean = torch.zeros(rows, width, dtype=F8, device=device)
    middle = torch.zeros(rows, dtype=F8, device=device).index_add_(0, own, share * log_size)
    for first in range(0, len(own), BLOCK):
        some = slice(first, first + BLOCK)
        mean.index_add_(0, own[some], share[some, None] * unit_of(some))
    mean, middle = mean / safe[:, None], middle / safe
    turned = torch.zeros(rows, dtype=F8, device=device)
    for first in range(0, len(own), BLOCK):
        some = slice(first, first + BLOCK)
        apart = unit_of(some) - mean[own[some]]
        turned.index_add_(0, own[some], share[some] * (apart * apart).sum(1))
    grown = torch.zeros(rows, dtype=F8, device=device).index_add_(0, own, share * (log_size - middle[own]) ** 2)
    return dict(unit=mean, log_size=middle, weight=torch.sqrt(total), known=asks, dispersion=dict(direction=turned, size=grown)), total


def summed(request, unit, log_size):
    """The comparison's two sums for own rows that stand at `unit` (unit states, rows x width) and `log_size`: what the rows leave of their
    requests, weighed, with the dispersion that no state of a row takes away. The direction's and the size's, apart."""
    weight = request['weight'].to(F8) ** 2
    direction = (weight * ((unit.to(F8) - request['unit']) ** 2).sum(1)).sum() + request['dispersion']['direction'].sum()
    size = (weight * (log_size.to(F8) - request['log_size']) ** 2).sum() + request['dispersion']['size'].sum()
    return float(direction), float(size)


def _sha256(path):
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


class _At:
    """One merge of a laid artifact: gap_targets.Node.asked's form, at the own rows."""

    def __init__(self, laid, node):
        self.laid, self.node, self.merge, self.space = laid, node, node.merge, node.space

    def asked(self, space, device=None):
        asked = self.node.asked(space, device)
        request = paired(asked, self.laid.pairs, self.laid.rows)
        self.laid.carried[self.merge] = dict(mass=request['mass'], dispersion={channel: float(each.sum()) for channel, each in request['dispersion'].items()},
                                             asking=int(request['known'].sum()), pairs=int(len(self.laid.pairs)))
        return dict(request, coverage=asked['coverage'])


class _Nearest:
    """One merge of a reference that is a set at every row (gap_sets), laid on the own rows: a row's request is found where the row
    stands. For each pair, the point of the reference row's set nearest the own row's state as the matrix about to be solved leaves it
    (set_points.onto: the unit state and the log of its size side by side, one mixture for both); then `reduced`, with the reference
    row's coefficient shared among the own rows laid on it, whichever of them are given."""

    def __init__(self, laid, node):
        self.laid, self.node, self.merge, self.space = laid, node, node.merge, node.space

    def asked_where(self, space, unit, log_size, take):
        """The request at the rows `take` marks (of all the own rows), which stand at `unit` and `log_size` after this merge:
        gap_targets.Node.asked's `unit`, `log_size`, `weight` and `known`, at those rows. A reference row that has no set at this
        merge asks nothing and its coefficient is in no sum: it is counted under `unknown` in what is carried, with that coefficient.
        A row given that does not stand at a finite point is refused: it is a reading that failed, not a row that asks nothing."""
        laid, device = self.laid, unit.device
        held = self.node.points(space, device)
        pairs = torch.as_tensor(laid.pairs, device=device)
        own, reference = pairs[:, 0], pairs[:, 1]
        there, rows = len(held['offsets']) - 1, int(take.sum())
        if take.shape != (laid.rows,) or unit.shape != (rows, held['table'].shape[1] - 1) or log_size.shape != (rows,):
            raise Refused(f'merge {self.merge}: the rows given are not the {laid.rows} own rows the requests were laid on, or stand in another width')
        if int(reference.max()) >= there:
            raise Refused(f'merge {self.merge}: a pair names a reference row past the {there} the sets are of')
        laid_on = torch.bincount(reference, minlength=there)
        if bool((laid_on == 0).any()):
            raise Refused(f'merge {self.merge}: {int((laid_on == 0).sum())} reference rows have no own row laid on them: every row of both histories is in the comparison')
        known = held['offsets'][1:] > held['offsets'][:-1]
        coefficient = held['weight'].to(F8) * known
        share = coefficient[reference] / laid_on[reference]
        place = torch.full((laid.rows,), -1, dtype=torch.long, device=device)
        place[take] = torch.arange(rows, device=device)
        asks = take[own] & known[reference]
        if not bool(asks.any()):
            raise Refused(f'merge {self.merge}: no row given is laid on a reference row whose set holds a point')
        mine, share = place[own[asks]], share[asks]
        standing = torch.cat([unit.to(F8), log_size.to(F8)[:, None]], 1)
        if not bool(torch.isfinite(standing).all()):
            raise Refused(f'merge {self.merge}: {int((~torch.isfinite(standing).all(1)).sum())} rows given do not stand at a finite point')
        found = set_points.onto(held['table'], held['offsets'], held['members'], reference[asks], standing[mine])
        point = found['point']
        request, total = reduced(mine, share, lambda some: point[some, :-1], point[:, -1], rows, unit.shape[1])
        apart = (standing[mine] - point) ** 2
        laid.carried[self.merge] = dict(
            mass=dict(reference=float(coefficient.sum()), laid=float(total.sum())),
            dispersion={channel: float(each.sum()) for channel, each in request['dispersion'].items()},
            asking=int(request['known'].sum()), pairs=int(len(own)), unknown=dict(rows=int((~known).sum()), mass=float(held['weight'].to(F8)[~known].sum())),
            stands=dict(direction=float((share * apart[:, :-1].sum(1)).sum()), size=float((share * apart[:, -1]).sum())),
            nearest=dict(pairs=int(asks.sum()), points_carrying=dict(least=int(found['support'].min()), most=int(found['support'].max()), mean=float(found['support'].double().mean())),
                         rounds=found['rounds'], widest_gap=float(found['gap'].max()), over_its_bound=int((found['gap'] > found['bound']).sum()),
                         sum_within=float(2 * (share * found['gap'].clamp(min=0)).sum())))
        return {name: request[name] for name in ('unit', 'log_size', 'weight', 'known')}


class Laid:
    """A target artifact laid on the rows of a model's own answers.

    The rows keep the artifact's order: every answer's own steps in the answers' order, then every answer's question positions. An
    answer's steps are laid on the reference's by its pairs ([own step, reference step]: a whole path of single steps that loses no step
    of either side, the rule the comparison holds its own pairs to, complete_overlap.validate_pairs). A question's positions are laid
    one on one: the question is the same input in both, whatever the model then writes. An answer the model writes as the reference did
    is the case of pairs that lay each step on itself, and takes no other way through here.

      targets   the artifact (gap_targets.Targets)
      own       each answer's own tokens ({id: tokens}), for exactly the artifact's answers
      pairs     each answer's pairs ({id: whole numbers, pairs x 2})
      bound     what the laying was read from, kept as given: `read` names the record, the capture and the model

    A sweep reads `sha256` (the artifact's), `held_to`, `answer_of_rows` and `of_matrix` as it reads a target's. `carried` keeps, by
    merge, what each request made here carries beside its rows: the reference's mass and the mass laid (one number), the dispersion that
    no state of a row takes away, how many own rows ask, and the number of pairs.

    `targets` may be a reference that is a set at every row (gap_sets.Sets) in the artifact's place. A merge's request is then each
    pair's nearest point of its set, found where the rows stand (`_Nearest`), and what is carried also says: `stands`, the set's own
    sum at that merge for the rows as the matrix about to be solved leaves them, direction and size apart; `unknown`, the reference
    rows that have no set there and the coefficients they would have had; and `nearest`, how the points were found (how many of a set's points carry one, the rounds
    of the search, the widest gap any pair ended at, how many pairs ended over rounding's own bound, and `sum_within`: `stands` is the
    exact sum to within that much)."""

    def __init__(self, targets, own, pairs, bound=None):
        from pipeline.overlap.complete_overlap import validate_pairs
        self.targets, self.sha256, self.bound, self.carried = targets, targets.sha256, bound, {}
        reference = targets.answers
        names = [each['id'] for each in reference]
        if sorted(own) != sorted(names) or sorted(pairs) != sorted(names):
            raise Refused("the own answers and their pairs are not the artifact's answers, each once")
        self.tokens = {name: [int(token) for token in own[name]] for name in names}
        steps = [len(self.tokens[name]) for name in names]
        answer_from = np.cumsum([0] + steps)
        question_from = answer_from[-1] + np.cumsum([0] + [each['question'] for each in reference])
        self.answers, lays = [], []
        for number, each in enumerate(reference):
            mine = np.asarray(pairs[each['id']])
            try:
                validate_pairs(mine, steps[number], each['steps'])
            except ValueError as why:
                raise Refused(f"{each['id']}: its pairs are no whole path of its {steps[number]} own steps on the reference's {each['steps']} ({why})") from None
            self.answers.append(dict(id=each['id'], steps=steps[number], question=each['question'], answer_rows_from=int(answer_from[number]),
                                     question_rows_from=int(question_from[number]), prompt_sha256=each['prompt_sha256'],
                                     text_sha256=gap_targets.of_tokens(self.tokens[each['id']])))
            lays.append(mine.astype('<i8') + np.array([int(answer_from[number]), each['answer_rows_from']], dtype='<i8'))
            lays.append(np.arange(each['question'], dtype='<i8')[:, None] + np.array([int(question_from[number]), each['question_rows_from']], dtype='<i8'))
        self.rows = int(question_from[-1])
        self.pairs = np.concatenate(lays)

    @classmethod
    def read(cls, targets, record, capture):
        """The laying of the record at `record` (`whole-gap-correspondence/2`), held to the capture it names: `capture` is the folder a
        model's own answers were captured into (own/text.json, own/<id>/answer.json and own/<id>/pairs-original.npy). Refused unless
        the record is of this artifact, the capture is the one the record was made from, and every answer's tokens and pairs are the
        files the capture bound when it closed: an answer the record names as changed by the record's own hashes of them, and an answer
        it names as unchanged by tokens that are the reference's with each step on itself. What model the capture is of is the
        record's to say (`bound['model_sha256']`) and the caller's to hold a pass's start to."""
        raw = Path(record).read_bytes()
        told, own = json.loads(raw), Path(capture)/'own'
        if told.get('format') != FORM or told.get('basis') != BASIS:
            raise Refused(f"the record is {told.get('format')!r} on {told.get('basis')!r}, not {FORM!r} on {BASIS!r}")
        if told.get('target_manifest_sha256') != targets.sha256:
            raise Refused('the record lays its answers on another target artifact')
        text_raw = (own/'text.json').read_bytes()
        text = json.loads(text_raw)
        if hashlib.sha256(text_raw).hexdigest() != told['capture']['text_json_sha256'] or text['model_sha256'] != told['model_sha256']:
            raise Refused('the capture is not the one the record was made from')
        changed, unchanged = told['answers'], told['unchanged']
        if sorted([*changed, *unchanged]) != sorted(each['id'] for each in targets.answers):
            raise Refused("the record's answers are not the artifact's, each once")
        tokens, pairs = {}, {}
        for each in targets.answers:
            name = each['id']
            bound = text['files'][name]
            if _sha256(own/name/'answer.json') != bound['answer.json']:
                raise Refused(f'{name}: answer.json is not the file its capture recorded')
            tokens[name] = [int(token) for token in json.loads((own/name/'answer.json').read_text())['tokens']]
            now = gap_targets.of_tokens(tokens[name])
            if name in changed:
                held = changed[name]
                if (held['answer_json_sha256'], held['current_text_sha256'], held['original_text_sha256'], held['steps'], held['original_steps']) != \
                        (bound['answer.json'], now, each['text_sha256'], len(tokens[name]), each['steps']):
                    raise Refused(f"{name}: the record's answer is not the capture's, or its reference is not the artifact's")
                if held['pairs_sha256'] != bound.get('pairs-original.npy') or _sha256(own/name/'pairs-original.npy') != held['pairs_sha256']:
                    raise Refused(f'{name}: pairs-original.npy is not the file its capture recorded')
                pairs[name] = np.load(own/name/'pairs-original.npy', allow_pickle=False)
                if not np.array_equal(pairs[name], np.asarray(held['pairs'])):
                    raise Refused(f"{name}: the record's pairs are not the capture's")
            else:
                if now != each['text_sha256'] or len(tokens[name]) != each['steps']:
                    raise Refused(f"{name}: the record names it unchanged, and its tokens are not the reference's")
                pairs[name] = np.column_stack([np.arange(each['steps'])] * 2)
        return cls(targets, tokens, pairs, bound=dict(record_sha256=hashlib.sha256(raw).hexdigest(), model_sha256=told['model_sha256'], reading_sha256=told.get('reading_sha256'),
                                                      capture_text_json_sha256=told['capture']['text_json_sha256'], protocol_sha256=(told.get('protocol') or {}).get('sha256'),
                                                      changed=sorted(changed), unchanged=sorted(unchanged)))

    def held_to(self, answers, tokens, fitting):
        """The caller's rows are the rows the requests were laid on: gap_targets.Targets.held_to's form, against the capture's own
        answers. An answer's question is named by the artifact's sha256 of its tokens, the same input in both; its text by the
        capture's own."""
        if [each['id'] for each in self.answers] != list(fitting):
            raise Refused('the laid answers are not the declared fitting answers in their order')
        if len(answers) != len(self.answers):
            raise Refused(f'the rows hold {len(answers)} answers and the laying {len(self.answers)}')
        for mine, theirs in zip(self.answers, answers):
            for name in ('id', 'steps', 'question', 'answer_rows_from', 'question_rows_from'):
                if mine[name] != theirs[name]:
                    raise Refused(f"{mine['id']}: its {name} is {mine[name]} where the requests were laid and {theirs[name]} in the rows")
            prompt, text = tokens[mine['id']]
            if gap_targets.of_tokens(prompt) != mine['prompt_sha256'] or gap_targets.of_tokens(text) != mine['text_sha256'] or len(text) != mine['steps']:
                raise Refused(f"{mine['id']}: the rows were read on another input than the capture the requests were laid on")

    def answer_of_rows(self, device=None):
        """Which answer every own row belongs to, as a number: the answer's place in the artifact's order."""
        out = torch.empty(self.rows, dtype=torch.long)
        for number, each in enumerate(self.answers):
            out[each['answer_rows_from']:each['answer_rows_from'] + each['steps']] = number
            out[each['question_rows_from']:each['question_rows_from'] + each['question']] = number
        return out.to(device)

    def of_matrix(self, name):
        """A matrix's own merge and, for a feed-forward, its layer's last one (or None), each asked at the own rows."""
        own, final = self.targets.of_matrix(name)
        at = lambda node: (_Nearest if hasattr(node, 'points') else _At)(self, node)   # a set at every row, or a point at every row
        return at(own), (None if final is None else at(final))

    def described(self):
        """What this laying is, for a sweep's description: what it was read from, each answer's own steps and tokens by their sha256,
        and the sha256 of every pair in the rows' order. A sweep taken up under other own answers or other pairs is another sweep."""
        return dict(bound=self.bound, rows=self.rows, answers=[dict(id=each['id'], steps=each['steps'], text_sha256=each['text_sha256']) for each in self.answers],
                    pairs_sha256=hashlib.sha256(np.ascontiguousarray(self.pairs, dtype='<i8').tobytes()).hexdigest())

    def carried_at(self, name):
        """What the requests last made for a matrix carry beside their rows: under `own` and, for a feed-forward, `layer_final`."""
        return {which: dict(merge=merge, **self.carried[merge]) for which, merge in self.targets.matrix_nodes[name].items() if merge in self.carried}
