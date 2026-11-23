"""A reference that is a set at every row, in the form a sweep takes a target in.

A target artifact (gap_targets) says where each row's state should stand after a merge: one point a row. A set reference says instead
what a row's state may stand in: everything some given points span, the same rows' own points of other answers (success_sets, whose
specification is records/retrospectives/2026-11-15-success-set-loop-freeze-astra.md). What a row is asked for is then the nearest
point of its set, which depends on where the row stands and is found when a matrix's requests are made (gap_requests._Nearest,
set_points). This file is only the reference's side of that: it holds a reference to what the solver needs of it and names each
matrix's merges. It chooses no point, no row and no coefficient, and makes no set.

What it reads of a reference (success_sets.Reference has each under this name):
  sha256      what names the reference: a record of pairs laid on its rows binds it
  answers     its rows, an answer each, in the order every answer's steps and then every answer's question positions: id, steps,
              question, answer_rows_from, question_rows_from, prompt_sha256, text_sha256
  schema      its merges in the model's order: merge, connection, layer, kind (attention, feedforward or layer), width, and space,
              the observable the merge's points are compared in
  offsets, donor_rows   row j's set is the points donor_rows[offsets[j]:offsets[j + 1]], each a row of the merge's points
  weights     each reference row's coefficient in the sum at one merge
  points(entry)   for a merge of the schema: every row's point there (rows x (width + 1): the unit state, then the log of its size) and
              which rows have one

A set is whole or it is not there: a reference row any of whose points is missing at a merge has no set at that merge. It asks nothing
there and is counted as unknown with the coefficient it would have had, which is neither a smaller set nor a set of nothing. The
comparison that measures a model against the reference holds a pair to the same rule (success_sets' scorer), so no row is asked toward a
set the measurement does not use."""
import numpy as np
import torch

from pipeline.weight_edit import gap_targets

F8 = torch.float64
KINDS = {'attention': 'attn_output', 'feedforward': 'ffn_down'}   # a merge's kind, and the matrix whose output it adds


class Refused(gap_targets.Refused):
    """The reference is not what the solver can read as a set at every row, or it is asked in another observable than its own."""


def _need(held, names, where):
    missing = [name for name in names if name not in held]
    if missing:
        raise Refused(f'{where} lacks {missing}')


class _Set:
    """One merge of the reference: the points its sets are drawn from, and which of them each row's set holds."""

    def __init__(self, sets, entry):
        self.sets, self.entry = sets, entry
        self.merge, self.layer, self.kind, self.width, self.space = (entry[name] for name in ('merge', 'layer', 'kind', 'width', 'space'))

    def points(self, space, device=None):
        """The merge's sets for rows read in `space`: `table` (every row's point, double precision), `offsets` and `members` (row j's
        set is the rows members[offsets[j]:offsets[j + 1]] of the table; a row whose set is not whole here has none) and `weight`
        (each reference row's coefficient, whether its set is there or not)."""
        if space != self.space:
            raise Refused(f'merge {self.merge} holds its points in the space {self.space!r}, and the rows are read in {space!r}')
        sets = self.sets
        table, valid = sets.reference.points(self.entry)[:2]
        table, valid = torch.from_numpy(np.ascontiguousarray(table, dtype='<f8')), torch.from_numpy(np.ascontiguousarray(valid, dtype=bool))
        if table.shape != (sets.rows, self.width + 1) or valid.shape != (sets.rows,):
            raise Refused(f'merge {self.merge}: its points are {tuple(table.shape)}, not a row of the unit state and the log of its size for each of {sets.rows} rows')
        if not bool(torch.isfinite(table[valid]).all()):
            raise Refused(f'merge {self.merge}: a row that is said to have a point holds a value that is not finite')
        offsets, members = sets.offsets, sets.members
        if not bool(valid.all()):
            lacking = ~valid[members]
            whole = torch.zeros(sets.rows, dtype=torch.long).index_add_(0, sets.set_of[lacking], torch.ones(int(lacking.sum()), dtype=torch.long)) == 0
            offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum((offsets[1:] - offsets[:-1]) * whole, 0)])
            members = members[whole[sets.set_of]]
        return dict(table=table.to(device), offsets=offsets.to(device), members=members.to(device), weight=sets.weights.to(device))


class Sets:
    """A set reference as a sweep reads a target: `sha256`, `answers` and `rows`, `matrix_nodes` and `of_matrix`. gap_requests.Laid
    lays it on a model's own answers as it lays an artifact, and a merge's request is then made where the rows stand."""

    def __init__(self, reference):
        self.reference, self.sha256 = reference, reference.sha256
        self.answers, self.rows = reference.answers, int(reference.rows)
        answer_from, question_from = 0, sum(each['steps'] for each in self.answers)
        for each in self.answers:
            _need(each, ('id', 'steps', 'question', 'answer_rows_from', 'question_rows_from', 'prompt_sha256', 'text_sha256'), 'an answer of the reference')
            if (each['answer_rows_from'], each['question_rows_from']) != (answer_from, question_from):
                raise Refused(f"{each['id']}: its rows do not lie where every answer's steps in order, then every answer's question positions, put them")
            answer_from, question_from = answer_from + each['steps'], question_from + each['question']
        if question_from != self.rows:
            raise Refused(f'the answers hold {question_from} rows and the reference says {self.rows}')
        self.offsets, self.members = (torch.from_numpy(np.ascontiguousarray(each, dtype='<i8')) for each in (reference.offsets, reference.donor_rows))
        self.weights = torch.from_numpy(np.ascontiguousarray(reference.weights, dtype='<f8'))
        sizes = self.offsets[1:] - self.offsets[:-1]
        if self.offsets.shape != (self.rows + 1,) or int(self.offsets[0]) != 0 or bool((sizes < 0).any()) or int(self.offsets[-1]) != len(self.members):
            raise Refused('the sets are not one run of points for each reference row, in the rows\' order')
        if len(self.members) and (int(self.members.min()) < 0 or int(self.members.max()) >= self.rows):
            raise Refused('a set names a point that is no row of the reference')
        if self.weights.shape != (self.rows,) or not bool((torch.isfinite(self.weights) & (self.weights > 0)).all()):
            raise Refused('a reference row\'s coefficient is not a positive number')
        self.set_of = torch.repeat_interleave(torch.arange(self.rows), sizes)   # the reference row each named point belongs to
        self.nodes, self.matrix_nodes = {}, {}
        for entry in reference.schema:
            _need(entry, ('merge', 'connection', 'layer', 'kind', 'width', 'space'), 'a merge of the reference')
            if entry['merge'] in self.nodes:
                raise Refused(f"the reference names merge {entry['merge']} twice")
            self.nodes[entry['merge']] = _Set(self, entry)
        for node in self.nodes.values():
            if node.kind == 'layer':
                continue
            if node.kind not in KINDS:
                raise Refused(f'merge {node.merge} is of the kind {node.kind!r}, which no matrix of a sweep writes')
            held = dict(own=node.merge)
            if node.kind == 'feedforward':
                final = [each.merge for each in self.nodes.values() if each.layer == node.layer and each.kind == 'layer']
                if len(final) != 1:
                    raise Refused(f'layer {node.layer}: its feed-forward has not one last merge in the reference')
                held['layer_final'] = final[0]
            name = f'blk.{node.layer}.{KINDS[node.kind]}.weight'
            if name in self.matrix_nodes:
                raise Refused(f'the reference holds two merges for {name}')
            self.matrix_nodes[name] = held

    def of_matrix(self, name):
        """A matrix's own merge and, for a feed-forward, its layer's last one (or None)."""
        if name not in self.matrix_nodes:
            raise Refused(f'the reference maps no merge to {name}')
        held = self.matrix_nodes[name]
        return self.nodes[held['own']], (self.nodes[held['layer_final']] if 'layer_final' in held else None)
