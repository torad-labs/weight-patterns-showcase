"""A whole-gap target artifact read and held to what it says of itself, and a merge's requests made from it for the solver.

The file format is records/development/2026-11-14-whole-gap-targets/FORMAT.md (`whole-gap-targets/1`): for every merge of the model and
every row of the fitting answers, where the state should stand after the merge (its unit direction and the log of its size, absolute, not a
step), each channel's coefficient in the loss, and each channel's status: resolved, or unknown for a reason the artifact names. This file
chooses nothing of that. It refuses an artifact that is not what the format says, and it turns one merge's targets into what
field.Matrix and field.Final take, without ever multiplying a missing value:

  read(folder, root)      the manifest held to its format: every required field, the row map, the records it is bound to, one node per
                          merge of the observation schema in the schema's own order. Arrays are opened a node at a time (`Node.arrays`),
                          each held to its sha256, its shape and its type, and to the status rules.
  Targets.held_to(...)    the artifact's rows are the caller's rows: the same answers in the same order with the same lengths and places,
                          the same prompt and text tokens, and exactly the declared fitting answers. Equal lengths are never enough.
  Node.asked(space)       finite buffers for the solver: the targets with nothing in the unknown rows' places, one weight a row (the
                          square root of the row's coefficient, nothing where the row is unknown) and which rows are known. The solver
                          takes one weight a row for both channels, so a merge where a row's two channels differ in status or in
                          coefficient is refused here by name and count (Unsupported): nothing is averaged, squared again or dropped.

Two spellings the format leaves open are fixed here and nowhere else: a node's four arrays are described under `files`, keyed by the file
names the format gives (unit.npy, log_size.npy, loss_weight.npy, status.npy), and the observation schema record is JSON with `merges`, a list
of merge, connection, layer, kind, width and space in the model's own order."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

FORMAT = 'whole-gap-targets/1'
ORDER = 'all_answers_then_all_questions'
FILES = {'unit.npy': ('<f4', '<f8'), 'log_size.npy': ('<f4', '<f8'), 'loss_weight.npy': ('<f8',), 'status.npy': ('<i4',)}
F8 = torch.float64


class Refused(ValueError):
    """The artifact is not what its format says, or its rows are not the caller's."""


class Unsupported(ValueError):
    """The artifact is sound and asks something the solver's one weight a row cannot hold."""


def of_tokens(tokens):
    """A token list's sha256, as the native readings name a prompt and a text: the JSON list of the integers, no spaces."""
    return hashlib.sha256(json.dumps([int(t) for t in tokens], separators=(',', ':')).encode()).hexdigest()


def _sha256(path):
    found = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 24), b''):
            found.update(block)
    return found.hexdigest()


def _need(held, names, where):
    missing = [name for name in names if name not in held]
    if missing:
        raise Refused(f'{where} lacks {missing}')


def _bound(root, entry, what):
    """A record the artifact is bound to: a path under the repository's root and the sha256 of its bytes."""
    _need(entry, ('record', 'sha256'), what)
    path = Path(entry['record'])
    if path.is_absolute() or '..' in path.parts:
        raise Refused(f"{what} names {entry['record']}: a record is named from the repository's root")
    if not (root/path).is_file():
        raise Refused(f"{what} names {entry['record']}, which is not there")
    if _sha256(root/path) != entry['sha256']:
        raise Refused(f"{entry['record']} is not the record {what} was bound to")
    return root/path


class Node:
    """One merge's targets: where every row's state should stand after it."""

    def __init__(self, targets, entry):
        _need(entry, ('merge', 'connection', 'layer', 'kind', 'width', 'space', 'files'), 'a node')
        self.targets, self.entry = targets, entry
        self.merge, self.connection, self.layer, self.kind, self.width, self.space = (entry[name] for name in ('merge', 'connection', 'layer', 'kind', 'width', 'space'))
        if sorted(entry['files']) != sorted(FILES):
            raise Refused(f'merge {self.merge} describes {sorted(entry["files"])}, not its four arrays')
        self._arrays = None

    def arrays(self):
        """The four arrays, each held to its descriptor and all four to the status rules: unit, log_size, loss_weight, status."""
        if self._arrays is not None:
            return self._arrays
        rows, found = self.targets.rows, {}
        shapes = {'unit.npy': [rows, self.width], 'log_size.npy': [rows], 'loss_weight.npy': [rows, 2], 'status.npy': [rows, 2]}
        for name, types in FILES.items():
            told = self.entry['files'][name]
            _need(told, ('file', 'sha256', 'shape', 'dtype'), f'merge {self.merge} {name}')
            if told['file'] != f'nodes/{self.merge}/{name}':
                raise Refused(f"merge {self.merge} keeps {name} at {told['file']}")
            if told['dtype'] not in types or list(told['shape']) != shapes[name]:
                raise Refused(f"merge {self.merge} {name} is declared {told['dtype']} {told['shape']}, and the format has {types} {shapes[name]}")
            path = self.targets.folder/told['file']
            if not path.is_file() or _sha256(path) != told['sha256']:
                raise Refused(f'merge {self.merge} {name} is not the file its descriptor names')
            held = np.load(path, mmap_mode='r', allow_pickle=False)
            if held.dtype.str != told['dtype'] or list(held.shape) != shapes[name]:
                raise Refused(f'merge {self.merge} {name} holds {held.dtype.str} {list(held.shape)}, not what its descriptor declares')
            found[name] = held
        unit, log_size, weight, status = (np.asarray(found[name]) for name in FILES)
        if bool((status < 0).any()):
            raise Refused(f'merge {self.merge} holds a negative status')
        unnamed = sorted(set(np.unique(status[status > 0]).tolist()) - {int(code) for code in self.targets.reasons})
        if unnamed:
            raise Refused(f'merge {self.merge} holds status {unnamed} with no reason')
        known = status == 0
        finite = np.stack([np.isfinite(unit).all(1), np.isfinite(log_size)], 1)
        missing = np.stack([np.isnan(unit).all(1), np.isnan(log_size)], 1)
        if bool((known & ~(finite & (weight > 0))).any()):
            raise Refused(f'merge {self.merge}: a resolved channel has a target that is not finite or a coefficient that is not positive')
        if bool((~known & ~(missing & (weight == 0))).any()):
            raise Refused(f'merge {self.merge}: an unknown channel has a target or a coefficient: unknown is NaN with a coefficient of exactly zero')
        self._arrays = dict(unit=unit, log_size=log_size, loss_weight=weight, status=status)
        return self._arrays

    def coverage(self):
        """How many rows' channels are resolved, and the unknown ones by reason: direction and size apart."""
        status = self.arrays()['status']
        told = lambda column: dict(resolved=int((column == 0).sum()), unknown={str(code): int((column == code).sum()) for code in np.unique(column[column > 0]).tolist()})
        return dict(rows=self.targets.rows, direction=told(status[:, 0]), size=told(status[:, 1]))

    def asked(self, space, device=None):
        """What the solver takes for this merge, for rows read in `space`: the targets as finite double-precision buffers with nothing in
        the unknown rows' places, one weight a row and which rows are known. A missing value is never multiplied: the buffers are filled
        from the resolved rows alone."""
        if space != self.space:
            raise Refused(f'merge {self.merge} holds targets in the space {self.space!r}, and the rows are read in {space!r}')
        held = self.arrays()
        known, weight = held['status'] == 0, held['loss_weight']
        one_channel = known[:, 0] != known[:, 1]
        both = known[:, 0] & known[:, 1]
        unequal = both & (weight[:, 0] != weight[:, 1])
        if bool(one_channel.any()) or bool(unequal.any()):
            raise Unsupported(f'merge {self.merge}: {int((known[:, 0] & ~known[:, 1]).sum())} rows know the direction alone, {int((~known[:, 0] & known[:, 1]).sum())} the size alone, '
                              f'and {int(unequal.sum())} weigh the two channels unequally; the solver holds one weight a row for both')
        unit = np.zeros(held['unit'].shape, dtype='f8')
        log_size = np.zeros(held['log_size'].shape, dtype='f8')
        root = np.zeros(len(both), dtype='f8')
        unit[both], log_size[both], root[both] = held['unit'][both], held['log_size'][both], np.sqrt(weight[both, 0])
        send = lambda values: torch.from_numpy(values).to(device)
        return dict(unit=send(unit), log_size=send(log_size), weight=send(root), known=send(both), coverage=self.coverage())


class Targets:
    """A target artifact: its manifest, its row map and its nodes, by merge."""

    def __init__(self, folder, manifest, sha256, row_map, schema):
        self.folder, self.manifest, self.sha256, self.row_map = folder, manifest, sha256, row_map
        self.rows, self.purpose, self.reasons = manifest['rows'], manifest['purpose'], manifest['status_reasons']
        self.answers = row_map['answers']
        self.nodes = {}
        for entry, merge in zip(manifest['nodes'], schema):
            node = Node(self, entry)
            if any(entry[name] != merge[name] for name in ('merge', 'connection', 'layer', 'kind', 'width', 'space')):
                raise Refused(f"node {entry['merge']} is not the observation schema's merge {merge['merge']} in its place")
            self.nodes[node.merge] = node
        if len(self.nodes) != len(schema) or len(manifest['nodes']) != len(schema):
            raise Refused(f"the artifact holds {len(manifest['nodes'])} nodes ({len(self.nodes)} apart) for the schema's {len(schema)} merges")
        self.matrix_nodes = manifest['matrix_nodes']
        for name, held in self.matrix_nodes.items():
            if 'own' not in held or set(held) - {'own', 'layer_final'} or any(merge not in self.nodes for merge in held.values()):
                raise Refused(f'{name} is mapped to {held}: a matrix has its own merge and, a feed-forward, its layer\'s last one, both of the artifact')

    def answer_of_rows(self, device=None):
        """Which answer every row belongs to, as a number: the answer's place in the row map."""
        out = torch.empty(self.rows, dtype=torch.long)
        for number, each in enumerate(self.answers):
            out[each['answer_rows_from']:each['answer_rows_from'] + each['steps']] = number
            out[each['question_rows_from']:each['question_rows_from'] + each['question']] = number
        return out.to(device)

    def held_to(self, answers, tokens, fitting):
        """The caller's rows are the artifact's: `answers` as a native row source records them (id, steps, question, answer_rows_from,
        question_rows_from, in its order), `tokens` each answer's prompt and text tokens ({id: (prompt, text)}), `fitting` the ids of the
        declared fitting answers in their order."""
        if [each['id'] for each in self.answers] != list(fitting):
            raise Refused('the artifact\'s answers are not the declared fitting answers in their order')
        if len(answers) != len(self.answers):
            raise Refused(f'the rows hold {len(answers)} answers and the artifact {len(self.answers)}')
        for mine, theirs in zip(self.answers, answers):
            for name in ('id', 'steps', 'question', 'answer_rows_from', 'question_rows_from'):
                if mine[name] != theirs[name]:
                    raise Refused(f"{mine['id']}: its {name} is {mine[name]} in the artifact and {theirs[name]} in the rows")
            prompt, text = tokens[mine['id']]
            if of_tokens(prompt) != mine['prompt_sha256'] or of_tokens(text) != mine['text_sha256']:
                raise Refused(f"{mine['id']}: the rows were read on other tokens than the artifact's targets were made for")
            if len(text) != mine['steps']:
                raise Refused(f"{mine['id']}: its text is {len(text)} tokens and its rows {mine['steps']} steps")

    def of_matrix(self, name):
        """A matrix's own node and, for a feed-forward, its layer's last one (or None)."""
        if name not in self.matrix_nodes:
            raise Refused(f'the artifact maps no merge to {name}')
        held = self.matrix_nodes[name]
        return self.nodes[held['own']], (self.nodes[held['layer_final']] if 'layer_final' in held else None)


def read(folder, root):
    """The artifact in `folder`, held to its format. `root` is the repository's root, which the bound records are named from."""
    folder, root = Path(folder), Path(root)
    raw = (folder/'target.json').read_bytes()
    manifest = json.loads(raw)
    _need(manifest, ('format', 'purpose', 'objective', 'factory', 'observation_schema', 'row_map', 'rows', 'status_reasons', 'nodes', 'matrix_nodes'), 'target.json')
    if manifest['format'] != FORMAT:
        raise Refused(f"the artifact's format is {manifest['format']!r}, not {FORMAT!r}")
    if manifest['purpose'] not in ('fixture', 'construction'):
        raise Refused(f"the artifact's purpose is {manifest['purpose']!r}")
    for what in ('objective', 'factory'):
        _bound(root, manifest[what], what)
    schema = json.loads(_bound(root, manifest['observation_schema'], 'observation_schema').read_text())
    _need(schema, ('merges',), 'the observation schema')
    for merge in schema['merges']:
        _need(merge, ('merge', 'connection', 'layer', 'kind', 'width', 'space'), 'a merge of the observation schema')
    if len({json.dumps(merge['merge']) for merge in schema['merges']}) != len(schema['merges']):
        raise Refused('the observation schema names a merge twice')
    _need(manifest['row_map'], ('file', 'sha256'), 'row_map')
    if manifest['row_map']['file'] != 'row-map.json' or _sha256(folder/'row-map.json') != manifest['row_map']['sha256']:
        raise Refused('row-map.json is not the row map the manifest names')
    row_map = json.loads((folder/'row-map.json').read_text())
    _need(row_map, ('order', 'rows', 'answers'), 'row-map.json')
    if row_map['order'] != ORDER or row_map['rows'] != manifest['rows']:
        raise Refused('the row map has another order or another count of rows than the manifest')
    answer_from, question_from = 0, sum(each['steps'] for each in row_map['answers'])
    seen = set()
    for each in row_map['answers']:
        _need(each, ('id', 'steps', 'question', 'answer_rows_from', 'question_rows_from', 'prompt_sha256', 'text_sha256'), 'an answer of the row map')
        if each['id'] in seen or each['steps'] < 0 or each['question'] < 0:
            raise Refused(f"{each['id']}: named twice, or with a negative length")
        if (each['answer_rows_from'], each['question_rows_from']) != (answer_from, question_from):
            raise Refused(f"{each['id']}: its rows do not lie where every answer's steps in order, then every answer's question positions, put them")
        seen.add(each['id'])
        answer_from, question_from = answer_from + each['steps'], question_from + each['question']
    if question_from != row_map['rows']:
        raise Refused(f"the answers hold {question_from} rows and the row map says {row_map['rows']}")
    for code, reason in manifest['status_reasons'].items():
        if not code.isdigit() or int(code) < 1 or str(int(code)) != code or not reason:
            raise Refused(f'status reason {code!r}: a reason is keyed by a positive whole number and says why')
    return Targets(folder, manifest, hashlib.sha256(raw).hexdigest(), row_map, schema['merges'])


def _asked(node, space, unit, log_size, take):
    """A merge's request at the rows `take` marks, which stand at `unit` and `log_size` after it as the matrix stands: where each should
    stand (`unit`, `log_size`), its weight and whether it asks (`known`). A merge whose reference is a point a row says so whatever the
    rows stand at (`asked`); one whose reference is a set a row finds each row's point where the row stands (`asked_where`,
    gap_requests)."""
    if hasattr(node, 'asked_where'):
        return node.asked_where(space, unit, log_size, take)
    held = node.asked(space, unit.device)
    return {name: held[name][take] for name in ('unit', 'log_size', 'weight', 'known')}


def matrix(targets, name, x, incoming, w, gain, space, last=None, rows=None):
    """field.Matrix for one matrix on a state's rows, its requests made from the artifact: at the matrix's own merge and, for a
    feed-forward, at its layer's last merge too. `x` and `incoming` are the matrix's input and the state its output is added to at every
    row, `w` and `gain` the matrix and the block's learned scale as the state's file holds them, `space` the observable the rows are read
    in, and `last` the layer's last merge as the model holds it (field.Final's gate, proj, scale, t and together). A request is the
    target less where the matrix as it stands leaves the row; an unknown row has no request and no weight, which is not a request for
    nothing. Where the reference is a set at every row, a row's target is the set's point nearest where the matrix as it stands leaves
    the row (`_asked`), found here once and held while the matrix is solved. `rows` marks the rows the matrix is given (all, when None): a row left unmarked is in no sum and no reading of it, as when
    some answers are kept out of a whole solve.

    Behind the last layer's feed-forward lies the model's last merge, which the artifact holds in a space of its own: what the output
    norm passes on of it. `last` then also holds `passes_on` (correction.terminal's: the norm's learned scale, the layer's output scale
    and the norm's epsilon) and `space`, the name of that observable, and the merge is read through field.Terminal. A last merge whose
    targets are in another space than the one its reading is in is refused."""
    from pipeline.weight_edit import field
    own, final = targets.of_matrix(name)
    take = torch.ones(x.shape[0], dtype=torch.bool, device=x.device) if rows is None else rows
    x, incoming8, gain8 = x[take], incoming.to(F8)[take], gain.to(F8)
    raw = torch.cat([(x[first:first + field.BLOCK] @ w).to(F8) for first in range(0, x.shape[0], field.BLOCK)])
    to, stands = field.left(incoming8, raw, gain8)
    asked = _asked(own, space, to, torch.log(stands), take)
    known = asked['known']
    direction, size = (asked['unit'] - to) * known[:, None], (asked['log_size'] - torch.log(stands)) * known
    del to, stands
    behind = None
    if final is not None:
        if last is None:
            raise Refused(f'{name}: its layer\'s last merge has targets, and the model\'s parts of that merge were not given')
        passes_on = last.get('passes_on')
        if (passes_on is None) != ('space' not in last):
            raise Refused(f'{name}: its last merge is given a space of its own or what the output norm passes on, and not both')
        parts = dict(gate=last['gate'], proj=last['proj'], scale=last['scale'], t=last['t'][take], together=last['together'][take])
        made = field.Final if passes_on is None else (lambda **given: field.Terminal(**given, passes_on=passes_on))
        nothing = torch.zeros_like(size)
        state = incoming8 + gain8 * raw / torch.sqrt((raw * raw).mean(1, keepdim=True) + field.NORM_EPSILON)
        to, stands = made(**parts, direction=torch.zeros_like(direction), size=nothing, weight=nothing).left(state, torch.arange(state.shape[0], device=state.device))
        there = _asked(final, last.get('space', space), to, torch.log(stands), take)
        turned, grown = there['unit'] - to, there['log_size'] - torch.log(stands)
        behind = made(**parts, direction=turned * there['known'][:, None], size=grown * there['known'], weight=there['weight'])
    return field.Matrix(x, incoming8, w, gain, direction, size, asked['weight'], targets.answer_of_rows(x.device)[take], behind)
