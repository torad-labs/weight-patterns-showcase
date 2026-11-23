"""Original-context absolute targets; no model execution or matrix selection.

The prospective objective is records/retrospectives/2026-11-14-contextual-targets-astra.md.
Numerical algebra, complete registration, explicit unknown channels and immutable
whole-gap-targets/1 writing live here. The source-only run is compile_context_targets.
"""
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.overlap.complete_overlap import member_means, native_targets, validate_pairs

OUTSIDE_SPAN, UNDEFINED_DIRECTION, UNDEFINED_SIZE = 1, 2, 3
REASONS = {
    "1": "Question context is outside the exact affine span of the fitting-success contexts.",
    "2": "Direction has a zero/nonfinite resultant or depends on an undefined donor direction.",
    "3": "Log size is nonfinite or depends on an undefined donor size.",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return dict(record=str(path.relative_to(root)), sha256=sha256(path))


def write_json(path, value):
    """Publish a new record atomically; never replace an existing record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    # Hard-link creation refuses concurrent publication rather than replacing it.
    path.hardlink_to(temporary)
    temporary.unlink()


def _fraction(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _reduce(rows, columns):
    """Exact Gauss-Jordan elimination; trailing columns are right-hand sides."""
    rows = [list(row) for row in rows]
    pivot = 0
    for column in range(columns):
        found = next((i for i in range(pivot, len(rows)) if rows[i][column]), None)
        if found is None:
            continue
        rows[pivot], rows[found] = rows[found], rows[pivot]
        scale = rows[pivot][column]
        rows[pivot] = [x / scale for x in rows[pivot]]
        for i in range(len(rows)):
            if i != pivot and rows[i][column]:
                factor = rows[i][column]
                rows[i] = [a - factor*b for a, b in zip(rows[i], rows[pivot])]
        pivot += 1
    return rows, pivot


def affine_weights(contexts, target):
    """Minimum-norm exact affine weights, or None for an unsupported context."""
    if not contexts or any(len(row) != len(target) for row in contexts):
        raise ValueError("Context rows must have the target's feature count")
    x = [[_fraction(v) for v in row] for row in contexts]
    y = [_fraction(v) for v in target]
    n = len(x)
    equations = [[row[j] for row in x] + [y[j]] for j in range(len(y))]
    reduced, rank = _reduce(equations, n)
    if any(row[-1] for row in reduced[rank:]):
        return None
    b, rhs = [r[:n] for r in reduced[:rank]], [r[-1] for r in reduced[:rank]]
    gram = [[sum(a*v for a, v in zip(left, right)) for right in b] + [value]
            for left, value in zip(b, rhs)]
    solved, gram_rank = _reduce(gram, rank)
    if gram_rank != rank:
        raise ValueError("Independent exact constraints have a singular Gram matrix")
    alpha = [sum(b[j][i]*solved[j][-1] for j in range(rank)) for i in range(n)]
    if any(sum(alpha[i]*x[i][j] for i in range(n)) != y[j] for j in range(len(y))):
        raise ValueError("Affine constraints did not reconstruct the exact context")
    return alpha


def mate_weights(contexts, target):
    """Balance all maximal-match donors by their changed-parameter sets."""
    if not contexts or any(set(row) != set(target) for row in contexts):
        raise ValueError("Mates must have exactly the declared question parameters")
    changed = [tuple(k for k in sorted(target) if _fraction(row[k]) != _fraction(target[k]))
               for row in contexts]
    least = min(map(len, changed))
    groups = defaultdict(list)
    for i, keys in enumerate(changed):
        if len(keys) == least:
            groups[keys].append(i)
    alpha = [Fraction(0) for _ in contexts]
    for indices in groups.values():
        for i in indices:
            alpha[i] = Fraction(1, len(groups)*len(indices))
    return alpha


def features(unit, log_size):
    """Float64 unit directions and sizes with independent undefined channels."""
    unit, log_size = np.array(unit, dtype="<f8", copy=True), np.array(log_size, dtype="<f8", copy=True).reshape(-1)
    if unit.ndim != 2 or len(unit) != len(log_size):
        raise ValueError("Feature row counts differ")
    norms = np.linalg.norm(unit, axis=1)
    valid = np.isfinite(unit).all(1) & np.isfinite(norms) & (norms > 0)
    unit[valid] /= norms[valid, None]
    unit[~valid] = np.nan
    status = np.zeros((len(unit), 2), dtype="<i4")
    status[~valid, 0] = UNDEFINED_DIRECTION
    status[~np.isfinite(log_size), 1] = UNDEFINED_SIZE
    log_size[status[:, 1] != 0] = np.nan
    return unit, log_size, status


def terminal_features(unit, log_size, weight, scale, eps):
    """correction.passed_on in float64, retaining the declared epsilon term."""
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(eps) or eps < 0:
        raise ValueError("Invalid original terminal scale or epsilon")
    unit, log_size, _ = features(unit, log_size)
    weight = np.asarray(weight, dtype="<f8")
    if weight.shape != (unit.shape[1],) or not np.isfinite(weight).all():
        raise ValueError("Invalid original output normalization weights")
    turned = unit * weight
    norm = np.linalg.norm(turned, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        # logaddexp avoids exp(-2*s) overflow without adding a numerical floor.
        log_epsilon = np.log(unit.shape[1]*eps) - 2*np.log(scale) - 2*log_size
        size = np.log(norm) + .5*np.log(unit.shape[1]) - .5*np.logaddexp(0, log_epsilon)
    return features(turned, size)[:2]


def combine(alpha, fields):
    """Do not let a zero-coefficient donor's unknown channel poison a target."""
    if len(alpha) != len(fields):
        raise ValueError("Donor coefficient count differs from the fields")
    result = np.zeros_like(fields[0], dtype="<f8")
    for coefficient, field in zip(alpha, fields):
        if coefficient:
            result += float(coefficient)*field
    return result


def row_map(members, sources) -> dict[str, Any]:
    answers, answer_from = [], 0
    question_from = sum(m["steps"] for m in members)
    seen = set()
    for member in members:
        if member["id"] in seen:
            raise ValueError("Duplicate member")
        seen.add(member["id"])
        if len(member["tokens"]) != member["question"] + 1 or len(member["recorded"]) != member["steps"]:
            raise ValueError("Tokens differ from the native row inventory")
        tokens_hash = lambda v: hashlib.sha256(json.dumps([int(t) for t in v], separators=(",", ":")).encode()).hexdigest()
        answers.append(dict(id=member["id"], steps=member["steps"], question=member["question"],
                            answer_rows_from=answer_from, question_rows_from=question_from,
                            prompt_sha256=tokens_hash(member["tokens"]), text_sha256=tokens_hash(member["recorded"]),
                            correct=member["correct"], sources=sources))
        answer_from += member["steps"]
        question_from += member["question"]
    return dict(order="all_answers_then_all_questions", rows=question_from, answers=answers, sources=sources)


def node_targets(members, pairs, prompts, places, prompt_places, unit, size, coefficients, merges):
    """One whole merge: correct identities and contextual wrong fields."""
    if len(pairs) != len(members) or len(prompts) != len(members) or merges < 1:
        raise ValueError("Incomplete correspondence or merge inventory")
    mapping = row_map(members, {})
    original_u, original_s, _ = features(unit, size)
    if len(original_u) != mapping["rows"]:
        raise ValueError("Native feature count differs from the row inventory")
    u, s = original_u.copy(), original_s.copy()
    spans = []
    for m, a, p, q in zip(members, mapping["answers"], pairs, prompts):
        validate_pairs(p, m["steps"], places)
        validate_pairs(q, m["question"], prompt_places)
        spans.append((slice(a["answer_rows_from"], a["answer_rows_from"] + a["steps"]),
                      slice(a["question_rows_from"], a["question_rows_from"] + a["question"])))
    donors = [i for i, m in enumerate(members) if m["correct"]]
    if not donors:
        raise ValueError("No fitting successes")
    counts = {flag: sum(m["correct"] == flag for m in members) for flag in (False, True)}
    groups = sum(count > 0 for count in counts.values())
    weight = np.empty((mapping["rows"], 2), dtype="<f8")
    outside = np.zeros(mapping["rows"], dtype=bool)
    for segment, (correspondence, length) in enumerate(((pairs, places), (prompts, prompt_places))):
        donor_u = [member_means(original_u[spans[j][segment]], correspondence[j], length) for j in donors]
        donor_s = [member_means(original_s[spans[j][segment]], correspondence[j], length) for j in donors]
        for i, m in enumerate(members):
            span = spans[i][segment]
            weight[span] = 1 / (groups*counts[m["correct"]]*(m["steps"] + m["question"])*merges)
            if m["correct"]:
                continue
            alpha = coefficients[m["id"]]
            if alpha is None:
                u[span], s[span], outside[span] = np.nan, np.nan, True
                continue
            n = m["steps"] if segment == 0 else m["question"]
            u[span] = native_targets(combine(alpha, donor_u), correspondence[i], n)
            s[span] = native_targets(combine(alpha, donor_s), correspondence[i], n)
    # Normalize constructed wrong fields only: keep each correct member's
    # already re-expressed original feature exactly, not another round of it.
    status = np.zeros((mapping["rows"], 2), dtype="<i4")
    for i, m in enumerate(members):
        for span in spans[i]:
            if not m["correct"]:
                u[span], s[span], status[span] = features(u[span], s[span])
            else:
                status[span, 0] = np.where(np.isfinite(u[span]).all(1), 0, UNDEFINED_DIRECTION)
                status[span, 1] = np.where(np.isfinite(s[span]), 0, UNDEFINED_SIZE)
    status[outside] = OUTSIDE_SPAN
    weight[status != 0] = 0
    return dict(unit=u, log_size=s, loss_weight=weight, status=status)


class ArtifactWriter:
    """Write immutable nodes; publish target.json only for the full inventory."""

    def __init__(self, folder, root, rows, schema, objective, factory, observation_schema, purpose):
        self.folder, self.root, self.schema = Path(folder), Path(root), schema
        self.folder.mkdir(parents=True, exist_ok=False)
        write_json(self.folder/"row-map.json", rows)
        matrices = {}
        for node in schema:
            if node["kind"] == "attention":
                matrices[f'blk.{node["layer"]}.attn_output.weight'] = dict(own=node["merge"])
            elif node["kind"] == "feedforward":
                final = [n for n in schema if n["layer"] == node["layer"] and n["kind"] == "layer"]
                if len(final) != 1:
                    raise ValueError("Feed-forward has no unique layer-final observable")
                matrices[f'blk.{node["layer"]}.ffn_down.weight'] = dict(own=node["merge"], layer_final=final[0]["merge"])
        self.manifest: dict[str, Any] = dict(format="whole-gap-targets/1", purpose=purpose, objective=objective, factory=factory,
                             observation_schema=observation_schema, row_map=dict(file="row-map.json", sha256=sha256(self.folder/"row-map.json")),
                             rows=rows["rows"], status_reasons=REASONS, nodes=[], matrix_nodes=matrices)

    def add(self, node, values):
        folder = self.folder/"nodes"/str(node["merge"])
        folder.mkdir(parents=True, exist_ok=False)
        files = {}
        for name, dtype in (("unit", "<f8"), ("log_size", "<f8"), ("loss_weight", "<f8"), ("status", "<i4")):
            data = np.asarray(values[name], dtype=dtype)
            path = folder/(name + ".npy")
            with path.open("xb") as stream:
                np.save(stream, data, allow_pickle=False)
            files[path.name] = dict(file=str(path.relative_to(self.folder)), sha256=sha256(path), shape=list(data.shape), dtype=data.dtype.str)
        entry = dict(node, files=files)
        # Validate using the consumer without loading or retaining other nodes.
        from pipeline.weight_edit.gap_targets import Node
        from types import SimpleNamespace
        target = SimpleNamespace(folder=self.folder, rows=self.manifest["rows"], reasons=REASONS)
        Node(target, entry).arrays()
        self.manifest["nodes"].append(entry)

    def finish(self):
        if [{k: n[k] for k in ("merge", "connection", "layer", "kind", "width", "space")}
                for n in self.manifest["nodes"]] != self.schema:
            raise ValueError("Cannot publish an incomplete or reordered target inventory")
        write_json(self.folder/"target.json", self.manifest)
        return binding(self.folder/"target.json", self.root)
