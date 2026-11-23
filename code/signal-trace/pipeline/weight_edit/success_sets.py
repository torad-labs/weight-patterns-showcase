"""Immutable fitting-success memberships on complete original native frames.

The sets contain individual native points, not donor or recipient means.  This
module never chooses a nearest point, reads an edited reference, or runs an engine.
Turn producers bind complete prompt/answer computation separately; no token-based
registration or missing-first-merge reconstruction substitutes for either domain.
"""
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.overlap.complete_overlap import distance_path, unit_steps, validate_pairs
from pipeline.weight_edit import context_targets as c
from pipeline.weight_edit.whole_context_score import NativeReading, _json, _layout, _local

FORMAT = "whole-gap-success-sets/1"
TURN_FORMAT = "whole-gap-reference-turns/1"
NODE_KEYS = ("merge", "connection", "layer", "kind", "width")
DOMAINS = (("answer", "steps", "answer_rows_from"), ("question", "question", "question_rows_from"))


def bound_path(root, record):
    """A named, hash-bound, non-linked file under the given repository root."""
    path = _local(Path(root).resolve(), record["record"])
    if c.sha256(path) != record["sha256"]:
        raise ValueError(f"Bound source changed: {record['record']}")
    return path


def labeled_answers(reading):
    """Join original cohort labels from the hash-bound native history table.

    Row maps describe positions and input identities, not outcomes.  The native
    table's group is the frozen original classification, never a new answer grade.
    """
    table = reading.record["table"]
    if [a["id"] for a in table] != [a["id"] for a in reading.answers]:
        raise ValueError("Original group table differs from the native row inventory")
    groups = {"correct": True, "wrong": False}
    answers = []
    for row, entry in zip(reading.answers, table):
        group = entry.get("group")
        if not isinstance(group, str) or group not in groups:
            raise ValueError(f"{row['id']}: original history table has no recognized group")
        correct = groups[group]
        if "correct" in row and (type(row["correct"]) is not bool or row["correct"] != correct):
            raise ValueError("Row-map outcome label conflicts with the bound original group")
        answers.append(dict(row, correct=correct))
    return answers


def schema_of(reading) -> list[dict[str, Any]]:
    nodes = [{key: node[key] for key in NODE_KEYS} for node in reading.record["merges"]]
    if not nodes or [n["merge"] for n in nodes] != list(range(len(nodes))):
        raise ValueError("Original merge inventory is empty, duplicated or reordered")
    layers = sorted({n["layer"] for n in nodes})
    expected = [(layer, kind) for layer in layers for kind in ("attention", "feedforward", "layer")]
    if layers != list(range(len(layers))) or [(n["layer"], n["kind"]) for n in nodes] != expected:
        raise ValueError("Reference needs every connected merge of every original layer")
    if reading.record["terminal"]["merge"] != nodes[-1]["merge"]:
        raise ValueError("Terminal is not the original path's last merge")
    if any(n["width"] != nodes[0]["width"] for n in nodes):
        raise ValueError("Residual width changed along the reference path")
    return [dict(n, space="output_norm" if n==nodes[-1] else "residual") for n in nodes]


def native_points(reading, entry, normalization):
    """The whole_context_score feature convention, including terminal evidence."""
    native = reading.record["merges"]
    merge = entry["merge"]
    if merge >= len(native) or any(native[merge][k] != entry[k] for k in NODE_KEYS):
        raise ValueError("Reading omits or changes a required native observable")
    supplied = native[merge]
    if supplied["space"] != "residual":
        raise ValueError("Native states must remain raw residual observations")
    raw = reading.array(supplied, f"nodes/{merge}/state.npy",
                        (reading.record["rows"], entry["width"]))
    state = np.asarray(raw, dtype="<f8")
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.log(np.linalg.norm(state, axis=1))
    terminal = None
    if entry["space"] == "output_norm":
        if merge != len(native)-1 or reading.record["terminal"]["merge"] != merge:
            raise ValueError("Only the last merge composes output normalization")
        unit, size = c.terminal_features(state, size, **normalization)
        held = reading.record["terminal"]
        l_out = reading.array(held["l_out"], "terminal/l_out.npy", raw.shape)
        engine = reading.array(held["result_norm"], "terminal/result_norm.npy",
                               (reading.answer_rows, entry["width"]))
        scaled = np.asarray(raw)*np.float32(normalization["scale"])
        reconstructed = unit[:reading.answer_rows]*np.exp(size[:reading.answer_rows, None])
        difference = np.abs(reconstructed-np.asarray(engine, dtype=np.float64))
        terminal = dict(
            engine_answer_rows=reading.answer_rows,
            derived_prompt_rows=reading.record["rows"]-reading.answer_rows,
            layer_scale_same_rows=int(np.all(scaled.view("<u4")==np.asarray(l_out).view("<u4"), axis=1).sum()),
            max_abs_state_difference_from_engine=float(difference.max()) if np.isfinite(difference).all() else None,
            prompt_output_norm="derived from native raw states, not recorded engine output",
        )
    elif entry["space"] == "residual":
        unit, size, _ = c.features(state, size)
    else:
        raise ValueError("Unknown reference feature convention")
    points = np.column_stack((unit, size))
    return points, np.isfinite(points).all(1), terminal


def _save_array(folder, name, values):
    path = folder/name
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return dict(file=name, sha256=c.sha256(path), shape=list(values.shape), dtype=values.dtype.str)


def _array(folder, descriptor, name, shape, dtype):
    if descriptor["file"] != name or descriptor["dtype"] != dtype or descriptor["shape"] != list(shape):
        raise ValueError("Set artifact array descriptor differs from its contract")
    path = _local(folder, name)
    if c.sha256(path) != descriptor["sha256"]:
        raise ValueError("Set artifact array bytes changed")
    values = np.load(path, allow_pickle=False)
    if values.shape != tuple(shape) or values.dtype.str != dtype:
        raise ValueError("Set artifact array header differs from its descriptor")
    return values


def _turn_record(reading, record, root):
    source, _ = _json(bound_path(root, record))
    expected = dict(format=TURN_FORMAT, reading_sha256=reading.sha256,
                    model_sha256=reading.record["model_sha256"], execution=reading.execution,
                    definition="unit(merged)-unit(incoming)",
                    merges=[{k:n[k] for k in NODE_KEYS} for n in reading.record["merges"]])
    if any(source.get(k) != value for k, value in expected.items()):
        raise ValueError("Turns are not bound to this complete original computation")
    if set(source["answers"]) != {a["id"] for a in reading.answers}:
        raise ValueError("Turn source changed the complete fitting membership")
    for answer in reading.answers:
        if set(source["answers"][answer["id"]]) != {"answer", "question"}:
            raise ValueError("Every member needs separate complete answer and prompt turns")
    return source


def _turns(source, answer, domain, count, schema, root):
    descriptor = source["answers"][answer["id"]][domain]
    expected = (count, len(schema), schema[0]["width"])
    if descriptor.get("shape") != list(expected) or descriptor.get("dtype") != "<f4":
        raise ValueError("Turn descriptor omits native rows, merges or coordinates")
    path = bound_path(root, descriptor)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != expected or values.dtype.str != "<f4":
        raise ValueError("Turn array header differs from its complete domain")
    return unit_steps(values) if count else np.empty((0, len(schema)*schema[0]["width"]), dtype="<f4")


def _memberships(answers, paths):
    """Validate every donor/member/domain path, then retain every mapped point."""
    rows = sum(a["steps"]+a["question"] for a in answers)
    donors = [a for a in answers if a["correct"]]
    if not donors or any(type(a["correct"]) is not bool for a in answers):
        raise ValueError("Reference requires original boolean labels and successful donors")
    if set(paths) != {a["id"] for a in answers}:
        raise ValueError("Correspondence omits a fitting recipient")
    points = [[] for _ in range(rows)]
    for recipient in answers:
        each = paths[recipient["id"]]
        if set(each) != {a["id"] for a in donors}:
            raise ValueError("Correspondence changed the fitting-success donor pool")
        for donor in donors:
            domains = each[donor["id"]]
            if set(domains) != {"answer", "question"}:
                raise ValueError("Correspondence lost an answer or prompt domain")
            for domain, length, offset in DOMAINS:
                pairs = np.asarray(domains[domain], dtype=np.int64) if not domains[domain] else np.asarray(domains[domain])
                if recipient[length] == donor[length] == 0:
                    if pairs.size:
                        raise ValueError("Empty prompt domain has unexpected pairs")
                    continue
                validate_pairs(pairs, donor[length], recipient[length])
                if recipient["id"] == donor["id"] and not np.array_equal(
                        pairs, np.column_stack([np.arange(donor[length])]*2)):
                    raise ValueError("A successful donor paired with itself keeps identity")
                for native, frame in pairs:
                    points[recipient[offset]+int(frame)].append(donor[offset]+int(native))
    if any(not values for values in points):
        raise ValueError("A reference frame has no successful native points")
    offsets = np.cumsum([0]+[len(values) for values in points], dtype=np.int64)
    return offsets.astype("<i8"), np.asarray([v for values in points for v in values], dtype="<i8")


def original_weights(answers, merges):
    counts = {flag: sum(a["correct"] == flag for a in answers) for flag in (False, True)}
    groups = sum(n > 0 for n in counts.values())
    result = np.empty(sum(a["steps"]+a["question"] for a in answers), dtype="<f8")
    for answer in answers:
        value = 1/(groups*counts[answer["correct"]]*(answer["steps"]+answer["question"])*merges)
        for _, length, offset in DOMAINS:
            result[answer[offset]:answer[offset]+answer[length]] = value
    return result


def compile_reference(folder, root, reading, turn_source, normalization, objective, provenance=None):
    """Compile immutable memberships; the declared command supplies original bindings.

    This small file API also serves fixtures. It does not admit an experiment or
    infer original-model identity from a score. No native feature is averaged.
    """
    folder, root = Path(folder), Path(root).resolve()
    if folder.exists():
        raise FileExistsError("Preserve a partial or finished successful-set artifact")
    if str(reading.record["visible"]) != "0":
        raise ValueError("Successful-set reference uses canonical card 0")
    schema = schema_of(reading)
    _layout(reading.row_map)
    bound_path(root, objective)
    source = _turn_record(reading, turn_source, root)
    answers = labeled_answers(reading)
    donors = [a for a in answers if a["correct"]]
    if not donors:
        raise ValueError("No original fitting successes")
    # Original turn arrays are consumed only here. Later readers use these frozen
    # memberships, not a regenerated mapping or a new hash taken after editing.
    paths: dict[str, Any] = {}
    for recipient in answers:
        paths[recipient["id"]] = {d["id"]: {} for d in donors}
        for domain, length, _ in DOMAINS:
            recipient_turns = _turns(source, recipient, domain, recipient[length], schema, root)
            for donor in donors:
                donor_turns = recipient_turns if donor["id"]==recipient["id"] else _turns(
                    source, donor, domain, donor[length], schema, root)
                if donor["id"] == recipient["id"]:
                    pairs = np.column_stack([np.arange(recipient[length])]*2)
                elif donor[length] == recipient[length] == 0:
                    pairs = np.empty((0, 2), dtype="<i8")
                else:
                    pairs = distance_path(donor_turns @ recipient_turns.T)
                paths[recipient["id"]][donor["id"]][domain] = pairs.tolist()
    offsets, donor_rows = _memberships(answers, paths)
    weight = np.asarray(normalization["weight"], dtype="<f8")
    # Validate the normalization's complete vector, positive scale and epsilon.
    c.terminal_features(np.ones((1, schema[0]["width"])), np.zeros(1), **normalization)
    folder.mkdir(parents=True)
    c.write_json(folder/"row-map.json", reading.row_map)
    c.write_json(folder/"paths.json", paths)
    manifest = dict(
        format=FORMAT, role="construction", original_model_sha256=reading.record["model_sha256"],
        group_source="original_reading.table.group",
        original_reading=c.binding(reading.folder/"reading.json", root),
        objective=objective, turn_source=turn_source, provenance=provenance,
        row_map=dict(file="row-map.json", sha256=c.sha256(folder/"row-map.json")),
        paths=dict(file="paths.json", sha256=c.sha256(folder/"paths.json")),
        rows=reading.record["rows"], donors=[a["id"] for a in donors], merges=schema,
        offsets=_save_array(folder, "offsets.npy", offsets),
        donor_rows=_save_array(folder, "donor-rows.npy", donor_rows),
        normalization=dict(weight=_save_array(folder, "output-norm.npy", weight),
                           scale=float(normalization["scale"]), eps=float(normalization["eps"])),
        weight_derivation="1/(represented groups * original group members * member prompt-plus-answer rows * merges)",
        representation="individual original native points in joint unit-direction/log-size space; mixtures are not renormalized",
    )
    c.write_json(folder/"manifest.json", manifest)
    return c.binding(folder/"manifest.json", root)


class Reference:
    """Hash-bound source points and CSR memberships shared by scorer and solver."""

    def __init__(self, folder, root):
        self.folder, self.root = Path(folder).resolve(), Path(root).resolve()
        self.manifest, self.sha256 = _json(self.folder/"manifest.json")
        m = self.manifest
        if m.get("format") != FORMAT or m.get("role") != "construction":
            raise ValueError("Not an original fitting-success set artifact")
        bound_path(self.root, m["objective"])
        path = bound_path(self.root, m["original_reading"])
        if path.name != "reading.json":
            raise ValueError("Original source is not a complete native reading")
        self.reading = NativeReading(path.parent)
        if self.reading.record["model_sha256"] != m["original_model_sha256"] or str(self.reading.record["visible"]) != "0":
            raise ValueError("Reference source is not its canonical original model")
        _turn_record(self.reading, m["turn_source"], self.root)
        self.row_map, _ = _json(_local(self.folder, "row-map.json"), m["row_map"]["sha256"])
        if self.row_map != self.reading.row_map:
            raise ValueError("Reference rows differ from the bound original fitting inventory")
        if m.get("group_source") != "original_reading.table.group":
            raise ValueError("Reference does not bind labels to the original history table")
        self.rows, self.answers = self.row_map["rows"], labeled_answers(self.reading)
        self.schema = schema_of(self.reading)
        if m["merges"] != self.schema or m["rows"] != self.rows:
            raise ValueError("Reference lost or changed a source merge or row")
        self.donors = [a["id"] for a in self.answers if a["correct"]]
        if m["donors"] != self.donors:
            raise ValueError("Reference donor pool is not exactly the fitting successes")
        self.paths, _ = _json(_local(self.folder, "paths.json"), m["paths"]["sha256"])
        offsets, donor_rows = _memberships(self.answers, self.paths)
        self.offsets = _array(self.folder, m["offsets"], "offsets.npy", offsets.shape, "<i8")
        self.donor_rows = _array(self.folder, m["donor_rows"], "donor-rows.npy", donor_rows.shape, "<i8")
        if not np.array_equal(offsets, self.offsets) or not np.array_equal(donor_rows, self.donor_rows):
            raise ValueError("Memberships differ from their complete donor-to-recipient paths")
        self.weights = original_weights(self.answers, len(self.schema))
        normal = m["normalization"]
        self.normalization = dict(
            weight=_array(self.folder, normal["weight"], "output-norm.npy", (self.schema[0]["width"],), "<f8"),
            scale=normal["scale"], eps=normal["eps"])
        c.terminal_features(np.ones((1, self.schema[0]["width"])), np.zeros(1), **self.normalization)

    def points(self, entry):
        return native_points(self.reading, entry, self.normalization)
