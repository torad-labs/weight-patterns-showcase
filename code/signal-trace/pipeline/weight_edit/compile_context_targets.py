"""Compile the two prospectively declared original-context full fields.

Run with the project's scientific environment:
  python -m pipeline.weight_edit.compile_context_targets <run folder>

Only original fitting slices are read. No engine, GPU, model write, no-fit
evaluation, candidate ranking or hidden fallback is part of this program.
"""
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.native_records.native_trace import checkpoint_values
from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROOT
from pipeline.weight_edit import context_targets as c


def read_json(path):
    return json.loads(Path(path).read_text())


def selected_rows(members, fitting):
    """Selected answer rows followed by selected question rows, never all data."""
    total = sum(m["steps"] for m in members)
    answer, question, offsets = [], [], []
    a, q = 0, total
    for member, use in zip(members, fitting):
        if use:
            answer.extend(range(a, a + member["steps"]))
            question.extend(range(q, q + member["question"]))
            offsets.append(dict(id=member["id"], answer_rows_from=a, question_rows_from=q,
                                steps=member["steps"], question=member["question"]))
        a += member["steps"]
        q += member["question"]
    return np.asarray(answer + question, dtype=np.int64), offsets, q


def read_selected(path, rows, total, width, root=ROOT):
    """Map only the declared file, then copy only fitting rows into arithmetic."""
    path = Path(path)
    expected = total*width*np.dtype("<f4").itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path.name}: native source byte count differs from inventory")
    mapped = np.memmap(path, dtype="<f4", mode="r", shape=(total, width))
    selected = np.array(mapped[rows], copy=True)
    import hashlib
    binding = dict(file=str(path.relative_to(root)), source_bytes=expected,
                   selected_shape=list(selected.shape), selected_dtype=selected.dtype.str,
                   selected_sha256=hashlib.sha256(selected.tobytes()).hexdigest())
    return selected, binding


def coefficient_record(members, keys):
    correct = [m for m in members if m["correct"]]
    contexts = [[1] + [m["parameters"][k] for k in keys] for m in correct]
    coefficients = {"affine": {}, "mates": {}}
    details = {}
    for member in members:
        if member["correct"]:
            continue
        alpha = c.affine_weights(contexts, [1] + [member["parameters"][k] for k in keys])
        mates = c.mate_weights([m["parameters"] for m in correct], member["parameters"])
        coefficients["affine"][member["id"]] = alpha
        coefficients["mates"][member["id"]] = mates
        def record(values):
            if values is None:
                return dict(status="outside_affine_span")
            return dict(status="resolved", exact=[str(v) for v in values],
                        float64=[float(v) for v in values], sum=str(sum(values)),
                        leverage=str(sum(v*v for v in values)),
                        negative=sum(v < 0 for v in values))
        details[member["id"]] = dict(context=member["parameters"], affine=record(alpha), mates=record(mates))
    # Metadata-only leave-one-success-out support. No reconstruction claim is
    # made without reading its full field; these diagnostics select nothing.
    left_out = {}
    for i, member in enumerate(correct):
        rows = contexts[:i] + contexts[i+1:]
        alpha = c.affine_weights(rows, contexts[i])
        left_out[member["id"]] = dict(supported=alpha is not None,
                                     coefficients=None if alpha is None else [str(v) for v in alpha])
    return coefficients, dict(parameter_keys=keys, donors=[m["id"] for m in correct],
                              wrong=details, leave_one_success_out_context_support=left_out)


def describe(values, unit, size, members):
    """Per-member complete-domain discrepancy and independent channel coverage."""
    u, s, _ = c.features(unit, size)
    result = {}
    mapping = c.row_map(members, {})
    for member, entry in zip(members, mapping["answers"]):
        spans = [slice(entry["answer_rows_from"], entry["answer_rows_from"] + entry["steps"]),
                 slice(entry["question_rows_from"], entry["question_rows_from"] + entry["question"])]
        direction = magnitude = 0.
        resolved = np.zeros(2, dtype=np.int64)
        unknown = np.zeros(2, dtype=np.int64)
        for span in spans:
            known = values["status"][span] == 0
            resolved += known.sum(0)
            unknown += (~known).sum(0)
            delta = u[span][known[:, 0]] - values["unit"][span][known[:, 0]]
            log_delta = s[span][known[:, 1]] - values["log_size"][span][known[:, 1]]
            direction += float(np.sum(values["loss_weight"][span][known[:, 0], 0]*np.sum(delta*delta, axis=1)))
            magnitude += float(np.sum(values["loss_weight"][span][known[:, 1], 1]*log_delta**2))
        if not np.isfinite(direction + magnitude):
            raise ValueError("Original comparison is undefined despite a resolved target")
        result[member["id"]] = dict(correct=member["correct"], direction=direction, size=magnitude,
                                    resolved=resolved.tolist(), unknown=unknown.tolist())
    return result


def compile_run(run):
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT/"studies/04-round-2-overlap/runs"):
        raise ValueError("Run must live under the round-two runs root")
    declaration = check_native_run(run, run/"declaration.json")
    if declaration["method_step"] != 3 or declaration.get("operation") != "original-context-targets":
        raise ValueError("This compiler only implements the source-only step-3 declaration")
    if any((run/name).exists() for name in ("affine", "mates", "factory.json", "result.json")):
        raise FileExistsError("Compilation already started here; preserve partial records")
    sources = {key: ROOT/value for key, value in declaration["sources"].items()}
    for key, path in sources.items():
        if not any(path == (ROOT/entry) or path.is_relative_to(ROOT/entry) for entry in declaration["inputs"]):
            raise ValueError(f"Source {key} is not a declared input")
    for record in declaration["implementation"]:
        if c.binding(ROOT/record["record"], ROOT) != record:
            raise ValueError(f"Compiler source changed since declaration: {record['record']}")
    from pipeline.weight_edit import tube
    native = read_json(sources["native_rows"])
    registered = read_json(sources["registration"]/"members.json")
    wide = read_json(sources["members"])
    if len({m["id"] for m in wide}) != len(wide):
        raise ValueError("Duplicate original member")
    by_id = {m["id"]: m for m in wide}
    if [m["id"] for m in registered["members"]] != [m["id"] for m in native["answers"]]:
        raise ValueError("Registration and native source order differ")
    all_members = []
    for member, native_row in zip(registered["members"], native["answers"]):
        original = by_id[member["id"]]
        if member["correct"] != native_row["correct"]:
            raise ValueError("Original group labels differ")
        all_members.append(dict(member, steps=native_row["steps"], question=native_row["question_positions"],
                                tokens=original["tokens"], recorded=original["recorded_tokens"],
                                parameters=original["parameters"]))
    fitting, folds = tube.picks(all_members)
    members = [m for m, use in zip(all_members, fitting) if use]
    rows, source_spans, total = selected_rows(all_members, fitting)
    keys = declaration["context_parameters"]
    if any(set(m["parameters"]) != set(keys) for m in members):
        raise ValueError("Question parameter inventory differs from declaration")
    baseline = read_json(sources["baseline"]/"index.json")
    if baseline["read_through_sha256"] != declaration["model_sha256"] or baseline["passes_read"] != "as generated":
        raise ValueError("Not the declared original native computation")
    if c.sha256(sources["model"]) != declaration["model_sha256"]:
        raise ValueError("Original model identity changed")
    tensors = read_json(sources["tensors"])
    weight = checkpoint_values(sources["model"], tensors["output_norm.weight"]).astype("<f8").ravel()
    width = len(weight)
    terminal = read_json(sources["terminal"])["terminal"]
    last = native["layers"][-1]
    scale_tensor = f"blk.{last}.layer_output_scale.weight"
    scale = checkpoint_values(sources["model"], tensors[scale_tensor]).astype("<f8").ravel()
    import hashlib
    if hashlib.sha256(weight.tobytes()).hexdigest() != terminal["output_norm_sha256"]:
        raise ValueError("Output normalization differs from original reference")
    if scale.shape != (1,) or float(scale[0]) != terminal["layer_output_scale"][scale_tensor]:
        raise ValueError("Last layer scale differs from original reference")
    schema = [dict(merge=i, connection=f'{tube.OUTPUT[node["kind"]]}-{node["layer"]}',
                   layer=node["layer"], kind=node["kind"], width=width,
                   space="output_norm" if node["layer"] == last and node["kind"] == "layer" else "residual")
              for i, node in enumerate(baseline["merges"])]
    expected = [(layer, kind) for layer in native["layers"] for kind in tube.MERGES]
    if [(n["layer"], n["kind"]) for n in schema] != expected or any(n["rows"] != total for n in baseline["merges"]):
        raise ValueError("Baseline is not the full native merge/row inventory")
    # Logical arrays plus their measured NumPy header lengths; no guessed
    # residency cap. JSON overhead is checked again before each node.
    from io import BytesIO
    logical = 0
    for shape, dtype in [((len(rows), width), "<f8"), ((len(rows),), "<f8"), ((len(rows), 2), "<f8"), ((len(rows), 2), "<i4")]:
        header = BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(descr=np.dtype(dtype).str, fortran_order=False, shape=shape))
        logical += int(np.prod(shape))*np.dtype(dtype).itemsize + header.tell()
    needed = len(schema)*len(("affine", "mates"))*logical
    floor = declaration["parameters"]["disk_floor"]["value_bytes"]
    if shutil.disk_usage(run).free - needed < floor:
        raise ValueError(f"Target arrays need {needed} bytes above the declared free-space floor")
    pairs = [np.load(sources["registration"]/"pairs"/(m["id"] + ".npy"), allow_pickle=False) for m in members]
    prompt_places, prompts, prompt_record = tube.prompt_places(members)
    places = registered["axis_steps"]
    coefficients, context_record = coefficient_record(members, keys)
    bindings = {key: c.binding(path, ROOT) for key, path in sources.items() if path.is_file() and key != "model"}
    bindings["baseline_index"] = c.binding(sources["baseline"]/"index.json", ROOT)
    bindings["registration_members"] = c.binding(sources["registration"]/"members.json", ROOT)
    bindings["registration_pairs"] = {m["id"]: c.binding(sources["registration"]/"pairs"/(m["id"] + ".npy"), ROOT) for m in members}
    row_sources = dict(inputs=bindings, fitting_rule="tube.picks: every other member within original outcome group",
                       selected_original_spans=source_spans,
                       fitting=[m["id"] for m in members],
                       folds=[[m["id"] for m, use in zip(all_members, fold) if use] for fold in folds])
    mapping = c.row_map(members, row_sources)
    c.write_json(run/"observation-schema.json", dict(merges=schema, native_sources=bindings, terminal=terminal))
    c.write_json(run/"coefficients.json", context_record)
    factory = dict(declaration=c.binding(run/"declaration.json", ROOT),
                   objective=c.binding(ROOT/declaration["objective"], ROOT),
                   implementation=declaration["implementation"], sources=bindings,
                   model_sha256=declaration["model_sha256"], rows=row_sources,
                   answer_registration=dict(places=places, source=bindings["registration_members"]),
                   prompt_registration=dict(places=prompt_places, record=prompt_record,
                                            pairs={m["id"]: p.tolist() for m, p in zip(members, prompts)}),
                   coefficients=c.binding(run/"coefficients.json", ROOT),
                   representation="float64 derived units/sizes/weights; int32 status; selected original float32 slice hashes",
                   coverage="Every fitting answer and question row, every native merge. Unknown channels retain full denominator.",
                   historical_registration="The existing original complete registration remains reference evidence, not new no-fit calibration.",
                   logical_array_bytes_including_headers=needed)
    c.write_json(run/"factory.json", factory)
    writers = {name: c.ArtifactWriter(run/name, ROOT, mapping, schema, factory["objective"],
                                     c.binding(run/"factory.json", ROOT), c.binding(run/"observation-schema.json", ROOT), "construction")
               for name in coefficients}
    reports = {name: [] for name in writers}
    for node in schema:
        if shutil.disk_usage(run).free - len(writers)*logical < floor:
            raise ValueError("Compilation stopped before crossing declared disk floor; partial data preserved")
        stem = f'{node["kind"]}-{node["layer"]}.f32'
        unit, unit_binding = read_selected(sources["baseline"]/("state-" + stem), rows, total, width)
        size, size_binding = read_selected(sources["baseline"]/("size-" + stem), rows, total, 1)
        size = size[:, 0]
        if node["space"] == "output_norm":
            unit, size = c.terminal_features(unit, size, weight, float(scale[0]), terminal["eps"])
        for name, writer in writers.items():
            values = c.node_targets(members, pairs, prompts, places, prompt_places, unit, size, coefficients[name], len(schema))
            writer.add(node, values)
            reports[name].append(dict(merge=node["merge"], members=describe(values, unit, size, members)))
            del values
        c.write_json(run/"source-slices"/f'{node["merge"]}.json', dict(merge=node["merge"], unit=unit_binding, log_size=size_binding,
                                                                    row_spans=source_spans))
        print(json.dumps(dict(merge=node["merge"], of=len(schema), status="written")), flush=True)
    # A completion record comes only after all source-counted nodes were emitted.
    targets = {name: writer.finish() for name, writer in writers.items()}
    summary = {}
    for name, report in reports.items():
        c.write_json(run/(name + "-readings.json"), dict(nodes=report))
        totals: dict[str, Any] = {m["id"]: dict(correct=m["correct"], direction=0., size=0., resolved=[0, 0], unknown=[0, 0]) for m in members}
        for node in report:
            for key, values in node["members"].items():
                for channel in ("direction", "size"):
                    totals[key][channel] += values[channel]
                for channel in ("resolved", "unknown"):
                    totals[key][channel] = [a + b for a, b in zip(totals[key][channel], values[channel])]
        summary[name] = dict(target=targets[name], members=totals,
                             direction=sum(v["direction"] for v in totals.values()), size=sum(v["size"] for v in totals.values()),
                             unknown=np.sum([v["unknown"] for v in totals.values()], axis=0).tolist())
    c.write_json(run/"result.json", dict(
        prediction_vs_result=dict(targets=summary, merges=len(schema), rows=len(rows), members=len(members),
                                  interpretation="Two constructed original-context fields, not a repair result; numerical context support is not native realizability."),
        failures_only=dict(unknown={name: s["unknown"] for name, s in summary.items()},
                           limitations=["Native corrected-model computation and no-fit transfer have not run.",
                                        "Affine and mate-context transport and complete DTW correspondence remain hypotheses.",
                                        "Leave-one-success-out trace reconstruction was not scored; context-span support alone is reported."]),
        what_this_changes="The contextual field is explicit and serialized. This compilation selects no candidate or ridge and authorizes no model run."))
    print(json.dumps(dict(status="complete", merges=len(schema), rows=len(rows), targets=targets)), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m pipeline.weight_edit.compile_context_targets <run folder>")
    compile_run(sys.argv[1])
