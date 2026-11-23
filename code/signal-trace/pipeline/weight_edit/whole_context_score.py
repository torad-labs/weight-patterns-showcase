"""Bind whole-gap-reading/1 native states to a complete contextual objective.

No capture, solve or model write occurs here. A supplied changed-history mapping
must be bound to both manifests; this module does not infer semantic alignment.
The command-line operation is an original-only reader/scorer qualification.
"""
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.native_records.native_trace import checkpoint_values
from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROOT
from pipeline.weight_edit import context_targets as c, gap_targets
from pipeline.weight_edit.context_score import own_answer, paired_loss
from pipeline.overlap.complete_overlap import validate_pairs


def _json(path, expected=None):
    raw = Path(path).read_bytes()
    found = hashlib.sha256(raw).hexdigest()
    if expected is not None and found != expected:
        raise ValueError(f"Record hash differs: {path}")
    return json.loads(raw), found


def _local(folder, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("A reading file must be relative to its artifact")
    path = folder/relative
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError(f"Reading file missing or linked: {name}")
    return path


def _layout(rows):
    if rows.get("order") != "all_answers_then_all_questions":
        raise ValueError("Unsupported native row ordering")
    answers = rows["answers"]
    first, question = 0, sum(a["steps"] for a in answers)
    seen = set()
    for a in answers:
        if a["id"] in seen or Path(a["id"]).name != a["id"]:
            raise ValueError("Duplicate or unsafe native member id")
        seen.add(a["id"])
        if a["steps"] < 1 or a["question"] < 0:
            raise ValueError("Invalid native member lengths")
        if (a["answer_rows_from"], a["question_rows_from"]) != (first, question):
            raise ValueError("Native row map has a hole, overlap or reordered span")
        first += a["steps"]
        question += a["question"]
    if question != rows["rows"]:
        raise ValueError("Native row count differs from its complete inventory")
    return first


class NativeReading:
    """A complete direct-state capture, held to its own immutable descriptors."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.record, self.sha256 = _json(self.folder/"reading.json")
        r = self.record
        if r.get("format") != "whole-gap-reading/1":
            raise ValueError("Not a whole-gap direct-state reading")
        identity = ("model_sha256", "recorder_sha256", "libraries_sha256", "card", "driver",
                    "visible", "fusion_disabled", "recorder_arguments")
        if any(key not in r or r[key] in (None, "") for key in identity):
            raise ValueError("Native reading lacks execution identity")
        self.execution = {key: r[key] for key in identity if key != "model_sha256"}
        if r["row_map"]["file"] != "row-map.json":
            raise ValueError("Native row map has an unexpected name")
        self.row_map, _ = _json(_local(self.folder, "row-map.json"), r["row_map"]["sha256"])
        self.answer_rows = _layout(self.row_map)
        if r["rows"] != self.row_map["rows"]:
            raise ValueError("Native reading and row map disagree")
        self.answers = self.row_map["answers"]
        ids = [a["id"] for a in self.answers]
        if r["answers_read"] != ids or [a["id"] for a in r["table"]] != ids:
            raise ValueError("Native history inventory omits or reorders a member")
        self.histories = {}
        for a, table in zip(self.answers, r["table"]):
            answer, _ = _json(_local(self.folder, f'answers/{a["id"]}/answer.json'), table["answer_json_sha256"])
            own_answer(answer, table["limit"])
            if table["the_whole_answer_is_this_models_own"] is not True:
                raise ValueError("Reading is not a complete own answer")
            if any(answer[key] != a[key] for key in ("id", "steps", "question", "prompt_sha256", "text_sha256")):
                raise ValueError("Answer evidence and native row map disagree")
            if gap_targets.of_tokens(answer["tokens"]) != a["text_sha256"]:
                raise ValueError("Answer token bytes differ from the native row identity")
            self.histories[a["id"]] = answer
        if r["answers_whose_text_is_not_this_models_own"] != 0 or r["answers_whose_ending_is_unresolved"]:
            raise ValueError("Capture contains a conditional or incomplete answer")

    def array(self, descriptor, name, shape):
        if descriptor["file"] != name or descriptor["shape"] != list(shape) or descriptor["dtype"] != "<f4":
            raise ValueError("Native array descriptor differs from its observable/row contract")
        path = _local(self.folder, name)
        if c.sha256(path) != descriptor["sha256"]:
            raise ValueError(f"Native array bytes changed: {name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype.str != "<f4" or value.shape != tuple(shape):
            raise ValueError("Native array header differs from its descriptor")
        return value


def score(reading, targets, normalization, correspondence=None) -> dict[str, Any]:
    """Score every native row and schema merge; normalize terminal exactly once.

    Optional correspondence is an already-qualified record with reading_sha256,
    target_manifest_sha256, basis='original-computation', and an answers map.
    Each entry binds current_text_sha256, original_text_sha256 and integer
    [current, original] pairs. The caller retains its construction provenance.
    """
    current, reference = reading.answers, targets.answers
    if [a["id"] for a in current] != [a["id"] for a in reference]:
        raise ValueError("Reading does not contain the target's complete fitting membership")
    if correspondence is not None and (
        correspondence.get("reading_sha256") != reading.sha256
        or correspondence.get("target_manifest_sha256") != targets.sha256
        or correspondence.get("basis") != "original-computation"
    ):
        raise ValueError("Correspondence is not bound to this reading and original target")
    paths = {}
    for now, original in zip(current, reference):
        if now["prompt_sha256"] != original["prompt_sha256"] or now["question"] != original["question"]:
            raise ValueError("Reading contains another question or prompt row count")
        if now["text_sha256"] == original["text_sha256"]:
            if now["steps"] != original["steps"]:
                raise ValueError("Identical history hashes have different step counts")
            pairs = np.column_stack([np.arange(now["steps"])]*2)
            if correspondence is not None and now["id"] in correspondence.get("answers", {}):
                if not np.array_equal(correspondence["answers"][now["id"]]["pairs"], pairs):
                    raise ValueError("Identical histories require identity correspondence")
        else:
            if correspondence is None or now["id"] not in correspondence.get("answers", {}):
                raise ValueError("Changed history lacks bound original-computation correspondence")
            held = correspondence["answers"][now["id"]]
            if held["current_text_sha256"] != now["text_sha256"] or held["original_text_sha256"] != original["text_sha256"]:
                raise ValueError("Correspondence token identities differ")
            pairs = np.asarray(held["pairs"])
            validate_pairs(pairs, now["steps"], original["steps"])
        paths[now["id"]] = pairs
    schema = [n.entry for n in targets.nodes.values()]
    native = reading.record["merges"]
    keys = ("merge", "connection", "layer", "kind", "width")
    if len(schema) != len(native) or any(any(a[k] != b[k] for k in keys) for a, b in zip(schema, native)):
        raise ValueError("Reading omitted or changed a target observable")
    if len({n["merge"] for n in native}) != len(native):
        raise ValueError("Duplicate native merge")
    members: dict[str, Any] = {
        a["id"]: dict(correct=a["correct"], direction=0., size=0., observed_direction=0.,
                      observed_size=0., complete=True, reference_comparisons=0)
        for a in reference
    }
    nodes = []
    terminal_record = None
    for expected, supplied in zip(schema, native):
        if supplied["space"] != "residual":
            raise ValueError("whole-gap-reading/1 stores raw residual states, including terminal")
        raw = reading.array(supplied, f'nodes/{supplied["merge"]}/state.npy', (reading.record["rows"], supplied["width"]))
        state = np.asarray(raw, dtype=np.float64)
        norm = np.linalg.norm(state, axis=1)
        with np.errstate(divide="ignore"):
            log_size = np.log(norm)
        if expected["space"] == "output_norm":
            if supplied["merge"] != reading.record["terminal"]["merge"] or expected != schema[-1]:
                raise ValueError("Output-normalized observable is not the bound terminal merge")
            state, log_size = c.terminal_features(state, log_size, **normalization)
            held = reading.record["terminal"]
            l_out = reading.array(held["l_out"], "terminal/l_out.npy", raw.shape)
            engine = reading.array(held["result_norm"], "terminal/result_norm.npy", (reading.answer_rows, supplied["width"]))
            scaled = np.asarray(raw)*np.float32(normalization["scale"])
            reconstructed = state[:reading.answer_rows]*np.exp(log_size[:reading.answer_rows, None])
            terminal_record = dict(
                engine_answer_rows=reading.answer_rows,
                derived_prompt_rows=reading.record["rows"]-reading.answer_rows,
                layer_scale_same_rows=int(np.all(scaled.view("<u4")==np.asarray(l_out).view("<u4"),axis=1).sum()),
                max_abs_state_difference_from_engine=float(np.max(np.abs(reconstructed-np.asarray(engine,dtype=np.float64)))),
                prompt_output_norm="derived from native raw states, not recorded engine output",
            )
            del l_out, engine, scaled, reconstructed
        elif expected["space"] != "residual":
            raise ValueError("Unsupported target observable space")
        # A fresh node keeps this merge's mmap lifetime local, instead of pinning
        # the loader's cached arrays for every merge throughout the score.
        arrays = gap_targets.Node(targets, expected).arrays()
        terms = {}
        for now, original in zip(current, reference):
            name = now["id"]
            terms[name] = {}
            for segment, offset, count in (("answer","answer_rows_from","steps"),("question","question_rows_from","question")):
                ns = slice(now[offset], now[offset]+now[count])
                rs = slice(original[offset], original[offset]+original[count])
                pairs = paths[name] if segment=="answer" else np.column_stack([np.arange(now[count])]*2)
                term = paired_loss(state[ns], log_size[ns], arrays["unit"][rs], arrays["log_size"][rs],
                                   arrays["loss_weight"][rs], arrays["status"][rs], pairs)
                terms[name][segment] = term
                member = members[name]
                member["complete"] &= term["complete"]
                member["reference_comparisons"] += original[count]
                for channel in ("direction", "size"):
                    member["observed_"+channel] += term["observed_"+channel]
                    if member[channel] is not None:
                        member[channel] = None if term[channel] is None else member[channel]+term[channel]
        nodes.append(dict(merge=expected["merge"], members=terms))
        del arrays, state, raw
    complete = all(m["complete"] for m in members.values())
    return dict(format="whole-gap-score/1", reading_sha256=reading.sha256, target_manifest_sha256=targets.sha256,
                model_sha256=reading.record["model_sha256"], execution=reading.execution,
                complete=complete, total=sum(m["direction"]+m["size"] for m in members.values()) if complete else None,
                members=members, nodes=nodes, merges=len(schema), reference_rows=targets.rows,
                comparisons_per_channel=sum(m["reference_comparisons"] for m in members.values()),
                terminal=terminal_record,
                numerical_representation="float64 features from native float32 states; serialized targets unchanged",
                interpretation="Error under the named constructed field; not a capability or transfer verdict")


def main(run):
    run = Path(run).resolve()
    if not run.is_relative_to(ROOT/"studies/04-round-2-overlap/runs"):
        raise ValueError("Score run must be under the round-two runs root")
    declaration = check_native_run(run, run/"declaration.json")
    if declaration.get("operation") != "original-context-score" or declaration["method_step"] != 3:
        raise ValueError("This command admits only the declared original-control qualification")
    for expected in declaration["implementation"]:
        if c.binding(ROOT/expected["record"],ROOT) != expected:
            raise ValueError("Scorer implementation differs from declaration")
    required = [declaration["reading"], declaration["model"], declaration["tensors"], declaration["terminal"], *declaration["targets"].values()]
    for path in required:
        if not any((ROOT/path).resolve().is_relative_to((ROOT/entry).resolve()) for entry in declaration["inputs"]):
            raise ValueError("Scorer source is not a declared input")
    if any(Path(name).name != name or name in (".", "..") for name in declaration["targets"]):
        raise ValueError("Target label escapes the score run")
    reading = NativeReading(ROOT/declaration["reading"])
    if reading.record["model_sha256"] != declaration["model_sha256"] or str(reading.record["visible"])!="0":
        raise ValueError("Qualification requires the original model on canonical card 0")
    # Only these two normalization tensors are read, and each is checked against
    # the original reference below. The capture binds the full model identity.
    model = ROOT/declaration["model"]
    tensors, _ = _json(ROOT/declaration["tensors"])
    terminal, _ = _json(ROOT/declaration["terminal"])
    terminal = terminal["terminal"]
    weight = checkpoint_values(model,tensors[terminal["output_norm"]]).astype("<f8").ravel()
    if hashlib.sha256(weight.tobytes()).hexdigest()!=terminal["output_norm_sha256"]:
        raise ValueError("Original terminal weight identity differs")
    scale_name, scale = next(iter(terminal["layer_output_scale"].items()))
    if checkpoint_values(model,tensors[scale_name]).ravel().tolist()!=[scale]:
        raise ValueError("Original terminal scale differs")
    normalization = dict(weight=weight,scale=scale,eps=terminal["eps"])
    outputs = {}
    for name, path in declaration["targets"].items():
        targets = gap_targets.read(ROOT/path,ROOT)
        result = score(reading,targets,normalization)
        c.write_json(run/(name+"-score.json"),result)
        outputs[name]=dict(score=c.binding(run/(name+"-score.json"),ROOT),total=result["total"],
                           complete=result["complete"],merges=result["merges"],reference_rows=result["reference_rows"],
                           comparisons_per_channel=result["comparisons_per_channel"],terminal=result["terminal"],
                           correct_identity_error=sum(m["direction"]+m["size"] for m in result["members"].values() if m["correct"]) if result["complete"] else None)
        print(json.dumps(dict(target=name,**outputs[name])),flush=True)
    c.write_json(run/"result.json",dict(prediction_vs_result=outputs,
        failures_only=dict(limits=["Original model and identical histories only; no changed-model or changed-history native qualification.",
                                  "Terminal prompt features are derived; engine output_norm is retained for answer rows only.",
                                  "The scored target fields remain withdrawn as primary native objectives."]),
        what_this_changes="The direct-state reader is bound to the complete contextual scoring arithmetic; no target is reinstated and no weights change."))


if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("usage: python -m pipeline.weight_edit.whole_context_score <run folder>")
    main(sys.argv[1])
