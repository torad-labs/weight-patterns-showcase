"""Declared, original-only compilation of successful native-point memberships."""
import hashlib
import json
import sys
from pathlib import Path


from pipeline.native_records.native_trace import checkpoint_values
from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROOT
from pipeline.weight_edit import context_targets as c, success_sets as sets
from pipeline.weight_edit.whole_context_score import NativeReading


def declared_source(root, declaration, name):
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("A source must be named from the repository root")
    path = (root/relative).resolve()
    if not any(path.is_relative_to((root/entry).resolve()) for entry in declaration["inputs"]):
        raise ValueError(f"Source is not declared: {name}")
    return path


def declared_record(root, declaration, binding):
    declared_source(root, declaration, binding["record"])
    return sets.bound_path(root, binding)


def compile_run(folder):
    run = Path(folder).resolve()
    if not run.is_relative_to(ROOT/"studies/04-round-2-overlap/runs"):
        raise ValueError("Compilation must write only inside its own round-two run")
    declaration = check_native_run(run, run/"declaration.json")
    if declaration["method_step"] != 3 or declaration.get("operation") != "original-success-sets":
        raise ValueError("This compiler needs its prospective original-success-sets declaration")
    if (run/"reference").exists() or (run/"result.json").exists():
        raise FileExistsError("Preserve partial or finished reference compilation")
    for entry in declaration["implementation"]:
        sets.bound_path(ROOT, entry)
    source = declaration["success_sets"]
    reading_file = declared_record(ROOT, declaration, source["reading"])
    if reading_file.name != "reading.json":
        raise ValueError("Expected the bound original native reading manifest")
    reading = NativeReading(reading_file.parent)
    if reading.record["model_sha256"] != declaration["model_sha256"]:
        raise ValueError("Reference computation is not the declared original model")
    if [a["id"] for a in reading.answers] != source["fitting"]:
        raise ValueError("Native reading changed the frozen fitting membership")
    declared_record(ROOT, declaration, source["turns"])
    turns = sets._turn_record(reading, source["turns"], ROOT)
    for domains in turns["answers"].values():
        for entry in domains.values():
            declared_source(ROOT, declaration, entry["record"])
    declared_record(ROOT, declaration, source["objective"])
    normal = source["normalization"]
    tensor_path = declared_record(ROOT, declaration, normal["tensors"])
    terminal_path = declared_record(ROOT, declaration, normal["reference"])
    tensors = json.loads(tensor_path.read_text())
    terminal = json.loads(terminal_path.read_text())["terminal"]
    model = declared_source(ROOT, declaration, normal["model"])
    if c.sha256(model) != declaration["model_sha256"]:
        raise ValueError("Normalization source is not the immutable original model")
    weight = checkpoint_values(model, tensors[terminal["output_norm"]]).astype("<f8").ravel()
    if hashlib.sha256(weight.tobytes()).hexdigest() != terminal["output_norm_sha256"]:
        raise ValueError("Original output-normalization bytes changed")
    name, scale = next(iter(terminal["layer_output_scale"].items()))
    if checkpoint_values(model, tensors[name]).ravel().tolist() != [scale]:
        raise ValueError("Original terminal scale changed")
    normalization = dict(weight=weight, scale=scale, eps=terminal["eps"])
    reference = sets.compile_reference(
        run/"reference", ROOT, reading, source["turns"], normalization, source["objective"],
        provenance=dict(declaration=c.binding(run/"declaration.json", ROOT),
                        implementation=declaration["implementation"], normalization=normal))
    artifact = sets.Reference(run/"reference", ROOT)
    c.write_json(run/"result.json", dict(
        prediction_vs_result=dict(reference=reference, members=len(artifact.answers),
                                  donors=len(artifact.donors), rows=artifact.rows,
                                  merges=len(artifact.schema), mapped_native_points=len(artifact.donor_rows)),
        failures_only=dict(limits=[
            "Complete computational correspondence does not prove semantic equivalence.",
            "Local convex sets may mix incompatible computations or charge legitimate context.",
            "No edited model, native update, capability closure or no-fit evaluation was run."]),
        what_this_changes="An immutable successful-set membership artifact exists; no nearest point or training loss was fitted here."))
    return reference


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m pipeline.weight_edit.compile_success_sets <run>")
    print(json.dumps(compile_run(sys.argv[1])))
