"""Register frozen reference stages on every prefix of the naturally generated query."""

import json
from pathlib import Path

import numpy as np
from pipeline.native_records.run_declaration import check_native_run, digest
from pipeline.observability.recorder import Recorder
from pipeline.overlap.complete_overlap import unit_steps
from pipeline.weight_edit import final_experiment as learner

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[3]


def checked(binding):
    path = ROOT / binding["record"]
    if digest(path) != binding["sha256"]:
        raise ValueError(f"Source changed: {path}")
    return path


def read(binding):
    return json.loads(checked(binding).read_text())


def endpoints(similarity):
    """Each endpoint minimizes summed DTW cost for the query prefix seen so far."""
    prev = np.full(len(similarity) + 1, np.inf)
    prev[0] = 0.0
    observed = []
    for col in similarity.T:
        cur = np.full_like(prev, np.inf)
        cost = 1.0 - np.clip(col, -1.0, 1.0).astype(np.float64)
        for i, local in enumerate(cost, 1):
            cur[i] = local + min(prev[i - 1], cur[i - 1], prev[i])
        best = cur[1:].min()
        observed.append(np.flatnonzero(cur[1:] == best).tolist())
        prev = cur
    return observed


def capture_turns(declaration):
    receipt = read(declaration["challenge"])
    result = read(receipt["raw"])
    capture = read(receipt["capture"])
    [case] = read(receipt["prompt"])
    if (
        not result["ended"]
        or result["conditioned_continuation"]
        or result["teacher_token_count"]
        or result["selection"] != "greedy"
        or len(case["tokens"]) != result["prompt_tokens"]
    ):
        raise ValueError("Invalid challenge generation")
    for b in capture["raw"].values():
        checked(b)
    app = learner.controller(learner.MEASURED)
    geometry = read(declaration["reading"])["merges"]
    raw = (ROOT / receipt["raw"]["record"]).parent
    native = app.held.raw_forward(
        raw, geometry, receipt["events"], geometry[0]["width"]
    )
    first = result["prompt_tokens"] - 1
    turns = app.turn_pair.turns(native, expected_merges=len(geometry))
    return unit_steps(
        np.stack([t[first:].float().numpy() for t in turns], axis=1)
    ), first


def run(d, rec):
    source = read(d["source"])
    prior = read(d["sites"])
    stages = {s["mode"]: s["row"] for s in prior["sites"]}
    stages["historical_control"] = d["historical_reference_event"]
    entries = {m["id"]: m for m in source["members"]}
    template = entries[source["reference_id"]]
    ref_start = template["prompt_events"] - 1
    ref = unit_steps(np.load(checked(template["turns"]), mmap_mode="r"))
    raw = {}
    members = [
        m["id"] for m in entries.values() if m["group"] == source["held_group"]
    ] + [d["challenge_id"]]
    for name in members:
        with rec.span("trace.prefix_correspondence", attributes={"member": name}):
            if name in entries:
                m = entries[name]
                first = m["prompt_events"] - 1
                query = unit_steps(np.load(checked(m["turns"]), mmap_mode="r"))
            else:
                query, first = capture_turns(d)
            paths = endpoints(ref @ query.T)
            raw[name] = dict(
                first_event=first,
                reference_endpoints=[[r + ref_start for r in row] for row in paths],
            )
            del query
    (RUN / "prefix-endpoints.json").write_text(
        json.dumps(raw, separators=(",", ":")) + "\n"
    )
    annotations = read(d["evaluation"])
    ref_pre = annotations["originals"][source["reference_id"]]["descent"][
        "prechoice_row"
    ]
    measured = {}
    for name, data in raw.items():
        ann = (
            annotations["originals"][name]
            if name in entries
            else annotations["controls"]["reordered"]
        )
        pre = ann["descent"]["prechoice_row"]
        eps = data["reference_endpoints"]
        first = data["first_event"]
        info = {}
        for key, stage in stages.items():
            hits = [j + first for j, row in enumerate(eps) if stage in row]
            info[key] = dict(
                reference_event=stage,
                query_hits=hits,
                relative_to_prechoice=[x - pre for x in hits],
            )
        matched = eps[pre - first]
        measured[name] = dict(
            query_prechoice=pre,
            reference_prechoice=ref_pre,
            reference_at_query_prechoice=matched,
            reference_offset_at_query_prechoice=[x - ref_pre for x in matched],
            stages=info,
        )
    result = dict(
        prediction_vs_result=measured,
        failures_only="An endpoint is the nearest cumulative correspondence available. It exists even if the query is unfamiliar. This test measures registration and supplies no class, trust margin, or gradient.",
        what_this_changes="Compare the time at which frozen reference stages can be reached online with actual choices in the different histories.",
    )
    (RUN / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    d = check_native_run(RUN, RUN / "declaration.json")
    for b in d["observability"]["sources"]:
        checked(b)
    rec = Recorder(
        RUN / "observability/events.jsonl",
        dict(
            project="weight-patterns",
            folder=RUN.name,
            operation=d["operation"],
            entrypoint=d["program"]["record"],
            declaration_sha256=digest(RUN / "declaration.json"),
        ),
    )
    try:
        run(d, rec)
    except BaseException:
        rec.close("FAILED")
        raise
    else:
        rec.close()
