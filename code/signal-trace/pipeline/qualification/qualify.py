"""Qualify the expanded recorder against native ordinary inference, not a theory test."""
import argparse
import json
import subprocess
from pathlib import Path


def run(binary, model, cases, backends, out):
    out.mkdir(parents=True, exist_ok=False)
    for mode in ("plain", "sham", "capture"):
        with (out / (mode + ".stdout")).open("w") as stdout, (out / (mode + ".stderr")).open("w") as stderr:
            subprocess.run([str(binary), str(model), str(out / mode), str(cases), str(backends), mode, "full", "trace"], stdout=stdout, stderr=stderr, check=True)
    comparisons = []
    for case in json.loads(cases.read_text()):
        cid = case["id"]
        arms = {m: json.loads((out / m / cid / "result.json").read_text()) for m in ("plain", "sham", "capture")}
        for mode in ("sham", "capture"):
            assert arms[mode]["tokens"] == arms["plain"]["tokens"], (cid, mode, "tokens")
            assert [s["logits_sha256"] for s in arms[mode]["steps"]] == [s["logits_sha256"] for s in arms["plain"]["steps"]], (cid, mode, "complete logits")
        event_count = write_count = read_count = transfer_gaps = 0
        weights = set()
        seen = {}
        cache_writes = {}
        execution = None
        vocabulary_seen = False
        for line in (out / "capture" / cid / "signal-events.jsonl").open():
            e = json.loads(line)
            if e["graph_execution"] != execution:
                seen.clear()
                execution = e["graph_execution"]
            assert e["event"] == event_count
            event_count += 1
            assert e["before"] < e["after"]
            if "cache_write" in e:
                write_count += 1
                w = e["cache_write"]
                assert w["slots"]
                if w["previous_event"] is not None:
                    assert cache_writes[w["previous_event"]] == w["storage"]
                cache_writes[e["event"]] = w["storage"]
            if e["operation"] == "FLASH_ATTN_EXT":
                read_count += 1
                assert e["inputs"][1].get("cache_state") is not None
                assert e["inputs"][2].get("cache_state") is not None
            for operand in e["inputs"]:
                if operand is None:
                    continue
                assert "weight" in operand or "snapshot" in operand
                if "weight" in operand:
                    weights.add(operand["weight"]["name"])
                producer = operand.get("producer_event")
                if producer is not None:
                    assert producer in seen, (e["event"], "producer outside graph execution")
                    assert seen[producer] == operand["snapshot"], (e["event"], "producer preimage mismatch")
                if operand.get("cache_state") is not None:
                    assert cache_writes[operand["cache_state"]] == operand["cache_storage"]
                if "backend copy" in operand.get("source_status", ""):
                    transfer_gaps += 1
            seen[e["event"]] = e["output"]
            vocabulary_seen |= e["name"] == "result_output"
        layer_ids = {int(w.split(".")[1]) for w in weights if w.startswith("blk.")}
        assert layer_ids == set(range(35)), sorted(layer_ids)
        assert event_count and write_count and read_count and vocabulary_seen
        comparisons.append({"case": cid, "decisions": len(arms["plain"]["tokens"]), "tokens_and_complete_logits_equal": True, "events": event_count, "layers": sorted(layer_ids), "weight_tensors": len(weights), "cache_writes": write_count, "attention_reads": read_count, "recorded_producer_and_cache_links_valid": True, "backend_input_transfer_gaps": transfer_gaps, "complete_end_to_end_lineage": False})
    report = {"qualification_only": True, "cases": comparisons, "not_a_capability_result": True}
    (out / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("binary", "model", "cases", "backends", "out"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    run(a.binary.resolve(), a.model.resolve(), a.cases.resolve(), a.backends.resolve(), a.out.resolve())
