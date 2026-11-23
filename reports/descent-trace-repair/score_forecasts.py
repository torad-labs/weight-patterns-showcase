"""Score a predeclared forecast against local outcomes from free generations."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

TRANSITIONS = {
    ("wrong", "right"): "repair",
    ("wrong", "wrong"): "persist",
    ("right", "right"): "preserve",
    ("right", "wrong"): "regress",
}


def indexed(document, fields):
    rows = document["operations"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("operations must be a nonempty list")
    result = {}
    for row in rows:
        for field in ("case", "operation", *fields):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"missing or empty {field} in {row!r}")
        key = row["case"], row["operation"]
        if key in result:
            raise ValueError(f"duplicate operation {key!r}")
        result[key] = row
    return result


def score(frozen, outcomes):
    for field in ("study", "source_model", "edit"):
        if not isinstance(frozen.get(field), str) or not frozen[field].strip():
            raise ValueError(f"forecast lacks {field}")
        if frozen[field] != outcomes.get(field):
            raise ValueError(f"outcome document differs on {field}")
    forecasts = indexed(frozen, ("family", "before", "prediction", "evidence"))
    observed = indexed(outcomes, ("after",))
    if forecasts.keys() != observed.keys():
        raise ValueError("operation keys differ; include absent or unscorable outcomes explicitly")
    counts = Counter()
    by_outcome = {}
    families = {}
    for key, row in forecasts.items():
        before, forecast, after = row["before"], row["prediction"], observed[key]["after"]
        if before not in ("right", "wrong", "unscorable"):
            raise ValueError(f"invalid starting validity for {key!r}")
        candidates = {v for (start, _), v in TRANSITIONS.items() if start == before}
        if forecast not in candidates | {"abstain"}:
            raise ValueError(f"prediction {forecast!r} incompatible with {before!r} at {key!r}")
        if after not in ("right", "wrong", "absent", "unscorable"):
            raise ValueError(f"invalid observed outcome at {key!r}")
        outcome = TRANSITIONS.get((before, after), "unmeasured")
        by_outcome.setdefault(forecast, Counter())[outcome] += 1
        family = families.setdefault(row["family"], Counter())
        for tally in (counts, family):
            tally["declared"] += 1
            tally["forecast_issued"] += forecast != "abstain"
            tally["abstained"] += forecast == "abstain"
            tally["absent_after"] += after == "absent"
            tally["unscorable_after"] += after == "unscorable"
            tally["unscorable_before"] += before == "unscorable"
            tally["transition_observed"] += outcome != "unmeasured"
            scored = forecast != "abstain" and outcome != "unmeasured"
            tally["scored"] += scored
            tally["forecast_correct"] += scored and forecast == outcome
            tally["unchanged_baseline_correct"] += scored and before == after
    return {"study": frozen["study"], "counts": counts, "by_family": families,
            "prediction_by_observation": by_outcome}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast", type=Path)
    parser.add_argument("outcomes", type=Path)
    parser.add_argument("--forecast-sha256", required=True,
                        help="digest registered before the edit was installed")
    args = parser.parse_args()
    try:
        data = args.forecast.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != args.forecast_sha256:
            raise ValueError("forecast differs from the registered digest")
        result = score(json.loads(data), json.loads(args.outcomes.read_bytes()))
        result["forecast_sha256"] = digest
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()
