"""Compare post-landmark transformations on natural responses, with group-disjoint witnesses."""

from functools import lru_cache
from hashlib import file_digest
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

import landmark_input

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
ROOT = HERE.parents[3]
PRIOR = RUNS / "descent-log-gain-and-repair-controls"
SITE = 61
SUFFIX = np.arange(SITE + 1, 105)
ROUTES = ("registered", "first_warning")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unit(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("Unresolved direction")
    return vectors / norms


class Original:
    def __init__(self, member):
        self.name = member
        p = RUNS / "wide-overlap/exported" / member
        index = read(p / "index.json")
        self.rows = np.asarray(index["input_positions"])
        self.turns = np.load(p / "turns.npy", mmap_mode="r")
        self.norms = np.load(p / "norms.npy", mmap_mode="r")
        self.locator = np.array(self.turns[:, SITE], dtype=np.float64)
        self.output = landmark_input.states(member)
        assert np.array_equal(self.rows, landmark_input.native_rows(member))
        assert sha(p / "turns.npy") == index["vector_sha256"]
        assert sha(p / "norms.npy") == index["norms_sha256"]

    def fields(self, indices):
        turns = self.turns[indices, SITE + 1 :].astype(np.float64)
        norms = self.norms[indices, SITE + 1 :].astype(np.float64)
        return turns, np.log(norms[:, :, 1] / norms[:, :, 0])


class Native:
    def __init__(self, folder, member, reader, inventory):
        raw = folder / "raw" / member
        self.name = member
        meta = read(folder / "measurement.json")
        capture = read(ROOT / meta["capture"]["record"])
        for name in ("stream.f32", "stream.jsonl"):
            assert sha(raw / name) == capture["raw"][name]["sha256"]
        self.rows = np.array(
            [v["input_position"] for v in read(raw / "result.json")["steps"]]
        )
        self.offsets = reader.rows_in(raw, meta["events"])
        self.values = np.memmap(raw / "stream.f32", dtype="<f4", mode="r")
        self.reader = reader
        self.inventory = inventory
        before, after = self.operands(self.rows, SITE)
        self.locator = unit(after) - unit(before)
        self.output = after

    def operands(self, rows, merge):
        node = self.inventory[merge]
        a, branch, b = [
            self.reader.taken(self.values, self.offsets[node[key]], rows)
            for key in ("incoming", "branch", "connection")
        ]
        if not np.array_equal((a + branch).view("u4"), b.view("u4")):
            raise ValueError("Native ADD failed at merge " + str(merge))
        return a, b

    def fields(self, indices):
        turns, gains = [], []
        for merge in SUFFIX:
            a, b = self.operands(self.rows[indices], merge)
            turns.append(unit(b) - unit(a))
            gains.append(
                np.log(
                    np.linalg.norm(b.astype("f8"), axis=1)
                    / np.linalg.norm(a.astype("f8"), axis=1)
                )
            )
        return np.stack(turns, axis=1), np.stack(gains, axis=1)


class Checkpoint:
    def __init__(self, step):
        import controls

        self.controls = controls
        self.step = step
        p = HERE / "controls"
        with np.load(p / f"{step}.npz") as data:
            self.rows = data["rows"]
            self.locator = data["ff20_turn"].astype(np.float64)
            self.output = data["ff20_output"]
        self.annotation = read(p / f"{step}.json")["evaluation"]["descent"]["row"]

    def fields(self, indices):
        turn, gain = self.controls.read_fields(
            self.step, self.rows[indices], merges=SUFFIX
        )
        return turn.astype(np.float64), gain.astype(np.float64)


def mixtures(contexts, query):
    """Project a unit query onto a small convex hull; check the simplex optimality gap."""
    offsets = contexts - query
    gram = offsets @ offsets.T
    n = len(gram)
    estimate = minimize(
        lambda x: (0.5 * x @ gram @ x, gram @ x),
        np.full(n, 1 / n),
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * n,
        constraints=[
            dict(type="eq", fun=lambda x: x.sum() - 1.0, jac=lambda x: np.ones_like(x))
        ],
        options=dict(ftol=np.finfo(np.float64).eps),
    )
    x = estimate.x
    gradient = gram @ x
    gap = float(x @ gradient - gradient.min())
    # This checks numerical precision of a convex projection, independent of its label.
    if (
        abs(x.sum() - 1) > np.sqrt(np.finfo(float).eps)
        or np.min(x) < -np.finfo(float).eps
        or gap > np.sqrt(np.finfo(float).eps) * max(1, np.linalg.norm(gram, 2))
    ):
        raise ArithmeticError((estimate.message, gap, x))
    return x, float(x @ gradient), gap


def warning_positions(scores):
    return np.flatnonzero((scores[1:] > 0) & (scores[:-1] > 0)) + 1


class Bank:
    def __init__(self):
        observations = read(
            RUNS / "descent-multiple-references-retrospective/result.json"
        )["readings"]
        timing = read(PRIOR / "timing-results.json")["members"]
        self.timing = timing
        self.cohort = read(HERE / "declaration.json")["cohort"]
        self.groups = {x["member"]: tuple(x["group"]) for x in observations}
        self.correct = {x["member"]: x["descent"] == "correct" for x in observations}
        self.originals = {k: Original(k) for k in self.groups}
        self.registered = {
            k: int(
                np.flatnonzero(
                    self.originals[k].rows == timing[k]["registered_row"]
                ).item()
            )
            for k in self.groups
        }
        self.anchors = {
            k: self.originals[k].locator[i] for k, i in self.registered.items()
        }

    @lru_cache(None)
    def direction(self, excluded):
        def mean(label):
            v = [
                v
                for k, v in self.anchors.items()
                if self.correct[k] == label and self.groups[k] not in excluded
            ]
            return np.mean(v, axis=0)

        good, bad = mean(True), mean(False)
        return (good + bad) / 2, unit(bad - good)

    def locator_scores(self, vectors, excluded):
        center, direction = self.direction(frozenset(excluded))
        return (vectors - center) @ direction

    @lru_cache(None)
    def anchor(self, member, index):
        original = self.originals[member]
        turns, gains = original.fields(np.array([index]))
        return unit(original.output[index]), turns[0], gains[0]

    def consensus(self, member, positions, excluded):
        candidates = self.originals[member].locator
        distances = {}
        for other in self.cohort:
            energy = self.groups[other]
            if energy in excluded:
                continue
            t = self.originals[other].locator
            end = self.registered[other]
            before = candidates[positions - 1] - t[end - 1]
            now = candidates[positions] - t[end]
            error = (before**2 + now**2).sum(axis=1)
            distances.setdefault(energy, []).append(error)
        grouped = [np.median(values, axis=0) for values in distances.values()]
        return int(positions[np.argmin(np.median(grouped, axis=0))])

    @lru_cache(None)
    def reference(self, group, route):
        selected = {}
        for k in self.cohort:
            if self.groups[k] == group:
                continue
            i = self.registered[k]
            if route == "oracle":
                target = self.timing[k]["choice_row"] - 1
                i = int(np.flatnonzero(self.originals[k].rows == target).item())
            if route in ("first_warning", "consensus"):
                positions = warning_positions(
                    self.locator_scores(
                        self.originals[k].locator, {group, self.groups[k]}
                    )
                )
                if not len(positions):
                    continue
                i = int(positions[0])
                if route == "consensus":
                    i = self.consensus(k, positions, {group, self.groups[k]})
            selected[k] = i
        classes = []
        for label in (True, False):
            ids = [k for k in selected if self.correct[k] == label]
            if not ids:
                raise ValueError("Missing class in fold")
            c, t, g = zip(*(self.anchor(k, selected[k]) for k in ids))
            classes.append((ids, np.stack(c), np.stack(t), np.stack(g)))
        return selected, classes

    def evaluate(self, trace, group, decision):
        if np.any(np.diff(trace.rows) != 1):
            raise ValueError("Non-contiguous trace")
        scores = self.locator_scores(trace.locator, {group})
        gated = warning_positions(scores)
        before = int(np.flatnonzero(trace.rows == decision - 1).item())
        indices = np.unique(np.r_[gated, before])
        turns, gains = trace.fields(indices)
        contexts = unit(trace.output[indices])
        output = {
            "decision": int(decision),
            "events": len(trace.rows),
            "landmark_rows": trace.rows[gated].tolist(),
            "variants": {},
        }
        for route in ROUTES:
            selected, classes = self.reference(group, route)
            readings = []
            for i, j in enumerate(indices):
                summaries = []
                for ids, c, t, g in classes:
                    coef, squared_distance, gap = mixtures(c, contexts[i])
                    delta = turns[i] - np.einsum("k,kmd->md", coef, t)
                    average = turns[i] - t.mean(axis=0)
                    summaries.append(
                        dict(
                            incoming=squared_distance,
                            conditioned=np.einsum("md,md->m", delta, delta),
                            unconditioned=np.einsum("md,md->m", average, average),
                            gain=(gains[i] - coef @ g) ** 2,
                            witnesses=dict(zip(ids, coef.tolist())),
                            gap=gap,
                        )
                    )
                good, bad = summaries
                local = good["conditioned"] - bad["conditioned"]
                entry = dict(
                    row=int(trace.rows[j]),
                    locator=float(scores[j]),
                    gated=bool(j in gated),
                    incoming=float(good["incoming"] - bad["incoming"]),
                    conditional=float(local.sum()),
                    average=float((good["unconditioned"] - bad["unconditioned"]).sum()),
                    gain=float((good["gain"] - bad["gain"]).sum()),
                )
                if j == before:
                    entry.update(
                        local_turn=local.tolist(),
                        local_gain=(good["gain"] - bad["gain"]).tolist(),
                        good_witnesses=good["witnesses"],
                        bad_witnesses=bad["witnesses"],
                        context_distances_squared=[good["incoming"], bad["incoming"]],
                    )
                readings.append(entry)
            event = next(x for x in readings if x["row"] == decision - 1)
            flagged = [
                x["row"] for x in readings if x["gated"] and x["conditional"] > 0
            ]
            output["variants"][route] = dict(
                predecision=event,
                alerts=flagged,
                donor_rows={
                    k: int(self.originals[k].rows[j]) for k, j in selected.items()
                },
                measured=readings,
            )
        return output


def counts(results, bank):
    groups = {
        "five_correct": [k for k in bank.cohort if bank.correct[k]],
        "thirteen_wrong": [k for k in bank.cohort if not bank.correct[k]],
        "ten_other_correct": [k for k in bank.groups if k not in bank.cohort],
    }
    out = {}
    for label, members in groups.items():
        out[label] = {}
        for route in ROUTES:
            samples = [results[k]["variants"][route] for k in members]
            v = [s["predecision"] for s in samples]
            out[label][route] = dict(
                n=len(v),
                gate=sum(x["gated"] for x in v),
                conditional=sum(x["conditional"] > 0 for x in v),
                joint=sum(x["gated"] and x["conditional"] > 0 for x in v),
                input_only=sum(x["gated"] and x["incoming"] > 0 for x in v),
                average=sum(x["gated"] and x["average"] > 0 for x in v),
                earlier=sum(
                    any(r < s["predecision"]["row"] for r in s["alerts"])
                    for s in samples
                ),
                anywhere=sum(bool(s["alerts"]) for s in samples),
            )
    return out


def main():
    bank = Bank()
    originals = {
        k: bank.evaluate(t, bank.groups[k], bank.timing[k]["choice_row"])
        for k, t in bank.originals.items()
    }
    summary = counts(originals, bank)
    out = dict(
        summary=summary,
        originals=originals,
        controls={},
        merge_indices=SUFFIX.tolist(),
        prediction_vs_result=summary,
        failures_only="Same-template, historically selected development routes. Flags away from the annotated descent have no operation labels.",
        what_this_changes="Checks whether the FF20 warning becomes specific after conditioning downstream transformations on its actual outgoing state.",
    )
    print(json.dumps(summary), flush=True)
    reader = load_module(
        "native_state_reader",
        ROOT / "records/development/2026-11-12-path-forward/states_reading.py",
    )
    inventory = read(
        RUNS / "vector-chance-working-from-update-3/loop/9/reading/reading.json"
    )["merges"]
    annotation = read(
        ROOT / "records/retrospectives/2026-11-17-natural-descent-decisions/scores.json"
    )["frozen_and_transplant"]
    for arm in ("early", "late_attention", "late_ffn", "late", "reordered"):
        if arm == "reordered":
            folder = RUNS / "vector-chance-isolated-pre-error-update/original/transfer"
            h = read(PRIOR / "result.json")["previously_missed_challenge"]
            member, decision, group = h["member"], h["decision_event"], (36, 20)
        else:
            folder = RUNS / "vector-chance-descent-sign-swap" / arm
            member = "discovery-05-channel-h1"
            decision = annotation["swap_" + arm]["choices"]["symbolic"][
                "input_position"
            ]
            group = bank.groups[member]
        trace = Native(folder, member, reader, inventory)
        out["controls"][arm] = bank.evaluate(trace, group, decision)
    for step in (175, 189, 387, 388):
        trace = Checkpoint(step)
        out["controls"][str(step)] = bank.evaluate(trace, (64, 40), trace.annotation)
    out["source_sha256"] = {
        p.name: sha(p)
        for p in (
            Path(__file__),
            HERE / "landmark_input.py",
            HERE / "controls.py",
            HERE / "declaration.json",
            HERE / "registration-check.json",
        )
    }
    filename = (
        "registration-results.json"
        if "--registration-check" in sys.argv
        else "result.json"
    )
    (HERE / filename).write_text(
        json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    if "--registration-check" in sys.argv:
        ROUTES = ("consensus", "oracle")
    main()
