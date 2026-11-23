"""Frozen two-turn native-operand phase reader for decode-only computational fields."""
from hashlib import file_digest
from pathlib import Path

import numpy as np


def scan(query, donors):
    """Use both consecutive FFN transforms. Returned indices are decode-relative."""
    query = np.asarray(query, dtype=np.float64)
    if query.ndim != 2 or len(query) < 2 or not np.isfinite(query).all():
        raise ValueError("Need a finite sequence of at least two native turn vectors")
    scores = np.full((len(query) - 1, 3), np.inf)
    reciprocal = []
    for turns, classes in donors:
        # Identical paired-event distances and class minima to operand.roles/local.
        turns = np.asarray(turns, dtype=np.float64)
        classes = np.asarray(classes)
        if turns.ndim != 2 or turns.shape[1] != query.shape[1] or len(classes) != len(turns)-1:
            raise ValueError("Donor coordinates disagree with query or phase labels")
        one = np.maximum((query*query).sum(1)[:, None] + (turns*turns).sum(1)[None, :]
                         - 2 * (query @ turns.T), 0)
        two = one[1:, 1:] + one[:-1, :-1]
        for role in range(3):
            selected = two[:, classes == role]
            if not selected.shape[1] or not np.isfinite(selected).all():
                raise ValueError("Each eligible frozen donor needs all three phase classes")
            minimum = selected.min(axis=1)
            scores[:, role] = np.minimum(scores[:, role], minimum)
            if role == 1:
                reciprocal.append((np.flatnonzero(minimum == minimum.min()) + 1).tolist())
    if not reciprocal:
        raise ValueError("No family-disjoint frozen donor")
    winners = np.argmin(scores, axis=1)
    down = (np.flatnonzero(winners == 1) + 1).tolist()
    friction = (np.flatnonzero(winners == 2) + 1).tolist()
    reciprocal_rows = set.intersection(*(set(rows) for rows in reciprocal))
    # A genuinely ambiguous equal-distance query is not accepted by a reciprocal policy.
    unanimous = sorted(r for r in reciprocal_rows if scores[r-1, 1] < scores[r-1, 2])
    return dict(descent=down, friction=friction, reciprocal_by_donor=reciprocal,
                reciprocal_two_way=unanimous, scores=scores)


class Matcher:
    """The fitter supplies donor annotations once; query registration reads only turns."""
    def __init__(self, bank, site, mode):
        self.bank, self.site, self.mode = bank, site, mode

    @classmethod
    def from_spec(cls, spec, root):
        if spec["format"] != "frozen-local-operand-phase/1":
            raise ValueError("Unknown frozen operand phase schema")
        binding = spec["bank"]
        path = Path(root) / binding["record"]
        with path.open("rb") as handle:
            if file_digest(handle, "sha256").hexdigest() != binding["sha256"]:
                raise ValueError("Frozen operand bank digest mismatch")
        with np.load(path, allow_pickle=False) as obj:
            donors = [(tuple(obj["groups"][i].tolist()), obj[f"turns_{i}"], obj[f"classes_{i}"])
                      for i in range(len(obj["groups"]))]
            site = int(obj["merge"].item())
        mode = spec["method"]
        if mode not in {"local", "reciprocal_two_way"}:
            raise ValueError("Unknown operand registration policy")
        return cls(donors, site, mode)

    def register(self, query_turns, spec):
        if query_turns.ndim != 3:
            raise ValueError("Pass decode-only [prediction_event,merge,width] transforms")
        query = query_turns[:, self.site]
        if hasattr(query, "detach"):
            query = query.detach().cpu().double().numpy()
        family = tuple(spec["query_group"])
        if len(family) != 2:
            raise ValueError("Need the original two-dimensional donor-exclusion group")
        if spec.get("method") != self.mode:
            raise ValueError("Operand registration method changed after binding")
        read = scan(query, [(turns, labels) for group, turns, labels in self.bank if group != family])
        candidates = read["descent" if self.mode == "local" else "reciprocal_two_way"]
        return dict(accepted=len(candidates) == 1, events=candidates if len(candidates)==1 else [],
                    candidates=candidates, local_descent=read["descent"], local_friction=read["friction"],
                    reciprocal_by_donor=read["reciprocal_by_donor"],
                    reciprocal_two_way=read["reciprocal_two_way"])
