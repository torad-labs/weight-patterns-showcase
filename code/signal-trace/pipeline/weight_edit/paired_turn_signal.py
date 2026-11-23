"""Read a signed turn signal at computation-matched stages of a free execution."""

import numpy as np
import torch


def _validate(artifact):
    if artifact["format"] != "paired-turn-stage/1":
        raise ValueError("Unknown paired-turn signal format")
    width = artifact["width"]
    anchors = np.asarray(artifact["anchor_pairs"], dtype=np.float64)
    axis = np.asarray(artifact["axis"], dtype=np.float64)
    if (
        axis.shape != (width,)
        or anchors.ndim != 2
        or anchors.shape[1] != 2 * width
        or len(anchors) == 0
        or not np.isfinite(anchors).all()
        or not np.isfinite(axis).all()
        or not np.isclose(np.linalg.norm(axis), 1)
        or not np.isfinite(artifact["offset"])
        or not np.isfinite(artifact["stage_radius_squared"])
        or artifact["stage_radius_squared"] < 0
    ):
        raise ValueError("Invalid signed direction, anchor pairs, or stage radius")
    return axis, anchors


def pair_distances(turns, anchors):
    """Squared distance to the nearest observed consecutive pair, in decode order."""
    data = np.asarray(turns, dtype=np.float64)
    if (
        data.ndim != 2
        or anchors.shape[1] != 2 * data.shape[1]
        or not np.isfinite(data).all()
    ):
        raise ValueError("Finite, equally wide turn observations required")
    pairs = np.concatenate([data[:-1], data[1:]], axis=1)
    distances = np.full(len(pairs), np.inf)
    for anchor in anchors:
        diff = pairs - anchor
        distances = np.minimum(distances, np.einsum("ij,ij->i", diff, diff))
    return distances


def match_turns(turns, artifact):
    _, anchors = _validate(artifact)
    distances = pair_distances(turns, anchors)
    # A strict cut abstains when a background event ties the nearest stage.
    return np.flatnonzero(distances < artifact["stage_radius_squared"]) + 1


def _turn(before, after):
    before, after = before.double(), after.double()
    a = torch.linalg.vector_norm(before, dim=-1, keepdim=True)
    b = torch.linalg.vector_norm(after, dim=-1, keepdim=True)
    if not bool(
        torch.isfinite(a).all()
        and torch.isfinite(b).all()
        and (a > 0).all()
        and (b > 0).all()
    ):
        raise ValueError("Cannot normalize a zero or non-finite native endpoint")
    return after / b - before / a


def locate(native, artifact, *, prompt_events):
    m = artifact["merge"]
    if not (
        0 <= m < len(native["merged"]) == len(native["incoming"]) == artifact["merges"]
    ):
        raise ValueError("Signal and execution have different merge inventories")
    first = prompt_events - 1
    if first < 0 or first >= len(native["incoming"][m]):
        raise ValueError("Prompt boundary lies outside this execution")
    with torch.no_grad():
        turns = (
            _turn(native["incoming"][m][first:], native["merged"][m][first:])
            .cpu()
            .numpy()
        )
    return [
        dict(key=artifact["key"], event=int(first + i), merge=m)
        for i in match_turns(turns, artifact)
    ]


def score(forward, cell, artifact):
    """Higher means closer to the failure side; retains the weight derivative."""
    m, event = cell["merge"], cell["event"]
    if m != artifact["merge"] or cell["key"] != artifact["key"]:
        raise ValueError("Selected cell has a different signal identity")
    turn = _turn(forward["incoming"][m][event], forward["merged"][m][event])
    axis = torch.as_tensor(artifact["axis"], device=turn.device, dtype=turn.dtype)
    return torch.dot(turn, axis) - artifact["offset"]
