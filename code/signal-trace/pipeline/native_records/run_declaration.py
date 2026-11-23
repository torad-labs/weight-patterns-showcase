"""Refuse a native run that the round-two verifier would reject.

The required fields and parameter rules come from
studies/04-round-2-overlap/round-2-method.json, the same source the
pre-commit verifier uses, so the two can't drift apart.
"""
import hashlib
import json
from pathlib import Path

from pipeline.research_guard import input_archives, recorded_losses
from pipeline.research_guard.verify_round2 import ROOT, verify


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def check_native_run(output, declaration_path):
    """Validate the run records without imposing commit or output timestamp order, and refuse a run whose own declared inputs or
    target are archived and not back on this disk, held to their inventory (input_archives)."""
    path = Path(declaration_path)
    declaration = json.loads(path.read_text())
    problems = verify(ROOT)
    if problems:
        raise ValueError('Round-two drift; refusing to run:\n' + '\n'.join(problems))
    # No run computes on a recorded loss: not the lost path itself, nothing below one, and no folder that holds one, however spelled
    # and in whichever field of the declaration it is named.
    method = json.loads((ROOT / recorded_losses.METHOD).read_text())
    gone = recorded_losses.named_in(recorded_losses.load(ROOT, method)[0], ROOT, declaration, method['runs_root'], own=path.parent.name)
    if gone:
        raise ValueError('This run declares inputs that are recorded as lost; refusing to run:\n' + '\n'.join(gone))
    waiting = input_archives.unrestored(ROOT, declaration)
    if waiting:
        raise ValueError('Archived inputs of this run are not restored; refusing to run:\n' + '\n'.join(waiting))
    return declaration
