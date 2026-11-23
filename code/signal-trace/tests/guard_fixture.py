"""Shared set-up of the guard tests' temporary trees."""
import json
from pathlib import Path

from pipeline.research_guard import input_archives as archives
from pipeline.research_guard.verify_round2 import ROOT


def copy_method(tmp, **changed):
    """The repository's method file written into a temporary tree, without its `recorded_losses`: the list it names is not in that tree,
    and what a made-up tree has lost is for its test to say (tests/test_recorded_losses.py). Returns the method as written."""
    method = json.loads((ROOT / archives.METHOD).read_text())
    method.pop('recorded_losses', None)
    method.update(changed)
    target = Path(tmp) / archives.METHOD
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(method))
    return method
