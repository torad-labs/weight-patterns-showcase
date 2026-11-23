"""Round two's population, set by rule, and the recorder case files for it.

The population is every physics question with a recorded original answer under
inputs/acquisition/discovery-*, expanded-* and historical-plain. Nothing is
added, dropped or matched.
"""
import json
import re

from pipeline.paths import ACQUISITION

SOURCE = re.compile(r'(discovery|expanded)-\d+|historical-plain')


def population():
    rows = []
    for folder in sorted(p for p in ACQUISITION.iterdir() if p.is_dir() and SOURCE.fullmatch(p.name)):
        for case_path in sorted(folder.glob('*/case.json')):
            case = json.loads(case_path.read_text())
            recorded = case_path.parent/'result.json'
            if case.get('group') == 'physics' and recorded.exists():
                rows.append(dict(source=folder.name, id=case['id'], case=case, recorded=recorded))
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('A case id repeats across sources')
    return rows


def token_ids(rows, words):
    """Token id of each word, read from the pieces the recorder saved with the recorded answers."""
    found = {w: set() for w in words}
    for row in rows:
        for step in json.loads(row['recorded'].read_text())['steps']:
            for entry in step['top']:
                if entry['piece'] in found:
                    found[entry['piece']].add(entry['id'])
    for word, ids in found.items():
        if len(ids) != 1:
            raise ValueError(f'{word!r} maps to {sorted(ids)} token ids; exactly one is needed')
    return {w: next(iter(ids)) for w, ids in found.items()}


def recorder_case(row, watched, **extra):
    case = row['case']
    return dict(id=case['id'], tokens=case['tokens'], max_tokens=case['max_tokens'], watched_tokens=watched, **extra)


def in_sample(source, template):
    """The construction sample of round two: round one's physics questions and the first screening set in the channel wording.

    64 questions, the scale of round one. Everything else that was recorded stays as test material.
    """
    return source == 'historical-plain' or (source.startswith('discovery-') and template == 'channel')
