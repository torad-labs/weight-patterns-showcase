"""Extend A's construction screening after the fully read first population.

Five of the first 96 responses have the target error as their first wrong
quantitative step. Hold the successful target-producing channel wording fixed
and vary loss as well as mass, speed, descent and final height. These are adaptive
construction questions, never fresh evaluation. Existing questions are reused
by reference rather than counted or generated twice.
"""
import hashlib
import itertools
import json

from pipeline.acquisition.prepare_a_cases import RUN, make, write_new


def main():
    prior = json.loads((RUN/'discovery-cases.json').read_text())
    prior += json.loads((RUN/'historical-cases.json').read_text())
    known = {c['prompt']: c['id'] for c in prior}
    fresh = {c['prompt'] for c in json.loads((RUN/'fresh-cases.json').read_text())}
    cases, reused = [], []
    for i, (m, v, drop, loss) in enumerate(itertools.product(
            (2, 3, 4, 5), (4, 6, 8), (1, 2), (12, 16, 20, 24))):
        pair = f'expanded-{i:03d}'
        for rise in (1, 3):
            c = make(f'{pair}-h{rise}', pair, m, v, drop, loss, rise, 'channel')
            assert c['prompt'] not in fresh
            if c['prompt'] in known:
                reused.append({'planned_id': c['id'], 'existing_id': known[c['prompt']]})
            else:
                cases.append(c)
    assert len({c['prompt'] for c in cases}) == len(cases)
    write_new(RUN/'expanded-cases.json', cases)
    chunks = []
    for start in range(0, len(cases), 24):
        path = RUN/f'expanded-{start//24:02d}-cases.json'
        write_new(path, cases[start:start+24])
        chunks.append(path.name)
    manifest = {'purpose': 'adaptive construction screening; no fresh outcomes used',
                'reason': 'First 96 yielded five first-departure height-double-count failures, below the proposed 20-50.',
                'condition': 'Same original model, greedy, 768 cap, GPU0 and historical channel wording',
                'planned': 192, 'new_questions': len(cases), 'reused_questions': reused,
                'chunks': chunks, 'fresh_generated': False,
                'sha256': {n: hashlib.sha256((RUN/n).read_bytes()).hexdigest()
                           for n in ['expanded-cases.json', *chunks]}}
    write_new(RUN/'expanded-question-manifest.json', manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
