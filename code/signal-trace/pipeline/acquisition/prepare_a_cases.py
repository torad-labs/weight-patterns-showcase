"""Prepare A's original-model screening and reserved evaluation questions.

Goal: vary physical quantities while holding the task and 768-token greedy
generation condition fixed. Each discovery pair changes only final climb.
This is case acquisition, not trace selection or edit construction.
"""
import hashlib
import itertools
import json
from pathlib import Path

from pipeline.paths import ROOT
RUN = ROOT / 'runs/legs/gemma4-e2b/signal-trace-A-2026-11-07'
HIST = ROOT / 'runs/legs/gemma4-e2b/physics-weight-edit-2026-11-07/cases.json'
ENDING = ('\nUse g = 10 m/s^2. Treat the object as a point mass and ignore rotation. '
          'There are no other energy losses and no motor or external push after the start.\n'
          'Will it reach the destination? Show a compact numerical calculation, not just a verbal outline. '
          'Write three lines: Available energy = your calculation in joules; Required energy = your '
          'calculation in joules; Verdict = YES or NO. Use the stated initial speed, descent, '
          'dissipation, and final climb.')


def make(cid, pair, m, v, drop, loss, rise, template):
    if template == 'channel':
        prompt = (f'A block sliding along a channel has mass {m} kg and initially moves at {v} m/s. '
                  f'It descends {drop} m along a smooth track, crosses a rough horizontal section '
                  f'that dissipates {loss} J, and then starts a smooth climb to a destination '
                  f'{rise} m above the bottom of that climb.')
    else:
        prompt = (f'A {m} kg block enters a smooth descent at {v} m/s. After descending {drop} m, '
                  f'it traverses a rough horizontal stretch that removes {loss} J of energy. '
                  f'It then ascends a smooth track toward a point {rise} m above the start of this final ascent.')
    initial = m*v*v/2
    gain = m*10*drop
    available = initial+gain-loss
    required = m*10*rise
    assert available > 0
    return {'id': cid, 'pair': pair, 'family': 'block', 'group': 'physics',
            'template': template, 'prompt': prompt+ENDING, 'max_tokens': 768,
            'parameters': {'m': m, 'v': v, 'drop': drop, 'loss': loss, 'rise': rise},
            'expected': {'initialKineticJ': initial, 'descentGainJ': gain,
                         'lossJ': loss, 'availableJ': available, 'requiredJ': required,
                         'marginJ': available-required, 'answer': 'YES' if available >= required else 'NO'}}


def write_new(path, value):
    text = json.dumps(value, indent=2)+'\n'
    if path.exists():
        assert path.read_text() == text, f'Refusing to change existing manifest {path}'
    else:
        path.write_text(text)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    historical = json.loads(HIST.read_text())
    write_new(RUN/'historical-cases.json', historical)
    discovery=[]
    # 24 independently varied physical configurations; two final heights each.
    # Each appears in two wordings, allowing same-problem wording contrasts.
    for i,(m,v,d) in enumerate(itertools.product((2,3,4,5), (4,6,8), (1,2))):
        loss = 2*m+4*d+1
        for template in ('channel','reordered'):
            pair=f'discovery-{i:02}-{template}'
            for rise in (1,3):
                discovery.append(make(f'{pair}-h{rise}',pair,m,v,d,loss,rise,template))
    fresh=[]
    for i,(m,v,d) in enumerate(itertools.product((2,3,4,5,6), (5,7), (1,2))):
        for rise in (2,4):
            fresh.append(make(f'fresh-{i:02}-h{rise}',f'fresh-{i:02}',m,v,d,3*m+2*d,rise,
                              'channel' if i%2==0 else 'reordered'))
    assert len(discovery)==96 and len(fresh)==40
    assert len({c['prompt'] for c in discovery+fresh})==136
    assert not ({c['prompt'] for c in discovery+fresh} & {c['prompt'] for c in historical})
    for pair in {c['pair'] for c in discovery}:
        rows=[c for c in discovery if c['pair']==pair]
        assert len(rows)==2 and rows[0]['expected']['availableJ']==rows[1]['expected']['availableJ']
    # Split acquisition into 24-question chunks. All chunks are declared before
    # outputs; they are not an optimization or a limit on needed evidence.
    for start in range(0,len(discovery),24):
        write_new(RUN/f'discovery-{start//24:02}-cases.json',discovery[start:start+24])
    write_new(RUN/'discovery-cases.json',discovery)
    write_new(RUN/'fresh-cases.json',fresh)
    manifest={'status':'questions fixed before new outputs; fresh responses not generated',
              'historical':len(historical),'discovery':len(discovery),'fresh_reserved':len(fresh),
              'construction_eligible':None,'construction_target_error':'final_climb_includes_descent',
              'note':'96 screening questions are 24 physical configurations with paired heights and two wordings, not 96 independent situations. No trace selected.',
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RUN.glob('*cases.json'))}}
    write_new(RUN/'question-manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
