"""Publish A's manually read baseline classifications with exact source spans.

This is an annotation exporter, not an automatic physics grader. Each diagnosis
below was made from the complete saved response. Byte/token references are
derived from the original token pieces and checked against the saved text.
"""
import collections
import hashlib
import json
from pathlib import Path

from pipeline.paths import ROOT
SOURCE = ROOT / 'runs/legs/gemma4-e2b/physics-weight-edit-2026-11-07'
OUT = ROOT / 'research/experiments/signal-trace-repair'

# category, exact excerpt, expected interpretation. Labels are about the first
# incorrect quantitative step, not the first potentially misleading prose label.
ANNOTATIONS = {
    'cart-base': ('downhill_gain_subtracted', r'E_i - \Delta PE = 36 \text{ J} - 40 \text{ J} = -4', '36 + 40 = 76 J after descent'),
    'cart-changed': ('downhill_gain_subtracted', r'E_i - \Delta PE = 36 \text{ J} - 40 \text{ J} = -4', '36 + 40 = 76 J after descent'),
    'sled-base': ('downhill_gain_subtracted', r'KE_i - PE_{loss} = 50 \text{ J} - 40 \text{ J} = 10', '50 + 40 = 90 J after descent'),
    'sled-changed': ('downhill_gain_subtracted', r'KE_i - PE_{loss} = 50 \text{ J} - 40 \text{ J} = 10', '50 + 40 = 90 J after descent'),
    'bead-base': ('reference_energy_misallocated', r'Potential Energy at the bottom of the rough section ($PE_1$): $mgh_1', 'The bottom was defined as h=0; the lost 30 J becomes kinetic, not bottom potential. Later subtracts it a second time.'),
    'bead-changed': ('downhill_gain_subtracted', r'E_i - PE_{loss} = 8 \text{ J} - 30 \text{ J} = -22', '8 + 30 = 38 J after descent'),
    'coaster-base': ('downhill_gain_subtracted', r'KE_i - PE_{loss} = 40 \text{ J} - 100 \text{ J} = -60', '40 + 100 = 140 J after descent'),
    'coaster-changed': ('downhill_gain_subtracted', r'KE_i - PE_{loss} = 40 \text{ J} - 100 \text{ J} = -60', '40 + 100 = 140 J after descent'),
    'block-base': ('numerical_success', None, '54 + 30 - 24 = 60 J; required 30 J; YES. Reversed top/bottom and potential-gain labels remain.'),
    'block-changed': ('final_climb_includes_descent', r'h_{final} = 1 \text{ m} (\text{descent}) + 3 \text{ m} (\text{climb}) = 4', 'Final climb is 3 m from its own bottom; required 3*10*3=90 J. Earlier numerical energy chain is correct; labels are inconsistent.'),
    'trolley-base': ('initial_kinetic_arithmetic', r'(8 \text{ m/s})^2 = 128', '0.5*2*8^2 = 64 J, not 128; later also subtracts downhill gain.'),
    'trolley-changed': ('initial_kinetic_arithmetic', r'(8 \text{ m/s})^2 = 128', '0.5*2*8^2 = 64 J, not 128; later also subtracts downhill gain.'),
    'cart-paraphrase': ('numerical_success', None, '36 + 40 - 16 = 60 J; required 40 J; YES. Reference-height and initial-climb-speed prose is inconsistent.'),
    'cart-paraphrase-changed': ('initial_kinetic_arithmetic', r'(6 \text{ m/s})^2 = 18', '0.5*2*6^2=36 J, not 18; later also omits descent gain.'),
    'coaster-paraphrase': ('reference_energy_misallocated', r'$E_{initial} = E_{bottom} \implies 40 \text{ J} = \frac{1}{2} m v_{bottom}^2 + 100', 'Lost height increases kinetic energy to 140 J; cannot place 100 J potential at the bottom relative to that same bottom.'),
    'coaster-paraphrase-changed': ('incomplete_correct_available_energy', None, '40+100=140 and 140-15=125 J are correct; response caps before required-energy evaluation and verdict. Not a complete success.'),
    'block-paraphrase': ('initial_kinetic_arithmetic', r'(6 \text{ m/s})^2 = 45', '0.5*3*6^2=54 J, not 45. Earlier prose also confuses entry/exit of descent.'),
    'block-paraphrase-changed': ('downhill_gain_omitted', r'E_{\text{bottom}} = \frac{1}{2} m v_i^2', 'Bottom energy must include 30 J descent gain: 54+30=84 J, then 60 J after dissipation.'),
    'cart-arithmetic-base': ('numerical_success', None, 'A=60, B=40, YES.'),
    'cart-arithmetic-changed': ('numerical_success', None, 'A=60, B=80, NO.'),
    'coaster-arithmetic-base': ('numerical_success', None, 'A=125, B=150, NO.'),
    'coaster-arithmetic-changed': ('initial_kinetic_arithmetic', 'A = 2*5*16 + 100 - 15', 'One-half became 2; first term is 40, not 160, and A=125.'),
    'block-arithmetic-base': ('initial_kinetic_arithmetic', 'A = 1.5 * 3 * 36 + 30 - 24', 'Mass was applied twice: (0.5*3)*36 = 54, not 1.5*3*36=162.'),
    'block-arithmetic-changed': ('numerical_success', None, 'A=60, B=90, NO.'),
    'block-m2-base': ('initial_kinetic_arithmetic', r'(6 \text{ m/s})^2 = 18', '0.5*2*6^2=36 J; later subtracts downhill gain and caps.'),
    'block-m2-changed': ('reference_energy_misallocated', r'KE_{B, \text{before}} = 36 \text{ J}', 'Kinetic energy after descent is 56 J. The response retains 36 J kinetic and assigns 20 J bottom potential. Final triple is nevertheless correct.'),
    'block-m4-base': ('numerical_success', None, '32+40-12=60 J; required 40 J; YES. Reversed top/bottom labels remain.'),
    'block-m4-changed': ('reference_energy_misallocated', r'Kinetic Energy ($KE_1$) = $32 \text{ J}$', 'Kinetic energy after descent is 72 J, not 32. Later subtracts 40 J again. Final triple is nevertheless correct.'),
}


def main():
    cases = json.loads((SOURCE / 'cases.json').read_text())
    prior = {r['id']: r for r in json.loads((SOURCE / 'original-assessment.json').read_text())['rows']}
    assert {c['id'] for c in cases} == set(ANNOTATIONS), 'Classify every source case exactly once'
    rows = []
    for case in cases:
        cid = case['id']
        path = SOURCE / 'original-plain' / cid / 'result.json'
        raw = path.read_bytes()
        result = json.loads(raw)
        content = bytes.fromhex(result['content_bytes_hex'])
        if 'contentSha256' in prior[cid]:
            assert hashlib.sha256(content).hexdigest() == prior[cid]['contentSha256']
        category, excerpt, interpretation = ANNOTATIONS[cid]
        spans = []
        offset = 0
        for step in result['steps']:
            piece = bytes.fromhex(step['piece_bytes_hex'])
            if offset == len(content):
                break  # final EOG is recorded but is not part of response text
            assert content[offset:offset+len(piece)] == piece
            spans.append((offset, offset+len(piece), step['step']))
            offset += len(piece)
        assert offset == len(content)
        anchor = None
        if excerpt is not None:
            quoted = excerpt.encode()
            start = content.find(quoted)
            assert start >= 0, (cid, excerpt)
            end = start + len(quoted)
            decisions = [i for a, b, i in spans if a < end and b > start]
            anchor = {'excerpt': excerpt, 'content_byte_start': start,
                      'content_byte_end_exclusive': end,
                      'first_decision': decisions[0], 'last_decision': decisions[-1],
                      'interpretation': 'Text-location evidence, not the onset of internal computation'}
        rows.append({'id': cid, 'group': case['group'], 'first_quantitative_error': category,
                     'target_error_member': category == 'final_climb_includes_descent',
                     'anchor': anchor, 'expected_interpretation': interpretation,
                     'full_response_review': 'Read complete saved response; manual classification',
                     'historical_judgment': prior[cid],
                     'historical_content_hash_present': 'contentSha256' in prior[cid],
                     'capped': result['stopped_limit'],
                     'decisions': len(result['steps']), 'source': str(path.relative_to(ROOT)),
                     'source_sha256': hashlib.sha256(raw).hexdigest(),
                     'content_sha256': hashlib.sha256(content).hexdigest()})
    counts = dict(collections.Counter(r['first_quantitative_error'] for r in rows))
    report = {'status': 'baseline audit; no new inference or internal trace finding',
              'scope': '28 historical original-BF16 responses; first wrong quantitative transformation separated from misleading prose',
              'source_cases': str((SOURCE / 'cases.json').relative_to(ROOT)),
              'source_cases_sha256': hashlib.sha256((SOURCE / 'cases.json').read_bytes()).hexdigest(),
              'counts': counts, 'capped': sum(r['capped'] for r in rows),
              'new_runs': 0, 'rows': rows}
    (OUT / 'A-baseline-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    text = ['# A: baseline error audit', '',
            'November 7, 2026 (CT). All 28 complete saved original-model responses were read. '
            'This is response classification, not a new generation or an internal trace finding.', '',
            '**The exact target error appears once in the historical 28:** `block-changed` '
            'adds the descent distance to the final climb. The predominant downhill-subtraction '
            'error is a different quantitative departure. We need additional same-error examples '
            'before claiming a many-response overlap for A.', '',
            'The table identifies the first incorrect quantitative transformation. Some responses '
            'mislabel top/bottom or potential/kinetic energy earlier; those inconsistencies remain '
            'in the original judgments and explanations. The seven numerical successes include '
            'four arithmetic controls and three physics responses with inconsistent prose. '
            'The 4 kg changed case has a correct final triple but an incorrect intermediate allocation.', '',
            '| First quantitative departure | Responses |', '| --- | ---: |']
    text += [f'| {k} | {v} |' for k,v in counts.items()]
    hashes_present = sum(r['historical_content_hash_present'] for r in rows)
    text += ['', f"Capped responses: {report['capped']}/28. The exporter checks the {hashes_present} "
                  'previously recorded content hashes, computes source/content hashes for all 28, '
                  'and verifies token-piece reconstruction for all 28. It maps each quoted error '
                  'back to its original token decisions.', '',
             '| Case | Classification | Text decision span | Interpretation |', '| --- | --- | --- | --- |']
    for row in rows:
        a=row['anchor']; span=f"{a['first_decision']}–{a['last_decision']}" if a else '—'
        text.append(f"| {row['id']} | {row['first_quantitative_error']} | {span} | {row['expected_interpretation']} |")
    text += ['', 'Decision spans locate wording. They are not an assumption that the relevant '
             'internal process starts there. Capture and compare the entire preceding progression.', '',
             '[Exact excerpts, byte ranges, token spans and source hashes](A-baseline-audit.json). '
             '[Experiment proposal and results notebook](A.md).', '']
    (OUT / 'A-BASELINE-AUDIT.md').write_text('\n'.join(text))
    print(json.dumps({'counts': counts, 'capped': report['capped'], 'verified_responses': len(rows)}, indent=2))


if __name__ == '__main__':
    main()
