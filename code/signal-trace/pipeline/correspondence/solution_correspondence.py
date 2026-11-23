"""Order-preserving response correspondence; no model-state or edit fitting.

The full source histories remain necessary. A token/section match supplies a
comparison index, not evidence of matching hidden computation or physical meaning.
"""
import collections
import hashlib
import re


def mandatory_matches(left, right):
    """Return the intersection of every longest common subsequence alignment.

    Forward/backward lengths enumerate which matched pairs can occupy each rank
    on an optimal path. A rank with one possible pair is compulsory. A rank with
    several pairs is ambiguous; no convenient traceback selects one for us.
    """
    n, m = len(left), len(right)
    forward = [[0] * (m + 1) for _ in range(n + 1)]
    backward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            forward[i + 1][j + 1] = (forward[i][j] + 1 if left[i] == right[j]
                                     else max(forward[i][j + 1], forward[i + 1][j]))
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            backward[i][j] = (backward[i + 1][j + 1] + 1 if left[i] == right[j]
                               else max(backward[i + 1][j], backward[i][j + 1]))
    ranks = collections.defaultdict(list)
    length = forward[n][m]
    for i in range(n):
        for j in range(m):
            if left[i] == right[j] and forward[i][j] + 1 + backward[i + 1][j + 1] == length:
                ranks[forward[i][j] + 1].append((i, j))
    return [ranks[k][0] for k in sorted(ranks) if len(ranks[k]) == 1]


def heading_role(heading):
    h = heading.lower().removeprefix('**').strip()
    if h.startswith('initial state'):
        return 'initial'
    if h.startswith(('state after descent', 'after descent')):
        return 'descent'
    if h.startswith(('state after rough section', 'after rough section')):
        return 'after_loss'
    if h.startswith('final state'):
        return 'destination'
    if h.startswith('available energy'):
        return 'available_summary'
    if h.startswith('required energy'):
        return 'required_summary'
    if h.startswith(('comparison', 'energy check', 'verdict')):
        return 'comparison'
    return None


def annotate(result, *, prompt_tokens):
    if not isinstance(prompt_tokens, list) or not all(isinstance(t, int) for t in prompt_tokens):
        raise ValueError('The actual supplied prompt-token IDs are required, not a token count')
    text = result['content']
    content = text.encode('utf-8')
    boundaries = [(0, 'opening', '')]
    for match in re.finditer(r'^\*\*[^\n]*', text, re.M):
        boundaries.append((len(text[:match.start()].encode('utf-8')),
                           heading_role(match.group()), match.group()))
    fields = list(re.finditer(r'^Available energy\s*=', text, re.M))
    if len(fields) == 1:
        boundaries.append((len(text[:fields[0].start()].encode('utf-8')), 'final_fields', ''))
    boundaries.sort()
    # A response beginning directly with a heading has no opening passage.
    boundaries = [x for i, x in enumerate(boundaries)
                  if i + 1 == len(boundaries) or boundaries[i + 1][0] != x[0]]
    counts = collections.Counter(x[1] for x in boundaries if x[1] is not None)
    sections = [dict(index=i, start_byte=start,
                     end_byte=boundaries[i + 1][0] if i + 1 < len(boundaries) else len(content),
                     role=role, heading=heading, automatic=role is not None and counts[role] == 1)
                for i, (start, role, heading) in enumerate(boundaries)]
    tokens, chunks, offset = [], [], 0
    for step in result['steps']:
        raw = bytes.fromhex(step['piece_bytes_hex'])
        end = offset + len(raw)
        section = next((s for s in sections if s['start_byte'] <= offset < end <= s['end_byte']), None)
        piece = raw.decode('utf-8', errors='replace')
        tokens.append(dict(prediction=step['step'], input_position=step['input_position'],
                           selected=step['selected'], piece=piece, start_byte=offset, end_byte=end,
                           section=section['index'] if section is not None else None,
                           numeric=bool(re.fullmatch(r'\s*[0-9]+(?:\.[0-9]+)?\s*', piece))))
        chunks.append(raw)
        offset = end
    if b''.join(chunks)[:len(content)] != content:
        raise ValueError('Recorded token pieces do not reproduce the response content')
    if offset < len(content):
        raise ValueError('Recorded token pieces end before the response content')
    return dict(content_sha256=hashlib.sha256(content).hexdigest(),
                prompt_tokens=prompt_tokens, sections=sections, tokens=tokens,
                end_tokens=[t['prediction'] for t in tokens if t['start_byte'] >= len(content)],
                annotation_limit='Textual roles identify comparison context, not hidden calculation onset')


def correspond(left, right):
    lsections = [s for s in left['sections'] if s['automatic']]
    rsections = [s for s in right['sections'] if s['automatic']]
    section_pairs = mandatory_matches([s['role'] for s in lsections], [s['role'] for s in rsections])
    pairs = []
    same_prompt = left['prompt_tokens'] == right['prompt_tokens']
    shared = 0
    for a, b in zip(left['tokens'], right['tokens']):
        if a['selected'] != b['selected']:
            break
        shared += 1
    # Selection d does not enter the model until execution d+1. At the first
    # different selection the generated input history is still shared.
    prefix_predictions = min(shared + 1, len(left['tokens']), len(right['tokens']))
    def row(a, b, section, ls, rs, reason, same_history=False):
        return dict(left_prediction=a['prediction'], right_prediction=b['prediction'],
                    section=section, left_section=ls, right_section=rs,
                    left_piece=a['piece'], right_piece=b['piece'],
                    left_selected=a['selected'], right_selected=b['selected'],
                    left_input_position=a['input_position'], right_input_position=b['input_position'],
                    numeric_pair=a['numeric'] and b['numeric'], reason=reason,
                    identical_incoming_history=same_history)
    for i in range(prefix_predictions):
        a, b = left['tokens'][i], right['tokens'][i]
        ls = left['sections'][a['section']] if a['section'] is not None else None
        rs = right['sections'][b['section']] if b['section'] is not None else None
        role = ls['role'] if ls and rs and ls['role'] == rs['role'] else 'shared_generated_prefix'
        pairs.append(row(a, b, role, a['section'], b['section'],
                         'identical_incoming_history' if same_prompt else 'shared_generated_history_different_prompt',
                         same_prompt))
    def key(token):
        return ('numeral',) if token['numeric'] else ('token', token['selected'])
    for li, ri in section_pairs:
        ls, rs = lsections[li], rsections[ri]
        lt = [t for t in left['tokens'] if t['section'] == ls['index'] and t['prediction'] >= prefix_predictions]
        rt = [t for t in right['tokens'] if t['section'] == rs['index'] and t['prediction'] >= prefix_predictions]
        for i, j in mandatory_matches([key(t) for t in lt], [key(t) for t in rt]):
            a, b = lt[i], rt[j]
            pairs.append(row(a, b, ls['role'], ls['index'], rs['index'],
                             'mandatory_numeric_position' if a['numeric'] else 'mandatory_same_token'))
    lpaired = {p['left_prediction'] for p in pairs}
    rpaired = {p['right_prediction'] for p in pairs}
    return dict(left_content_sha256=left['content_sha256'], right_content_sha256=right['content_sha256'],
                same_prompt=same_prompt, shared_generated_prefix_tokens=shared,
                shared_generated_history_predictions=prefix_predictions, predictions=pairs,
                unmatched_left=[t['prediction'] for t in left['tokens'] if t['prediction'] not in lpaired],
                unmatched_right=[t['prediction'] for t in right['tokens'] if t['prediction'] not in rpaired],
                unmatched_reason='Absent, ambiguous or differently ordered passage; boundary/end token; no interpolation',
                limit='Textual correspondence alone does not establish equal physical roles or a capability trace')
