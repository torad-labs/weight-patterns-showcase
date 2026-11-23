"""What is left of the a2-pattern module: what other programs take from it.

KINDS      for each kind of matrix (attention output, feed-forward down): the engine's labels of its input, its tensor, its norm scale, the state it adds to,
           its contribution and the merged state, and the index of its merge inside the layer
WIDE       the wide overlap's run folder
BATCH      how many answers are recorded before their raw streams are used and deleted
model_of   where a run keeps its edited model
questions  the recorded cases of the wide overlap and of round one, each with its expected value and where its original recording is

The edit itself (`record`, `build`, `replay`, `moved`, `asked`, with `shared_gap`, `unit` and `travel`, and the module's commands) left this file on
November 15, 2026. No token placed it: it took each wrong answer's step at every place of the trace-overlap axis, which the computation lays. It left
because it is a construction on the values of the stream, a kind withdrawn that day, and because nothing used it: at every place and merge it added the
shared part of the gap between the two groups' average turns to the direction of an answer's merged state and asked the matrix for the contribution that
gives the state that direction, and it read the result by how far one group's average pattern travelled towards the other's. The trace this work corrects is
what each merge does to its input, not where the stream stands after it. The runs' records stay as history (studies/04-round-2-overlap/runs/a2-pattern, a2-sent and a2-diverge;
each one's retired-programs.json names the commits that hold the code). The commands that had the edited model answer the questions and judged the answers
had left on November 14, 2026.
"""
import json

from pipeline.paths import ROUND_TWO_RUNS

WIDE = ROUND_TWO_RUNS/'wide-overlap'
BATCH = 10  # answers recorded before their raw streams are used and deleted; only bounds the disk in use
# kind -> (matrix input, tensor, norm scale, incoming state, contribution, merged state, index of the merge inside its layer)
KINDS = {'attention': ('kqv_out', 'attn_output', 'post_attention_norm', None, 'attn_post_norm', 'attn_out', 0),
         'feedforward': ('ffn_geglu', 'ffn_down', 'post_ffw_norm', 'attn_out', 'ffn_post_norm', 'pe_in', 1)}


def model_of(run):
    return run/'model'/f'{run.name}.gguf'


def questions():
    wide = json.loads((WIDE/'members.json').read_text())
    first = json.loads((ROUND_TWO_RUNS/'full-overlap'/'questions.json').read_text())
    cases = json.loads((ROUND_TWO_RUNS/'full-overlap'/'cases.json').read_text())
    case = {c['id']: c for c in cases}  # some are stored as tokens and some as their text; the recorder takes either
    rows = [dict(id=m['id'], set='overlap', group='wrong' if m['role'] == 'failure' else 'correct', expected=m['expected'],
                 case=dict(id=m['id'], tokens=m['tokens'], max_tokens=m['max_tokens']), original=WIDE/'source'/'original-gpu0'/m['id']/'result.json') for m in wide]
    rows += [dict(id=q['id'], set='round one', group=None, expected=q['expected'], case=case[q['id']],
                  original=ROUND_TWO_RUNS/'full-overlap'/'capture'/q['id']/'result.json') for q in first]
    return rows
