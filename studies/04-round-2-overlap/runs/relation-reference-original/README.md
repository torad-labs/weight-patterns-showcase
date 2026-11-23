# relation-reference-original

The frozen reference of the first relational construction (`records/retrospectives/2026-11-15-relational-gradient-specification-astra.md`), compiled from the original alone. Declared and compiled November 15, 2026 by `run.sh` (the arrays on card 0 as arithmetic, the reference on the processor); no engine ran, no model was changed, no answer was read.

## What is here

- `declaration.json`: the construction, its bindings and its prediction, written before the compile (the preparation's scratch compile is disclosed in it).
- `reference/reference.json`: the ten members, the thirty recipients, each question's regime and number, each recipient's mate and the candidates it was chosen among, the tie rule, the coverage, the masses.
- `reference/tolerance.npy` ([merge, original row], float32, sha256 `6ae1ff5f2391…`) and `reference/tolerance-contributions.npy` (each member's part of it). Bulk: in the bucket, not in git.
- `reference/calibration.json`: the loss on the original, by answer, by group, by layer, with the turn's part and the gain's apart; and on the matched sweep's model on its own answers.
- `arrays/`: the tables the compile read (`relation_mates.py`). Bulk: in the bucket.
- `result.json`: the prediction beside the result.

## What it reads on the original

The objective is 0.81871, the sum over the thirty answers. The wrong answers' mean loss is 0.0355, the members' 0.0156, the other successes' 0.0262. A wrong answer stands above a member in 0.82 of pairs; 3 of the fifteen stand above every member. Before the tolerance the distance is 0.0577 for the wrong answers and 0.0278 for the members, of which the gain is 0.00089 and 0.00042: the turn carries nearly all of it. The prompts' rows carry 0.00041 and 0.00043; the answers' rows 0.0464 and 0.0215.

By answer, the loss times a thousand, least first: e004-h1 0.01, d01-h1 0.01, d08-h3 3.27, d10-h3 6.76, d11-h3 9.15, e064-h1 15.39, e076-h1 16.26, e023-h1 16.72, e029-h1 17.61, e046-h3 19.47, d05-h1 19.78, e045-h1 19.82, e078-h1 22.16, e092-h1 23.06, d10-h1 23.30, e068-h1 24.33, d06-h1 24.35, d15-h1 25.60, e070-h1 26.45, e094-h1 28.33, e095-h1 33.28, e039-h3 33.92, d17-h1 33.96, e020-h1 40.41, e034-h3 44.66, e071-h1 46.32, e042-h3 47.23, d00-h3 49.17, e087-h3 68.57, e001-h3 79.39.

The two members that stand highest, e071-h1 and e039-h3, have mates of another route (e064-h1 and e004-h1): no member of their own route shares their regime. The reference is a provisional hypothesis (Astra), and this is where it shows.

## What it is not

No target of state values, no window, no threshold of closure and no verdict. A later model is measured under these frozen bytes (`relation_reference.py frozen=`); the reference is never compiled again, because a card's sums do not repeat bit for bit: the declared compile and the preparation's, both on card 0, agree on every number read and differ in bytes.
