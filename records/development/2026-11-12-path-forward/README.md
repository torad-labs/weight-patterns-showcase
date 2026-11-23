# The path forward, November 12, 2026

Marcos, to this session, about 1:12 AM CT: "spend a good few turns brainstorming with weight-research-fable, weightr-research-astra on the
path forward. Take your guys time, plan a robust path forward, then act on it, dont wait on my approvals". Before it: "We have clear evidence
that corecting the path of computation corrects capabilities ... We still dont know how to achieve it reliably, but we've already proven it
owkrs".

This folder holds what weight-research-perfomance measured and built for that plan. The objective, the target and every declaration are
weight-research-astra's (`journal/2026-11-12-correction-not-protection.md`, `records/retrospectives/2026-11-12-correction-objective-astra.md`);
the solver and card 0 are weight-research-fable's. Everything here reads the original's recordings and the 30 fitting answers. The answers no
fit uses are never generated or read, reserved-40b stays sealed, and nothing here is a bound, a loss, a target or a selector.

## What was measured

**A model has to be read on the text it writes.** `own_compare.py` (in `../2026-11-10-speed-review/as-we-go/`) lets a model answer the fitting
questions by itself on card 1 and lays its own traces on the round's axis. Three models were read with it: the unchanged original (what the
instrument itself leaves), the state after controlled-bf16-4's nineteen matrices, and the first training's model `a2-path`. The captures are
kept as the controls of `runs/correction-at-19` (`own/controls/{original,after-19,a2-path}`, each with its provenance and checksums); the
records are in `own-readings/`, side by side in `own-three-models.txt`.

| on its own answers, places 142 to 313 | original | after 19 matrices | a2-path |
|---|---|---|---|
| wrong group, part of the gap along | +0.063 | +0.347 | +0.807 |
| wrong group, beside the gap (gaps) | 0.23 | 0.45 | 1.13 |
| correct group, travelled (gaps) | 0.17 | 0.44 | 1.26 |
| complete answers, wrong and correct of 15 | 0 and 15 | 3 and 10 | 8 and 11 |

Replayed on the ORIGINAL's text the same two edited models read 0.027 and 0.385, which is what this session first reported as "the protected
sweeps held the model still". That was wrong by an order of magnitude for the protected state, and it is corrected in the speed review's
README. `a2-path` read this way reproduces its own record (0.79 along and 1.04 beside on 60 answers on card 0).

**Where the rungs' own traces lie** (`rung_map.py`, `rung-map.txt`; `given_text_map.py`). On their own text `a2-path`, `c2-path` and
`c2-protected` put both groups on the correct side at places 142 to 199 at every depth, with only layers 0 to 8 written, and from about place
210 both groups read near the middle together. The claim made earlier that night, that no run had walked the path in depth, is withdrawn.

**What one answer shows** (`sharp_stretch.py`, `sharp-stretch.txt`). At places 172 to 201 one answer's coordinate along the gap separates the
original groups by 8 to 13 standard deviations; elsewhere by about 2. In the edited rungs 51 to 55 of 60 answers read correct there, and the
answers that are wrong another way read like the right ones. What the overlap's gap shows per answer is the descent-sign decision; the errors
left after a repair are further along the answer.

**Do texts repeat, and how firm are the decisions** (`token_histories.py`, `token-histories.txt`; `rollout_tokens.py`, `rollouts/`).
- Greedy decoding repeats exactly on each card: on card 0 the original's second answers carry the recordings' tokens in 60 of 60 answers; on
  card 1 two rollouts by two recorder modes agree in 30 of 30. Between the cards 9 of 30 differ.
- The original's decisions (60 recordings, 30,546 of them) have a median margin of 9 to 10 between the first and second choice, but 5.9% of
  the wrong answers' decisions and 4.1% of the correct answers' lie under 0.5, and 0.37% and 0.11% under 0.02. One wrong answer has a median
  of 26 decisions under 0.5, one correct answer 17. The two cards' margins differ by 0.0065 at the median and 0.058 at the 99th centile on
  the same text, and all 9 cross-card rewordings start at a card-0 margin under 0.08.
- After nineteen protected matrices 22 of the 30 answers write other tokens than the original on the same card, the first one at step 25 to
  363. In 20 of the 22 the original's margin at that decision was under 0.5 (13 at 0.1 or less), and the new token was the original's second
  or third choice in all 22. `a2-path` rewords all 30, 19 of them from step 3.
- What it means for the plan: texts change early and mostly by wording at near-ties, not by decisions. Correcting on the model's own text
  therefore rests on registering a reworded answer onto the fixed axis, and that registration's floor has to be known answer by answer.

**Is a given text still the model's own** (`history_check.py`). Any pass that runs whole writes the engine's own first choice beside each
supplied token. Replaying the card-0 recordings on card 1 flags exactly the 9 answers the card-1 rollout rewords, at the same first steps
(18, 69, 289, 300, 337, 379, 689 and 72, 294), in 138 s for the 30 answers. So the check needs no generation and no second arithmetic.

**One answer at a time** (`members.py`): each fitting answer's distance from the fixed original correct pattern at every place and merge, its
coordinate along the gap, what lies beside, its log size, and its movement from its own recording, with how the registration covers each place.
The three controls' values are kept beside them (`own/controls/<name>/members.npz`), the reference in `members-cache/` (out of git; `members.py
reference` and `held-out` make it again in 15 minutes).

- *What a radius around the correct pattern calls agreement* (`member-envelope-held-out.txt`; the 60 original recordings, each read against
  patterns made without it; asked by Astra). At places 172 to 201, a radius equal to the correct answers' root-mean-square distance puts an error
  on all 30 wrong answers but also on a third of the correct answers' cells (8 to 10 of 30 correct answers in over half their cells). A radius
  equal to the correct answers' greatest distance puts none on a correct answer and sees 1 of 30 wrong ones. The reason is that beside the line
  through the two patterns the wrong answers lie no farther from the correct pattern than the correct answers do: the whole distance is mostly an
  answer's own context. Taking only the part along the gap, the greatest-distance radius (no error on any correct answer) leaves 29 to 30 of the
  30 wrong answers in error over most of the stretch, in 98.6% of cells at layers 14 to 23. The log size identifies no single answer.
- *What rewording alone does* (`member-floor.txt`; the unchanged original on card 1 against its own recordings). The 21 answers with the same
  tokens read the same to three decimals: the instrument is exact on the same text. The 9 reworded answers move 1 to 5 member variances from
  their own recordings from the first other token on, as far as a different answer of the same group would lie. An answer's context does not
  survive rewording, so nothing answer-specific can be carried from the original across it.
- *On its own text an answer's coordinate reads what the answer does* (`member-outcomes.txt`; three models, 90 answer-readings, 61 reworded).
  Every answer that subtracts the descent reads between -0.10 and +0.23 at places 172 to 201, with one at +0.44 (24 readings). Every answer that
  states the available energy correctly reads +0.51 or more, 52 of 55 at +0.90 or more, including the 15 formerly wrong answers that now get it
  right (1 in the unchanged original on card 1, 4 after nineteen protected matrices, 10 in `a2-path`). One of those is the unchanged original
  itself: `expanded-001-h3` parts from its recording at step 18 on a margin of 0.004 and then treats the descent correctly, with the same weights.
- *The projected envelope on the models' own answers* (`member-projected.txt`, `member-projected/`; asked by Astra, specification
  `records/retrospectives/2026-11-12-next-loop-astra.md`). Every fitting answer of the three controls, at every place and merge, along the
  original gap against the original correct answers' greatest leave-one-out projected distance, no multiplier, the last merge as the output
  norm passes it on. The envelope can see the original wrong pattern almost only on the sign's line: the visible error is 0.38 of the squared
  gap at places 169 to 201 and 0.001 at the places before the sign. On that line it holds on real histories for the failure it was made
  from: the 24 readings of an answer that subtracts the descent on its own text have an error of 0.14 to 0.48 of the squared gap, and 46 of
  the 47 readings of an answer without a failure have 0.000 to 0.002 (one reworded original 0.062). As a target it fails three ways: it asks
  nothing where the decision is made; it does not see other failures on the line (`a2-path`'s six answers with another wrong available
  energy read 0.001 to 0.002); and away from the line it marks answers without a failure (12 of 19 in `a2-path` read 0.07 to 0.33 at places
  0 to 148), which is their wording.
- What this says for the plan: for one answer, "agrees with the correct computation" is measurable along the recurring contrast and not in the
  whole vector; the whole vector belongs to the group pattern, where the answers' contexts average out. That is evidence for Astra's decision on
  the answer-level target, not a decision.

**The specification's whole-model L of the three controls** (`whole_model_l.py`, `whole-model-l.txt`; the fields beside each control in
`whole-model-l/`). Read with the reference whose last merge is what the output norm passes on (Astra's ruling of about 2:20 AM CT; record
sha256 74d1eafe…), through the solver's own `correction.passed_on`. With no movement the tool reproduces the record's own objective.

| own traces, card 1, 30 fitting answers | L | direction | size | wrong group | correct group |
|---|---|---|---|---|---|
| the original recordings (the reference's own) | 0.248 | 0.121 | 0.127 | 0.248 | 0 |
| original | 0.247 | 0.112 | 0.135 | 0.244 | 0.003 |
| after 19 matrices | 0.217 | 0.069 | 0.148 | 0.170 | 0.046 |
| a2-path | 2.704 | 1.000 | 1.705 | 1.325 | 1.379 |

The same three read with the reference as first made (the last merge raw) are in `whole-model-l-raw-terminal.txt`: 0.247, 0.220 and 2.701.
Only the last of the 105 merges differs between the two references, and no reading's order or sign changes with it.

The unchanged original reads within 0.6% of its recordings, so at the level of the groups the instrument's floor is small. The state after
nineteen protected matrices closes 0.127 of L: its direction part falls 43% and its size part rises 17%, and a fifth of what is left is the
correct group's. `a2-path` reads 10.9 times the original's L in both groups alike: at places 172 to 201 its wrong group's mean is on the
correct pattern along the gap (0.90 to 1.08) while its L there is still above the original's (1.07 to 1.59 times), all of it beside the gap
and in size, and at places 0 to 141, where it rewords the opening of 19 of 30 answers, it reads 25 to 121 times the original's L there. Its repair and its other
displacement are both real. Two cautions on any mean: after nineteen matrices the wrong group's mean at places 172 to 201 is +0.38 along,
which answer by answer is four answers at +1.00 to +1.11, one at +0.86 and ten between -0.04 and +0.23; and outside that stretch a reworded
answer's coordinate moves toward the middle with unchanged weights (the original's correct group reads -0.01 to -0.15 along with 2 of 15
answers reworded), so a mean's movement along the gap there is partly rewording. After nineteen matrices the correct group's mean reads
-0.05 to -0.10 along at places 0 to 201 and -0.20 to -0.40 from place 202 on.

**Where the capability is decided, in the text and in depth** (`sign_decision.py`, `sign-decision/`). The gap is a difference of two group
patterns; part of it is the text the groups have read by then and part is what the model computes on the same text. The state before the
first merge is the embedding of the token a step reads, so the groups' difference there is text and nothing else.
- *The text* (`text-map.txt`; 60 recordings; shuffled groups stay under 9.2). The two groups read apart at 16 of the 480 places: the heading
  at 149 to 160, the sign at 169 and 183, then 219 to 233, 266 to 269 and the closing NO or YES at 479. At place 169 the wrong answers read
  ' -' (29 of the 29 that have the line) and the correct ones ' +' (26 of 26): "KE_i - PE_descent" against "KE_i + PE_descent". The heading
  before it is 'Kinetic Energy at bottom of descent' in 29 of 30 wrong answers; the correct answers write 'Total Energy before rough
  section' (16) or the wrong answers' own heading (11).
- *Before the sign the gap is the heading* (`own-text.txt`). One answer's coordinate along the gap at places 161 to 168 (layers 14 to 23),
  original on card 1: the wrong answers that write ' -' read -0.10 to +0.03; the correct answers with the wrong answers' heading +0.00 to
  +0.02; the correct answers with the other heading +1.26 to +1.36. After the sign (172 to 201) the same four correct answers read +0.78 to
  +1.10. So before the sign no envelope along the gap asks anything of a wrong answer, and a pull along it is a pull toward the other
  heading. (The fifteenth wrong answer, `expanded-001-h3`, writes ' +' on card 1 in another layout and reads +1.10 there.)
- *On the same text, the difference enters at layer 14's attention* (`same-line.txt`). 17 originals have the identical line before the sign
  (' descent ($KE_{bottom}$): $KE_i'), 10 wrong and 7 correct. Their separation at the nine steps of the line (squared distance of the two
  means over the spread; shuffled groups stay under 7.1): never above 3.7 at layers 0 to 8, single steps at layers 9 to 13 (10 to 11),
  every step of the line from layer 14's attention on (7 to 16), and at the step that chooses the sign 12 at layer 14, 18.7 at layer 20, 28
  at layer 23 and 43.5 at the last merge. That same-text difference is nearly orthogonal to the group gap at the same place: cosine -0.23
  to +0.12 through layer 18 and +0.17 to +0.34 from layer 20 on, at a seventh to a third of the gap's length. It agrees with
  weight-research-fable's turn map (the step at layer 14's attention, the band to layer 20) from the other side.
- *The decision in the engine's own scores* (`text-map.txt`, last lines). The score of ' +' less the score of ' -' at the step that chooses:
  -6.86 to -0.22 in the wrong answers (median -3.89), +0.07 to +8.46 in the correct ones; with the identical line, -4.22 to -1.10 against
  +0.07 to +4.35.
- *The same step on the same text, per model* (`sign-scores.txt`; `history_check.py` on the recordings' text, one plain pass of 140 to 160 s
  on card 1; the original reproduces its recordings within 0.2). After controlled-bf16-4's nineteen matrices the score changed by 0.00 at
  the median in both groups (-0.51 to +0.82 and -0.77 to +0.26), and ' +' scores higher in 0 of the 15 wrong answers. In `a2-path` it
  changed by +5.98 at the median in the wrong group (+3.50 to +9.67), ' +' higher in 11 of 15, and by +0.99 in the correct group. On their
  own text the nineteen-matrix state's newly correct answers reached the line by other wording (`discovery-10-channel-h1` writes the 'Total
  Energy' heading), and the correct answer it lost (`discovery-15-channel-h1`) is the one whose original margin was +0.07. So that
  configuration's movement of L and of the mean came through rewording before the line, with the decision where the original had it, and
  `a2-path` moved the decision itself. This is about that configuration; `a2-path`'s +6 is the technique moving the decision.
- What it means for the plan: the score is a per-answer, continuous reading of the capability on a fixed text, one plain pass per candidate,
  kept beside L as description (never a loss or a selector). And a contrast between answers that have read the same text isolates the
  decision, where the group gap mixes it with the correct answers' wording. The target is Astra's to decide.

**The cheaper reading between two full ones** (`dose_reading.py`, `dose-reading-check/`). It replays a kept capture's text on another model,
the pass run whole with every merge kept, and gives each answer the capture's own registration, so that what differs from the capture is the
weights alone, step for step; `whole_model_l.py` and `members.py` read its output as they read any capture. The engine's own first choice at
every step says from which step an answer's text is no longer what the model writes; the rows after it are a conditional replay, named as
such per answer and never counted as the model's own trace. Its check, 2:50 to 2:58 AM CT on card 1: the state after nineteen matrices
replayed on its own text flags 0 of 30 answers and reads L = 0.21661824412969002, the capture's own value to the last digit, every field
equal. It took 257 s of recorder, 379 s with the export, and 99 s for L, against 10 to 15 minutes for a full own reading before its L.

**What the pooled gap is made of** (`members.py decompose`, `pooled-contrast.txt`, `pooled-contrast/decomposition.json`; the 60 originals,
files only). With C and W the two groups' means and C_s, W_s the means of the members that share some text,
C - W = (C_s - W_s) + (C - C_s) + (W_s - W): the contrast within the shared text and two pooling terms. For the 17 originals that read the
identical line before the sign (7 correct, 10 wrong), at places 149 to 168 the within term is 0.03 of the pooled squared gap (0.01 to 0.05
place by place; correct pooling 1.54, wrong pooling 0.44), and it is still 5.9 times its own spread (11.6 at layers 14 to 23). At places 169
to 201 it is 2.35 of the pooled squared gap and 20.9 times its spread (33.0 at layers 14 to 23). For the 'Kin' heading (11 correct, 29
wrong) it is 0.44 before the sign and 1.44 after it. So before the sign the pooled gap is 94 to 100% a mixture of wordings, and from the
sign on the contrast within one text is larger than the pooled one. Astra's reading of it is in
`records/retrospectives/2026-11-12-next-loop-astra.md`: a files-only diagnostic, not a target and not a crop.

**Experiment A's one candidate, read** (`runs/correction-at-19`: `blk.9.ffn_down.weight` from the state after controlled-bf16-4's
nineteen matrices; ridge 2^-4, round 40 of 40, 11.46% of the matrix; the capture is in the run at `own/candidate/`, its conditional
readings and controls under `own/diagnostics/`). All on card 1, the 30 fitting answers, 3:43 to 4:55 AM CT. The run's own verdict is
Astra's, in its `result.json`: the declared weakening condition is met. What the readings show about why:

| reading | the state it started from (after-19) | the candidate |
|---|---|---|
| own answers complete, wrong group of 15 / correct group of 15 | 3 / 10 | 7 / 3 |
| own answers cut at the token limit | 0 / 0 | 3 / 6 |
| own answers' median length, steps | 453 / 486 | 682 / 735 |
| own-trace L (the recordings' own 0.24802) | 0.21662 | 5.19145 |
| of it, in the 28 merges before the written matrix | 0.04658 | 1.12802 |
| L on after-19's own text, the text held (`dose_reading.py`) | 0.21662 | 0.20664 |
| sign score on the recordings' text against the original, wrong (15) / correct (13) | 0.00 / 0.00, ' +' higher in 0 / 12 | +10.63 / +1.19, ' +' higher in 15 / 13 |
| the supplied token's margin, size of its change over 14,592 steps, median | 0.055 | 0.378 |
| first choices that became another token, of 14,582 | 66 | 616 |
| the question's positions, distance of the unit state from the original's at layer 9's feed-forward | 0.006 | 0.140 |

- *The decision moved, for every answer, on the old text.* The candidate's scores at the sign step are +4.98 to +8.66 for the wrong
  answers and +5.88 to +7.86 for the correct ones (the original: -6.90 to -1.04 and +0.15 to +7.00). By depth (`sign-decision/depth.txt`,
  each answer's state at that step against the original on the same card, as a part of the same-line difference): at layers 23 to 32 the
  wrong answers moved by +0.71 to +0.97 of it at the median where they read the identical line and +0.44 to +0.87 in other wording, and
  the correct answers by -0.13 to +0.21. `a2-path`, the repair that transferred, reads the same there (+0.75 to +1.19, +0.53 to +0.82;
  correct -0.12 to +0.46), and the nineteen-matrix state +0.08 to +0.14. At layers 14 to 19 both `a2-path` and the candidate move every
  identical-line answer AGAINST that difference (-0.4 to -1.7, correct and wrong alike), so the direction there is not what either
  follows. Along the difference lies 5 to 13% of the wrong answers' squared movement at layers 21 to 29; the rest is elsewhere.
- *The own-trace L of 5.19 reads another text first.* None of the 30 answers keeps its recording's tokens, and a fifth of L lies in merges
  whose weights are after-19's bit for bit; per merge it is 0.040 before the matrix and 0.053 from it on (the recordings' own 0.0020 and
  0.0025). With the text and registration held L falls 4.6% (direction 0.0687 to 0.0540, size 0.1479 to 0.1527; wrong group 0.170 to
  0.155, correct group 0.046 to 0.052), and it is identical before the matrix to the digit.
- *Where the text leaves.* On the recordings' text the candidate's first choice differs at step 3 in 26 of 30 answers: ' compact' against
  ' calculation'. ' compact' less ' calculation' is +0.02 to +0.50 in the original (median +0.25), +0.02 to +0.43 after nineteen matrices,
  -0.27 to +0.18 in `a2-path` and -1.20 to -0.76 in the candidate. The answer then takes a longer layout ("Here is the calculation to
  determine if the block reaches the destination", givens listed first), in which 9 of 30 run out of tokens.
- *The scores moved everywhere, most where the capability is decided* (`score_movement.py`, `sign-decision/score-movement.txt`). The size
  of the change of the supplied token's margin, median by stretch of places: 0.24 at 0-141, 0.86 at 142-171, 0.53 at 172-201, 0.36 to
  0.52 after (the nineteen-matrix state 0.02 and 0.07 to 0.09; `a2-path` 0.19, 0.80, 0.67, 0.31 to 0.40). At the sign step it is 28 times
  the median for the wrong group and 3 times for the correct one (`a2-path` 20 and 3). At the answer's steps 3 to 6 it is 1.0 to 1.3, the
  largest of the first twenty, and 320 of the 687 steps the original decides by under half a logit flipped.
- *The question's positions moved, and no cell reads them.* The specification of November 12 records question changes and gives them no
  target. The candidate moved the unit state at the question's positions by 0.140 at layer 9's feed-forward, 0.171 at layer 11, 0.021 at
  the last merge (after-19 0.006, 0.009, 0.002; the original read again on card 1 0.004 at layer 9); Fable's row-by-row reading of the
  matrix agrees (0.121 a row at the median, `../2026-11-12-correction-objective/rows-moved-correction-at-19.txt`). With the text held, the
  candidate's state at the answer's step 3 differs from after-19's by 0.005 at the matrix's merge and by 0.107 at layer 13's exit: at the
  opening the movement arrives through later layers' attention, not through the matrix's own output there. This fits the question's rows as
  the channel and does not prove it.

**Do the originals differ at the question's positions?** (`prompt_overlap.py`, `prompt-overlap/`; the 60 originals' kept rows, files only.)
Counted from the prompt's start or its end, the two groups' squared distance over the spread is 1.00 to 1.11 at the median position and
merge (1 is no difference; shuffled groups reach 5.9 to 6.2 somewhere by chance), 6 of 171 positions counted from the start and 11 from
the end are apart at some merge, four of them where the questions' own tokens differ, and the groups' squared distance there is 0.2 to
1.4% of the reference's squared gap at the answer's places 169 to 201. Where the model reads the question, the correct and the wrong
originals compute alike. Whether the question's positions enter the objective, and with what target, is Astra's decision.

**The opening, crossed inside each model** (`opening_controls.py`; declared by Astra as the run's `opening_diagnostic_phase`; records in
`runs/correction-at-19/own/diagnostics/opening-controls/`). The 26 fitting answers whose two own answers part at step 3 with ' compact'
and ' calculation' (12 wrong, 14 correct; the other 4 part elsewhere and are not forced). Each model is fed the three shared tokens and
the other model's fourth, token by token, and writes the rest by itself within the same limit of 768 tokens:

| complete answers, wrong 12 / correct 14 (cut at the limit) | opens ' compact' | opens ' calculation' |
|---|---|---|
| the state after nineteen matrices, no further change | 2 / 10 (0 / 0), its own answers | 3 / 2 (3 / 10), forced |
| the candidate | 4 / 4 (3 / 2), forced | 5 / 3 (2 / 5), its own answers |

  The unchanged starting state loses the correct group on the other opening by itself: 10 complete become 2, and 10 of 14 run out of
  tokens, some after subtracting the descent on a question it answers correctly in its own layout (`discovery-01-channel-h1`:
  "KE_bottom = KE_i - PE_1 = 16 J - 40 J = -24 J", then a re-evaluation to the limit). So the fall of the correct group in the
  candidate's own answers is reproduced with no weight changed, by one token. And the candidate held to ' compact' still completes 4 of
  14 where its starting state completes 10, with other failures than the descent's sign (descent omitted 3, other 2, climb plus descent
  2, verdict 1, subtracted 1, missing 1), and leaves the starting state's text again by step 11 in 20 of 26: its damage is not that one
  branch alone. Forced answers describe a branch; they replace no result. The check of the forced pass itself: each model fed its OWN
  fourth token writes its natural answer token for token in 26 of 26 cases (`start-own/`, `candidate-own/`; ended 4:59 and 5:19 AM CT).

**Experiment B's one candidate, read** (`runs/correction-at-17-original`: `blk.17.attn_output.weight` alone, from the ORIGINAL model; ridge
2^-3, round 25, 9.52% of the matrix; the capture is in the run at `own/candidate/`, the conditional readings under `own/diagnostics/`, the
script as it ran and its times in `own/logs/`). The 30 fitting answers on card 1, 4:49 to 5:29 AM CT, the files-only readings to 5:42 AM
CT, against the unchanged original read the same way on the same card (`runs/correction-at-19/own/controls/original`). The run's verdict is Astra's. What the readings show:

| reading | the original, read again on card 1 | B's candidate |
|---|---|---|
| own answers with the available energy right, wrong group of 15 / correct group of 15 | 1 / 15 | 8 / 15 |
| own answers that subtract the descent | 14 / 0 | 3 / 0 |
| own answers complete (available, required and verdict) | 0 / 15 | 6 / 12 |
| own answers with the verdict right | 1 / 15 | 11 / 15 |
| own answers cut at the token limit | 1 / 0 | 0 / 0 |
| own answers' median length, steps | 489 / 422 | 533 / 473 |
| projected error short of the correct pattern at places 169 to 201, median over the answers (a point on the wrong pattern: 0.382) | 0.429 / 0.000 | 0.010 / 0.002 |
| own-trace L (the recordings' own 0.24802) | 0.24664 | 0.42049 |
| of it, in the merges before layer 17, whose weights are the original's | 0.11325 | 0.21727 |
| L on the original's own card-1 text, the text held (`dose_reading.py`) | 0.24664 | 0.22418 |
| of it, from layer 17 on | 0.13339 | 0.11093 |

- *The capability moved in the answers the model writes by itself, and the correct group kept it.* Of the 15 wrong answers, 8 now state
  the right available energy (the original: 1) and 2 more add the descent with the kinetic energy doubled (`discovery-05-channel-h1`,
  `expanded-020-h1`: "128 J + 40 J = 168 J" where ½·2·8² is 64 J; these are the two answers that take the other opening at step 3); 3
  still subtract it (`expanded-042-h3`, `-046-h3`, `-070-h1`) and 2 omit it (`-034-h3`, `-094-h1`). All 15 correct answers keep the right
  available energy and the right verdict. What B cost is elsewhere: the REQUIRED energy is no longer right in 3 correct answers
  (`discovery-06` and `-11`: another number; `discovery-08-channel-h3`: the climb plus the descent, 120 J for 90 J) and in 1 wrong one
  (`expanded-087-h3`: climb plus descent; `discovery-10-channel-h1` has it as in the original), while 2 wrong answers whose required
  energy the original had wrong are right now (`expanded-020-h1`, `-095-h1`). No answer runs out of tokens, and the layout stays the recordings' ('Kin' or 'Total' heading at place 149 in 27
  of 30). For scale, `a2-path` on the same card: available energy right 10 / 12, complete 8 / 11, cut at the limit 2 / 2.
- *The trace agrees with the answers, answer by answer* (`member-projected.txt`, the projected envelope on each answer's own text). At
  places 169 to 201 the 8 wrong answers whose available energy is right lie at 0.000 to 0.009 of the squared gap short of the correct
  pattern, but for `expanded-087-h3` (0.109); the two with the doubled kinetic energy at 0.010; the two that omit the descent at 0.049
  and 0.075; the three that still subtract it at 0.175, 0.218 and 0.248. The correct group stays at 0.000 to 0.009 in 13 of 15;
  `discovery-00-channel-h3` (0.277) and `expanded-029-h1` (0.157) lie off it there though their answers are complete.
- *The own-trace L rose, and the rise reads the new text.* L is 0.42049 against the original's 0.24664 (direction 0.121 against 0.112,
  size 0.299 against 0.135; wrong group 0.276 against 0.244, correct group 0.144 against 0.003), so the run's declared prediction on L
  is not met. 0.104 of the 0.174 rise lies in the merges before layer 17, where no weight changed and the state on the same text is the
  original's to the bit (`state-distance-from-the-original.npz`: zero at all 51 merges): none of the 30 answers keeps its recording's
  tokens (first other step 3 to 171 for the wrong group, 11 to 264 for the correct one). The group patterns say the same
  (`own/logs/compare-with-the-original.out`): at layers 0 to 4 the wrong pattern lies +0.47 of the gap along it and the correct pattern
  0.63 of a gap from where the original's was, by the text alone. With the text held to the original's own answers L FALLS 9.1%, all of
  it from layer 17 on (0.13339 to 0.11093): wrong group 0.2441 to 0.2212 (layers 17 to 23: 0.0421 to 0.0257; places 142 to 171: 0.0505
  to 0.0416; 172 to 201: 0.0348 to 0.0294), correct group 0.0025 to 0.0029. So what an own-trace L, or any pooled own-trace pattern,
  says of an edit is mixed with what the edit made the model WRITE; the held-text reading and the per-answer one separate them.
- *The movement is where the capability is decided, and smaller elsewhere.* With the text held, the unit state at layer 17's attention
  lies from the original's by 0.150 for the wrong group and 0.043 for the correct one at places 172 to 201 (medians over the answers),
  0.125 and 0.043 at 142 to 171, 0.020 and 0.022 at 0 to 141, 0.076 and 0.039 at 202 to 259, and 0.066 to 0.087 against 0.054 to 0.064
  from place 260 on: late in the answer the two groups move nearly alike. It fades with depth (wrong group at 172 to 201: 0.189 at layer 19, 0.116 at 23,
  0.043 at 29, 0.021 at 34). On the recordings' text the sign score moved by +7.02 at the median for the wrong group (+2.00 to +12.21;
  ' +' higher in 12 of 15, not in `discovery-10-channel-h1`, `expanded-034-h3`, `-042-h3`) and by -0.34 for the correct group (' +'
  higher in 13 of 13); the supplied token's margin changed by 0.174 at the median over all steps (A: 0.378), 365 first choices became
  another token (A: 616), and the sign's movement is 40 times that median for the wrong group and 2 times for the correct one. The
  question's positions moved by 0.061 at layer 17 (A: 0.140 at layer 9), 0.041 at layer 23, 0.011 at the last merge.
- *What it is not.* These are the 30 fitting answers the reference was made from, read on card 1; the answers no fit uses were not
  generated and reserved-40b was not opened, so nothing here says the repair transfers.
- *By depth at the sign step, B moves the wrong answers the whole same-line difference and leaves the correct ones* (`sign-decision/depth.txt`,
  the recordings' text replayed with every merge kept, 5:47 to 5:56 AM CT; the state is the original's through layer 16, as it must be).
  At layers 23 to 32 the wrong answers moved by +0.65 to +1.03 of the same-line difference at the median where they read the identical
  line (+0.96 to +1.03 at layers 28 to 32) and by +0.58 to +0.75 in other wording; the correct answers by +0.31 to +0.56 on the
  identical line (4 answers) and by -0.05 to -0.00 in other wording (9 answers). A reads +0.71 to +0.97 and +0.44 to +0.87 there for the
  wrong group, `a2-path` +0.75 to +1.19 and +0.53 to +0.82. From layer 17's attention to layer 20's attention B moves the wrong answers
  and the identical-line correct ones AGAINST that difference (-0.24 to -0.98), as A and `a2-path` do at layers 14 to 19, and turns
  along it at layer 20's feed-forward. The whole movement, squared, over the squared difference: 34 for the wrong group and 3.0 for
  the correct one at layer 17's attention, 7.9 and 0.28 at layer 23, 4.0 to 4.2 and 0.10 to 0.12 at layers 28 to 32.

**Text by weights: what an own-trace L of an edited model is made of** (`text_and_weights.py`, `text-and-weights/`: `reading.json` holds the
reasoning and the prediction written before the run, `read.sh` the script as it ran, `text-and-weights.txt/json` the tables; card 1, 9:10
to 10:05 AM CT). From a start to an edited model two things change at once: the weights, and the text the model writes with the
registration that lays it on the round's axis. A replay holds a text and changes the weights alone, so four readings of the frozen L make
a table per edit: (a) the start on its own answers, (b) the edited weights on the start's answers, (c) the START's weights on the edited
model's own answers, (d) the edited model on its own. (b) and (c) are conditional replays; (c) is the cell no reading had.

| edit (start) | a | b | c | d | rise d − a | of it the text, at the start's weights (c − a) | the weights, on the edited model's own text (d − c) |
|---|---|---|---|---|---|---|---|
| B: one matrix at layer 17 (the original) | 0.24664 | 0.22418 | 0.42554 | 0.42049 | +0.17385 | +0.17890, 1.03 of it | −0.00505 |
| A: one matrix at layer 9 (the state after nineteen matrices) | 0.21662 | 0.20664 | 5.21514 | 5.19145 | +4.97483 | +4.99852, 1.00 of it | −0.02368 |
| `a2-path`: nineteen matrices (the original) | 0.24664 | 0.26109 | 2.62943 | 2.70408 | +2.45744 | +2.38279, 0.97 of it | +0.07464 |

- *The rise of the own-trace L is the wording, three edits of three.* The unedited start, fed the edited model's words, reads 0.97 to
  1.03 of the whole rise. What the edited weights add on their own text is at most 0.03 of it in size, and it lowers L for B and A. So the three
  own-trace numbers that rose while answers were repaired (0.42, 5.19, 2.70) say how far the new wording lies from the recordings' on this
  axis, at ANY weights; they do not say the edit moved the computation off the correct pattern. The prediction of `reading.json` held for B
  and A in sign and size; for `a2-path` the weights' part on its own text is small but positive (+0.075 on 2.70), so its sign did not.
- *The correct group's "damage" in B is text.* B's correct group reads 0.14436 on its own answers against the original's 0.00253; the
  original's weights on those same answers read 0.14402. On the original's text B moves that group from 0.00253 to 0.00294.
- *Where the weights act on the text B writes.* From layer 17's attention on, wrong group: 0.12738 at the original's weights, 0.12199 at
  B's; layers 17 to 23, 0.0288 to 0.0249; places 142 to 171, 0.0132 to 0.0105; places 172 to 201, 0.0094 to 0.0077. The correct group:
  0.08089 to 0.08123.
- *At the capability's places the text carries the path.* Wrong group, places 172 to 201, all merges: 0.0348 (original on its own text),
  0.0294 (B's weights on that text), 0.0177 (the ORIGINAL's weights on B's text), 0.0160 (B on its own). Of the fall from 0.0348 to 0.0160,
  0.0171 is there at the original's weights once the words are B's: 91%. Which of those words B's weights had to write is the question
  of the two readings below.
- *The split is conditional on the text held* (Astra's assessment, `records/retrospectives/2026-11-12-next-loop-astra.md`, `3332cd17`).
  B's weights lower L by 0.02246 on the original's text and by 0.00505 on B's own; the remainder that is neither alone is +0.01740. In
  `a2-path`'s wrong group the weights' part is −0.03118 on the original's text and +0.03910 on its own. So no subtraction yields a
  history-free "true repair" number, a held-text improvement cannot stand in for the own-path result, and the table does not say the
  weights matter little: they wrote the changed text, with its repairs and its regressions.
- *What it is not.* (c) says what the start computes on the edited model's words, not that it would write them. The table describes the
  30 fitting answers on card 1; it selects nothing, bounds nothing and regrades no run.

**The decisions an edit changed on its own path** (`decisions_changed.py`, `text-and-weights/decisions/decisions-changed.txt/json`,
`read-decisions.sh`; card 1, 9:22 to 10:13 AM CT). An edited model's greedy answer is a chain of decisions. Fed that text token by token,
the model it started from says at every step what IT would write next. The steps where the two differ are the only steps at which the two
sets of weights part on the path the edited model takes; every other token both would have written. The check: fed its own answers each
edited model chooses every token of them, 30 of 30 answers in all three.

| edit | steps of its own 30 answers | decisions the start would make otherwise | start prefers its own by under 0.5 / 0.5 to 2 / over 2 logits | per answer, wrong / correct group (median) |
|---|---|---|---|---|
| B | 15,709 | 379, 2.4% | 206 / 139 / 34 | 14 / 10 |
| A | 20,683 | 660, 3.2% | 359 / 241 / 60 | 21 / 27 |
| `a2-path` | 17,388 | 373, 2.1% | 272 / 85 / 16 | 11 / 11 |

- *An edit changes two or three decisions in a hundred, and most of them barely.* These are token choices on one path, not a count of
  changed internal computation. Nineteen matrices (`a2-path`) change no more decisions than one matrix does (B). In all three, over half lie under half a logit and the written token is the start's second choice in 79 to 88%.
- *Where they lie in B.* By stretch of places (all, and those the start prefers by over 2 logits): 0 to 141: 30, 5; 142 to 171: 76, 19;
  172 to 201: 7, 0; 202 to 259: 18, 2; 260 to 313: 45, 5; 314 to 399: 94, 2; 400 to 479: 109, 1. The firm ones are at the heading and the
  line before the sign; the many late ones are near-ties. The pairs that recur (the start's token → B's): ' descent' → ' rough' in 14
  answers, ' bottom' → ' the' in 10, 'Kin' → 'Total' in 7, 'State' → 'After' in 7, ' -' → ' +' in 7, 'Kin' → 'Potential' in 5.
- *At the sign step the original, given B's words, mostly writes the plus itself. What it writes NEXT is the ladder's question, and the
  ladder shows this reading of scores overstates the wording.* At the step where B writes the descent's sign, the original, given B's
  words up to there, writes ' +' ITSELF in 6 of the 8 wrong answers whose available energy is right in B's own answers (by +0.91 to
  +6.61; 7 of the 8 are new, `expanded-001-h3` had that field right in the original's own answer; 6 of the 8 are complete), and
  prefers ' -' by 0.15 and 0.01 in the other 2 (`discovery-17-channel-h1`, `expanded-092-h1`). What B changed before it in those 8 is the
  line: ' rough' where the original would write ' descent' (5 answers, the original's preference 2.8 to 7.3 logits: "bottom of the
  **rough section**" for "bottom of the **descent**" in 4, and the heading "After Descent (Top of rough section)" in
  `discovery-10-channel-h1`), a 'Total' heading for 'Kin' (2 answers, 2.3 and 3.4), and in `expanded-087-h3` "Height gained" for "Height
  dropped" with a 'Potential' heading. Example, `discovery-17-channel-h1`: the original writes "Kinetic Energy at bottom of descent: KE_i −
  PE_descent = 128 − 80 = 48"; B writes "Kinetic Energy at the bottom of the rough section: KE_i + Energy gained from descent = 128 + 80 =
  208". Where B wrote ' +' over a FIRM preference for ' -' under the old wording (4 answers, 2.2 to 4.1 logits: `discovery-05-channel-h1`,
  `expanded-020-h1`, `-034-h3`, `-094-h1`), none has the available energy right: two double the kinetic energy, two omit the descent. One
  answer shows the wording is not sufficient: `expanded-046-h3` has the new line, the original writes ' +' there too, and what follows is
  "Energy lost during descent (negligible …)". The ladder below finds the same in three of the six: the original writes the plus and then
  subtracts.
- *The correct group at the sign.* The original writes B's sign in 14 of 15; in `discovery-00-channel-h3` B writes ' -' where the original
  prefers ' +' by 0.74, and the answer stays complete.
- *A and `a2-path`.* Their first changed decision is the opening word in most answers (' compact' → ' calculation' in 26 and 19), after
  which the layout is another and few answers lay a sign at places 165 to 173 at all (A: 7 wrong and 11 correct answers; `a2-path`: 5 and
  4). Where they do, the start writes the same ' +' in 3 of 3 (A, wrong group) and 4 of 5 (`a2-path`, wrong group).
- *What it is not.* After an answer's first changed decision the start would have written another text, so every later step is its choice
  on a history it would not have produced: the question asked is where the weights part on THIS path. That the original writes ' +' after
  B's words is a reading of scores; what the original then goes on to write is the ladder's question, below. Nothing here is a loss, a
  target or a selector.

**The decision ladder: what each changed decision carries** (`decision_ladder.py`, `text-and-weights/ladder/`: `reading.json` is the
reasoning and prediction committed before any rung was made (`9ebe1158`), `result.json` the result against it, `B/ladder.txt/json` the
ladders; card 1, 10:14 to 10:46 AM CT). For every step of B's own answers at which the original would write another token, the original
is fed B's tokens through that step, token by token, and writes the rest by itself. A rung's answer is the one the question gets when
B's weights act through that step and the original's after it; these are all the switch points there are (379 rungs in 30 answers). A
rung feeds B's whole prefix, so it is a CUMULATIVE intervention on the history, never a test of its one token alone. The
method's checks hold exactly: the two own captures part at the first changed decision in 30 of 30 answers, at 115,961 fed steps the
original's first choice is the one its plain pass kept, and the last rung of every answer writes B's own answer token for token.

| the wrong group's 15 answers | available energy right | required energy right | complete |
|---|---|---|---|
| the original's own answers | 1 | 12 | 0 |
| every changed decision before the sign fed, the sign left to the original | 4 | 10 | 2 |
| the sign fed too | 5 | 10 | 3 |
| B's own answers (every changed decision) | 8 | 13 | 6 |

- *The prediction that the wording before the sign is enough did not hold.* It said 6 answers right before the sign and all 8 with it;
  the ladder reads 4 and 5. In `discovery-17-channel-h1`, `expanded-045-h1` and `expanded-068-h1` the original, given B's line and a ' +',
  writes "KE_i + (Energy **lost** during descent …) − PE" and subtracts after all. What settles those three is the word after the sign:
  B's ' gained' where the original prefers ' lost' by 1.04, 0.93 and 1.06 logits. So what the changed-decisions reading saw in scores at the
  sign (the original writes ' +' too) overstated what the original then does; the ladder is the reading that lets it write.
- *Where the available energy last turns right, in the 8 answers that end with it right.* Three counts stay apart: 8 wrong answers have
  the available energy right at B's endpoint; 7 of them are new (`expanded-001-h3` had that field right in the original's own answer,
  and loses it at its first rung); 6 answers become complete. The last rung at which the field turns right and then stays right:
  'State' → 'After' (the original's preference 0.08, place 77) in `discovery-10-channel-h1`; 'Kin' → 'Total' (3.41, place 148) in
  `expanded-001-h3`; ' descent' → ' rough' (3.74, place 156) in `expanded-095-h1`; the sign itself (0.01, place 168) in
  `expanded-092-h1`; ' lost' → ' gained' (place 170) in the three above; ' -' → ' =' (0.18, place 172) in `expanded-087-h3`. Seven of
  the eight lie at places 148 to 172, and in six the original prefers its own token by under 1.1 logits. These are measured locations
  and no more: any ladder that ends right after being wrong has a last transition, the prefix before it holds every earlier change of
  B's, and nothing here shows that one decision alone would repair an answer.
- *The outcome depends on the history, rung by rung.* A field of the judge changes at 3 rungs of a wrong answer at the median and at up
  to 10 (by number of such rungs: 1: 2 answers, 2: 2, 3: 4, 4: 2, 5: 1, 7: 1, 8: 1, 10: 2). Handed B's text midway, the original often
  writes an answer neither model writes alone: in the correct group, whose available energy is right in all 15 of the original's own
  answers and all 15 of B's, it is not right at 16 of 163 rungs (8 answers), and the required energy is not at 32 (7 answers).
- *B's cost to the correct group is not at one place.* The required energy first fails early in two of the three answers B cost
  (`discovery-06-channel-h1` at ' descent' → ' rough', 2.85, place 82; `discovery-11-channel-h3` at place 102), is right again at
  later rungs in all three, and the last rung from which it stays wrong is late in all three: places 425, 422 and 367, the original's
  preference 0.14, 1.27 and 0.09. The same word change, ' descent' → ' rough', is on the path of a wrong-group repair in one history
  and of a correct-group failure in another. Places 314 to 479 hold 203 of B's 379 changed decisions, 3 of them over 2 logits.
- *The seven B did not repair.* No rung has the available energy right in five of them. In `expanded-070-h1` (1 rung of 16) and
  `expanded-094-h1` (4 of 20) a midway answer is right and B's later decisions undo it.
- *What it is not.* A rung is written by two sets of weights one after the other; it is no model's own result and no score of a model.
  A token's identity, the place of a last transition and a small margin do not make a correction mask that transfers, and neither "fix
  the sign" nor "train only where the last transition lay" follows (Astra's assessment, `3332cd17`). Nothing here is a loss, a target
  or a selector; C goes on as specified.

**B on the original's text, by the corrected definition: the baseline C is read against** (`conditional_members.py` at definition
version 2 and `prompt_features.py`; `conditional-members/`: `READING-OF-C.md` is the definition, Astra's `9a436042`, fixed before any
candidate of C existed; `conditional-reading.txt/json`, `b.json`, `b-prompt.json`; the first draft with its output is kept under
`first-draft/`; card 1, 10:46 to 10:59 AM CT, committed at `a4906352` while C's solve was running and its run folder held no model).
The question: on a text held the same (the original control's 30 answers replayed on B), how much of the correction the wrong group
still lacks does B's one matrix make, and what does it do to the correct members and to their questions' positions, which should not
move at all? It is conditional on that text, which B does not write (its own answers part from it in 30 of 30), and is never the own
result.

- *What is counted.* For a wrong cell (a merge and a place), `a0` is the control's wrong-group mean movement from the recordings, as
  the frozen L counts it, `G` the recorded gap and `t` the part of `G − a0` outside the ball: the correction the control still lacks.
  `D` is what B adds on the same text. `W0` is the control's wrong-group L, `W` the same with `a0 + D`, and `P = Σ w⟨D, t⟩ / W0` the
  share of the lacking correction made along it. `C_err` is each correct member's own squared movement from the control in the
  reference's weights (the members, not their mean, and no ball), `Q_err` the same at every position of their questions, and
  `R = sqrt(C_err + Q_err) / P`.
- *The checks hold.* `W0` equals the control's frozen wrong-group L cell by cell (largest difference 2.0e-11) and `W` the replay's
  (1.2e-11). The questions' positions are paired on card 1 from a pass that prefills the question and makes one decision; it reads the
  full captures' prompt rows exactly (largest difference 0.0 for the original and for B), B's first decision is the original's in 30
  of 30, and no position differs before merge 51, the matrix's own (layer 17, attention). Coverage is whole: 105 merges by 480 places
  for the 15 correct members, and 2,579 question positions.

| B on the control's text | whole path | direction | size |
|---|---|---|---|
| `W0`, the control's wrong-group L | 0.24411 | | |
| `W`, B's | 0.22124 | | |
| the loss's fall, `W0 − W` | +0.02287 | +0.01614 | +0.00673 |
| `P`, the share of the lacking correction made | +0.10800 | +0.06805 | +0.03994 |
| `C_err`, the correct members | 0.03732 | 0.02091 | 0.01641 |
| `Q_err`, their questions' positions | 0.00943 | 0.00480 | 0.00463 |
| `R` | 2.0020 | | |

- *B makes about a tenth of what is lacking.* The wrong group's loss falls by 9.4% of `W0` and `P` is 0.108. At the matrix's own merge
  the cells' loss falls from 0.000948 to 0.000105, 89% of what stood there, so the fit is met where it is made; the whole path's small
  share is what the layers after it pass on. Of `P`, that merge holds 0.00405, layers 17 to 23 0.06995 and layers 24 to 34 0.03804.
- *Nearly half of `W0` stands before the matrix.* Layers 0 to 16 hold 0.11273 of the 0.24411 (0 to 8: 0.05263; 9 to 13: 0.05292; 14
  to 16: 0.00719), and on a held text no matrix at layer 17 moves a cell there: those cells are in `W0` and add nothing to `P`. Of the
  0.13138 that stands from layer 17 on, B removes 17% (layers 17 to 23: 0.04210 to 0.02565, a fall of 39%; layers 24 to 34: 0.08928 to
  0.08286, 7%), and a `D` equal to `t` in every cell from layer 17 on would give a `P` of 0.538, not 1. The same holds for C, whose
  matrix is the same one, so the comparison of the two `R` is not changed by it; what `P` says about either alone is.
- *Most of what the correct members pay is not in their mean.* The group's mean movement carries 0.01325 of `C_err`'s 0.03732; the other
  0.02407 (64%) is members moving differently from one another, which a reading of the group's mean cannot see. Every member pays
  (0.00133 to 0.00456), and each question's positions pay nearly the same (0.00062 to 0.00064 a member), so `Q_err` is the edit's
  uniform touch on the question and `C_err` the part that depends on the answer. `sqrt(C_err) / P` is 1.7887 and `sqrt(Q_err) / P`
  0.8993.
- *Where, along the answer* (the loss's fall, the part of `P`, `C_err`): places 0 to 141: +0.00036, +0.00147, 0.00307; 142 to 171:
  +0.00889, +0.04140, 0.00869; 172 to 201: +0.00545, +0.02789, 0.00208; 202 to 259: +0.00243, +0.01023, 0.00216; 260 to 313: +0.00249,
  +0.01025, 0.00640; 314 to 399: +0.00065, +0.00457, 0.00963; 400 to 479: +0.00261, +0.01218, 0.00529. Places 142 to 201 hold 63% of the
  fall and 64% of `P` for 29% of `C_err`; places 314 to 479 hold 14% of the fall and 16% of `P` for 40% of `C_err`. By band of layers:
  17 to 23: +0.01645, +0.06995, 0.01942, and 0.00566 of `Q_err`; 24 to 34: +0.00642, +0.03804, 0.01789, and 0.00378.
- *What it is and is not.* It is the number C is read against: Astra's prediction, stated before C, is that C's `R` is below 2.0020 when
  both are defined. It is conditional on a text neither B nor C writes, it is no loss, target or selector, and nothing is fitted to it.
  The stretches are where B's progress and cost lie on this text; no crop of places follows from them (Astra's assessment, `3332cd17`).
  The first draft's statistic, (C + Q) / (W0 − W) = 1.7855 from a zero origin, measured another thing and is history only.

**Experiment C's one candidate, read** (`runs/correction-members-at-17-original`: B's matrix, rows, folds and wrong-group cells, with the
correct group's mean-and-radius cells replaced by each correct member's own error at every place and question position. Fable's solve ran
10:07:07 to 11:30:29 AM CT and selected ridge 2^-5, round 40; the change is 0.2007 of the matrix against B's 0.0953. Model sha256
`2f0eb953…`, checked against the one Fable sent and only read. Read on card 1 from 11:31:57 AM CT by the run's `own/logs/read-own.sh`, as
`conditional-members/READING-OF-C.md` fixed it before the candidate existed; records in the run's `own/` and in `conditional-members/`.)

| the 30 fitting answers, each model answering by itself | original | B | C |
|---|---|---|---|
| wrong group (15): complete | 0 | 6 | 6 |
| wrong group: available energy right; subtracted; omitted; other | 1; 14; 0; 0 | 8; 3; 2; 2 | 6; 6; 1; 2 |
| wrong group: required energy right; the right verdict | 12; 1 | 13; 11 | 14; 9 |
| correct group (15): complete | 15 | 12 | 10 |
| correct group: available energy right | 15 | 15 | 10 (4 descent omitted, 1 missing) |
| correct group: required energy right; the right verdict | 15; 15 | 12; 15 | 14; 11 |
| correct group: written with the recording's own tokens | 13 | 0 | 6 |
| cut at the token limit | 1 (wrong) | 0 | 1 (correct) |
| whole-model L on its own answers; wrong group; correct group | 0.24664; 0.24411; 0.00253 | 0.42049; 0.27613; 0.14436 | 0.27380; 0.17821; 0.09559 |

- *Astra's two predicted counts are met, and the count hides a swap* (`own/diagnostics/answers/answers.txt`, by `answers.py`). C has six
  complete repairs, as B, and one originally correct answer with a required-energy error against B's three. Of B's six repairs C retains
  two (`expanded-001-h3`, `expanded-095-h1`) and loses four: `discovery-17-channel-h1`, `expanded-045-h1` and `expanded-068-h1` subtract
  the descent again, the original failure, and `expanded-092-h1` is "other". C has four new ones: `discovery-05-channel-h1`,
  `discovery-10-channel-h1`, `expanded-087-h3`, `expanded-094-h1`. B and C together repair 10 of the 15; `expanded-020-h1`, `-034-h3`,
  `-042-h3`, `-046-h3` and `-070-h1` are repaired by neither.
- *C undoes B's damage to the correct group and does another.* B's three answers with a required-energy error
  (`discovery-06-channel-h1`, `discovery-08-channel-h3`, `discovery-11-channel-h3`) are complete again. Five others are not complete, none
  of them B's: `discovery-15-channel-h1`, `expanded-029-h1`, `expanded-076-h1` and `expanded-078-h1` leave the descent out of the
  available energy, and `discovery-00-channel-h3` runs to the token limit (available energy missing, required energy climb plus descent:
  C's one required-energy error). B never broke the available energy of a correct answer. So the correct group has 10 complete against
  B's 12.
- *On the original's text the conditional statistic is a third of B's, as predicted* (`conditional-members/conditional-reading.txt`;
  definition version 2; both cell-by-cell checks hold):

  | on the original's card-1 text | B | C |
  |---|---|---|
  | `W`, from `W0` 0.24411; the loss's fall | 0.22124; 0.02287 | 0.22249; 0.02162 |
  | `P`, the share of the lacking correction made | 0.10800 | 0.09264 |
  | `C_err`, the correct members; of it the group's mean | 0.03732; 0.01325 | 0.00372; 0.00100 |
  | `Q_err`, their questions' positions | 0.00943 | 0.00018 |
  | `R` | 2.0020 | 0.6735 |
  | at the matrix's own merge: the loss's fall; `C_err` | 0.000843; 0.001096 | 0.000817; 0.000074 |

  C makes 86% of B's progress with a tenth of the correct members' error and a fiftieth of their questions' positions'. By stretch its
  `P` is 0.03354 at places 142 to 171 and 0.02856 at 172 to 201 (B 0.04140 and 0.02789), and its `C_err` is largest at places 142 to 171
  (0.00140).
- *The held-text statistic did not predict the own answers; its ranking inside C did.* The correct members' held-text error fell
  tenfold from B to C while the correct answers not complete rose from three to five. The five C breaks are five of the six correct
  members with the largest `C_err` under C (`expanded-029-h1` 0.00040, `expanded-078-h1` 0.00035, `discovery-00-channel-h3` 0.00034,
  `expanded-076-h1` 0.00030, `discovery-15-channel-h1` 0.00029; the sixth and largest, `discovery-01-channel-h1` at 0.00110, is one of
  the two whose control text is not the recording's, and stays complete; the other nine lie at 0.00006 to 0.00015). The same five are
  among the six whose error under B sat mostly at places 142 to 171, at ten times the level, and B broke none of them.
- *On decisions the member term did what it was built for, and one line undid five answers* (`conditional-members/b-replay-own.json`,
  `c-replay-own.json`: each model's own first choice at every step of the original's card-1 text). The correct group: under B all 15
  answers leave that text (first at step 97 at the median, 11 to 264) and B's choice differs at 125 of its steps; under C 10 of 15
  leave it (step 145.5 at the median, 72 to 337) and C's choice differs at 31 steps; five correct answers are written exactly as the
  original writes them. The wrong group: 248 steps under B, 199 under C, all 15 leaving under both. But the five correct answers C
  breaks first leave the original's text at steps 145, 146, 146, 150 and 166, just before the sign, with one or two differing steps
  each, and four of them write the same new line there. `expanded-076-h1`, the original: "$KE_i + \text{Work done by gravity} = 40 J +
  100 J = 140 J$"; C: "$KE_i + \text{Energy lost during descent (0, smooth track)} = 40 J$ (Since the track is smooth, $KE$ is conserved
  during descent ... but the problem implies the initial speed is used for the subsequent motion)". `expanded-029-h1` and
  `expanded-078-h1` the same; `discovery-15-channel-h1` goes from "(negligible as track is smooth)", after which the original adds the
  descent in its next line, to the same parenthesis without it. The three repairs of B that C loses write the original's subtraction
  again where B wrote "$KE_i + \text{Energy gained from descent}$"; these are the three the ladder on B found settled by ' gained'
  against ' lost'.
- *At the sign, on the recordings' text, C moves the score a third as far as B* (`own/diagnostics/sign/sign-scores.txt`). The score of
  ' +' less ' -' at the step that chooses, against the original: the wrong group +2.27 at the median (+1.11 to +12.39), ' +' ahead in 8
  of 15; B +7.02 (+2.00 to +12.21) and 12 of 15. The correct group +0.13 at the median (-0.15 to +1.97); B -0.34 (-2.67 to +5.66). Over
  every step of that text C's first choice is another token at 230 steps (B 365), and the size of a margin's change at the median is
  0.068 (B 0.174) (`own/diagnostics/sign/score-movement.txt`). On the trace the two edits make nearly the same progress (`P` 0.093
  against 0.108; Fable's rows read +0.72 of the gap at the sign for both), so the trace measure and the decision's score do not move
  together here.
- *The fall of C's own wrong-group L is in its wording* (`text-and-weights/logs/text-and-weights-with-c.out`; the four cells, C read
  12:24 to 12:33 PM CT). The original's weights on C's own answers read L 0.27928 (wrong group 0.18210, correct group 0.09719); C on
  them 0.27380 (0.17821, 0.09559). So of the wrong group's fall from 0.24411 to 0.17821, the wording at the original's weights gives
  -0.06201 and the weights on that wording -0.00388; the correct group's rise of 0.093 is the wording too (+0.09466). As for B, A and
  a2-path, what an own-trace L of this edited model reads is the text it writes.
- *C changes half as many decisions as B* (`text-and-weights/decisions/decisions-changed.txt`). The original would write another token
  at 185 of C's 14,917 own steps (1.2%; B 379 of 15,709, 2.4%): 141 by under half a logit, 35 by half a logit to 2, 9 by over 2. Per
  answer the median is 7 in the wrong group and 3 in the correct one (B 14 and 10), and five correct answers have none. By stretch:
  0 to 141: 16; 142 to 171: 29; 172 to 201: 9; 202 to 259: 13; 260 to 313: 41; 314 to 399: 36; 400 to 479: 41. C writes ' +' where
  the original prefers ' -' in 4 answers, all wrong-group, by 2.62 to 4.32 logits, and the available energy is right in one of them.
- *By depth at the sign step, on the recordings' text* (`own/logs/depth.out`; the movement along the same-line difference, median over
  the answers). The six wrong answers on the identical line: +0.34 at layer 25, +0.61 at layer 30, +0.74 at layer 34 (B +0.65, +0.96,
  +0.56). The four correct answers on that line: +0.25, +0.40, +0.25 (B +0.31, +0.32, -0.57). The nine correct answers of other
  wording stay within 0.03 of zero at those layers under both. So on the one line both groups read, at layers 25 and 30 C moves the
  wrong answers 0.52 and 0.64 as far as B and the correct answers 0.81 and 1.25 as far: the wrong less the correct is +0.09 and
  +0.21, where B's was +0.34 and +0.64. The member term held the correct answers of other wording; where the two groups read the
  same text it parted them less than B did.
- *The ladder on C: one token, and the unchanged original writes the other outcome* (`text-and-weights/ladder/C/ladder.txt`, 185 rungs,
  1,070 s, 12:38 to 12:56 PM CT; the checks hold: the two own captures part at the first changed decision in 30 of 30, 58,639 fed
  steps with none differing from the plain pass, the last rung is C's own answer in 25 of 25 answers with a rung. Read at an answer's
  FIRST rung by `ladder_first_rung.py`, `text-and-weights/ladder/first-rung.txt`). A first rung is the one rung that is not a
  cumulative intervention: before the first changed decision the two models write the same text, so the original is fed its own
  answer and one token it did not choose, and writes the rest.

  | after its own text and one token of the edited model, the original writes | B's token | C's token |
  |---|---|---|
  | wrong group: complete (its own answers: 0 of 15) | 3 of 15 | 5 of 15 |
  | of them, the edited model's own answer is complete too | 1 | 4 |
  | correct group: not complete (its own answers: 15 of 15 complete) | 6 of 15 | 5 of the 10 that have a rung |
  | of them, the edited model's own answer is not complete either | 2 | 5 |

  - Four of C's six complete repairs are this: `discovery-05-channel-h1` ('2' → 'before', the original preferring its own by 0.25,
    step 72), `discovery-10-channel-h1` ('gain' → 'des', 0.02, step 92), `expanded-087-h3` ('_{' → '_', 0.02, step 82) and
    `expanded-095-h1` ('State' → 'After', 0.01, step 69): a token of the lines before the sign's (58 to 69 steps before it in the
    first three), after which the original adds the descent by itself, and the available energy is right at every later rung. The other two (`expanded-001-h3`,
    `expanded-094-h1`) are right at 5 of 8 and 7 of 10 rungs and stay right from the fifth. `expanded-034-h3` is right after one token
    (' ($' → ':', 0.00, step 26) and at the next rung, and C's later decisions leave the descent out.
  - All five correct answers C breaks are this too. `expanded-029-h1`, `expanded-076-h1`, `expanded-078-h1`: 'Work' → 'Energy' (0.28,
    0.18, 0.05; steps 145, 146, 146); `discovery-15-channel-h1`: 'negl' → '0' (0.08, step 150); after it the original writes the
    answer with the descent left out, and so does every later rung (0 of 6, 9, 7 and 3 right). `discovery-00-channel-h3`: ' (' → ':'
    (0.68, step 166), right at 1 of 38 rungs. The steps are those Fable's kept rows name (150, 145, 146, 146; commit `99edd068`); the
    ladder's places for them (172, 169, 152, 152) are those of C's own registration, not the original's 169.
  - The same holds on B's ladder, read again this way: three wrong answers turn complete after one token, 'State' → 'After' in all
    three (0.08, 0.04, 0.01), and six correct answers turn not complete after one token (0.06 to 2.85), of which B's later decisions
    bring four back. 'Work' → 'Energy' at step 145 of `expanded-029-h1` is the same rung in both ladders; under B the answer ends
    complete, under C it does not.
  - The repairs that are NOT this are B's three on the identical line (`discovery-17-channel-h1`, `expanded-045-h1`,
    `expanded-068-h1`): no first or early rung rights them, B's ladder settles them at the word after the sign, and their sign score
    on the recordings' text moves from -3.57, -2.05 and -3.46 to +3.79, +5.41 and +3.94 under B and to -2.01, +0.56 and -1.66 under C.
    C loses all three. No rung of C's ladder has the available energy right in them or in `expanded-020-h1`, `-046-h3`, `-070-h1`.
    On B's OWN path those three begin on the heading, not at the sign (`text-and-weights/ladder/B/ladder.txt`, lines 51 to 57, 133
    to 138 and 164 to 170; `text-and-weights/decisions/decisions-changed.txt`, lines 21 to 23, 38 to 40 and 44 to 46): ' bottom' →
    ' the' (by 1.52, 1.32, 1.50) and ' descent' → ' rough' at place 156 (by 7.16, 5.93, 7.33) turn "Kinetic Energy at bottom of
    descent" into "Kinetic Energy at the bottom of the rough section". After that heading the original writes ' +' itself in two of
    the three (by 2.72 and 0.91) and prefers ' -' by 0.15 in the third, and at every rung through the sign it still writes ' lost'
    and subtracts the descent. The rung that turns each is ' lost' → ' gained' at place 170, by 1.04, 0.93 and 1.06.
  - So the answers show an edit doing more than one thing here, and a count of complete answers adds them. One: it tips a near-tie
    of wording early in the answer, and the ORIGINAL's weights then write the calculation right, or wrong. That is four of C's six
    repairs, all of C's breaks and part of B's, and it says the original already writes both outcomes on these questions, a token
    apart. Two: on the line both groups write, B rewrites the heading (by 6 to 7 logits) and then prefers ' gained' to ' lost' after
    the sign (by about one). That is B's three. C makes neither change in them: on its own path it changes 0 and 1 decisions
    before the sign in `discovery-17-channel-h1` and `expanded-068-h1` and 5 in all in `expanded-045-h1`, none by over 2 logits, and
    their sign score on the recordings' text stays below zero in two. Three: it writes ' +' where the original prefers ' -' by more
    than 2 logits, and that alone seldom completes an answer: of B's four such flips (2.18 to 4.05) none does, of C's four (2.62 to
    4.32) one does, `expanded-094-h1`, right from its fifth rung (`decisions-changed.txt`, lines 10 and 309).
  - **Correction, November 12, 2:29 PM CT.** As first written (commit `db69f5ab`) the second kind read "it changes what is written
    at the sign on the line where the two groups read the same text. That is B's three, and C's member term, holding the four
    correct answers on that line, gave it up." Both halves said more than was read. On B's own path its work in those three is on
    the heading and at the word after the sign, as above; the rise of their sign score by about 7 logits is a reading on the
    recordings' text, where the heading is the original's. And nothing here says which of C's two differences from B (the member
    term, a change of the matrix 2.1 times as large) is why C makes neither change. Stage 3 of the analysis in `deep-think-analysis/`
    found it; the lines above are checked against the two files.
  - *Every answer B and C turn, by the decision that turns it* (`turned_by.py`; `text-and-weights/ladder/turned-by.txt` and `.json`;
    files only, 2:38 PM CT). For each answer whose completeness an edit turns: its first rung, the TURNING rung (the first from
    which the original writes the edited model's outcome at that rung and at every later one), the largest changed decision up to
    it, and the sign on the edited model's own text, each with the text read before it. Nothing is classed; the lines are the
    answers' own.
    - B's six repairs. Three turn at ' lost' → ' gained' (1.04, 0.93, 1.06), read after "$KE_i + \text{Energy", and the largest
      decision before it is ' descent' → ' rough' on the heading (7.16, 5.93, 7.33): the three of the correction. `expanded-095-h1`
      turns at ' descent' → ' rough' itself (3.74), and was complete already after its first token. `expanded-092-h1` turns at the
      sign, ' -' → ' +' by 0.01, after 'Kin' → 'Total' on its heading by 2.26. `expanded-001-h3` turns at a token of 0.01 at step
      310, after 'Kin' → 'Total' by 3.41.
    - C's six repairs. Four turn at their first rung (0.25, 0.02, 0.02, 0.01): the four above. `expanded-094-h1` turns at a token
      of 0.16 at step 195, and the largest decision before it is the sign itself, ' -' → ' +' by 2.65. `expanded-001-h3` turns at a
      token of 0.06 at step 379, after ' height' → ' final' by 3.34.
    - `expanded-001-h3`, a repair under both, is the answer the original writes with both energies right on this card and does
      not end within the token limit: what each edit turns in it is the answer ending.
    - C's five breaks. Four turn at their first rung, on the token after "$KE_i + \text{" ('Work' → 'Energy', 0.28, 0.18, 0.05) or
      inside "Energy lost during descent (" ('negl' → '0', 0.08); the original has written ' +' there itself.
      `discovery-00-channel-h3` turns at its 13th rung of 38 (step 303, 0.25), after 'B' → 'A' by 4.33.
    - B's three breaks, all of the required energy, turn late in the answer (steps 418, 461, 512) at tokens of 1.27, 0.09 and 0.14;
      the largest decision before it is ' descent' → ' rough' in an earlier heading (2.28, 2.85) in two and ' height' →
      ' destination' (1.50) in the third.
    - Counted over the twelve repairs: four are one early token (C); three are the heading rewritten and then the word after the
      sign (B); one is the heading rewritten after an early token (B); one turns at a sign the original nearly ties (0.01) after
      its heading went from 'Kin' to 'Total' (B); one has a sign written against the original by 2.65 as its largest decision and
      turns five rungs in (C); two are `expanded-001-h3` ending. In none of the twelve is the turning decision a sign the edited
      model writes against a preference of the original of more than 0.01 logit on that text. This is the edited models' own text,
      where earlier decisions have changed what is read; it is not the reading on the recordings' text, where B's and C's sign
      scores are those of the table above.
    - What these readings cost for one model on card 1, from C's own logs (`runs/correction-members-at-17-original/own/logs/chain.log`
      and `text-and-weights/ladder/logs/chain.log`): the model's own answers with every merge kept, which L needs, 17 min 41 s; the
      two plain passes of the changed decisions 5 min 12 s; the ladder 5.8 s a rung (C's 185 rungs in 1,070 s; B's 379 in 32 min).
      The first rungs alone are at most one a fitting answer, about 3 minutes at that rate; `decision_ladder.py` makes every rung
      and has no such option yet. As the tools stand, the first rungs and the changed decisions of one stage end of a solve would
      be about 26 minutes of card 1, 18 of them the capture. No solve keeps its stage ends yet (the solver is Fable's, the
      declaration Astra's), so none has been read this way.
  - What it is not: a token after which the original writes another outcome is a measured sensitivity of the original on that text.
    It is the token the edit chose, not any near-tie (the two cards' own differences reword 9 of 30 of the original's answers and
    change no group's count of complete answers); it is no mask, no target and no selector.
- *What this is and is not.* One candidate is one draw. C differs from B in two things at once: the objective, and a change of the
  matrix 2.1 times as large, selected among ridges the solver's own spread does not tell apart (Fable's record). The conditional
  reading is on the original's card-1 text, which is the fitting text itself in 21 of the 30 answers. The answer counts are evidence
  beside the trace readings, never a reward or a selector. No second candidate, dose or opening control is read, the answers no fit
  uses were not generated and reserved-40b was not opened.

**The label of the line: where the available energy is decided, and what one swapped label does** (`deep-think-analysis-2/frame_margin.py`
and `frame-margin.json`, files only; `label_by_model.py`, `label-swap/label-by-model.txt`, files only; `label_swap.py`, `label-swap/`,
card 1 under its lock, 4:50 to 4:53 PM CT, 141 s of the recorder, the original model, text only, the verifier first). It came out of the
second deep-think-analysis (`deep-think-analysis-2/`), whose stage 2 asked where the answer commits before the sign.

- *What the recorded answers show (files only).* The line that first writes the initial kinetic energy with a sign begins `Kin`
  ("Kinetic Energy at …") or `Total` ("Total Energy …") in 201 of the original's recorded channel answers (the no-fit thirty are not
  read, the reserved set is not opened). Of the 103 that begin `Total` the available energy is right in 103, and 80 are complete; the
  required energy is wrong in 22, in 20 of them by adding the descent to the climb, 17 of those where the kinetic energy is larger
  than the descent's gain. Of the 98 that begin `Kin` the available energy is right in 43, the descent is subtracted in 15, left out
  in 15, and 25 are missing or otherwise wrong; 39 are complete. Every recorded answer that subtracts the descent began the line
  `Kin`. Where the kept eight scores hold `Total`, the answers that subtract prefer `Kin` to it by a median of 3.38 logits (quartiles
  2.17 and 3.49, least 0.01, in 13 of the 15).
- *The same in every model's own answers on file (files only).* Over the own fitting answers of the original on card 1, the state
  after nineteen matrices, `a2-path`, `d2-path`, A, B and C, 59 lines begin `Total` and the available energy is right in 58 (the one
  that is not, `d2-path`'s `expanded-095-h1`, writes the plus under `Total` and states the subtracted number in its last line). B's
  answers hold 10 such lines and C's 11, against the original's 7. C's five breaks are all among the correct group's `Kin` answers,
  where the available energy is right in 1 of 6. A writes neither label in 26 of its 30 answers.
- *`d2-path` is the run that moved the label, and it broke the other wording with it.* Its own fitting answers begin the line `Total`
  in 21 of 30 (11 of the 15 wrong answers), and the available energy is right in 20 of those 21. But 13 of the 21 then speak of a
  total height reached or of the starting point, against 23 of 103 in the original's recorded `Total` answers, and the descent is
  added to the climb in 14 of the 21 (11 of the 13, and 3 of the 6 that give the destination's height only, which the original's
  `Total` answers never do in 48). Under a weights edit the two wordings moved together (stage 3 of the second analysis found it;
  the counts above are `label_by_model.py`'s).
- *What the word does (card 1).* The original was fed each fitting answer of its own up to that label and the OTHER label in its
  place, one token, and wrote the rest by itself; 29 passes, one answer has no such line. At all 29 steps its first choice was its
  own label.
  - The fifteen wrong answers, all `Kin`, fed `Total` (the original prefers `Kin` by 0.02 to 4.06, median 3.39, in the 13 where the
    pass keeps both scores): the available energy is right in 10 of 15 against 1 in its own answers, and 8 are complete against 0.
    One of the fifteen, `expanded-001-h3`, was already right in both energies on this card, so the word turned 10 of the 14 that
    subtract. By the second heading above the label: where it reads "(2 m)", "(Top of descent)" or "(Height h₁ = 2 m)", 9 of 9
    turn; where it reads "(Point A)" or "(Point B)", 1 of 6. `expanded-092-h1`, `-094-h1` and `-095-h1` have the same mass, speed,
    drop and climb; 092 wrote "(2 m)" and turned, the other two wrote "(Point A)" and did not.
    Of the ten, two add the descent to the climb. The five that do not come right: `expanded-094-h1` and `-095-h1` write "Total
    Energy at Point A ($E_A$): $KE_i - PE_{gain}$" and still subtract; `expanded-034-h3` and `-042-h3` are laid out otherwise (the
    label found is a heading two lines above the line, which stays `$$E_B = KE_A - PE_B`); `expanded-001-h3`, right in both energies
    in its own card-1 answer, comes out with another available energy.
  - The seven correct answers that wrote `Kin`, fed `Total`: six stay complete; `discovery-00-channel-h3` comes out with another
    available energy.
  - The seven correct answers that wrote `Total`, fed `Kin`: two stay complete. Two subtract the descent ("Kinetic Energy at top of
    descent … $KE_i - PE_{descent}$"), two end with no available energy (one writes "24 J - 30 J = -6 J"), one adds the descent to
    the climb.
- *What it shows.* On these answers the label is not only a sign of a state that was already wrong: with the weights and the prompt
  the original's, one token turns the available energy right in ten wrong answers (fed `Total`) and wrong in four right ones (fed `Kin`). In
  the recorded answers no line labelled `Total` subtracts; under a fed `Total` two answers still do, so the label does not settle
  the sign in every layout, and the wording of the heading before it has a part. What is left under `Total` is the other use of the
  descent, the error `d2-path` shows and the one round one's edit A repaired (' gained' over ' reached', a tie of 0.006,
  `studies/03-round-1-single-pair-edits/A-weight-edit/STUDY.md:169-174`).
- *The heading and the label together, and the climb's wording (`second_feeds.py`, `label-swap/second-feeds/`; card 1 under its lock,
  5:19 to 5:21 PM CT, 19 passes, 93 s, the original model, text only, the verifier first).* Stage 3 of the second analysis named these
  two feeds as the cheapest way to prove the path wrong.
  - *The heading.* The six fitting wrong answers whose second heading reads "(Point A)" or "(Point B)" were fed the digit of their
    question's drop for 'Point' ("State after Descent (2 m)"). The heading alone turns none of the five that subtract: each still
    begins the line `Kin` and subtracts (`expanded-087-h3` now writes "90 J - 100 J = -10 J. Since kinetic energy cannot be
    negative …"). With `Total` then fed for `Kin` on that continuation, `expanded-087-h3`, `-094-h1` and `-095-h1` come out complete
    ("Total Energy before rough section ($E_1$): $KE_i + PE_{gain}$"); the two that had kept the minus under "Total Energy at
    Point A" are among them. `expanded-034-h3` and `-042-h3`, which write the line as a display equation
    ("$$E_B = KE_A - PE_B$$") under their heading, still subtract. `expanded-001-h3`, right already, stays right.
  - *The climb.* Seven continuations of the label swap write "height reached" or "total vertical displacement" in their final-state
    paragraph and were fed "gained" or "height gained" there. All four that had added the descent to the climb come out with the
    required energy right (`expanded-020-h1` and `-068-h1` complete; `-094-h1` and `-095-h1` right in the required energy, their
    available energy still subtracted on that text). The two that were complete stay complete. One whose required energy was
    otherwise wrong (`expanded-064-h1`, a correct answer fed `Kin`) then adds the descent to the climb.
  - *Counted over the fourteen fitting answers that subtract:* with one fed word the unedited model writes the available energy
    right in 10; with the heading and the label, in 12; with the climb's word as well, 12 come out complete. The two that do not
    share one layout.
- *The opening and the sign: which text picks the label, and what the sign's own row carries (`prefix_transplant.py`,
  `label-swap/transplant/`; card 1 under its lock, 7:59 to 8:03 PM CT, 41 passes, 225 s, the original model, text only, the verifier
  first).* Fable's `same-prefix-135-original` holds the answer's text fixed and lets the question's number differ. This is the other
  half: the question is held fixed and the text before the label differs.
  - *A right answer's opening under a wrong question.* Thirteen of the fourteen subtracting fitting answers have a recorded answer
    to another question of the same mass, speed and drop that begins the line `Total` and is complete (`expanded-087-h3` has none).
    Fed that answer's text up to the label, the label not included, the original writes `Total` by itself in 9 of the 13, with the
    available energy right in all 9 and the whole answer right in 8. `expanded-034-h3` and `-042-h3`, which no fed word had turned,
    are among them. The four that still write `Kin` do so by 0.02 to 0.20 logits: `discovery-10-channel-h1`, whose donors' text is
    its own text token for token (the five answers of the same-prefix run), and `expanded-092-h1`, `-094-h1`, `-095-h1`, whose
    donors climb 3 m where they climb 1 m.
  - *How far the opening moves the label.* On its own opening a wrong answer prefers `Kin` by a median 3.25 logits (0.02 to 3.78).
    Under the right answer's opening the same question prefers `Total` by 0.61 to 1.27 in most, 2.98 and 3.46 under the heading
    "(at the start of the rough section)", and `Kin` by 0.02 to 0.20 in the four. Answer by answer the opening moves the
    preference by a median 4.1 logits toward `Total`; what is left to the question's numbers is under about one logit.
  - *The other way.* Four of the seven fitting correct answers that begin the line `Total` have a recorded subtracting answer of
    the same mass, speed and drop. Fed its opening, they write `Kin` by themselves in 5 of 6 passes (by 2.93 to 3.29 where the
    scores are kept) and subtract the descent in all 5; none comes out complete.
  - *Which openings.* The right answers' second headings read "(Top of descent)", "(before rough section)" or "(at the start of
    the rough section)" and their line then reads "Total Energy before rough section". The subtracting answers' read "(2 m)",
    "(Height $h_1 = 2$ m)" or "(Point A)" and their line reads "Kinetic Energy at bottom of descent" or "at Point A". So the
    heading feed above gave the answers "(2 m)", a heading the subtracting answers themselves write; that it turned none by itself
    says nothing against the heading. Both kinds write "Potential Energy gained" for the descent in the line before the label.
  - *The sign's row.* Each of the fourteen subtracting answers was fed its own text up to the sign of the line and ' +' for ' -'.
    The available energy comes right in 6 and the whole answer in 6. In five the original then writes "$KE_i$ + Energy lost during
    descent (0, smooth track)" and leaves the descent out; in three it writes the plus and then "- $PE_{gain}$". A perfect change
    of the sign's own token carries less of the outcome (6 of 14) than the label's (10 of 14) or the opening (9 of 13 with the
    label left to the model).
  - *Counted over the fourteen:* under some fed text the unedited model now writes every one of them whole.
  - *What it does not show.* Donors, tokens and the passes kept are chosen by this reading; 21 passes share 10 donors; three passes
    are marked because the donor's climb equals a shared quantity; nothing here is a rate, a target, a mask or a selector, and no
    weights were changed.
- *The second wording, in the recorded answers (`height_wording.py`, `label-swap/height-wording.txt`; files only, patterns written
  after reading eight paragraphs).* Under `Total`, the final-state paragraph speaks of "the total vertical height reached" or of a
  displacement "from the starting point" in 23 answers, and the required energy is right in 2 of them and has the descent added to
  the climb in 20: all twenty such errors under `Total`. Where it speaks of the height "gained during the climb" (32) or gives the
  destination's height only (48), the required energy is right in 79 of 80. Those are round one's two words: its repair went from
  "The total height reached" to "The total height gained during the climb".
- *What it does not show.* That an edit which moves the label repairs the model: this is a fed token, not a weight. Why the label is
  what it is: the wording of the second heading goes with it (the two that still subtract under `Total` both read "Point A"), and
  the question's numbers lean it. That it holds outside these thirty answers and the recorded pool: no answer of the no-fit thirty
  or the reserved set is read. It is a measured sensitivity of the original and no score, target, mask, threshold or selector.
- *A correction to the line above on the two cards.* "It is the token the edit chose, not any near-tie (the two cards' own
  differences reword 9 of 30 … and change no group's count)" said more than the nine show: six of them start after the available
  energy is written (steps 289 to 689, `token-histories.txt:14-22`), and of the three before it one (`expanded-001-h3`, step 18)
  has both energies right on card 1. Nine rewordings are too few to say a near-tie does nothing.

**The first words, the sign's row and the question's numbers: three readings of the evening of November 12** (files only unless said;
the recorded answers with a label line, the answers no fit uses left out, the reserved set never opened).

- *The route is taken in the first words (`opening_forks.py`, `opening-forks/`).* Of the 201 recorded answers, the 35 that open
  "Here is the calculation" all write the `Kin` line (0 `Total`; 9 complete), and of the 166 that open "Here is the compact" the
  103 that go on "**Initial State (" write `Total` in 99 (available energy right in 99, complete 77) while the 63 that go on
  "**Initial State:**" write `Kin` in 59 (the descent subtracted in 11, complete 33). The written word leads the other by a median
  0.18 logits at the third token (greatest 0.52) and 0.21 at the token after "Initial State" (greatest 1.37); the heading's first
  word leads its second by a median 0.92. The transplant of this afternoon, read again from its passes' own records: of the 124 to
  183 tokens of a supplied opening, the ones that were not the model's own first choice there number 0 to 4 a pass (0 in two, 1 in
  fifteen, 2 in eight, 4 in two), and the model preferred its own by 0.005 to 0.25 logits at 37 of the 39 (0.46 and 0.98 at two
  reverse controls): " compact" for " calculation" at the third token, " (" for ":**" after "Initial State", "before" for "2" or
  "Height" in the heading. So "a right answer's opening" was one to four near-ties, and the rest of the opening is what the model
  writes itself after them. On the `Total` route the available energy is right in 103 of 103 and what goes wrong is the other
  computation: 20 of the 23 answers there that are not complete add the descent to the climb.
- *The sign goes with the question's numbers (`sign_by_regime.py`, `sign-by-regime/`).* On the `Kin` line (98 answers), where the
  initial kinetic energy is above the descent's energy (38) the first sign after KE_i is a minus in 28 (the descent subtracted in
  13, left out in 4) and a plus in 10, all ten with the available energy right; where it is below (60) the sign is a plus in 43
  and a minus in 17 (the descent subtracted in 2). With the kinetic energy above and a rise of 1 m: a minus in 21 of 23; with a
  rise of 3 m: 7 of 15. In the fitting thirty: 13 of the 15 wrong answers have the kinetic energy above the descent's, and 6 of the
  7 correct answers that write the same `Kin` line have it below. So a comparison of the wrong answers with the same-line correct
  ones inside the fitting thirty compares two regimes of the question's numbers before it compares right with wrong. Ten recorded
  answers write the `Kin` line with the kinetic energy above and add the descent; one is a fitting answer (`expanded-071-h1`),
  nine are outside the registered sixty and have tokens on record and no trace.
  The sign also goes with the words of the line before it. After a line that begins "Potential Energy gained", on a `Kin` line
  with the kinetic energy above the descent's: a minus in 15 of 15 (the descent subtracted in 11, left out in 4); below it: a plus
  in 29 of 36. After "Height dropped": a plus in 11 of 11 (3 of them above). After "Kinetic Energy after descent": a minus in 13 of
  13. On a `Total` line after "Potential Energy gained": a plus in 89 of 89 (58 above). On the one pair of lines that six of the
  fitting wrong answers write word for word ("Potential Energy gained ..." then "Kinetic Energy at bottom of descent
  (KE_{bottom}): KE_i", 36 answers): above, a minus in 6 of 6 (those six); below, a plus in 27 of 30. No recorded answer writes a
  plus above on that pair. Three answers outside the registered sixty write a minus below on it (`expanded-060-h3`,
  `expanded-061-h1`, `expanded-062-h1`, each 8 J below; the sign leads by 1.75, 0.19 and 0.27 logits).
- *Which row a change keyed on a matrix's input can address (`row_precision.py`, `row-precision/`; `sign_rows.py`, `sign-rows/`;
  card 1 for the arithmetic, no engine).* At each of the 70 matrices, a
  key was built from the rows of the wrong fitting answers outside a fold (the mean input at the row, whitened by the second
  moment of every other row, float64, no ridge; five folds by case) and read at every row of the fold's answers: 1 is the request
  met, 0 a row left alone. At the label's row the held-out wrong answers answer about 1.0 and the correct answers' label rows 0.69
  to 1.09 (0.82 or more at every feed-forward matrix); at the heading's row correct and wrong answer alike (1.05 to 1.15 at
  feed-forward 14 to 34). Neither row is addressed apart from the correct answers by a matrix's input. At the sign's row the first
  table kept the correct answers as one median (1.02 at layer 0, 0.30 at feed-forward 16 to 19, 0.15 at 34), and I told both seats
  that the sign's row could be addressed without touching the correct answers. **That was wrong, and Astra said why before the
  second reading showed it:** the median was the eight correct answers that write the `Total` line (their median 0.07 to 0.37 from
  feed-forward 14). Kept answer by answer (`sign-rows/sign-rows.json`, by kind in `sign-rows-by-kind.txt`): the four correct
  answers that write the same two lines as six of the wrong ones answer the key at a median 0.91 to 1.24 of the request at every
  feed-forward input from 0 to 34 (0.91 to 1.01 from layer 21; the least single answer 0.84), beside the wrong ones' median 0.96
  to 1.18; asked to be zero in the mean over the same-line correct answers outside the fold, they still answer a median 0.63 to
  0.81 held out from feed-forward 14 (the wrong ones 0.75 to 0.95). Of the three other correct answers on a `Kin` line, the two
  whose line before begins "Height dropped" fall from 0.89 and 0.93 at feed-forward 14 to 0.66 and 0.69 at 21, 0.40 at 25 and 0.15
  at 34, and the third (`discovery-00-channel-h3`, after "Kinetic Energy at the bottom of the descent") stays at 0.68 to 0.92 from
  layer 11. Rows that are no sign's row answer at most 0.17 to 0.48 at the feed-forward inputs from layer 9 (root mean square
  0.006 to 0.025). So a matrix's input at the sign's row tells the `Kin` line from the `Total` line and one wording from another;
  it does not, by a key of this kind and
  thirty answers, tell the wrong sign from the right sign on the same text.
- *Where the trace parts on the same text (`same_text_sign.py`, `same-text-sign/`; the processor only).* The six wrong answers and
  the four correct answers on that one pair of lines, read at the row that writes the sign in the input each matrix received: no
  difference to speak of through layer 6 (at most 0.017 of the mean input's size, under each group's own spread; a row left out of
  its group's mean falls nearer its own group in 2 to 5 of 10 from feed-forward 1 to feed-forward 6, in 8 of 10 at the two
  matrices of layer 0 and 6 at attention 1); a difference from layers 7 to 11 (0.02 to 0.08; 5 to 9 of 10); from layer 12 to 19
  the left-out rows are nearer their own group in 8 to 10 of 10 (10 at feed-forward 12, 14 and 15 and attention 14 and 18), and
  from layer 20 to 32 in 10 of 10 at 23 of the 26 inputs (9 at attention 20, 27 and 28); the difference is twice the pooled spread
  or more at every feed-forward input from 21 (2.1 to 3.1) and at every attention input from 21 but those of layers 27, 28 and 29
  (1.4 to 1.9), and 0.19 to 0.30 of the mean at
  feed-forward 28 to 34. The six are all above the descent's energy and the four all below, so this is where the numbers and the
  sign together part the trace, and these files cannot say which of the two it is. The correct `Total`-line answers above against
  below, which write the same sign, never part (3 to 6 of 8), but their text is not word for word the same and their spread is two
  to three times larger, so that is no clean reading of the numbers alone. The three answers that write a minus below on the same
  pair, and the nine that write a plus above under another wording, have tokens on record and no trace: replayed, the three
  against the four read the sign alone and the three against the six the numbers alone.

**Which merges write a fork: the direct path to two tokens' scores** (`sign_direct.py`, `sign-direct/`; `fork_direct.py`,
`fork-direct/`; night of November 12; files only, the processor; the kept own answers and fed continuations of
`fed-continuations-replay-original`, card 1, and four tensors of the model file read only).

The engine's last steps are exact arithmetic: the last layer's output times its output scale, the final norm, the token's row, the
cap. The last layer's output is a sum of the scaled embedding and the 105 added parts, each multiplied by the output scales of its own
and every later layer. So before the cap the difference of two tokens' scores at a row is exactly a sum of 106 terms, one per merge:
what the merge added, read along the one direction that separates the two scores. Check on every row read: the eight recorded scores
rebuilt from the kept parts within 0.0002, and the terms sum to the difference within 2e-6. A term is the direct path only: it says
where a choice between two tokens is written, not where its reason is computed.

- *The layers' output scales set what can reach the output directly.* A part added in layer 14 reaches the output multiplied by
  9e-7, in layer 20 by 0.001, in layer 28 by 0.05, in layer 33 by 0.12, in layer 34 by 0.17 (the scales run from 0.018 to 0.87;
  0.088 and 0.029 at layers 13 and 14). At every one of the 136 rows read here the embedding and layers 0 to 19 together add at most
  0.16 to a difference of scores. That much is the model's build, not a finding about the capability: an early layer acts on a
  token's score only through what later layers make of it.
- *The sign (27 fitting own answers at the row that writes the first sign after KE_i; 13 wrong, 14 correct; the three whose text on
  card 1 departs before the sign are listed, not read).* The attention merges of layers 20 to 34 add -1.5 to -2.0 to
  score(' -') - score(' +') in every answer, wrong and correct alike. The feed-forward merges of layers 29 to 34 carry the
  difference: summed, +0.5 to +8.5 in all 13 wrong answers and -1.0 to -10.7 in all 14 correct ones. Feed-forward 33 alone parts
  all 27: -0.8 to +2.7 in the wrong, -1.7 to -6.9 in the correct. On the two lines six wrong answers write (five read, against the
  four correct on the same text; mean difference 5.58): feed-forward 33 gives +2.72 of it, 34 +0.99, 32 +0.72, 31 +0.33, 29 +0.28,
  24 +0.27, all attention -0.13. It goes with the sign written and not with the question's numbers on the two answers that can
  tell: `expanded-087-h3` (a minus, kinetic energy below the descent's) has the wrong answers' feed-forward 33 term (-0.32), and
  `expanded-071-h1` (a plus, kinetic energy above) the correct ones' (-4.77). On a `Total` line the plus already leads by 3 to 6
  at layer 28 (feed-forward 23, 24 and 28). The difference of the groups' traces at layers 7 to 20 (the bullet above) is material
  brought to the row; it is no part of the sign's score.
- *The label (29 rows: the 22 answers that wrote `Kin`, wrong and correct, and the 7 that wrote `Total`).*
  score('Kin') - score('Total') is +0.03 to +9.88 in the first and -4.42 to -1.13 in the second; of the mean difference of 7.43,
  feed-forward 33 gives +2.41, attention 34 +1.33, feed-forward 28 +1.07, 24 +0.63, 23 +0.54, 27 +0.47 (feed-forward of layers 20
  to 28 together +3.21, of 29 to 34 +2.61, attention of 29 to 34 +1.52). Feed-forward 33's term: -0.18 to +2.78 where the answer
  wrote `Kin`, -1.96 to -0.50 where it wrote `Total`. The seven correct answers that wrote `Kin` have the wrong answers' terms.
- *The opening's two near ties (the thirty own answers).* The word after "Here is the" (' calculation' in 4, ' compact' in 26; the
  difference of the two scores -0.83 to +0.52) is a balance of large terms that cancel: attention 34 -12.8, feed-forward 27 +4.7,
  24 +4.3, 34 -3.1, 42 logits of terms in all for a difference under 1. What tilts it between the answers is spread thin:
  feed-forward 34 -0.38, attention 34 +0.22, attention 29 +0.17, no other merge above 0.10. The token after "**Initial State"
  (':**' in 20, ' (' in 10; -2.64 to +0.47): feed-forward 34 -8.2, 32 +4.2, attention 31 +3.1; the tilt is feed-forward 33 +0.44,
  24 +0.21, 34 +0.20, 23 +0.20.
- *The heading's word and the climb's word (rows of fed continuations, 3 to 6 each).* The same layers: attention 34 and
  feed-forward 28 to 34 hold the largest terms (`fork-direct/fork-direct.txt`).

So on the output's own axis every fork of the answer is written from layer 20 on, the late feed-forward merges tell the answers
apart where attention does not, and feed-forward 33 is the largest single share at both the sign and the label. This picks no site
and makes no target: where a score is written is where a change reaches it most directly, and a change that near the readout has to
be shown to move the computation and not only the token. What these tables are, after Astra's reading of them (commits 3b1138b5
and ca66a905): at the sign's row the late feed-forward terms separate the wrong answers from the correct ones, and a construction
at feed-forward 33 from correct traces is admissible for the bounded sign request. At the other forks they are attributions of a
choice between two tokens and not yet a difference between wrong and correct: the correct answers that write `Kin` have the wrong
answers' terms at the label, the heading's rows have no correct contrast, and at the climb's word feed-forward 33's mean term is
+5.18 against +5.08.

**The twelve counterexamples: the sign's row where the sign and the regime disagree** (run `sign-counterexamples-replay-original`,
`studies/04-round-2-overlap/runs/sign-counterexamples-replay-original/result.json`; `sign_direct_counterexamples.py`,
`sign-direct/sign-direct-counterexamples.txt`). Twelve screening answers outside the registered sixty, read again on card 1 under
their own recorded tokens: three that write the minus with the kinetic energy below the descent's energy on the six's two lines,
nine that write the plus with it above under two other wordings. Astra admitted them as diagnostic observations only (round five,
47249c20): no fitting data, no target, no score.

- *The declared reading.* Beside the five fitting minus answers and the four fitting plus answers on the same two lines, the three
  lie nearer the plus mean at layers 12 to 19, as predicted (18, 24 and 24 of 24 merges). At layers 20 to 34 the prediction was
  that they lie nearer the minus mean, since they write the minus. Two of the three do not, at any of the 45 merges; the third,
  expanded-060-h3, crosses at layer 27's feed-forward merge and stays (23 of 45). The split "the middle layers follow the numbers,
  the late layers follow the sign" is not what the record holds.
- *What it holds instead (read after).* The three are not clear counterexamples: each has the kinetic energy 8 J below the
  descent's, the regime's boundary. The fitting plus answer with the same numbers (discovery-15-channel-h1) is a near tie on card 1
  as well. The four answers at 8 J below, minus over plus: -0.15, +0.08, +0.10, +1.70. Over the twelve answers on those two lines
  the place of the state between the two group means ranks with kinetic-minus-descent at 0.92 to 0.96 in every band of layers from
  12 on (with the signs' margin at 0.86 to 0.96; the numbers and the margin rank together at 0.90). Four answers with the same
  numbers hold nearly the same place through layer 26 whatever sign they write (0.09 to 0.55, where 0 is the plus mean and 1 the
  minus mean); the one whose sign is clear against its numbers leaves it from layer 27's feed-forward on and ends at 0.90.
- *The direct path at the twelve.* Feed-forward 33's term goes with the sign written, not with the regime: the nine plus answers
  above the regime have -2.77 to -5.27 (the fitting correct answers: -1.70 to -6.89), expanded-060-h3 has -0.61 (the fitting wrong
  answers: -0.81 to +2.68), and the two near ties have -1.58 and -1.55, between the two ranges. Attention of layers 20 to 34 adds
  -1.47 to -2.03 in all twelve, as in every fitting answer.
- *The two under "Height dropped"* lie within their two fitting partners' distance from each other at 45 of 45 merges of layers 20
  to 34.
- *Feed-forward 33's term against the question's numbers, wording by wording* (`sign_by_numbers.py`,
  `sign-direct/sign-by-numbers.txt`, which also writes the place-between-the-groups table quoted above; the 27 fitting answers of
  `sign-direct.json` and the twelve of `sign-direct-counterexamples.json`, 39 rows on card 1). On the six's two lines (12 answers) the term rises with
  kinetic-minus-descent, rank correlation 0.92: -2.73 to -3.76 at 36 and 60 J below, -0.61 to -1.70 at 8 J below, -0.81 to -0.28
  at 24 and 36 J above, +0.48 and +0.62 at 48 J above. The whole difference of the two signs' scores follows (rank 0.90): -2.41 to
  -4.51, then -0.18 to +2.13, then +1.27 to +4.94, crossing zero near 8 J below. Under the wordings where every recorded answer
  writes the plus the term does not move with the numbers: `KE_{B, before} = KE_A` -2.77 to -4.77 from 4 J below to 44 J above
  (8 answers); "Height dropped" -4.77 to -5.27 from 6 J below to 48 J above (4); a `Total` line -2.71 to -6.89 from 24 J below
  to 88 J above (7). So the wrong answers' feed-forward 33 is not a switch thrown the wrong way: on one wording its output is
  graded in the question's numbers, where on the wordings the model gets right it is the same whatever the numbers.

What it says to a construction at feed-forward 33 built from correct traces: on the six's two lines "the five minus the four"
cannot be told from the difference of their numbers through layer 26, so a target taken from the four's mean state carries the
four's numbers with it. What goes with the sign beyond the numbers is what the feed-forward merges from layer 27 on add. Limits:
twelve rows, a projection on one line, numbers and earlier text that differ between every case and the answers beside it, and no
recorded answer that writes the plus above the regime on the six's two lines.

## The plan the three seats merged (v4, sent to both at about 1:55 AM CT; their replies decide what changes)

Principles
1. One objective: the discrepancy from the FIXED original correct pattern, direction and log size, both groups, the whole registered trace. No
   correction where the current computation agrees within the original-measured resolution.
2. On the model's own text. Unchanged tokens: the affected inputs and merges are still refreshed. Changed tokens: new rollout, registration and
   rows, per answer.
3. Answer-level cells, not group means. After a refresh the error at a step's start is unpaired against the fixed reference; the movement
   inside a step is paired on the current rows.
4. Every reading carries, per cell, how the registration covers it, whether a well-registered cell identifies the capability per answer (two
   separate maps), and whether the text read there agrees with the correct pattern. Unknown is marked unknown, never counted as agreement and
   never dropped from the result.
5. Dose: a written matrix is kept at the stage where the whole-model fitting error over the merges it can change is least, the unchanged
   matrix among the candidates. A search that runs out means "this construction found no further correction", nothing more.
6. Decisions on the fitting 15 + 15 only, cross-validated; answers are described beside and never a loss; the no-fit answers once after
   freezing, reserved-40b once after that.
7. Card 0 is canonical for rollouts and acceptance, because there the original's rollout is the recording itself. Card 1 does paired readings
   and development, labelled.

Sequence
0. Experiment A, `runs/correction-at-19`: reference (Fable), controls (in place), one candidate matrix (Fable), its own-trace reading against
   the after-19 control and the original's floor (this seat), assessment (Astra). An instrument test; no sweep follows from a local curve.
1. Instruments (this seat): answer-level readings and the wording floor; the registration maps; the dose reading with the history check, made
   cheaper; the classes of cells.
2. Solver (Fable): answer-level cells; the token ids a pass replays and the origin of movement as parameters.
3. Experiment B, proposed for Astra to declare or decline: the same one-matrix correction from the original at a layer where the depth makes
   the 160-to-199 stretch (layers 14 to 20 in Fable's table), beside A's layer-9 result, to inform the order of the loop.
4. The loop from the original, declared by Astra when 1 and 2 exist: per matrix, rows on the current own text, the fit, the write, the dose
   reading with the history check, refreshed inputs; at each layer's end a re-rollout of the answers whose text changed and a full own-trace
   reading; passes until the error stops falling; then freeze, the no-fit answers once, reserved-40b once.

Measured unit costs on card 1 for the 30 fitting answers: tokens-only rollout 148 s; history check 138 s; capturing rollout 262 to 320 s;
registration and reading on top 365 to 612 s; a whole given-text replay keeping all 105 merges 519 s with its reading.

What the dose reading's pass costs by how much it keeps (`replay_cost.py`, `replay-cost.txt`; the original on the recordings' text, 14,592
steps, the pass run whole so that it also writes the engine's own first choice at every step):

| merges kept from layer | outputs kept | recorder | written |
|---|---|---|---|
| 34 | 3 | 153 s | 0.36 GB |
| 17 | 54 | 163 s | 6.6 GB |
| 9 | 78 | 182 s | 9.5 GB |
| 0 | 105 | 199 s | 12.7 GB |

The pass itself is the cost: keeping a merge's output adds about 0.45 s for the 30 answers, so a reading of everything from the written
layer on costs nearly the same at any depth, and each of the four passes flags the same 9 answers as not the model's own text on this card.
The reading of the kept rows comes on top and is not timed here; the 519 s above is an earlier pass with its reading, and the two are not
reconciled.

## The loop's steps on the instruments' side (November 13)

The loop of the plan above (sequence 4) left the state page with the compactions of November 12; a peer session pointed at it on November 13
at about 3:00 AM CT, and the four late constructions running that night each declare that they are not it. The solver's side is
`code/signal-trace/pipeline/weight_edit/loop.py` (Fable, c5588be5). This seat's side of a step is two things, and one reading the chains get
while they are built.

**The own text of a state** (`own-text.sh <run> <step>`, which finds the state the step starts from by the hash the last written step
holds and the text before it, and runs `own_text.py <cohort> <state> <recordings | folder holding own/> <steps/n/own>` on a card under its
lock; card 0 is canonical for rollouts). A step from the loop's first state reads the recordings and gets no folder; a step whose state
is the one an earlier folder was made for (the steps between wrote nothing) gets that folder copied, with no card. The history check first; an answer whose every token is still the state's own choice keeps its text and its
registration (from the recordings: the recorded tokens and the overlap's frozen registration); an answer whose text changed is rolled out
again by the state itself and laid on the frozen axis by its computation (`tube.registered`). First run, the primary chain's model after
link 29 (`runs/sign-chain-29-34-original/own/diagnostics/own-text/after-link-29/own/text.json`): 2 of 30 changed, 38.7 s of rollout and
6.0 s of registration after the 147 s check, and both rollouts carry the recorded text up to the step the check named.

**The reading of a stage before it is kept** (`dose-stage.sh <run> <step or link> <power | none>`). `loop lay` writes one stage of a walk into
`<step>/dose/<run>.gguf`. Since 4:15 AM CT November 13 the reading is the stage's own history, captured whole (Astra: "The loop needs an
own-history capture for every dose it may select, not a score of the stale replay after a mismatch ... mismatch triggers refresh, never
exclusion or an answer-token gate"): first the stage's own text (`own_text.py` on the text the step's rows were read on: the history
check, and each answer the stage no longer writes answered again by the stage and registered), then all thirty fitting answers replayed
on that text with every merge kept (`dose_reading.py`; the text is the stage's own, so the engine's first choice must be the supplied
token at every step, and an answer where it is not is named and kept), the arrays staying in `<step>/dose-readings/<power>/capture/` for
the member-level reader, about 10 GB a stage. `whole_model_l.py`'s pooled L of that capture, whole and from the matrix's own merge on (the
merges it can change), by group, is kept in `reading.json` as a description and not the selector
(`records/retrospectives/2026-11-12-next-loop-astra.md`:135-151); the dose score is Astra's, read from the capture. `reading.json` is the
file `loop take` is given: the stage's hash, how many answers it rewords, where the capture lies with every file's hash. `none` reads the
state the step starts from, the unchanged matrix among the candidates. `tab_dose.py` lays a step's stages side by side. Which stage is
kept is the declaration's to say. Its second form reads a state no walk laid (a chain's link as
written, from a copy held to its hash): without `OWN=1` a replay on the recordings' text with each answer's first step not its own, with
it the own-history capture. First run, the replay form (before the change), the primary chain's model after link 29 on the recordings' text on card 0
(3:47 to 4:01 AM CT; `runs/sign-chain-29-34-original/own/diagnostics/written-links/link-29/dose-readings/written-card-0/reading.json`):
before the matrix's merge the reading is the recordings' to 1.5e-11 at every merge and place, as it must be; from feed-forward 29 on, the
wrong group's L against the fixed reference (`correction-at-19`'s, the whole-trace one) is 0.0478174 where the recordings read 0.0478447,
0.99943 of it. Per merge it rises 0.03% at layer 29 and attention 30, falls 0.08 to 0.11% through layers 31 and 32 and 0.45 to 0.52% at
feed-forward 33, layer 33 and attention 34, rises 0.12% at feed-forward 34 and ends 0.03% under at the last merge; by place the largest
changes are at 168 (-4.2%), 182 (-2.3%) and 181 (+2.6%), 252 places falling and 218 rising. The correct group's L is 0 in the recordings
and stays 0. So one link of the sign chain, asked at its sign rows, moves the whole-trace error by about one part in two thousand: the
dose rule the plan names (least whole-model error over the merges the matrix can change) reads nearly flat on such a link, which is for
the loop's declaration to weigh. A local reading on the recorded text, two of whose answers the model no longer writes past steps 228 and
289; no verdict.

**The history check of each written link of a chain** (`history-link.sh <run> <link> [card]`, `history-links.sh <card>`, `tab_history.py`):
is the recorded text still the model's own up to each sign row, which is whether the next link's rows are the model's own computation. A
chain's one model file is written again at its next link, so the check reads a copy held to the hash `written.json` names. After link 29
of the primary chain, on card 0: the recording is the model's own text in 28 of 30 answers, the two others leave at steps 228 and 289 on
near ties (0.0004 and 0.006 logits), and at all 21 sign rows the request names every earlier token is the recording's
(`runs/sign-chain-29-34-original/own/diagnostics/on-the-recordings/link-29-card-0/history.json`). After link 30 (4:00 AM CT) it is 29 of
30: the answer that left at step 228 by 0.0004 logits writes the recording again, expanded-020-h1 still leaves at step 289 (0.005), and
all 21 sign rows stand. The six-supports chain after its link 29 reads the same: 29 of 30, expanded-020-h1 at step 289 (0.008), all 20 of
its sign rows (`link-30-card-0/` and that run's `link-29-card-0/`). Experiment A's candidate, feed-forward 9
on the nineteen-matrix state, left the recordings' text at step 3 in 26 of 30 (above, "Where the text leaves"); this one, at feed-forward
29, leaves none before step 228.

What a step costs, measured (card 0 unless said; the solver's numbers are Fable's, from link 30's log at 12288 wide):

| part | seconds | where |
|---|---|---|
| rows on the step's text | 197 | card 0 (Fable) |
| the request | 26 | card 0 (Fable) |
| one walk of the path, no folds, to 2^0 / to 2^-3 | 335 / 620 | card 0 (Fable) |
| the history check of a state, thirty answers | 147 (138 on card 1) | a card |
| a changed answer's rollout and registration | about 19 and 3 | a card, then the processor |
| a stage's replay with every merge kept | 373 of recorder in three passes on card 0 (257 on card 1, November 12); 593 with the exports between them, which hold no lock since the first run | a card |
| the whole-model reading of a replay | 153 alone, 248 beside a solve and a model copy | the processor |
| the write | 93 | Fable |

A reading stays behind every job of this repository on a card's lock (`after-the-others.sh`): flock wakes waiters in no order, and on
November 13 a history check and an own-text run went ahead of three rows passes that feed the rented boxes. While one of our jobs holds
the lock or waits for it the reading looks again every five seconds and does not queue; when the lock is another project's only, it
waits ON the lock like any card job (since ec6b13aa, November 14: until then it took a free lock only, and a back-to-back series of
another project kept step 2's first capture out for 8.5 minutes; speed review `README.md`, "A reading that only took a free lock").

## The closed chains read whole on their own path (November 13, evening)

Each is the thirty fitting answers the candidate writes itself on card 1, kept at every merge, read by the same instruments as the earlier
candidates (`read-whole.sh`, taken up after the disk stop with `read-whole-again.sh`; `tab_own.py <run>` lays a reading out). A reading
is a description of the candidate on its own answers. It selects nothing and is no assessment: `result.json` is Astra's. Its seat was out when these were read and
is active again since about 8:12 PM CT the same evening; it assessed both from these readings, with no capture made again (cfbdde4c,
`records/retrospectives/2026-11-13-closed-sign-chains-astra.md`: solving link 33 moves the sign farther and completes no repair).

**The six supports' chain (B), 5:52 to 6:18 PM CT** (`runs/sign-chain-29-34-six-supports-original`, model 3a3958f6…, made again in 24.8 s
and released at 6:07:45 PM CT; 20,848 positions, 105 merges; under the run's `own/`: `record/own.json`, `direct/own-direct.json`,
`direct-control/own-direct-control.json`, `whole-model/l.json`, `paired/paired.json`, `row-map/beside.json`).
- **The sign, same row and same history.** All 15 wrong answers still read the original's own tokens up to the original's sign row, so
  what differs at that row is the weights. 8 of the 14 that wrote the minus write the plus (expanded-034-h3, 042-h3, 087-h3, 094-h1,
  095-h1, 045-h1, 046-h3, discovery-05-channel-h1); the primary chain turned 2 (discovery-05, expanded-046). score(' -') - score(' +')
  moves toward the plus in 15 of 15, by 0.94 to 7.88 logits, median 4.58 (the primary: 0.97 to 4.85, median 2.70). Feed-forward 29 to 34
  carry it (median -4.57, of it feed-forward 33's own term -2.67; everything through layer 28: +0.03). The six that still write the
  minus keep 0.48 (discovery-17), 0.73 (expanded-020, 070), 2.80 (092), 3.51 (discovery-10) and 3.93 (068). The history check on the
  recordings' text named the same eight answers (KNOWN, "after link 34 (the chain's final file)").
- **It is the weights, not the wording.** On this model's own text the original's weights favour the minus at 8 of the 9 first rows that
  write a plus (`direct-control`; ten wrong answers have another text and a control, five keep the original's text). On a2-path's text
  the original's own weights favoured the plus at 5 of 6.
- **The text.** 10 of the 15 wrong answers leave the original's text on this card (first at steps 141 to 657, median 142; the primary:
  4). At that token the original held its own choice by 0.01 to 6.77 logits, median 2.07, and the weights moved the margin by a median
  of 3.33 (the primary: 0.53 and 0.77): decisions the edit turned, not near ties. The ten reworded answers are 125 steps shorter to 213
  longer than the original's, median 78.5 longer (7,952 steps of wrong answers become 8,707; the primary: 8,071). expanded-042-h3 now
  runs to the step limit; expanded-001-h3 does under the original too.
- **What the answers say** (the judge's fields: a description of the text, never the measure). Two wrong answers come out whole and
  right, which neither the original nor the primary does for any: expanded-034-h3 (64 J against 90, NO) and expanded-087-h3 (166 J
  against 150, YES). Three more reach the right verdict with the available energy still wrong: discovery-05 and expanded-045 omit the
  descent (51 J for 91, 80 for 140), expanded-095 states 86 J for 236 and a required energy of 0. expanded-046 writes the plus, omits the
  descent (76 J for 136) and answers NO. Seven state the descent subtracted, expanded-094 among them although it writes the plus at
  its sign row (40 J for 240); expanded-042 is cut at the limit before its numbers, and expanded-001 states its available energy right,
  as under the original, and is cut before its verdict.
- **The 15 correct answers.** 14 are the original's text token for token, whole and right, and all 14 sign rows still write the plus
  (the margin moves by -1.58 to +0.26, median -0.05). discovery-01-channel-h1 leaves the original's text at step 503, where the original
  held its choice by 0.00 logits, and then states the required energy as 60 J for 20: the same answer at the same step as under the
  primary.
- **The trace** (whole-model L on the own answers against the reference's frozen target; the original's own answers on this card:
  0.24664, wrong 0.24411, correct 0.00253). L = 0.26323: wrong 0.26086 (direction 0.09102, size 0.16984), correct 0.00237. The primary:
  0.21581 (wrong: direction 0.09440, size 0.11903, correct 0.00237); the single site: 0.22137; the overlap chain: 0.29961 with its
  correct group at 0.07899. B's direction part is below the primary's and the single site's; its L is above theirs by size. Split at
  feed-forward 29 (no link touches a weight before it, so the merges before it differ only through the text they read): before it the
  wrong group's direction is 0.08260 and its size 0.12333 (the primary: 0.08616 and 0.08514); from it on, direction 0.00842 and size
  0.04651 (the primary: 0.00825 and 0.03389). Of the 0.04742 by which B's wrong-group L stands above the primary's, 0.03463 lies before
  feed-forward 29: it is the text of the four answers B alone rewords (the table below), as in the three earlier edits whose rise of
  the own-trace L was the wording (line 321). What the changed weights do to the trace on a fixed text is not separated by this reading.
- **Beside the visit on card 0.** The capture kept 12 answers in 5 min 16 s (5:53:00 to 5:58:16 PM CT), stopped twice on its 40 GiB
  margin while the visit's second stage was being captured (5:58:16 and 6:00:24 PM CT) and went on by itself at 5:59:24 and 6:04:32 PM
  CT; all thirty were kept at 6:05:34 PM CT. The instruments ran 9 min 28 s. Fable accepted that the visit's next stage waits at its 55 GB
  bar while a reading's model is home.

**The primary's links 33 and 34 again, 6:18 to 6:39 PM CT** (`runs/sign-chain-33-34-from-primary-32`, model c36f9158…: the primary's state
after link 32 with link 33 solved, the second ridge read with the roundings held; its 8 kept answers back from the bucket at 6:18:51, the
model made again at 6:22:09 and released at 6:32:17 PM CT; 20,398 positions; the same files under the run's `own/`).
- **The sign, same row and same history.** 5 of the 14 that wrote the minus write the plus: expanded-034-h3, 087-h3, 045-h1, 046-h3 and
  discovery-05-channel-h1. The primary, whose link 33 never solved, turned the last two only. The margin moves toward the plus in 15 of
  15, by 1.43 to 7.48 logits, median 3.78 (the primary 2.70, the six supports 4.58); feed-forward 29 to 34 carry it (median -3.79, of it
  feed-forward 33's own term -1.97). The nine that still write the minus keep 0.22 (expanded-042), 1.07 (095), 1.36 (094), 1.50
  (discovery-17), 1.53 (020), 1.77 (070), 4.37 (092), 4.95 (discovery-10) and 5.39 (068).
- **It is the weights.** On this model's own text the original's weights favour the minus at 5 of the 6 first rows that write a plus
  (seven wrong answers have another text and a control).
- **The text.** 7 of the 15 wrong answers leave the original's text (first at steps 141 to 657, median 141); at that token the original
  held its own choice by 0.01 to 4.71 logits, median 1.58, and the weights moved the margin by a median of 2.25. They are 125 steps
  shorter to 213 longer (7,952 steps become 8,257); expanded-001-h3 alone runs to the limit, as under the original.
- **What the answers say.** The same two wrong answers come out whole and right (expanded-034-h3, expanded-087-h3); discovery-05 and
  expanded-045 reach the right verdict with the descent omitted, expanded-046 omits it and answers NO, nine state the descent subtracted.
- **The 15 correct answers.** As under the six supports and the primary: 14 the original's text, whole and right, 14 of 14 sign rows
  still the plus (the margin moves by -1.59 to +0.24, median -0.03), and discovery-01-channel-h1 reworded at step 503 into the same
  wrong required energy.
- **The trace.** L = 0.19167 (direction 0.07947, size 0.11220): wrong 0.18930, correct 0.00237. It is the lowest own-trace L of the
  candidates read whole in this record (the primary 0.21581, after-19 0.21662), 0.777 of the original's 0.24664, with the correct
  group where the original's is.

**The five read whole, side by side** (`chains_beside.py`, `closed-chains/beside.txt` and `.json`; files only; a description, no ranking).

| on its own answers | sign rows turned, of 14 | wrong answers whole and right / verdict right | wrong answers reworded | correct answers whole and right | L | wrong: before ff 29 + from it | correct |
|---|---|---|---|---|---|---|---|
| the original (card 1) | - | 0 / 1 | - | 15 | 0.24664 | 0.19829 + 0.04582 | 0.00253 |
| the primary (link 33 unsolved) | 2 | 0 / 2 | 4 | 14 | 0.21581 | 0.17130 + 0.04214 | 0.00237 |
| the single site 33 | 2 | 0 / 2 | 2 | 15 | 0.22137 | 0.17692 + 0.04192 | 0.00253 |
| the primary's links 33 and 34 again | 5 | 2 / 4 | 7 | 14 | 0.19167 | 0.15189 + 0.03741 | 0.00237 |
| the six supports (B) | 8 | 2 / 5 | 10 | 14 | 0.26323 | 0.20593 + 0.05493 | 0.00237 |
| the overlap chain (C) | no such row: all 15 take another wording before it | 10 / 12 | 15 | 14 (10 reworded) | 0.29961 | 0.18428 + 0.03634 | 0.07899 |

- **An answer the four sign-request candidates turn ends at one text.** For every two of the primary, the single site, the six supports
  and the links again, every answer both reword is the same text token for token: 2 of 2, 4 of 4, 5 of 5, 2 of 2, 2 of 2 and 7 of 7.
  The seven answers the links again reword are each a text another of them wrote; the six supports add four of their own (expanded-042,
  092, 094, 095) and keep the original's text of expanded-020, which the primary and the links again reword at step 289. The overlap
  chain shares no reworded text with any of them (0 of 5, 2, 11 and 8). On its own path such a candidate is the set of answers it turns:
  what follows a turned sign is the model's one continuation of it.
- **Before feed-forward 29 only the text is read, to the bit.** Over every pair of candidates and every answer both wrote with the same
  tokens (5 to 27 answers a pair), the largest difference of a cell before feed-forward 29 is 0; from it on it is 0.004 to 0.021 of the
  answer's largest cell among the four, 0.14 against the overlap chain. So the part of an own-trace L before feed-forward 29 moves only
  with the answers a candidate rewords. Of the fall of the links again's wrong-group L from the original's (0.05481), 0.04640 lies there
  and 0.00841 from feed-forward 29 on; the six supports stand 0.00764 above the original there and 0.00911 above it from 29 on, and
  0.05404 above the links again there, which is the five answers whose text differs between the two.
- **On a text that stays the original's, the weights alone leave the trace where it was, to a thousandth, and move the wrong answers'
  more than the correct ones'.** Over the merges from feed-forward 29 on, an answer's squared distance from the original correct
  pattern changes by this part of the original own answer's sum there. The links again: -0.00084 to -0.00019 for the 8 wrong answers
  that keep their text (median -0.00043, nearer in 8 of 8) and -0.00004 to +0.00001 for the 14 correct ones (median -0.00001). At the
  medians the primary -0.00031 and -0.00001, the six supports -0.00026 and -0.00001, the single site -0.00010 and -0.00000. The gross
  change (absolute changes summed) is 0.0003 to 0.0009 on wrong answers and 0.0001 to 0.0002 on correct ones at the medians; for a
  wrong answer one registered place carries 0.20 to 0.60 of it under the three chains (0.04 to 0.47 under the single site): place 168,
  among the prompt's last four, or place 182, in the answer's opening. The overlap chain, on the five correct answers that keep their
  text, changes the same sum by -0.027 to +0.005 with a gross of 0.031 at the median. So the fall of the own-trace L under the
  sign-request candidates is the text they come to write: they turn a decision, and on the text they leave alone the trace is the
  original's. (Against the original's own answers the cells before feed-forward 29 differ by at most 2.4e-07, one step of single
  precision; between candidates by exactly 0.)
- **Each reworded answer against the correct pattern** (the mean coordinate along the gap over all places and merges: 0 at the original
  wrong pattern, 1 at the correct one; and the answer's summed squared distance from the correct pattern over the original own
  answer's). The seven the links again reword all move toward the correct pattern along the gap: expanded-034 -0.167 to +0.359 and
  expanded-087 -0.293 to +0.160 (the two whole and right; distances 0.765 and 0.945), expanded-045 +0.116 to +0.385 (1.115),
  expanded-046 +0.090 to +0.294 (1.057), expanded-020 +0.132 to +0.314 (0.850), discovery-05 +0.187 to +0.323 (1.427), expanded-001
  +0.499 to +0.537 (0.968). The four the six supports add: expanded-095 -0.315 to -0.051 (1.071), expanded-094 -0.148 to -0.109 (1.299),
  expanded-042 -0.033 to -0.056 (1.263, cut at the limit), expanded-092 +0.039 to -0.017 (1.011).
- **What the two readings cost the visit.** One wait: its stage 2^15 stood at the 55 GB bar from 6:22:16 to 6:40:16 PM CT while the
  second reading ran (its model 8.7 GiB until 6:32:17, the instruments' copy until 6:39:20, and the first reading's 7 GiB of kept and
  control arrays). Those were read back from the bucket by 6:28:10 PM CT and left this disk only at 7:49:46 PM CT; the second
  reading's were read back by 7:57:41 and left at 7:59:37 PM CT (`own/kept-archive.json` in each run).
- **A waiter of this seat fired wrongly once.** The waiter for the visit's end tested `grep -q canonical-visit /proc/<pid>/cmdline`, which
  does not match here (grep is a shell function on this machine and the arguments are separated by NUL bytes), so it reported the
  visit ended at 8:10:03 PM CT, one minute after it was started, with the visit running. It only reads, so nothing was touched; it runs
  again since 8:10:55 PM CT with `tr '\0' ' ' < /proc/<pid>/cmdline | grep -q`, tried first on the running visit and on another pid.
- **What neither does.** No candidate here repairs the wrong group: under the links again nine sign rows still write the minus and under
  the six supports six, and of the answers that turn the sign three in each omit the descent's energy where the correct answers add it.
  Whether either is a repair of the path is the assessment (`result.json`), which is Astra's.

### The split at feed-forward 29, per merge and per band (`l_by_layer.py`, `closed-chains/l-by-layer.txt`; CPU, files only, 9:21 PM CT)

The table above gives the original's wrong group 0.19829 before feed-forward 29 and 0.04582 from it on. That cut says where the late
chains' first matrix is, not where the distance gathers: 88 of the model's 105 merges lie before it. Per merge the original's wrong
group carries 0.002253 before it and 0.002695 from it on, so 0.812 of the distance stands where 0.838 of the merges are. A cell is
where the state stands after a merge, not where a difference is written: a difference written early is counted again at every later
merge that still carries it.

- **Along the depth.** The layers that carry most are 34 (0.0181, of it 0.0102 at the model's last merge and 0.0065 at feed-forward
  34's), 11 to 13 (0.0391 together, 0.16 of the whole), 25 to 27 (0.0295) and 2 (0.0108). The bands were laid out on November 12
  (above, "Nearly half of `W0` stands before the matrix"); these are the same sums, layer by layer.
- **The two parts lie in different places.** Of the direction part (0.1118) layers 9 to 16 hold 0.39, layers 17 to 28 0.36 and
  layers 29 to 34 0.09. Of the size part (0.1324) layers 25 to 34 hold 0.49. The prompt's 172 places hold 0.239 of the distance.
- **Each candidate against the original, band by band** (the wrong group's L at layers 0 to 8, 9 to 16, 17 to 28, 29 to 34): the links
  again -17.8%, -18.9%, -30.1%, -18.7%; the primary -13.1%, -10.5%, -16.1%, -8.3%; the single site -10.7%, -9.7%, -11.6%, -8.6%; the
  six supports +8.3%, -5.1%, +7.1%, +20.0% (its direction part falls in every band and its size part rises in every band); the overlap
  chain +24.3%, -5.9%, -27.3%, -20.7%. For the primary and the single site the band that holds the matrices they changed falls less,
  in proportion, than every band before it; for the links again it falls as much as layers 0 to 16 and less than layers 17 to 28.
  Before feed-forward 29 a candidate moves a cell only through the text, so the fall of these candidates' L is the rewording of the
  answers they turn, at every depth: what the fixed-text reading above says from the other side.

A description. It names no site to edit: Astra's reading of the same numbers is `records/retrospectives/2026-11-13-after-first-visit-astra.md`
("a numerically registered group distance is not automatically a causal capability discrepancy").


## The readers made ready for the line chain at feed-forward 18, 20 and 21 (November 13, 10:30 PM CT)

Astra adopted `line-chain-18-20-21-original` (`records/retrospectives/2026-11-13-earlier-line-chain-astra.md`): its reading contract asks
for after-link history checks and the final whole own-path reading, and says of this seat's fixed-input reader that it "is qualified
only at layers 29–34; do not apply it to these earlier layers and call it validated". Its first rows pass waits for the card-0 visit to
end, so the readers were made ready before the first link exists. Nothing here reads a candidate: no line-chain link is written yet.

- **The fixed-input reader's own error at feed-forward 18, 20 and 21** (`fixed_input_movement.py check <layer>`, on the processor, files
  only, 80 s, 121 s and 56 s; `fixed-input-movement/check-feedforward-18.json`, `-20.json`, `-21.json`). The check rebuilds the block's
  added part from the kept card-1 states and the original's matrices and lays it against the one the engine kept, at the same 56,110
  positions of the same three stores as at layers 29 to 34 (78 fed continuations, 19 opening transplants, 12 counterexample replays).
  The distance over the kept part's size:

  | feed-forward | median, the three stores | greatest | least cosine |
  |---|---|---|---|
  | 18 | 3.40e-06 to 3.65e-06 | 2.88e-03 | 0.9999961767 |
  | 20 | 3.25e-06 to 3.37e-06 | 2.41e-03 | 0.9999971740 |
  | 21 | 4.52e-06 to 4.69e-06 | 3.48e-03 | 0.9999941898 |
  | 29 to 34, as checked on November 12 | 8.81e-07 to 4.49e-06 | 2.48e-03 to 5.07e-03 | 0.9999914703 to 0.9999969279 |

  At the three earlier layers the block is rebuilt as closely as at the six late ones, and with the same shape of error: a median of a
  few millionths, and a tenth of the positions of two stores above 1.3 to 2.2 thousandths (the opening transplants' greatest is 4.1e-06
  to 6.5e-06 at the three layers). That is the instrument's arithmetic of the block, and no more than that. What a fixed-input reading says
  at these layers is narrower than at 29 to 34: the block's direct term to a score leaves out what every later layer does with the
  changed state, and here sixteen to thirteen layers follow. At the line's rows the reading that carries is the block's own movement
  (the added part and the merged state, direction and size, at every kept position), not the sign's direct term. Whether Astra's
  contract takes the reader at these layers on this check is Astra's to say; until then a line-chain link's fixed-input reading is
  kept as a development reading and named so.
  (Later that evening the chain's declaration, 471cdded, and Astra's contract said it plainly: the reader "is not applied to these
  links". The declared reading makes no fixed-input reading at 18, 20 or 21: `read-whole.sh` leaves the site reader out outside 29 to 34, below.)
- **The history tabulation reads a several-row request** (`tab_history.py`, 480f7aaf). A sign chain's `asked.json` names one row an
  answer (`sign_rows`); the line chain's names every request as a key (answer, back, step, place: `line_request.py`, Fable's, not
  committed at the time). Where a request has keys, the rows of each back (0 is the sign's row) are counted as on the model's own
  path when every token before the row is still the recording's, for the model and for the original on the same card. It was run on
  an existing link's check with a made-up four-back request, to exercise the lines: it is no reading.
- **What needs no change.** `history-link.sh` and `read-links.sh` take the link from the run's own `links/<L>/written.json`, and the
  whole own-path reading (`read-whole.sh`, `read-whole-again.sh`) reads any candidate's model on all thirty fitting answers and 105
  merges. The three kept stores the fixed-input reader opens are whole on this disk (the counterexample store went to the bucket at
  10:04 PM CT for the visit's disk bar and came back at 10:31 PM CT: `runs/sign-counterexamples-replay-original/kept-archive.json`).

**Added at 11:45 PM CT, after the chain was declared and Astra named the readings** (its message of about 11:32 PM CT; the declaration's
`rule.own_reading`): the after-link history checks on card 0 on the exact written link, the whole own-path reading of all thirty and 105
merges with same-history original-weight controls, and "each link's request read at its own merge key by key from the capture".

- **Each link's keys read in the engine at their own merge** (`request_keys.py`; its tests, `code/signal-trace/tests/test_request_keys.py`).
  Astra's reader of a request on exact histories (`pipeline/overlap/request_realization.py`) reads a sign chain's one row an answer at
  links 29 to 34. The line chain asks 52 keys a link (answer, back, step, place) at feed-forward 18, 20 and 21, so the keys needed a
  reader: for every key of every link, the state the final file has on its own answer at the key's row right after the link's merge, and
  the original's own at that row on the same card, each against the key's saved target in the request's joint features. One row's
  arithmetic is Astra's own function, called as it is. A key is read only where the prompt and every token before the key's row are the
  recording's, in the candidate's own answer and in the original's: so a changed token leaves the rows after it "unsupported history"
  and the rows before it read, key by key (a row three before the sign's can be read where the sign's row cannot). A row a capture lacks
  is a "source gap". Neither enters a sum, and both are listed. Beside each key lies what the solver's own reading kept of it on the
  recordings' text (`links/<L>/candidate/keys.json`), to be compared by what a key keeps and never sum against sum (the solver weighs
  each key by the reference's cell). It writes `runs/<run>/own/request-realization/record.json`, where Astra's records of the sign
  chains lie. It was run on a made-up capture in its tests and on no candidate: no link of the chain is written.
- **The whole reading leaves the site reader out outside feed-forward 29 to 34** (`read-whole.sh`): it reads the fixed-input sites only
  at the layers the reader is qualified at, says in its log which it left out, and, for a chain that declares keys, runs `request_keys.py`
  once the captures are kept.
- **The readers' processor threads are a variable** (`PIN`, in `read-whole.sh`, `read-whole-again.sh`, `read-candidate.sh` and
  `history-link.sh`; the default is what they had, 5,6,7,13,14,15). A dose stage's capture is pinned to those same six by
  `dose-stage.sh`, so beside a visit of the loop a reading is started with `PIN=2,3,4,10,11,12`.
- **How they are started for this chain:** after each `links/<L>/written.json`,
  `SCRATCH=<scratch> PIN=2,3,4,10,11,12 bash history-link.sh line-chain-18-20-21-original <L>` (card 0, behind every chain job, the
  link's file held to its sha256); after `model/written.json`, `SCRATCH=<scratch> PIN=2,3,4,10,11,12 bash read-whole.sh line-chain-18-20-21-original` (card 1).

## The first visit's eighteen stages, member by member (November 14, 2:47 AM CT)

`tab_stage_members.py on-policy-33-original 1 canonical` lays every member's score terms at every stage beside the unchanged start's, all
on card 0 (`stage-members/on-policy-33-original-step-1-canonical.txt` and `.json`). It is a description: it does not enter the selection
(2^7, by Astra's rule: the least held sum among the stages that lower both sums; `steps/1/selection.json`) and grades nothing again.

- At every one of the eighteen stages the two terms (direction and size) together rise in all 15 correct members, 4 held and 11 fitted,
  and fall in 13 or 14 of the wrong members that have a term in the unchanged start (14 of 14 at 2^17 and from 2^7 down; one has none:
  expanded-001-h3). No stage lowers a correct member's terms.
- The dose trades one against the other. At the stage taken, 2^7: the wrong members fall by 1,413 (4 of 5 held) and 3,692 (10 of 10
  fitted) and the correct members rise by 487 (held) and 1,355 (fitted), in units of 1e-9; the wrong members stand at 0.88 to 1.00 of
  their terms in the unchanged start. At 2^0 the wrong members stand at 0.23 to 0.48 of theirs and the correct members' rise is
  35,366 and 108,988: 2.2 and 2.7 times what the wrong members fall by.
- Seventeen stages write every answer's text as the unchanged start does; 2^0 rewords one wrong answer (expanded-046-h3, from its step
  141). A compact reading holds only where the text is the base's, so that stage would have been read whole.

The correct members' rise is what the adopted next loop is made to see: its steps alternate feed-forward 34 and 33 and score the known
requests at both sites (`records/retrospectives/2026-11-14-multi-matrix-loop-astra.md`).

## The line chain after link 18, on the recordings' text (November 14, 3:09 to 3:12 AM CT)

`line-chain-18-20-21-original` wrote its first link at 3:08 AM CT (`links/18/written.json`: `blk.18.ffn_down.weight`, ridge 2^-5,
13,070,938 weights changed, 0.0196 of the matrix's size; model sha256 0c54708c…). Its history check on card 0 (`history-link.sh`, 119 s,
14,592 steps, the engine's first choice under the recorded history, no generation and no repair read):
`runs/line-chain-18-20-21-original/own/diagnostics/on-the-recordings/link-18-card-0/history.json`.

- **The sign's row.** Of the 14 requested wrong answers, 13 still reach their sign row on the recording's own text. One writes the plus
  there: expanded-087-h3, ' +' over the recorded ' -' by 1.468 logits (step 141). The other 12 write the recorded minus, ahead of their second choice
  by 0.054 to 5.776 logits. discovery-10-channel-h1 leaves the recording 17 steps before its row ('Total' over 'Kin' by 0.063) and its
  row is off its own path. For comparison, after their four links 29 to 32 the primary chain had turned 2 of 14 and the six supports 4.
- **The rows before it.** At the 38 requested rows one, two and three before the sign's (14, 12 and 12 keys) the model's first choice is
  the recorded token in every answer that is still on its own path there; the check reads which token is first only where an answer
  first leaves, so a margin is not a direction at any other row.
- **The wording.** Six of the 15 wrong answers leave the recording's text (the original on this card: none), five of them at near ties of
  0.013 to 0.063 logits and one at the sign. Of the 15 correct answers 14 are the recording's text whole; expanded-078-h1 leaves at step
  146, four steps after its sign row, 'Energy' over 'Work' by 0.079 logits. All seven requested correct sign rows keep the recording.

What this does not say: what any answer writes after it leaves (a rollout's finding: the whole own-path reading, after link 21), and
whether the requested keys were realized at their own merges (`request_keys.py`, the same reading). Links 20 and 21 follow from this state.

## The parent-weights control made ready for the line's continuation (November 14, 4:38 AM CT)

Astra adopted the reading contract of the next construction before it is declared (`records/retrospectives/2026-11-14-through-readout-astra.md`,
9ea72b1f): eight links from the closed line chain's model (feed-forward 22, 23, 24, 27, 29, 31, 33, 34; 188 link-and-key requests), and of
this seat the after-link history checks, the final whole own-path reading of all thirty answers and 105 merges, and one control no
reader here had: "Include the parent weights on the final candidate's exact history, alongside the existing original-weight same-history
control, to distinguish new weight effects from wording." Nothing here reads a candidate: the continuation is not declared and its parent
has one link of three written.

- **Built: `pipeline/overlap/own_parent_control.py`** (`prepare`, `capture`, `paired`; `tests/test_own_parent_control.py`, 8 cases on
  made-up kept folders). The parent is held to being the candidate's start: its model file's sha256 is the one its `model/written.json`
  closes on (final; a partial parent is refused), the one its own answers were read with, and one the candidate's declaration names; and
  each question's own answer of the two models was read by the same card and recorder (their execution records), or it stops: "a seed
  reading on another card is not a canonical baseline". An answer whose text is the parent's own is not read again (the parent's kept
  own answer is that reading); for each other answer the parent reads the candidate's tokens, the last left to it, kept whole under
  `own/parent-control/`. `paired` lays the candidate against the parent on the same tokens in the fields of the original's control
  (direction, log size, the weighted cost, what each merge adds) under `own/paired-parent/`, and says the first merge that moves: the
  two models are one before the first continued matrix.
- **It changes no existing file.** `own_capture.py`, `read-candidate.sh`, `read-whole.sh`, `own_direct_control.py` and `request_keys.py`
  are bound to the parent chain's own pending reading (the contract: "Do not change a file while the running parent has a command bound
  to it"). What waits for that boundary: the three commands in `read-candidate.sh` and the line in the whole reading; the sign rows read
  with the parent's weights on the same text (`own_direct_control.py` names the original's control folder; it takes the parent's the
  same way); `request_keys.py` on a key list of each link's own, once Fable's declaration gives its form.
- **Not made:** a native pass. Card 1 holds the chain's solve of link 20 (14,096 MiB of the card in use at 3:55 AM CT), so the capture command has run nowhere; it is
  `own_capture.control_capture` with the parent's model and no position left out.
- **What the reading will need on this disk:** the parent's kept own answers (its `own/kept`, which a finished reading sends to the
  bucket), at `prepare` and at `paired`; the control itself is at most the candidate's positions whole, 0.63 MiB a position (17.3 GiB
  if all thirty answers left the parent's text and ran to their limit of tokens; the closed chains' readings were 20,212 to 20,577
  positions, 12.4 to 12.6 GiB whole).
- **The fixed-input site reader** is qualified at feed-forward 29 to 34: of the eight links it reads 29, 31, 33 and 34 as before, and
  22, 23, 24 and 27 are outside it (its arithmetic was checked at 18, 20 and 21 only, above).

## A whole exact reading in 264 s, and native data taken by its identity (November 14, 5:55 to 6:09 AM CT)

For the whole-gap construction (`records/retrospectives/2026-11-14-whole-gap-restart-astra.md`): every step judges the edited model's whole
computation, so a state's whole reading is paid at every step. Preparation only; no model is changed and nothing here is a run of a
construction.

- **`whole_reading.py` makes `dose_reading.py`'s reading in one recorder pass, each answer kept while the card reads the next: 264 s
  where step 1's reading of the same state took 707** (511 s of recorder in three passes, 195 s of exports between them). Same command
  line, same recorder command on the same cases, the same export of every answer, the same files. The thirty answers are one pass (one
  load of the model), and the recorder's line for a case, printed after the case's stream is closed and its `result.json` written, starts
  that answer's export in a pool of processes; the model's hash is read beside the pass.
- **It keeps the same bytes.** The original on the recordings' text, card 0, read twice with it: `turns.npy`, `norms.npy`, `start.npy`
  and `pairs-axis.npy` of all thirty answers, 120 of 120 files, have the sha256 the archive of step 1's unchanged start lists
  (`whole_reading_check.py <out> steps/1/dose-readings/none/capture-archive.json pairs-axis.npy`; no answer flagged, 14,592 steps).
- **The keepers need cores of their own.** Recorder and four keepers on the same three cores: 442 s (recorder 436). Recorder on cores 2
  and 3 (`taskset -c 2,3,10,11`) and keepers on cores 4 and 5 (`KEEP_ON=4,5,12,13`): 264 s (recorder 255), the machine at a load of 16 to
  27 from other projects both times. The recorder works one processor thread hard between the card's pieces.
- **What a whole exact reading still costs:** the recorder, 255 s here against about 215 s on a quieter machine (7 s an answer; a plain
  pass is half of that). Reading 32 tokens at once would be 8 to 9 times less and is not adopted (Astra, 6:03 AM CT: the token-by-token
  method stays; speed review, "What a whole reading of one state costs").
- **Native data is taken by what was computed, not by the step that asked** (the layout this seat proposes for the new construction's
  declaration; Astra's contract `records/retrospectives/2026-11-14-native-reuse-astra.md`). A reading is named by the model file's
  sha256, the recorder's, the card and the sha256 of the text's tokens, and a state is read where it is first made: the reading of the
  candidate a step keeps IS the next step's start, with nothing made again (the loop read it twice, about 1 h 20 min a step). A scorer
  or a target builder asks for `the reading of <model> on <text>` and gets the folder, read or bound; a new request set scores those
  bytes again and captures nothing. The original's readings are in hand: the recordings' own capture (`tube.NATIVE`, local) and step
  1's unchanged start (the bucket, 217 files). Not built: the lookup itself, which is bound to the run layout the new declaration takes.
- **The whole reading is also the check of the text, so a state needs no pass before it.** The loop's stage first replayed the text
  with the vocabulary's scores to see whether it was still the model's own (`own_text.py`: 121 s for step 1's start; 1,265 s for step
  2's, most of it waiting on the card behind another project's agreed hold), then captured the same text whole. The capture runs the
  same computation and keeps the engine's choice beside every supplied token (`first_step_not_its_own`). So a state is read whole on
  the text its parent wrote, and only the answers that reading flags are answered again, registered and read again; where no text
  changed (thirty of thirty at step 2's start and at its first stages) the one pass is all.
- **Rows of several matrices can come from one pass**: the recorder keeps every label a case's `stream_labels` names under `projection`
  (`attn_norm`, `ffn_norm`, `kqv_out`, `ffn_geglu` of any layers: observe_stream.cpp:86-88), where today each matrix has a pass of its
  own (1 to 4 minutes). For matrices solved from one state that is one pass; not measured at many labels.

## Every matrix's rows from one pass, and a reading found by what it computed (November 14, 6:31 to 6:51 AM CT)

Preparation for the whole-gap correction: no model was changed, nothing was written into a model and no answer no fit uses was read.
What it is for: Fable's solve of one matrix is now seconds to minutes (`records/development/2026-11-14-whole-gap-solver/README.md`), so
what a correction over all 70 matrices waits for is its rows and its readings.

- **`rows_reading.py`: the rows of all 70 matrices, on one state, from one recorder pass.** `tube.passed` reads one matrix a pass. Here
  the thirty fitting answers go through the model once, the recorder keeps the input and the incoming state of every matrix (140 outputs),
  and each answer's rows are laid into place beside the pass, in the order `tube.passed` returns: 19,751 rows a matrix, 42.0 GB in all,
  one `.npy` a label and a `rows.json` that names each matrix's rows0 and rows1.
- **Measured twice on the original, card 0** (`rows-reading/rows-first.json`, `rows-second.json`). First, at load 11 to 21: the recorder
  234 s (about 7 s an answer), the whole reading 333 s, because four keepers each waited for the disk at every label. That wait is gone
  (one flush at the end, 2.8 s): the second reading had its rows in place 8 s after the recorder. The second recorder pass itself took
  489 s after 109 s of waiting for the card and its first answer, about 17 s an answer, at load 19 to 23. What slowed it is not
  separated: this seat's own fetches from the bucket ran beside it (sixty starts of the bucket tool, unpinned), and a load report 25 s
  after the pass showed ten cores busy with other projects' checks and one of this repository's scripts (`hostshieldctl load`,
  6:50 AM CT). The recorder waits for processors, the wall this page already names. Seventy single passes of the same answers are 53 minutes of decode (speed review, "What a sweep's passes cost, part by part").
- **The rows are the bits a matrix's own pass gives, and the bits the original's recordings hold.** Both readings against the rows two
  single-matrix passes kept of the original (feed-forward 29 and 33; `rows_reading_check.py <reading> <rows.npz>`): 4 of 4 arrays, 19,751
  of 19,751 rows the same bits, both times (`rows-reading/second-run-and-kept-files.txt`). The first reading against the rows
  input-overlap recorded while the original generated (`rows_reading_check.py <reading> stored`; only the fitting answers' rows of those
  files are opened): 140 of 140 arrays, every row the same bits (`rows-reading/against-stored.txt`). So for the original a fresh pass and
  the stored rows are one thing, at every matrix, with today's recorder on card 0.
- **What it changes for a correction of the whole gap.** Solved from one state: the original needs no pass (its rows are stored and are
  these), and any other state needs one pass of 4 to 10 minutes for all its matrices. Solved in path order: one pass a matrix, 55 minutes
  a sweep, and the state a matrix's output is added to (rows1) is the engine's own state after the write before it, so the passes of a
  sweep already hold what each write did to the trace at its own merge, with no pass added.
- **`reading_identity.py`: a reading found by what it computed** (Astra's contract, `records/retrospectives/2026-11-14-native-reuse-astra.md`:13-23
  and 39-57). `describe` gives a reading's identity from its own records: the model's sha256, the recorder, the engine's libraries, the
  card, how the text was read, and for each answer the prompt's and the text's tokens and whether the text is the model's own whole
  answer. `find` gives, answer by answer, the reading under some roots that holds what a run asks, or why none does. `bind` writes the
  asking run's binding inside that run: the origin's place, its record's sha256, its files and their hashes, `computed_here: false`, the
  misses with their reasons, and no score. It refuses to write inside the reading it binds. 16 tests on made-up readings
  (`test_reading_identity.py`; the sixteenth is Astra's, for the correction below).
- **A replay is a model's own answer only with its ending** (`whole_reading.ending`, same record:51-57): every token the model's choice
  proves the text up to its last token; the end needs an end token (then chosen too) or the generation's limit. `whole_reading.py` now
  keeps that per answer, with the engine's libraries (one sha256 over their names and hashes), the files that made the kept arrays, and
  the driver. An older reading has none of it in its record: its endings are read from its text and the engine's own end tokens, and
  its libraries stay unresolved (the correction below).
- **Tried on a real reading**: step 1's unchanged start, whose arrays are in the bucket. From its two kinds of small records alone
  (`own.json`, 12 kB, and thirty `answer.json`, brought back to a scratch folder) `describe` gives the original's model, recorder and
  card and all thirty texts as the original's own whole answers: 29 close on the engine's end token and one stops at its limit of 768
  (`rows-reading/step-1-start-described.json`). That record's sha256 is the one step 1's `reading.json` names for its capture. Its
  libraries it cannot give: the record holds no hash of them.
- **Corrected, November 14, 7:20 AM CT: the step-1 readings are NOT found, and this section's "30 of 30" is withdrawn.** As first
  written here, a made-up next run asked for three models' own answers on the recordings' text, on card 0, and the original was bound
  from step 1's start 30 of 30 and the model step 2 started from (40ce40b8) from step 1's own reading of stage 2^7, 30 of 30. That
  rested on this seat's own shortcut: an older reading's libraries were taken to be today's when no library file had been written
  after the reading began. A copy or a restore that keeps a file's time can replace its bytes, and Astra reproduced the false match
  that follows and removed the shortcut (afcc558b; `records/retrospectives/2026-11-14-native-cache-accounting-astra.md`, "A concrete
  identity failure found in preparation"). Asked again at 7:19 AM CT with the corrected code: 0 of 30 bound for each of the three
  models, the two real readings missed for "another libraries_sha256" (`rows-reading/bound-for-a-made-up-run.json`, which keeps the
  withdrawn result named as withdrawn). No record under step 1, the speed review, this folder, the rented-solve folder or the
  recorder's folder names a library file with a hash, so nothing on record resolves them. What does not rest on the shortcut stands:
  the tokens, the endings, the 140 of 140 arrays.
- **What that costs.** A reading made before today is read again the first time a run needs it: 264 s for a whole reading of one
  state with `whole_reading.py`, which now writes the libraries' hash, so that reading is found by every run after it. The loop's
  re-read of a step's start (about 1 h 20 min a step) is a lookup only for readings made from now on.
- **A reading made with today's tool is found** (8:21 to 8:26 AM CT, the unchanged original on the recordings' text, card 0, in a
  scratch folder and in no run; `rows-reading/a-new-reading-found.json`). `whole_reading.py` had not run since its record changed. It
  took 217 s (the recorder 214.5 s, the machine at a load of about 5; 264 s at a load of 16 to 27). Its record holds the libraries'
  hash, the files that made the kept arrays, the driver and every answer's ending: 29 on the end token, one at its limit, none
  unresolved. Its 120 arrays are the bytes of step 1's archived reading (`whole_reading_check.py`, 120 of 120). Asked for by a made-up
  next run it is bound 30 of 30, its libraries taken from its own record; the same asked on card 1, of a model nobody read, or under
  other libraries is 30 misses each, with its reason. The arrays (8.9 GB) were removed after: a reading is kept where a declared run
  makes it, and the first run that needs the original's pays these 217 s once.
- **What this asks of an archive from now on:** when a capture's arrays go to the bucket, its `own.json` and `answer.json` stay on this
  disk (kilobytes). The step-1 captures released them with the arrays, so a reading that is whole in the bucket cannot be found here
  without asking the bucket file by file (about 7 s a file with the tool as it is).
- **Not built:** the driver's use of a binding (Fable's), the scorer's use of bound bytes (Astra's), and a reading from a merge on
  (`dose_reading_from.py`'s), which `describe` refuses instead of calling whole.

## The engine's own state at every merge and every row: `states_reading.py`, and the original read with it (November 14, 11:02 to 11:12 AM CT)

Astra's whole-history scorer compares a model's direction and log size with every original frame at every merge, at the answer rows and
the prompt rows (`records/development/2026-11-14-whole-gap-targets/SCORING.md`, "Remaining native reader dependency"). `whole_reading.py`
keeps, at the answer steps only, the float32 turn of the direction a merge and two sizes: a direction rebuilt from those is a sum of
rounded turns, and the prompt rows are not there. Astra asked this seat for the direct states, on the original first. The declared run
is `studies/04-round-2-overlap/runs/whole-states-original` (its `declaration.json`, `run.sh`, `result.json`).

- **The reader** (`states_reading.py <declared run> <cohort run> <original | model file> <recordings | capture folder whose text is
  given>`): `whole_reading.py`'s pass, command for command (one recorder process, `capture residual stream`, each fitting answer's prompt
  and then its text one token at a time, no filter, no stop, so every decision has the engine's scores), after the research verifier has
  passed on the run's declaration. What it keeps is the engine's output itself. Nothing is derived, normalized or rounded.
- **The contract, `whole-gap-reading/1`** (in `<run>/reading/`):
  - `nodes/<merge>/state.npy`, merge 0 to 104 (3L the attention's, 3L+1 the feed-forward's, 3L+2 the layer's last): float32
    [19751, 1536], the engine's own output of that merge's ADD, the bytes the recorder read off the card. All 105 are in the residual
    stream as the engine holds it (`space: residual`), **the last one included: merge 104 is the RAW last merge**, before the last
    layer's output scale and before the output norm. The whole-gap target file's merge 104 is in output-norm space: whoever compares
    composes that normalization once, from this raw state.
  - `terminal/l_out.npy`: float32 [19751, 1536], `l_out-34`, the last merge times the last layer's output scale as the engine put it
    out (what the output norm reads), at every row. `terminal/result_norm.npy`: float32 [14592, 1536], the engine's own output norm of
    it, at the answer rows only. The engine computes it only where it computes scores: every answer step, and in the prompt one
    position (127, the last of the prefill's first call, in all thirty answers), which is no field over the prompt and is not kept.
    So a composition of the output norm can be held to the engine at the answer rows and at no prompt row.
  - `row-map.json`: order `all_answers_then_all_questions`, the order of `rows_reading.py` and of the target file's row map; per
    answer `id`, `steps`, `question`, `answer_rows_from`, `question_rows_from`, `prompt_sha256`, `text_sha256`
    (`reading_identity.of_tokens`). Answer row s of an answer is input position question + s, where the decision for its token s is
    computed; question row q is input position q.
  - `answers/<id>/answer.json`: the tokens read, the first step that is not the model's own choice (none here), its margin, the steps
    under half a logit, whether it ended on an end token, the prompt positions that had an output norm.
  - `reading.json`: the model's, the recorder's and the engine libraries' sha256 (`reading_identity.libraries`), the card, the driver,
    the fusion switch, the recorder's arguments, the declaration's sha256, where the text came from; every array's file, shape, dtype
    and sha256; each merge's connection, layer, kind and its two operands; the table of answers with each one's ending
    (`whole_reading.ending`: a text is the model's own whole answer only with every token its choice and an end token or its limit).
  - A direction and a log size are not stored: they are x / |x| and ln |x| of a stored row, in the precision the consumer declares.
  - It is not named `own.json` and `own/`, so `reading_identity.describe` does not take it for a reading of turns.
- **How a row is held before it is kept:** the engine's ADD is made again from the two operands the same pass recorded (the state it
  adds to and the block's normalized output: `export_stream_traces.connections`' three a layer) and must give the kept row bit for bit,
  and the record's operation and source names must be that ADD's; one row that does not ends the pass. It held at all 2,073,855 rows
  (19,751 by 105).
- **The original, read:** 11:06:32 to 11:10:37 AM CT on card 0, 235.7 s (the recorder 225.1), load 8 to 10; 141 files, 12.95 GB. In 30
  of 30 answers every supplied token is the model's own first choice; 29 end on the end token and `expanded-001-h3` at its limit of
  768: all thirty recordings are this model's own whole answers.
- **Held to the two readings of the original on record** (`states_reading_check.py`, `qualification/record.json`; a comparison is a
  count of rows that are the same bits, no tolerance):
  - input-overlap's raw rows (`state-feedforward-<L>.f32`, the state each feed-forward's contribution is added to, recorded while the
    original generated): the 35 attention merges, 510,720 of 510,720 answer rows and 180,565 of 180,565 question rows.
  - the baseline's float32 direction and log size (`controlled-bf16-1/baseline`), by the baseline's own derivation
    (`tube.unit_and_log`, float32, card 0): all 105 merges, 1,532,160 of 1,532,160 answer rows and 541,695 of 541,695 question rows,
    for the direction and for the log size; the last merge's question rows are also its second reading of the prompts alone, 5,159 of
    5,159.
  - Only the fitting answers' rows of those sixty-answer files are read (`tube.rows_of`, through a memory map).
  - For a consumer deriving in float64 from the raw rows: x / |x| is at most 2.59e-07 in a coordinate from the baseline's float32
    direction (2.7e-08 for the median row at the median merge), ln |x| at most 4.51e-07 from its float32 log size. Described, not a
    bound.
- **The first pass was stopped** (11:03:20 to 11:04:55 AM CT, no answer kept): this reader's own check wanted the output norm at the
  answer steps and nowhere else, and the engine writes one at prompt position 127. The check was wrong; the declaration's revision says
  so. The reader now also ends the pass at an answer's first failure (the raw streams of answers nobody keeps were filling the disk: 16
  GB in 95 s).
- **Room:** 12.95 GB a state. The reader starts only if the arrays and the raw streams of six answers fit beside the 40 GiB margin
  (`fed_replay.MARGIN_GIB`). The reading is in the bucket with its bytes read back (`logs/bucket.log`) and is kept here for the scorer.
- **Found by what it computed, so the original's states are read once (11:25 AM CT).** `reading_identity.py` now describes, finds
  and binds a states reading beside a reading of turns, and never takes one for the other (`asked(..., kept='states')`; 22 tests, 6 of
  them new). Asked for the original's own answers as states on card 0, under today's recorder and libraries, the thirty are bound 30 of
  30 to `whole-states-original/reading` in 0.05 s, from records alone: each answer with its rows in the row map and its ending, the
  reading's 107 arrays with their sha256 named once. The same asked on card 1, of a model nobody read, or as turns is 30 misses each
  with its reason, and with one answer's last token changed 29 are found and that one is "another text"
  (`states-reading/found-by-identity.json`; the binding was written in scratch, in no run). A run that needs the original's direct
  states binds this reading and makes no pass; a changed state is read where it is first made and found the same way after.
- **Bound by the scorer (Astra, eddd6b35, 11:39 AM CT).** `whole-context-score-original` binds this reading and scores both
  diagnostic target artifacts over all thirty fitting answers, 19,751 rows and 105 merges with no other capture; the raw terminal
  rows times the layer's scale give `terminal/l_out.npy` bit for bit at all 19,751 rows, and the output norm composed in float64 is
  within 3.49e-05 in a coordinate of the engine's own on the 14,592 answer rows
  (`records/retrospectives/2026-11-14-direct-state-score-astra.md`). **The reading is now that run's declared input**
  (`whole-context-score-original/declaration.json`, `inputs`), so it stays on this disk under the verifier, and it leaves only
  through the declared-input archive (`archive_inputs.py`), never by a plain release after the bucket's read-back.
- **A changed model's own history, as the instruments stand** (read from the code at 12:12 PM CT November 14; no changed model was
  read). `own_text.py <cohort run> <model> recordings <out>` checks every recorded token against the state's own first choice, rolls
  out only the answers whose text changed, and lays each of those on the original by its computation alone (`tube.registered`):
  `<out>/own/<id>/answer.json` holds the state's own tokens, and `pairs-original.npy` holds `[this answer's step, the original's
  step]`, the column order and the basis ("original-computation") `whole_context_score.score` asks of a changed history's
  correspondence. `states_reading.py <run> <cohort run> <model> <out>` then reads the states on that text: it takes the tokens from
  the same `own/<id>/answer.json` by the two lines `whole_reading.py` takes them with, which the loop's stage readings ran on changed
  states. The record that binds a reading's sha256 and a target's to those pairs was left here as "the scorer's caller's to write",
  and nothing wrote it: the next bullet.
- **The record between the two readers and the scorer** (`states_correspondence.py`, 12:31 PM CT November 14; files only, no card, no
  model). `whole_context_score.score` refuses a reading that holds one changed history without a correspondence bound to the
  reading's sha256 and the target's, and its one caller (`main`) passes none, so the first changed state of a sweep would have reached
  the scorer with no way in. `python states_correspondence.py <reading folder> <capture folder> <target folder> <out.json>` writes that
  record once (`whole-gap-correspondence/1`, basis `original-computation`): for every answer whose text changed, both texts' hashes and
  `pairs-original.npy` as integers, with the file's sha256; an answer with the original's text gets no entry, as the scorer lays it by
  identity. It refuses, and writes nothing, when the capture and the reading are of two models, when the reading is not the target's
  answers in its order, when an `answer.json` is not the file its capture recorded (`text.json`), when the states were read on another
  text than the capture holds, when the cohort's recording of the original is not the text the target was compiled from, when the
  original's text is laid any other way than one on one, or when a changed answer's pairs are not the scorer's own rule
  (`complete_overlap.validate_pairs`). **Seen:** 7 tests on the scorer's own changed-history fixture (`test_states_correspondence.py`:
  the scorer refuses the reading without the record and scores it whole with it, read back from its file). On the captures stored
  under the runs, with no native run: of the 41 that hold `pairs-original.npy`, 168 of 168 changed histories' pairs are the scorer's
  rule against the affine target's row map and 1,042 of 1,042 unchanged ones are the identity; from the 32 that are `own_text.py`'s
  output the record is made 32 of 32 (109 changed histories, 851 unchanged; the largest record 73 KB), a stand-in stating the
  capture's own text where no states of that model exist (the other 9 are other tools' captures, without `text.json` or with an
  older one). **Not seen:** no changed model has been read by `states_reading.py`, so the scorer has not run on a real changed
  history. The line chain's link-18 model is on this disk (7 of 30 answers reworded); reading it is a native read of a stopped
  construction's model and is Astra's to admit (asked at 12:31 PM CT). The record states where the pairs come from and holds them to
  their bytes; whether a score through them is admitted is Astra's.
- **Astra's decision, and the gap it found, closed at the producer (12:42 PM CT November 14).** Astra accepts the record as file and
  format preparation, not as qualified native correspondence, and does not admit the link-18 reading: a changed history is scored for
  the first time on the first changed state of an admitted construction
  (`records/retrospectives/2026-11-14-changed-history-correspondence-astra.md`). It found a real gap in what is described above:
  `own_text.py` hashed `answer.json` and `pairs-axis.npy` when a capture closed and never `pairs-original.npy`, so the record's
  sha256 of the pairs was taken when they were consumed, and any other whole one-step path put in their place after the capture
  would have been given the name "original computation". Closed where the pairs are made, not where they are read:
  `own_text.py` now writes into `text.json`, when the capture closes, the sha256 of every answer's `pairs-original.npy` (`files`), the
  original's turns each changed answer was laid on with their sha256 (`original_computation`: `wide-overlap/exported/<id>/turns.npy`,
  what `tube.registered` reads; an answer kept from a capture before keeps what that capture bound), and the card, its driver and the
  engine's libraries beside the recorder. `states_correspondence.py` (now `whole-gap-correspondence/2`) carries only those hashes,
  holds the files to them, and refuses a capture without them: **every capture made before this change is refused** (the 32 above
  among them; they stay diagnostic sources and no hash taken today stands in their place). It also refuses a capture and a reading
  that are not of one execution (recorder, libraries, card, driver, the card's number, each as both recorded it) and a changed answer
  whose capture names no original computation. What it still does not establish, and says in the record itself: that the
  registration is right, that the original computation named is the reference a comparison calls for, and that the execution is the
  admitted one or the original control's; those are the declared caller's to hold (Astra's, same record). **Seen:** 12 tests on the
  scorer's fixture, among them another whole monotonic path put in the pairs' place after the capture closed (refused), a capture
  that never bound its pairs (refused) and each of the five execution fields differing (refused). The identity lines `own_text.py`
  now writes give, on this machine today, the five values the states reading of the original recorded (5 of 5), and the original's
  turns are on this disk for 30 of 30 fitting answers. **Not seen:** `own_text.py` has not run since the change (it needs a card and
  a declared run; it compiles), so no capture holds the new lines yet.
- **The reader's protocol for the two declared experiments** (`reader_protocol.py`, `reader-protocol/controlled-bf16-1.json`, made
  1:04 PM CT November 14, sha256 8d764d29…; files only). Astra re-admitted both contextual fields as native experiments at 12:56 PM CT
  (`records/retrospectives/2026-11-14-contextual-native-experiments-astra.md`) and left to this seat to "bind the full native reader
  and producer-bound whole-gap-correspondence/2 to the declared original computation and admitted execution". Each instrument
  recorded its own execution and nothing said which one a construction is held to, or that the original the pairs are laid on is the
  original the comparison is made against. `python reader_protocol.py make <cohort run> <control reading> <out.json>` writes one file a
  declaration names by its sha256: the original control (`whole-states-original/reading`, reading.json c027fe34…, 19,751 rows, 105
  merges) and the programs that made it; the one execution a changed state's capture and reading must both be of (recorder
  0b4ea036…, engine libraries 06ffecec…, NVIDIA GeForce RTX 5080, driver 615.71.10, card 0, fusion off, `capture residual stream`),
  each equal or refused, no tolerance; each fitting answer's original computation (`wide-overlap/exported/<id>/turns.npy` with the
  sha256 its own export recorded, the file held to it: 30 of 30); the cohort's frozen reference and registration source; and the 15
  programs of the acquisition with their sha256. `reader_protocol.held`, which `states_correspondence.py` now calls before it writes
  (its fifth argument is the protocol), refuses a capture or a reading of another execution than the control's, of another cohort,
  reference or registration source, registered by another `tube.py` or read by another `states_reading.py`, `whole_reading.py` or
  `reading_identity.py` than the protocol names, and an answer laid on another original computation than the protocol's; the
  record carries the protocol's sha256. **Measured, not asserted: the original the pairs are laid on is the control's own
  computation.** For every fitting answer and every merge whose incoming state the control keeps (70 of 105: each layer's
  feed-forward and layer merge), the exported turn is made again from the control's two stored float32 states and the export's own
  two float64 sizes, as `export_case` makes it, and is the exported turn bit for bit at every answer step: 2,100 of 2,100 arrays
  (57 s the first time, 25 s with the files in memory). A layer's first merge adds to the layer before's scaled output, which the
  control keeps only at the last layer, so those 35 turns are not made again; each state they involve is the incoming state of
  the next merge's turn, so every stored state of the control at the 14,592 answer steps and all 105 merges enters a compared
  turn. **Disclosed:** the control was read with `reading_identity.py` at 650b18d0… (afcc558b) and the file is fed8fcbd… now
  (213bc55c added the finding of a states reading); what the reader takes from it is `libraries`, and `libraries`, `digest` and
  `ENGINE` are the same source in both (`readers_that_differ_now` in the protocol). `states_reading.py` and `whole_reading.py` are
  the control's. **A protocol is made again when a bound program changes** (`tube.py`, `ladder.py`, `loop_requests.py` and
  `gap_targets.py` are among the 15), under another name, and the declaration revised. 17 tests (`test_states_correspondence.py`).
  **Not established here:** that the registration pairs rows that are the same reasoning; that a score through it is admitted
  (Astra's entry point); and nothing has run on a changed model: the first changed state of the first declared construction is
  the first.
- **The record and the protocol through Astra's entry point, as this seat writes them (1:15 PM CT November 14).** Astra's caller
  (`pipeline/weight_edit/whole_context_compare.py`, 4c0703e9) consumes the protocol file above directly. Its own eight tests
  hand-make the correspondence record and the inner protocol, so the seam between the two seats' code was read, not run. Run now, files
  only: `states_correspondence.record`'s own output and a protocol with the keys `reader_protocol.make` writes go through the entry
  point's `comparison_protocol`, `verify_correspondence_files` and `compare` on its own changed-history fixture and score whole
  (`test_states_correspondence.py`, the 18th test). And the real files: `reader-protocol/controlled-bf16-1.json` and the real control
  reading pass the entry point's control checks (reading, model, rows, merges, answers, the programs that made it: 6 of 6) and its
  execution equality, with one match for each reader and for `tube.py`; with the control stood in as a candidate the entry point
  refuses it ("Candidate was made by another reader than the protocol binds"), which is the disclosed older `reading_identity.py` and
  is what it should do: a candidate is read with the programs the protocol binds now.
- **The two declarations held to this seat's files, and one sentence of them that is not true of `own_text.py` (2:07 PM CT
  November 14).** Fable declared `whole-gap-affine-original` and `whole-gap-mates-original` at 2:02 PM CT (4f649cd7) and started
  the first sweep at 2:06 PM CT. Read against the files: each run's `reader-protocol/controlled-bf16-1.json` is this seat's file
  byte for byte (8d764d29…) and is the sha256 its declaration names; the 15 bound programs are the files as they lie now, 15 of
  15, in both; all 42 declared inputs are on this disk; `run.sh`'s `own`, `reading` and `correspondence` cases give the three
  programs the arguments they take, on card 0 under `after-the-others.sh 0`. **What does not hold:** `execution.before_each` says
  every native pass and both readers call the research verifier through `check_native_run`. The sweep's passes do
  (`run_a_observation.run`) and `states_reading.py` does (its line 180); `own_text.py` and `history_check.py` never did: line 77
  starts the check and line 117 the recorder, directly, and neither file names the verifier. The defect is this seat's (the
  command was given to Fable without saying so). `own_text.py` is not changed while its sha256 is bound in the protocol and in
  both declarations; for these two runs the verifier is run immediately before `run.sh own`, and the sentence is corrected in each
  declaration's next revision: both are Fable's, told at 2:06 PM CT. Fable did both at 2:08 PM CT (6673e544, revision 1 of each
  declaration: the verifier before the `own` step, in the mates run's `run.sh` and by hand in the affine run, whose script was
  running), and Astra's review holds the affine run's hand-run call necessary (bb9dea83). Open, this seat, once neither run reads with the bound
  revision any more: `own_text.py` calls `check_native_run` itself when its output lies in a declared run, and the protocol is
  made again for that revision.
- **Where the two experiments run, for their declarations (this seat's placement, 1:06 PM CT November 14; arithmetic from measured
  parts, nothing timed end to end).** Every recorder pass and both changed states' readings are on card 0 under `gpu0.lock`: it is the
  protocol's execution, and a reading on card 1 is refused by the protocol. A sweep in path order is one chain (a pass, a solve, a
  write, the next pass), so two cards do not shorten one experiment; they let the two experiments run side by side. The first
  experiment's solves go on card 1 under `gpu1.lock`, the second's on card 0 under `gpu0.lock` between the passes, and every solve
  of one experiment is on the one card its declaration names (the two cards' sums differ at 3.8e-10 and each repeats itself: speed
  review, "What is left of a solve"). Why not both on card 1: the solves are the larger part (about 44 s a matrix on card 1 and
  0.77 of that on card 0, against 9.5 to 11.9 s a late pass), so two chains solving on one card wait on each other for most of a
  solve, where a pass waiting behind the other experiment's solve on card 0 waits a part of one. By that arithmetic the two finish
  in about an hour and a half side by side where one after the other is about two and a half hours; a late pass counts only once
  the driver's comparison across a real write has held, and until then a pass is whole (38 to 102 s). Card 0's lock is shared with
  another project's sessions, which are not ours to stop. **Disk:** 72 GB free at 1:05 PM CT; a run keeps about 22 GB (the working
  model 9.31, the changed state's reading 12.95) and a reading needs about 9 GB more while it is made, so the two runs end 12 GB
  under the 40 GB floor unless the three 9.31 GB model files of the stopped constructions (Fable's, 27.9 GB) leave before the
  second run's reading.
- **Not shown:** a changed model read with it (no changed model was read; the declaration's rule); the same on card 1.

## The same-history controls, and the readers on an explicit list of answers (November 14, 7:08 to 7:50 PM CT)

Asked by Astra after it assessed both whole-gap runs (`records/retrospectives/2026-11-14-whole-gap-native-assessment-astra.md`; the
evaluation's contract: `records/retrospectives/2026-11-14-final-transfer-astra.md`). Two things, both preparation or control: nothing
here regrades either run, no model was changed, and no answer outside the fit was generated, read or listed.

- **The same-history control, made for both frozen candidates.** A changed model's score on its own answers differs from the
  original's for two reasons at once: its weights and the text it now writes. The control is the ORIGINAL's weights read on each
  candidate's own text (`states_reading.py <control run> controlled-bf16-1 original <candidate run>`; card 0, every row, all 105
  merges), so with the mapping, the reference and the target held fixed the difference splits into what the new text does under the
  old weights and what the new weights do on that text. Runs `same-history-control-mates` and `same-history-control-affine`
  (declared 93a3e692 before any pass; results 6946c866): 18,375 and 21,443 rows a merge, the recorder 199.1 s and 299.0 s. A
  control is by construction not the reading model's own answer, and the primary scorer goes on refusing it; it has a record of its
  own (`states_conditional.py`, format `whole-gap-conditional-correspondence/1`), which lays no pair and takes the candidate's own
  mapping from its capture and its primary record, and a check (`states_conditional_check.py`, `whole-gap-conditional-check/1`) that
  holds the control by bytes to the original control wherever both were given one input: every question row, and each answer's rows
  up to the first step where the candidate's text is not the recording's (6,750 and 6,614 rows a merge, the same bits at all 105
  merges). Beside it: the original's own choice is not the supplied token at 0.072 of the matched-context text's steps and at 0.220
  of the affine text's. The loader that scores a conditional reading is Astra's and is not written. 17 tests
  (`test_states_conditional.py`).
- **Which answers a reader reads is fixed before it generates or opens anything (`cases.py`).** The four readers (`own_text.py`,
  `history_check.py`, `states_reading.py`, `reader_protocol.py`) took `tube.picks` each for itself. They now take their answers
  from `cases.chosen`: with `CASES` unset, the fitting answers, exactly as before (their records' strings are byte for byte the old
  ones); with `CASES=<manifest>`, exactly the manifest's answers after `cases.held` has held it to the cohort's membership (format
  `whole-gap-cases/1`, role `no-fit`: the whole complement of the fitting selection, in the cohort's order, each in its outcome
  group; another membership, a duplicate, an unknown id, a fitting answer or an omission is refused). Every capture, reading and
  protocol records its selection, and `reader_protocol.held` refuses a capture or a reading of another selection than the
  protocol's, so a fitting protocol cannot be relabelled for an evaluation, nor the other way. The protocol is made again as
  `reader-protocol/controlled-bf16-3.json` (83e41020…, 19 instruments; the same control, execution, cohort, reference and
  registration; 2,100 of 2,100 turn arrays made again from the control). `python cases.py make <cohort run> no-fit <out.json>`
  writes the manifest; it has not been run on the real cohort and is run when an evaluation is declared, not before.
- **One evaluation's steps, taken up where they stood (`evaluation.py`, 11 tests on made-up files).** `plan <evaluation.json>`
  says what is made and what is next; `run <evaluation.json> <step>` runs one step after its bindings are held. Nine steps:
  the original on the cases' recordings (the evaluation's own control), the protocol on it, the frozen model on the recordings
  token by token (the supplied-history replay), the frozen model's own answers, their reading and its record, the original on those
  answers (the same-history control), its record and its check. The manifest, the model and the protocol are named by sha256 and
  hashed again before a step that reads them. A step counts as made only when its file is of this evaluation's model and manifest:
  a file of another model or another selection is refused and never overwritten, so nothing is chosen on a restart. A reading begun
  and not finished is set aside under `<run>/failed/` with every file kept and read again whole. It writes no declaration: each
  reading is a declared run, and `states_reading.py` runs the verifier on that run's declaration before it reads. It scores
  nothing.
- **What is there for the answers outside the fit, by name and limit only.** The limit of tokens each answer was recorded under
  is the same for all 60 (768, `wide-overlap/members.json`), and the registration every capture binds
  (`trace-overlap-complete-registered`) holds a pairs file for all 60, so the capture's sources resolve for an explicit list as for
  the fitting one. Read of those answers: their ids, their prompts' lengths and their limits. Nothing else.
- **The budget (`evaluation_budget.py`, `evaluation-budget.json`).** At the bound, every answer taken at its limit: 28,200 rows
  a reading (23,040 answer steps and 5,160 prompt positions), 17.24 GiB a reading (the fitting thirty's five readings: 11.22 to
  13.10), and the reader asks 25.21 GiB beside its 40 GiB margin before it starts. On card 0, from the rates of the five finished
  readings (10.8 to 17.4 ms a row, quiet to loaded) and the two finished captures (65 to 102 ms a generated step, check, rollouts
  and registrations together): 306 to 491 s a reading, 1,490 to 2,359 s a model's own answers, 40 to 64 minutes a model and 85 to
  136 minutes for both with the one shared control. Seven readings are 120.65 GiB at the bound; with each model's three sent to
  the bucket and read back before the next model starts, 68.94 GiB are here at once. An expectation below the bound would need the
  answers' lengths, which are not read.
- **Not shown.** No step of `evaluation.py` has run on a real model, card or answer: its commands are the readers' own usage
  lines, held to the two finished runs' `run.sh`, and its plan was printed in the real environment on a made-up `evaluation.json`.
  The target, the reference and the scoring rule of an evaluation are Astra's and are not in it.

## The evaluation's scorer held to the readers, and the fields in its room (November 14, 8:35 to 8:50 PM CT)

Astra's request (its a5f9ffdd): hold `pipeline/weight_edit/evaluation_native.py` and
`records/retrospectives/2026-11-14-whole-gap-evaluation-rule-astra.md` to this seat's `cases.py`, `reader_protocol.py`, the producers
and `evaluation.py`, name any seam that stops the freeze, and put the target fields in the room. Read from the code and from fitting
records; no answer outside the fit was generated, read or listed (two facts about them are file existence only, named below).

**What holds, seam by seam.**
- *The selection.* The scorer calls `cases.held(manifest, members)` with the members of the membership file its declaration binds.
  `held` and `tube.picks` read each member's id and original group and nothing else, in the file's order, so the correspondence's
  `members.json` (what `ladder.cohort` starts from) is a membership it can bind. A reading's `reading.json`, the capture's
  `text.json` and the protocol each carry `cases` with the role and the sha256 of the manifest's bytes, which is what the scorer
  compares with its own binding of that file.
- *The four roles are `evaluation.py`'s four runs*: original = `baseline` (`states_reading.py … original recordings`), supplied =
  `supplied` (`… <model> recordings`), candidate = `own-reading`, control = `control`; `capture_text` is `<own run>/own/text.json`,
  the two records are `own-record` and `control-record`, the check is `control-check`. Each reading is a run of its own.
- *The producers it calls again* have the arguments it gives them: `states_conditional.primary_record.record(reading, targets,
  capture, recorded, protocol, protocol_sha256)` and `states_conditional.record(…)`; its `recorded` (the baseline's own tokens) is
  what the fitting path passes from the cohort.
- *A supplied replay loads as its conditional reading*: `ConditionalReading` asks of a reading only what `states_reading.py` writes
  for any model on any text (the two same-history controls are that reader's, on a text not the model's own).
- *The original's stored computation exists for the answers outside the fit*: `wide-overlap/exported/<id>/turns.npy` and
  `index.json`, and the frozen registration's `pairs/<id>.npy`, 30 of 30 by file existence. The protocol and the registration of a
  changed answer read them.

**What must be settled before the freeze.**
1. *The freeze reaches three seats' files.* The scorer refuses when any program the protocol binds is not at the protocol's bytes
   at scoring time, and a capture or reading made under other bytes is refused by `held`. The protocol binds 20 programs: this
   seat's 13 (`own_export.py` among them since tonight), the recorder, `tube.py`, `ladder.py`, `loop_requests.py`, `gap_targets.py`,
   `complete_overlap.py` and `export_stream_traces.py`. From an evaluation's baseline to its last score none of them may change:
   at the bound 85 to 136 minutes of card 0 for both models, and the fields' compilation and the scores beside it. A change in
   between does not damage a result; it refuses the score and the acquisition is made again.
2. *The protocol measures the original on these answers before it is written.* `reader_protocol.py make` makes the original's
   exported turns again from the baseline's stored states, bit for bit, for every evaluated answer, and writes nothing when one
   differs. On the fitting thirty it was 2,100 of 2,100. On the answers outside the fit it is first known when their baseline
   exists. A refusal there stops the acquisition before any edited model is read, and no way around it is built: it would say
   that card 0 today does not give the original's stored computation of those answers.
3. *The target comes before the two records, not before the readings.* `own-record` and `control-record` bind the target
   manifest's sha256; `evaluation.json` names the target and `plan` shows both steps blocked until it stands. The baseline, the
   supplied replay, the own text, both readings and the control need no target.
4. *No supplied replay of a changed model has been made on real answers.* The reader is the one that made both controls, and the
   first real one is the evaluation's own. One on the fitting thirty for a frozen model is 5 to 7 minutes of card 0 after its
   model is restored (the two sweeps' own readings took 312 and 403 s), and would also be the fitting L(W1,H0); whether that is
   wanted is Astra's.

**The fields in the room** (`evaluation_budget.py`, made again; `evaluation-budget.json`). A `whole-gap-targets/1` field is
1,293,600 bytes a reference row over its 105 nodes (each node: the unit direction in float64, the log size, two loss weights, two
status words), read from a compiled field's own manifest: 23.80 GiB on the fitting thirty's 19,751 rows, which is the store's
size. At the bound (28,200 rows) one model's field is 33.97 GiB and both are 67.95. The scorer declares every scored array of the
candidate, the supplied replay and the control, and the field's, as inputs that must be here, so one model is scored with the
baseline, its three readings and its field on this disk: 102.92 GiB at the bound, 142.92 free before it starts with the readers'
margin. One batch of the own text's raw capture is 14.0 GiB at the bound and gone when the batch is exported. The order that
keeps it there: per model, the three readings beside the shared baseline; the candidate's model file leaves (no later step reads
it); its field comes back for the three scores; the field and the three readings go to the bucket and are read back before the
next model starts. What is here of the fitting constructions when an evaluation is declared (the control readings, the fields,
the original's control) leaves through its own records first.

Not shown: anything about an answer outside the fit beyond its count, its prompt's length, its limit and the existence of the
three files named above; the scorer on real evaluation records (none exist); the bound's figures as an expectation (an answer is
taken at its limit).

## The instruments that measured by tokens are removed (November 14, from 9:49 PM CT)

Marcos's direction of the evening of November 14 (relayed by claude-research at 9:34 PM CT; Astra's request to this seat at 9:47 PM
CT): for five days the work was brought back to the trace and each time returned to measuring by tokens, so whatever takes a
token, a token's score, a word, a stated number or a judged answer as a measurement of an edit is taken out of the project, and
is not kept behind a warning. The judge is the computational trace: where the edited model's path overlaps the correct one it
stays, and where it diverged the gap closes.

What this means for this folder: a program stays only if a trace cannot exist without it (the model writing its own text, the
registration of that text on the axis by the computation, the state at every merge and row, each row's place on the reference,
the hashes that say which text a reading was made on) or if it reads states alone. The findings these instruments made stay in
this record above and in their folders as history; no instrument is left to make them again, and they are not what a result is
judged by.

**Removed in the first pass** (37 files; git keeps them; no finished run's declaration holds one as an input):
- the sign's score and its direct path: `sign_on_texts.py`, `sign_direct.py`, `sign_direct_counterexamples.py`, `own_direct.py`,
  `own_direct_control.py`, `score_movement.py`, `sign_by_numbers.py`, `sign_by_regime.py`, `sign_decision.py`, `fork_direct.py`;
- decisions and answers counted or judged: `decisions_changed.py`, `decision_ladder.py`, `ladder_first_rung.py`,
  `corrected_answers.py`, `turned_by.py`, `label_swap.py`, `label_by_model.py`, `height_wording.py`, `opening_forks.py`,
  `opening_controls.py`, `second_feeds.py`, `token_histories.py`, `sharp_stretch.py`;
- the tables of those: `tab_history.py`, `tab_movement.py`, `tab_own.py`, `tab_stage_members.py`;
- the scripts that ran them on a candidate or a chain: `history-link.sh`, `history-links.sh`, `read-candidate.sh`,
  `read-links.sh`, `read-whole.sh`, `read-whole-again.sh`, `label-swap/read.sh`, `label-swap/second-feeds/read.sh`,
  `text-and-weights/ladder/read.sh`, `text-and-weights/read-decisions.sh`.

**Removed in the second pass** (November 14, 10:46 to 10:55 PM CT; source only, no engine run; 141 and 104 tests pass):
- `own_text.py` runs no check of a text and counts no wording: every answer is generated from its prompt and registered, and
  its record no longer holds how many answers' text changed, in which group, from which step, or whether a rollout followed a
  check (`text_changed`, `text_changed_by_group`, `rollouts_that_do_not_follow_the_check`, `the_check`, the per-answer
  `first_step_not_its_own`, `steps_before` and the two fields after them). `ROLL_ALL` is no longer read: it is the only way.
  One identity of inputs stays, because the scorer lays it so (`context_score.py`, `whole_context_score.py`: a history that
  is the reference's token for token is compared position for position, and anything else is refused for it): such an answer
  keeps the reference's registration. On the original the computation writes those same files, 30 of 30 (speed review,
  "Registering every answer by its computation"); for a changed model Astra decided the same that evening
  (`records/retrospectives/2026-11-14-identical-input-correspondence-astra.md`): identical complete inputs keep the identity,
  since the changed model's states are then measured at the same computational events and a path made again there could
  relocate the change being measured;
- `rollout_tokens.py` (tokens only, "what the judge reads of it"; nothing called it);
- the readers' scores of tokens: `states_reading.py` and `whole_reading.py` keep no margin and no count of steps under half a
  logit, and no line saying whether an answer's tokens are the recording's; whether a SUPPLIED token is the engine's own
  choice stays, because it says whether a fed pass is the model's own generation (the identity a reading is reused by);
- `history_check.py` is out of the reader protocol's programs (a protocol made from now on does not bind it), and nothing of
  this seat's runs it; `own-text.sh` and `judge_profile.py` print seconds and how many answers were read;
- the box image's tracker and the Decode Machine's paired-results import went at 10:12 PM CT (e881246b, 1d1d8556).

**Removed in the third pass** (November 14, after the 11:01 PM CT commit; source only, no engine run; 97 tests pass:
`tests/test_own_parent_control.py`, `test_remote_dose_payload.py`, `test_whole_context_compare.py` and this folder's five):
- programs that existed to judge or compare answers' text, removed whole: `chains_beside.py` (the five closed candidates'
  judged answers side by side; its record `closed-chains/beside.json` stays), `prefix_transplant.py` with
  `label-swap/transplant/read.sh` (what a pass wrote after a transplanted opening; its record
  `label-swap/transplant/prefix-transplant.json` stays and `pipeline/overlap/opening_replay.py` reads that file, carrying no
  `then_wrote` or `judged` into a case), `conditional-members/answers.py` (each answer's judged fields per model and the
  count of "complete repairs"), `deep-think-analysis-2/frame_margin.py` and `regime_frame_judged.py` (the judge over the
  recorded answers; `frame-margin.json` stays);
- the speed review's `as-we-go` readers read the trace alone: `as_we_go.py` has no `answers` reading (ten answers generated,
  judged and counted; the argument is refused), `own_compare.py read` judges nothing and its `own.json` holds no `written` or
  `table`, `report.py` and `states.py` print no answer counts, and `follow.py`, `read_states.sh`, `run_card1.sh` no longer ask
  for the removed reading;
- `members.py`: `outcomes` lists each answer's coordinate by its recording's group and `projected` each answer's error by
  stretch, neither beside the judge's available energy or "what fails first"; no count of answers by wording is printed. The
  arrays that say from which step an answer's rows have other INPUTS than the recording's stay (`same_tokens`,
  `first_other_step`, `first_other_place`): `floor` splits a trace movement by them, as a statement about the rows' inputs;
- `whole_model_l.py` keeps no count of answers with the recording's tokens; `conditional_members.py` keeps no flag of whether
  the control wrote the recording's tokens; `prompt_features.py` keeps no first decision's token or scores and pairs none;
- `pipeline/overlap/own_capture.py record` computes no judge's fields, transitions or scores at the first differing token
  (the import scan for the judge now finds `pipeline/overlap/wide.py` and the judge's own test only).
A record made before any of these keeps its old fields; nothing reads them.

**Removed in the fourth pass** (November 15, from 12:00 AM CT; source only, no engine run; the pipeline's whole test suite
passes after it, 767 tests, and this folder's five test files), once Fable's dated revisions
of the two stopped runs' declarations no longer held them in the tree (6e78051e: `on-policy-33-original` revision 12,
`line-chain-18-20-21-original` revision 4; each bound hash has beside it the commit whose blob is those bytes):
- `history_check.py` (a supplied text fed token by token and the engine's highest scores read at every step);
- the stopped loop's stage readers: `dose_reading.py`, `dose_reading_from.py`, `dose_record.py`, `dose_compact.py`,
  `dose-stage.sh`, `tab_dose.py`, `compare_box_readings.py`, and what only carried them: `dose_bind.py`, `whole_from.py`,
  the rented box's dose payload (`dose_files.py`, `dose-stage-box.sh`, `dose-on-box.sh`, added by Astra for that loop) and
  the four programs under `dose-from/`; with them the six test files that pinned them (`tests/test_dose_compact.py`,
  `test_dose_reading_from.py`, `test_dose_stage_box.py`, `test_remote_dose_payload.py`, `test_dose_bind.py`,
  `test_whole_from.py`);
- kept: every measurement the family made (`dose-from/README.md` and its 134 records, `dose-reading-check/`, the runs' own
  folders). What a replay of a given text needs is `whole_reading.py`, which writes the same files in one pass and keeps no
  token score; the three launchers that called `dose_reading.py` call it (`conditional-members/read.sh`, its first draft,
  `text-and-weights/read.sh`).

**Still here, named as gaps, with what holds each:**
- Fable's part of the stopped loop: `pipeline/weight_edit/loop_dose.py` and `loop.py`, and the driver `loop-step.sh` under
  `records/development/2026-11-12-correction-objective/`, which calls the removed `dose-stage.sh`. Fable retires its part
  after the new loop's connection (the rows on the model's own answers, the paired request, the start from a reached model).
  Astra's compact scorer went at 12:18 AM CT November 15 (5fbea45a: `compact_dose_score.py`, `read_loop_visit.py`,
  `revise_33_34.py`, three tests; `records/retrospectives/2026-11-15-retired-compact-scorer-astra.md`); no payload is restored;
- `pipeline/weight_edit/loop_requests.py:120-149` (Fable's, the stopped loop's path) requires the fields `own_text.py` no
  longer writes, and refuses a record made from now on; it leaves with `loop_dose.py`;
- the three whole-gap declarations (`whole-gap-affine-original`, `whole-gap-mates-original`,
  `whole-gap-recipient-prefix-original`) still name `history_check.py` among their instruments, unrevised because the
  input-archive records hash those declarations as provenance; their bindings name bytes git holds (Fable: e3d1d57a at
  2118b14d, ed4830fc at 0117c49d). This seat's `sign-route-whole-gap-mates`, whose four passes were that program's, carries
  a dated revision naming the same blob (e3d1d57a at 2118b14d);
- `pipeline/overlap/wide.py` runs the judge once, to make the frozen cohort's two groups: the cohort's provenance, not a
  reading of any model since;
- the programs of closed card-1 runs that chose or located rows by the recorded text, kept because each run's kept store is
  still read and its declaration names its program: `pipeline/overlap/counterexample_replay.py` (cases chosen by the
  recorded sign and regime, with the recording's historical labels beside them), `row_map.py` (the rows of a path found by
  what its own text states), and `deep-think-analysis-2/regime_frame.py`, `regime_frame_reordered.py` (the original's
  recorded sign against the question's numbers, no judge);
- the development scripts of November 12 that read the recorder's scores (`records/development/2026-11-12-correction-objective/`
  `same_text_*.py`, `same_line_direction.py`, `own_part_same_line.py`, `terminal_check.py`), history of findings already recorded.

**What a reading of a fed run is now**: `fed_path.py` (the trace of the fed run against the unfed run and the two original
patterns, both laid on the axis by the computation alone).

## The fed runs' path, read on the trace alone (November 14, 10:11 to 10:41 PM CT)

**What was asked** (Marcos, the evening of November 14, through claude-research): from the fed word to the sign, where and at
which merges the fed run's trace parts from the same question's unfed trace, and whether it arrives on the correct answers'
trace before the sign is written; on the original weights, with the question and the history held fixed, with no card and no
new run. No such reading existed: the November 12 fork tables read token scores, and the replay's kept states had never been
laid on the overlap. It is made now from those kept states (`runs/fed-continuations-replay-original`, 78 cases, every merge at
every position, the original model on card 1; brought back from the bucket 9:39 to 10:15 PM CT through
`f312069c9a54-kept.json`), files only, on the processor, 10:11 to about 10:30 PM CT:

    (from code/signal-trace)
    python fed_path.py read   <out> [.label | .climb | .heading | .heading-label]   one reading.<kind>.json and one .npz a run
    python fed_path.py tell   <out>                                                 the windows and bands (fed-path/told.txt)
    python fed_path.py places <out> 90 200                                          place by place (fed-path/places-90-200.txt)
    python fed_path.py bands  <out> 149 ... 200                                     band of layers by band (fed-path/bands-149-200.txt)

The runs' arrays (`fed-path/*.npz`, 86 MiB, 48 runs) stay on this disk and out of git; `read` makes them again in about twenty
minutes while the store is here.

**How it is read.** Each run's answer steps are laid on the frozen axis by the computation alone (`distance_path` over the turns
at all 105 merges, as the overlap was registered), and the fed run's steps on the unfed run's the same way. No step is paired by
what it writes and no token's score is read. Two coordinates come from records, not from this reading: the fed row (the first row
whose input is the fed word, from the replay's own record; the rows before it are the unfed run's bit for bit, which the reader
asserts), and the axis places 168 and 169, where the earlier registration records put the row that writes the sign and the rows
that read it (`records/KNOWN.md`, "members_in_the_gap.py, row 168"). They say where on the axis a table is read. For a run, at
place p and merge m, with z its mean unit state there, c0 and c1 the two original patterns and g = c1 - c0: `along` =
(z - c0).g / |g|^2 (0 at the wrong pattern, 1 at the correct one), `to correct` = |z - c1|^2 / |g|^2 (1 is the whole gap), `moved`
= |z_fed - z_unfed|^2 / |g|^2; cells are summed over the merges of a band and the places of a window, the last merge read as what
the output norm passes on. Beside each run: the fitting members of each group, each against its own group's pattern made of the
other members.

**What it is read against, and what that limits.** The reference is the frozen one, and it is not fitting-only (Fable's note of the
same evening): the axis is expanded-089-h3, which is not a fitting answer; c0 and c1 are the centres of all thirty and thirty
members; and they are card 0's recordings while the store is card 1's (each unfed run against its member's recording: the same
count of steps in 22 of 29 questions, `along` 3.6e-4 apart at the median cell and 25.1 at the most). So everything below is a
description. By Astra's ruling of the same evening it sets no target, selection or closure threshold of a loop.

**1. Where the fed run parts from the unfed run: at the fed row itself, at every merge, most in layers 9 to 13, and it does not
come back.** The 15 wrong answers fed the label ('Total' for 'Kin'), median over the runs, the distance between the two runs' raw
states over the unfed state's size at the steps the registration sets together:

| steps after the fed word | all layers | 0 to 8 | 9 to 13 | 14 to 23 | 24 to 34 |
|---|---|---|---|---|---|
| 1 | 0.50 | 0.46 | 0.77 (cosine 0.72) | 0.32 | 0.54 |
| 2 to 5 | 0.44 | 0.46 | 0.75 | 0.41 | 0.36 |
| 6 to 20 | 0.28 | 0.19 | 0.54 | 0.33 | 0.25 |
| 21 to 60 | 0.15 | 0.08 | 0.34 | 0.20 | 0.11 |
| 61 on | 0.33 | 0.30 | 0.64 | 0.29 | 0.30 |

The first row that reads the fed word, merge by merge: 0.05 at merge 0, 0.24 at 6, 0.45 at 18, 0.61 at 24, 0.77 at merge 36
(layer 12), 0.29 at 42, 0.20 at 48, then 0.43 to 0.59 from merge 60 to the last. Over the whole path from the fed row the two runs'
place means lie 6.9 squared gaps apart.

**2. From the fed row (place 149) through the sign's row (168) and the rows that read it (169), the wrong group's fed runs stand
on the correct pattern, in every band of layers.** All 105 merges summed at each place, median over the 15 runs
(`places-90-200.txt`):

| place | along: unfed | fed | fed higher in | to correct: unfed | fed | fed lower in | correct members, held out: along, to correct | wrong members, held out |
|---|---|---|---|---|---|---|---|---|
| 148 | +0.007 | +0.007 | 0 | 1.67 | 1.67 | 0 | +1.18 ± 2.21, 11.9 ± 25.9 | -0.006 ± 0.174, 2.67 ± 2.11 |
| 149 | -0.029 | +1.613 | 15 | 1.09 | 0.59 | 15 | +1.05 ± 0.80, 1.33 ± 1.32 | -0.028 ± 0.011, 1.11 ± 0.03 |
| 156 | -0.005 | +1.666 | 15 | 2.36 | 0.97 | 13 | +1.00 ± 0.82, 2.21 ± 1.92 | +0.045 ± 0.259, 2.85 ± 1.05 |
| 160 | -0.020 | +1.504 | 15 | 1.32 | 0.72 | 13 | +1.02 ± 0.73, 1.91 ± 2.45 | -0.009 ± 0.091, 1.46 ± 0.45 |
| 166 | -0.107 | +1.288 | 15 | 1.51 | 0.63 | 15 | +0.93 ± 0.65, 1.82 ± 2.27 | -0.018 ± 0.258, 1.52 ± 0.20 |
| 168 | -0.050 | +0.958 | 14 | 1.31 | 0.59 | 15 | +0.93 ± 0.58, 1.80 ± 2.35 | -0.033 ± 0.039, 1.39 ± 0.34 |
| 169 | -0.033 | +1.021 | 15 | 1.15 | 0.33 | 13 | +0.97 ± 0.05, 0.86 ± 0.93 | -0.028 ± 0.021, 1.13 ± 0.05 |
| 173 | -0.024 | +0.959 | 14 | 1.17 | 0.40 | 15 | +0.96 ± 0.15, 1.07 ± 1.03 | -0.019 ± 0.027, 1.27 ± 0.19 |

At place 168 by band (`bands-149-200.txt`), unfed to fed: layers 0 to 8 -0.06 to +0.44, 9 to 13 -0.04 to +0.83, 14 to 23 -0.06 to
+1.11, 24 to 34 -0.01 to +1.01. At every place from 149 to 175 the fed runs' whole distance to the correct pattern is under the
unfed runs' at the median.

**3. What the gap between the two patterns is made of on that stretch: the label.** The replay's own record of what was fed says
which label each answer wrote: all 15 wrong fitting answers wrote the one label, and of the 14 correct fitting answers it fed, 7
wrote the same label and 7 the other. Their unfed runs, the same tables (median over each set; all layers, and at 168 the last
band):

| place | 15 wrong: along, to correct | 7 correct with the wrong group's label | 7 correct with the other label |
|---|---|---|---|
| 149 | -0.029, 1.09 | -0.029, 1.09 | +1.636, 0.60 |
| 160 | -0.020, 1.32 | -0.007, 1.18 | +1.515, 0.71 |
| 166 | -0.107, 1.51 | -0.096, 1.36 | +1.304, 0.60 |
| 168 | -0.050, 1.31 | +0.067, 1.12 | +1.164, 0.56 |
| 168, layers 24 to 34 | -0.014, 1.28 | +0.170, 1.04 | +1.305, 0.50 |
| 169 | -0.033, 1.15 | +0.968, 0.76 | +0.974, 0.30 |
| 169, layers 0 to 8 | -0.050, 1.13 | +1.072, 0.86 | +0.899, 0.30 |
| 173 | -0.024, 1.17 | +1.150, 1.38 | +1.015, 0.36 |

So between the label and the sign's row the correct group is two routes, and the gap between the group patterns there is the
label's own difference: a fed run reads the other label and lands where the seven answers that wrote it are (1.5 to 1.7, past 1,
as they are), while the seven correct answers that kept the wrong group's label stand on the wrong pattern, in every band of
layers, until place 168, and reach the correct pattern at 169, in the first layers already, where the sign is in the row's input.
Before the sign is read, what sets those seven apart from the fifteen wrong ones in this coordinate is +0.17 against -0.01 of the
gap in layers 24 to 34 at place 168, the sign's own row. The correct members' spread says the same: ± 0.80 along the gap at place
149, ± 0.05 at 169.

**4. After the sign the fed runs stay toward the correct pattern and fall part of the way back.** Windows counted from the fed
row's place, all layers (`told.txt`): 20 to 40: along -0.03 unfed, +0.78 fed (14 of 15), to correct 1.65 to 0.92 (15 of 15; the
correct members held out 1.50 ± 1.01, the wrong 1.76 ± 0.24); 40 to 80: +0.95, 2.02 to 1.21; 80 to 160: +0.65, 4.32 to 3.71
(correct 2.89 ± 0.62, wrong 5.06 ± 1.94); 160 to the end: +0.21 unfed, +0.54 fed (12 of 15), 6.78 to 6.65 (10 of 15; correct
5.78 ± 1.40, wrong 7.94 ± 3.29). The whole path from the fed row: along +0.10 to +0.87 (14 of 15; correct 0.92 ± 0.28, wrong
0.15 ± 0.13), to correct 4.12 to 3.47 (12 of 15; correct 3.37 ± 0.62, wrong 4.92 ± 1.76), the same in each band of layers (0.79
to 0.92 along).

**5. The other feeds.** The whole path from the fed row, all layers, median, unfed to fed:

| what was fed | runs | along | to correct |
|---|---|---|---|
| wrong: the label | 15 | +0.10 to +0.87 (14 higher) | 4.12 to 3.47 (12 lower) |
| wrong: the label and the climb's word (two wordings) | 2, 3 | +0.17 to +0.91, -0.08 to +0.58 (5 of 5) | 3.71 to 3.12, 5.48 to 3.75 (5 of 5) |
| wrong: the heading alone ('1' or '2' for 'Point', fed at place 81) | 3, 3 | +0.13 to +0.14, -0.11 to -0.09 | 5.49 to 5.51, 6.17 to 5.94 |
| wrong: the heading and the label | 3, 3 | +0.13 to +0.63, -0.11 to +1.06 (6 of 6) | 5.49 to 4.49, 6.17 to 2.89 (6 of 6) |
| correct with the other label, fed the wrong group's | 7 | +1.21 to +0.36 (0 higher) | 2.77 to 4.23 (0 lower) |
| correct with the wrong group's label, fed the other | 7 | +0.82 to +0.96 (3 higher) | 3.39 to 5.77 (3 lower) |
| correct: the label and the climb's word | 1, 1 | +0.99 to +0.95, +1.18 to +0.40 | 4.65 to 6.46, 2.65 to 3.69 |

The heading alone moves the path where it is fed (0.6 to 41 squared gaps at places 97 to 148) and leaves it where it was from
the label on (moved 0.02 to 0.05 at places 149 and 150; along -0.01 and -0.04 there). The correct answers fed the wrong group's
label go to the wrong pattern's coordinate at once (-0.02 at place 149) and at the sign's row lie far from both patterns (to
correct 5.0, 7.4 and 6.2 at places 166 to 168, from 0.60, 1.23 and 0.56). The correct answers that kept the wrong group's label
and are fed the other move onto it as the wrong answers do (+1.62 at place 149) and end farther from the correct pattern over the
whole path than they were, at the median (4 of 7).

**What it says.** On the original weights, one word upstream moves the whole path that follows, at every layer from the first
row, and for the wrong answers the path it moves to is the correct pattern's from that row through the sign's row and the rows
that read it. And the same tables say the group patterns' gap between the label and the sign is which label an answer wrote: half
the correct answers are on the wrong pattern there, and are right. For answers that share the label, the two groups' paths in
this reference differ before the sign only at its own row, in the last layers, by about 0.18 of the gap; from the row that reads the
sign on they differ in every layer. A target made of the correct group's centre therefore asks a wrong answer, from the label to
the sign, for the other label's route, which no correct answer with its label takes; what those seven do differently before the
sign is not resolved by the two centres. Which reference resolves it is Astra's decision and Fable's fitting-only cut (the nearest
mixture of the fitting correct answers' own states on one card); this reading is evidence for it, not a target. The link to what
was read before: on November 12 the answers that read the identical line before the sign (10 wrong and 7 correct of the sixty
recordings) were found to part in their states from layer 14's attention on, with no cell above the shuffled level at layers 0 to
9 (`records/KNOWN.md`, "same-text, layer 14 attention onward" and "own_part_same_line.py"; all sixty recordings, so descriptive as
this is). The two agree: on a shared line the difference before the sign is late and at the sign's own row, and it is small beside
the label's.

**What it does not say.** Nothing here is a repair, a closure or a floor. The reference holds the answers no fit uses and is of
another card. The first places after the fed row carry the fed word itself in the row's input, and after it each run reads its own
text, so a later place compares two runs that have read different text, laid together by their computation. The medians are of
15, 7, 6, 3, 2 and 1 runs. The 14 feeds of the sign alone were never replayed, so there is no run in which only the sign's row
changes. The sets of seven are named by the replay's record of what it fed, not by anything read here.

## The question rows' turns at every merge, from records already kept (November 15, 2026, 1:05 AM CT; files only)

Astra asked, for the compiler of the success-set reference (`records/retrospectives/2026-11-15-success-set-loop-freeze-astra.md`), which
existing record of the original supplies the turns at the prompt's rows. `prompt_turns.py`, `prompt-turns/held.json`: 53 s on the
processor, no engine, no card, only the fitting answers' rows read (`tube.rows_of`, through a memory map).

**What was missing.** A turn is unit(merged) - unit(incoming) at a merge (`export_stream_traces.export_case`). The original's exports
(`wide-overlap/exported/<id>/turns.npy`) hold it at the answer steps only. `whole-states-original/reading` keeps every merge's merged
state at every answer row and every question row, and inside a layer a merge's incoming state is the merge before it: 70 of the 105
merges, which is what `reader_protocol.turns_made_again` makes again. Each layer's first merge adds to the layer before's exit
(`l_out-<L-1>`) or, at layer 0, to the scaled token embedding (`inp_scaled`), and the reading keeps neither.

**The record that keeps them.** `input-overlap/inputs`: `state-attention-<L>.f32` and `question-state-attention-<L>.f32`, the state
the attention's contribution is added to (`input_overlap.record`: the label `inp_scaled` at layer 0, `l_out-<L-1>` after), read off the
recorder's stream while the original generated each of the cohort's sixty answers, at every answer step and at every position of the
question. Each file's hash is in the registry record `input-archives/0ac0603daa26-inputs.json` (the bucket's copy verified November 14,
6:37 PM CT); the files are on this disk.

**What holds it** (rows that are the same bits; nothing is a tolerance):
- Answer rows, the 35 attention merges: the turn made from the kept incoming row, the reading's merged row and the export's own two
  sizes is the exported `turns.npy`, 1,050 of 1,050 arrays (30 answers by 35 merges); the kept incoming row's float64 size is the
  export's, 510,720 of 510,720. With the 70 merges the protocol already makes again, the reading and this record make the exports'
  turns at all 105 merges.
- Every row, layers 1 to 34: the kept layer exit is float32(the reading's `pe_out` row of the layer before times the model file's
  `layer_output_scale` of that layer), 496,128 of 496,128 answer rows and 175,406 of 175,406 question rows. The reading's own terminal
  `l_out-34` the same way, 19,751 of 19,751.
- Every row, layer 0: the kept `inp_scaled` row is float32(the model file's token row of the row's input token times
  float32(sqrt(1536))), 14,592 of 14,592 answer rows and 5,159 of 5,159 question rows.
- Question rows, all 105 merges: both sizes of the turn are positive and finite, 541,695 of 541,695.

**So a turn at a question row** is: merged, the reading's state of the merge at the row; incoming, inside a layer the reading's state
of the merge before, and at a layer's first merge the kept `question-state-attention-<L>` row, which is the same bits as a fixed
function of what the reading keeps and the model file. On the one edited model's reading on this disk
(`whole-gap-mates-original/reading`, brought back from the bucket at 12:56 AM CT) the terminal exit is that function too, 18,375 of
18,375 rows: a states reading of a changed model carries what its own question rows' turns need, with no further pass.

**What it does not say.** The input-overlap record does not name its card; its feed-forward states are the card-0 reading's rows bit
for bit (`whole-states-original/qualification/record.json`) and its attention states are held as counted here. Whether the record is
admitted as a source of a reference is Astra's: the registry holds it as a backup of bytes that admits nothing by itself. The two
fixed functions are counted on the original at every layer and on an edited model at the terminal only. Nothing here says a
correspondence laid on these turns pairs rows that are the same reasoning.

## The producer of the reference's turns (`reference_turns.py`; source, fixtures and a trial on the real files, November 15, 1:19 AM CT)

Astra accepted the source above for the fitting answers and asked for the producer of what her compiler reads
(`success_sets.py`, `whole-gap-reference-turns/1`; `records/retrospectives/2026-11-15-success-set-implementation-astra.md`). No
artifact is made here: the turns are written inside the reference's own declared run.

**What it writes.** `reference_turns.produce(folder, root, reading, kept, exports, scales, bound)`: for every answer of the reading,
`<id>/question.npy`, float32 [question positions, 105, 1536], the turn at every merge as `export_case` makes it (float64 from the
two float32 rows, each size a row at a time), and `turns.json`, the record the compiler binds: the reading's hash, model and
execution, the definition, every merge's keys, and each answer's two descriptors. The answer steps' descriptor names the original's
export itself (`wide-overlap/exported/<id>/turns.npy`), after every merge of it is found to be the bytes made here from the reading
and the kept incoming rows; an export that is not is refused, and nothing is copied. Beside them the record keeps each kept file's
hash, each answer's span in it, the qualifying records' bindings and what was counted.

**What it reads of the kept files.** Each answer's own span, a slice of a memory map (`Kept`: the spans come from the kept folder's
own index and are held to the reading's steps and question positions). Each file is held whole to the bytes and the sha256 the
registry record names, which hashes it and reads no row. With the model's 35 layer output scales given, every kept layer exit used is
held to the reading's state of the merge before, bit for bit, and one that differs refuses the whole.

**The command.** `python reference_turns.py <declared run>` writes `<declared run>/turns`. Its method-step-3 declaration names, under
`reference_turns`: `reading`, `incoming` (`folder`, `registry`), `exports`, `qualification` (a list), `model`, `tensors`; bindings
are `{record, sha256}` from the repository's root and every source lies inside a declared input.

**Fixtures** (`test_reference_turns.py`, 5 tests): the turns are the export's formula a row and a merge at a time; the record is
the one `success_sets._turn_record` takes and `compile_reference` compiles from; a kept file with an answer the reading does not hold
between its answers gives the same bytes whatever that answer's rows hold; a kept file that is not the registry's bytes, a kept exit
that is not the state before times the scale, an export that is not the reading's computation, other steps in the kept index and a
folder that already exists are each refused; with no exports the answer steps are written too, and a question of no position is an
empty array.

**A trial on the real files** (1:17:38 to 1:18:44 AM CT, the processor, written to the scratchpad and removed: not the artifact):
65.4 s for the thirty answers. 3,150 of 3,150 answer arrays (30 answers by all 105 merges) are the exported bytes, 671,534 of 671,534
kept layer exits are the reading's state times the scale, 70 kept files are the registry's bytes, and the question turns are 3.2 GB.

**One seam it found.** `success_sets.compile_reference` takes each answer's group from `reading.answers[...]["correct"]`, and the real
reading's row map has no such key (`whole-states-original/reading/row-map.json`: id, steps, question, the two row starts, the two text
hashes; the group is in `reading.json`'s table, as `group`). The fixtures' row maps carry it, so the suite passes. Astra's to settle.

**The correspondence record takes a set reference as its target** (`states_correspondence.target_of`): the folder's own manifest
says whether it is a target file (`gap_targets.read`) or a set reference (`success_sets.Reference`); `record` reads the same two
things of either, the manifest's sha256 and each answer's id, steps and text. A reader protocol made from now on names the new bytes
of `states_correspondence.py`.

## A capture belongs to its declared run, and reads the fitting answers alone (November 15, 2:19 to 2:31 AM CT)

Astra found two seams in the capture of a state's own answers before the new loop's acquisition was frozen (her message of November 15,
received before 2:19 AM CT; `records/retrospectives/2026-11-15-first-pass-acquisition-astra.md`). Both were this seat's instrument, and both are closed in
source and on fixtures. No engine pass was made, and no exception to the checker.

**What was wrong.** `own_text.py` took a cohort and an output folder. Its one check was `tube.declared` on the COHORT's declaration,
which sets how that cohort's old passes read a text and holds its old allowance: it says nothing of the run a capture is made for.
So the new loop's capture command, given `controlled-bf16-1`, would have started the recorder without the research checker having
run on the loop's own declaration or on its own restored inputs. And every capture was bound and registered through the frozen axis
of the first overlap: `loop_requests.capture_sources` hashed the axis answer's turns, and `tube.registered` laid every generated
answer on them (`pairs-axis.npy`). That axis answer is `expanded-089-h3`, which is not one of the thirty fitting answers
(membership alone, from `trace-overlap-complete-registered/members.json` and `tube.picks`; no array opened for it). No reader of the
new loop reads that path: the correspondence record, the measurement and the requests all take `pairs-original.npy`.

**What it is now** (`capture_bound.py`, with `own_text.py`, `reader_protocol.py`, `states_reading.py`):

- **The command names the declared run first**: `[PASS=<n>] own_text.py <declared run> <cohort run> <original | model> <recordings |
  earlier capture> <out>`, as `states_reading.py` does. The form that named a cohort alone is refused with the usage line.
- **The checker runs before anything is made**: `check_native_run(out, <declared run>/declaration.json)` (the research verifier, and
  the run's own archived inputs back on this disk), then `capture_bound.admitted`, then the folder, then the recorder.
- **A pass is bound** (`PASS=<n>`): the output is `<run>/loop/<n>/capture/own` (`gap_run.of_pass`) and no other folder; the cohort is
  the declaration's `sweep.cohort`; the model is the state the pass starts from BY ITS SHA256 (`gap_run.begins`: the declared start
  for pass 1, the state the pass before swept to, by its `swept.json`, after it); the reference history is the one the
  declaration's `set_measurement.history_before` names; and the kept streams (`KEEP_RAW`) lie inside the run. Any other is refused
  before the recorder starts. The capture's record carries `run`, `declaration_sha256`, `declared_step`, `pass_number` and
  `starts_from`. A capture that is no pass (an evaluation's) is checked the same way and held inside its run.
- **The sources are of the answers read and of no other** (`capture_bound.original_sources`): per answer its prompt, its recorded
  text, its steps and the sha256 of the original's exported turns, each file hashed once and held to its export's own record.
  `reference_sha256` is one sha256 of those in the cohort's order; `registration` names the basis (`original-computation`), the
  rule and each answer's original turns. No axis, no list of the sixty, no pairs of the frozen registration, and the cohort's old
  bound reference (`tube.reference_sha256`) is not read. The reader protocol binds the same function's output, so `held` compares
  like with like; `loop_requests.py` left the protocol's instruments and `capture_bound.py` entered them.
- **The registration is on the original's recording of the same question only** (`capture_bound.laid_on_original`): the arithmetic
  `tube.registered` used for `pairs-original.npy` (`complete_overlap.distance_path` on `unit_steps`), made once and never again. A
  finished answer holds `answer.json` and `pairs-original.npy`; `pairs-axis.npy` is not made, bound or checked. When the capture
  closes each path is held to the scorer's own rule (`validate_pairs`), the model is hashed again, and the original's turns are
  held by size and time of change since their one hashing.
- **A pass's reading is of that pass's capture** (`states_reading.py`, `PASS=<n>`): the folder given must be
  `<run>/loop/<n>/capture` and its record must say this run and this pass.

**For the launcher** (Fable's `run.sh`), pass n: `PASS=<n> [KEEP_RAW=<run>/loop/<n>/streams] own_text.py <run> <cohort> <the state
pass n starts from> recordings <run>/loop/<n>/capture/own`, on card 0 under `RECORDER_UNDER`; then `PASS=<n> [ROLLOUTS=<run>/loop/
<n>/streams] states_reading.py <run> <cohort> <that state> <run>/loop/<n>/capture`; then `states_correspondence.py` as before.

**Fixtures** (`test_capture_bound.py`, 11 tests; `test_states_correspondence.py` and `test_evaluation.py` brought in line): a pass's
folder, cohort, start and history each refused when another; a later pass refused until the pass before is swept and then bound to
that state alone; the original a start only where declared; the sources equal for the same answers and different for another
selection, cohort or recorded text, with no other answer's folder present; turns that are not their export's bytes refused; an
answer whose computation is the original's laid one on one, a longer one by a whole path, and nothing but `pairs-original.npy`
written.

**What is left and whose.** `own-text.sh`, the stopped on-policy loop's wrapper, is removed: it called the form that is no longer
run, for a loop that read `pairs-axis.npy` (`loop-step.sh` still names it and leaves with that loop, Fable's). `whole_reading.py`'s
own replay of a given text, `members.py`, `whole_model_l.py`, `conditional_members.py` and `export_own.py` read `pairs-axis.npy` of
captures made before today; they are the closed runs' readers, outside the new loop, and cannot read a capture made from now on.
Every protocol made before this change names other bytes and another registration source: the protocol is made again at the freeze.

### The reader protocol frozen from this acquisition (November 15, 2:35 AM CT; files only, no pass)

Asked by Astra once the acquisition was committed (1f2bee12): `reader_protocol.py make controlled-bf16-1
studies/04-round-2-overlap/runs/whole-states-original/reading` wrote `reader-protocol/controlled-bf16-6.json`
(`whole-gap-reader-protocol/1`, made 2:35:35 AM CT; the file's sha256 is
`243a7ea7209e363aaddc3271b967d2f4bdd58bd399ac17bd9ee43d635a9872ff`). What it binds, each as the record says it:

- **The answers:** the thirty fitting answers of `controlled-bf16-1` (`tube.picks`) and no other; `reference_sha256`
  `3f8113f410db…` and the registration (basis `original-computation`, the rule, each answer's original turns by sha256) are
  `capture_bound.original_sources`' own. No axis and no member list of the sixty is in it.
- **The original control:** `whole-states-original/reading` (reading sha256 `c027fe345cf8…`, the original model `6a7371313624…`,
  19,751 rows, 105 merges, 30 answers), found by what it computed; no pass was made for it.
- **Its true versions:** the control was made by the readers of November 14 (`states_reading.py` `e493e1805b33…`,
  `whole_reading.py` `1045550c0829…`, `reading_identity.py` `650b18d0b396…`), and the record keeps those under
  `original_control.made_with` and names all three under `readers_that_differ_now`: it does not say today's readers made it. A
  candidate reading's `made_with` is held to the protocol's INSTRUMENTS (today's files), the control's to its own.
- **The original computation is the control's:** 2,100 of 2,100 turns made again from the control's stored states are the exported
  bytes (30 answers at the 70 merges that can be made again; the 35 first merges of a layer are held as the incoming state instead).
- **The instruments:** 19 files by sha256, each the file on this disk and the tree at 1f2bee12 when it was made
  (`capture_bound.py` in, `loop_requests.py` out); the execution it names is card 0's (the recorder `0b4ea036c2c3…`, the libraries
  `06ffecec44d0…`, driver 615.71.10, fusion disabled).

A change to any of the 19 files after this makes the scorer refuse it ("Reader implementation changed since its declared protocol"):
the protocol is then made again and the declaring seat told. The record establishes neither that the registration pairs rows of the
same reasoning nor that a score through it is admitted (`not_established_here`), and it admits no capture and no model run.

### The rows reader had the capture's seam: it belongs to its declared run too (November 15, 2:48 to 2:52 AM CT; source and fixtures)

`rows_reading.py` is the one pass that gives every matrix's rows on one state, which is what a correction that solves all seventy
matrices from one state's rows needs in place of seventy late passes ("A pass of the new loop as its launcher stands", speed review).
It had the seam Astra found in `own_text.py`: it took a cohort and an output folder, called `tube.declared` on the COHORT's
declaration and never the research checker on the run the rows were for, wrote wherever it was told, and held neither the model nor
the text to a pass. Found by this seat while pricing that pass; nobody had run it for the new loop.

**What it is now.** `[PASS=<n>] [TEXT=<capture folder>] [KIND=attention|feedforward] rows_reading.py <declared run> <cohort run>
<original | model> <out folder> [<first>-<last>]`:
- The declared run is named first, and `check_native_run` on its declaration runs before a folder is made or the recorder started.
  The output lies inside that run. Both older forms (a cohort first, with or without a range of layers) print the usage and stop.
- With `PASS=<n>` (`rows_reading.admitted`): the output is `<run>/loop/<n>/rows`; the cohort is the declaration's `sweep.cohort`;
  `TEXT` is that pass's own capture (`gap_run.of_pass`), closed, whose record says it is this run's pass n on the same model; and the
  model is the state the pass starts from by its sha256 (`gap_run.begins`). The model is hashed before the pass for this and again
  beside it.
- The text is the capture's bound text (`rows_reading.text_of`): each `answer.json` is held to the sha256 the capture bound when it
  closed (`own/text.json`'s `files`).
- `rows.json` says what the reading belongs to (`run`, `declaration_sha256`, `declared_step`, and for a pass `pass_number` and
  `starts_from`); the cohort it read stands under `cohort`, where `run` named it before.

`rows_reading.py` is not one of the frozen protocol's 19 files, so `controlled-bf16-6` stands. On fixtures: `test_rows_reading.py`,
10 tests (a pass's folder, capture and state; a later pass after the sweep before it; the original only where declared; another
folder, cohort or capture; a capture not closed or saying another run, pass or model; a reading that is no pass; pass numbering; a
text file changed since its capture closed). The suite is 949 tests and 115 subtests. The real script was run only in its two older
forms, which stop at the usage with no folder made; it has not read a capture's text on an engine (`TEXT=` was source only before
this too), and no pass is admitted by it. Where a pass's driver looks for these rows is Fable's.

**The same seam elsewhere among this seat's readers** (a scan of this folder's programs for the ones that start the recorder, 2:54
AM CT November 15). `own_text.py`, `states_reading.py` and now `rows_reading.py` call the checker on a declared run. Three do not:
- `whole_reading.py` as a script (its replay of a given text, line 132): it starts the recorder on a cohort and an output folder
  alone. It is one of the frozen protocol's 19 files because `states_reading.py` and `rows_reading.py` import its functions; its
  script form is no command of the new loop and cannot read a capture made since `capture_bound.py` (it reads `pairs-axis.npy`).
  Giving it the check changes a bound file, so it is done when the protocol is next made again, not before the first declared pass.
- `prompt_features.py` and `replay_cost.py`: development tools of closed work (experiment C's reading of the question's positions,
  and a timing of a given-text replay), not bound and not the loop's. Nothing runs them under the hold.

### A reading's stored states are every merge's turn (November 15, 4:01 AM CT; files only, no pass, nothing measured)

Why it was looked at. Marcos's clarification, relayed at 3:44 AM CT November 15 (`journal/2026-11-15-computation-not-state-values.md`):
the trace of computation is what each merge does to what it is given, the relation between a merge's input and its output, not the
values in the stream; the loop on state values was stopped at 3:46 AM CT and the loss on that relation is open design work (Astra's).
What this seat owes that design is the instrument's side: for a model a reading was made of, where is what each merge was given?

It is already in the reading, one file earlier. The engine's graph (`dependencies/llama.cpp/src/models/gemma4.cpp`: the three adds
at lines 260, 338 and 361, the layer's output scale at 366, `l_out` at 371) gives a layer's feed-forward merge its attention merge's
output, its third merge its feed-forward merge's output, and the next layer's attention merge the third merge's output times that
layer's output scale: one number a layer, which a direction does not see while it is positive. The model's first merge is given the
token's embedding times sqrt(1536), the token's row of the model file. So the turn, `unit(merged) - unit(incoming)` as
`export_stream_traces.export_case` makes it from the recorder's streams, is `unit(state[m]) - unit(state[m - 1])` for every merge
after the first, and `unit(state[0]) - unit(embedding[token])` at the first, at every row a reading has.

`turns_reading.py` is that (`turns_at(reading, merge, rows[, embedded])`, `embedding(model, tensors, tokens)`), and its `check`
holds it to the turns the recorder's own streams gave for the original (`wide-overlap/exported/<answer>/turns.npy`) against the
original's kept reading (`whole-states-original/reading`), at every position both have: all thirty fitting answers, 14,592 answer
rows, every one of the 105 merges (`turns-reading/check-original.json`; 26 s on four processor threads). The distance between the
two over the exported turn's size:

| Merges | Rows by merges | Largest | Median |
|---|---|---|---|
| a layer's attention merge after the first layer (34): given the layer before's exit | 496,128 | 8.3e-07 | 7.6e-08 |
| feed-forward (35) | 510,720 | 5.1e-08 | 2.5e-08 |
| third (35) | 510,720 | 3.5e-08 | 2.5e-08 |
| the model's first merge, given the embedding (each answer's second row on) | 14,562 | 4.1e-08 | 3.1e-08 |

Every layer's output scale is therefore positive in this model (a negative one would turn the direction over and read near 2, not
8e-07). The first merge at an answer's first row is not in the count: the token given there is the prompt's last, which an answer's
record does not keep. Question rows are not in this check, because the exports hold answer positions; the
question rows' turns of the original were held to the incoming states of `input-overlap/inputs` when the reference's were compiled
(`reference_turns.py`, 2:08 AM CT November 15).

What it changes:
- **A model's turns need no kept stream and no new recorder mode.** A capture's generation and its reading (205 s and 28.9 s in the
  stopped loop's pass 1) already hold every merge's turn at every row, question and answer. `own_text.py`'s `KEEP_TURNS` (about
  8.5 GB a pass) is not needed for it, and neither is the reading mode this seat was about to add for the attention merges' inputs.
- **The stopped loop's pass 1 has them.** Its reading (`success-sets-loop-from-mates/loop/1/reading`, here and in the bucket) is the
  matched sweep's model on its own answers at all 105 merges: that model's turns exist from files, with no pass.
- **A correction of what this seat wrote at the stop** (`STATE.md` as of 2083a8fa, and a note to claude-research): that a state's
  turns exist "only where two stored states give them (70 of 105 merges)". That was wrong. The attention merges' incoming is the
  stored state before them times a positive number; this seat's own `reference_turns.py` had counted exactly that at 2:08 AM CT
  (671,534 of 671,534 layer exits the reading's state times the scale, the same bits) and the sentence was written without it.

What it is not: no model is compared with a reference here, no loss is computed and no closure level is read. Which observable
states the relation (this turn, or one that also keeps how much a merge changes the size of what it is given), where the successes
agree, and what the first merge's dependence on the token means for it are Astra's and Marcos's to settle.

### The whole model in torch, with gradients: `torch_forward.py` (November 15, 4:21 AM CT; files and card 1's arithmetic, no engine pass)

Why. Marcos's statement of the loss (relayed 3:44 to 3:52 AM CT November 15): the shortfall of a run's trace from the correct
answers' overlap is taken back through the weights, the model is run again and the trace measured again. Taking anything back
through the weights needs the model's computation as one differentiable graph. The engine gives none and the tree held two pieces
of one (`fixed_input_movement.Block`, `field.Final`). Fable had read the same source and written nothing; by his reply that hour
this seat builds the forward and the update taken from it stays the solver's.

What it is. `torch_forward.Model` reads a model file's tensors at the original's offsets and computes, for one text held fixed, the
engine's own graph: `dependencies/llama.cpp/src/models/gemma4.cpp` whole; the attention of `llama-graph.cpp` (build_attn_mha, 3028;
the key-value form, 3534); which layers make keys (`llama-model.cpp`:2989: the first 15 make them, a later layer reads layer 13's if
it is windowed and layer 14's if not); the window (`llama-hparams.h`:421, 512 back); the card's rotary kernel
(`ggml-cuda/rope.cu`:188). The architecture is read from the file's own keys with the engine's reader
(`torch-forward/architecture.json`: 35 layers, width 1536, eight query heads over ONE head of keys and values, heads 512 wide on
the seven whole-text layers and 256 on the 28 windowed ones). The file's docstring holds the arithmetic line by line.

The seam the update is taken through (as Fable asked for it in that reply):
- `Model(model file, tensors index, architecture, device, leaves=..., cache=None)`: the leaves are float32 copies of the file's BF16
  values that take a derivative: the 35 `attn_output` and 35 `ffn_down` by default (651,165,696 weights), any tensors of the file
  on request (`leaves=lambda name: ...`: a q, k, v, gate, up or a norm's weight).
- `forward(tokens)` returns `incoming[m]` and `merged[m]`, each `[position, 1536]`, for the 105 merges, as one graph. Position `p`
  of an answer is the reading's question row `p` below the question's length and its answer row `p - question` from there on.
  `turns(pairs)` is `unit(merged) - unit(incoming)`. A function of the pairs is one number; `number.backward()` leaves its
  derivative in every leaf's `.grad`. Each layer is made again during the backward, so the card holds one layer's graph.
- `forward(tokens, given=...)` feeds each layer an input from elsewhere, which is how the check reads a block alone.

Held to the engine (`check`, `torch-forward/check-original.json`): the original's file against the original's kept reading
(`whole-states-original/reading`), all thirty fitting answers, 19,751 positions, on card 1 under its lock. Each distance is over
the engine's own size.

| What | Median | Largest |
|---|---|---|
| a feed-forward merge alone (what it adds to the ENGINE's own input) | 4.1e-06 | 4.8e-03 |
| a third merge alone | 1.3e-06 | 1.3e-02 |
| an attention merge alone | 3.5e-04 | 6.4e-03 |
| ... windowed layers, positions the engine made in a batch (the prompt) / one at a time (the answer) | 1.6e-03 / 2.4e-04 | 6.4e-03 / 2.9e-03 |
| ... whole-text layers, in a batch / one at a time | 1.9e-03 / 5.1e-04 | 5.0e-03 / 3.1e-03 |
| a state end to end (every layer fed by this forward's own) | 7.0e-04 | 1.46 |
| a turn end to end, attention / feed-forward / third merges | 1.3e-03 / 1.2e-03 / 3.8e-04 | 1.41 / 1.38 / 2.70 |
| the last merge end to end | 5.0e-04 | 4.9e-02 |

The cosine between this forward's turn and the engine's is 0.9999995 at the median over every merge and position.

Two things the table says, each with what was run to see it:
- **The attention's residual is the engine's own arithmetic, not this forward's.** The feed-forward and third merges alone stand at
  1e-06 of the engine's. The attention alone stands a hundred times further, and how far depends on HOW THE ENGINE MADE THE POSITION:
  seven times further on the prompt's positions (made in batches of 32) than on an answer's (made one at a time) at windowed layers,
  four times at whole-text layers. A wrong formula here could not know which positions the engine batched. The source says why:
  the cache holds keys and values in float16 (`llama-context.cpp`:4315) and the card's attention kernels compute on float16
  operands, a tensor-core kernel for batches and for the 512-wide heads, a vector kernel for one position at 256 wide
  (`ggml-cuda/fattn.cu`:1015-1042). So the engine's attention carries its own noise of 2e-04 to 2e-03 of what the merge adds, and
  the engine's prompt rows and answer rows are not made by the same arithmetic.
- **The model computes a few positions unstably, and there the forward and the engine part.** 11 of the 19,751 positions stand more
  than 0.1 from the engine at their worst merge (63 over 0.03, 509 over 0.01; 2 over 0.5), all of them answer rows, between positions
  235 and 485 of six answers, first passing 0.03 at merges 25 to 36 (layers 8 to 12) and back within 0.023 by the last merge
  (the record lists each). It is not the forward's error accumulating: the same forward run twice, the second time only with the
  cache's float16 rounding of keys and values (a change of about 1e-04), stands more than 0.1 from ITSELF at 15 positions (82 over
  0.03, 4 over 0.5). A change the size of the engine's own rounding moves those positions' middle states by tenths to more than
  their whole size. The two sets share 4 positions of the 11 and 15: each perturbation trips its own subset of a larger population.
  With the cache's rounding applied (`cache='float16'`, `torch-forward/check-original-cache-float16.json`) the forward stands
  FURTHER from the engine (20 positions over 0.1, 117 over 0.03, 1,065 over 0.01), because its rounding falls on slightly different numbers than the engine's and is one more independent
  perturbation; so the forward computes keys and values in float32 unless asked.

What it costs (card 1, RTX 5070 Ti, torch 2.15.0+cu132; `check-original.json`, `backward`): the weights load in 1 s from the page
cache; a forward of one answer is 0.18 s at the median (6.4 s for the thirty); a forward and a backward to all 70 matrices is 0.60 s
an answer at the median (0.55 to 1.03), and **the thirty answers' derivative, added up answer by answer over all 19,751 positions,
is 20.1 s**, the card holding 10.3 GB at most. The backward was of a stand-in number, the sum of every merge's squared turn: no loss
of anything.

The same on an edited model (`torch-forward/check-matched-own.json`, 4:27 AM CT): the matched sweep's model
(`whole-gap-mates-original/sweep/working.gguf`, the original's layout with other values, read only) against the ENGINE's reading of
that model on its own answers (the stopped loop's pass 1, 18,375 positions). Each merge alone stands where it stood on the original
(feed-forward 4.3e-06, third 1.3e-06, attention 3.5e-04); a turn end to end agrees at a cosine of 0.9999996 at the median and
0.984 at the least; the last merge stands at 3.8e-04 at the median and 9.3e-03 at most; the thirty answers' derivative is 18.8 s.
So the forward follows an edited model on its own text as it follows the original. One difference is recorded and not explained:
this model's answers hold far fewer of the unstable positions (1 over 0.1 from the engine against the original's 11, 10 over 0.03
against 63, 129 over 0.01 against 509; and the forward from itself under the cache's rounding, 1 over 0.1 against 15). Its texts are
other texts, so this does not say the edit made the computation steadier.

What it is not, and what was not done:
- No loss, target, update or write. Which function of the pairs is the loss is Astra's; the step taken from a derivative, and
  writing it to a model file in BF16, are Fable's.
- It is the computation of a given text. It writes no text and scores no token; the engine stays the recorder of a model's own
  answers and of the trace that is measured. At the 1e-03 level the engine's states are the engine's, with its kernels' noise.
- Two files were run: the original and the matched sweep's model. No derivative of either was used for anything.
- The 11 positions, and the larger population they are drawn from, are not excluded, weighted or marked by anything here. Their
  trace is decided by arithmetic at the 1e-04 level and not by the weights alone, on the original as on any candidate. What a loss
  on the trace does with such rows is the objective's (Astra's) to decide; the record gives their answers, positions and merges.
- The output norm and the scores of tokens are not part of the check (the forward computes `result_norm`, nothing reads it).

### The torch forward writes a model's answers: `write` (November 15, 4:49 AM CT; files and card 1's arithmetic, the original model, no engine pass)

Why. In a loop whose step is taken from a derivative, the engine's generation is what sets a step's time (speed review, "A step
taken by the gradient: what sets its time now"), and the forward computed a GIVEN text only. `Model.write(prompts, limit, ends)`
writes the text: the vocabulary's first choice at each position, as the recorder selects at temperature 0
(`recorder/observe_stream.cpp`:368-370), the answers side by side, one new position of each at a time, reading a cache of the
fifteen key-making layers' keys and values (the twenty later layers read layer 13's or 14's). `scores` is the head: the final
norm's product with the token table, which the file ties to the output (there is no `output.weight` among its 601 tensors),
soft-capped at 30 (gemma4.cpp:397-402). While it writes, every stored matrix is its float32 on the card in place of its stored
bits (`holding`) and is its stored bits again after.

Held to the recorder (`torch-forward/write-original.json`; `torch_forward.py write whole-states-original wide-overlap <out>`; the
thirty fitting answers, the text `whole-states-original/reading` was made on, the recorder's kept scores of that same text):
- **Under the recorded text held fixed**, at its 14,592 decisions: this forward's scores stand 0.0125 from the eight the recorder
  kept at each decision at the median (0.094 at the 99th percentile, 1.31 at most). Its first choice is the recorder's token at
  14,582 decisions and another token at 10, in 9 answers. At nine of those ten the engine's own first two choices are less than
  0.015 apart (0.0019 to 0.0145), at the tenth 0.079. Over all decisions the engine's first two are nearer than 0.03 at 42 and
  nearer than 0.1 at 168.
- **Written from the prompts**: the thirty answers in 27.3 s in the record (5:08 AM CT; 32.4 s in the run of 4:49 AM CT, which
  this record replaced, with the test suite running beside it), 36 ms for each token of the longest text, 14,643 tokens, 11.4 GB
  of card 1 at most. 21 answers are the recorded text token for token. Each of the other 9 leaves the recorded text at exactly the
  decision found under the fixed text (steps 11, 18, 26, 69, 69, 135, 294, 300, 689) and is another text from there (380 to 768
  tokens against 409 to 768).
- So the cache and the side-by-side writing compute what `forward` computes on a given text (the texts leave where the fixed-text
  scores said and nowhere else), and this forward is one more arithmetic beside the two cards', which word 9 of 30 answers
  differently from each other (line 42 of this page; whether these nine are those nine was not compared). A wording is decided
  by near ties a hundredth of a score wide.
- **A written text is laid on the original's rows by the capture's rule** (`registered(pairs, question, original turns)`, 5:08
  AM CT): the forward's turns of the text at its steps, each step's 105 turns as one unit vector, their product with the
  original's exported turns of the same question, and `complete_overlap.distance_path` over it, as
  `capture_bound.laid_on_original` does for a capture. On the thirty written answers: 11.9 s, the forward of each text included;
  the 21 texts written as recorded lie one on one on the original's steps, 21 of 21, and each of the 9 texts that left the
  recorded one gets the path the capture's own arithmetic gives on the processor from the same turns, 9 of 9. So an own-text
  trial has its text and its correspondence in about 40 s, and the forward it needs for a loss is the one the registration
  already made.
- The attention was changed to take texts side by side (`...` axes in its two sums). On three real answers the changed forward's
  states and derivatives are the committed forward's (79de25fd) bit for bit (a scratch comparison at 4:46 AM CT, not kept).
  Tests: 17 in `test_torch_forward.py` (six new: the scores, each written token is the forward's first choice on the text so
  far, side by side is alone, an ending token, the stored bits after `holding`, attention side by side).

What it is for, and what it is not:
- Astra chose, in a message that arrived before 4:50:07 AM CT November 15, that the engine's complete own trace is the measured
  loss at every step, and that torch supplies gradients and fixed-history proposal trials, never the reading of progress or
  closure. `write` changes none of that: a text it writes is no acquisition of the canonical protocol.
- What it adds is a trial of a proposed step on the model's OWN text in torch (about 30 s to write, then the forward) before an engine
  pass of five minutes is spent on that step. Whether an own-text trial may shape a proposal is the objective's to say (put to
  Astra with this record); the solver (Fable) has the call: `model.write(prompts, 768, [106])` on the weights the model holds.
- No answer is read and no token judged: the ending token (106) and the limit (768) are taken from the recorded texts and cases,
  and the comparison is of two acquisitions of one model. Not run on a candidate (`model=` takes one, as `check` does).
- Astra's answer (arrived before 4:57:19 AM CT November 15; `records/retrospectives/2026-11-15-relational-gradient-specification-astra.md`,
  "Connected gradient and measured progress"): a torch-written own-history trial may shape a proposal, optional, on the thirty
  fitting prompts with the same frozen reference over the complete remapped trace; its generation, correspondence and selection
  rule are bound beforehand in the step's declaration by the solver; a fall on it is neither native progress nor closure.

### The reader protocol made again: `controlled-bf16-7` (November 15, 5:04 AM CT; files only, no card, no pass)

Three of the frozen protocol's nineteen instruments changed after `controlled-bf16-6` was made, none in what it computes:
`own_text.py` (each batch is finished while the recorder writes the next), `capture_bound.py` (`one_behind`, which orders that
work in time) and `complete_overlap.py` (`distance_path` fills its table by anti-diagonals: the loop's path bit for bit). The
record of each is in the speed review ("A step taken by the gradient: what sets its time now"). `controlled-bf16-6` stays what
the stopped loop's declaration names. `reader_protocol.py make controlled-bf16-1
studies/04-round-2-overlap/runs/whole-states-original/reading` wrote `reader-protocol/controlled-bf16-7.json` (sha256
`87a2f2f45d5ec4ddad36377cece93bb0f5d553af9cabc234763f18fa570c7d6e`) at 5:04:37 AM CT from the files as they stand: the same
cohort, fitting answers, original control and execution, and the original's computation made again from the control's states is
the exported turns bit for bit at 2,100 of 2,100 (answer, merge) arrays, as before. A step declaration of the next loop binds
this file; a change to any of its nineteen instruments means it is made again and the declaring seat told. A file of this name
made at 5:00 AM CT, before `complete_overlap.py` changed, was never committed or named to anyone and was removed.

### What a record of pairs covers when an answer is shorter than the original's: `pairs_coverage.py` (November 15, 5:44 AM CT; files only)

Asked by the session Marcos talks to, after the first torch steps wrote answers that end at 141 to 145 steps: when a model's own
answer no longer reaches a part of the reference path, how is that part counted, and is the coverage recorded each step.

**A record of pairs always covers both whole traces.** The path is laid from the first step of both to the last step of both, and
`complete_overlap.validate_pairs` refuses a record that lost a step of either side. So an answer shorter than the original's still
has a pair for every original step: the original steps it has no step for are laid, several together, on ONE of its steps. A loss
summed on the original's rows through the record counts each of those rows in full, against a step that is not theirs. The
loss's own completeness check (an original row with no pair is refused) cannot see it, since no row is ever without a pair.

**What the capture records of it today:** per computed registration, the answer's steps, the original's steps and the most original
steps on one step (`own/text.json`, `answers[<id>].registration`), and the record of pairs itself (`pairs-original.npy`). What it
does not record is where along the original's path the model has a step of its own.

**`pairs_coverage.py <capture folder> <out.json>` reads that from the records of pairs alone** (no turns, no states, no text, no
token; 5 tests): per answer the original steps laid one on one (`own`) and the piles (`shared`: the answer's step, the first and
last original step laid on it); along the path, for every original step, how many answers reach it and how many hold it one on
one. It is not one of a protocol's instruments and makes nothing a measurement reads. A step laid one on one is no agreement;
the two words say how the record of pairs lays the reference on the model's own trace, and nothing of what the model computed.

On the one changed model the engine has read whole on its own answers (the matched sweep's model, the stopped loop's pass 1;
`pairs-coverage/success-sets-loop-from-mates-pass-1.json`): its thirty answers are 13,216 steps against the original's 14,592,
18 of them shorter than their original (11 wrong, 7 correct). Of the 14,592 original steps 11,279 have a step of its own and 3,313
lie in piles: 2,323 of the wrong answers' 7,916 and 990 of the correct answers' 6,676. Along the path the share with a step of its
own is 0.923 over original steps 0 to 99, 0.845 over 100 to 199, 0.781 over 200 to 359 (the stretch the relation reading found the
shortfall in) and 0.587 from 360 on. One pile holds 124 original steps: `expanded-001-h3` has the original's steps 233 to 356 on
its one step 207; `expanded-087-h3` has 184 to 302 on its step 189.

**What the two words mean and how the record is used, decided by Astra** (November 15, after this section's first numbers;
`journal/2026-11-15-follow-the-changing-trace.md`, "A shortened history and reference coverage"): the coverage record stands
beside every measured loss, bound to that loss's capture and model, with `controlled-bf16-7` and the loss unchanged. A loss keeps
every original row at its full mass with its actual pairwise shortfall: piled rows are not dropped, their shortfalls are not
averaged away and no penalty is put on an answer's length. `own` and `shared` are exclusive and shared correspondence in the
record of pairs, not proof that a reasoning operation occurred or was absent; the record's complete, monotonic coverage is forced
by the registration and certifies no closure, so the shared spans and what they contribute to the loss stay visible. The stopped
pass's 3,313 shared rows are evidence of how that capture was mapped, not a measurement of any later proposal.

**The record is tied to its capture since format `pairs-coverage/2`** (the reader as changed for that decision): it names the
capture's run, pass and model, carries the sha256 of the capture's `own/text.json`, and each record of pairs is held to the sha256
the capture bound when it closed (`text.json`, `files`), or the reading is refused; it carries the meaning above as its `meaning`.
The keys a reader takes (`own`, `shared`, `stretches`, `along_the_path`) and pass 1's numbers are as before. Beside a measured loss
of a declared run it is `pairs_coverage.py <run>/loop/<n>/capture <run>/loop/<n>/pairs-coverage.json` (the step's driver is
Fable's, who has the requirement from Astra); joining a shared span to its rows' shortfalls is done where the loss's rows are.

### The readers run the kept scheduler: `controlled-bf16-8`, and what a pass says of its own engine (November 15, 8:27 AM CT; files only, no card, no pass)

Astra accepted `recorder-kept-original` (the same bytes, 30 of 30 answers, in 0.756 of the installed time under load) and adopted
the kept scheduler for the readers through a protocol made again, on one condition: a record says which engine a pass RAN, the
installed engine apart from the effective one, with the search order and both places
(`records/retrospectives/2026-11-15-recorder-kept-admission-astra.md`). That is `recorder_engine.py` (edf7934e, 0497753a), and
`own_text.py`, `states_reading.py` and `whole_reading.py` start every recorder pass through it: the kept folder ahead of the
caller's library path, `GGML_SCHED_EVAL_KEPT=1`, fusion off, built as the qualification's kept side was. `rows_reading.py` and the
late passes are not changed: they give the recorder other arguments, which no comparison qualified.

**The identity comes from the loader, and two libraries are other files, not one.** The recorder is asked which file it takes for
each library it needs, by the loader's own trace (`LD_TRACE_LOADED_OBJECTS=1`: the program does not run), once as a reader set its
environment until now and once as a pass is started. Bound are the libraries taken from either engine folder and every library the
two traces take from different files:

| library | file a pass takes | sha256 | |
|---|---|---|---|
| `libggml-base.so.0` | `dependencies/engine-f089c9c-sm120-kept/libggml-base.so.0.21.0` | `de163ac07215…` | the overlay's, the qualified bytes (installed: `1dbfd54200c1…`) |
| `libgomp.so.1` | `/usr/lib/x86_64-linux-gnu/libgomp.so.1.0.0` (libgomp1 15.2.0-4ubuntu4) | `fa075918dc2e…` | the system's (installed folder's copy: `d46f9225c188…`) |
| `libllama.so.0`, `libggml.so.0`, `libggml-cpu.so.0`, `libggml-cuda.so.0` | `dependencies/engine-f089c9c-sm120/…` | in the protocol | the installed folder's, as before |

The second line is a correction of what I wrote of the qualification. Its pages say one library is another file
(speed review, "A split computed as one graph…", `recorder_kept.py`'s own text, the run's declaration). The kept library was built
on this machine without the `$ORIGIN` runpath the installed libraries carry, so its own need of `libgomp.so.1` is met by the
system's library, not by the installed folder's copy, which is other bytes. The qualification's kept side was started in the
environment built here, so by the loader's rule it ran that way too; that is inferred, because the comparison read only the
scheduler library's map. Its result stands for what it ran: the kept scheduler with the system's libgomp gave the installed
bytes. The same holds for the deferred library, built the same way and not adopted. What follows from binding it: a system update
of libgomp changes the engine's identity, and a protocol made before it refuses the capture. Putting the installed folder second
in the pass's library path would leave one library apart, but that is a combination no run compared: Astra's to decide.

**A pass says for itself what it ran** (`Engine.watch`, `Engine.ended`). While a pass runs its own map is read (the started
process, or one under it when a pass is started under a lock's holder). When it has ended it is refused unless every bound library
was mapped from the bound file (the scheduler's library by its inode too), no bound file changed since it was hashed, the
scheduler's own line and counts are in the pass's log, and every split was marked and computed as one range, which is what was
qualified. A capture's `text.json` and a reading's `reading.json` carry `libraries_sha256` (the engine's identity,
`8c8b96aaa287…`; the installed folder's own, `06ffecec44d0…`, is what the original control carries and stays in
`engine.installed`) and `engine`: the identity whole and each pass's account. A reading made from a generation's kept streams
carries the generation's accounts and is held to them.

**The protocol, form 2** (`reader-protocol/controlled-bf16-8.json`, sha256 `064c773f62351feac20c3a890f113e9efc21eea1464c79e70b2f3f0aeed56ecc`,
made 8:27:03 AM CT from the instruments at 0497753a; 2,100 of 2,100 turn arrays the exported bytes, as before): `execution` is the
original control's, untouched; `capture_execution` is what a changed state's capture and reading must be of (the control's
recorder, card, driver, card number, fusion switch and arguments on the engine above); `the_two_executions` says what is the same,
what differs and what the difference rests on (the qualification's declaration, result and comparison by sha256). `held` holds a
capture and reading to the capture execution and to their passes' accounts; a form-1 protocol is held as it was.
`controlled-bf16-7` and every earlier file are as they were and hold no capture made by the instruments as they now stand.

**What a consumer meets.** A reading under `-8` does not carry the control's `libraries_sha256`. A scorer that requires the
candidate's execution to equal the control's refuses it until its owner takes the protocol's pair
(`whole_context_compare.py`:88, `whole_context_conditional.py`:143, `evaluation_native.py`:60: Astra's, of stopped
constructions). `states_conditional_check.py` names that one difference and compares the rows across it. A rented box has no kept
library: a capture there refuses at its start until one is built for the box.

**Not run.** No pass was made for this. The checks are 19 fixtures of `recorder_engine.py` on a stand-in process that maps files
and writes a scheduler's lines (`test_recorder_engine.py`), and 116 of the readers' and scorers' fixtures (38 subtests) with the
new cases for form 2. The first pass through it is the first declared capture; a refusal there is this instrument's to answer.
As arithmetic on the measured ratio a capture's generation goes from 204 s to about 154 s; no capture has measured it.

**Asked before a capture, not found out after it: `engine_check.py`** (8:42 AM CT November 15; files and the loader's trace of
the recorder, no pass, no card). A capture's engine is decided by the loader in the environment the capture is started in, and the
readers refuse a capture of another engine only once it has closed, about four minutes of a card later.
`python3 -I engine_check.py reader-protocol/controlled-bf16-8.json`, run from the shell that will start the capture, exits 0 when
the engine `recorder_engine.py` finds there is the protocol's (it is, in this seat's session: `8c8b96aaa287…`), 1 with each
library that is another file and each folder of that environment's `LD_LIBRARY_PATH` that holds a bound library's name, and 2 for
a protocol that binds no capture engine (`controlled-bf16-7` and earlier). It is no instrument of the protocol and changes none.
**A limit of `recorder_engine.py` met while trying it:** the instrument binds the installed folder's libraries and whatever the
overlay makes resolve to another file than the caller's environment resolves by itself. With the deferred scheduler's folder ahead
in the caller's `LD_LIBRARY_PATH`, the pass would load the same six files as the protocol's engine (the kept folder is still put
first), but the record leaves `libgomp.so.1` out of the bound set and names the deferred folder's scheduler library as the installed
one; its identity is then `1b811f208095…`, not the protocol's, and the capture is refused. The refusal is on the safe side and no
launcher of ours puts an engine folder in that variable; the instrument is bound by `controlled-bf16-8` and is left as it is, and
the preflight says which folder is in the way.

### What a step overlaps: the answer written again with its own original, not with the correct answers (November 15, 8:54 AM CT; read from the programs, nothing run)

Marcos asked whether every step overlaps, and that what is wrong be written where the next reader finds it before more steps are
taken (relayed by claude-research at 8:48 and 8:50 AM CT November 15). He pointed at the words entering the measurement. What
this seat's instruments do there, all of it, read from the programs:

- **The torch loop.** `relation_step.py`:725 has the model write the thirty answers and :736 lays each on
  `exported/<the same answer>/turns.npy`, the ORIGINAL's recording of that answer, through `torch_forward.registered`
  (`torch_forward.py`:313): every step's turns at all 105 merges as one unit vector, the product of the two answers' steps, and
  `complete_overlap.distance_path` over it, a path that keeps every step of both sides.
- **The native capture** under `controlled-bf16-8` does the same on the engine's turns (`capture_bound.laid_on_original`,
  `capture_bound.py`:122), and the correspondence a scorer takes is on that basis (`states_correspondence.py`,
  `original-computation`).
- **The correct answers are never overlapped with at a step.** They are reached second-hand, as constants: "for positions laid
  on the original's rows, the mate's turn and the tolerance at every merge" (`relation_step.py`:208). Which rows of a success
  correspond to an original row was laid once, on the original's texts (`relation_mates.py`), and the tolerance was made once
  for those rows (`relation-envelope-original`).

**So a step overlaps the model with its own past, and reads the correct answers through an index made for the old wording.**
Three ways the words enter by it: (1) a pair joins a position of the new text with a row of the old one that may have read
another word, and the reference at that row (the mate's rows, the tolerance) was made for the word and the history the original
had there; (2) the path must keep every step of both sides, so an answer that is longer or shorter than its original piles
positions on one row or rows on one position (`pairs_coverage.py`, the section of 5:44 AM CT above: full mass is not support);
(3) a wrong answer is laid on the wrong trace, the one the correction is meant to take it away from, and the nearer the
correction comes to working the less that path means.

**What it reads, measured by Fable** (`records/development/2026-11-15-relation-reading/same-history.json`, 8:18 AM CT): on
answers written again at the plain loop's steps 2, 3, 8 and 13, under the ORIGINAL's own weights, the correct answers read 0.045,
0.096, 0.050 and 0.039 where they read 0.0006 on the original's texts, and a wrong answer reads above a right one in 0.54, 0.60,
0.34 and 0.48 of pairs where it is 0.86 on the original's texts. Only the words moved between those readings.

**Not known.** Whether laying each rewritten answer directly on the correct answers' traces at every step, and reading a
departure only where the correct answers agree with one another, keeps a right answer near nothing and a wrong one above it
whatever the wording. Nothing has measured it. No instrument makes that overlap at a step: the direct overlap exists only for
the original's own texts (the complete overlap of the method's first steps, and `relation_mates.py`'s memberships). Whether the
form of the trace itself (the turn carries the word a row reads) lets words in beyond the correspondence is not separated from
it by any reading so far.

**What this seat does about it.** The overlap of a model's own answers with the correct answers at every step is this seat's
instrument and is built before anything else of the speed work: for any thirty answers as a model now writes them, each one's
path onto each correct answer's trace by the computation alone, and the places where the correct answers agree with one another.
The measure read on it is Astra's and the step taken from it Fable's.

### The answers written again, overlapped with the correct answers directly: `overlap_step.py` (November 15, 9:10 AM CT; files and card 1's arithmetic, the original's weights only, no engine pass, no loss)

The instrument the section above says did not exist. `overlap_step.py` lays every answer, as a model writes it, directly on each
of the fifteen original correct answers by the rule the reference itself was compiled with (`success_sets.compile_reference`:
`complete_overlap.distance_path` over the product of the two answers' unit steps, the answer's steps and the question's positions
each their own domain), and keeps, before anything is pooled, the pairs, and at every step and merge the cosine of the answer's
turn with each correct answer's turn at its rows laid there, the sizes and the log gains (`<answer>.npz`: `pairs.<correct answer>`,
`alike`, `size_by`, `gain_by`, each by correct answer; the correct answer of the same name is laid and kept like the others and
left out of everything pooled). Each correct answer stays its own frame; nothing is laid on a common axis (Astra's shape, 9:02
AM CT). It holds no tolerance, mate, threshold, selection of rows or loss. Eight fixtures on made-up turns (`test_overlap_step.py`).

**On the original's own texts it is the reference's overlap.** `overlap_step.py original` on the engine's exported turns made
870 paths (each of the thirty on every other correct answer, both domains) and all 870 are the reference's own
(`success-set-reference-original/reference/paths.json`), pair for pair (`overlap-step/original/overlap.json`,
`paths_of_the_reference_come_again`). 48 s for the thirty on card 1.

**One plain reading of its tables, kept so two overlaps can be held beside each other, and not the objective:** `departure`, the
mean over an answer's cells (every row of the question and the answer, 105 merges) of `shared * (1 - toward)`: `shared` is the
length of the mean of the correct answers' unit turns at the rows laid on a step (1 where they do one thing, toward 0 where they
do different things: nothing is asked there), `toward` the cosine of the answer's turn with what they share. A correct answer is
read by the other fourteen. The question's rows read 0.0014 and 0.0015 for the two groups (the fifteen agree at 0.9986 there); what tells the
groups apart is in the answer's rows.

**What it reads, under the ORIGINAL's weights throughout** (`overlap_step.py torch kept=<step>.npz`: the torch forward of the
original's file on the texts the plain torch loop kept at a step; `overlap-step/summary.json`, each set's `overlap.json`):

| texts | originally wrong | originally correct | a wrong one reads above a right one | mean steps, wrong / correct |
|---|---|---|---|---|
| the original's own, the engine's recording | 0.0710 | 0.0467 | 0.711 of pairs | 528 / 445 |
| the original's own as torch writes them (step 1's texts; 9 of 30 worded otherwise) | 0.0695 | 0.0467 | 0.733 | 530 / 446 |
| written after the loop's first step (step 2's texts) | 0.0487 | 0.0503 | 0.489 | 440 / 448 |
| after its second (step 3's) | 0.0963 | 0.1024 | 0.422 | 637 / 642 |
| step 8's | 0.0559 | 0.0710 | 0.240 | 469 / 510 |
| step 13's | 0.0553 | 0.0647 | 0.258 | 462 / 482 |

- **Two wordings of the unchanged model read the same.** The correct answers read 0.0467 on the engine's wording and 0.0467 on
  torch's; the wrong ones 0.0710 and 0.0695. The direct overlap does not read the nine reworded answers as departures.
- **The correct answers did not rise seventy-fold after the first step.** The loop's own loss read them at 0.0006 on the original's
  texts and 0.043 on step 2's (0.045 under the original's weights, Fable's same-history record). Laid directly on the correct
  answers they read 0.0467 and 0.0503. That rise was the second-hand correspondence and its tolerance, not the computation.
- **On the texts written after the FIRST step, ten of the fifteen originally wrong answers stand where a correct answer stands.**
  0.1285 to 0.0694, 0.0984 to 0.0361, 0.0928 to 0.0379, 0.0851 to 0.0387, 0.0780 to 0.0388, 0.0685 to 0.0406, 0.0603 to 0.0381,
  0.0527 to 0.0400, 0.0524 to 0.0342, 0.0437 to 0.0275, and their lengths fall to the correct answers' (740 to 453 steps, 623 to
  388, 621 to 419). One stays far (`expanded-001-h3`, 0.156 to 0.163), two stay where they were and two rise a little (0.035 to
  0.046, 0.033 to 0.043). Of the fifteen originally correct, eight stay or fall (`discovery-00-channel-h3`, the correct answer
  that read like a wrong one, 0.1015 to 0.0794) and seven rise, three of them far and with a much longer text:
  `discovery-15-channel-h1` 0.051 to 0.099 (461 to 592 steps), `discovery-06-channel-h1` 0.046 to 0.078 (409 to 547) and
  `discovery-01-channel-h1` 0.020 to 0.062 (388 to 556); the other four rise by 0.003 to 0.015.
- **After the second step both groups stand at twice the original correct answers' distance, with texts half again as long**
  (0.096 and 0.102; 637 and 642 steps), and at steps 8 and 13 the originally correct answers read above the originally wrong ones.
- **The amounts say the same** (the mean distance of an answer's log gain from the correct answers' at the rows laid on it, the
  answer's rows): 0.0110 and 0.0088 on the original's texts, 0.0087 and 0.0091 on step 2's, 0.0136 and 0.0139 on step 3's.
- **The plain reading does not locate a stretch: it reads the route.** By band of five layers the originally wrong answers stand
  at 1.41 to 1.61 times the correct ones at every depth on the original's texts, and at 0.95 to 0.98 at every depth on step 2's
  (`overlap_depth.py`, `overlap-step/depth-profile.json`). A group difference that is the same share at every depth is not a
  place where a computation errs; it is how much of an answer lies on the route the correct answers take. Under one set of
  weights a text's path is mostly the text's, so this reading says the model after its first step WROTE, for ten wrong questions,
  answers that take the correct answers' route. Where on a route a computation departs is what a definition of local shared
  agreement has to read from the kept tables, and this reading does not.
- **The share of pairs told apart by the ORIGINAL's labels is not a check of a measure once the texts change.** At step 2 it is
  0.49 because most of the answers labelled wrong no longer stand apart, which is what a closing gap reads as; a label belongs to
  the text it was given for.

**What this does and does not establish.** Every number here is the ORIGINAL's weights on a kept text, in torch on card 1: it
says where the path of a text lies when the unchanged model is held to it, and the stepped model is the one that wrote the text.
The stepped weights were not kept, so their own trace on these texts is not read, and nothing here is the engine's trace. It is
one plain reading of the tables and no objective; no answer was read as text and no outcome is claimed for any answer. What it
does establish: (1) the loop's first step, which was taken from the original's own texts where no wording had moved, was followed
by texts on which most of the wrong answers' paths lie with the correct ones and three correct answers' lie much further off; (2) the
loop's own loss could not see either, read all thirty as far, and the second step, taken on that reading, is where both groups
left; (3) a measure laid directly on the correct answers at every step reads the same on two wordings of one model.

- **The model's own answers in one command, and a trap met making it (9:21 to 9:24 AM CT November 15; `244f875f`).** With no
  `kept`, the torch mode has the model write its thirty answers first (`torch_forward.Model.write`, the recorder's limit and
  ending token from `torch-forward/write-original.json`), keeps them as `<out>/texts.npz` and overlaps those (its `beside=`
  count of texts written as a kept set left that afternoon: the page's last section). A `kept=` reading computes the tables it computed before.
  Fable's `state_texts.py` (`3384abf3`) does the writing as its own step, and the readings of the first step's state are Fable's
  run's (`plain-first-step-state`, revision 1), not a run of this seat: the separate run begun here for them was withdrawn
  before it had a folder. **The trap:** Fable's revision hashed `overlap_step.py` while this edit was in the working tree and
  not committed, so it bound bytes no commit held (`8b2dfa8c…`). Not knowing that, this seat put the committed file back at
  9:23:19 AM CT (`92bc2a8a…`) and the bound bytes at 9:23:36; the run's `own` reading started at 9:23:31, on the committed
  file's code. The two compute the same tables for a `kept=` reading, but the record did not hold the binding; Fable was told at
  once and the bound bytes are committed as they stood. A program another seat may bind is edited and committed in one act.

**What follows for the step, for its owners** (the measure is Astra's, the step and its state Fable's): the state after the first
step is the one worth reading whole: made again from the original (the step is one derivative and one length, in the loop's
record), written as a model, and read on its own answers by its own weights through this overlap, natively when admitted. The
kept arrays of every set are in `overlap-step/<set>/` (156 MB a set, not committed; made again in a minute from the model, the
texts and this program, each named by sha256 in the set's `overlap.json`).

### One question in two wordings: the reading follows the token a step is given, and hardly the question (November 15, 9:34 AM CT; files only)

Marcos's question, relayed 8:48 AM CT: on the unchanged model, does a correct answer read the same whatever its wording, and does
a wrong one read above a right one on every wording? `records/KNOWN.md` held it as open: no form of the relation had been read on
one question in two wordings. `overlap_wordings.py` reads it from the tables `overlap_step.py` kept, with no model and no card
(`overlap-step/wordings-original-weights-on-plain-step-1.json`, `wordings-original.json`; five fixtures). Every set here is under
the ORIGINAL's weights: the thirty questions on the original's own texts as torch writes them, and on the texts the plain loop
kept after its first, second, seventh and twelfth steps, which under the original's weights are conditional traces on other histories of the same
questions, not certified equivalent correct computations (all thirty texts differ from the first step on; they part from the original's at places 3 to 76, 23 of them at 11 or
68 to 76); and the engine's recording beside torch's writing (9 of 30 worded otherwise).

**How two wordings are held beside each other.** Both wordings of a question are laid on a correct answer S (never the
question's own) by `overlap_step.py`'s paths, so a step of S has steps of each wording laid on it. At that step and every merge
each wording has its cosine with S's turn (the direction) and its gain and turn size (the amounts). A step of S is `before` (all
laid steps come before the texts part: the same history), `same_given` (after that, and the laid steps were given the same token
in both wordings) or `other_given`. Beside it, in one set, two DIFFERENT questions at the same step of S, by the same two kinds.

| at a step of a correct answer, mean over the 105 merges | given the same token | given another token |
|---|---|---|
| ONE question in two wordings: how far the wording alone moves the direction | 0.030 (correct), 0.029 (wrong) | 0.196, 0.196 |
| two different correct questions | 0.029 | 0.184 |
| a wrong and a correct question | 0.034 | 0.191 |
| two wrong questions | 0.032 | 0.194 |
| ONE question in two wordings: how far the wording moves the gain | 0.0065, 0.0066 | 0.0234, 0.0235 |
| two different correct questions, the gain | 0.0061 | 0.0216 |
| a wrong and a correct question, the gain | 0.0068 | 0.0220 |

(First wording: the original's own texts in torch; second: the texts after the loop's first step. Before the texts part the
wording moves the direction by 0.0004 to 0.0008 and the gain by nothing. Against the texts after steps 2, 7 and 12 the same two
columns read 0.027 to 0.040 and 0.195 to 0.216. On the engine's recording the between-question rows read 0.029, 0.035, 0.030 and
0.184, 0.191, 0.192; the unchanged model's own two wordings, engine against torch, read 0.012 to 0.033 and 0.176 to 0.198 on the 9
questions worded otherwise, and 0.00005 on the 21 worded the same, which is the two arithmetics.)

What it says, in order of size:

- **The reading is strongly associated with the mapped input-token stratum.** In `other_given`, a question's readout differs by 0.196 across two
  histories, comparable to the between-question readouts (0.184 to 0.194). Given the same token it stands
  0.030 from itself, again as two different questions do (0.029 to 0.034). The factor between the columns is six for the
  direction and three and a half for the gain. It holds for every one of the thirty questions by itself: the wording moves a
  question by 0.016 to 0.055 where the token is the same and by 0.155 to 0.247 where it is not, and the two ranges do not meet.
- **These local direction and gain readouts are not invariant across the supplied histories.** Their mapped scalar changes
  are comparable to the between-question values. Original weights are fixed, but correctness and computational equivalence of
  the alternative histories are not established. This does not define an acceptance failure at every cell or rule out every
  useful amount-based construction (the reported gain movement is 1.2 times its reference level in `other_given`, and the
  direction movement 0.8 times).
- **The original-label group contrast is smaller than the contrast between the reported token strata.** In `same_given`, a historically wrong
  and a correct question stand 0.0335 apart where two correct questions stand 0.0292: 1.15 times, in every band of five layers
  (1.13, 1.13, 1.11, 1.21, 1.18, 1.16, 1.17 from the first band to the last; the gain 1.10 to 1.13). With another token given the
  same ratio is 1.02 to 1.08. In the mean that excess is 0.0043 beside a wording term of 0.166 (0.196 less 0.030), 39 times its size.
- **The token-stratified contrast is a substantial confound for this descriptor.** The pooled difference between strata is
  much larger than the original-label group contrast. That does not identify an additive 0.16 nuisance term or 0.004 capability
  error at each cell, nor prove what every whole-path loss must do. Direct correspondence removes the old-row detour but not
  this measured history/context sensitivity; original-label separation on changed histories remains descriptive, not acceptance.

**Limits.** Descriptive tables under the original's weights; no measure, tolerance or loss is proposed, and the definition of
what the correct answers share is Astra's. The labels are the original's on its own texts, not a judgement of another wording.
`same_given` is read at steps the path laid together, and the path is itself found by the turn's direction, so the direction
still chooses which steps are compared (Astra's journal page of November 15 says this of the correspondence). The means are over
all steps and merges: where along an answer the 1.15 sits is not read here. All but the engine's set are torch's arithmetic.

**What follows, for its owners.** The definition asks for shared computation, not a same-token mask. These tables do not
establish that the reference level of one exists wherever input tokens match, or that the pooled 0.029 is a local tolerance.
`same_given` tests equal sets of given tokens among mapped positions, not necessarily one token or identical histories; the
directional quantity is a difference of mapped cosine readouts, not a full transformation distance. Holding original weights
fixed does not certify that the supplied histories compute the same correct thing. The excess of 0.004 remains a descriptive
group contrast, not an isolated capability-error component. Astra's assessment is
`records/retrospectives/2026-11-15-paired-wording-assessment-astra.md`; it admits no token mask, threshold or correction. Fable's
`wording_split.py` separately reads the split among correct answers (3.4 times by direction, 2.0 to 2.3 by amounts) and reports
larger shortfall for originally wrong members in 0.76 of pairs with the word held; where that contrast lies remains to be read.

### The first step's state: the pooled contrast changes mainly with the supplied history (November 15, 9:40 AM CT; files only)

Fable made the closed plain loop's first step again and kept the state as a model file (`runs/plain-first-step-state`, a
reconstruction; model `9d666de8…`), and read it through `overlap_step.py` under its own declaration: the state writes, token for
token, the thirty texts the loop kept after that step, and four sets of tables were kept, the state's and the original's weights
each on the state's texts and, at this seat's asking, the state's weights on the original's own texts. This section reads those
kept tables with `overlap_wordings.py`, files only: no model, no card, nothing of the state run again
(`overlap-step/first-step-*.json`). What is read is the section above's `same_given` group contrast:
how far a historically wrong and a historically correct question's readouts stand apart beside two historically correct ones.
This token-set stratum retains history/context variation; it is not a wording-free capability quantity.

| weights | texts | two correct questions | a wrong and a correct | ratio | wrong questions outside the correct questions' range |
|---|---|---|---|---|---|
| the original's | the original's own | 0.0292 | 0.0335 | 1.147 | 3 of 15 |
| the state's | the original's own (the wording held) | 0.0282 | 0.0323 | 1.143 | 3 of 15 |
| the original's | the state's own | 0.0250 | 0.0250 | 1.001 | 1 of 15 |
| the state's | the state's own | 0.0242 | 0.0243 | 1.003 | 1 of 15 |

(Same token given, the cosine with a correct answer's turn, mean over the merges; the gain reads 1.113, 1.114, 0.995, 0.997 in
the same order. By band of five layers the first two rows read 1.11 to 1.21 and the last two 0.99 to 1.02. The last column is
each question's own distance from the correct questions, a correct question from the other fourteen:
`each_question_from_the_correct_questions_same_token_given`.)

- **With the history held, the weights move the readout in both groups while leaving its pooled group ratio nearly unchanged.** On the original's own texts the state's weights move a question's reading by 0.0051 (wrong) and
  0.0047 (correct) and bring both groups 0.001 to 0.002 nearer the correct answers; the ratio stays 1.143 against 1.147 and the
  same three wrong questions stand outside the correct ones' range. Fable's bound reading says the same by the plain reading
  (0.0682 and 0.0459 against the original's 0.0695 and 0.0467).
- **The new generated histories reproduce the lower pooled group contrast under the ORIGINAL's weights too.** All thirty texts part from the original's within the first 76 places. On them a wrong and a correct question stand as
  two correct ones do (1.001 under the original's weights, 1.003 under the state's), at every depth, by the gain as by the
  direction. Fourteen of the fifteen originally wrong questions stand inside the correct questions' range; their texts are 373
  to 453 tokens where the original's were 528 at the mean. The unchanged weights reproduce the lower
  scalar contrast when supplied these histories. That supports history mediation of this reading, not equality of the full
  computation or proof of correct answers. Changing which history is generated can itself be a mechanism of a persistent
  weight edit; the observation neither establishes nor rules out capability repair.
- **It is not shown that the new texts are correct answers.** Every question, the correct ones included, stands nearer the
  correct questions on the new texts (two correct questions 0.0250 where they stood 0.0292), which a common form of answer would
  also produce. One question, `expanded-001-h3`, runs to 736 tokens and stands further than before (0.064 from 0.043); two
  correct questions stand further (`discovery-01-channel-h1` 0.0292 from 0.0241, `discovery-15-channel-h1` 0.0283 from
  0.0278). No text is read here, and Astra's assessment holds: this selects no state and establishes no repair.
- **The same thing a third time.** a2-path's plus rode on its wording, with the original's own weights favouring the plus on
  a2-path's text (`runs/sign-route-a2-path/result.json`); a fed word took the wrong answers onto the correct pattern from the fed
  row on (`fed_path.py`, "The fed runs' path, read on the trace alone"); and here one small step changes early choices and the
  original's weights carry the rest. These three records show downstream readings that change when the supplied history changes under original
  weights. Their observables and scope differ; they do not establish that all downstream computational differences are only
  wording or that the edited weights have no further effect. A whole-path reference must account for this history dependence
  without treating every new expression as error or a smaller pooled ratio as closure.

**Limits.** Torch, not the engine; a reconstruction of the state; means over all steps and merges; the path that lays steps
together is found by the turn's direction; the labels are the original's on its own texts. Where the state's texts first part
from the original's and how near a tie the original stood there is not read yet.

### Where the first step's texts part from the original's: nine pairs of tokens at near ties, most shared by wrong and correct questions (November 15, 9:44 AM CT; the original's weights, torch on card 1, 11 s)

`parting_places.py` (`c7e532c6`; one fixture) finds, for two sets of texts of the same thirty questions, the first place each
question's texts differ, runs a model on the prompt and the tokens before it, which both texts share, and keeps the scores of
the two tokens at that decision: token ids and scores, as the recorder keeps them, nothing read for what it says. Here the
original's own texts in torch against the texts the first step's state writes, under the ORIGINAL's weights
(`overlap-step/parting-original-to-first-step.json`). The state's half, the same decisions under the state's weights, is
Fable's to run inside `plain-first-step-state`, which binds this program.

- **Every text parts within the first 76 places** (3, 3, 11 seven times, 15, 17, 29, 29, 68 six times, 69, 69, 71 four times,
  72, 72, 73, 76, 76), of answers 373 to 768 tokens long.
- **The original stood near a tie at most of them.** The token the state writes is the original's second choice at 24 of the
  30, its third at 5 and its fourth at 1. The original chose its own token by 0.33 logits at the median (0.36 for the correct
  questions, 0.31 for the wrong ones): under 0.1 at 7, under 0.5 at 19, under 1 at 24. The five at 1.6 to 1.9 are one pair of
  tokens at place 68.
- **The thirty partings are nine pairs of tokens, and 28 of the thirty are at seven pairs that several questions share.** One
  pair at place 11 or 17 for eight questions (2 wrong, 6 correct), one at 71 or 72 for six (2 and 4), one at 68 for five (3
  and 2), one at 68 or 69 for three (2 and 1), and three pairs for two questions each; six of the seven shared pairs are shared
  between wrong and correct questions.

What that says and does not. The step moved a small number of choices in the opening of an answer, the same choices for many
questions whether the original answered them wrong or correctly, at places where the original was near a tie. That fits every
question standing nearer the correct ones on the new texts, the correct ones included (the section above), and it does not say
whether the new opening is the form a correct answer takes or only another form: no token is read for what it is.

The other half, how far the state's weights moved those thirty decisions, is Fable's bound reading with this program
(`studies/04-round-2-overlap/runs/plain-first-step-state/result.json`, `where_the_texts_part`; its record
`parting/under-the-state-s-weights.json`): under the state's weights its own token is the first choice at 30 of 30, over the
original's by 0.417 at the median (0.015 to 1.319), and the step moved the difference between the two tokens by 1.22 at the
median (0.07 to 2.20), the wrong questions' median 1.19 and the correct ones' 1.23.

### The overlap in half the time (10:21 AM CT November 15; `overlap_step.py`, `overlap-step/speed.json`)

Why now. A call of `overlap_step.py` took 48 to 63 s (the eleven kept records), and the solver's next step asks for it twice at
every step: the answers written again, and the step before's texts under the new weights. That made the overlap the largest
part of a torch step, where the thirty forwards are 5.4 s. The call was profiled as it stood (the original's weights on the
texts the plain loop's first step kept, card 1, the card's launches blocking so each second is charged where it is spent),
changed, and held to the kept record of the same call.

| where a call's 56.4 s went, before | s | what was done | s after |
|---|---|---|---|
| copying a correct answer's turns to the card, 900 times (thirty answers by fifteen correct answers by two domains; 5.56 GiB each answer) | 16.5 | the correct answers' turns are copied once and kept on the card as far as its room allows | 6.0 (14 of 30 kept when profiled, 16 as run) |
| the path search, 900 times one after another (`complete_overlap.distance_path`) | 10.4 | the fifteen paths of an answer are searched together, on contiguous anti-diagonals | 2.0 |
| compressing the thirty files | 6.4 | compressed and written a thread behind | not waited for |
| the thirty forwards | 8.4 | unchanged | unchanged |
| the card's arithmetic for the 900 pairs (cosines, means, norms) | about 13 | one array of the answer's size less (the direction is made in the mean's own array, the rows are scaled in place) | about 11 |

- **As run, a call is 28.1 s where the kept record took 55.1 s** on card 1 beside the desktop (16 of the 30 arrays of turns
  kept on the card, 4.11 of 5.56 GiB), and **18.7 s where it took 48.1 s** in the mode that reads the original's own texts,
  where no model is on the card and all thirty are kept. A named model file is read through for its sha256 beside the work
  and no longer before it. The machine was loaded both times (load average 7 to 18), so these are not a quiet machine's seconds.
- **The paths are the same, every one**: 900 of 900 against the kept record on the step's texts, 900 of 900 on the original's
  texts, and the reference's own 870 paths between the original's answers come again as before. The new search
  (`paths_at_once`) makes each cell by the bound search's comparisons and its one addition; it is held to
  `complete_overlap.distance_path` on 1,386 made-up matrices with ties, nothing but ties and unequal sides (and in
  `test_overlap_step.py`, 12 fixtures). `complete_overlap.py` itself, which `controlled-bf16-8` binds, is not touched.
- **The other arrays are the kept record's as far as the instrument repeats itself.** 412 of 540 arrays are the kept record's
  bit for bit, and the unchanged program run again gives 409 of 540: a card's `index_add_` does not sum a step's rows in a
  fixed order, so the last bit of a mean moves between two runs of one program. The largest difference is 7e-07 in the
  float32 arrays and one float16 step in the per-correct-answer cosines, with the change as without it, and the summed
  departures agree to eleven places (0.06945378074 wrong, 0.04666464157 right).
- **What sets the room.** The model holds 4.71 GiB of a card and the fifteen correct answers' turns are 5.56 GiB (4.01 the
  answers' steps, 1.55 the questions' positions). Laying the longest answer holds 3.35 GiB more as measured (`most_held_bytes`
  less the model and what was kept), 4.09 GiB as `working_bytes` counts it from the program's own arrays; the room given to
  the kept turns is what the card has free less that count, so nothing about a card is assumed and a card without the
  desktop keeps all thirty. If a card runs out with turns kept on it, they are given back and the answer is laid again as
  before. Pinned memory would not have done it: turns reach card 1 at 11.2 GiB a second from pageable memory and 13.3 pinned.
- **A first form of this was slower and is not kept**: reading the card's free room after the longest answer had been laid
  gave 0.55 GiB, because torch keeps what it has once held, and the call took 68 s with the card nearly full.

For a step's own process (`overlap_step.py`'s header, "From another program"): `Reference()` once, then
`read_answers(reference, out, device, by_forward(model, texts), lengths)` for each set of texts with the model already in
memory, and `reference.back()` to give the card's room back to a derivative; the copies to the card are about half a second.

**Limits.** One call of each kind, on a loaded machine, on card 1; no quiet-machine time and nothing on card 0. What is left
of a call is the card's arithmetic over 900 pairs and the forwards, and a further factor would change the order of that
arithmetic and with it the last bits of every array, which this change did not.

## A step's generation by three recorder contexts, its states kept from that generation's own streams (declared 11:03 AM CT November 15; nothing run)

Astra approved source preparation only (`records/retrospectives/2026-11-15-recorder-workers-result-astra.md`): three-worker
kept generation under a new reader protocol that tells the control's, the generation's and the reading's executions apart,
with `controlled-bf16-8`, the installed recorder and the bound readers unchanged, and one integrated original-only
qualification declared and assessed before any candidate capture uses it. This section is that preparation. No recorder has
started under it.

**What sets the shape: where a step's states come from.** A step needs the model's own answers and their states at all 105
merges. Two records price the two ways of having the states:

| the states | record | seconds |
|---|---|---|
| a separate pass of the installed recorder over the thirty answers | `runs/whole-states-original/reading/reading.json` (`made_by`: one recorder pass) | 235.7 |
| kept from the streams the generation itself wrote, no pass | `runs/success-sets-loop-from-mates/loop/1/reading/reading.json` (`made_by`: from the streams, KEEP_RAW) | 28.9 |

A faster generation followed by a separate pass is therefore slower than `controlled-bf16-8`'s step, which already keeps
its states from its generation (about 154 s of generation as arithmetic, then 28.9 s). This seat's first form 3 (committed
10:41 AM CT, `f1c290d0`) had the workers generation followed by a separate pass of the installed recorder: it priced the
generation alone. It was replaced at 11:01 AM CT (`2c743431`) before a protocol was sent or anything ran. The states are
kept from the workers generation's own streams, and the reading says that is what it is.

**Why new files and not the bound ones.** `states_reading.py` refuses streams written by another recorder than the installed
one, and `states_conditional_check.check` refuses to compare two readings that differ in more than the libraries. Both
refusals are right for what they bind and neither is widened: a reading kept from the workers recorder's streams is not the
installed recorder's reading and is not labelled as one. The new programs stand beside the old, and no file `controlled-bf16-8`
binds is edited.

| program | what it is |
|---|---|
| `generation_engine.py` | the workers recorder's identity (`5c893cd713e3…`, three contexts, three scheduler totals a context) and what a call must leave: every context's account, each case taken once |
| `own_text_workers.py` | the capture: `own_text.py`'s, ten answers a call on three contexts; its `text.json` says `execution_role: generation`, the recorder, the libraries, the generation's sha256, the worker count and every call's account |
| `states_reading_roles.py` | keeps a workers generation's streams as a reading (`ROLLOUTS`), makes no pass and refuses to; the record says `execution_role: kept from the generation`, the generation's identity and the `text.json` it was kept from |
| `reader_protocol_roles.py` | form 3 (`whole-gap-reader-protocol/3`): `execution` (the control's), `generation_execution`, `reading_execution` (a reading by its own pass, as form 2), `the_executions`, `integrated_qualification`, `admits` |
| `states_correspondence_roles.py` | `whole-gap-correspondence/3`: `execution` holds `generation`, `reading` and `reading_role`, each from its own record |
| `generation_qualification.py` | the qualification's account (`account`) and the card's memory while the capture runs (`watch`) |

`reader_protocol_roles.held` takes a reading by the role it says. One that says it was kept from the generation is held to
`generation_execution` and to the capture's own `text.json`; one that says nothing (the installed recorder's own pass) is
held to `reading_execution`; any other role is refused. Forms 1 and 2 are read by the programs that read them before, with
their meaning unchanged (their fixtures pass unchanged: 21 and 18; the new ones: 11, 10 and 9).

**The protocol as it stands qualifies nothing.** `reader-protocol/controlled-bf16-9-qualifying.json` (sha256
`6fe185584079…`, made 11:02:05 AM CT November 15 from files, 24 instruments) has `integrated_qualification: null`, and with
that `held` takes a capture of the original model only. `controlled-bf16-9.json` is made after the qualification below has
run and Astra has assessed it, with the run bound in.

**The declared run: `generation-workers-original`** (`runs/generation-workers-original/declaration.json`, declared 11:03:49
AM CT; `run.sh capture reading held account`). The thirty fitting answers of the original are generated in the capture's
three calls of ten on card 0 with their streams kept; the reading is kept from those streams; the protocol's `held` takes
both; and the account lays every call's and every answer's accounting beside the kept reading compared bit for bit with
`whole-states-original/reading` at every merge. That comparison is across two executions (the workers recorder on the kept
scheduler, and the installed recorder as the control ran it) and is made inside the qualification by
`generation_qualification.against_the_control`, which says so in its record, because the bound checker refuses the pair.
The launcher refuses `capture` until Astra's admission stands in the declaration, and checks before every step that each of
the 26 files the declaration binds is still those bytes. While the capture runs nothing else of this seat runs on threads
2, 3, 4, 10, 11 and 12 (the workers comparison shared them with this seat's own file analyses, and its ratios carry that).

**Revision 1, after Astra's review** (`records/retrospectives/2026-11-15-generation-workers-review-astra.md`; made 11:37 AM CT
November 15, before admission, nothing run). The review held admission for two gaps, both real. *The scope was checked after
the act:* the original model and the fitting population were held only by the protocol's `held`, which runs after the
capture and the reading, while the capture takes a selection (`CASES`), another solver file (`TUBE`) and a pass (`PASS`) from
the environment. Now `generation_qualification.py` runs each step: it refuses those and six other overrides before any source
of the repository is loaded, then holds the model's sha256, the cohort, the selection's record and the thirty ids the capture's
own selection gives against the declaration's `population` and the protocol's control, in the process that goes on to run the
capture, before a selected trace is opened or a recorder started (`in_scope`, `scope`); the reading does the same. My first
version of that check called the selection before refusing the overrides, and a `CASES` naming a manifest was opened by it:
the trial with the variable set showed it, and the refusal now comes first, with a fixture holding the order. *A commit's
name stood for bytes:* 26 files were bound and the other pipeline sources left to the commit `result.json` would name. Now 61
files are bound: the 54 Python sources the four steps load (`generation_qualification.py closure` lists them from the
programs' imports and by importing what they name), the two recorders, the lock's script, the launcher and three protocol
instruments no step loads. `run.sh` holds all of them before every step, and each step holds every Python source it loads to
that list at import (`Held`): a file not bound, or bound at other bytes, is not loaded and the step ends. Forty of the 54 are
`code/signal-trace/pipeline` files other seats work in; a change to one of them before the run means a new revision, not a
relaxed check. The protocol is unchanged (none of its instruments changed). Fixtures: 18 for this program where 10 were.

**Limits.** Nothing here is measured: the time of a three-worker generation of thirty answers is not known (the ten-answer
comparison gave 37.65 s a call against 79.38 s installed on a loaded machine, and any thirty-answer figure from it is an
extrapolation). Answers 11 to 30 have never been generated by the workers recorder. Three kept contexts held 7,878 MiB in
that comparison, so the generation does not share a card with a wide solve. The scorers (Astra's) read forms 1 and 2 and do
not read form 3 yet. Nothing is adopted: the readers and `controlled-bf16-8` are as they were.

## The engine's own trace overlapped with the correct answers directly (11:16 AM CT November 15; `overlap_reading.py`; the original only, files and card 1, no engine pass)

The direct overlap had two sources for an answer's turns: the torch forward of a text, and the original's exported turns. A
step of an admitted loop is measured on the engine's own trace, with torch supplying the derivative (Astra's choice of
November 15), and what the engine computed for a model's own answers is a reading. Until now nothing laid a reading on the
correct answers: the engine's trace reached them only through the withdrawn route (each answer laid on its own original
recording, then the mate's rows). `overlap_reading.py` is that seam. It takes a reading's stored states as every merge's turn
and log gain (what a merge was given is the state one merge earlier, the layer's output scale and the token's row from the
model file: `turns_reading.py`'s header) and hands them to `overlap_step.read_answers` unchanged, so the tables, the paths and
`overlap.json` are the same things made from the engine's numbers. No forward is made.

**Held on the original** (`overlap-step/reading-original/overlap.json`; `whole-states-original/reading`, card 1; made twice,
11:16:27 to 11:16:53 AM CT and, with the program as committed, 11:20:45 to 11:21:11 AM CT, which is the record kept; the
tables agree between the two):

| | the engine's reading | the kept overlap from the exported turns (`overlap-step/original`) |
|---|---|---|
| the reference's own paths between the original's answers | 870 of 870 come again | 870 of 870 |
| paths (thirty answers on fifteen, two domains) | 900 of 900 the same as the kept overlap's | |
| the pooled arrays (`shared`, `toward`, `size`, `gain`, theirs) | within 4.8e-07 of the kept overlap's | |
| the arrays by correct answer (kept in float16) | within 4.9e-04, the float16 step | |
| mean departure, wrong and correct answers | 0.070999398, 0.046685225 | 0.070999398, 0.046685225 |
| seconds for the thirty answers | 20.9 and 17.8 in the two calls (7.1 and 3.8 of them reading the states and making the turns) | 48.1 as first kept; 28.1 since the 10:21 AM CT change |
| most held on the card | 8.90 GiB | |

So a native measurement on the direct overlap costs 18 to 21 s once a step's reading exists, with no model on the card. In the
stopped loop's first pass the same place was held by the registration (43.1 s) and the set measurement (250.8 s).

**How far the torch forward's overlap stands from the engine's** (`overlap-step/torch-beside-the-engine-original.json`; the
original's weights on the engine's own thirty texts, `overlap-step/original-weights-on-the-engine-s-texts`, 29.2 and 31.0 s, held beside
the reading's overlap from files). This is the distance between where a step's derivative is taken and where it is measured:

| | |
|---|---|
| paths the same | 893 of 900; the seven others share 510 of 520 pairs in the worst case and all but 1 to 3 pairs in the other six |
| an answer's departure, torch against engine | largest difference 1.0e-05 (0.128536 against 0.128546), median 2.2e-06 |
| mean departure, wrong and correct | 0.0709989 and 0.0466843 by torch; 0.0709994 and 0.0466852 by the engine |
| `toward` cell by cell | 9.5e-06 apart at the median, 4.5e-04 at the 99th hundredth, 0.011 at the 9,999th of 10,000, 0.47 at the largest |
| rows whose worst merge is over 0.01 and over 0.1 apart in `toward` | 65 and 3 of 19,751 |

The few far cells are the positions the model computes unstably ("The whole model in torch, with gradients": 11 of 19,751 over
0.1 from the engine at their worst merge). They do not move an answer's departure in the fifth place.

**For a step's own program.** The folder is one `path_step.py` already reads: `texts=<folder>/texts.npz tables=<folder>` with
the reading's model takes the overlap as the engine made it (its `texts_sha256` and `model_sha256` are what that program holds
a kept overlap to), so a derivative can be taken along the paths the measured trace gave. In a process:
`turns_of, by_id, texts = overlap_reading.by_reading(reading, model_file, tensors, scale, width, device)` and then
`overlap_step.read_answers(reference, out, device, turns_of, lengths)`.

**On a changed model's reading** (bound before it was read: `overlap-step/reading-matched-own-bound.json`, commit `4ab6ddc1`;
run 11:24:54 to 11:26:08 AM CT; an instrument check and no assessment of the model). The one changed model's own-path reading
on this disk is the stopped loop's first capture: the matched sweep's model (`05b169076ada…`) on the thirty answers as it wrote
them, 18,375 rows. The program laid it (`overlap-step/reading-matched-own/overlap.json`): 31.3 s, 17.2 of them reading the
states, which were not in the page cache (the original's were, at 3.8 and 7.1 s). The torch forward of the same model on the
same texts beside it (`overlap-step/torch-beside-the-engine-matched-own.json`): 895 of 900 paths the same (the five others
differ by 1 to 14 pairs of 465 to 536); an answer's departure 1.9e-05 apart at most and 1.9e-06 at the median, where the
least gap between two answers' departures is 7.2e-05; `toward` 7.6e-06 apart at the median cell, 2.8e-04 at the 99th
hundredth, 0.12 at the largest; 12 and 1 of 18,375 rows over 0.01 and 0.1 at their worst merge. The prediction held: torch
and the engine give one overlap to the fifth place on a changed model too. The pooled departure on those texts is in the
record as the program always keeps it and is not read: it follows the words ("One question in two wordings").

**Limits.** One changed model, a state-value sweep's and not a step of the loop being designed; the torch-against-engine
distance is measured there and at the original, not after a step taken on the overlap. Card 1, a machine at a load of about
9, two calls on the original and one on the changed model. `departure` stays what it was, a plain reading of the tables and
no loss. The definition of what is locally shared is Astra's and is not here.

## The three-worker generation qualified on the original: the same bits at every row (`generation-workers-original`, 11:59:37 AM to 12:03:32 PM CT November 15; card 0)

Astra admitted revision 1 at 11:52:03 AM CT (`records/retrospectives/2026-11-15-generation-workers-admission-astra.md`) and
`run.sh capture reading held account` ran once, in order, from commit `81f6ea20`, each step exiting 0. The record is the run's
`result.json` and `compare/integrated.json`.

**What it found, as predicted.** The thirty fitting answers were generated in three calls of ten by three engine contexts a
call, with the recordings' tokens at all 14,592 answer steps; every call accounts for each of its ten cases once and for its
nine scheduler contexts (27 in all, no split unmarked, every split one range). The states were kept from that generation's own
streams, with no second pass, and the protocol's check took the capture and the reading as its generation. Every row is the
original control's bits: 19,751 rows at all 105 merges, `l_out` at 19,751 rows and `result_norm` at 14,592, largest difference
0.0, no answer apart. The comparison is across two executions and says so: the workers recorder on the kept libraries against
the installed recorder on the installed ones, same model, card and driver.

**What it took** (recorded, no condition; load average 7.0 to 11.1). The capture 108.3 s, of which 101.1 in the three recorder
calls (37.4, 33.1 and 30.6 s, a model load in each); the reading 49.1 s; the recorder 7,878 MiB of card 0 at most, nothing else
listed on the card. For scale, and not like with like: the stopped loop's first pass spent 205.1 s generating its thirty answers
with one context of the installed recorder, and a separate pass of the installed recorder for the control's states took 235.7 s.

**Limits.** The original only, once, on a loaded machine. Nothing is adopted: the readers and `controlled-bf16-8` are as they
were, and `controlled-bf16-9`, binding this run as its integrated qualification, is made only after Astra assesses the result.
The scorers do not read form 3 yet. The kept reading's arrays (12.95 GB, the control's bits) stay in the run's folder until
that assessment; the run's 110 record files are in the bucket, read back at 12:07 PM CT.

**Accepted, closed and frozen** (12:12 to 12:15 PM CT November 15). Astra accepted the result at 12:12:21 PM CT
(`records/retrospectives/2026-11-15-generation-workers-result-astra.md`, 6a4fcd29) and named the duplicate arrays for the
declared close. The kept reading's 107 arrays left at 12:14:10 PM CT (12,952,782,208 bytes; the two readings' inventories name
the same files at the same sha256, and the folders held exactly those files); free disk went from 173,970,636,800 to
186,928,041,984 bytes and no deleted file of the run is held open. `reading.json`, `row-map.json`, the per-answer records and
every account stay. The run's `result.json` then took its `status`, `accepted` and `release` (it is another file than the one
assessed, and says so). **`reader-protocol/controlled-bf16-9.json`** was made at 12:15:10 PM CT, sha256
`500ecec06f63394c74f2f519fe1213c3ba57ec3be0fc7b120d71322cc3e68704`: form 3, the same 24 instruments at the same bytes as the
qualifying file, with the integrated qualification bound by its declaration (`02a949f63b0f…`), result (`dde99a1538c6…`) and
comparison (`2c3fb665754b…`), so it admits a changed state's capture and the reading kept from it. The control, the generation
(workers recorder, three contexts) and a reading by its own pass (installed recorder) stay three executions.
`controlled-bf16-9-qualifying.json` (`6fe185584079…`) and `controlled-bf16-8.json` (`064c773f6235…`) are unchanged. No capture
has run under the frozen file; a run that uses it binds it in its own declaration, and the scorers' reading of form 3 is Astra's.

## The fifth pass on what compares tokens: two readings left, and what is held (November 15, from 1:44 PM CT)

Marcos asked whether tokens were being compared again, and that it stop (relayed by claude-research at 1:44 PM CT). It continues
"The instruments that measured by tokens are removed" above, and is written here at the page's end so that no line the index
points at moves. Source only: nothing ran on a card or an engine.

This seat's programs were listed again from the folders, not from the list sent: 37 files of this folder, of
`2026-11-10-speed-review` and of `tools` name a token, a text or a wording. Two of them were readings decided by tokens, written
on November 15 after the direction of November 14, bound by no protocol and by no run that is open or paused, and they left
with their tests: `overlap_wordings.py` (the overlap's tables split by whether a row and a correct answer's row were given the
same token) and `parting_places.py` (where two sets of texts first part, token against token, and how near a tie the scores
stand there). Their tables stay as records of what was found (`overlap-step/wordings-*.json`, `parting-*.json`,
`first-step-*.json`), and the sections above that report them stay as written; nothing is to be built on either reading. The
MLflow view's token-decided readings left earlier the same afternoon (`tools/box-image/README.md`, the `loop.json` row).

One closed run names them: `plain-first-step-state` (the solver's, closed 9:48 AM CT November 15 with its result) bound
`parting_places.py` for its reading of where the texts part (sha256 `ab36e4f98569…`, commit c7e532c6) and lists
`overlap_wordings.py` among what that reading loads (`59d8fd0091dd…`), and its two closing scripts name the programs
(`2026-11-15-relation-reading/close_first_step_state.py`, `bind_first_step_parting.py`). Both programs stand in git at those
bytes in every commit up to b385993d (`git show b385993d:records/development/2026-11-12-path-forward/parting_places.py`), so the
closed run's bindings stay readable; its declaration is its owner's and was not touched, and the verifier passes without the
files.

**What of this seat's still compares tokens, and what holds each** (none is a loss; none was touched in this pass):
- `overlap_step.py`, `same_texts` and its `beside=` option: how many answers were written token for token as a kept set holds
  them. Added November 15 as acquisition, and a count of texts all the same. Only this file's own command line and its test
  call it; the paused loop does not (its record's counts of answers written as the step before and as the original are made in
  the solver's `2026-11-15-relation-reading/path_step.py`:671-673). `shared-relation-loop-torch`'s declaration binds this file
  by sha256, so the count leaves when the loop's owner says the paused loop does not start again or revises its declaration.
- `torch_forward.py`, `write`'s own record (`torch-forward/write-original.json`: how many of the thirty the torch forward writes
  as the recorder did): the qualification of the writing, bound by the same declaration.
- the protocol-bound readers of `controlled-bf16-8` and `-9`: `capture_bound.py` (an answer whose complete inputs are the
  reference's keeps its identity: the objective owner's decision of November 14), `states_conditional_check.py` (a same-history
  control held to the original where both were given the same input), `generation_qualification.py` (the generated tokens
  against the recordings', beside the states' bits). Each is the identity of an acquisition, not a reading of a model; a
  change to any of them is a new protocol.
- the programs of closed card-1 runs named in the fourth pass, which found rows by the recorded text.

Whether the first two go is the loss and loop owner's to say, since the file they sit in is bound by a run that is hers to
resume or close; the request went to her at 1:55 PM CT.

**Her decision, and what was done on it** (Astra, answered before 1:56 PM CT November 15): the token-conditioned paused loop
does not resume, and the solver retires its driver by a dated revision of its declaration; `same_texts`, the `beside=` option
and their test leave; the writing's qualification record stays as it is (acquisition evidence of the original qualification,
not a score of progress, and taking a count out of a historical record is no new qualification); the identity checks of the
protocol-bound readers stay, with no new protocol and no run made again.
- `overlap_step.py` no longer holds `same_texts` or `beside=`: a text the model writes is kept and overlapped, and is compared
  with no other text. Its test of the count left with it; the file's other eleven tests pass, and the verifier passes. The
  file's sha256 was `ae6115ebfb0b…` as `shared-relation-loop-torch`'s declaration binds it (in git at those bytes up to
  f134bd5f) and is `e822f9912df6…` now, so the paused loop as declared cannot be started on this file, which is the intent.
  A record made before this that holds `written.as_held_by` stays as written.
- `torch-forward/write-original.json` and `torch_forward.py`: unchanged.
- `capture_bound.py`, `states_conditional_check.py`, `generation_qualification.py`, `controlled-bf16-8` and `-9`: unchanged.

## The older readers that a token, a label or an answer decided left the tree (November 15, from 2:56 PM CT; source only)

The fourth pass had left these "still here, named as gaps" (above, "Still here, named as gaps, with what holds each"), held
because each closed run's kept store is still read and its declaration names its program. Astra, who holds the loss and the
loop, asked this seat that afternoon to finish the retirement in its source: executable judgments of progress leave with their
consumers and tests; the original cohort labels, the identity and completeness of acquisition and every historical record
stay; no original label is made again or read differently. Nothing ran on a card or an engine. The standalone record, with
each program's sha256, what named it and what stays, is `retired-progress-readers.json` in this folder; every program's bytes
are in git at efdfc802. No declaration, registry record or reader protocol was rewritten and no run folder was written to.

**Out of the tree:**
- `pipeline/overlap/counterexample_replay.py` (`sign-counterexamples-replay-original`, closed): its twelve cases are chosen by
  the sign an answer writes and its reading is of the row that writes the sign. The run's cases, its kept store of twelve
  cases and its records stay.
- `pipeline/overlap/own_parent_control.py` and its test: a weighted cost of state values between a continued candidate and
  the model it was continued from; no declaration named it and no run holds an output of it.
- `pipeline/decisions/answers.py`, the answer judge, and its test. Its one caller was `pipeline/overlap/wide.py`'s `select`,
  which made the frozen cohort's two groups; `select` left with it. `runs/wide-overlap/members.json` is the cohort as it was
  committed on November 9 (a06609c9) and is the record of those labels. `wide.py` keeps `capture` unchanged.
- in this folder, the readings placed at the row where an answer writes the sign or found by the words of a text:
  `sign_rows.py` with `sign-rows/read.sh` and `sign-rows/summarize.py`, `same_text_sign.py`, `row_precision.py` with
  `row-precision/read.sh`, `row_map.py` (not the `row-map.json` a reading keeps, which `states_reading.py` writes from
  positions alone), and `deep-think-analysis-2/regime_frame.py`, `regime_frame_reordered.py`. Their tables and logs stay
  where they are (`sign-rows/`, `same-text-sign/`, `row-precision/`, `row-map/`, `sign-direct/`, `sign-by-regime/`,
  `deep-think-analysis-2/`), as history of findings already recorded; nothing is built on them.

**Read and left, as acquisition:** `pipeline/overlap/opening_replay.py` (the transplant's passes listed from their own
records and held to their recorded tokens; it carries the cohort's group of a question and judges nothing),
`fed_replay.py` (the capture and record the replays share; protocol-bound instruments import it), `wide.py`'s `capture`,
and `pipeline/decisions/population.py` (the rule that sets round two's population; nothing calls it now).

**Read and left, because they feed none of these:** `pipeline/overlap/calibrate_labels.py` and `member_scores.py`
(statistics of the original cohort's recorded traces by the cohort's own labels; no changed model, no token, no answer's value). **Corrected November 15, 2026, on Fable's reading: "no token" was wrong.** They read no token's score, but the rows they score are the token-matched correspondence's (the sections destination and descent), so both left the tree in the cleanup's third wave (47fdeb8b; `retired-progress-readers.json`, `corrected_by`). This seat had read what they scored by and not which rows they scored.

**Not this seat's in this pass:** `request_keys.py` in this folder, with `pipeline/overlap/request_realization.py` and the a2
legacy constructors, are the solver's to retire on Astra's word.

**Still here, and put to the objective owner:** readings of a model by state values that take no token or word
(`pipeline/overlap/own_capture.py`'s `paired` and `moved`, `members.py`, `whole_model_l.py`, `conditional_members.py`, and
`2026-11-10-speed-review/as-we-go/`). None was touched.

## The readings of a model by state values left the tree (November 15, from 3:08 PM CT; source only)

Astra's answer to the paragraph above, at about 3:08 PM CT: the executable progress readings of the withdrawn state-value
objective leave with their dedicated consumers and tests; what they wrote stays as history, and the old measurement programs
do not stay to run that history again; native acquisition, raw-state extraction, correspondence and capture completeness stay
where the same files give them. No pass, no card, no reading was made: source only. The record, with each program's sha256,
what it read and what of its output stays, is `retired-state-value-readers.json` (standalone: no declaration, registry record
or reader protocol is rewritten, and no run folder is written to).

**Removed (24 programs; their bytes are in git at 5a354bf2 and before).** From this folder: `members.py` (each answer against
the fixed original correct pattern), `whole_model_l.py` (the whole-model L), `conditional_members.py` (experiment C's
conditional reading), and their dedicated consumers and runners `text_and_weights.py`, `text-and-weights/read.sh`,
`l_by_layer.py`, `conditional-members/read.sh`, `conditional-members/first-draft/read.sh` and `given_text_map.py`. From
`2026-11-10-speed-review/as-we-go/`: `as_we_go.py`, `own_compare.py`, `along.py`, `arrivals.py`, `cells.py`, `follow.py`,
`lines.py`, `means_outside.py`, `read_states.sh`, `report.py`, `run_card1.sh`, `size_toward.py`, `states.py`, `thirds.py`
and `where_it_binds.py`. None had a test.

**Changed.** `pipeline/overlap/own_capture.py` lost `paired` with its command, `moved`, and the distances and cosines `record`
made with it; `record` now says only what the capture holds (where the histories part, whether the answer ended by itself, the
positions, the shared positions and the first merge that differs). Its capture, its control and the functions that give kept
states back (`parts`, `outgoing`, `control_states`) are as they were. `export_own.py`, `prompt_overlap.py` and `replay_cost.py`
changed in their heads only.

**What stays, and is no progress reading.** `as-we-go/matrices.py`, `checkpoint.py` and `candidate.py` (model files made from
the original and a sweep's kept matrices; they measure nothing, and `kept-matrices/` needs the first to give a state back);
`as-we-go/readings/` and `kept-matrices/`; every table, `l.json`, `paired/` folder and reading these programs wrote, here, in
the speed review and in the runs. The sections of this page and of the speed review's that report those readings stay as
records of what was read; the commands they give are no longer in the tree.

**Named in files not touched.** `own_text.py` and `own_text_workers.py` (bound by the reader protocols) point at
`own_compare.py` for the `TUBE` convention: with `TUBE` set, that file is loaded in `pipeline.weight_edit.tube`'s place.
`prompt_features.py` names an `own_compare.py` capture (the form `<capture>/own/<answer>/start.npy`, `turns.npy`, `norms.npy`,
`answer.json`). `fed_path.py` takes its two centres from `members.py`'s cache.

**Still here, and put to the objective owner:** state-value readings her answer did not name and that are not dedicated
consumers of the ones it did, told from their heads and not read line by line: `fed_path.py`, `rung_map.py`,
`fixed_input_movement.py` and `prompt_features.py`'s `pair`. None was touched.

**`request_keys.py` left as well** (d9c43f66, 3:08:10 PM CT, by the solver's seat on Astra's decision, with
`request_realization.py` and their tests): the sections above that describe it, from line 1263 on, are records of a program
whose bytes are in git before that commit, and the limit at line 1346 (files held unchanged while the line chain's own reading
was pending) has no pending reading behind it since the chain was stopped on November 14.

## The same category, counted from the source (November 15, from 3:36 PM CT; source only)

Astra's word on the section above, read at 3:36 PM CT: the scope is the category, not the names she had been given. Every
executable progress measurement against the withdrawn state-value targets leaves, the four programs named to her and any other
of the kind, decided from the source; capture, raw-state extraction and generic forward and derivative arithmetic stay. So
every program of this seat's folders was counted from the folders and classed; the count, each class and each file's sha256
are in `retired-state-value-readers.json` (`second_pass`). No pass, no card, no reading was made.

**The rule applied.** A program, or one command or field of it, is in the category when running it makes a reading of how
far states stand from a state-value target of a withdrawn objective (the two original centres, the tube's bounds and loss,
the reference's L, the success sets), or of how far one run's states moved from another's. The states may be a changed
model's, an intervention's on the original, or a solve's result. Seconds, bytes held equal, and the difference between two
forms of one arithmetic are not such readings.

**Removed whole (10; bytes in git at 6aaad3ec and before).** `fed_path.py` (a fed run's path against the two centres and
against the unfed run's) and `rung_map.py` (the rungs' traces against the original correct path). From the speed review:
`calibration/back_caps.py` with `table.py` and its three queue scripts (what a setting of the old solver's bringing back
reaches against the reference's bounds), `solver-as-of-2125/driver.py` (the old solver's choice at one matrix read against
the bounds, once on the thirty answers no fit uses), and `solver-build/make_driver.py` and `run.sh`, which wrote and ran
such a driver.

**Cut inside a file that stays (3).** `fixed_input_movement.py` lost `read`, `direct_axis` and `same_bytes`; its `check`
and the block's arithmetic (`tensor`, `rms`, `Block`, `entering`, `cosine`, `apart`, `cases`) are byte for byte what they
were, and `rows_from_states.py` imports `Block` as before. `prompt_features.py` lost `pair` and keeps `capture`.
`pipeline/overlap/fed_replay.py` lost `beside` and the paired same-history row its `record` made; no frozen reader protocol
lists that file among its instruments, and its capture, `states` and `scales` are unchanged. `declared_inputs.py` names
`own_capture.py` as the reader of the fed replay's store.

**What is left, by class** (159 programs: 42 here, 72 in the speed review, 13 in the rented solve, 32 in `pipeline/overlap`).
Here: acquisition 20, raw-state extraction 5, correspondence 4, identity and completeness checks with the reader protocols 7,
generic forward and derivative arithmetic 2 (`torch_forward.py`, `fixed_input_movement.py`), the direct overlap on each
merge's relation 3 (`overlap_step.py`, `overlap_reading.py`, `overlap_depth.py`: an instrument and no loss, no state-value
target), and one description of the original cohort by state values that reads no changed model (`prompt_overlap.py`). The
speed review: model files from kept matrices 3, disk 4, time from records 6, the machine 4, recorder arrangements held to the
installed recorder's bytes 19, acquisition checks 4, the old solvers' arithmetic timed or held to another form 30, the
verifier's scan 2. The rented solve: a solve's hosting 11, its memory and basis 2. `pipeline/overlap`: this seat's acquisition
4, the original cohort's overlap and statistics 28. Of these the path-forward programs and every file changed were read in
their source; the others were classed from their heads and a search for the old objectives' readings by name.

**State-value arithmetic that stays and is no progress reading:** the check in `prompt_features.py capture` that its raw rows
are the rows a full capture read, the two checks that two files hold the same arrays (`export_own.py check`,
`rows_reading_check.py`), and `fixed_input_movement.py check` (the block rebuilt against the engine's kept part).

**What the removals leave unread:** the sections above that report `fed_path.py`, `rung_map.py`, fixed-input movement and the
paired prompt error are records of what was read; `fed-path/`, `fixed-input-movement/` and the tables stay.

## The third pass: a contrast of the cohort by state values left (November 15, from 4:11 PM CT; source only)

On Marcos's direction relayed at 4:11 PM CT November 15 the cleanup of the tree is Fable's and this seat's to decide between
them, so that Astra keeps to the loss and the loop. As this seat understands him, files left in the tree end up influencing the
work: what leaves is anything executable that takes tokens, words, answers or state values as a measure of progress, and
whatever only those things use; capture, raw extraction, arithmetic still used and the records as they were written stay.

`prompt_overlap.py` left (`retired-state-value-readers.json`, `third_pass`, with its sha256). The second pass had kept it as a
description of the original cohort, but it splits the answers' two groups by state values at the question rows against the
withdrawn correction reference, and nothing takes it. Its tables stay in `prompt-overlap/`, and the sections above that report
them are records of what was read. This folder holds 41 programs after it.

**Not done by this pass, and why.** `states_correspondence.py`, `states_correspondence_roles.py` and `states_conditional.py` only
bind a reading to a state-value target for the old scorer, so they are in the category. The frozen reader protocols list them,
and the bound instruments take a few names from the old solver's modules (`tube`, `ladder`, `gap_targets`, `gap_run`) and the
old scorer's loader of a native reading. The approach agreed with Fable is one joint step: those modules cut in place to the
names the staying instruments take, the three records removed, and one protocol made again, so that the protocol changes once.

## The fourth pass: what only the old sweep solver and the old scorer used (November 15, from 4:38 PM CT; source only)

The old sweep solver fits weights to a target on state values, so it is in the category of the section above, and Fable is
cutting it. On this seat's side what only that solver or the old scorer used leaves with them. From this folder, nine files
(`retired-old-solver-readers.json`, with each file's sha256 and reason; the bytes are in git at 570673c8):

- the rows readers, `rows_reading.py` with its test, `rows_reading_check.py` and `rows_from_states.py`: every matrix's input
  rows, from a recorder pass or from a state reading. Rows were the sweep solver's input and nothing else took them.
  `fixed_input_movement.py` keeps the block arithmetic `rows_from_states.py` imported.
- `reference_turns.py` with its test: it wrote the original's turns in the withdrawn set reference's form. Its trial is on
  record and no artifact was made. `turns_reading.py`, which the direct overlap uses, stays, and so does `prompt_turns.py`.
- `prompt_features.py`, `export_own.py`, `replay_cost.py`: a capture through the old solver's `correction.passed_on`, a kept
  reading in the form the old original-reference instruments read, and the price of a whole replay for the old loop.

**What the sections above still say.** "Every matrix's rows from one pass", "The rows reader had the capture's seam", "The
question rows' turns at every merge" and the sections on the prompt's features are records of what was built and read; the
tables and kept arrays they name stay. This folder holds 34 programs after the pass: the capture and the reading of a model's
own computation with their checks, protocols and qualification, the torch forward, the direct overlap on the relation, and the
three records named in the section above, which wait for the joint step.

## What the next reader protocol has to bind (read 5:00 PM CT November 15; source only, nothing run)

`controlled-bf16-9` no longer holds the capture as it stands. Astra changed `capture_bound.py` in the working tree for the new
loop's joint-state acquisition (sha256 `263870a1a22a…` at 5:00 PM CT, against the `14f2ab9c0ee5…` the protocol binds): a
declaration that carries `functional_loop` goes to `pipeline.weight_edit.functional_capture.admitted`, which checks the actual
initial and reached model hashes, the complete joint installation, and the cohort, history, output and protocol instruments.
By her word no native capture under that branch is declared or run, and she asks that the change go into the protocol made
again after the cleanup.

The instrument lists of `-8` and `-9` were named by hand in `reader_protocol.py` and `reader_protocol_roles.py`, and they are
not what a capture loads: `capture_bound.py` imports `gap_run.py` (`begins`, `of_pass`), which neither list names, so a change
to it never changed a protocol's identity. The protocol made again binds the capture's whole import closure, read from the
source and not from a list:

- `capture_bound.py` as it then stands, with `functional_capture.py`, `functional_install.py`, `functional_reference.py`,
  `functional_layout.py` and `functional_reading.py`;
- `gap_state.py` (`Working`, `digest`, `encoded`, which `functional_install.py` takes) and what those reach in `a2.py` and
  `native_trace.py`;
- what Fable's cut leaves of `gap_run.py`, `gap_targets.py`, `tube.py` and `ladder.py`, and `whole_context_score.py`'s
  `NativeReading`.

It is made once, after the joint step in which the three records of the section above leave, and Astra is given its hash with
both seats' commits and a record that each definition kept is the bytes it was.

**The closure as the tree stands, read by syntax at 5:16 PM CT** (every `import` and `from` of the fifteen entry programs of
the capture, the two readings and the protocol, followed through this folder and `pipeline/`; files only). It is 65 files: the
fifteen, and fifty of the pipeline. Among the fifty is the whole old solver (`correction.py`, `field.py`, `gap_sweep.py`,
`gap_rows.py`, `gap_late.py`, `gap_sets.py`, `gap_requests.py`, `fused.py`, `context_score.py`, `context_targets.py`,
`success_sets.py`, `a2_pattern.py`), reached through `tube.py`, `ladder.py`, `gap_run.py` and `whole_context_score.py`. Of
`pipeline/weight_edit` this seat's staying programs take 19 names and no more: `a2.NORM_EPSILON`; `gap_run.begins`, `of_pass`; `gap_targets.of_tokens`;
`ladder.CORRESPONDENCE`, `INPUTS`, `TENSORS`, `baseline_of`, `cohort`, `first_direction`; `tube.NATIVE`, `ORIGINAL_MODEL`, `WIDE`,
`digest`, `picks`, `rows_of`, `unit_and_log`; `whole_context_score.NativeReading`; `functional_capture.admitted`; and each
module's `__file__` where a record names the source it ran. Two things follow for the protocol made again:

- the closure is taken after Fable's cut, when those four files hold only the names above: bound now, it would bind the solver
  that is leaving;
- the closure also reaches the gate (`run_declaration.py`, and through it `research_guard/`: the verifier with its parameter,
  archive and forbidden-bytes checks). `run_declaration.py` is taken by the capture and is bound. The files of
  `research_guard/` decide whether a capture may start and read no model and no trace: this seat will name them in the protocol
  as reached and not bound, with that reason, so that a new rule in the verifier does not change a reader protocol's identity.
  Whether that boundary stands is Astra's to say when she is given the hash.

Two loads are by a computed name and are not followed by syntax: `own_text.py`:57 and `whole_reading.py`:89 put a named
version of `tube.py` in the module's place when `TUBE` is set. A capture records the `tube.py` it ran, `held` refuses another,
and the functional capture refuses `TUBE` outright.

## The joint step: three records, the late recorder and the fused writing leave; a protocol binds what its programs import

Committed 5:42 PM CT November 15, 2026 (cca5253e). Source and files only: nothing ran on a card. The record is
`retired-joint-step.json` (each removed file's sha256; each changed file's before and after; the bytes are in git at
eb569efa). It is this seat's half of the step agreed with Fable, whose cut of the old solver is eb569efa
(`records/development/2026-11-15-tree-cleanup/README.md`, wave 2).

**What left.**

- `states_correspondence.py`, `states_correspondence_roles.py` and `states_conditional.py`, with their tests. Each wrote a
  record whose only reader was the old scorer: a reading bound to a compiled target of state values. The scorer left in the
  cut. `states_conditional_check.py`, the control reading's own check of its inputs, stays, and its five tests stand in
  `test_states_conditional_check.py` on fixtures of their own (they stood on the old scorer's).
- The late-start recorder: `observe-stream-late`, `observe_stream.late.cpp`, its identity record,
  `pipeline/acquisition/late_start.py` and its tests. It read one matrix's rows for the old sweep without computing the layers
  before them, and nothing else took a late pass. `run_a_observation.run` runs one recorder and no longer takes a program's
  name; a case that carries `late_start` is refused, since `observe-stream` would read it whole and say nothing.
  `observe-stream` and its source were never changed by it. `STREAM-RECORDER.md` says where the section and the bytes are.
- The fused writing's six programs under the speed review's `fused-writing/` (that page, last section).

What these programs measured stays as it was written: the late start's comparisons and Astra's admission, the fused writing's
card readings, every run's records.

**What a protocol binds now.** Through `controlled-bf16-9` the instruments were a list named by hand, and the list was not what
a capture loads: it never named `gap_run.py`, which `capture_bound.py` imports. `reader_protocol.py` now reads the list from
the source (`closure`, `instruments_record`): the acquisition's programs (`ENTRIES`) and every file of this folder and of the
pipeline they import, by syntax, inside functions too, with each package's `__init__.py`; then the card lock's wrapper and the
recorder by name. `reader_protocol_roles.py` takes form 3's the same way (the workers capture in place of `own_text.py`, with
its engine, its reading and the workers recorder). As the tree stands that is 41 files and two for form 2, and 44 files and
three for form 3. They include Astra's joint-state branch: `capture_bound.py` as she changed it, `functional_capture.py`,
`functional_install.py`, `functional_reference.py`, `functional_layout.py` and `functional_reading.py`.

Three things a reader of a protocol should know:

- **The gate is named and not bound.** `run_declaration.py` is bound. The files it reaches under `pipeline/research_guard/`
  (`__init__.py`, `input_archives.py`, `verify_round2.py`) are written into the protocol with their sha256 under
  `reached_and_not_bound`, and what they import is not followed. They decide whether a capture may start and read no model and
  no trace, so a new rule in the verifier does not make a new protocol. Whether that boundary stands is Astra's to say.
- **A load the source does not spell stops the protocol.** A module loaded by a computed name cannot be followed by syntax. The
  only such loads a protocol is made with are the capture's and `whole_reading.py`'s named `tube.py` under `TUBE`
  (`own_text.py`:57, `own_text_workers.py`:70, `whole_reading.py`:89): a capture records the `tube.py` it ran and `held`
  refuses another. One anywhere else in the closure raises before a protocol is written.
- **The form's name did not change.** A protocol made now is still `whole-gap-reader-protocol/2` or `/3`; it carries three
  new keys (`instruments_are`, `reached_and_not_bound`, `loaded_by_a_computed_name`), and `instruments_are.since` says the list
  is read from the source.

**The frozen protocols.** All ten files under `reader-protocol/` (`controlled-bf16-1` to `-9` and `-9-qualifying`) name one of
the removed records among their instruments, beside the solver's files as they were before the cut. None can hold a capture
on this tree, and no capture ever ran under `-8` or `-9`. They stay as records. The protocol is made again from the files as
they stand, once, after this commit.

**Checks.** This folder's tests, 148 passed (seven of them new: `test_reader_protocol_instruments.py`, which holds that a file
reached by any import of a named program is bound, that the gate is named and not bound, that a computed load stops the
protocol, and that the tree as it stands binds `gap_run.py`). The pipeline's suite, 364 passed and 27 subtests.

This folder holds 31 programs after the step.

**Whether a protocol still holds the tree** (added 5:48 PM CT November 15, files only):
`reader_protocol.py stands <protocol.json>`, from `code/signal-trace`, names the protocol's instruments that are other bytes now
or are gone, and the files its programs import now that it does not name. Both lists empty: a capture started now runs what the
protocol binds. Asked of `controlled-bf16-8` and `-9` it names eight and nine files (the three the cut changed, the removed
records, `capture_bound.py`, the capture's program and the protocol programs). A protocol is made in 33 s on this machine with
the processor alone (a trial written to the session's scratch, not a protocol of record: every one of the 2,100 turn arrays the
original's exported bytes). **No protocol of record is made yet**: at 5:48 PM CT two of the files a protocol binds were being
edited by their holders, `pipeline/native_records/group_observation.py` in the cleanup's next wave and Astra's joint-state
capture branch, and a protocol frozen on bytes in the middle of an edit holds nothing an hour later. It is made when those are
committed.

**The evaluation's budget no longer budgets the old scorer** (source only, November 15, 2026). `evaluation_budget.py` lost the
compiled target field's bytes and what the old scorer's scoring held on the disk at once (`target_field_gib`,
`scoring_at_once_gib`): it named `evaluation_native.py`, which left in the cleanup's first wave, and sized a field from the
manifest of a compiled target of state values. It budgets what `evaluation.py`'s seven steps make: the readings, the capture and its raw batch.
`evaluation-budget.json`, the record of November 14, stays as it was written, with those two entries. The program was not run
again: no evaluation is admitted, and what a new measure of these readings needs on the disk is its owner's to say.

## A loop on named prompts: a selection of named fitting answers, and a call of fewer answers than workers

Written 6:07 PM CT November 15, 2026. Source, fixtures and files only: no pass, no card.

**What Astra decided** (her answer to this seat, read after 5:51 PM CT November 15): the loop's acquisition is form 3, the
generation's own states with no second replay; the gate's files are named in a protocol with their hashes and not bound, while
`run_declaration.py` stays bound and runs the verifier as it stands; the unchanged recorder and reading qualification is
reused, with no second native qualification; the loop's final scope is one prompt, bound before the run by an explicit `CASES`
manifest; and the protocol is frozen only after her source handoff, her capture admission being still in the making.

**What stood in the way, in this seat's readers.** `cases.py` knew one role, `no-fit`, the whole complement of the fitting
answers, and refused any manifest that names a fitting answer. The capture, the kept reading and `reader_protocol.make` all take
their selection through it, so a one-prompt manifest was refused three times over before her capture admission was asked
anything. And a protocol asked for a control read on its own selection, and form 3 for a qualification of its own count.

**What was built.**

- `cases.py`: a second role, `fitting-named`, fitting answers named one by one
  (`python cases.py make <cohort run> fitting-named <out.json> <id> [<id> ...]`). Its answers are chosen, and the manifest says
  by whom: the declaration that names it by sha256. It refuses a name that is not a fitting answer, so no answer that no fit uses
  can enter under it, and a manifest that names none. `no-fit` is as it was: whole, taking no names.
- `reader_protocol.make` under `CASES=<such a manifest>`: the control of the thirty serves, with no pass. Each named answer must
  be one of the control's, with the cohort's prompt, recorded text and steps, and its turns are made again from the control's
  own rows and held to the export bit for bit. The protocol's `cases` is the manifest's selection, so `held` refuses a capture or
  a reading of the thirty, or of another manifest; `original_control.serves_a_named_selection` says which answers and how. No
  other selection borrows a control: an evaluation's answers still need one read on them.
- Form 3 binds the integrated qualification as the one of the control's thirty answers that contain the named ones, and
  `integrated_qualification.of` says that no run qualified the generation on the named answers alone.
- **A call of fewer answers than workers was refused by this seat's own account, after the call had run.**
  `generation_engine.ended` asked every one of the three contexts for its schedulers' totals. But every worker makes its context
  before it asks for a case (`dependencies/recorder-workers/observe_stream_workers.cpp`:473 to 485), and a scheduler says its
  totals only after it computed a split (`recorder-kept/ggml-backend-eval-kept.patch`:255, `eval_kept_splits > 0`). So in a
  call of one answer two contexts take nothing, compute nothing and leave no line, and the call was counted as one whose
  contexts left no account. The contexts held to an account are now the ones that took a case, in `ended` and in `held`; a
  context that took a case and left none, and totals from more contexts than took a case, still end the capture. The integrated
  qualification's own three calls agree with the rule (ten answers each, three workers took them, nine totals).

**Trials, written to the session's scratch and not of record.** A manifest naming one fitting answer and a form 3 protocol
under it, made in 8 s: one answer, 70 of 70 turn arrays the original's exported bytes, 46 instruments. A form 3 protocol of the
thirty in 17 s: 2,100 of 2,100.

**Not run, and not known.** No capture of one answer has run. The first one is the first time a call of fewer answers than
workers runs on a card: three contexts are still made (three kept contexts held 7,878 MiB in
`runs/recorder-workers-original/result.json`) and one of them writes the answer.

**Checks.** This folder's tests, 162 passed. Two assertions of `test_generation_qualification.py` pinned a call of two answers
that left three contexts' totals, a shape no call leaves; its fixture now leaves the totals of the contexts that took an answer.

## The reader protocol made again: `controlled-bf16-10`, form 3, of the one answer Astra's manifest names

Written 6:17 PM CT November 15, 2026. Files only: no pass, no card.

**What it is.** `reader-protocol/controlled-bf16-10.json`, sha256 `1f4bf3be17b0741b08c9f2ac77cf2911caf40790987306e5efb6aaaa7fdfbc2f`,
made at 6:13:48 PM CT on Astra's word, once her capture source was final (b30c0ed9, 6:12:09 PM CT). Form 3: the three-worker
kept generation, a step's states kept from that generation's own streams, no second replay. It is made under her manifest,
`records/retrospectives/2026-11-15-single-reference-training-cases.json` (sha256 `9974fedb7e19…`; role `fitting-named`, cohort
`controlled-bf16-1`, the one answer `discovery-10-channel-h1`), and it admits a changed state's capture and reading. It binds 46
files, the import closure read from the source, and names the gate's three with their sha256; every one of them was committed
when it was made. `reader_protocol.py stands` on it: no bound file of other bytes or gone, nothing imported and not named.
The control of the thirty serves the named answer with no pass: it is held by its prompt, text and steps, and its turns are made
again from the control's own rows, 70 of 70 arrays and none differing. A run that uses the protocol sets `CASES` to the manifest
for every reader; the protocol's `acquisition` names the three commands.

**What it binds against `-9`** (`controlled-bf16-10-against-9.json`, definition by definition; a file `-9` bound is compared
with the bytes it bound, a file it did not name with the tree as `-9` was frozen on it, c9035167). Of the 48 files either names:

- **22 are the same bytes.** Twelve that `-9` bound: both recorders, `export_stream_traces.py`, `own_export.py`,
  `states_reading.py`, `states_reading_roles.py`, `whole_reading.py`, `recorder_engine.py`, `reading_identity.py`,
  `states_conditional_check.py` and the lock's two files. Ten it never named and a capture always imported, `gap_state.py`,
  `native_trace.py` and `run_declaration.py` among them.
- **19 are other bytes.** In the pipeline every kept definition is the same bytes but four: the removed definitions are the
  cleanup's (`tube.py` 60, `ladder.py` 35, `gap_targets.py` 21, `context_targets.py` 17, `gap_run.py` 15, `a2_pattern.py` 13,
  `group_observation.py` 12, `a2.py` 9, `whole_context_score.py` 3, `context_score.py` 2, `complete_overlap.py` 2), and the four
  changed are `run_a_observation.py`'s `command_of`, `run` and command line (the second recorder a caller could name is gone; no
  reader named one) and `fed_replay.py`'s `record` (the paired-row movement left). In this folder: `capture_bound.admitted`
  (Astra's route to `functional_capture.admitted` for a declaration with `functional_loop`; a declaration without it takes the
  lines it took), `cases.py` (the role `fitting-named`), `generation_engine.py` (`ended` and `held` count the contexts that took
  a case), `own_text_workers.py` (its description only, 54 definitions the same bytes), and the two protocol makers, which are
  no part of a pass.
- **5 are new:** Astra's joint-state branch, `functional_capture.py`, `functional_install.py`, `functional_layout.py`,
  `functional_reading.py`, `functional_reference.py`.
- **2 that `-9` bound are gone:** `states_conditional.py` and `states_correspondence_roles.py`, which only bound a reading to a
  state-value target for the old scorer.

The four execution records (`execution`, `generation_execution`, `reading_execution`, `the_executions`) and every field of the
integrated qualification but `of` are the same in both protocols. `reference_sha256` differs, by the selection as far as was checked:
a protocol of the thirty made to scratch on this tree at 6:04 PM CT had `-9`'s.

**What the reuse of the qualification does not cover.** Astra reuses the recorder's and the reading's qualification
(`runs/generation-workers-original`) with no second native qualification, and accepts the subset qualification and the count of
the contexts that took a case (her words read after 5:51 PM CT and at 6:13 PM CT). The protocol says of what the qualification is: the
thirty that contain the named answer. Not run, and so not known:

- no capture and no reading has run under `-10`, of one answer or of thirty;
- no call of one answer on three workers has run: two contexts are made and take no case, and the loop's first native capture
  is the first time that arrangement runs;
- Astra's joint-state branch has run on fixtures and tests only;
- the files `-9` did not name are compared with the tree at its commit, not with what the qualification's processes loaded at
  11:59 AM CT: no record holds that.

**Checks.** This folder's tests, 162 passed at 6:16 PM CT with the protocol in place.

## The capture that only acquires: form 4, `controlled-bf16-11` (6:53 PM CT November 15; files only, no capture run)

Written 6:54 PM CT November 15, 2026. Sources: commits abeeca29, 75033fae and 695f67f7; the two protocol files and what
`native_events_protocol.py make` printed at 6:52:57 and 6:53:00 PM CT; this folder's tests at 6:51 PM CT.

**`controlled-bf16-10` is withdrawn for the loop, and why.** Marcos's sweep, relayed at 6:22 PM CT November 15: as this seat
understands him, the events of the wrong run are paired with the events of the reference, at every step, by the relation of
computation the loss uses, and nothing in that chain is made of a turn vector or a token id. The capture `-10` bound, frozen by this
seat at 6:13 PM CT, held two things that are neither:

1. **The registration.** `laid_on_original` laid every regenerated answer on its OWN original recording by
   `complete_overlap.distance_path` on the unit turn vectors and wrote `pairs-original.npy`. The pairing was made of turn vectors, and
   it paired the wrong answer with its own past, not with the reference: the construction stopped at 8:55 AM CT that day.
2. **A token comparison deciding a pairing.** `same_inputs`: whether a new answer's token list equalled the recorded one decided
   whether it was registered "by the identity of its inputs" or by the computation.

This seat wrote the caution about the first that morning (this page, line 2582, and the board) and froze it into `-10` anyway:
the error is this seat's. No capture ever ran under `-10`.

**What form 4 is** (`whole-gap-reader-protocol/4`, `acquisition: native-events-only`). A functional loop's capture is
`native_events.py`. Each named prompt is given to the model, greedy, to its limit, by the three-worker recorder on the kept scheduler
(the recorder call is own_text_workers.py's: same recorder, environment and arguments). What is kept, per answer: `answer.json`
(the token ids the model itself chose, how the answer ended, its limit, its prompt's sha256) and the recorder's own stream, moved
untouched to `KEEP_RAW`. `states_reading_roles.py` then keeps every merge's state from those streams, with no second pass, and
refuses a functional loop's capture that does not say `native-events-only` or that carries `registration`,
`original_computation` or `reference_sha256`. The declaration says so first: `capture_bound.native_events` refuses a
`functional_loop` whose `acquisition` is not `native-events-only`, before anything is generated. Pairing events is the loop's own
work, by its own relation, after the capture: nothing here pairs.

**What left the loop's path.** No `pairs-original.npy`, no registration record, no turn export, no array of the original opened, no
recorded answer loaded. The registration (`laid_on_original`, `original_sources`, `unchanged_since`) moved to
`capture_registration.py`, each definition the same bytes; forms 1 to 3 (`own_text.py`, `own_text_workers.py`,
`reader_protocol.py`) still import it and the loop's programs do not. `ending` moved from `whole_reading.py` to
`reading_identity.py`, so the reading no longer imports the replay program. The capture, the reading and the protocol's maker do not
call `ladder.cohort`, which hands every member the original answer's token ids and length: they take the cohort's membership
(`cases.cohort`: each answer's id and outcome group, in the cohort's order, from the member list of `ladder.CORRESPONDENCE`; none of
that run's arrays) and each NAMED answer's prompt and limit (`cases.prompts`, from `runs/wide-overlap/members.json`, whose
`recorded_tokens` are not taken). The closure of the eight entry programs, read from the source, is 33 files, with no
`complete_overlap.py`, `own_export.py`, `own_text*.py`, `capture_registration.py`, `whole_reading.py`, `overlap_step.py` or
`overlap_reading.py` in it and no module loaded by a computed name.

**The token ids that remain, each by what it decides.** Counted from the source of the three programs of this folder on the path:

- the prompt: the ids the model is run on, hashed per answer and as one (`prompts`, `prompts_sha256`);
- the answer's own history: the ids the model chose, kept in `answer.json` and hashed in the reading's row map;
- inside ONE run of the recorder: that it was given the prompt's count of tokens (`native_events.answer_of`), that no token was
  supplied to it (`teacher_token_count`, `conditioned_continuation`), and at each step whether the selected token is the model's own
  first choice (`greedy_token != selected`, which is what "the whole answer is this model's own" rests on);
- between the two records of ONE run: the reading holds its rows to the capture's `answer.json` by sha256 and by the same
  hashes of prompt and history (`reading_identity.py`:167).

None compares an answer with another answer, with the reference or with the original recording, and none decides a pairing or
enters a measure. The outcome group of an answer is read as membership only: it is what the fitting split (`tube.picks`) and a case
manifest's identity are made from, and the reading writes it beside each answer's id as a label. **Not traced by this seat:** every
function of the 25 pipeline files in the closure. They are bound because a program on the path imports the file; what this folder's
programs take from them by name is `BACKENDS`, `ROOT`, `check_native_run`, `MARGIN_GIB`, `ROUND_TWO_RUNS`, `tube.ORIGINAL_MODEL`,
`tube.WIDE`, `tube.picks`, `ladder.CORRESPONDENCE` and `ladder.INPUTS`, and `capture_bound.py` takes Astra's
`functional_capture.admitted`. Bound and not called from here: the turn exporter (`export_stream_traces.py`), the baseline's
`tokens_equal` (`run_a_baseline.py`) and the fed replay's checks (`fed_replay.py`). What the kept pipeline files do with tokens and
turn vectors is Fable's record (`records/development/2026-11-15-tree-cleanup/README.md`, commit 4a9d11bc).

**The two protocols.** Both name the same 36 instruments by sha256: the 33 files of the closure, `after-the-others.sh`, the installed
recorder and the workers recorder. 35 are the bytes of commit 695f67f7; the workers recorder is a build git does not track, bound by
its sha256 as in `-9`. The gate's three files are named and not bound, as before; the three makers
(`native_events_protocol.py`, `reader_protocol.py`, `reader_protocol_roles.py`) are named as they stood and not bound.

| Protocol | sha256 | Made | Astra's manifest | Answers |
|---|---|---|---|---|
| `reader-protocol/controlled-bf16-11.json` | `5a01da55f5f60c93fc0a800759c4f5073278a258fb5445d1ed62119bf96929d6` | 6:52:57 PM CT | `records/retrospectives/2026-11-15-single-reference-training-cases.json` (`9974fedb7e19…`) | `discovery-10-channel-h1` |
| `reader-protocol/controlled-bf16-11-post-closure.json` | `ddfed1cc4f15701aade4348b1bfd3941a23b6fb00bdd38f7cfd85b296d893f5b` | 6:53:00 PM CT | `records/retrospectives/2026-11-15-single-reference-post-closure-cases.json` (`a3d3b0ecf9e8…`) | the 29 other fitting answers, 14 wrong and 15 correct |

A declaration binds one by `functional_loop.reader_protocol = {record, sha256}`, says `functional_loop.acquisition =
'native-events-only'`, and names the manifest as `functional_loop.cases`; `functional_capture.admitted` holds a capture to all three.
The commands, from `code/signal-trace`, card 0 under its lock for the first:

    PASS=<n> KEEP_RAW=<folder in the run> CASES=<the manifest> native_events.py <declared run> controlled-bf16-1 <the state's model> <the history the declaration names> <declared run>/loop/<n>/capture/own
    PASS=<n> ROLLOUTS=<that KEEP_RAW folder> CASES=<the manifest> states_reading_roles.py <declared run> controlled-bf16-1 <the state's model> <declared run>/loop/<n>/capture
    native_events_protocol.py held <the protocol> <the capture folder> <the reading folder>

The capture's `text.json` no longer has `registration`, `original_computation`, `reference_sha256`, `tube_sha256` or
`the_text_before`; it has `acquisition`, `prompts`, `prompts_sha256`, `answers` (steps, ending, limit) and the generation's
execution. `held` refuses a capture that carries a pairing key or any `pairs` file.

**What is not run and not established.**

- No capture has run under either protocol, and `native_events.py` has run on made-up folders with a stand-in process only (20
  tests): that its call writes the bits the control holds rests on `generation-workers-original`, which ran own_text_workers.py's
  call of the same recorder on the thirty answers, not this program.
- No call of one answer on three workers has run on an engine (two contexts take no case).
- Nothing here establishes that two events are the same computation, or admits a measure: both are the loop's.
- The original control (`whole-states-original/reading`) is named as the execution the generation was qualified against; its
  record and row map are read, to hold each named prompt to the prompt it was read on; no array of it, and nothing of it is paired.

**The older forms.** `-9` and the forms before it bind files that changed today and hold no capture on the tree as it stands.
`own_text.py`, `own_text_workers.py`, `own_export.py`, `capture_registration.py`, `reader_protocol.py`'s forms 1 to 3 and the two
direct-overlap instruments (`overlap_step.py`, `overlap_reading.py`, which pair by `distance_path` on unit turn vectors) are still
in the tree, outside form 4's closure; whether they leave is this seat's next cleanup pass.

**Checks.** This folder's tests, 182 passed at 6:51 PM CT; the pipeline's suite, 339 passed and 24 subtests at 6:44 PM CT (before
75033fae, which changed no file a pipeline test imports); the verifier clean at each commit.

## The programs that paired by turn vectors or by token identity leave the tree (7:19 PM CT November 15; files only)

Written 7:19 PM CT November 15, 2026. Sources: `retired-pairing-programs.json` (each file's sha256 and lines as commit c4882734
holds them, and why it left), the verifier's output, this folder's tests, and the Eli subagent's audit of that evening
(`records/retrospectives/2026-11-15-token-audit-eli.md`).

**Why.** Marcos's direction relayed at 4:11 PM CT November 15 put the tree's cleanup with Fable and this seat, since what a
withdrawn construction leaves in the tree ends up steering the work; his sweep relayed at 6:22 PM CT says a pairing is made by the
relation of computation the loss uses, and by nothing made of a turn vector or a token id. With form 4 in place nothing of the loop
takes the programs below, and the audit found at 6:52 PM CT that the loop's acquisition still called the old capture (its finding 5;
`pair_native.py` calls `native_events.py` since).

**What left whole: sixteen files, 2,442 lines.**

- the registration, `capture_registration.py`, and `own_export.py`, the turn export the two old captures made for it;
- the direct overlap and what read it: `overlap_step.py`, `overlap_reading.py`, `overlap_depth.py`, and `turns_reading.py`,
  which only the second took. They laid a model's answers on the fifteen original correct answers by
  `complete_overlap.distance_path` on unit turn vectors and read a pooled departure from that pairing;
- `pairs_coverage.py`, a reading of `pairs-original.npy` records, which no capture writes any more;
- `generation_qualification.py`, the program of the finished `generation-workers-original`: its capture step ran
  `own_text_workers.py`. The run, its declaration, result and comparison stay, and form 4 still rests on them;
- `evaluation.py` and `evaluation_budget.py`, the old evaluation on an explicit list of answers, whose capture step ran
  `own_text.py`; `evaluation-budget.json` stays. An evaluation after closure is the loop owner's (`pair_evaluation.py`);
- their five test files, and the speed review's `own-export/registration_by_computation.py`.

**The two old captures are stubs, because the verifier refused their removal.** With `own_text.py` and `own_text_workers.py`
gone the verifier named five violations: the declarations of `recorder-deferred-original`, `recorder-kept-original`,
`recorder-workers-original`, `recorders-beside-original` and `generation-workers-original` cite one of them as the source of a
parameter (ten answers a recorder call). A declaration is not revised, so each file stays as a text that holds which declarations
cite it, the line of its last form they rest on (`git show c4882734:<path>`, lines 214 and 227) and why the program left, and
nothing that runs: the form `correction.py`, `field.py` and `mixture.py` have in the pipeline.

**The two makers are cut to what form 4 takes.** `reader_protocol.py` keeps `digest`, `imported`, `closure`, `instruments`,
`instruments_record` and `stands` (359 lines to 130): forms 1 and 2 (`make`, `held`, `turns_made_again`) left, the entries are the
caller's, and a module loaded by a computed name always stops a protocol, since the exception was for the captures that put a
named `tube.py` in the module's place. `reader_protocol_roles.py` keeps `integrated`, the same bytes (212 lines to 35): form 3 left.
`test_capture_bound.py` lost the four tests of the registration.

**What stays, and one cut that waits.** `torch_forward.py` is untouched, with its `registered` (the same pairing, in torch) and its
module-level import of `complete_overlap`: the loop's declarations bind that file by sha256 (`single_reference` loads the declared
forward by a computed name and uses only `Model`; the audit, "Checked and found clean"), so the cut is made on Astra's word, when
the binding can move, and Fable's cut of `distance_path` and `unit_steps` waits on it. `states_reading.py`, `whole_reading.py` and
`prompt_turns.py` stay as raw extraction; `states_conditional_check.py` stays as a check of one acquisition against another, bit
for bit. The programs of the four recorder qualifications in the speed review name `own_text.py` in comments only.
`capture_bound.py`, `generation_engine.py`, `recorder_engine.py`, `native_events.py` and `states_reading_roles.py` still name the
old captures in their descriptions: they are bound by form 4 and were not edited.

**Form 4 holds the tree after the pass.** `reader_protocol.py stands` answers `controlled-bf16-11` and its post-closure twin with
nothing of other bytes or gone and nothing imported and not named. The protocols of forms 1 to 3 name files that are gone and hold
no capture, as before.

**What the audit says of this seat's files, beside finding 5** (its lines there): `native_events.py` and
`native_events_protocol.py held` are clean; the fields the loop's stop reads (`ended_with_an_end_token` and the own-step fields,
`states_reading_roles.py`:184) are facts of the acquisition this seat records, and what the loop does on them is its owner's
(finding 4); a reading's `group` label is the cohort's recorded outcome flag, the removed grader's verdict on the ORIGINAL answer,
written as membership (finding 7); until 7:32 PM CT this line said nothing of the loop reads it, which was wrong: the ORIGINAL reading's label is what `functional_reference.py`:53-61 takes as who is correct and who is wrong, and `pair_native.py`:110 holds the pair to it; the label a new capture's reading writes again is the one nothing reads. It did not read through `fed_replay.py`,
`export_stream_traces.py`, `run_a_baseline.py`, `group_observation.py` or `native_trace.py`, which form 4 binds.

**Checks.** This folder's tests, 129 passed at 7:05 PM CT on the cut tree and again with the stubs in place (53 tests left with
their programs); the verifier, five violations before the stubs and clean after; the pipeline tests that load this folder's
programs (`test_functional_capture`, `test_model_trace_response`, `test_after_the_others`, `test_compact_dose_archive`), 23 passed
at 7:19 PM CT. The pipeline's whole suite, started at 7:07 PM CT, was still running when this was written; this pass changed no
file it loads. This folder now holds 20 programs and the two stubs. Added 7:22 PM CT: this seat stopped that run after 14 minutes
and ran the suite without the four test files being edited in the tree at the time (`test_pair_evaluation`, `test_pair_loop`,
`test_pair_native`, `test_joint_inverse`): 329 passed and 24 subtests in 25 s at 7:22 PM CT. Which of the four held the first run is
not established.

## `torch_forward.registered` left: no program of this folder pairs by turn vectors (7:28 PM CT November 15)

The section above left one cut waiting: `torch_forward.registered`, the registration's rule copied into torch (each step's turns at
all merges as one unit vector, the product of two answers' steps, `complete_overlap.distance_path` over it). It waited because
seven declarations name `torch_forward.py` by sha256 and the loop's owner had to say when that binding could move. Astra said so in
a message read at 7:26 PM CT: the one-reference construction and its compilation are withdrawn after Eli's finding on the Taylor
remainder, no compilation or fixture process of it ran at her check of 7:25:56 PM CT, the earlier declarations keep their hashes
as historical provenance and nothing is requalified, and no replacement pair or native loop is admitted.

**What left.** From `torch_forward.py` (601 lines to 564; sha256 `ab0abc44eaa3…` to `64c4bae55812…`): `registered` whole; in
`written`, the part that laid each written text on the original's exported turns and held a changed text's path to
`distance_path(unit_steps(...) @ unit_steps(...).T)`, with the second import of `complete_overlap`; the `laid_on_the_original` key
of the command's printed summary; the docstring's sentence on it. From `test_torch_forward.py` (255 lines to 240; `dc40f56afcdb…`
to `45099653dbb6…`): the one test of `registered`. The bytes before are commit 73cb6b25's.

**What stays, and what it is.** `Model` (the forward, the scores, `write`), `turns`, `check`, `backward_once` and `written`'s two
other parts. `written` is the torch writing held to the recorder's: it lays the token list one acquisition wrote beside the other's
of the same model, and Astra's decision of 1:55 PM CT keeps it as acquisition evidence; it measures no progress and nothing reads
it for a loop. The record it once wrote, `torch-forward/write-original.json`, keeps its `laid_on_the_original` table as written.

**What this changes for others.** Outside `studies/`, the only programs that import `complete_overlap` or call `distance_path` or
`unit_steps` are `complete_overlap.py` itself (`align`) and `tests/test_complete_overlap.py`, counted from git's list of tracked
programs; this folder's `test_native_events.py` names the module only to refuse it. So Fable's cut of the two functions no longer
waits on this seat. The seven declarations that name `torch_forward.py` (`envelope-loop-balanced-torch`,
`envelope-loop-plain-torch`, `functional-baseline-original`, `plain-first-step-state`, `relation-reference-original`,
`shared-relation-loop-torch`, `single-reference-original`) hold the earlier bytes' sha256 and are not rewritten. Form 4's two
protocols never bound this file, and `reader_protocol.py stands` answered both with nothing changed after the cut.

**Checks.** This folder's tests: 128 passed at 7:28 PM CT (one test left). The pipeline tests that load the forward or the capture's
admission (`test_model_trace_response`, `test_complete_overlap`, `test_functional_capture`): 17 passed at 7:28 PM CT. The verifier:
no drift.

## The removals of 7:20 and 7:28 PM are undone: the programs are back as they were (7:41 PM CT November 15)

Marcos's word, relayed by claude-research at 7:37 PM CT: as this seat understands him, the right code was in this tree and today's
removals broke it, and his three papers and the loss document are his statement of the method (they are on this machine,
`~/.cache/claude-notes/papers/`). In the same message claude-research withdrew the premise its 6:22 PM sweep had given, that a
pairing by turn vectors is made of tokens (its reading, it says, not his), quoted the path-edit paper's Section 4 for the overlap
(two steps are alike when their turns at all 105 merges point the same way, no token, number or word used), and asked that no
program be removed until he says which code stays and that the overlap come back from history as it was.

**This seat's part, said plainly.** The two sections above removed, on that premise, the programs that lay a model's rewritten
answers on the correct answers by the turns at all 105 merges (`overlap_step.py`, `overlap_reading.py`, `turns_reading.py`), which
is the overlap that paper defines as quoted, with the captures of forms 1 to 3, their registration, their protocol makers and
`torch_forward.registered`. This seat had not read the papers, and did not hold the sweep's premise against his own statement of
the method before it removed. That is the mistake of `controlled-bf16-10` (frozen at 6:13 PM CT) again: acting on a description
without checking what it describes.

**What is back.** The 22 files commit c231714a removed or changed, as its parent holds them: `capture_registration.py`,
`evaluation.py`, `evaluation_budget.py`, `generation_qualification.py`, `overlap_depth.py`, `overlap_reading.py`, `overlap_step.py`,
`own_export.py`, `own_text.py`, `own_text_workers.py`, `pairs_coverage.py`, `reader_protocol.py`, `reader_protocol_roles.py`,
`turns_reading.py`, the tests `test_capture_bound.py`, `test_evaluation.py`, `test_generation_qualification.py`,
`test_overlap_step.py`, `test_pairs_coverage.py`, `test_reader_protocol_instruments.py`, `test_turns_reading.py`, and the speed
review's `own-export/registration_by_computation.py`; and `torch_forward.py` with `test_torch_forward.py` as commit 73cb6b25 holds
them. Each was written from history and held to history's blob: 24 of 24 the same bytes, nothing changed in any.
`torch_forward.py` is sha256 `ab0abc44eaa3…` again, the hash seven declarations hold; `overlap_step.py` is `e822f9912df6…`.

**What was not touched.** `native_events.py`, `native_events_protocol.py`, `cases.py` and form 4's two protocols stay, and
`reader_protocol.py stands` answers both with nothing changed. The two sections above stay as the record of what was done;
`retired-pairing-programs.json` says under `restored` that its removals are undone. No reader protocol of forms 1 to 3 holds the
tree as it stands (`controlled-bf16-10` finds nine of its files with other bytes, from the changes of 6:44 to 6:52 PM CT that came
before c231714a and are not undone here), so a capture with a registration is admitted by no standing protocol; none is made here.

**Not brought back by this pass, and where each is.** This seat's earlier removals of November 15, each program with its sha256
and the commit that holds it: `retired-progress-readers.json`, `retired-state-value-readers.json`,
`retired-old-solver-readers.json` and `retired-joint-step.json` in this folder, the speed review's `retired-run-readers.json`, and
`records/development/2026-11-13-rented-solve/retired-solve-senders.json`. Which of them comes back is his to say. The old solver
and `pipeline/overlap` are Fable's.

**Checks.** This folder's tests: 182 passed at 7:42 PM CT (the 129 of the cut tree and the 53 that had left with their
programs). `test_model_trace_response`, `test_complete_overlap`, `test_functional_capture`, `test_after_the_others` and
`test_compact_dose_archive`: 29 passed at 7:42 PM CT. The verifier: no drift.

## This seat's other removals of November 15, held against the papers (7:49 PM CT; nothing restored by this section)

Read at 7:45 to 7:48 PM CT, from the files Marcos sent: the path-edit paper whole, the trace-derived guide whole, and the loss
document's text (its equations are objects the text extraction does not carry). What follows is this seat's reading, for him to
decide from; which code stays is his to say.

**What the papers define, where this seat's instruments touch it.** The trace is the turn of the stream's direction at each of
the 105 merges of a token (guide 4.1 and 4.2). Answers are laid on one another by dynamic time warping in which two steps are
alike when their turns at all 105 merges point the same way, using no token, number or word (path edit 4.1). An edited model is
read by recording its own answers, laying them on the same places by their computation, and measuring each against the ORIGINAL
patterns, 0 at the wrong pattern and 1 at the correct one, with what lies to the side beside it (path edit 6.3, Figure 6, Table
3). The path edit aims at the state after each matrix's merge and not at the turn, for two measured reasons (path edit 5.1), and
the loss document says the two observables are not interchangeable and must be named. A reading on given text did not predict the
model's own answers (path edit 6.4; guide 9.3).

| What left, and when | Where it is | Against the papers |
|---|---|---|
| The direct overlap, the captures of forms 1 to 3 with their registration, `torch_forward.registered` (7:20 and 7:28 PM CT) | back in the tree since 7:44 PM CT (1e053266) | the pairing rule of path edit 4.1: dynamic time warping in which two steps are alike by their turns at all 105 merges. The paper lays every answer on one axis, the median-length answer's 480 places; `overlap_step.py` lays each answer on each of the fifteen original correct answers by that rule, and the captures' registration lays a rewritten answer on its own original recording by it. That last is the form of November 15; until then a capture registered an edited model's answers through the frozen axis of the first overlap (`tube.registered`, the solver's, as `capture_bound.py`'s head says), which is the laying on the places that 6.3 describes |
| The readings of a model by state values (3:18 to 3:36 PM CT, on Astra's decision): `members.py`, `rung_map.py`, `fed_path.py`, the speed review's `as-we-go/along.py`, with `whole_model_l.py`, `conditional_members.py` and the as-we-go readers | `git show 5a354bf2:<path>`; `retired-state-value-readers.json` | `members.py`'s `along` and `beside`, and `rung_map.py`, ARE the measurement of path edit 6.3 and Table 3 (each answer's coordinate along the gap between the two original patterns, and what lies off that line). They read the state after each matrix, the paper's choice; his direction relayed at 3:46 AM CT that day withdrew the constructions on the state's values (`STATE.md`; claude-research's reading of it at 7:37 PM CT: the target on the turn and not on the state after the merge). `whole_model_l.py` and `conditional_members.py` are later constructions and not in the two papers read here. The as-we-go readers are the reading made during a build (6.4) |
| The readings a token, a word or an answer decided (about 3:01 PM CT, on Astra's request, after his question relayed at 1:44 PM CT) | `git show efdfc802:<path>`; `retired-progress-readers.json` | `pipeline/decisions/answers.py` is the judge of the papers' Scoring, which counted 29 of 30, 17 complete and 7 of 21. His direction of the evening of November 14 is that tokens are not the judge. The sign-row readers are not in the papers |
| What only the old solver, the old scorer or the stopped loops used (4:39 to 5:42 PM CT, on the 4:11 PM CT criterion): 29 readers and timers, 10 senders of a solve to a rented box, 5 readers of a run's time, 6 records that bound a reading to a state-value target, the late-start recorder (5 files), the fused writing (6 files) | `570673c8`, `64bca1ab`, `fdde25a0`, `eb569efa`; `retired-old-solver-readers.json`, the rented solve's `retired-solve-senders.json`, the speed review's `retired-run-readers.json`, `retired-joint-step.json` | instruments around the solver of path edit 5.3 to 5.6, most of them this seat's work on its speed; none is in the papers. None runs without the solver's whole modules. One is taken by the solver's later form: since November 14 the path-order sweep read each matrix's rows after a write through the late-start recorder, and `run_a_observation.run` refuses a late-start case since 5:42 PM CT, so a sweep restored as it stood that afternoon needs those five files back |

**What this means for a restore.** `members.py`, `rung_map.py` and the solver's instruments take the solver's own functions
(`tube.py` whole, through `TUBE=`), so they load only once the solver is back, and they fit the solver as it stood when each left:
a seat-by-seat restore to different commits leaves pieces that do not fit each other. One restore point for every program, chosen
by him or by the seats together, avoids that. This seat brings its files back to whatever point is chosen, as they were.

## This seat's removals after 4:22 PM CT are undone: 78 files back as commit 69b7d13f holds them (8:03 PM CT November 15)

**The point.** `69b7d13f` (4:22:25 PM CT November 15) is the last commit before either seat's first program removal under the
tree's cleanup: this seat's `3a884b51` at 4:34 PM CT, Fable's `abaaa932` at 4:49 PM CT. Fable named it at about 7:55 PM CT and wrote
the solver's side back to it (`records/development/2026-11-15-tree-cleanup/README.md`, "The programs are back"). This seat's side
followed in `e7a773e3` at 8:03:03 PM CT.

**What is back, counted from git's listing of the commits and not from the retirement records.** 78 files, each the bytes and the
mode git holds at the point:

| Commit, time | Deleted there, back now | What |
|---|---|---|
| `3a884b51`, 4:34 PM CT | 6 | `prompt_overlap.py` and five programs of the speed review that only the removed drivers used |
| `16c99826`, 4:41 PM CT | 29 | the rows readers (`rows_reading.py`, `rows_reading_check.py`, `rows_from_states.py`), `reference_turns.py`, `prompt_features.py`, `export_own.py`, `replay_cost.py`, two tests, and the speed review's 20 programs on the old solver (its arithmetic, the cost of a set, timers and benches, the late start's maker and check, the pass split, `as-we-go/candidate.py`) |
| `fd2c31c7`, 5:13 PM CT | 10 | the rented solve's senders (`bundle.py`, `fits_16.py`, `small_side.py`, `box-start.sh`, `next-link.sh`, `queue-links.sh`, `pull-solve.sh`, `box-pause.sh`, `pause-for-rollouts.sh`) and `tests/test_rented_solve_bundle.py` |
| `0b933275`, 5:21 PM CT | 5 | the readers of a run's time (`sweep_profile.py`, `judge_profile.py`, `visit_stage_times.py`, `correction-solves/solve_time.py`, `build-as-it-ran/where_the_time_went.py`) |
| `cca5253e`, 5:42 PM CT | 17 | `states_correspondence.py`, `states_correspondence_roles.py`, `states_conditional.py` and their tests; the late-start recorder (`observe-stream-late`, `observe_stream.late.cpp`, its identity record, `pipeline/acquisition/late_start.py`, `tests/test_late_start.py`); the fused writing's six files |

Every one of the 67 had at its removal the same bytes it has at the point. Eleven more files were rewritten after the point only to
cut, and are set back to the point's bytes: `pipeline/acquisition/run_a_observation.py` (a pass names the late recorder again:
`STREAM_RECORDERS`, `program`), `recorder/STREAM-RECORDER.md`, `evaluation.py` and `test_evaluation.py` (the two record steps and
`target`), `evaluation_budget.py` (the old scorer's field), `fixed_input_movement.py` (one line of its head), the rented solve's
`box-engine.sh` and `box-link.sh`, and the box image's `Dockerfile`, `box-image.sh` and `README.md`.

**How.** The 11 had no uncommitted change and a copy of each is in this session's scratch. Each file was written from git's blob at
the point with the mode git holds there, and its `git hash-object` is the point's blob, 78 of 78. `git checkout <commit> -- <paths>`
is refused by the destructive-commands guard and was not used.

**What was not set back, and what that leaves.** What this seat built after the point and did not cut stays as it is: the import
closure of a reader protocol and `stands` (`reader_protocol.py`, `reader_protocol_roles.py`), the named fitting answers (`cases.py`,
`generation_engine.py`), the registration moved out of `capture_bound.py` into `capture_registration.py` with the same bytes,
`ending` moved from `whole_reading.py` to `reading_identity.py`, and form 4's programs. Three things of the point are therefore not
as they were, and are said here so nobody finds them by accident:
- The two makers no longer name the three record programs (`reader_protocol.py`'s `ENTRIES`, `reader_protocol_roles.py`'s `NEW`):
  a protocol of forms 1 to 3 made now would not bind `states_correspondence.py`, `states_correspondence_roles.py` or
  `states_conditional.py`, which the point's makers did. The three programs run as they did (each calls `held`, which is there).
- `states_reading_roles.py` refuses `recordings` as its fourth argument since 6:51 PM CT (`75033fae`); `states_reading.py` still
  reads the recordings' text.
- A reading kept from a generation no longer names `whole_reading.py` in its `made_with`.
None is changed here: each is decided when a protocol of forms 1 to 3 is asked for.

**Three tests were fitted, not the programs (`9fd4728e`, 8:07 PM CT).** The restored `test_states_correspondence_roles.py` made up a
generation's account of nine schedulers for two taken cases; since `8190354b` (6:11 PM CT) a call is held to the contexts that took
a case, on the measured fact that a worker that took no case leaves no totals, so the made-up account is six. `test_native_events.py`
held `complete_overlap.py` out of form 4's import closure; the whole `tube.py` and `ladder.py` bring it in, and the test beside it
still holds that the capture itself imports no pairing program. `tests/test_recorder_command.py`'s third test held that no other
recorder can be named; it holds the name again.

**Checked.** This folder's tests: 247 passed at 8:04 PM CT. The pipeline's suite with both seats' restores on disk: 819 passed and
105 subtests at 8:06 PM CT, Astra's in-flight `test_pair_native.py` left out. The verifier: no drift at both commits. Nothing ran on
a card.

**Form 4 no longer holds the tree.** Its import closure is 52 files where 33 were bound: `tube.py`, `ladder.py` and `gap_run.py`
whole bring the solver in. `reader-protocol/controlled-bf16-11.json` and its post-closure twin describe the tree of 6:52 PM CT; no
capture ran under either. The protocol is made again when Astra says which acquisition her loop takes.

**The box image.** Its `Dockerfile` and `box-image.sh` stage the solver by default again (`correction`, `tube`, `field`, `gap_run`
and what they import) and check both recorders. The only image on this machine is `wp-box:watch`, built at 4:22:40 PM CT from the
point itself, which the MLflow view runs; no `box` image exists and none was built. The check that a rented card is ours alone
is in both places now: `box-start.sh` holds it again before it sends a solve, and `box-link.sh`, which got it at `fd2c31c7` when
`box-start.sh` left and lost it when the restore set it to the point's bytes, has it back as an addition in the commit after the
restore, since a box that only gets readings never passes through `box-start.sh`. `box-link.sh` is therefore the one restored
program whose bytes are not the point's.

**What stays out.** This seat's removals before the point: the readings by state values (`members.py`, `rung_map.py` and the others,
in git at `5a354bf2`), the readings a token or an answer decided (`efdfc802`), the cuts inside files of 3:36 PM CT
(`fixed_input_movement.read`, `prompt_features.pair`, `fed_replay.py`'s paired row, `own_capture.py`'s `paired` and `moved`), and
the token instruments of the evening of November 14. Which of them return is Marcos's or Astra's to say; the table above holds each
against the papers.

## The loop's acquisition stays the capture that only acquires; form 4 is not made again yet (Astra, read 8:23 PM CT November 15)

Astra's answer to this seat's notice that the two form 4 protocols no longer hold the tree: the new loop's acquisition remains
`native-events-only`, with no registration of an answer against its own old wrong recording. Her original-only compiler needs no
capture, so `controlled-bf16-11` is not frozen again now. The training manifest the frozen one was made under (the one answer
chosen by the prompts' similarity) is withdrawn; she supplies the final one-prompt inventory first and asks for the replacement
then. No capture is admitted. So the tree holds no reader protocol of record, of any form, and this seat makes none until she asks.

## The capture that only acquires, frozen again on the restored tree: `controlled-bf16-12` (8:40 PM CT November 15; files only)

Astra asked for it (read at 8:37 PM CT): the replacement protocols of form 4 against her two finished inventories, binding the
restored import closure, with the original control's and the generation's execution identities kept apart; no capture, no loss
evaluation and no weight write is asked or admitted. Her record of the selection is
`records/retrospectives/2026-11-15-turn-acquisition-scope-astra.md`: the training prompt is the first originally wrong member in
the definition's own order, chosen from membership alone.

| Protocol | sha256 | Made | Manifest (Astra's), sha256 | Answers |
|---|---|---|---|---|
| `reader-protocol/controlled-bf16-12.json` | `e5ec4c7a12d49bcfc48f5e52df5a43a5658e88e474fa821b85813e057c3dcee9` | 8:39:58 PM CT | `2026-11-15-turn-training-cases-astra.json`, `3c15a4ebda86…` | `discovery-05-channel-h1` |
| `reader-protocol/controlled-bf16-12-post-closure.json` | `3f32d80d89b0805c127928ae3505cc3119dcc51af9baedcbb9f7d035e34c2542` | 8:40:02 PM CT | `2026-11-15-turn-post-closure-cases-astra.json`, `db50a2762839…` | the 29 other fitting answers (14 originally wrong, 15 originally correct), held until closure |

Both are `whole-gap-reader-protocol/4`, `acquisition: native-events-only`, made by `native_events_protocol.py make` on
`controlled-bf16-1`, the original control's reading (`whole-states-original/reading`) and the integrated qualification
(`generation-workers-original`), from records and row maps only. Each binds 55 instruments: the 52 files the capture's and the
reading's programs import as the tree stands (33 when `controlled-bf16-11` was frozen; the whole `tube.py`, `ladder.py` and
`gap_run.py` bring the solver's modules in), `after-the-others.sh` and the two recorders. No module is loaded by a computed name.
The two name the same instruments; their answers do not overlap and together are the control's thirty. The control's execution
(recorder `0b4ea036…`) and the generation's (`5c893cd7…`) are written as two identities with what differs between them.

**Qualification, from the source and on fixtures.** `reader_protocol.py stands` on each: no instrument with other bytes, nothing
imported and not named, read from the source (8:40 PM CT); on `controlled-bf16-11` the same check finds 8 instruments with other
bytes and 19 files imported and not named, so the old one holds nothing. `test_native_events.py`: 20 passed; with the tests of
the programs it binds (`test_cases.py`, `test_capture_bound.py`, `test_generation_engine.py`, `test_reading_identity.py`,
`test_reader_protocol_instruments.py`): 91 passed; Astra's `tests/test_functional_capture.py`: 8 passed (8:41 PM CT); this folder
whole: 247 passed at 8:39 PM CT. **Not run:** no capture has run under form 4 on any model; `native_events.py` has run on made-up
folders with a stand-in process only.

**The reason written into the protocol is corrected first (`9bbb4f33`, 8:39 PM CT).** `controlled-bf16-11` and the heads of
`native_events.py`, `native_events_protocol.py` and `capture_registration.py` gave, as Marcos's direction relayed at 6:22 PM CT,
that nothing made of a turn vector may pair. The session that relayed it withdrew that at 7:37 PM CT as its own reading; his
papers pair answers by those turns. What stands is Astra's design: the capture does not lay a regenerated answer on its own old
wrong recording, and the loop pairs events against the fixed correct reference, over complete turns, after the capture. The new
protocols' `pairs_nothing` says that and nothing about token-carrying vectors.

**What they do not establish.** That any two events are the same computation; that a measure on these states is admitted. A
protocol's `admits` field says what kind of capture it can hold a run to, a changed state's; it starts nothing. A change to any
of the 55 instruments means the protocol is made again and Astra told.
