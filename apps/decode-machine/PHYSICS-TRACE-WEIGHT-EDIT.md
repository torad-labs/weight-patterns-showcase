# A persistent weight edit — November 7, 2026

One coordinated edit to the actual layer-16 attention-output weights repaired
the targeted numerical calculation. The original response gave **60 J available,
120 J required, NO**. The edited checkpoint gives **60 J, 90 J, NO**, using the
recipient's own 3 m climb. The paired 1 m case remains token-for-token unchanged
at **60 J, 30 J, YES**.

The same edit repaired a second numerical calculation, but also introduced a
wrong final answer in a different block problem. This is evidence that the
selected parameter change can alter a useful computation, with collateral
effects. It is not a general physics repair or recovery of a complete capability.

This is the external native llama.cpp **BF16** research arm. The previous Q4
model was not edited. No training, optimizer, runtime activation replacement,
logit override, production QGRE change or factory UI change was used. The original
checkpoint remains intact; the edited GGUF is a separate file. All layer and
prediction indices are zero-based.

## What was changed

The preceding [contribution transplant](../../02-physics-trace-observation/physics-intervention/STUDY.md) changed
one recorded normalized attention contribution at a selected prediction. It
changed that prediction's vocabulary scores, but none of its generated tokens.
The user then requested a comparison using actual weights in the same recognized
computational area.

The recorded native dependency is:

```text
kqv_out-16 (2,048 inputs)
    → blk.16.attn_output.weight (1,536 outputs × 2,048 inputs)
    → raw projection output
    → RMS normalization and unchanged learned normalization weights
    → attn_post_norm-16
    → residual addition and the remaining computation
```

The observation identifies a connected transformation. It does not assign a
physical concept to a particular scalar weight. The input and raw output were
captured from both original executions at prediction 235, immediately after
the shared complete calculation of 60 J available energy.

Let `x_f` be the failing case's recorded projection input, `y_f` its raw output,
and `y_s` the paired successful case's raw output. The single declared candidate
was:

```text
u = y_s − y_f
v = x_f / (x_fᵀ x_f)
ΔW = u vᵀ
W_edited = BF16_RNE(W_original + ΔW)
```

Before rounding, this is the smallest matrix change by Frobenius norm satisfying
`ΔW x_f = u`. At another input `z`, its unrounded effect is `u (vᵀ z)`: the same
parameter change can contribute differently as its input changes. This is a
candidate derived from the recorded computation, not an independently recovered
physics direction. The paired prompts also differ in destination height, so the
source difference includes that contextual change.

All 3,145,728 native values in the matrix were read. The edit used native ggml
BF16 encoding and round-to-nearest-even. There was one formula, one strength,
one layer and one edited checkpoint; none was tuned after observing results.
The unchanged inference program then used these weights throughout prefill and
every generated prediction.

## Native representation and checkpoint integrity

| Measurement | Result |
| --- | ---: |
| Values in the selected matrix | 3,145,728 |
| Values whose stored representation changed | 200,965 (6.3885% of this matrix) |
| Changed bytes | 204,565 |
| Intended change / original matrix norm | 0.036654% |
| Rounded change / original matrix norm | 0.029656% |
| Intended versus realized matrix-change cosine | 0.702831 |
| Remaining distance to donor projection at the fixed source input, CPU equation diagnostic | 44.4103% of the original distance |
| Zero-update BF16 decode/re-encode | Every value byte-identical |
| Bytes compared across the complete original and edited GGUF | 9,311,304,896 |
| Bytes changed outside the selected matrix | **0** |

The matrix norm percentage is not a percentage of capability or information.
The proposed change was small enough that native BF16 rounding removed or
altered many individual increments. It still closed about 56% of the fixed-input
projection distance in the diagnostic calculation. Native inference, rather than
that calculation, determines the behavioral result below.

The ordinary CPU matrix equation agreed with the captured unmodified native
projection to relative error `8.97e-8`, supporting the interpreted matrix layout.
The full-file verification also recomputed the original model's known SHA256.
Native before/after matrix bytes are saved beside the edited checkpoint under
`models/`; activations, responses and reports are under `runs/legs/`.

## All 28 behavioral comparisons

The suite is the original 24 physics, paraphrase and supplied-arithmetic questions
plus the four existing numerical variants. These are previously observed
questions, not a new blind evaluation set. Only the original block pair supplied
the input and output vectors used to construct the edit.

| Measure | Original | Edited |
| --- | ---: | ---: |
| Complete correct numerical calculation | 7 / 28 | **9 / 28** |
| Correct explicit YES/NO verdict | 16 / 28 | **16 / 28** |
| Reached the 768-token cap | 9 / 28 | 9 / 28 |
| Token-for-token identical to original | — | 18 / 28 |

Ten continuations changed. All six supplied-arithmetic continuations remained
token-for-token identical. All seven previously correct complete numerical
calculations remained correct. Nevertheless, one previously correct final
verdict was lost. Numerical correctness, final verdict and explanation quality
are separate measurements.

| Case | Original available / required / verdict | Edited | Interpretation |
| --- | --- | --- | --- |
| Original 3 kg block, 3 m climb | 60 / **120** / NO | 60 / **90** / NO | Targeted numerical error repaired; the earlier 1 m descent is no longer added to the 3 m climb |
| Paired 3 kg block, 1 m climb | 60 / 30 / YES | Identical response | Existing numerical success preserved |
| Sled, 1 m climb | **0** / 40 / **NO** | **80** / 40 / **YES** | Correctly adds the 40 J downhill gain, then subtracts the 10 J loss |
| 4 kg block, 2 m climb | 60 / 80 / NO | 60 / **40** / **YES** | New failure: subtracts the earlier 1 m descent from the destination height |

The repaired responses still contain inconsistent top/bottom labels. They count
as repaired numerical chains, not fully correct physical explanations. The
4 kg case already contained an incorrect intermediate kinetic-energy calculation
before editing despite its previously correct final triple; editing then also
made its required energy and verdict wrong. That regression must not be hidden
by the increase from seven to nine correct calculations.

The other altered responses remain numerically wrong or incomplete. The complete
texts and per-case judgments, including partial calculations and contradictions,
are in `edited-assessment.json` and the ordinary-run result files.

## Following the effect

The six edited captures reproduce ordinary edited inference exactly across
**3,396 decisions**, including every token and complete-vocabulary score hash.
Paired internal comparison covers the two original source cases; the four
numerical variants have retained edited traces without paired internal analysis
in this report.

The repaired recipient retains its original first **269 generated tokens**.
Predictions 0–269 therefore have identical incoming histories, including the
conditioning point 235. Across those 270 predictions, and all 450 predictions of
the unchanged paired response, the native projection and final normalized state
change on every prediction. Across the paired captures, all **19,152 retained
upstream checks** remain byte-identical, including the projection input and
preceding layer exits. These checks include prompt positions.

At prediction 235 in the failing recipient, the connected change is:

| Boundary | Difference from original, relative L2 |
| --- | ---: |
| Input to the edited projection | Exactly zero |
| Raw projection output | 0.9191% |
| Normalized attention contribution | 1.0273% |
| Layer 16 exit | 0.3570% |
| Layer 34 exit | 0.02357% |
| Final normalized state | 0.05797% |

Each row uses its own original boundary vector as the denominator. The full
progression is retained rather than assumed to shrink monotonically. Native
inference leaves 44.4103% of the original raw-output distance to the donor at
this input, agreeing with the separately reported rounding diagnostic.

The first visible divergence is a narrow choice at prediction **269**:

| Candidate token | Original logit | Edited logit |
| --- | ---: | ---: |
| ` reached` | **16.181353** | 16.177919 |
| ` gained` | 16.174959 | **16.186314** |

The original continuation says “The total height reached” and subsequently adds
the earlier descent to the climb. The edited continuation says “The total height
gained during the climb” and uses the stated 3 m climb. This is the observed
first branch in the text, not proof that this word alone caused the numerical
repair. The parameter edit acts at every prediction. After this divergence,
the two internal histories are no longer treated as matched contexts.

The edit is not confined to failures: complete-score hashes change at all
**11,053 comparable predictions across the 28 cases**, including those whose
generated tokens stay unchanged. The original block-base response has changed
internal states and scores while all 450 selected tokens remain the same. Thus
behavioral preservation here means preserved outputs, not untouched internal
computation.

Unlike the earlier one-time substitution, the parameter change is applied again
on each use of the projection. It does not require the previous layer-16 change
to survive in K/V storage in order to change the next prediction. The evidence
now connects an actual weight edit, unchanged incoming signals, altered native
projection outputs, downstream transformations and changed ordinary responses.

## Experimental conditions and scope

The successful original-model replay reproduced all **14,578 decisions across
28 questions**, including every selected token and complete-vocabulary score
hash. The additional input/projection capture reproduced all **930 decisions**
for the original pair, and its selected post-normalization source bytes matched
the earlier activation experiment exactly. The graph verified the projection's
actual weight, input, normalization and residual dependencies.

A preliminary attempt on the second GPU failed exact baseline qualification:
all seven completed cases differed in vocabulary-score hashes, and six differed
in tokens. It was stopped before source extraction or any weight edit. Its
artifacts remain separate. The results reported here use the original **RTX 5080
device 0** for both original and edited checkpoints.

Settings remained greedy decoding, 768-token limit, context 2048, batch 128,
microbatch 32, eight CPU threads, native flash attention, and the same disabled
fusion setting. Model and engine identities and every command outcome are in
[the evidence index](run/EVIDENCE-INDEX.md).

The trace does not recover individual scalar contributions through every fused
operation, runtime KV-slot contents, a complete causal circuit, or the whole
physics capability. No unrelated-language benchmark or matched-size random edit
was run. The experiment establishes the effect of this particular persistent
parameter edit on these cases. It does not establish that the source difference
is exclusively a success/failure direction or that similar edits are generally
safe.

The earlier activation substitution and this checkpoint edit also have different
exposure: one substitution at one prediction versus changed parameters at every
use. They answer the requested practical comparison, not an isolated comparison
of weight editing against repeated activation steering with otherwise identical
exposure.

The original and edited launch phases, source extraction, native editor,
comparison analysis and trace summary all completed with exit code 0. Independent
review found no material issue in the instruments, all ten changed response
assessments, or this report against its saved evidence. The full command lines,
binary/source hashes and review are retained in the evidence directory.
