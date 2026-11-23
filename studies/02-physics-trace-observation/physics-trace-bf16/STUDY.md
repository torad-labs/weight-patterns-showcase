# BF16 physics trace comparison — November 7, 2026

The same 24 questions now have BF16 ordinary and recorded executions. BF16
completed more correct numerical calculations, while several physical mistakes
persisted. The most useful new observation is a pair that correctly calculates
54 + 30 = 84, then 84 − 24 = 60, throughout a shared generated history. Its
recording connects concrete successful intermediate steps to actual changes in
the computation.

The local progression previously observed in Q4 also broadly persists:
relative separation falls at the layer-14 feed-forward addition, then rises at
the layer-15 addition, at **1,707/1,708 eligible decisions**.
The one exception remains in the evidence. These correlated observations are
not independent demonstrations, a recovered physics signature, or a safe edit target.

This arm was planned under the November 6 study directory and executed on
November 7 CT. It uses external llama.cpp, as authorized, and makes no change to
QGRE production precision, inference or training. No training or parameter edit occurred.

## What the model did

| Observation across the same questions | Q4_K_M | BF16 |
| --- | ---: | ---: |
| Correct explicit final verdict, all cases | 8/24 | 13/24 |
| Complete correct arithmetic calculation and verdict | 1/6 | 4/6 |
| Complete correct physical numerical chain and verdict, including paraphrases | 0/18 | 2/18 |
| Physically consistent completed explanation | 0/18 | 0/18 |
| Responses reaching the 768-token cap | 10/24 | 8/24 |

The two correct physical numerical chains are **block-base** (60 J available,
30 J required, YES) and **cart-paraphrase** (60 J available, 40 J required, YES).
Their prose nevertheless contradicts parts of the physical setup: the block
response swaps top/bottom labels, and the cart explanation gives inconsistent
reference-height and starting-kinetic-energy descriptions. Numerical success
and those contradictions are both retained. A correct verdict reached with an
incorrect calculation is a separate outcome.

Other responses preserve useful partial applications. Both block variants
correctly obtain 60 J; the changed variant later adds the earlier descent to
the final climb and asks for 120 J instead of 90 J. The capped coaster
paraphrase correctly obtains 140 J after descent and 125 J after friction.
The bead original correctly calculates 38 J, then 30 J, before an extra
potential-energy subtraction incorrectly turns that into zero.

An independent reader checked all 24 full BF16 texts, verdicts, partial
successes and truncation classifications. This is a bounded prompt-set result,
not a general model capability score. Behavior remains mixed in BF16, so
the earlier failures cannot be attributed exclusively to Q4 from these results.
The [per-case assessment](run/behavior-assessment.json) preserves the detail.

## A correct partial calculation with a controlled comparison

The block questions differ in one prompt token: the destination is 1 m or 3 m
above the start of the climb. Both require the same available-energy calculation.
Their BF16 responses share 253 generated tokens, including the complete
54 + 30 − 24 calculation. The first divergent output is decision 253, when
the model restates the destination height. The Q4 block pair shared only three
generated tokens, giving four eligible predictions before its text diverged.

Four calculation moments were selected from the BF16 ordinary text before
examining their internal vectors. Each has a trail through all 35 layer exits
and additional attention/FF boundaries at layers 12–16:

| Observable calculation moment | Decision, zero-based | FF14 relative ratio | FF15 relative ratio |
| --- | ---: | ---: | ---: |
| Adds downhill gain: 54 + 30 | 166 | 0.493 | 1.471 |
| Produces 84 J | 177 | 0.546 | 1.425 |
| Subtracts friction: 84 − 24 | 222 | 0.527 | 1.425 |
| Produces 60 J | 233 | 0.583 | 1.281 |

![Recorded transformations at four calculation moments](run/block-calculation-trails.svg)

These panels describe the difference between the two recorded states at each
prediction. They do not claim that a curve is the physical energy itself, or
stitch different token histories into one spatial path. Number anchors select
the first digit of the number. Emitted calculations locate observable behavior;
they are not assumed to be a faithful account of the internal computation.

Nineteen anchors in total include these four moments in both block responses,
the later erroneous height addition, and the bead/cart/coaster progressions.
Eight anchors remain inside the block pair's common history. Eleven are later
individual observations and are explicitly excluded from prompt-only contrasts.
The collector verified 190 source-linked additions at those anchors.
[Anchor selection](run/CALCULATION-ANCHORS.md) and
[the recorded trails](run/milestone-trails.json) retain graph identities,
tensor-byte offsets, state changes and output candidates.

## How broadly the earlier progression persists

Each within-precision pair changes one prompt token and reverses the expected
verdict. Eligibility includes every decision through the first divergent output
token, while generated input histories still agree. Later single-run traces
remain available, but are not treated as matched prompt-only contrasts.

| Pair | Q4 progression / eligible | BF16 progression / eligible | BF16 median FF14 ratio | BF16 median FF15 ratio |
| --- | ---: | ---: | ---: | ---: |
| cart | 94/94 | 297/297 | 0.510 | 1.537 |
| sled | 104/104 | 306/306 | 0.502 | 1.519 |
| bead | 4/4 | 20/20 | 0.482 | 1.530 |
| coaster | 223/223 | 270/270 | 0.502 | 1.547 |
| block | 4/4 | 253/254 | 0.526 | 1.492 |
| trolley | 251/251 | 307/307 | 0.503 | 1.489 |
| cart-paraphrase | 17/17 | 35/35 | 0.510 | 1.613 |
| coaster-paraphrase | 16/16 | 50/50 | 0.497 | 1.606 |
| block-paraphrase | 40/40 | 47/47 | 0.515 | 1.565 |
| cart-arithmetic | 25/25 | 72/72 | 0.541 | 1.682 |
| coaster-arithmetic | 34/34 | 25/25 | 0.563 | 1.714 |
| block-arithmetic | 34/34 | 25/25 | 0.585 | 1.657 |

![First-decision trails in both checkpoint representations](run/precision-comparison.svg)

The figures show individual first-decision trails rather than average paths.
Legend counts cover all eligible positions. Different full-window medians have
different generated histories and denominators, so they are not a controlled
estimate of the size of a precision effect.

The exception is **block, decision 50**: FF14 relative separation grows by a
factor of 1.0010507 rather than falling; FF15 grows by 1.788667. The emitted token
is a space/backslash in the initial kinetic-energy expression, before the marked
calculations. Its small magnitude does not justify silently discarding it.

Absolute separation grows at all 1,708 FF14 additions and all
1,708 FF15 additions. At FF14, the whole state usually grows
faster than the difference, producing a relative decrease. Across the twelve
pair medians, absolute growth is 1.635–2.001 while state-length growth is
3.308–3.423. At FF15 those ranges are 3.624–4.063 and 2.360–2.447.
Update alignment and changes of direction are retained alongside these scales.
The measured addition includes the MLP's output normalization and learned scaling.
These observations do not identify a semantic axis or show that information was erased.

Two additional comparisons keep their populations distinct:

- **Same prompt, Q4 versus BF16:** 411 eligible predictions
  across 24 questions, with the relative progression at 411/411.
  Many shared histories end in the introductory text; these do not provide
  matched later physical reasoning across precisions.
- **Same generated history across all four runs in a prompt pair:**
  131 eligible positions, with the progression at
  131/131 in Q4 and
  131/131 in BF16. Each original physical pair
  contributes only four early predictions here. This stricter comparison is a
  reuse of recorded observations, not another experiment.

The figures also prompted a post-hoc examination of layer 16. At the first
prediction, the segment from layer 15's exit through layer 16's attention
addition has a larger relative-separation ratio in BF16 in all twelve pairs.
Across the 131 four-way matched positions, this holds at 103, so it is not
universal over the later shared text. The first-prediction ratios range from
0.828–1.023 in Q4 and 1.254–1.921 in BF16. This is a contextual difference in
the connected computation, not attribution to individual attention weights or
an explanation of the behavioral improvement. The full source-linked rows,
including the other 28 positions, are in
[the exploratory layer-16 comparison](run/posthoc-layer16-comparison.json).

The recurrence across prompt changes, arithmetic controls and checkpoint changes
supports treating this as a broad participating transformation to investigate.
It does not yet distinguish the contribution to energy accounting from other
computation at those stages. A useful trace may be shared and partial; the
next question is what specific dependency a fragment helps preserve or change.

## Can the recordings support this interpretation?

The short two-case qualification matched ordinary, sham and capture tokens and
complete vocabulary-score hashes. The full 24-case BF16 ordinary/capture study
then matched all **12,023 generated decisions** and hashes covering
**3,151,757,312 native vocabulary scores**, with finite-score checks.
Hashes establish equality within BF16; they cannot reconstruct full logits.
Top-eight candidates, exact token bytes and full response texts are retained.
Native prompt tokenization matches the corresponding Q4 prompt exactly.

The capture contains 7,735,750 callback events:
751,750 retained tensor-payload records and
6,984,000 explicit payload omissions under the focused scope.
Native tensor data occupies 5,937,475,584 bytes. Actual graph sources,
byte intervals and positions were checked. Within-BF16 analysis verified
34,160 attention/FF additions with native rounded operands;
the cross-precision analysis verified 8,220
such additions in its own population. These counts overlap and are not
independent tests of model behavior.

Retained boundaries cover the input, all layer exits, final norm, and selected
internal states/operands at layers 12–16 for prefill positions and generated
decisions. These are native float32 intermediate snapshots; the checkpoint
remains the published BF16/F32 GGUF. Scalar weight contributions, fused attention
internals and runtime KV-slot lineage remain gaps. Selected-state vectors and
all output scores were checked finite, not every stored intermediate scalar.

Completed event JSONL files were losslessly gzipped. Every archive was fully
decompressed and checked for original SHA256 and byte length before its redundant
raw JSONL was removed. Tensor bytes and previous studies were unchanged.

## Checkpoint and execution provenance

The BF16 file is the 9,311,304,896-byte artifact from
[the publisher's Gemma 4 E2B repository](https://huggingface.co/bartowski/google_gemma-4-E2B-it-GGUF),
at revision 81012ba3538e061d5ee003f11f25335b17f82e2d. Its verified SHA256 is
6a737131362433969a148503c92ef28cf7cf80648ac8ec74af19413a331b575c.
The publisher's Q4 checksum matches the earlier study checkpoint.
All 601 tensor names and dimensions match: BF16 has 318 BF16 and 283 F32 tensors;
the Q4_K_M file has 194 Q4_K, 70 Q5_K, 53 Q6_K, one BF16 and 283 F32 tensors.

The existing observer binary and immutable llama.cpp build were unchanged.
Both arms use greedy decoding, the same 768-token limit, unfused standalone
execution, flash attention, context 2048, batch 128, microbatch 32 and eight
CPU threads, on device 0 (RTX 5080). Native arithmetic paths vary with tensor
type. The two published representations therefore do not isolate pure
quantization error from every other numerical difference. The earlier
server/default-fusion versus standalone discrepancy remains separate.

The [evidence index](run/EVIDENCE-INDEX.md) connects prompts, responses,
recorded transformations, comparisons, limitations and preserved analysis
sources. [Commands](run/COMMANDS.md) and [independent review](run/REVIEW.md) are
retained. The factory UI has not yet imported this raw research trace format.

This arm gives us recorded correct partial progressions, their later departures,
and a recurring computational segment that survives the checkpoint change.
A next bounded causal probe can start from those concrete moments. No claim
of a full capability signature, geometrical reconstruction or successful edit
is made by this observational study.
