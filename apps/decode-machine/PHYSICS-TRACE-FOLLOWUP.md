# Physics trace follow-up — November 6, 2026

The predicted local progression occurs at **846/846 eligible decisions**
across the twelve recorded pairs. The table below preserves the denominator for
each pair. These are correlated positions in related executions, not 846
independent demonstrations or a statistical confidence claim.

We have extended observation of a connected transformation beyond the original
coaster pair. Its occurrence in supplied-arithmetic comparisons is part of the
finding. It does not establish a physical role, a shared complete capability
signature, or an edit target. Partial/shared influence remains the research target.

## What changed from the pilot

The same quantized Gemma 4 E2B checkpoint, llama.cpp build, greedy decoding and
unfused standalone configuration were retained. The model file's SHA256 was
rechecked. Twenty-four cases comprise six physical pairs, three paraphrased
pairs and three supplied-arithmetic pairs. Each pair changes one prompt token
and reverses the expected verdict. The original coaster pair is a replication;
the five other original physical pairs add new internal observations of questions
already included in the earlier behavioral pilot. Paraphrases and arithmetic
comparisons are related contrasts, not independent physical tasks.

Instead of seven sampled positions, analysis includes every decision whose
generated input histories still match, through the first divergent output token.
It stops the prompt-only contrast after that point. The original coaster data
were also examined separately: the progression holds at all 223 eligible
decisions there. That reuse of the pilot is not additional independent evidence.
The fresh coaster pair contributes 223 positions to this table.

## The connected progression

At each inspected decision we follow the actual recorded attention addition,
feed-forward addition, per-layer input/scaling and next layer. The detailed local
region covers layers 12–16; every layer exit supplies broader context. Before and
after states link to operation/source identities and byte offsets in the capture.

The predicted sequence was a decrease in relative state separation at the
layer-14 feed-forward addition followed by an increase at the layer-15 addition.
Relative separation is the distance between the two states divided by their
average length. Values below one in the table's last two columns mean that
relative separation decreased during the corresponding addition.

| Pair | Kind | Predicted sequence / eligible | Median FF14 ratio | Median FF15 ratio |
| --- | --- | ---: | ---: | ---: |
| cart | physics | 94/94 | 0.559 | 1.506 |
| sled | physics | 104/104 | 0.567 | 1.515 |
| bead | physics | 4/4 | 0.506 | 1.510 |
| coaster | physics | 223/223 | 0.556 | 1.497 |
| block | physics | 4/4 | 0.493 | 1.405 |
| trolley | physics | 251/251 | 0.574 | 1.513 |
| cart-paraphrase | paraphrase | 17/17 | 0.535 | 1.504 |
| coaster-paraphrase | paraphrase | 16/16 | 0.552 | 1.530 |
| block-paraphrase | paraphrase | 40/40 | 0.545 | 1.531 |
| cart-arithmetic | arithmetic | 25/25 | 0.563 | 1.625 |
| coaster-arithmetic | arithmetic | 34/34 | 0.556 | 1.690 |
| block-arithmetic | arithmetic | 34/34 | 0.568 | 1.682 |

Across these observations, FF14 relative ratios span 0.473–0.790 and
FF15 ratios span 1.271–1.987. The connected detail retains every observed
case, including any departures, rather than only a favorable aggregate.

![Connected first-decision trails](run/connected-followup.svg)

The figure shows actual first-decision trails, each normalized to its own start;
it is not an averaged path. The legend counts include all eligible decisions.
Plot coordinates and distance summaries are viewing measures, not recovered
capability coordinates or a geometric shape discovered in the weights.

Absolute separation, state-length growth, update alignment and turns are retained
alongside the relative measure. A relative decrease is not an erasure of
information. Absolute separation did not fall at 846/846 FF14
additions and 846/846 FF15 additions. The following ranges are
the minimum and maximum of the twelve pair medians, not pooled medians:

| Layer | Absolute separation ratio | State-length growth | Update alignment cosine | Before/after difference cosine |
| --- | ---: | ---: | ---: | ---: |
| 14 | 1.721–1.942 | 3.284–3.387 | -0.067–0.029 | 0.514–0.552 |
| 15 | 3.508–4.041 | 2.382–2.547 | -0.041–0.090 | 0.232–0.360 |

Update alignment measures the difference between the two added branches against
the incoming difference. Zero means perpendicular in the native residual
coordinates; it does not identify a semantic axis. The last column measures
the turn from the incoming difference to the outgoing difference. These views
retain the distinction between scale and direction without assuming that a
shared scalar pattern implies a shared vector direction across cases.
The captured implementation normalizes the MLP output before adding it to the
incoming state; the observation therefore includes that normalization and its
learned scaling, not just the unnormalized matrix product.

## What the model actually did

| Kind | Cases | Correct explicit verdicts | Complete correct solutions | Capped |
| --- | ---: | ---: | ---: | ---: |
| physics | 12 | 3 | 0 | 7 |
| paraphrase | 6 | 2 | 0 | 3 |
| arithmetic | 6 | 3 | 1 | 0 |

An independent reader checked all 24 full response texts and their manual
classifications. None of the eighteen physical responses completed a correct
solution. Several subtract or omit gravitational gain on descent. The sled
original first computes the correct 90 J after descending, labels the 10 J
alternative incorrect, then uses that 10 J to continue. The bead original
computes the correct bottom kinetic energy but later mixes energy quantities and
height references. These behaviors remain attached to their own captures.

The only complete correct solution is cart-arithmetic-changed: A=60, B=80, NO.
The other cart arithmetic case computes A=60 and B=40 but answers NO. Coaster
arithmetic turns one-half into 2; block arithmetic multiplies mass twice.
Correct verdicts with incorrect calculations are not counted as correct solutions.
The capped block original's verbal reachability conclusion is preserved in the
assessment but does not count as an explicit final YES/NO.

These results describe this prompt set and execution configuration. They are not
a general capability score for Gemma. The recurrence is primarily observed
during unsuccessful reasoning; this study does not identify a successful physics
trace or show that the local progression causes the physical mistakes.

## Connecting concrete calculation moments

Eight anchors were selected after reading the ordinary responses and before
examining their internal vectors. They preserve partial correct applications as
well as departures, even where no complete physical solution succeeded:

| Execution | Observable progression | Recorded decision indices (zero-based) |
| --- | --- | --- |
| sled-base | Adds downhill gain; obtains 90 J; rejects 50−40; later uses the rejected 10 J | 272, 283, 331, 538 |
| cart-paraphrase-changed | States addition of downhill gain; later substitutes subtraction | 387, 466 |
| cart-arithmetic-base | Calculates 60 and 40 correctly; answers NO incorrectly | 120 |
| cart-arithmetic-changed | Calculates 60 and 80 correctly; answers NO correctly | 79 |

Each anchor now has an actual recorded trail from the input through all 35 layer
exits, with the extra internal boundaries at layers 12–16, source identities,
tensor offsets and top eight output candidates. The collector verified all
80 local additions at these anchors.
At the sled's adoption of 10, the selected first digit 1 scored 27.7549 and the
alternative first digit 9 scored 25.4546. Those are native logits, not
probabilities or proof that the model internally maintained the correct 90 J.

These anchors make it possible to revisit concrete moments in the developing
computation. They do not add matched-prefix observations: their generated
histories differ. Emitted calculations locate observable behavior, without
assuming faithful internal reasoning or treating a later error as evidence that
a particular layer caused it. Selection and limitations are preserved in
CALCULATION-ANCHORS.md; the source-linked records are in milestone-trails.json.

## Can the recordings support the comparison?

The short ordinary/sham/capture qualification matched tokens and full-score
hashes, and matched actual vocabulary-score bytes from the original pilot.
Native tokenization was exercised and matched the previous prompt IDs.
All 24 full ordinary/capture comparisons then matched their 13462
generated decisions and SHA256 values for 3,528,982,528
native vocabulary scores. Every score was checked finite. Full score bytes are
not retained in this follow-up; hashes establish agreement, while the top eight
scores, exact token bytes and text remain available.

There are 8,653,832 callback events, with
840,968 retained tensor payload records and
7,812,864 explicitly omitted by the focused
scope. Tensor data occupies 6,485,630,976 bytes.
Analysis checked graph references, binary intervals and position metadata, and
verified 16,920 recorded attention/feed-forward additions
against their native rounded operands with zero difference. Both addition operands
are connected by the actual source graph. Selected-state vectors and every
vocabulary score are checked finite; this does not claim a scan of every stored scalar.

Callback boundaries remain the original detailed observer's boundaries; only
persisted payloads were narrowed. Flash-attention internals, scalar products,
runtime KV-slot lineage and some backend copies remain gaps. Intermediate
snapshots retain the native float32 values of this execution; the checkpoint
remains in its original mixed GGUF Q4_K_M representation.

The pilot's default-fusion/server versus standalone differences remain unresolved.
This study consistently uses the qualified unfused standalone execution. It does
not label those different execution settings as interchangeable.

## What this supports next

There is a reproducible connected segment to investigate across related requests.
Its presence in arithmetic may reflect a broader participating transformation;
exclusivity to physics is not required. Recurrence alone does not tell us which
part carries the physical dependency, which supports arithmetic or language, or
which contribution would be safe to change.

Many eligible prefixes include introductions, restated quantities or setup text.
Matching generated histories isolates the prompt change, but does not align all
pairs to the same physical reasoning step. Full later per-run observations remain
available, explicitly outside the prompt-only contrast once histories diverge.
Observing repeated successful physical applications is still a gap.

A BF16 comparison is a useful next possible arm with this exact prompt set and
qualified inference entry, each precision checked independently. It could show
whether behavior and this progression change with checkpoint precision; the
current data do not establish that quantization caused the errors. No BF16 model
was downloaded or run, no training or weight edit occurred, and no production
engine change was made. The raw trace format is not yet imported into the factory UI.

The [evidence index](run/EVIDENCE-INDEX.md) connects prompts, responses, fidelity
checks, source-linked transformations, behavioral assessments and preserved
instrumentation. The framework remains the project’s RESIDUAL-TRACE-HANDOFF.md.
