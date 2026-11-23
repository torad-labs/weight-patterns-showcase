# Independent source-to-code review

November 15, 2026, 9:51 PM CT. Marcos directly assigned this implementation-peer session to
keep the research anchored to its hypothesis, read his four documents, then
check the code against the day's repeated mistakes. This review does not
transfer Astra's ownership of the loss and loop or authorize another method.

**Finding: the current mathematical proposal is coherent; the native loop is
not connected yet.** The remaining work is implementation, not another search
for a definition of the trace. No repair or whole-build approval follows from
the component tests. The independent subagent asked to perform the requested
The independent reviewer whole-build review reached the same finding after reading all four sources
directly, including the DOCX's three equation images. This is a new independent
review, not approval from the standing reviewer session.

## Sources and hypothesis

Both reviewers read the three supplied PDFs and
`Iterative_Learning_from_Computational_Traces.docx`, rather than a seat's summary.
Permanent copies are under `reports/`. The supplied path-edit PDF is an earlier
version than the repository's: Sections 1–5 agree; later sections reword the
judgment in terms of computation and add a replay/native mismatch limitation.
The adjacent `2026-11-15-independent-source-review-peer.json` records the exact
source and implementation hashes. The review includes the governing method,
machine method, November 7 clarifications, and November 15 clarifications.

The hypothesis is that a capability leaves a recurring difference between
correct and wrong computational traces, and editing the weights responsible
for that difference repairs the capability. The current one-wrong-prompt,
one-fixed-original-reference experiment tests the correction mechanism first.
Closing that one trace would not by itself establish transfer to unused
answers; the subsequent experiment remains necessary.

The source distinctions that prevent another substitution are:

| Source | What it supplies | What the current experiment must not silently inherit |
|---|---|---|
| Fixing a reasoning mistake by editing the weights | Local contributions can recur through a connected path; the entire capability need not live in one isolated weight | A large weight alone as the causal explanation |
| Trace-derived guide, Sections 4 and 7 | Full turn `T = unit(output) - unit(input)`; differentiation and inverse response through the computation | Its historical token-selected support and same-token references |
| Path-edit paper, Sections 4.1–4.2 and 5 | Complete turn-based correspondence, whole-vector gap versus chance, response-based inverse correction | Its post-merge state target, original-text-only sweep, or group-mean variance relabeled as single-execution variance |
| Iterative learning document, read with later instructions | Fixed original reference, regenerated current trajectories, discrepancy beyond original variation | Extra keep/behavior objectives or generating other answers during the current one-prompt training experiment |

## What the current source actually does

- `turn_pair.py:16` computes every merge's turn from its own incoming and
  outgoing state, retaining magnitude and both derivatives. These turns are
  the papers' computation trace; calling them forbidden token vectors would
  repeat the coordinator's withdrawn interpretation.
- `turn_correction.py:30` and `complete_overlap.py:62` align complete traces
  by their turn fields and require coverage of both event inventories.
  `turn_reference.py:33` chooses an observed original successful trace from
  computational distances. No prompt similarity or token equality selects
  this reference or the current loss support.
- `vector_chance.py:55` compares the full current path against that fixed
  original, with original event mass. The same local boundary supplies the
  correction request and complete outside count; a lower aggregate loss or
  smaller represented population is not closure.
- `vector_chance_reference.py:48` measures predictive squared variation of
  individual original successes around the fixed trace. This is Astra's
  explicit adaptation of the paper's signal-versus-chance distinction,
  **not** the paper's variance of two group means. Scalar variance can hide
  variation in a stable direction; neither independent statistical coverage
  nor capability repair is established. Perfect whole-trace coverage of every
  original success is not added as a new gate.
- `joint_inverse.py:118` solves the joint linear least-squares response;
  `trace_response.py:82` derives the length from the combined response.
  The step is not a raw gradient multiplied by a guessed learning rate.
- `native_events.py:6` acquires the model's own history without pairing it
  to its original wrong answer. Tokens and natural ending remain acquisition
  provenance. They do not grade the computation or establish convergence.

## The concrete missing connection

`pair_loop.py:1` explicitly has no native backend or command and is used only
by its simulated test backend. `turn_correction.py:5` explicitly installs
nothing. `functional_install.py:26` can write and publish the complete joint
BF16 update, but no current production caller connects these components.

The next implementation is the existing one-prompt path:

`native capture -> native reading -> fixed-reference measurement -> joint
proposal -> complete BF16 installation -> reached-state record -> fresh native
capture and correspondence -> same fixed-reference measurement`.

The actual backend must exercise these connections, reject stale readings or
incomplete installed states, and keep the prospectively chosen other-answer
evaluation after full fresh-trace closure. This connects the agreed method;
it introduces no second objective, new calibration scheme, or extra study.

At `turn_correction.py:76`, candidate evaluation calls the anchored Torch
graph on the held history. Its BF16 weights are real rounded values, but its
reported objective is a **surrogate proposal measurement**, not a changed
native engine measurement. `native_graph.py:9` correctly discloses that its
reconstructed attention cache is not captured native cache bytes. Fresh native
acquisition must supply the outer measurement. An accepted proposal, inner
stationarity, generation limit, or initially inside wrong trace cannot be
reported as a repair.

## Checks and action

Reviewed HEAD `c894cc92` and the working-tree vector implementation, whose
hashes are kept alongside this record. The repository verifier reports
`round-2 verifier: no drift`. Eighteen focused test files covering turn loss,
reference compilation/loading, complete correspondence, native graph/readings,
joint inverse/response, BF16 search/install, capture admission and controller
behavior passed **96 tests and 10 subtests**. One explicitly opt-in CUDA fixture
was skipped. These are CPU fixture results, not a native experiment.

Command, from `code/signal-trace`:

```sh
uv run --offline --with torch --with numpy --with pytest python -m pytest -q tests/test_turn_pair.py tests/test_turn_reference.py tests/test_vector_chance.py tests/test_vector_chance_reference.py tests/test_vector_chance_run.py tests/test_turn_artifact.py tests/test_turn_correction.py tests/test_joint_inverse.py tests/test_joint_functional_step.py tests/test_trace_response.py tests/test_native_graph.py tests/test_functional_reading.py tests/test_functional_capture.py tests/test_functional_install.py tests/test_bf16_direction.py tests/test_pair_loop.py tests/test_pair_evaluation.py tests/test_complete_overlap.py
```

The exact integration gap and proposal/native distinction were sent to Astra
through Splice; Fable was also told the review's scope and asked for any current
connection work. At this snapshot those messages were submitted, with no reply
yet. No claim is made that either seat has acknowledged or fixed the gap.
No research run, model write, code cleanup, or change to their uncommitted
implementation was performed by this review. Future review starts from these
sources and the current entrypoint, and replaces this snapshot when the backend
exists; it does not reclassify the restored turn trace or repeat retired-box
findings as if they describe current code.


## Follow-up: native connection and Marcos's correction

November 15, 2026, 10:17 PM CT. The earlier missing-backend finding is resolved in
Astra's `vector_loop.py`: the shared controller now connects native capture,
kept reading, computational measurement, joint proposal, BF16 installation,
reached-state verification and fresh measurement. Root's independent
`test_vector_loop_connection.py` exercises the real writer/loader/controller
with a stand-in proposal and engine. It checks the accepted update, stale
reading refusal, incomplete matrix refusal and incomplete published-state
refusal. It also calls the real CGLS solver through the backend's progress
callback. Astra corrected that callback's argument mismatch. The focused
connection/capture/install/controller check passed 26 tests; a later rerun
passed the four connection cases against the tightened prospective bindings.
The callback and prospective cases/protocol checks are implementation fixes,
not new research criteria. Compact-state integration is still being reviewed.

Marcos directly corrected the discussion of shared prompt positions. I had
made position a deciding issue and sent a proposed positional cut to Astra;
that guidance was withdrawn before any scope change. Fable likewise withdrew
his recommendation and his inference that exact agreement made closure
impossible. A measured computational agreement or divergence is the relevant
fact. Token identity, prompt sameness and textual position provide no rule for
including, excluding, preserving or grading computation. Astra confirms no
positional scope change was made. No radius floor or outcome-based row removal
was introduced.

Marcos also explicitly requires the independent reviewer to review the whole build, always given
the research papers and the goal. The existing independent reviewer has been
resumed on the native backend, supplied again with the full goal and all four
original documents. `AGENTS.md` now makes that requirement persistent for
future reviews. A source-grounded build assessment will identify the exact
reviewed implementation; it will not be presented as a native repair result.

## Final connection and storage checks

November 15, 2026, 10:26 PM CT. Root's final focused set passes **41 tests and
2 subtests** against backend SHA-256
`254013e043947a93f14d5dc46190669e81ed9227ad964648d8e61b4d09dda737`.
The six independent connection cases cover complete publication followed by
fresh measurement, stale-reading refusal, incomplete weight and publication
refusal, and both packed-offset and original-model-identity corruption in a
previous compact archive. Focused Pyright for the new test file has no errors
or warnings. The actual writer, loader and controller run in these fixtures;
acquisition and the research proposal are stand-ins, so this is no empirical
repair or inverse-convergence result.

The independent reviewer reproduced both storage defects independently. The fixed release path
checks the saved compact record's canonical path and SHA-256, reconstructs the
previous whole-model hash from original bytes and saved tensors, and checks
the full copy before removing it. Both corruption witnesses now preserve the
full model. The earlier reconstruction-only fix was insufficient for damaged
restore-critical metadata; the saved-record binding closes that defect.

The independent reviewer also caught a concrete acquisition mismatch in the prospective declaration:
its cohort was `wide-overlap`, while copied protocols named a different cohort.
Astra remade both protocols with the existing maker and rejects such mismatches
before acquisition. This binds capture provenance; it adds no text-based
comparison criterion. Training and post-closure inventories stay separate,
and the latter is not acquired during training.

Final verification command, from `code/signal-trace`:

```sh
uv run --offline --with torch --with numpy --with pytest python -m pytest -q tests/test_vector_loop.py tests/test_vector_loop_connection.py tests/test_state_tensors.py tests/test_functional_capture.py tests/test_functional_install.py tests/test_pair_loop.py tests/test_turn_correction.py
```

The continuing-review requirement is committed in `AGENTS.md` at `a9bcdb63`.
The independent test file was released to Astra for the backend's atomic
commit. The independent reviewer's final source-bound verdict is **YES: ready to start the declared
one-pair experiment**, recorded in `2026-11-15-whole-build-review.md` and
its adjacent JSON. She independently checked all 82 implementation bindings
and constructed the actual backend with acquisition forbidden. This review
performed no native acquisition or model update and supplies no repair claim.
