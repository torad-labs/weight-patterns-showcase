# Ghost Geometry: measuring before drawing

For the current research target, read [the residual-trace handoff](../../../hypothesis/RESIDUAL-TRACE-HANDOFF.md)
and [full reference](../../../hypothesis/RESIDUAL-TRACE-THEORY.md). This document describes the earlier
instrument's actual capabilities. Its activity inventories and magnitude
shortlists do not establish the connected recurring trace of a capability.

The purpose is to give Marcos and an AI the same inspectable evidence about changing activity over fixed parameters, then test whether proposed relationships help predict controlled interventions. The ghost prototype owns the spatial representation. These artifacts do not prescribe a sphere, springs, distances, a hierarchy or rigid structures.

`qgre dev ghost-inspect` reads saved artifacts. It cannot launch inference, write weights, apply a patch or reconstruct a forward pass. There are two instruments: an observational inventory and a four-arm contrast. Neither reports a causal importance score.

## 1. Follow actual activity to candidate parameter groups

```sh
cargo run -p qgre -- dev ghost-inspect observe \
  --study .dev/decode-machine/public/evidence/e2b-30k/weight-study.json \
  --output runs/legs/gemma4-e2b/ghost-inspection-2026-11-05/observations.json \
  --top-per-layer 4
```

The reader checks the study schema, identities, matrix/input shapes, file bounds and SHA-256 of every consumed activation plane. It inventories all 32-coordinate groups of every shape-compatible routed `geglu → mlp_down` matrix, not just a picked subset. Missing or incompatible captures remain in the denominator and in `missing`. A matrix with no compatible capture gets no groups or ranking; its expected groups remain in `expectedGroups`.

Each observation identifies the saved capture, layer, input group, raw file range, original 32 code bytes and scale byte, nonzero coordinate count and exact maximum magnitude. It distinguishes a different stored encoding from a different numerical value. An encoded zero does not prove that a value was zero before quantization or that its parameter can be removed.

The shortlist sorts maximum observed magnitude **within each layer**, with stable coordinate ordering for ties. This is a reproducible place to inspect, not evidence of importance. It can favor outliers; recurrence and distinct states remain separate measurements. Equal states in eight snapshots do not establish rigidity, and different snapshots are not controlled interventions.

The existing E2B data names an MLP boundary for 35 layers and eight diagnostic positions, but inspection exposed a capture-coverage gap: the saved manifest's `geglu` stride is 6,144 for every layer, while checkpoint down-projection inputs are 12,288 wide in layers 15–34. The prepared index faithfully copied that original capture geometry. Those 160 rows are preserved as shape mismatches and excluded from ranking; layers 0–14 remain available. No missing tail or parameter mapping is inferred. The study also contains attention and other parameters; they are counted in the source inventory but are outside this first instrument's scope. It does not manufacture observations for the remaining generated tokens.

The logical group is linked to its down-projection matrix and the matrix's storage tensors. Its **binding to stored parameter columns remains unresolved**: runtime WHT, quantization and the actual captured executable must establish that mapping before proposing exact weight indices. This artifact never relabels tap coordinate 32 as weight column 32.

### Relational artifact: `qgre.ghost-observations.v1`

Read the tables as a path, rather than a bag of numbers:

`capture → observation → input group → matrix → storage tensor → source`

| Table | Identity and joins | What it establishes |
| --- | --- | --- |
| `captures` | Original `id`, reply and token correspondence | Which saved execution was sampled; the original correspondence caveats remain |
| `observations` | `id`, `captureId`, `groupId`, `activationId` | A measured state of one group, and exact byte ranges for inspecting it |
| `groups` | `id`, `matrixId`, half-open `coordinateRange` | A fixed identity across captures, with activity/state counts and coverage |
| `shortlists` | `layer`, `matrixId`, ordered `groupIds` | Which groups to inspect first under one explicit observational ordering |
| `activations` | Original `id`, code/scale paths, SHA-256, offsets | Recoverable raw recorded values, interpreted relative to the input study directory |
| `matrices` | Original `id`, `tensorIds`, input/output | The expected architectural relationship, not a measured physical transfer |
| `tensors`, `sources` | Original identities | Named parameter storage and the inherited evidence inventory |
| `relations` | `from`, `to`, `kind`, `basis` | Explicit architectural group-to-matrix edges |

`peakMagnitude = coefficient × 2^exponent2` is an exact description of a stored value. Counts and exponents are integer metadata. No floating tensor or alternate precision path is created. Value-state hashes use ordered canonical `(signed i16 coefficient, i16 exponent)` pairs, little-endian, after removing powers of two; all zero encodings map to `(0, 0)`. Encoded-state hashes use the 32 original codes followed by their scale byte. These hashes detect equality, not similarity.

For the prototype: keep each `group.id` fixed, vary its visible state from `observations`, and make a click open the original capture and matrix links. How those states become movement is deliberately left to the prototype. Unresolved coordinate bindings should remain visible.

## 2. Ask what A, B and A+B actually changed

```sh
cargo run -p qgre -- dev ghost-inspect compare \
  --experiment path/to/experiment.json \
  --output path/to/new-comparison.json
```

An experiment contains exactly four saved arms: `baseline`, `A`, `B`, `AB`. Each arm points to a `qgre.weight-probe.v1` JSON with one capture and matching recorded boundaries. The current offline extractor can produce that probe format from a saved case; a future intervention capture must use the same contract. The comparison cannot generate an arm for you.

The manifest has this shape; angle-bracket values are placeholders, not executable defaults:

```json
{
  "schema": "qgre.ghost-experiment.v1",
  "id": "<experiment identity>",
  "studyFingerprint": "<64 lowercase hex characters>",
  "hypothesis": "<the relationship being tested>",
  "prediction": "<expected boundary and behavioral changes, before collecting data>",
  "refutation": "<what would count against the prediction>",
  "interventions": [
    {"id": "a", "description": "<bounded edit A>", "patchSha256": "<digest>"},
    {"id": "b", "description": "<bounded edit B>", "patchSha256": "<digest>"}
  ],
  "arms": [
    {
      "role": "baseline",
      "probe": {"path": "baseline.json", "sha256": "<digest of that JSON>"},
      "interventionIds": [],
      "controls": {
        "baseCheckpointSha256": "<digest>",
        "engineSha256": "<digest>",
        "policySha256": "<digest>",
        "prefixSha256": "<digest>",
        "initialStateSha256": "<digest of the starting state, including KV state>",
        "parameterBasisSha256": "<digest of the coordinate/transform binding>",
        "position": 86,
        "inputToken": 2094,
        "seed": 1,
        "precision": "<actual representation>",
        "captureMode": "<actual capture mode>"
      }
    }
  ]
}
```

Add the other three arms with identical controls: `A` names `["a"]`, `B` names `["b"]`, and `AB` names `["a", "b"]`. Each has its own relative probe path and digest. Probe paths must stay inside the experiment directory. The checkpoint control identifies the common **base**, not an edited checkpoint. Patch digests and control identities are declarations at this stage; their equality is checked, but this reader cannot prove their application or that predictions predated the captures. It says that explicitly in its output.

### Relational artifact: `qgre.ghost-comparison.v1`

`experiment → intervention set → arm → recorded boundary → exact contrast`

Each comparison retains the boundary's layer, stage, shape, axes and coordinate binding. Its coordinate lists show:

- A, B and joint changes relative to baseline;
- nonadditivity: `AB − A − B + baseline ≠ 0` in the stored values;
- joint return to baseline after at least one individual change;
- a change present only in the joint arm;
- encoding-only changes, counted separately from numerical changes.

Calculations use exact dyadic coefficients and integer exponent carries. A tiny effect is not rounded away beside a large value. Non-finite codes, zero-scale/nonzero-code pairs, differing shapes/boundaries, duplicate identities, bad hashes and incompatible control declarations are rejected. Outputs are atomically published to new paths; an existing result is never replaced.

Nonadditivity can come from ordinary nonlinearities and quantization. Return to baseline can come from simple cancellation. Neither by itself establishes a stable structure or a useful edit. Behavioral benefit is explicitly `not measured`; observed contrast data must be paired with fixed-prefix logits or scores and independent behavioral evaluation before claiming success.

## What the first causal capture still needs

The next execution instrument must bind logical coordinates through the runtime transforms to exact parameter bytes; record the actual base/patch/executable/policy identities and starting prefix/KV state; capture baseline, A, B and A+B at matching boundaries; and repeat the unedited baseline to quantify nondeterminism. It must measure the preregistered behavioral target and held-out behavior, retaining harmful and null results. Every arm stays on QGRE's production megakernel path and precision policy.

This implementation makes existing observations inspectable and gives those future captures an exact comparison consumer. It does not yet apply interventions or establish which weights cause a behavior. No model generation is needed to run either reader.
