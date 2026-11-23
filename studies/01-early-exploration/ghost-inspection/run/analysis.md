# First Ghost Geometry observation pass

Source: the saved `gemma4-e2b-30k-2026-11-05` study, fingerprint `78a2f828e9e489769affb4d1b54c32221cd804e98c993940cb04bcf518061cff`.

Result: `observations.json`, schema `qgre.ghost-observations.v1`. This was produced by `qgre dev ghost-inspect observe` from existing files. No new inference, parameter edit or intervention was performed.

## Coverage

The source study contains 317 matrices. This instrument considers the 35 routed MLP down projections, at eight saved diagnostic positions. It expects 280 input planes and 10,560 groups of 32 coordinates across those matrices.

Only 120 planes from layers 0–14 have input widths matching the checkpoint. They yield 2,880 groups, 23,040 group observations and 737,280 sampled coordinate values. Sixty groups are shortlisted: four per admitted layer, ranked by maximum recorded absolute value within that layer. The shortlist is observational, not a causal importance ranking.

The other 160 planes, from layers 15–34, are explicitly excluded from ranking. The saved capture manifest allocates a 6,144-wide `geglu` stride uniformly. Those layers' checkpoint matrices have 12,288 input columns. The prepared study accurately reflects the original capture geometry, so rebuilding the index cannot recover the absent coverage. This is not evidence that generation skipped those weights; it is a limitation of what these recorded taps establish. The exact write-side behavior and captured executable must be bound before interpreting any of those narrow rows as a specific portion of a wider layer.

## What the values say

Of the 737,280 admitted coordinate observations, 694,917 have nonzero stored values. Every admitted 32-coordinate group has at least one nonzero value at all eight positions. No group has an identical canonical value state across all eight positions.

Consequently, a binary active/inactive group display would provide little separation in this sample. The exported code bytes, scale bytes, exact peak magnitudes and stable group identities allow the prototype to represent more than a switch. Neither widespread activity nor changing states establishes influence on a chosen output.

For a concrete first inspection, layer 0's highest-peak group is the half-open tap-coordinate range `[5216, 5248)`. Its maximum recorded magnitude is exactly 56 (`7 × 2^3`), it is active in eight of eight captures, and all eight recorded value states differ. Its architectural relationship is the input to `model.language_model.layers.0.mlp.down_proj`. These are tap coordinates, not certified checkpoint column indices.

## What remains before a causal verdict

1. Correct and verify the capture geometry for the wider MLP layers, retaining the old recordings as limited evidence.
2. Bind the tap coordinates through the runtime transforms and quantization to exact parameter bytes.
3. Capture baseline, A-only, B-only and A+B under the same prefix and starting state, plus an unedited repeat. The new offline comparison reader can consume those arms, but no such experiment was executed here.
4. Measure the predicted behavioral effect and held-out behavior. Nonadditivity or restoration of a stored value is not by itself a discovered constraint or an improvement.

The artifact imposes no spatial layout, distances, clusters or stiffness. It supplies measurements and relationships for the separate ghost prototype.
