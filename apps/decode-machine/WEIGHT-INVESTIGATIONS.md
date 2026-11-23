# From a passage to a testable intervention

Purpose: Marcos and an AI should be able to inspect the same generation, select an exact token span, follow its recorded and architectural relationships to model internals and named parameters, and prepare an experiment whose results can be compared without confusing correlation with causation.

This extension is preparation only. It never starts inference, applies an edit, or changes the checkpoint. The next hypothesis comes from Marcos before any new generation.

## User journey

1. Select a reply and a half-open output-token range. Read the original text and exact vocabulary pieces; mark a second range as a comparison, without assigning either a quality label automatically.
2. See which recorded positions intersect that span. A token-ID correspondence from a separate diagnostic execution is not an identical production trajectory or a production timestamp.
3. Select a layer and a named matrix. Trace its input and output boundaries, inspect existing stored-code distributions, and open the same operation in the mechanical replay.
4. Read a bounded block of actual checkpoint bytes with its row, columns, scale bytes, file hash and storage basis. The view never presents a stored/rotated coordinate as a proven semantic neuron or an access trace.
5. Save a hypothesis, bounded weight-slice proposal, prediction, refutation and independent confirmation spans. Export a relational case and a capture request. The view takes in no outcome of an experiment: its import of paired results (answers marked correct, rescues, regressions, token cost) was removed on November 14, 2026, when instruments that measure progress by answers, tokens or words left this project; what an edit did is read from its computational trace, by the round-two readers.

## Relationships and evidence

The `qgre.weight-investigation.v1` document contains normalized sources, replies, captures, matrices, storage tensors and activation summaries. Selection and experiment state are stored separately in a `qgre.weight-case.v1` document that binds the dataset by hash. Exports include the current case, reading order and explicit relationships.

`recorded` means a saved observation or exact file metadata. `derived` means a deterministic computation over those observations. `architecture` means an expected input/output relationship; it is not a measured transfer. `proposed` means an analyst's experiment. Every relationship names its basis.

Token ranges are zero-based, start inclusive and end exclusive. A decode capture at position p consumes the generated token at p minus prompt length and predicts the following token. A two-token match is recorded as such; missing or disagreeing correspondence stays visible. No nearest capture is silently substituted.

Histograms count encoded 32-element blocks, not neuron importance. Raw checkpoint slices retain E2M1 codes and their stored scale bytes; no new precision or alternate forward implementation is introduced. Unmapped checkpoint tensors remain in the inventory. Existing timing rejection remains a rejection.

The prepared E2B study contains 187 replies and 30,000 generated tokens, eight saved diagnostic positions, 4,216 activation summaries and 2,797 stored tensors. Of 317 matrix entries, 205 have an architectural operation route, 40 follow a shared K/V owner, and 72 remain unmapped. A local K/V matrix is not marked active where the recorded schedule reuses an earlier layer's K/V state. These counts describe the saved study, not comprehensive telemetry for all 30,000 tokens.

## Next capture contract

Each proposed probe names execution identity, sample/span, token positions, layer and boundary, axes, representation, parameter basis and source artifact hash. Needed evidence includes token-aligned residuals before/after the selected operation, MLP inputs before down projection, attention probabilities where relevant, and exact parameter slices. Readiness is an inventory of what exists, not permission to infer missing values.

Shape-preserving edits need an exact sparse patch, base hash, changed indices, and original/new codes and scales. This application records proposals; it imports no outcomes and has no model-execution or weight-write endpoint.

## Research basis

- [Original work](https://rikiyat.github.io/blog/weight-space-programming/)
- [Qwen experiment and scope](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp8_math)
- [Actual editor prompt and measurement interfaces](https://github.com/RikiyaT/lm_understand-logs/blob/main/exp8_math/runs/p50c/turns/t001/prompt.txt)

The published Qwen edit is not evidence of a defect in this Gemma checkpoint. Repetition in the saved conversation is a selectable observation, not a diagnosed mechanism. No default hypothesis is supplied.

## Offline use

`bun tools/prepare-weight-study.ts <saved-run-directory> <public-evidence-directory>` rehashes the recorded checkpoints, inventories their tensors, indexes saved replies and admits existing native tap-range reports. It copies the raw activation planes alongside the index. It never runs a model.

`bun tools/extract-weight-probe.ts <weight-study.json> <downloaded-case.json> <new-output.json>` extracts the selected matrix block and the matching saved activation rows in their original encoded representation. The result includes offsets and hashes, the exact token correspondence, and missing evidence. The output path must be new. This provides an AI-readable numerical packet without inventing a decoder, evaluating the network, or requiring the AI to scrape a screenshot.

The local Vite preview offers a bounded read-only `/__weight-block` endpoint. It accepts a prepared study, a known matrix ID, a row and a 32-column block. It verifies the checkpoint hash and tensor bounds before returning code and scale bytes. Static builds retain the full investigation and export UI, but arbitrary checkpoint reads require the local preview or the offline extractor.

The native [`ghost-inspect` instrument](ghost-inspection/STUDY.md) now inventories saved MLP input groups across all available layers and captures, and compares baseline/A/B/A+B probe packets with exact stored-value arithmetic. Its relational exports are independent of the ghost's visual geometry. Activity candidates remain observational; binding them to exact transformed parameter coordinates and executing causal interventions are separate, still-needed steps.
