# Gemma E2B: repeated numeric answers to association questions

The completed twenty-case original-model run produced **8 numeric-substitution errors**, **5 correct values**, **4 other listed values**, and **3 answers cut off at eight tokens**. Two numeric failures were previously observed anchors; six are new cases. This is a targeted development set, not an estimate of general model accuracy.

The numeric failures span five requested values: blue, Falcon, Violin, Silver, and Train. Their outputs include 1, 2, 3 and 24. Only two of the eight numeric answers match the requested row number. The observed failure class is therefore number-instead-of-associated-word, not a consistently correct row-index operation.

## Actual outputs

| Case | Requested value | Requested row | Actual text | Short-run classification |
|---|---|---:|---|---|
| known-01 | blue | 2 | 2 | numeric-substitution |
| known-03 | blue | 2 | 3 | numeric-substitution |
| probe-03 | mango | 1 | Mango | correct-value |
| probe-04 | mango | 2 | Mango | correct-value |
| probe-05 | mango | 3 | Mango | correct-value |
| probe-06 | falcon | 1 | Panda | wrong-listed-value |
| probe-07 | falcon | 2 | 3 | numeric-substitution |
| probe-08 | falcon | 3 | The question asks for the value of " | incomplete-at-cap |
| probe-09 | violin | 1 | Please provide the table you are referring to | incomplete-at-cap |
| probe-10 | violin | 2 | 3 | numeric-substitution |
| probe-11 | violin | 3 | 3 | numeric-substitution |
| probe-12 | silver | 1 | Copper | wrong-listed-value |
| probe-13 | silver | 2 | 24 | numeric-substitution |
| probe-14 | silver | 3 | Silver | correct-value |
| probe-15 | oslo | 1 | Tokyo | wrong-listed-value |
| probe-16 | oslo | 2 | The question asks for the value of " | incomplete-at-cap |
| probe-17 | oslo | 3 | Lima | wrong-listed-value |
| probe-18 | train | 1 | Train | correct-value |
| probe-19 | train | 2 | 1 | numeric-substitution |
| probe-20 | train | 3 | 1 | numeric-substitution |

The classifications above preserve incomplete answers. The existing runner's strict correct=false on a truncated text does not establish that its eventual answer would be wrong.

## A matched failure/success relationship

The same three metal mappings and the same question yield:

- Row 1: Nek=Silver, Jup=Copper, Foz=Gold → Copper.
- Row 2: Jup=Copper, Nek=Silver, Foz=Gold → 24.
- Row 3: Jup=Copper, Foz=Gold, Nek=Silver → Silver.

The requested association remains Nek=Silver. Reordering its presentation changes the outcome. The row-2/row-3 pair gives a concrete failed and successful computation to inspect. It does not establish which internal component caused the change.

For comparison, Vem=Mango returns Mango in all three positions. Pav=Train returns Train in row 1 and 1 in rows 2 and 3. A theory has to explain both the failures and these preserved or changed behaviors.

## Run and evidence identity

- Original published TQ4 E2M1 checkpoint, no weight edits.
- Checkpoint root: a4e166e6798bfc833a99ee08460b58f0c305695ca077bc7c810f89646a7caab2.
- Original saved qgre-infer SHA256: 7bf8f6a1151b41887c6a9da6ec45a6db16b7a0ea91d9e27c8105c4be12dd1524.
- Corpus SHA256: db2da35d0badc63dc0c674af1126b0d2db83a751efd2cd2a5e56e787ae423a6d.
- Production planner/megakernel, device 0, fresh process and prefill for each case, greedy, cap 8 as in the original discovery experiment.
- Twenty questions: two exact known-failing prompts and eighteen variants from six three-row tables, each requested association moved through all three rows.
- Historical anchors used device 1; both now reproduce their exact answer and emitted token IDs on device 0. This does not establish universal cross-device equivalence.
- The saved executable's source revision is not established. Its executable and checkpoint identities are recorded; no source-head or performance claim is made.

Corpus, raw run results, run identity. Each case directory preserves stdout and stderr, including emitted token IDs.

## Other question wording remains valid evidence

An earlier What is <key>? batch was stopped after an assistant overcorrection about preserving exact wording. The user clarified that different wording can reveal the same mechanism. Exact wording is not a requirement for membership in a failure group; actual observed behavior determines membership.

That earlier batch has four completed semantic successes, one 128-token incomplete explanation, and one interrupted request with no answer. Its files remain in ../behavior-pattern-20-2026-11-06/. It was not a completed twenty-case run and does not show that its question form cannot exhibit the failure.

## Follow-up plan and scope limits

The short pass left three cut-off cases (probe-08, probe-09, probe-16). The planned follow-up used the same prompts with cap 128 and checked their first eight emitted token IDs against the short runs. The Silver row-2/row-3 pair received answer-boundary captures and named KV views, with output-token agreement and prompt positions checked before joining those captures. Completion and remaining limits are recorded below.

No native alternative-token probabilities, common internal direction, causal component or successful edit has been established by this behavioral run. It supplies repeated failures across different content plus successful comparisons. The proposed geometric explanation remains a hypothesis to investigate with internal observations and controlled interventions.

## Follow-up observations completed

All three extended runs preserved the exact first eight token IDs of their short runs, and their prompt-token counts matched.

| Case | Longer observation | Classification |
|---|---|---|
| probe-08 | Explicitly states that Sop's value is Falcon, continues explaining, reaches 128-token cap | Correct association reached; verbose and unfinished |
| probe-09 | "Please provide the table you are referring to. I need the table to answer the question." Stops naturally after 19 tokens | Fails to use the supplied table; distinct from numeric substitution |
| probe-16 | Explicitly states Zef is associated with Oslo, continues reconsidering the question, reaches 128-token cap | Correct association reached; verbose and unfinished |

These follow-ups do not increase the eight-case numeric-substitution cohort. The two 128-token runs remain incomplete; no eventual termination or final answer is inferred.

The Silver pair's native captures both preserve the entire original emitted token sequence: probe-13 produces [236778, 236812, 107, 106] (24 plus newline/EOT), and probe-14 produces [48659, 106] (Silver/EOT). Each has 46 prompt tokens and captures position 45, the final prompt position that produces the first answer token. Each contains 17 named stage categories over the 35-layer architecture and 35 named KV layer views, including shared-layer views. This is one captured position per run, not a full generated-token trajectory.

Failure capture, successful capture, and relational evidence link questions, runs, row-variant groups, failure observations and token-matched captures. All five follow-up executions completed successfully. No weights were edited.

### What this now supports

There is a reproducible number-instead-of-associated-word failure across different inputs, requested values and returned numbers. The literal-row-index-only explanation is contradicted by six of its eight numeric responses. The Silver pair isolates an observable change in behavior while preserving the requested association and question wording.

The internal cause is still open. One candidate is a context-sensitive error in choosing or executing the requested answer relation; the observed classes could also contain more than one cause. The captures permit inspection of that candidate. They do not establish a unique vector, a safe edit or the proposed causal role of geometry in stability.
