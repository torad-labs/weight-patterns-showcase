# Follow a question through both checkpoint representations

1. `PLAN.md` states the authorized comparison and its interpretation boundaries.
   `cases.json` connects the unchanged questions, pair identities, physical
   quantities, expected calculation and response limits. The preceding Q4 study
   lives at `../physics-trace-followup-2026-11-06/`.
2. `publisher-metadata.json`, `provenance.json` and `metadata-comparison.json`
   connect the files to their publisher revision, checked model checksum,
   unchanged observer/library and matching tensor names/dimensions. The native
   metadata dumps preserve tensor types. Matching topology does not prove
   identical effective weights or isolate every source of arithmetic difference.
3. `qualification-comparison.json` and `study-comparison.json` establish recording
   fidelity within BF16 and prompt-tokenization agreement with Q4. They do not
   require the two precisions to produce identical answers or scores. Full-score
   hashes cannot reconstruct logits; top eight values and exact token bytes remain.
4. `study-{plain,capture}/<case>/result.json` connects the actual generated text
   to every selected token, output candidate, input position and stop condition.
   Judge the complete calculation separately from its final verdict.
5. A capture's `records.jsonl` or lossless `records.jsonl.gz` connects each event
   to its call, position, graph node and native tensor-byte interval. The gzip
   companion JSON records a verified full decompression checksum and length.
   `graph.json` preserves source operations; `tensors.bin` holds original native
   float32 intermediate bytes. Runtime KV-slot lineage and fused attention
   internals remain unobserved. Omitted payloads are explicitly marked.
6. `connected-contrasts.jsonl` and `analysis.json` follow each within-BF16 prompt
   pair through its common generated history. `cross-precision-analysis.json`
   and `cross-precision-connected-contrasts.jsonl` instead follow each identical
   prompt with Q4 on the left and BF16 on the right, stopping after the first
   divergent output decision. Sources and tensor offsets make those different
   comparisons distinguishable. Later histories remain separate observations.
7. `behavior-assessment.json` records the reading of every response. The report
   connects that behavior to measured changes, preserves failures and truncation,
   and states what remains unestablished. `instrumentation/` preserves the source
   used to obtain and analyze this evidence. No training or weight edit occurred.

8. Read [REPORT.md](REPORT.md) for the completed comparison. summary.json joins the within-BF16, Q4 and cross-precision populations without merging their denominators. four-way-matched-comparison.json further limits comparison to generated histories shared by all four runs.
9. CALCULATION-ANCHORS.md describes the nineteen text-selected moments. milestones.json binds literal text bytes to prediction indices; milestone-trails.json binds those predictions to actual graph nodes and tensor offsets. The block figure shows four such moments; the precision figure shows first-decision within-pair trails.
10. posthoc-layer16-comparison.json retains every eligible observation behind the exploratory layer-16 comparison, including all 28 positions where BF16 did not have the larger ratio. The earlier FF16-only slice is supplemental exploration, not an independent experiment. COMMANDS.md and instrumentation/SHA256SUMS preserve reproducibility. behavior-notes.json is the initial twelve-case draft; behavior-assessment.json is the reviewed complete twenty-four-case assessment.
