# Read the follow-up as connected evidence

1. `PLAN.md` fixes the question and scope before this run. `cases.json` connects
   each question to a physical family, variation, matched pair, expected energy
   progression and response budget. Arithmetic comparisons supply the operations;
   they do not test deriving them from physical circumstances.
2. `qualification-comparison.json` connects the new recorder to ordinary/sham
   execution and the original pilot's actual logit bytes. `study-comparison.json`
   checks ordinary/capture agreement across every decision of all 24 cases.
   Full-vocabulary SHA256 values preserve byte-agreement evidence, not all scores
   for future reconstruction. Top eight values and selected tokens are retained.
3. `study-{plain,capture}/<case>/case.json` contains the actual tokenized prompt.
   `result.json` retains every generated token, its raw bytes, full-score hash,
   top candidates and stop reason. Judge calculations from the actual response.
4. Each capture's `records.jsonl` connects an event to its input position, phase,
   operation, call, microbatch and byte interval in `tensors.bin`. `graph.json`
   connects source operations. Only the declared tensor payloads are retained;
   other callback events explicitly say they were omitted. Structural identity
   does not recover mutable KV-slot provenance or fused attention internals.
5. `connected-contrasts.jsonl` follows every eligible shared-prefix decision
   through layer exits and the connected internal additions at layers 12-16.
   Before/after events retain their source-node identities and binary offsets.
   Absolute differences, relative differences, state growth, update alignment
   and direction changes are views of recorded vectors, not a declared geometry.
6. `analysis.json` groups these progressions by physical pair, paraphrase and
   arithmetic contrast, preserves denominators and full responses, and reports
   actual-addition checks. `behavior-assessment.json` records the manual reading
   of calculation quality separately from final verdict and truncation.
7. `CALCULATION-ANCHORS.md` explains the post-hoc selection of eight observable
   calculation moments. `milestones.json` fixes their token/byte locations;
   `milestone-trails.json` joins those moments to actual connected state changes,
   graph sources and raw tensor offsets. Different generated histories remain
   distinct; these anchors do not establish causal responsibility or faithful
   internal reasoning.
8. `REPORT.md` states the findings and their limits. `instrumentation/` preserves
   the scripts/source used; `instrument-sha256.txt` binds recorder and library
   binaries. This is an external llama.cpp study, with no checkpoint edits.
