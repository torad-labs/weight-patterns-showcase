# Read this experiment as a connected story

1. **The physical question.** `cases.json`, `cases.after-calculation.json`, and
   `cases.numeric-calculation.json` keep the 18 prompts, their shared families,
   changed condition, and independently computed expected quantities. PLAN and
   the three AMENDMENT files explain what changed and why.
2. **What the model actually said.** Each response-format directory preserves
   the request, tokenization, response, token IDs, and the stopping condition.
   `behavior-assessment.json` separates verdict correctness from the physical
   calculation. `server-native-comparison.json` prevents conflating different
   executions of the same prompt.
3. **Can the observation be trusted?** The `qualification-*-comparison.json`
   files compare ordinary execution, observation without saving, and observation
   with persistence. `deep-unfused-comparison.json` and
   `deep-boundary-comparison.json` extend that comparison through the complete
   recorded runs. Only scores with matched generated histories are compared.
4. **Follow an execution.** `deep-v2-unfused-capture/<case>/case.json` identifies
   the input; `result.json` joins generated decisions to full vocabulary scores
   in `logits.f32`. `records.jsonl` joins each tensor event to its phase, input
   position, call, microbatch, structural node, native type and binary interval
   in `tensors.bin`. `graph.json` carries source-node edges and operations.
   Structural identity alone is not runtime KV provenance. Default-fusion,
   layer-exit observations use the same organization in
   `deep-default-boundary-capture`.
5. **Inspect the connected changes.** `connected-observations.json` links
   selected generated-token contexts to consecutive recorded transformations;
   its counterfactual joins the two coaster executions at equal generated
   prefixes. `connected-observations-boundary.json` provides the coarser view.
   The two `counterfactual-*.svg` figures show relative separation. They are
   derived views of the data, not recovered geometric coordinates.
6. **Inspect the exploratory recurrence.** `local-recurrence.json` enumerates all
   35 layers across seven sampled decisions. `local-segment.svg` overlays the
   connected layer-14/layer-15 progression. The sample is one physical prompt
   pair. The region was noticed afterward; this is a candidate to follow, not
   independent confirmation, an exclusive physics signature, or an edit target.
7. **Reproduce and account for limitations.** `instrumentation-v2` preserves the
   recorder and analysis sources and their execution provenance. The project
   report is copied to `REPORT.md`. Earlier failed recorder attempts are retained
   and described in provenance. Identical plain/capture logit files may share
   filesystem storage after a complete byte comparison; all paths and bytes are
   preserved.

No model parameters changed. No training or BF16 run took place. All raw captures
remain on the project disk; the factory UI has not imported this new trace format.
