# Measured native keys and finite forecasts in the installed learner

The prospective launcher uses the same recorded single-pair learner, reference and local objective. The optional graph captures the pre-cache key at each executed producer during the ordinary full decode, and anchors each later candidate to those observed keys. On a registered local signal the learner reports the continuous JVP, rounded-displacement JVP, finite rounded-weight forecast, and the next observed free generation separately. [Public commands and output paths](../../../code/signal-trace/TRAINING.md)

## Computation and scope

For each producer, the native index supplies the exact operator, call, microbatch, positions, head width and key-head count. Every original recorded producer and position must be present once, and the native producer calls must agree with the measured projection geometry. The original model provides the frozen adapter key on its own incoming residuals. For a trial, the graph retains the current model's norm, projection and positional rotation, adds its change from that frozen key to the original measured FP32 key, and applies the historical FP16 cache cast. Native Q and V have no new anchoring operation. The original table of shared producer keys still supplies later consumer layers. [Key reader and graph wrapper](../../../code/signal-trace/pipeline/weight_edit/native_key_anchor.py)

The public path requests the existing recorder's projection scope with a per-case label filter for the old residuals and the executed key outputs. The prompt and generation budget are unchanged. Every capture binds both the original case file and its effective recorder case, installed model identity, actual invocation and output stream hashes. The historic residual reader receives its original two-axis records, and the native keys are read in native three-axis order. On an actual mixed Gemma stream, the incoming and merged arrays at the shared ADD were bitwise identical to the previous residual-only acquisition over all 578 events. Mixed stream check · [Recorder and controller integration](../../../code/signal-trace/pipeline/weight_edit/native_key_driver.py)

The optional finite reader reuses the physical candidate bytes already obtained by the normal preview, storing each changed coordinate and its decoded value. During one held-history graph forward, a sparse overlay materializes tensors on demand and restores the derivative model on exit. No checkpoint or full candidate model is duplicated. The registered original sites and the original native anchors stay fixed. Ambiguous visits keep their individual scores. A missing site is an empty result. A failed optional finite read is reported as unavailable. Physical installation and fresh decoding still belong to the original learner. [Finite scoring](../../../code/signal-trace/pipeline/weight_edit/native_key_finite.py)

## What ran

The production finite reader was exercised on a historical Gemma pair using all 15 measured key producers, all 105 residual merges and the exact rounded candidate. At the previously studied signed-turn cell its finite forecast was **+0.0001276433**, the rounded JVP was **−0.0000049260**, and the observed native change was **+0.0001103947**. At the earlier automatically registered cell its finite forecast was **+0.0000333020** and the observed native change was **+0.0000428412**. The latter stayed within tolerance. [Held Gemma replay](../../../studies/04-round-2-overlap/runs/finite-installed-diagnostic-replay/README.md)

Fourteen focused CPU tests passed across the native key, registered diagnostic, full graph and installed-weight suites. The combined fixture constructed the complete 35-block graph with all 15 synthetic producer captures, measured both JVPs and the finite endpoint, installed a rounded edit, and regenerated from the updated native fixture. Other checks compared reverse and forward key derivatives, shared-cache consumers, missing or shifted native calls, repeated/absent sites, source restoration, and the previous baseline. No full Gemma learner step ran in this validation. [Native-key tests](../../../code/signal-trace/tests/test_native_key_anchor.py) · [Combined controller test](../../../code/signal-trace/tests/test_trace_diagnostic.py)

The public dry preparation and check from step 390 bound 191 implementation files and 538 eligible tensors. At the declared 2,048-event capacity, the additional key data reserve is 37,748,736 bytes. A widened index estimate adds 27,617,280 bytes, for 65,366,016 additional bytes per capture. The persistent native and computed key arrays require at most 75,497,472 bytes at this context. A finite observation also stores indices and F32 endpoint values for each changed coordinate, materializes requested changed tensors, and incurs a forward pass. Its full device and host high-water marks during native training remain unmeasured. The preflight found 59,889,266,688 bytes available against a total reserve of 60,533,054,736 bytes. Measured sizing · Preflight

To repeat the read-only capacity check from the repository root, run the following in Bash. The temporary file is the draft declaration. Neither command creates a research run or reserves disk.

```bash
export PYTHONPATH="$PWD/code/signal-trace"
native=(uv run --offline --python 3.12 --with torch==2.15.0 --with numpy==2.5.5 --with matplotlib==3.11.2 python -m pipeline.weight_edit.native_key_experiment)
runs=studies/04-round-2-overlap/runs
output="$runs/installed-keys-review-only"
draft=$(mktemp /tmp/installed-native-keys.XXXXXX.json)
"${native[@]}" prepare "$output" --from-run "$runs/vector-chance-final-experiment-installed-348-recovery" --review reports/descent-trace-repair/review.md --intent records/development/2026-11-18-installed-native-key-anchor/preflight-intent.json --trace-diagnostic records/development/2026-11-18-whole-prefix-diagnostic/diagnostic-spec.json > "$draft"
"${native[@]}" check "$output" --declaration "$draft"
```
