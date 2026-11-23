# Running the installed-weight baseline

The Gemma 4 E2B experiment edits the weights to bring one regenerated case closer to one fixed original reference trace. Local generations 0–42 in the [recorded continuation](../../studies/04-round-2-overlap/runs/vector-chance-final-experiment-installed-348-recovery/declaration.json) correspond to historical steps 348–390. The [systems architecture guide](ARCHITECTURE.md), [experiment specification](../../FINAL-EXPERIMENT.md), and [working paper](../../reports/descent-trace-repair/README.md) describe the implementation and empirical findings.

## What an update does

The native decoder records incoming and outgoing residual states at three merges in each of 35 blocks. At each event and merge, the fitted turn is the difference between the unit outgoing and unit incoming state. The controller aligns the sequence of turns to the fixed reference by computational similarity and a monotone path. Each aligned cell supplies its own vector residual outside its calibrated radius. Their weighted squared norms sum to the scalar objective. The original calibration includes zero-radius cells.

The negative weight gradient includes the individual residual derivatives over the full applicable graph. It reaches 538 eligible learned tensors. The controller computes the response to that direction with a Jacobian-vector product, sets the step length to the retained empirical rate times the response-scaled length, predicts native rounding, and installs the resulting BF16/F32 values with FP32 remainders. The derivative uses a straight-through estimator for rounded operands. The next observation comes from a fresh decode of the same prompt using the installed native descriptor. Alignment, history and gradients can all change. The reference stays fixed. Direction/log-magnitude overlays and operation-specific descent scores remain diagnostic in this baseline.

The [executed recovery controller](../../records/development/2026-11-17-unmeasured-state-recovery/run.py) first measured the installed historical step 348 because the preceding job stopped between its update and acquisition. For a complete, measured checkpoint such as 390 the [measured continuation controller](../../records/development/2026-11-17-final-experiment/run.py) runs the same update code with the measured-state admission. Both use one mutable working model with chunked writes and one complete exit checkpoint. A clean boundary after a requested stop can be restored. An incomplete installation is marked partial and cannot serve as a measured starting state.

## Inspect or prepare without starting training

Use Bash from the repository root. The recorded runtime uses Linux, Python 3.12, the CUDA build of PyTorch 2.15.0, NumPy 2.5.5, and Matplotlib. The command below pins Matplotlib to the version used for the packaging checks. Research execution also requires the bound local generation binary, libraries, qualified GPU, model seed, original reference and the complete native and remainder payloads. The declaration records the hardware and source files.

```bash
export PYTHONPATH="$PWD/code/signal-trace"
py=(uv run --offline --python 3.12 --with torch==2.15.0 --with numpy==2.5.5 --with matplotlib==3.11.2 python)
trainer=("${py[@]}" -m pipeline.weight_edit.final_experiment)
runs=studies/04-round-2-overlap/runs
prior="$runs/vector-chance-final-experiment-installed-348-recovery"
"${trainer[@]}" --help
```

A new continuation has its own research question, review, output directory, and declaration. Supply a JSON object in `intent.json` with five text fields, `why_this_direction`, `repeating_pattern_check`, `hypothesis`, `prediction`, and `would_weaken_it`. The project’s whole-build review is a local file under the repository. Substitute its path and the intended new run name below. The first two commands only read research inputs.

```bash
next="$runs/<new-run-name>"
review="<whole-build-review-path>"
"${trainer[@]}" prepare "$next" --from-run "$prior" --review "$review" --intent intent.json > /tmp/next-declaration.json
"${trainer[@]}" check "$next" --declaration /tmp/next-declaration.json
```

The draft lists each implementation binding and the input inventory. The check shows the exact seed, trace, tolerance, case, payload digests from the checkpoint manifest, hardware identity and available storage. It hashes the declared implementation, verifies the checkpoint metadata and checks payload sizes. Full payload hashes are checked when the native state is restored. The disk reserve includes room for the exit checkpoint, next capture and existing project free-space floor.

## Execute an approved continuation

```bash
"${trainer[@]}" prepare "$next" --from-run "$prior" --review "$review" --intent intent.json --write
# Record the run name and prepared status in studies/04-round-2-overlap/ROUND-2.md.
"${trainer[@]}" run "$next"
```

The native controller runs the project verifier, validates its input and source bindings, takes the sole-controller lock and acquires a fresh baseline. Its local `loop/N/measurement.json` ties each fresh objective to the installed model revision, prompt, capture, and correspondence. The MLflow metric is `native.objective`. Predicted responses and actual fresh loss have separate names. Send SIGUSR1 to request a transition-boundary stop. For a measured exit use its directory as the next `--from-run`. A new output directory preserves the old records. Recovery from a fully installed but unmeasured predecessor is limited to the explicitly documented reserve-failure admission. The release has no automatic policy for other failed acquisitions.

## Bounded CPU verification

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "${py[@]}" -m unittest tests.test_final_experiment
```

Or using the `pyproject.toml` project configuration inside `code/signal-trace/`:

```bash
cd code/signal-trace
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m unittest tests.test_final_experiment
```

The test builds a tiny 35-block model with the same parameter families and native storage types. It decodes from the same prompt on independently reopened bytes, applies two trace-derived updates, and compares the frozen reference and calibrated radii after each. At the boundary it rejects a deliberately corrupted remainder before installation, restores a complete checkpoint into the original seed, and compares native and remainder bytes. This exercises the learner plumbing on CPU. The scientific measurements and the CUDA/native recorder evidence come from the recorded continuation above.

## When to stop following the fitted reference

Add `--stopping-policy path/to/policy.json` to the `prepare` command. The whole-trace policy uses the learner's own native acquisition, full-coverage correspondence, local tolerance and weighted loss. It makes an operational checkpoint decision. A behavioral evaluator can subsequently inspect the selected checkpoint. Capability acquisition, duration and generalization remain separate measurements.

Register the fitted prompt, the development prompts, their frozen targets and the family identifiers reserved for the final evaluation. The trained gradient still receives one fitted trajectory. At the initial revision and the scheduled boundaries, the controller regenerates every development response to its natural end on the currently installed weights. At the initial revision it performs a second acquisition at unchanged weights for both the fit and every development prompt. The observed difference measures the resolution for that prompt and that revision. One repeat does not characterize the distribution of possible future histories.

The local maps retain all reference/current correspondence pairs and every merge. The controller marks the reference cells for which **all** baseline matches lay inside the local tolerance. For each fresh trace it reports the ordinary whole-trace discrepancy, how many of these cells now have an outside match, and the summed local loss on those cells. This distinguishes a decrease in total loss from movement of previously agreeing computations outside tolerance. The initial mask remains fixed throughout. Loss on later newly repaired cells contributes to the ordinary local maps. The scalar retention summary specifically protects the initial in-tolerance set.

The history in the calibration file contains the original contiguous fresh fitted observations. One more than the longest dry spell that recovered to a new fitted low sets the interval. The number of historical updates sets the finite compute budget. The median absolute deviation of changes between adjacent windows in the historical loss measures training variation. The 348–390 trajectory yields an eleven-update interval and a 42-update budget. A calibration history with no recovered reversal uses every update for evaluation and provides a budget-only endpoint. Where fewer than two historical window changes are available, the selector also uses the budget endpoint.

A checkpoint becomes eligible when at least one recorded discrepancy improves beyond baseline replay variation and the other discrepancies across all traces remain within their selected values plus replay variation. Ordinary updates can still move losses in either direction. A departure of previously agreeing cells beyond the replay range at two consecutive scheduled checks can end the run. Two complete intervals with no resolved development gain can also end it. The receipt distinguishes a still-declining fitted loss from a fit that has stopped improving beyond measured variation. An incomplete trajectory or a partial output cannot establish a favorable result. The budget ends the study if these events do not. A zero fitted loss and a universal percentage reduction have no special stopping status.

### Register the whole-trace policy

All bindings below contain the repo-relative `record` and its `sha256`. Each development member supplies one native case file with its unique `id`, prompt `tokens`, and a declared `max_tokens`. Prompt tokens serve only to generate each trajectory and verify input identity. The policy compares computation across the generated sequences. The calibration file contains `{"fit_history": [...]}` with ordered bindings to native `measurement.json` records of one fitted trajectory. Registration verifies their model/capture and prompt identity, completeness and loss identity. The final-test family IDs are excluded from fit and development. An evaluator must establish the substantive independence of the named families.

```json
{
  "format": "whole-trace-development-stopping/1",
  "fit": {"id":"<fitted native ID>", "family":"<fitted family>", "tokens":[1,2]},
  "held_families": ["<final test family ID>"],
  "panel": [
    {"id":"<development ID>", "family":"<development family>",
     "cases":{"record":"<one native prompt JSON>", "sha256":"<digest>"}}
  ],
  "calibration":{"record":"<fit history JSON>", "sha256":"<digest>"}
}
```

Each case defaults to the learner's frozen reference and tolerance. To register another frozen development target, add `target` with the same `reference`, `tolerance`, and `tolerance_policy` fields accepted by the fitted native controller. A different target may be appropriate when the computations needed for that prompt differ. This choice must be made before the run. At evaluation the same native loader checks its original-model provenance, complete geometry and declared tolerance. The baseline's original zero-radius cells remain visible.

Fresh complete or capped native measurements are under `validation/<revision>/<member>/`. The measured fit is already in `loop/<revision>/`. The development assessments retain native measurement bindings, the original masked cells, full local maps and the unchanged-weight range. The selected complete state is recorded by `validation-decision.json`. A selected intermediate checkpoint is saved once per promoted revision. Continuation replays the hash-bound decision history and preserves the remaining budget. The final held prompts and judgments stay outside this selector. The focused CPU fixture exercises registration, acquisition through the documented native routine, local retention under changed correspondence, stoppage and recovery. Native training with this new selector and a prospective scientific panel remains unmeasured.

### Earlier physics diagnostic

The older `development-operation-stopping/1` format is retained for replay of the separately identified physics diagnostic. It registers a domain assessor, manually selected operation regions and arithmetic labels. Its observations answer the domain-specific audit question. The research stopping path above takes no scorer, physical operations, answer outcomes or hand-selected cells. [Archived format and executable examples](tests/test_validation_stopping.py)

The measured-continuation preflight also checks the archived retired predecessor when restored. It verifies the historical file digest, archive registry and readback evidence, current restore receipt, original-model identity and retired writer. A changed inode is accepted only after the complete restored file passes the bound SHA-256. Historical declarations retain their original admission.

## Observing a pre-decision signal during an installed update

Pass `--trace-diagnostic path/to/policy.json` to the same `prepare` command. The `installed-trace-diagnostic/1` registration binds the `reader` Python source, the frozen JSON `artifact`, and lists of source `dependencies` and data `inputs`. Each binding carries a repository-relative `record` and its SHA-256. The prospective declaration records the reader, hooks, runtime and artifact. The first reader is [paired_turn_signal.py](pipeline/weight_edit/paired_turn_signal.py). Its artifact specifies the merge, bank of paired turns, signed direction and match radius. Increasing score means movement toward the failure side of this fitted contrast. Its scientific specificity must be established separately.

The reader locates eligible sites from a complete generated computation. Its `locate(native, artifact, *, prompt_events)` returns logical keys, event indices and merge indices. `prompt_events` is the number of consumed prompt tokens, so event `prompt_events - 1` predicts the first generated token. The reader's differentiable `score(forward, cell, artifact)` evaluates the selected site. During the existing two forward-mode products, the runtime keeps that registration fixed. The continuous tangent is multiplied by the learner's applied length. The second tangent uses the actual native-rounded displacement with coefficient one, including zero physical movements. Neither observation contributes to the gradient or chooses the update.

Each `loop/N/trace-diagnostic/fresh.json` records the native score, selected sites, their local discrepancy, and the fitted measurement. `proposal.json` beside it stores the two predictions about the next installation. The next revision's `transition.json` compares them against an independently registered fresh decode. Missing and ambiguous matches remain visible. When the entire consumed history is naturally identical after the edit, the receipt also scores the original sites on the new native values. Otherwise the actual fixed-history observation is unavailable. A separately forced replay was not executed. A gap between a JVP and the ordinary regenerated score includes finite-step, history and registration effects.

The recorder stores `native.trace_signal.match_count` and, when uniquely observed, `native.trace_signal.score`, `proposal.trace_signal.continuous_change`, `proposal.trace_signal.rounded_change` and `native.trace_signal.fresh_change`. Per-site values and the corresponding source and target model identities remain in the local receipts. The CPU test invokes the public command and its actual diagnostic driver with a tiny native-format model, the research reader, both existing JVPs, an installed edit and a fresh autoregressive decode. The test substitutes the machine-specific run admission and capture. [Runtime test](tests/test_trace_diagnostic.py)

The first [registered signal](../../studies/04-round-2-overlap/runs/discovered-turn-stage/diagnostic-spec.json) can be supplied directly to `--trace-diagnostic`. Its [discovery record](../../studies/04-round-2-overlap/runs/discovered-turn-stage/result.json) selected merge 0 on a 105-merge scan. A downstream match and ambiguity on a valid control limit its use as a precursor. It remains an observational signal. [Public preflight and source bindings](../../records/development/2026-11-18-installed-trace-diagnostic/preflight.json)

The [whole-prefix policy](../../records/development/2026-11-18-whole-prefix-diagnostic/diagnostic-spec.json) selects the earlier fixed-reference direction and log-gain score through the same `--trace-diagnostic` option. For each generated event, it aligns all available turns across all merges against the frozen reference from its first generated event. The policy reports every visit to the selected reference event. At a visit its scalar is the local combined turn and log-gain distance minus its frozen successful cutoff. Positive values exceed the cutoff. Zero visits and multiple visits retain their ordinary diagnostic statuses. The full field used for registration stays outside the derivative. Both weight-direction products differentiate the registered local scalar.

On archived Gemma responses, the earlier score detected one of two same-family mistakes 15 events before the audited prechoice. The correct response stayed within its cutoff. All six visits on the reordered challenge occurred after the wrong descent. [Frozen site, native replay and controller verification](../../records/development/2026-11-18-whole-prefix-diagnostic/README.md)

## Run with native pre-cache keys and a finite candidate forecast

A prospective continuation can use the [native-key launcher](pipeline/weight_edit/native_key_experiment.py) with the existing installed-weight learner. Its `prepare` accepts the same arguments as the baseline. Use `--trace-diagnostic` to name a source-bound reader whose registered cells should receive finite and first-order forecasts. For example, from the repository root with the shell variables above and the research review and intent for the proposed run:

```bash
key_trainer=("${py[@]}" -m pipeline.weight_edit.native_key_experiment)
policy=records/development/2026-11-18-whole-prefix-diagnostic/diagnostic-spec.json
"${key_trainer[@]}" prepare "$next" --from-run "$prior" --review "$review" --intent intent.json --trace-diagnostic "$policy" --write
# Record the new declaration in ROUND-2.md before launch.
"${key_trainer[@]}" run "$next"
```

Each ordinary free generation now retains its original residual data and the post-RoPE FP32 keys at all executed key producers. Each graph reconstructs the key as the measured native value plus the change in the current adapter's post-RoPE key on its bound history, then applies the original FP16 cache cast. The derivative follows current weights and incoming states. Consumer layers reuse their original producer. The next free generation reads the newly installed native model as usual. The ordinary baseline entry point and all earlier run declarations retain their source versions.

For each registered source site, `loop/N/trace-diagnostic/proposal.json` records the two first-order products. Beside it, `finite.json` records a separate finite graph evaluation at the exact prospective rounded weight values and fixed source history. The graph overlays the changed coordinates from the existing read-only native preview and retains its original anchors. The next `fresh.json` and `transition.json` report the observed native successor. Multiple matching rows keep their individual forecasts and their ambiguity. An absent match produces zero site forecasts. These observations do not select the gradient, alter its step length or decide whether to install it. [Gemma validation and capacity](../../records/development/2026-11-18-installed-native-key-anchor/README.md)

## Train with the common early repair direction

Add `--prefix-objective path/to/objective.json` to either public `prepare` command. The [frozen development objective](../../studies/04-round-2-overlap/runs/prefix-direction-target/objective.json) uses the trace directions shared by two successful physical repairs before their generated histories diverged. A continuation of a prefix-trained state inherits the bound objective unless an explicit new record is supplied. The predecessor is admitted by its complete native state and ordinary measurement identity. The restored weights then produce a new free generation that measures the combined training objective.

```bash
target="$runs/prefix-direction-target/objective.json"
"${key_trainer[@]}" prepare "$next" --from-run "$prior" --review "$review" --intent intent.json \
  --prefix-objective "$target" > /tmp/next-prefix-declaration.json
"${key_trainer[@]}" check "$next" --declaration /tmp/next-prefix-declaration.json
```

The `common-prefix-turn-objective/1` record binds an `arrays` NPZ, the existing whole-prefix `registrar`, an absolute `first_source_event` for provenance, and optional source `inputs`. Each binding contains its repository-relative `record` and `sha256`. `reference_turns` contains the complete frozen source decode from its first prediction, with dimensions event × merge × coordinate. `early_turns` is its supported leading interval. Every event and merge in that interval has an entry in `axes` and in `targets`. Positive targets give a lower bound on displacement along a frozen unit axis, measured from the early turn. Zero targets denote unsupported cells.

Registration uses all observed turns across all merges, with each query event matched on its available prefix. For a registered cell, the new residual is `max(0, target − dot(current_turn − early_turn, axis))`. Each supported source cell has mass one divided by the number of supported cells. Multiple visits divide that cell's mass among them. Reaching the projection bound gives zero direct gradient from that cell. Unvisited cells are visible in coverage. The signed residual in turn units and the ordinary vector residuals share a single squared objective, existing weight pullback and continuous and rounded response products. Both products use the current correspondence and the next independent native generation computes its own. The fitted complete history remains the sole source of gradient requests.

`loop/N/measurement.json` retains the ordinary objective in `objective`, the registered directional discrepancy in `prefix_objective`, and their sum in `combined_objective`. It reports supported, represented, matched and active cell counts. `prefix-correspondence.json` binds the visits to the fresh native measurement. `proposal.json` keeps the separate ordinary and prefix held-history moments for both response products. The recorder exports `native.objective`, `native.prefix_repair.objective`, `native.combined_objective` and the prefix support counts. Existing stopping policies still measure their own declared ordinary trace criteria. [Installed CPU verification](tests/test_common_prefix_training.py)

Add `--evaluate-after path/to/fit-history.json` to run one fixed interval before independent evaluation. The JSON `fit_history` contains ordered source bindings for a contiguous run of complete fresh measurements of the ordinary fitted objective, starting at its measured baseline. One more than the longest dry spell ending at a new fitted low determines the interval. The documented 348–390 history yields eleven updates. After the final acquisition and its archive call return, the controller takes its normal measured exit. `evaluation-boundary.json` binds that update's measurement and the clean exit checkpoint. An earlier native terminal condition keeps its ordinary receipt. Evaluation prompts and outcomes do not enter the interval or the optimizer.

Use `--resume-evaluation-interval` with `--from-run` set to the interrupted run to finish its original interval from a complete measured native state. The new declaration binds the original calibration, previous run, final state, launcher exit and fresh source measurement. The first local acquisition links to that source and regenerates from its restored native values and FP32 remainders. The receipt reports both the total updates in the interval and the number installed by this continuation. For the interval that began at 390 and paused at local step 7, the continuation starts at historical step 397, installs four updates, and selects historical step 401 at local step 4.

## Train on signed failing and preservation computations

For a continuous joint fit, set `updates` to `"observed_joint_outcome"` and bind the fit, preservation, and plateau criteria in `update_basis`. The native controller keeps regenerating the cohort after installation. The research controller requests its existing graceful stop at a measured boundary once a fitted endpoint is selected or progress no longer supports continuing. The held panel is read after selection. A numeric `updates` still runs the historical bounded calibration.

The group launcher uses the same native recorder, anchored graph, full-parameter weight pullback, response scaling, preview, and native installation. Each installed update is followed by a new free generation for every included case. The next update uses those refreshed failing and preservation requests. The physical embedding row inventory is the union across those histories. A single case may supply a failing descent and a valid friction operation.

```bash
cohort=("${py[@]}" -m pipeline.weight_edit.group_contrast_experiment)
legacy="$runs/vector-chance-empirical-eta-continuation-4"
origin="$runs/prefix-balance-known-error-origin/origin.json"
"${cohort[@]}" prepare "$next" --from-run "$legacy" --development-origin "$origin" \
  --group-objective "$cohort_policy" --review "$review" --intent "$intent"
```

Supply `--write` to create the declaration. After recording the new run in `ROUND-2.md`, call `check "$next"` and `run "$next"` on the group launcher. The materialized origin binds the exact legacy native observation and its original reference, the authorized seed, and the full native state and remainder payloads. Fresh observations from the restored state drive the new training. For a previously measured full checkpoint, omit `--development-origin` and pass its run to `--from-run`.

The `signed-group-turn-objective/1` JSON binds `registrar`, `updates`, `update_basis`, and `members`. Every member has `id`, `case`, `arrays`, `first_source_event`, and `operations`. Each operation specifies `name`, `role` (`failing` or `preservation`), and an NPZ key prefix in `field`. Cases contain one unrestricted native prompt each. Members can share a bound array file and the registration method. The existing origin prompt is one member.

Every array file contains full `reference_turns[event,merge,coordinate]` and each operation supplies `FIELD_cells[N,2]`, with the source row relative to the first prediction and its merge. A signed field supplies `FIELD_axes[N,width]` and `FIELD_offsets[N]`. Its residual is the signed projection minus the frozen target, with immediate pullback on either side of zero. An affine preservation field supplies `FIELD_centres[N,width]` and `FIELD_variation[N,K,width]` with orthonormal or zero-padded rows. Its full displacement from the centre is projected outside the sampled correct variation. Its receipt records that orthogonal vector norm. A signed field separately reports motion orthogonal to its axis relative to the frozen native turn.

The baseline freezes the supported cells it actually represents in each operation. Roles, members within each role, and operations within each member have equal mass before dividing across these cells and their visits. Each `group-measurement.json` records original support, frozen baseline support, newly registered and missing cells, and paired loss changes on common cells. Partial coverage makes the complete objective unavailable, and its available residuals feed the next update while each operation retains a measurement. An entirely unmeasured operation stops further updates. Stored native stream pairs for consumed members are released through the guarded upload and full readback after their successor is measured. The endpoint record distinguishes finishing the declared update count from having complete, comparable group measurements.

A new cohort can select `"support_policy": "live_donor_support"` in its bound objective JSON. Each frozen donor-qualified source cell then owns a fixed share of its operation's mass. Every cell matched on the fresh generation contributes its local residual and divides that share across its current visits. Newly matched cells enter the next weight gradient immediately. Missing cells are listed in `missing_donor_support`, with changes from the baseline listed separately. The operation and group report their full objective only when all required cells are represented and the group additionally requires completed trajectories. `measured_contribution` gives the subtotal on observed cells. If an entire operation disappears, other observed operations can still provide a gradient and the complete objective stays unavailable. A comparison between installations uses the intersection of their measured source cells. A cohort with no measured operation supplies no update. Omitting the field selects the historical baseline policy.
