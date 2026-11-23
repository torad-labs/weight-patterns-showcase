# Decode machine — inside Gemma 4

> **Research home:** This application lives in `apps/decode-machine` at the repository root. Start with the [repository guide](../../README.md) for the theory, studies, code, figures, and manuscripts.

For the current research direction, start with `STATE.md` at the repository
root and the method in `studies/04-round-2-overlap/ROUND-2.md`; the journal
pages of November 7 and later hold wherever an older page disagrees.
[The residual-trace handoff](RESIDUAL-TRACE-HANDOFF.md) and
[the full theory reference](RESIDUAL-TRACE-THEORY.md) record Marcos's
November 6 clarification as it was understood then: seek recurring portions of
the connected changes in computation across capability-focused executions; a
useful partial trace does not require recovery of the complete mechanism. They
and the earlier experiment documents here are historical records, not the
current definition of the research target (this paragraph corrected November
15, 2026: it had still named the two as the current direction).

[The physics observation pilot](PHYSICS-TRACE-PILOT.md) records the first bounded
trial under that definition: behavioral results, connected prompt/generation
captures, and measured limits on observer fidelity.

[The physics trace follow-up](PHYSICS-TRACE-FOLLOWUP.md) extends the connected
comparison to twelve physical, paraphrased and supplied-arithmetic pairs. It
records all eligible shared-prefix decisions, separates scale from direction,
and links concrete calculation moments to their actual captured state changes.

[The BF16 comparison](PHYSICS-TRACE-BF16.md) repeats the same 24 questions with
the published BF16 checkpoint. It records correct partial energy calculations,
preserves the recurring layer progression and its exception, and compares both
precisions within their actual shared generated histories.

[The contribution-transplant experiment](PHYSICS-TRACE-INTERVENTION.md) tests a
recorded attention contribution with exact controls. It changes current scores
but no generated tokens, and identifies why the chosen layer cannot carry this
one-time change through the model's cache into later predictions.

[The persistent weight-edit experiment](PHYSICS-TRACE-WEIGHT-EDIT.md) repeats the
comparison with an actual native BF16 projection edit. It repairs the targeted
calculation and a second case, introduces a new wrong final answer elsewhere,
and follows the resulting changes through matched internal histories.

**Follow the reason** adds eight connected profiling lenses to the evidence inspector:
dependencies, instruction supply, useful work, buffer lifetimes, supply/activity,
value distributions, token trajectories, and measurement conditions. Select a piece
in a drawing to inspect its source and follow the next related question. **Open selected
machine** returns to the same operation in the mechanical world.

Use **Explore an example** for authored, reversible mechanical motion; **Whole example**
sets the total duration in seconds. Its slider does not seek the recorded decode.
**Recorded evidence** shows only captured facts. The supplied recordings lack the deeper
hardware observations; **Import profiling evidence** accepts a normalized, capture-bound
JSON bundle, and **Download example bundle** provides its complete schema in practice.
See [PROFILING-LENSES.md](PROFILING-LENSES.md) for collection semantics and the import contract.
Native Nsight report files require normalization before import.

The inspection export now includes `profiling`, with the full imported bundle, normalized
relationships, operation joins, missing evidence and derivation semantics. Shared links
also retain lens, selected piece and example position. Imports stay in memory; after a
reload, a shared link names the bundle to re-import instead of substituting example data.

Choose **Diagnose recording** to open a synchronized worker timeline and evidence inspector alongside the 3D machine. From the illustrated story, this action switches to its recorded capture and pauses. From a recording, it keeps the current time and playback state. Follow connected operation nodes, select a dot in the worker arrival plot, or inspect the last observed exit. **Time window** magnifies the whole recording, a layer or an operation without changing time. Selecting an interval or **Go to entry** explicitly seeks and pauses. The comparison ranges use a shared scale; opening the other recording explicitly goes to that worker's entry.

**Share inspection link** reveals a link that restores capture, operation, worker, investigation area, time and window. It also attempts clipboard copy; the visible field remains available if the browser does not provide clipboard access. **Download evidence JSON** and the read-only `window.decodeInspection.snapshot()` expose a relational `qgre.decode-inspection.v2` document: normalized entities, typed relationships, source-linked calculations and an ordered story. The snapshot returns `null` outside recorded diagnostic mode. `decode-inspection-change` announces an explicit worker/operation selection change; read a new snapshot for the current clock time and investigation area.

**From observation to an experiment** provides seven architecture branches. Each connects alternative explanations to the missing evidence and a conditional change, prediction and refutation. [EVIDENCE-ONTOLOGY.md](EVIDENCE-ONTOLOGY.md) explains the export and investigation method, drawing on `.dev/research/nvidia-pro` and official NVIDIA documentation. It preserves the boundary between an observed interval and a proven hardware cause.

These are thread-0 arrival stamps, not proof of whole-block completion. The recorded-order graph is incomplete as a causal graph. Exit spread is an investigation aid, not recoverable time. Unassigned time is not asserted to be hardware idleness, and the mechanical supply paths are still illustrative. See `DIAGNOSTICS.md`.

The default page follows **“What’s the capital of France?” → “Paris.”** through the Gemma-shaped machine. The question, current tray and answer remain visible while you open its layers and equipment. A physical intake feeds 35 layers, amber balls remain in storage while read copies feed presses, and the answer printer sends selected tokens back through the same stack.

Choose **Open the moving layer** to slow that layer to a 12-second inspection window. **Open the equipment** magnifies a housing in the same journey. **Return to the whole journey** releases that window without changing its position or play/pause state. The duration control explicitly names **Whole animation** or **This inspection**; the whole story keeps its own duration (120 seconds by default). Play, pause, scrub and replay work at either scale.

The tabs describe inspection depth: **Megakernel floor → Layer journey → Inside a layer → Cart & supplies**. The floor shows reusable workers doing overlapping jobs and opens with a stable camera. **Follow active worker** deliberately follows one representative worker; **Follow tray** and **Follow active equipment** follow the material at the other scales. Switching views or opening equipment while paused stays paused. Worker counts distinguish working, waiting and idle benches.

From **Megakernel floor**, choose a worker and **Open SM & warps** to inspect its hardware context. A fixed representative SM housing surrounds the selected block, shared staging and execution machinery. Its eight source-defined warp groups—one controller, one loader and six consumers—each open into 32 selectable logical lanes with private register holders. Choose a group on the machine or in **Warp group**, then **Open 32 lanes**. **SM bay** and **Worker floor** return through the same selection; only **Watch this block** explicitly seeks and plays.

The 63 recorded blocks are not 63 physical SMs. SM identity, other resident blocks, per-warp timing, active-lane masks, register values and role-to-thread binding were not captured. Role counts come from the separately identified source anatomy, not proof of the capture's build. Lane strokes explain grouping using the selected block's activity window; a wait or idle window stops those strokes and the operand feed. A lane is a logical thread position, not a dedicated physical core.

**Cart & supplies** opens a connected cutaway of the material at the current position. Click a travelling cart, a mechanism label, or choose one of the nine topics in **Explore**. **Look closer** changes only the camera; **Watch this process** explicitly starts an inspection window on the same clock. The cutaway shows:

- A cartridge with twelve signed numerical channels, its incoming/current values, and continuity into the next layer.
- Numbered K/V shelves that retain context, with eligible read copies and an optional three-position teaching window on sliding layers.
- Six partial-work trays, their collecting gate, retained weight balls, a loader and three reusable staging slots.
- Controller, loader and consumer roles, readiness, and a low-precision code/shared-scale gauge with its rounding difference.
- Candidate scores, selection and the chosen token's physical return route.
- An optional second copy of the France request sharing machinery with separate context and output. Select A or B to inspect it; the other request remains visible in the journey. Their timing offset is authored, not a measured batching schedule or throughput result.

The twelve-channel calculation, six contributions, staging schedule and candidate scores are small illustrative examples. The six consumer roles reflect the separate source anatomy, not measured physical core placement. Recorded mode keeps unmeasured channels, K/V contents, transfers, staging occupancy and candidates explicitly unknown. A recorded wait does not establish its producer or cause. Pausing, changing viewing duration and reversing the timeline preserve these distinctions.

This France story, its token boundaries, pin shapes and timing are authored explanations on the captured topology, not model output or numerical tensor simulation. It uses 35 Gemma layers, eight query heads, owned/shared K/V drums, wider late feed-forward presses and 63 reusable logical worker blocks. No inference engine is run or changed. See `CONNECTED-JOURNEY.md` for the repair’s intent.

Select **Recorded token 1067** or **1015** to inspect the original timing. These are independent captures, not consecutive answer tokens. The prompt and selected answer were not recorded. Layer selection preserves recorded time until an explicit play/seek action; the hand crank pauses the recording for a separate mechanical illustration. The layer journey retains arrival-based navigation and the worker floor retains early waits and all participating block intervals. **Open the source anatomy** searches the separate source snapshot. Each housing opens into equipment with explicit inputs, supplies and outputs; see `INTERNAL-EQUIPMENT.md`.

`GEMMA-REPLAY-PLAN.md` describes the evidence, mechanical mapping and limits. `tools/prepare-replay.mjs` imports the local lifecycle-ready bundle using the existing decode-replay normalizer and topology admission. It verifies tape/launch-table hashes and exports browser-ready data under `public/replay/`:

```sh
bun run replay:prepare
```

The importer and provenance integration test require the original files under `runs/legs/`; the browser and static build use the committed display data.

## Viewer structure

- `src/replay-playback.ts` owns mode selection, transport, inspection windows and duration. `replay-model.ts` supplies the recorded clock; `connected-journey.ts` supplies the authored story.
- `src/recorded-motion.ts` resolves overlapping observations into worker activity, wait intervals and weight-read progress. `machine-motion.ts` is the sole pose writer for the stack, layer and worker machines.
- `src/replay-world.ts` owns scene geometry, view framing and projected labels. `journey-world.ts` carries trays and weight copies along connected paths. `workshop-world.ts` expands the selected equipment and shares the same press stroke.
- `src/material-model.ts` derives the cart, context, partials, staging and optional second request from the existing playback position. `material-world.ts` owns the connected cutaway's geometry and poses; `material-panel.ts` connects its controls, numeric readouts and inspection actions.
- `src/hardware-model.ts` combines source warp roles with the shared worker activity resolver. `hardware-world.ts` owns the representative SM and lane machinery; `hardware-panel.ts` provides hardware navigation and selection without a separate clock.
- `src/replay-app.ts` connects controls and readouts. `source-inspector.ts` owns lazy source-anatomy loading and search; `replay-dom.ts` holds small DOM helpers.
- `src/replay.css` owns the default viewer's styling at every inspection depth and screen size. The teaching scene retains its separate entry point and styles.

All animation poses are reconstructed from the current replay position. Pausing, changing duration and scrubbing backward preserve a consistent scene; inspecting a part does not introduce a second running clock. Weight copies and press strokes remain symbolic movement, even when their activity windows come from a recording.

## Preserved teaching example

At `/teaching.html`, the earlier observable 3D teaching machine answers the fixed question **“What’s the capital of France?”** The scripted answer is **“Paris.”**, followed by an end signal.

The machine cuts the question into illustrative tokens and assembles carriers with numerical pins. Prefill moves the prompt positions through 32 layers and fills each layer’s paired K/V storage racks. Prefill produces the scores for the first answer token. The selector prints Paris and feeds it back to predict the period; the period makes another pass to predict End. The return gate closes and the answer remains visible.

This is an authored explanation, not a running language model, measured QGRE trace, or numerical tensor simulation. Token boundaries, vector pins, candidate scores, layer/head counts and timing proportions are illustrative. There is no arbitrary-question input. No inference engine code is changed.

## Run

From this directory:

```sh
bun install --frozen-lockfile
bun run dev --port 4318 --strictPort
```

Open http://127.0.0.1:4318/. The preview is local to this machine and owned by this chat. Stop it with Ctrl+C in its terminal.

```sh
bun run test
bun run build
bun run preview --port 4319 --strictPort
```

The static build is `dist/`. Models and fonts are local; the viewer requires no external service.

## Explore the teaching example

- Drag to orbit; scroll or pinch to zoom; right-drag pans.
- Inspect the question intake, weight reservoir, any layer, or the candidate selector. Individual mechanisms are isolated for a clear view.
- A layer contains a normalization jaw, three representative parallel attention heads, K/V racks and visible read copies, a second sizing step, feed-forward press and two residual routes. Lift its cover to inspect the internals.
- Open a layer’s **weight drawer** to inspect twelve stored sample values. Read copies travel to that layer’s heads and press; stored balls stay unchanged between token passes.
- Click a prompt token, a moving carrier, or **Inspect a token carrier**. Compare four example values at the layer entrance with their current values. Selection preserves the playhead; **Show this carrier in a layer** explicitly seeks when the carrier is waiting or has finished. Paris and the period have their own carriers too.
- Choose **Follow the story** to move between intake, active layers, selector and overview as playback advances.
- Set **Total animation** to 1–3600 seconds (default 120). The entire question-to-answer cycle scales together without jumping to another position.
- Pause, scrub, advance an operation, use the Question / Prefill / Paris / Period / End chapter buttons, or replay. Playback stops at the end. Reverse scrubbing reconstructs earlier output and cache contents.
- Keyboard: Space plays/pauses or replays a completed story, arrows move half a viewing second, Home restores the overview, and C lifts the selected layer cover. Focused controls keep their native keyboard behavior.

Reduced-motion preferences start paused and snap camera/cover transitions. Playback remains available on demand. On phones playback sits immediately beneath the visible machine, followed by the inspector.

## Teaching representation

Amber balls represent sample weights. Each numbered layer owns a distinct set of twelve illustrative values, reused unchanged on every pass. Read copies travel along local pipes while the stored balls remain in their drawer. Teal angular carriers keep their token identity while four numerical pins change. K/V shelves keep paired context markers; read copies move from the shelves toward the heads while stored markers remain. The weights are not balls labeled with words, and Paris is not a reservoir lookup.

The four-pin demonstration uses authored vectors and a small deterministic sequence of normalization, weighted mixing, residual blending and a nonlinear press. One layer’s output becomes the next layer’s input. This is a teaching calculation, not a Transformer implementation or a causal explanation of the scripted answer. Pin height increases with the signed value; the numeric readout gives the sign. Finished prompt carriers retain their final values during later decode passes.

The sizing jaws and carrier height changes stand for normalization. The press expands and compresses the carrier to represent a feed-forward transformation. Candidate bars show authored relative scores, not model probabilities. Choosing End closes the feedback gate without printing a literal End token.

Projection matrices, positional encoding, causal-mask arithmetic, individual tensor instructions and real sampling distributions are not expanded. The representative mechanisms explain dependencies and information reuse; their geometry is not a literal GPU memory layout.

## Assets

`assets/layer-kit.blend` is the editable Blender source. `tools/build-assets.ts` authors the recipe and invokes Blender’s Python API as an export bridge:

```sh
bun run assets
BLENDER_BIN=/path/to/blender bun run assets
```

Default Blender executable: `blender` (override with `BLENDER_BIN`, tested with Blender 4.5.14 LTS). The kit exports reusable layer, supply, intake, selector and receiving-tray GLBs. The teaching scene loads the first four; the receiving tray remains a reusable kit component. Teaching layer modules are reused 32 times. Roller, jaw, press, cutter, selector and stop-gate pivots stay independently movable after export.

Babylon.js applies cel shading and owns animation. `src/timeline.ts` defines the complete normalized story, layer operations and position-derived output/cache state. `src/representation.ts` reconstructs illustrative carrier values and stable weight identities. `src/weights.ts` animates local drawers and read copies. `src/story.ts` creates the surrounding mechanisms, pin carriers, cache shelves, read copies and candidate bars. `src/main.ts` connects scene rendering and controls.

## Validation and limits

The eight-lens profiling extension passes `bun run test` (60 passed, 0 failed,
exit 0), `bun run build` (exit 0), and `git diff --check` (exit 0). Browser checks
covered every lens at 3840 × 2050 and 390 × 844, plus the 1440 × 900 layout;
file-chooser import, rejection of an incorrect tape without losing loaded data,
relational download, shared-link restoration and the re-import notice, reversible
example positions, an unchanged recorded clock, reduced motion, and return to the
selected machine. Review-driven checks also confirmed one chart tab stop per evidence
entity, focus retention after activation, visible selection, and a third imported
distribution appearing in its comparison. The finish reviewer scored all four
requested corrections resolved; this is a verdict on those fixes, not a fresh
whole-application approval. Final captures and the illustrative exported inspection
are under `evidence/profiling/`.

This extension provides visualization and normalized ingestion. It does not collect
new GPU data, run experiments, authenticate imported physical claims, or directly
parse binary Nsight reports. The pre-existing recording and software-WebGL limits
below still apply.

`bun run test` covers the narrative ordering, automatic stop, reverse scrubbing, cache retention, chosen-duration scaling, parallel branches, layer seeking and distance-based pipe travel. Browser checks drive real controls and loaded models across the full story, including replay and mobile controls. Captures cover 3840×2050, 1440×900 and 390×844.

The evidence-story revision passes all 55 tests (`bun run test`, exit 0) and the production build (`bun run build`, exit 0). Browser checks cover both captures, shared selection, exact equipment entry, keyboard navigation, restored investigation links, resized timeline hit-testing and the downloaded relational report. All relationship endpoints in the downloaded example resolve. Final captures, a representative report and independent visual reviews are under `evidence/ontology/`. These are functionality checks, not performance measurements of the GPU engine.

The final independent visual review scores the desk layout 7.0/10 and compact layouts 6.5/10, below the 8/10 target. Nested scrolling is removed, but machine framing and compact label density, distance between evidence and machinery, and the text-heavy investigation plan remain design limitations. The implementation is functional; the visual finish is not fully approved.

The automated browser uses software WebGL; these checks establish functionality, not GPU performance. WebGL2 is required. Vite still reports a large Babylon.js production bundle. `evidence/` holds local validation material and is excluded from the shipped site.

## Credits

All mechanical geometry and the favicon were authored for this prototype. No generated raster artwork or reflection environment is used. The local Manrope font comes from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/manrope), unchanged under the SIL Open Font License included as `public/assets/Manrope-OFL.txt`.
