# Decode replay

A local, offline 3D inspector for the real production composition and recorded execution. No GPU launch, remote service, CDN or inference implementation is involved.

## Use

This library now lives in the weight-patterns research repository. For the current observer, run the [Decode Machine](../decode-machine/README.md). Existing replay inputs are local at `runs/legs/internal-decode-replay/`; preparing them requires no QGRE checkout:

```sh
# From the repository root:
bun apps/decode-machine/tools/prepare-replay.mjs runs/legs/internal-decode-replay/lifecycle-ready/bundle.json /tmp/replay-preview
bun test apps/decode-replay/*.test.js
```

The following commands document the original QGRE acquisition/export path. They are historical engine commands, not the entry point for ongoing weight-pattern research.

```sh
cargo build -p qgre
cargo build -p qgre-infer --bin qgre-phase-tape-report
target/debug/qgre dev decode-replay \
  --output runs/legs/replay/my-capture \
  --label "My recorded decode" \
  --phase-tape path/to/pos_01067.phaseclock.bin \
  --nsys path/to/capture.nsys-rep \
  --ncu path/to/capture.ncu-rep
```

Open the generated `index.html` directly in a browser. It is self-contained. `bundle.json` can be loaded with **Open bundle**. Output directories must be new; existing captures/artifacts are never overwritten. `--repo` selects the source checkout; the default discovers the Git root from the current directory.

- **Mechanism world:** a full-viewport scene with connected layer mechanisms, not a scrolling report. Drag to pan, wheel to zoom, Shift-drag to orbit, or use the minimap/area selectors to travel. **Parts** lists selectable machinery for keyboard access and close-up framing. Double-click a part to enter; Escape returns. **World** frames the complete recorded stack.
- **Token lifecycle:** **Intake** and **Output** travel to embedding/PLE preparation and final norm/readout/selection/stop machinery. The logical return wire closes the world without inventing a next execution. Playback ends at the layer tape boundary and shows the untimed output station.
- **Layer handoffs:** input norm, activation staging, weight transport and residual publication sit beside the measured machinery. Readiness gates expose existing wait spans. PLE parts share the down-consumer span; they add no separate duration. Shared KV portals travel to the captured owner layer. Supply/shared routes appear for the selected area to avoid an all-to-all wire cloud.
- **Snapshots:** recorded positions are independently selectable snapshots, not consecutive events. The drawer shows captured cache windows and sharing; it never invents cache contents or updates between files. Missing topology remains unknown.
- **Compare layers:** two independently explorable mechanisms stay side by side (stacked on a phone), with one recorded clock and root-span strips on the same duration scale. Select either pane's layer without manufacturing aligned execution times.
- **Play decode:** follows all recorded layers and measured inner parts. Manual navigation turns following off; **Follow token** restores it. Space plays/pauses; previous/next and scrubbing use the same globaltimer domain. Untimed embedding/logits/sampling are never animated as measured work.
- **Evidence on demand:** **Steps** lists every recorded step in the selected layer. Selecting a part opens only its name, timing and actions; **Block evidence** exposes the original timestamps, cycles and SM IDs. Shared-span operations remain labelled shared, storage untimed. Scene geometry and machinery glyphs explain composition, not physical GPU placement or simulated tensor contents.
- **Source assembly:** the actual `qgre_schedule::source_plan::production_plans` projection, grouped by operation directory. Open a group, then an invocation, then its declared lower-tier modules. Pagination retains every piece. This view is static and does not claim that a current-source module ran in an old recording.
- **Recorded activity:** 3D event blocks from the selected timeline. Play, pause, scrub and step; sort by longest interval, then focus a selected event to inspect neighboring gaps. Exact intervals remain available in the table. One Nsight event can contain many generated tokens.
- **Model / layer:** all discovered production recipes and layer kinds from source configs whose SHA matches the policy pin. No handwritten duplicate model roster.
- **NCU counters:** separate measured launch aggregates, with original metric units. They are not aligned to nsys by name.
- All original profiler timestamps travel as decimal text; integer subtraction precedes browser-number conversion. Independent captures retain separate clocks.

## Inputs

| Flag | Input |
|---|---|
| `--nsys` (repeatable) | CUDA 13.3 `.nsys-rep`, or Nsight SQLite export. Uses `nsys export --type=sqlite` and read-only `sqlite3`. Exports allowlisted kernel/API fields only, not process environments. More than 200,000 events is refused rather than truncated. |
| `--ncu` (repeatable) | `.ncu-rep` or NCU raw/details CSV. Uses CUDA 13.3 `ncu --import --csv --page raw --print-units base` for binaries. |
| `--phase-tape` (repeatable) | Raw `pos_*.phaseclock.bin` with its original sibling `launch_tables.json` and `phaseclock.jsonl`, or a `qgre.mrl-m1-58.phase-tape.v3` report from the existing reader. |

For raw phase tapes, build `cargo build -p qgre-infer --bin qgre-phase-tape-report` in the same profile as `qgre`. The reader remains responsible for raw-tape digests, launch geometry and wrap validation. Its refusals propagate unchanged. The importer also exports `qgre.capture-topology.v1` from the exact hash-bound launch tables: layer kinds, shared-tail count, PLE presence and per-layer windows. It rechecks the table digest after the reader returns. Imported JSON reports and their optional `capture_topology` are supplied report evidence, not independently recertified raw bytes. Topology is validated against the tape before replacing a mounted world.

Whole-decode navigation uses the root steps’ last-participating-entry → last-participating-exit spans and advances to the next layer after the previous layer’s last root exit. Gaps are retained, and early free-running/bystander entries do not pull playback ahead of unfinished work. Child spans are not added to parent durations; all individual observations remain available in the cutaway and block details.

The phase reader exports available block observations as checked signed globaltimer offsets and same-block cycle differences. The graph preserves overlapping observations; it does not arrange parallel work into a fictitious serial timeline. Arrows mean per-block program order, not a complete hardware dependency DAG. Missing operations remain explicitly untimed. Legacy JSON reports without the internal projection remain usable as coarse supporting evidence.

### E2B owner populations

The saved E2B long-context tape now reads successfully. The diagnostic reader previously rejected layer-varying FFN owner populations. It now reports each validated layer separately while retaining exact coverage, missing-owner and wrong-owner refusals. The internal export preserves those layer-specific populations rather than intersecting them across layers.

Attention S1–S4 are final-tile stamps; earlier tiles are included in the entry span. On two-pass heads, Q-tail inner marks retain the last visit and must not be summed as independent operations. Wait spans include lead-in as well as waiting. These are instrumented observations, not per-instruction profiling or production performance.

## Privacy and interpretation

Generated bundles contain kernel names, paths, capture metrics and source ports. Keep them local unless explicitly approved for sharing. The source snapshot has a Git revision and dirty-state indicator; it is not automatically the binary/source identity of an older capture. 3D coordinates are composition/record-order diagrams, never reconstructed physical SM placement. No utilization animation or internal timings are manufactured.

## Checks

```sh
cargo test -p qgre --test qgre_suite decode_replay:: --no-fail-fast
cargo test -p qgre-infer --bin qgre-phase-tape-report
bun test ./.dev/decode-replay/model.test.js ./.dev/decode-replay/internal-model.test.js ./.dev/decode-replay/mechanism.test.js ./.dev/decode-replay/decode-sequence.test.js ./.dev/decode-replay/world-model.test.js ./.dev/decode-replay/world-lifecycle.test.js
```
