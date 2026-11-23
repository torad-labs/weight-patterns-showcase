# Eight ways to follow a delay

## Goal

Marcos and an AI can follow the same selected operation through dependency readiness,
instruction supply, useful work, buffer lifetimes, time-resolved activity, numerical
distributions, successive tokens, and measurement conditions. Each view shows relationships
and relative quantities before explanation. Selection, recorded time, and export agree.

The existing cel-shaded machine, amber material, teal information, violet waiting, and
paper/ink interface are the visual authority. This extends the evidence inspector.

## Interaction

Eight illustrated lens buttons share a single large drawing and a contextual evidence
rail. Selecting a component highlights connected marks and exposes its evidence and a
next investigation. Recorded observations can seek the existing machine. An explicitly
selected illustrative example has its own hand crank and never advances recorded time.
Lens selection and example mode survive a shared inspection link. File import attaches
a normalized profiling bundle to its exact capture/tape; exports retain the whole bundle
and its relationships. Unknown quantities remain unknown, never zero.

## Evidence boundary

The supplied recordings contain thread-0 intervals, not the eight new hardware data sets.
Recorded mode uses those intervals and known capture metadata and displays missing
connections. A separate, labeled example exercises every data path. This change adds
visualization and ingestion, not CUDA instrumentation or a new inference path.
Critical-path and speedup claims require more than the last observed prerequisite.
Imported producer links are explicit. Aggregate samples have interval width and pass
identity; samples from different passes are not presented as simultaneous observations.
Numerical histograms are summaries, not permission to skip arithmetic or widen precision.
`saturated: null` means clipping cannot be established from the captured data. An optional
`population` names the counting unit, such as encoded 32-element blocks. Top-bin code
occupancy must not be relabeled as input saturation.

## Acceptance

- Each lens has a distinct visual relationship, a selectable component, and a useful
  missing-data state; all eight work with the same imported bundle.
- Validation rejects wrong-capture joins, dangling references, impossible intervals,
  incompatible clock domains, invalid counts and malformed distributions.
- Derived readiness, useful fractions, and buffer residency use explicit source values.
- Export includes lens focus, provenance, links, raw rows and derivation semantics.
- The desktop, laptop and phone retain readable diagrams and keyboard interaction.
- Example motion is reversible, honors reduced motion, and remains outside the replay.

## Capture contract

`src/profiling-model.ts` defines `qgre.profiling-evidence.v1`. All timestamps are nanoseconds
in the named capture's `globaltimer-offsets:<captureId>` domain. Instrument records name
their method, source, pass, cache control, clock control and sample interval (null means
unknown). A producer row carries an instrument reference. Record IDs must belong to the
capture; logical worker IDs must be within its recorded grid. A tape hash binds the join,
but does not substitute for binary, checkpoint or policy identity.

Data tables: tasks, dependencies, instructions, work, buffers, samples, values, tokens,
and experiments. Experiments preserve proposed/tested/refuted outcomes and supporting
row IDs. They never run automatically. Native `.ncu-rep`/`.nsys-rep` files must first be
exported and normalized into this contract; the browser does not parse their binary formats.

The bundled downloadable example is a complete template, explicitly marked `illustrated`.
Changing that label is not a method of obtaining real evidence. Real producers should
write `recorded` only for observations actually collected on that capture.

### What each producer must establish

| Lens | Required relationship | Do not infer |
| --- | --- | --- |
| Dependencies | Producer task → generation-specific readiness event → consumer task; explicit common clock | A global critical path or recoverable latency from the last observed input |
| Instruction supply | Compiled location and instruction/fetch evidence → task → instrument/pass | Instruction-cache misses from a long task interval alone |
| Useful work | Valid work and issued work in the same counting unit; requested and transferred bytes at a named boundary | Padding cost from matrix shape alone, or DRAM traffic from logical weight bytes |
| Buffer lifetime | One allocation generation, capacity, request, ready, first use, last use and release/observation end | An actual release when an object is retained, or saved time from byte-nanoseconds |
| Supply/activity | Metric, units and observation interval → collection pass | Simultaneity between separate replays, or zero activity in unsampled gaps |
| Values | Named boundary, shared bin edges, bin counts, zero and saturation counts | A speedup from sparsity without an execution path that exploits it |
| Tokens | Position, sequence, interval, context and batch → instrument; clocks/power/limits when captured | Continuity between independently recorded positions or causality from correlation |
| Instruments | Every row → method, source, pass, cache treatment, clock treatment, sampling width and binary identity where known | Production timing from a diagnostic replay |

IDs are bundle-wide unique. Relational export uses `measured_by`, `publishes`,
`required_by`, `investigates`, and `observes_operation`. The complete bundle is exported,
even when the diagram displays a selected subset. Large sample series currently show
up to 2,000 intervals per metric/pass group and disclose that limit in the reading note.
The selected operation controls the local lenses; token and instrument views retain
the wider run context. Prerequisite closure includes producers outside that operation.

The schema is a normalized interchange boundary, not a claim that Nsight provides all
these facts in one pass. A producer must explicitly correlate its clocks and identities.
The viewer checks structural joins and ranges; it cannot authenticate a caller's physical
claims. Imports live only in the open page. Sharing retains the imported bundle ID and
asks for that file again after reload. Reduced-motion users can scrub the example manually.

## Recorded conversations and companion passes

`?recording=/local/path/model.json&run=/local/path/run.json` opens a saved recording
with its run-level evidence. The recording and its sibling `anatomy.json` must be
same-origin. A run manifest may point to same-origin profiling sidecars; these load
automatically for the selected capture and restore after following an inspection link.
Arbitrary file imports still require the file again after reload.

`qgre.run-evidence.v1` organizes sources, conversation turns, CUDA launches, sampled
metrics, diagnostic captures, observations and remaining gaps. A turn references its
production source; a metric references its own source and launch. Turn totals must
reconcile. A counter cannot reference a launch from another pass. Optional configuration,
telemetry summaries and explicit relation edges retain the preparation tool's provenance.
Independent Nsight Compute counter passes retain the native replay reader's metric rows
and source, with `timelineBinding: null`; a kernel name alone cannot join them to a phase.

The conversation chart uses reply width for generated-token count and height for mean
GPU decode time. It excludes the prefill-produced first token from the decode denominator.
Selection shows the corresponding retained-input fraction and the prefill/decode share
of request time when both launch durations are recorded. The attention chart compares
the last-entry-to-last-exit tail averaged over layers of each kind; it does not claim a
critical path or recoverable time. Clicking a capture opens an observed working participant
and the capture's value distributions. Hardware rails share a 0–100% scale and retain the
distinction between production and diagnostic passes. Lines never connect attention
points from different capture passes; the later cold-prefill probe remains separate.

Prepare an existing phase replay with `bun tools/prepare-replay.mjs <bundle> <output>`.
The E2B run's `tools/prepare-run.mjs <run-directory> <output>` adapter preserves raw
artifact paths and emits run-level evidence plus the recorded tap-range sidecars.
It reads profiler exports; it does not launch GPU work or change the inference engine.

Rejected internal timelines remain visible as `unavailableCaptures`; their original
artifacts and rejection reasons survive in the export. Independently admitted stored
values may remain under `valueCaptures` with no timing binding. The adapter never repairs
a failed clock bound by rewriting the original receipt. Ranked instruction sites retain
their counter-pass identity and selection rule, without inventing PC-to-operation joins.
The scheduler detail compares independent residency limits; its smallest limit is not
a promised occupancy improvement. Nsight launch memory includes static plus dynamic
storage, while the Compute counter table also retains its separate driver allocation.
