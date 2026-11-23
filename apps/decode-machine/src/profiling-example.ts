import type { Capture, RecordedStep } from "./replay-model";
import type { ProfilingBundle } from "./profiling-model";

/** Authored demonstration. These values never become recorded telemetry. */
export function profilingExample(capture: Capture, step: RecordedStep, worker: number): ProfilingBundle {
  const start = Math.max(capture.start, step.start), end = Math.min(capture.end, Math.max(start + 1, step.end));
  const span = end - start, at = (p: number) => start + span * p;
  const row = { recordId: step.id, worker, instrumentId: "example:trace" };
  return {
    schema: "qgre.profiling-evidence.v1", id: `example:${capture.id}:${step.id}:${worker}`, label: "A delivery, a waiting press, and a full tray",
    origin: "illustrated", captureId: capture.id, tapeSha256: capture.tapeSha, clockDomain: `globaltimer-offsets:${capture.id}`,
    instruments: [
      { id: "example:trace", label: "Lifecycle recorder", method: "Authored event example", source: "profiling-example.ts", pass: "Lifecycle pass", cache: "Warm (illustrated)", clocks: "Fixed (illustrated)", sampleNs: null, binary: null },
      { id: "example:counters", label: "Counter sampler", method: "Authored samples", source: "profiling-example.ts", pass: "Counter replay · separate execution", cache: "Flushed (illustrated)", clocks: "Fixed (illustrated)", sampleNs: span / 12, binary: null },
    ],
    tasks: [
      { ...row, id: "task:weights", label: "Weight delivery", startNs: at(.02), endNs: at(.48) },
      { ...row, id: "task:activation", label: "Activation tray", startNs: at(.04), endNs: at(.24) },
      { ...row, id: "task:slot", label: "Slot released", startNs: at(.06), endNs: at(.37) },
      { ...row, id: "task:press", label: "Tensor press", startNs: at(.63), endNs: at(.95) },
    ],
    dependencies: [
      { id: "dep:weight", producerId: "task:weights", consumerId: "task:press", readyNs: at(.48), generation: "tile 17 / generation 4", instrumentId: row.instrumentId },
      { id: "dep:activation", producerId: "task:activation", consumerId: "task:press", readyNs: at(.24), generation: "activation 17", instrumentId: row.instrumentId },
      { id: "dep:slot", producerId: "task:slot", consumerId: "task:press", readyNs: at(.37), generation: "slot 2 / generation 4", instrumentId: row.instrumentId },
    ],
    instructions: [
      { ...row, id: "instruction:press", label: "Tensor press control", pc: "example:0x180", startNs: at(.48), endNs: at(.95), dataReadyNs: at(.48), fetchWaitNs: span * .15, instructions: 384, bookkeeping: 96 },
      { ...row, id: "instruction:loader", label: "Loader control", pc: "example:0x240", startNs: at(.02), endNs: at(.48), dataReadyNs: at(.02), fetchWaitNs: span * .02, instructions: 160, bookkeeping: 100 },
    ],
    work: [
      { ...row, id: "work:tail", label: "Tail tile", valid: 40, issued: 64, requestedBytes: 1280, transferredBytes: 2048 },
      { ...row, id: "work:full", label: "Full tile", valid: 64, issued: 64, requestedBytes: 2048, transferredBytes: 2048 },
    ],
    buffers: [
      { ...row, id: "buffer:weights", label: "Weight tray", generation: "slot 2 · generation 4", bytes: 32768, requestNs: at(.02), readyNs: at(.48), firstUseNs: at(.63), lastUseNs: at(.78), releaseNs: at(.96), memory: "shared", outcome: "released" },
      { ...row, id: "buffer:activation", label: "Activation tray", generation: "slot 1 · generation 9", bytes: 8192, requestNs: at(.04), readyNs: at(.24), firstUseNs: at(.63), lastUseNs: at(.91), releaseNs: at(.94), memory: "shared", outcome: "released" },
      { ...row, id: "buffer:kv", label: "Context page", generation: "page 7 · generation 1", bytes: 65536, requestNs: at(.01), readyNs: at(.12), firstUseNs: at(.18), lastUseNs: at(.86), releaseNs: at(.99), memory: "KV", outcome: "retained" },
    ],
    samples: ["DRAM delivery", "Tensor activity"].flatMap((metric, track) => [15, 85, 90, 10, 0, 0, 70, 95, 80, 15, 5, 0].map((value, i) => ({
      ...row, instrumentId: "example:counters", id: `sample:${track}:${i}`, metric, unit: "% of sustained peak", startNs: at(i / 12), endNs: at((i + 1) / 12), value: track ? [0, 0, 15, 40, 50, 0, 5, 80, 95, 90, 40, 10][i] : value,
    }))),
    values: [
      { ...row, id: "values:before", label: "Before rotation", boundary: "Rotation inlet", edges: [-4, -3, -2, -1, 0, 1, 2, 3, 4], counts: [2, 3, 8, 49, 55, 7, 2, 2], zeros: 34, saturated: 4 },
      { ...row, id: "values:after", label: "After rotation", boundary: "Rotation outlet", edges: [-4, -3, -2, -1, 0, 1, 2, 3, 4], counts: [4, 10, 20, 30, 30, 20, 10, 4], zeros: 8, saturated: 0 },
    ],
    tokens: [10, 10.2, 10.1, 10.4, 15, 10.6, 10.8, 11].map((ms, i) => ({ id: `token:${i}`, position: capture.position - 4 + i, startNs: i * 20e6, endNs: i * 20e6 + ms * 1e6, context: 1020 + i, batch: 1, sequence: "Illustrated sequence", instrumentId: row.instrumentId, coreMHz: i === 4 ? 2200 : 2400, powerW: 210, limit: i === 4 ? "Illustrated clock change" : null })),
    experiments: [
      { id: "experiment:staging", lens: "lifetimes", hypothesis: "A tray stays occupied after its last use", change: "Release the demonstrated slot earlier", prediction: "The next delivery starts earlier without overwriting live data", outcome: "proposed", result: null, evidenceIds: ["buffer:weights"] },
      { id: "experiment:prefetch", lens: "bursts", hypothesis: "Additional prefetch could fill the quiet interval", change: "Illustrated extra prefetch", prediction: "Shorter token duration with no extra rereads", outcome: "refuted", result: "Example outcome: more traffic; token completion did not advance.", evidenceIds: ["sample:0:4", "work:tail"] },
    ],
  };
}
