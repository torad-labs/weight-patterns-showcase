import { expect, test } from "bun:test";
import json from "../public/replay/gemma4-e2b.json";
import { recordingUrls, type ReplayData } from "./replay-model";
import { profilingExample } from "./profiling-example";
import { lifetime, parseProfilingBundle, profilingRelations, readiness, sampleGroups, usefulFraction } from "./profiling-model";
import { readRunEvidence, turnMeanMs, type RunEvidence } from "./run-evidence";

const capture = (json as ReplayData).captures[0];
const step = capture.steps.find(s => s.kind === "work")!;
const example = () => profilingExample(capture, step, 0);

const runExample = (): RunEvidence => ({
  schema: "qgre.run-evidence.v1", id: "conversation", label: "Conversation", targetTokens: 30_000, generatedTokens: 101,
  sources: [{ id: "normal", method: "Production", artifact: "normal.log", binarySha256: null, warnings: [] }, { id: "trace", method: "Trace", artifact: "trace.sqlite", binarySha256: null, warnings: [] }],
  turns: [{ id: "turn:1", sourceId: "normal", ordinal: 1, promptTokens: 20, cachedTokens: 0, generatedTokens: 101, cumulativeTokens: 101, hostNs: 900_000_000, gpuNs: 500_000_000, decodeTokens: 100, finish: "stop", request: "request.json", response: "response.json" }],
  launches: [{ id: "launch:1", sourceId: "trace", name: "decode", startNs: 20, endNs: 120, registers: 242, grid: 63, threads: 256, sharedBytes: 51200 }],
  metrics: [{ id: "metric:1", sourceId: "trace", launchId: "launch:1", name: "SM activity", unit: "%", mean: 75, min: 70, max: 80, samples: 10 }], captures: [], gaps: [], observations: [],
});

test("a recorded conversation reconciles its tokens and uses the measured decode population", () => {
  const run = readRunEvidence(runExample());
  expect(turnMeanMs(run.turns[0])).toBe(5);
  run.generatedTokens++;
  expect(() => readRunEvidence(run)).toThrow("total");
});

test("run counters cannot be attached to a launch from a different measurement pass", () => {
  const run = runExample(); run.metrics[0].sourceId = "normal";
  expect(() => readRunEvidence(run)).toThrow("same pass");
});

test("independent profiler counters remain unbound to a diagnostic timeline", () => {
  const run = runExample();
  run.counterPasses = [{id:"ncu:0",sourceId:"trace",timelineBinding:null,name:"decode",metrics:[]}];
  expect(readRunEvidence(run).counterPasses?.[0].timelineBinding).toBeNull();
  run.counterPasses[0].sourceId = "missing";
  expect(() => readRunEvidence(run)).toThrow("counter pass");
});

test("a shared recording link keeps its own source anatomy", () => {
  expect(recordingUrls("http://localhost:4318/?recording=/evidence/run/gemma.json#capture=86")).toEqual({ recording: "http://localhost:4318/evidence/run/gemma.json", anatomy: "http://localhost:4318/evidence/run/anatomy.json" });
  expect(recordingUrls("http://localhost:4318/").recording).toBe("http://localhost:4318/replay/gemma4-e2b.json");
  expect(() => recordingUrls("http://localhost:4318/?recording=https://elsewhere.test/data.json")).toThrow();
});

test("stored-code histograms preserve unknown clipping and their counting unit", () => {
  const b = example(); b.values[0].saturated = null; b.values[0].population = "32-element encoded blocks";
  const parsed = parseProfilingBundle(b, capture);
  expect(parsed.values[0].saturated).toBeNull();
  expect(parsed.values[0].population).toBe("32-element encoded blocks");
});

test("all eight views accept a relational example while retaining its illustrative origin", () => {
  const bundle = parseProfilingBundle(JSON.parse(JSON.stringify(example())), capture);
  expect(bundle.origin).toBe("illustrated");
  const ids = new Set([bundle.instruments, bundle.tasks, bundle.dependencies, bundle.instructions, bundle.work, bundle.buffers, bundle.samples, bundle.values, bundle.tokens, bundle.experiments].flat().map(r => r.id));
  for (const relation of profilingRelations(bundle)) { expect(ids.has(relation.from)).toBe(true); expect(ids.has(relation.to)).toBe(true); }
  expect(bundle.experiments.find(e => e.outcome === "refuted")?.result).toContain("did not advance");
});

test("readiness distinguishes the last prerequisite from delay after all recorded inputs arrive", () => {
  const b = example();
  b.tasks[3].startNs = 80;
  b.dependencies.forEach((d, i) => d.readyNs = [50, 20, 40][i]);
  const result = readiness(b, "task:press")!;
  expect(result.lastReadyNs).toBe(50);
  expect(result.afterReadyNs).toBe(30);
  expect(result.inputs.map(i => i.slackNs)).toEqual([0, 30, 10]);
  expect(readiness(b, "task:weights")).toBeNull();
});

test("useful work and retained capacity are derived from different denominators", () => {
  const b = example();
  expect(usefulFraction(b.work[0])).toBe(.625);
  expect(usefulFraction({ ...b.work[0], valid: 0, issued: 0 })).toBeNull();
  expect(lifetime({ ...b.buffers[0], bytes: 8, requestNs: 0, readyNs: 10, firstUseNs: 20, lastUseNs: 30, releaseNs: 50 })).toEqual({ deliveryNs: 10, waitingNs: 10, activeNs: 10, heldNs: 20, byteNs: 320 });
});

test("different measurement passes never merge into one activity series", () => {
  const b = example(); b.samples.push({ ...b.samples[0], id: "other-pass", instrumentId: "example:trace" });
  expect(sampleGroups(b).length).toBe(3);
  expect(sampleGroups(b).at(-1)!.rows.length).toBe(1);
});

test("invalid imports cannot silently become recorded evidence", () => {
  const cases = [
    (b: ReturnType<typeof example>) => b.tapeSha256 = "different",
    (b: ReturnType<typeof example>) => b.clockDomain = "host-clock",
    (b: ReturnType<typeof example>) => b.tasks[0].recordId = "unknown",
    (b: ReturnType<typeof example>) => b.tasks[0].worker = capture.grid,
    (b: ReturnType<typeof example>) => b.dependencies[0].producerId = "unknown",
    (b: ReturnType<typeof example>) => b.work[0].valid = b.work[0].issued + 1,
    (b: ReturnType<typeof example>) => b.buffers[0].releaseNs = b.buffers[0].readyNs - 1,
    (b: ReturnType<typeof example>) => b.values[0].edges = [0, 0],
    (b: ReturnType<typeof example>) => b.samples[0].value = Number.NaN,
    (b: ReturnType<typeof example>) => b.experiments[0].evidenceIds = ["unknown"],
  ];
  for (const mutate of cases) { const b = example(); mutate(b); expect(() => parseProfilingBundle(b, capture)).toThrow(); }
});
