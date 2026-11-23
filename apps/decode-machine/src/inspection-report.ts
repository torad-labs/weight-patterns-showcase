import type { DiagnosticModel } from "./diagnostic-model";
import { exitDistribution } from "./diagnostic-model";
import type { Capture, RecordedStep } from "./replay-model";
import { architectureAreas, evidenceSources } from "./diagnostic-ontology";

export function inspectionReport(model: DiagnosticModel, time: number) {
  const { data, capture, step, block } = model;
  const id = (c: Capture, type: string, key = "") => `${type}:${c.id}${key ? `:${key}` : ""}`;
  const opId = (c: Capture, s: RecordedStep) => id(c, "operation", s.id);
  const obsId = (c: Capture, s: RecordedStep, index: number) => id(c, "observation", `${s.id}:${index}`);
  const window = model.bounds();
  const inWindow = (model.motion.workers[block] ?? []).filter(s => s.end > window.start && s.start < window.end);
  const scopes = data.captures.map(c => {
    const selected = new Set(c.steps.filter(s => s.layer === step.layer).map(s => s.id));
    if (c.id === capture.id) for (const s of inWindow) selected.add(s.step.id);
    const byId = new Map(c.steps.map(s => [s.id, s]));
    // Include direct order neighbors, then all enclosing records. Do not expand the entire order chain.
    const seeds = new Set(selected);
    for (const s of c.steps) {
      if (seeds.has(s.id)) for (const p of s.predecessors) if (byId.has(p)) selected.add(p);
      if (s.predecessors.some(p => seeds.has(p))) selected.add(s.id);
    }
    for (const key of selected) {
      let parent = byId.get(key)?.parent;
      while (parent && byId.has(parent) && !selected.has(parent)) { selected.add(parent); parent = byId.get(parent)?.parent; }
    }
    return { capture: c, records: c.steps.filter(s => selected.has(s.id)) };
  });
  const entities = {
    runs: scopes.map(({ capture: c }) => ({ id: id(c, "run"), tokenPosition: c.position,
      instrument: "thread-0 device stamp recording", startNs: c.start, endNs: c.end,
      clockDomain: `globaltimer-offsets:${c.id}`, source: c.source, tapeSha256: c.tapeSha,
      launchTablesSha256: c.tablesSha, semantics: c.semantics, coverage: c.coverage,
      gpuIdentity: null, policyIdentity: null, checkpointIdentity: null, exactBuildIdentity: null })),
    layers: scopes.flatMap(({ capture: c, records }) => c.layers.filter(l => records.some(s => s.layer === l.id)).map(l => ({
      id: id(c, "layer", String(l.id)), runId: id(c, "run"), index: l.id, kind: l.kind,
      kvOwnerLayerIndex: l.owner, window: l.window, headDim: l.headDim, ffnWidth: l.ffnWidth, startNs: l.start, endNs: l.end }))),
    operations: scopes.flatMap(({ capture: c, records }) => records.map(s => ({
      id: opId(c, s), sourceRecordId: s.id, runId: id(c, "run"), layerId: id(c, "layer", String(s.layer)),
      name: s.name, kind: s.kind, role: s.role, fromTap: s.fromTap, toTap: s.toTap,
      description: s.description, anomalies: s.anomalies }))),
    workers: scopes.flatMap(({ capture: c }) => Array.from({ length: c.grid }, (_, b) => ({
      id: id(c, "worker", String(b)), runId: id(c, "run"), logicalBlock: b, physicalSm: null }))),
    observations: scopes.flatMap(({ capture: c, records }) => records.flatMap(s => s.observations.map((o, index) => ({
      id: obsId(c, s, index), operationId: opId(c, s), workerId: id(c, "worker", String(o[0])),
      startNs: o[1], endNs: o[2], reportedSmField: o[3], physicalPlacementVerified: false,
      validity: !Number.isFinite(o[1]) || !Number.isFinite(o[2]) ? "non-finite" : o[2] < o[1] ? "reversed" : o[2] === o[1] ? "zero-length" : "valid",
    })))),
    segments: inWindow.map((s, index) => ({ id: `segment:focus:${index}`, operationId: opId(capture, s.step),
      workerId: id(capture, "worker", String(block)), startNs: Math.max(window.start, s.start), endNs: Math.min(window.end, s.end),
      kind: s.step.kind === "work" ? "work" : s.step.kind === "wait" ? "wait" : "mixed",
      observationIds: s.step.observations.flatMap((o, i) => o[0] === block && o[2] > s.start && o[1] < s.end ? [obsId(capture, s.step, i)] : []) })),
    areas: architectureAreas.map(a => ({ id: a.id, name: a.label, physicalMetaphor: a.physical, mapping: "conceptual architecture; placement and per-area hardware counters unrecorded" })),
    checkpoints: architectureAreas.flatMap(a => a.checkpoints.map((name, index) => ({ id: `checkpoint:${a.id}:${index}`, areaId: a.id, name, availability: "missing", meaning: "Investigation checkpoint; no per-event timing captured" }))),
    sources: evidenceSources.map(s => ({ ...s })),
    measurements: [
      { id: "measurement:worker-window", subjectId: id(capture, "worker", String(block)), unit: "ns", scope: window,
        method: "Disjoint resolved intervals clipped to window; remainder unassigned; nested parents never added", value: model.totals(block),
        inputIds: inWindow.map((_, i) => `segment:focus:${i}`) },
      ...scopes.flatMap(({ capture: c, records }) => records.filter(s => s.id === step.id).map(s => ({
        id: id(c, "measurement", "exits"), subjectId: opId(c, s), unit: "ns", scope: { start: s.start, end: s.end },
        method: "Valid thread-0 exits; median, last and their difference; no recoverable-time claim",
        value: { median: exitDistribution(s).median, last: exitDistribution(s).last, tail: exitDistribution(s).tail, count: exitDistribution(s).rows.length },
        inputIds: s.observations.flatMap((o, i) => Number.isFinite(o[1]) && Number.isFinite(o[2]) && o[2] > o[1] ? [obsId(c, s, i)] : []) }))),
    ],
    hypotheses: architectureAreas.flatMap(a => a.alternatives.map((name, i) => ({ id: `hypothesis:${a.id}:${i}`, areaId: a.id, name, status: "unproven" }))),
    requirements: architectureAreas.map(a => ({ id: `requirement:${a.id}`, areaId: a.id, name: a.need, tool: a.tool,
      availability: "missing", scope: "Same workload, exact binary/policy/device; align clocks before joining" })),
    experiments: architectureAreas.map(a => ({ id: `experiment:${a.id}`, areaId: a.id, status: "conditional—not authorized by evidence yet",
      change: a.experiment, prediction: a.prediction, primaryMetric: "Production end-to-end token latency on a controlled workload",
      companions: [...a.companions], refutes: a.refutes,
      controls: ["Same GPU, clocks, checkpoint, batch/context, token count and instrumentation", "State predicted magnitude and band before running", "Existing correctness and coherence checks", "Production megakernel; geometry changes through policy"] })),
  };
  type Relation = { id: string; from: string; to: string; kind: string; basis: string };
  const relations: Relation[] = [];
  const link = (from: string, to: string, kind: string, basis: string) => relations.push({ id: `relation:${relations.length}`, from, to, kind, basis });
  for (const l of entities.layers) link(l.runId, l.id, "contains", "recorded hierarchy");
  for (const w of entities.workers) link(w.runId, w.id, "contains", "recorded logical grid");
  for (const s of entities.operations) link(s.layerId, s.id, "contains", "recorded hierarchy");
  for (const o of entities.observations) { link(o.id, o.operationId, "observed_operation", "raw tuple"); link(o.id, o.workerId, "observed_worker", "raw tuple"); }
  for (const { capture: c, records } of scopes) {
    const included = new Set(records.map(s => s.id));
    for (const s of records) {
      for (const p of s.predecessors) if (included.has(p)) link(id(c, "operation", p), opId(c, s), "recorded_before", "program order; causal producer unproven");
      if (s.parent && included.has(s.parent)) link(opId(c, s), id(c, "operation", s.parent), "nested_in", "recorded enclosure; not additive");
    }
    const corresponding = records.find(s => s.id === step.id);
    if (corresponding && c.id !== capture.id) link(opId(capture, step), opId(c, corresponding), "analogous_record", "same source ID; not a controlled A/B");
  }
  for (const s of entities.segments) for (const o of s.observationIds) link(s.id, o, "derived_from", "existing worker interval resolver");
  for (const m of entities.measurements) for (const input of m.inputIds) link(m.id, input, "derived_from", m.method);
  for (const a of architectureAreas) {
    a.checkpoints.forEach((_, i) => {
      const checkpoint = `checkpoint:${a.id}:${i}`;
      link(a.id, checkpoint, "investigation_checkpoint", "conceptual investigation model, not measured placement");
      if (i) link(`checkpoint:${a.id}:${i - 1}`, checkpoint, "inspect_next", "suggested investigation sequence, not recorded execution order");
    });
    for (const h of entities.hypotheses.filter(h => h.areaId === a.id)) {
      link(h.id, `requirement:${a.id}`, "requires", "distinguishing evidence absent");
      link(h.id, `experiment:${a.id}`, "tested_by", "conditional experiment, not a diagnosis");
    }
    for (const source of a.sources) link(a.id, source, "informed_by", "documented investigation method");
  }
  const candidate = model.area;
  link(opId(capture, step), candidate.id, model.areaId ? "investigates_area" : "candidate_area", model.areaId ? "analyst selection; not measured placement" : "heuristic from operation role/kind; not measured placement");
  for (const h of entities.hypotheses.filter(h => h.areaId === candidate.id)) link(id(capture, "measurement", "exits"), h.id, "motivates", "observation suggests a question, not a cause");
  return {
    schema: "qgre.decode-inspection.v2", ontology: "EVIDENCE-ONTOLOGY.md",
    focus: { runId: id(capture, "run"), layerId: id(capture, "layer", String(step.layer)), operationId: opId(capture, step), workerId: id(capture, "worker", String(block)), areaId: candidate.id, areaBasis: model.areaId ? "analyst selection" : "heuristic", timeNs: time, window: model.window, boundsNs: window, stampState: model.stampState(step, time) },
    scope: { description: "Selected layer in each capture, direct order neighbors and ancestors, plus focus-worker window inputs", runs: scopes.map(s => ({ runId: id(s.capture, "run"), includedOperations: s.records.length, availableOperations: s.capture.steps.length })) },
    provenance: { bundle: data.source, bundleSha256: data.sourceSha, anatomyRevision: data.sourceRevision, anatomyBinding: "Separate snapshot; not verified as capture build" },
    story: [
      { id: "story:location", question: "Where is the inspected work?", refs: [id(capture, "run"), id(capture, "layer", String(step.layer)), opId(capture, step), id(capture, "worker", String(block))] },
      { id: "story:time", question: "What occupied this worker?", refs: ["measurement:worker-window", ...entities.segments.map(s => s.id)] },
      { id: "story:order", question: "What precedes, follows or encloses it?", refs: relations.filter(r => [r.from, r.to].includes(opId(capture, step)) && ["recorded_before", "nested_in"].includes(r.kind)).flatMap(r => [r.from, r.to]) },
      { id: "story:arrivals", question: "How do observed exits differ between workers and recordings?", refs: entities.measurements.filter(m => m.id.endsWith(":exits")).map(m => m.id) },
      { id: "story:investigation", question: "Which evidence would distinguish the possible causes?", refs: [candidate.id, ...entities.checkpoints.filter(c => c.areaId === candidate.id).map(c => c.id), ...entities.hypotheses.filter(h => h.areaId === candidate.id).map(h => h.id), `requirement:${candidate.id}`, `experiment:${candidate.id}`] },
    ],
    entities, relations,
    boundaries: { units: "nanoseconds in separate capture-relative globaltimer domains; negative offsets valid", semantics: capture.semantics,
      comparison: "Different recorded token positions; workload/build controls incomplete; no speedup conclusion",
      missing: ["causal wait producer", "complete critical path", "recoverable time", "transfer timings", "cache outcomes", "per-warp issue", "verified physical placement", "exact source-build binding"],
      benchmark: "Instrumented recording; not production latency", machinery: "Symbolic movement inside recorded intervals" },
  };
}
