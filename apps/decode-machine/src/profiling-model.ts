import type { Capture } from "./replay-model";

export const lenses = [
  { id: "dependencies", label: "What holds it up?", noun: "Dependencies", area: "coordination", table: "tasks", need: "Producer identities, readiness generations and consumer start times", action: "Trace the last required producer before changing assignment." },
  { id: "instructions", label: "Is it being told what to do?", noun: "Instruction supply", area: "execution", table: "instructions", need: "Instruction-fetch samples, data readiness and exact compiled-code locations", action: "Inspect the hot instruction footprint and unrolling in the same binary." },
  { id: "work", label: "How much work is useful?", noun: "Useful work", area: "execution", table: "work", need: "Valid and issued elements, instruction mix, requested and transferred bytes", action: "Test a tile or assignment that removes padding or repeated work." },
  { id: "lifetimes", label: "What is still holding space?", noun: "Buffer lifetimes", area: "transfer", table: "buffers", need: "Request, arrival, first/last use and release for each buffer generation", action: "Inspect the release that prevents the next tray from arriving." },
  { id: "bursts", label: "When does the supply stop?", noun: "Supply & activity", area: "memory", table: "samples", need: "Time-bounded memory and execution samples, with collection-pass identities", action: "Distinguish empty supply intervals from a saturated memory path." },
  { id: "values", label: "What happens to the values?", noun: "Value distributions", area: "correctness", table: "values", need: "Comparable histograms, zero and saturation counts at named boundaries", action: "Test scaling or rotation; retain the same precision and correctness checks." },
  { id: "tokens", label: "When does the run change?", noun: "Across tokens", area: "host", table: "tokens", need: "Per-token intervals, sequence identity, context length and active batch", action: "Compare like workloads around the transition, including host and GPU time." },
  { id: "measurement", label: "How was this measured?", noun: "The instruments", area: "host", table: "instruments", need: "Collection mode, pass, cache treatment, clocks, sample width and exact build", action: "Match the collection conditions before attributing a difference to code." },
] as const;
export type LensId = typeof lenses[number]["id"];
export type EvidenceMode = "recorded" | "illustrated";
export type Instrument = { id: string; label: string; method: string; source: string; pass: string; cache: string | null; clocks: string | null; sampleNs: number | null; binary: string | null };
type Row = { id: string; recordId: string; worker: number | null; instrumentId: string };
export type Task = Row & { label: string; startNs: number; endNs: number };
export type Dependency = { id: string; producerId: string; consumerId: string; readyNs: number; generation: string; instrumentId: string };
export type Instruction = Row & { label: string; pc: string; startNs: number; endNs: number; dataReadyNs: number; fetchWaitNs: number; instructions: number; bookkeeping: number };
export type Work = Row & { label: string; valid: number; issued: number; requestedBytes: number; transferredBytes: number };
export type BufferLife = Row & { label: string; generation: string; bytes: number; requestNs: number; readyNs: number; firstUseNs: number; lastUseNs: number; releaseNs: number; memory: "shared" | "KV" | "L2"; outcome: "released" | "evicted" | "retained" };
export type Sample = Row & { metric: string; unit: string; startNs: number; endNs: number; value: number };
export type Distribution = Row & { label: string; boundary: string; edges: number[]; counts: number[]; zeros: number; saturated: number | null; population?: string };
export type Token = { id: string; position: number; startNs: number; endNs: number; context: number; batch: number; sequence: string; instrumentId: string; coreMHz: number | null; powerW: number | null; limit: string | null };
export type Experiment = { id: string; lens: LensId; hypothesis: string; change: string; prediction: string; outcome: "proposed" | "supported" | "refuted"; result: string | null; evidenceIds: string[] };
export type ProfilingBundle = { schema: "qgre.profiling-evidence.v1"; id: string; label: string; origin: EvidenceMode; captureId: string; tapeSha256: string; clockDomain: string; instruments: Instrument[]; tasks: Task[]; dependencies: Dependency[]; instructions: Instruction[]; work: Work[]; buffers: BufferLife[]; samples: Sample[]; values: Distribution[]; tokens: Token[]; experiments: Experiment[] };

/** Input is local but untrusted. Validate before making any visual or causal join. */
export function parseProfilingBundle(input: unknown, capture: Capture): ProfilingBundle {
  const object = (x: unknown, label: string): Record<string, unknown> => {
    if (!x || typeof x !== "object" || Array.isArray(x)) throw Error(`${label} must be an object.`);
    return x as Record<string, unknown>;
  };
  const root = object(input, "Evidence");
  const string = (x: unknown, label: string) => { if (typeof x !== "string" || !x.trim()) throw Error(`${label} must be nonempty text.`); };
  const num = (x: unknown, label: string, nonnegative = false) => { if (typeof x !== "number" || !Number.isFinite(x) || (nonnegative && x < 0)) throw Error(`${label} must be a finite${nonnegative ? " nonnegative" : ""} number.`); };
  const count = (x: unknown, label: string) => { num(x, label, true); if (!Number.isSafeInteger(x)) throw Error(`${label} must be an integer.`); };
  const nullableText = (x: unknown, label: string) => { if (x !== null) string(x, label); };
  if (root.schema !== "qgre.profiling-evidence.v1") throw Error("Choose a qgre.profiling-evidence.v1 JSON bundle.");
  if (root.origin !== "recorded" && root.origin !== "illustrated") throw Error("Evidence origin must be recorded or illustrated.");
  if (root.captureId !== capture.id || root.tapeSha256 !== capture.tapeSha) throw Error("This bundle belongs to a different capture or tape. Select its recording first.");
  if (root.clockDomain !== `globaltimer-offsets:${capture.id}`) throw Error("Align timestamps to this capture's clock domain before importing.");
  string(root.id, "Bundle id"); string(root.label, "Bundle label");
  const tables = ["instruments", "tasks", "dependencies", "instructions", "work", "buffers", "samples", "values", "tokens", "experiments"] as const;
  const ids = new Set<string>();
  const rows = Object.fromEntries(tables.map(table => {
    const items = root[table]; if (!Array.isArray(items) || items.length > 100000) throw Error(`${table} must be an array of at most 100,000 rows.`);
    return [table, items.map(item => { const r = object(item, table); string(r.id, `${table}.id`); const id = r.id as string; if (ids.has(id)) throw Error(`Duplicate evidence id: ${id}`); ids.add(id); return r; })];
  })) as Record<typeof tables[number], Record<string, unknown>[]>;
  const instruments = new Set(rows.instruments.map(r => r.id));
  const records = new Set(capture.steps.map(s => s.id));
  const instrument = (r: Record<string, unknown>) => { if (!instruments.has(r.instrumentId)) throw Error(`${r.id}: instrument reference is missing.`); };
  const row = (r: Record<string, unknown>) => { instrument(r); if (!records.has(String(r.recordId))) throw Error(`${r.id}: operation is not in this capture.`); if (r.worker !== null && (!Number.isInteger(r.worker) || Number(r.worker) < 0 || Number(r.worker) >= capture.grid)) throw Error(`${r.id}: worker is outside this capture.`); };
  const ordered = (r: Record<string, unknown>, keys: string[], bounded = true) => {
    for (const k of keys) num(r[k], `${r.id}.${k}`);
    if (keys.some((k, i) => i > 0 && Number(r[k]) < Number(r[keys[i - 1]]))) throw Error(`${r.id}: timestamps must be ordered.`);
    if (bounded && (Number(r[keys[0]]) < capture.start || Number(r[keys.at(-1)!]) > capture.end)) throw Error(`${r.id}: timestamps fall outside this capture.`);
  };
  for (const r of rows.instruments) {
    for (const k of ["label", "method", "source", "pass"]) string(r[k], `instrument.${k}`);
    for (const k of ["cache", "clocks", "binary"]) nullableText(r[k], `instrument.${k}`);
    if (r.sampleNs !== null) { num(r.sampleNs, "Sample width", true); if (r.sampleNs === 0) throw Error("Sample width must be positive or null."); }
  }
  for (const r of rows.tasks) { row(r); string(r.label, "Task label"); ordered(r, ["startNs", "endNs"]); }
  const tasks = new Map(rows.tasks.map(r => [r.id, r]));
  for (const r of rows.dependencies) {
    instrument(r); string(r.generation, "Readiness generation"); num(r.readyNs, "Readiness time");
    const producer = tasks.get(r.producerId), consumer = tasks.get(r.consumerId);
    if (!producer || !consumer || producer === consumer) throw Error(`${r.id}: dependency endpoints must be distinct existing tasks.`);
    if (Number(r.readyNs) < Number(producer.startNs) || Number(r.readyNs) > Number(consumer.startNs)) throw Error(`${r.id}: readiness must follow producer start and precede consumer start.`);
  }
  // A cycle would turn a dependency story into invented causality.
  const visited = new Set<unknown>(), visiting = new Set<unknown>();
  const visit = (id: unknown) => { if (visiting.has(id)) throw Error("Dependency graph contains a cycle."); if (visited.has(id)) return; visiting.add(id); for (const d of rows.dependencies.filter(d => d.producerId === id)) visit(d.consumerId); visiting.delete(id); visited.add(id); };
  for (const id of tasks.keys()) visit(id);
  for (const r of rows.instructions) {
    row(r); string(r.label, "Instruction label"); string(r.pc, "Instruction PC"); ordered(r, ["startNs", "endNs"]); num(r.dataReadyNs, "Data readiness");
    if (Number(r.dataReadyNs) < capture.start || Number(r.dataReadyNs) > Number(r.endNs)) throw Error(`${r.id}: data readiness is outside the observation.`);
    num(r.fetchWaitNs, "Fetch wait", true); count(r.instructions, "Instructions"); count(r.bookkeeping, "Bookkeeping");
    if (Number(r.fetchWaitNs) > Number(r.endNs) - Number(r.startNs) || Number(r.bookkeeping) > Number(r.instructions)) throw Error(`${r.id}: instruction counts or wait exceed their total.`);
  }
  for (const r of rows.work) { row(r); string(r.label, "Work label"); for (const k of ["valid", "issued", "requestedBytes", "transferredBytes"]) count(r[k], k); if (Number(r.valid) > Number(r.issued)) throw Error(`${r.id}: valid work exceeds issued work.`); }
  for (const r of rows.buffers) {
    row(r); string(r.label, "Buffer label"); string(r.generation, "Buffer generation"); count(r.bytes, "Buffer bytes"); ordered(r, ["requestNs", "readyNs", "firstUseNs", "lastUseNs", "releaseNs"]);
    if (!["shared", "KV", "L2"].includes(String(r.memory)) || !["released", "evicted", "retained"].includes(String(r.outcome))) throw Error(`${r.id}: invalid buffer type or outcome.`);
  }
  for (const r of rows.samples) { row(r); string(r.metric, "Metric"); string(r.unit, "Unit"); ordered(r, ["startNs", "endNs"]); num(r.value, "Sample", true); if (r.startNs === r.endNs) throw Error("A sample needs a nonzero time interval."); }
  for (const r of rows.values) {
    row(r); string(r.label, "Distribution label"); string(r.boundary, "Value boundary");
    if (!Array.isArray(r.edges) || !Array.isArray(r.counts) || !r.counts.length || r.counts.length > 128 || r.edges.length !== r.counts.length + 1) throw Error(`${r.id}: histogram edges must bracket every bin (at most 128 bins).`);
    r.edges.forEach((n, i) => { num(n, "Bin edge"); if (i && n <= (r.edges as number[])[i - 1]) throw Error("Bin edges must increase."); }); r.counts.forEach(n => count(n, "Bin count"));
    count(r.zeros, "Zero count"); if (r.saturated !== null) count(r.saturated, "Saturation count"); if (r.population !== undefined) string(r.population, "Population unit"); const total = (r.counts as number[]).reduce((a, b) => a + b, 0);
    if (Number(r.zeros) > total || (r.saturated !== null && Number(r.saturated) > total)) throw Error("Zero and saturation counts cannot exceed the population.");
  }
  for (const r of rows.tokens) { instrument(r); string(r.sequence, "Sequence identity"); for (const k of ["position", "context", "batch"]) count(r[k], k); if (!r.batch) throw Error("Active batch must be positive."); ordered(r, ["startNs", "endNs"], false); for (const k of ["coreMHz", "powerW"]) if (r[k] !== null) num(r[k], k, true); nullableText(r.limit, "Limit reason"); }
  for (const r of rows.experiments) {
    if (!lenses.some(l => l.id === r.lens) || !["proposed", "supported", "refuted"].includes(String(r.outcome))) throw Error("Experiment lens or outcome is invalid.");
    for (const k of ["hypothesis", "change", "prediction"]) string(r[k], k); nullableText(r.result, "Experiment result");
    if (!Array.isArray(r.evidenceIds) || r.evidenceIds.some(id => !ids.has(id))) throw Error("Experiment evidence reference is missing.");
  }
  return structuredClone(input) as ProfilingBundle;
}

export function readiness(bundle: ProfilingBundle, taskId: string) {
  const task = bundle.tasks.find(t => t.id === taskId), inputs = bundle.dependencies.filter(d => d.consumerId === taskId);
  if (!task || !inputs.length) return null;
  const lastReadyNs = Math.max(...inputs.map(d => d.readyNs));
  return { lastReadyNs, afterReadyNs: task.startNs - lastReadyNs,
    inputs: inputs.map(d => ({ ...d, slackNs: lastReadyNs - d.readyNs, last: d.readyNs === lastReadyNs })) };
}
export const usefulFraction = (w: Work) => w.issued ? w.valid / w.issued : null;
export const lifetime = (b: BufferLife) => ({ deliveryNs: b.readyNs - b.requestNs, waitingNs: b.firstUseNs - b.readyNs, activeNs: b.lastUseNs - b.firstUseNs, heldNs: b.releaseNs - b.lastUseNs, byteNs: b.bytes * (b.releaseNs - b.readyNs) });
export function sampleGroups(bundle: ProfilingBundle) {
  const groups = new Map<string, Sample[]>();
  for (const s of bundle.samples) { const key = `${s.instrumentId}:${s.metric}:${s.unit}`; const rows = groups.get(key) ?? []; rows.push(s); groups.set(key, rows); }
  return [...groups.entries()].map(([id, rows]) => ({ id, rows: rows.sort((a, b) => a.startNs - b.startNs), instrument: bundle.instruments.find(i => i.id === rows[0].instrumentId)! }));
}
export function profilingRelations(bundle: ProfilingBundle) {
  const relations: { from: string; to: string; kind: string }[] = [];
  for (const table of [bundle.tasks, bundle.instructions, bundle.work, bundle.buffers, bundle.samples, bundle.values, bundle.tokens]) for (const row of table) relations.push({ from: row.id, to: row.instrumentId, kind: "measured_by" });
  for (const d of bundle.dependencies) { relations.push({ from: d.producerId, to: d.id, kind: "publishes" }, { from: d.id, to: d.consumerId, kind: "required_by" }, { from: d.id, to: d.instrumentId, kind: "measured_by" }); }
  for (const e of bundle.experiments) for (const id of e.evidenceIds) relations.push({ from: e.id, to: id, kind: "investigates" });
  return relations;
}
