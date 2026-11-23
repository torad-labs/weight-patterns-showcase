import { type Capture, type RecordedStep, type ReplayData } from "./replay-model";
import { RecordedMotion } from "./recorded-motion";
import { inspectionReport } from "./inspection-report";
import { architectureAreas, candidateArea } from "./diagnostic-ontology";

export type TimeWindow = "layer" | "operation" | "capture";
export type Bounds = { start: number; end: number };
export const observed = (s: RecordedStep) => s.observations.filter(o =>
  Number.isFinite(o[1]) && Number.isFinite(o[2]) && o[2] > o[1]);
export function exitDistribution(step: RecordedStep) {
  const rows = observed(step).map(o => ({ block: o[0], start: o[1], end: o[2], duration: o[2] - o[1] }))
    .sort((a, b) => a.end - b.end || a.block - b.block);
  const middle = Math.floor(rows.length / 2);
  const median = rows.length ? (rows[middle].end + rows[Math.floor((rows.length - 1) / 2)].end) / 2 : null;
  const last = rows.at(-1)?.end ?? null;
  return { rows, median, last, tail: median === null || last === null ? null : last - median,
    lastBlocks: rows.filter(r => r.end === last).map(r => r.block) };
}

/** Same records and resolved intervals as the machinery; no second replay clock. */
export class DiagnosticModel {
  readonly byId: Map<string, RecordedStep>;
  readonly successors = new Map<string, RecordedStep[]>();
  readonly ranked: { step: RecordedStep; tail: number }[];
  stepId: string;
  block = 0;
  window: TimeWindow = "layer";
  areaId: string | null = null;
  get area() { return architectureAreas.find(a => a.id === this.areaId) ?? candidateArea(this.step); }
  selectArea(id: string) {
    if (!architectureAreas.some(a => a.id === id)) return false;
    this.areaId = id; return true;
  }

  constructor(readonly data: ReplayData, readonly capture: Capture, readonly motion: RecordedMotion) {
    this.byId = new Map(capture.steps.map(s => [s.id, s]));
    for (const s of capture.steps) for (const id of s.predecessors) {
      const list = this.successors.get(id) ?? [];
      list.push(s); this.successors.set(id, list);
    }
    this.stepId = capture.steps.find(s => s.kind !== "sentinel" && observed(s).length)?.id ?? capture.steps[0].id;
    this.ranked = capture.steps.filter(s => !s.parent && s.kind === "work")
      .map(step => ({ step, tail: exitDistribution(step).tail ?? 0 }))
      .sort((a, b) => b.tail - a.tail || a.step.id.localeCompare(b.step.id));
  }
  get step() { return this.byId.get(this.stepId)!; }
  select(id: string, block = this.block) {
    const step = this.byId.get(id);
    if (!step || step.kind === "sentinel" || !Number.isInteger(block) || block < 0 || block >= this.capture.grid) return false;
    if (id !== this.stepId) this.areaId = null;
    this.stepId = id; this.block = block; return true;
  }
  bounds(): Bounds {
    if (this.window === "operation" && observed(this.step).length) {
      const rows = observed(this.step);
      return { start: Math.min(...rows.map(o => o[1])), end: Math.max(...rows.map(o => o[2])) };
    }
    if (this.window === "capture") return { start: this.capture.start, end: this.capture.end };
    const l = this.capture.layers[this.step.layer];
    return { start: l.start, end: l.end };
  }
  totals(block: number, bounds = this.bounds()) {
    let work = 0, wait = 0, mixed = 0;
    for (const r of this.motion.workers[block] ?? []) {
      const duration = Math.max(0, Math.min(bounds.end, r.end) - Math.max(bounds.start, r.start));
      if (r.step.kind === "wait") wait += duration;
      else if (r.step.kind === "work") work += duration;
      else mixed += duration;
    }
    return { work, wait, mixed, unassigned: Math.max(0, bounds.end - bounds.start - work - wait - mixed) };
  }
  at(block: number, time: number) {
    return this.motion.workers[block]?.find(r => r.start <= time && time < r.end);
  }
  recordAt(block: number, time: number) {
    // Operation zoom draws only this record. Blank space must not select hidden work.
    return this.window === "operation" ? this.step : this.at(block, time)?.step;
  }
  neighbors(direction: "before" | "after") {
    return direction === "before"
      ? this.step.predecessors.map(id => this.byId.get(id)).filter((s): s is RecordedStep => !!s)
      : this.successors.get(this.stepId) ?? [];
  }
  stampState(step: RecordedStep, time: number, block = this.block) {
    const o = step.observations.find(o => o[0] === block);
    return !o ? "Not observed on this worker" : o[2] < o[1] ? "Invalid interval" :
      time < o[1] ? "Entry not yet observed" : time < o[2] ? "Between entry and exit" : "Exit observed";
  }
  snapshot(time: number) {
    return inspectionReport(this, time);
  }
  link(time: number, base: string) {
    const url = new URL(base);
    url.hash = new URLSearchParams({ diagnose: this.capture.id, record: this.stepId, block: String(this.block),
      time: String(time), window: this.window, ...(this.areaId ? { area: this.areaId } : {}) }).toString();
    return url.href;
  }
}

export function inspectionFromHash(hash: string, data: ReplayData) {
  const p = new URLSearchParams(hash.replace(/^#/, ""));
  const capture = data.captures.find(c => c.id === p.get("diagnose"));
  if (!capture) return null;
  const step = capture.steps.find(s => s.id === p.get("record") && s.kind !== "sentinel");
  const block = Number(p.get("block")), time = Number(p.get("time"));
  const window = p.get("window") as TimeWindow;
  if (!step || !p.has("block") || !p.has("time") || !Number.isInteger(block) || block < 0 || block >= capture.grid || !Number.isFinite(time) ||
    !["capture", "layer", "operation"].includes(window)) return null;
  const areaId = architectureAreas.some(a => a.id === p.get("area")) ? p.get("area") : null;
  return { capture, step, block, time: Math.max(capture.start, Math.min(capture.end, time)), window, areaId };
}
