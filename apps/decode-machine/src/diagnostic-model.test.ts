import { describe, test, expect } from "bun:test";
import dataJson from "../public/replay/gemma4-e2b.json";
import { DiagnosticModel, exitDistribution, inspectionFromHash } from "./diagnostic-model";
import { RecordedMotion } from "./recorded-motion";
import { ReplayPlayback } from "./replay-playback";
import type { ReplayData, RecordedStep } from "./replay-model";
import { equipmentPathForRole, workshopParts, mechanismForRole } from "./workshop-model";

const data = dataJson as ReplayData;
describe("shared recorded inspection", () => {
  for (const capture of data.captures) {
    test(`${capture.id}: selection and exported evidence retain raw identity and uncertainty`, () => {
      const m = new DiagnosticModel(data, capture, new RecordedMotion(capture));
      const step = capture.steps.find(s => s.kind === "wait" && s.predecessors.length)!;
      const block = step.observations[0][0], time = step.observations[0][1];
      expect(m.select(step.id, block)).toBe(true);
      const evidence = m.snapshot(time);
      expect(evidence.schema).toBe("qgre.decode-inspection.v2");
      const operation = evidence.entities.operations.find(s => s.id === evidence.focus.operationId)!;
      expect(operation.sourceRecordId).toBe(step.id);
      expect(evidence.entities.observations.filter(o => o.operationId === operation.id).map(o => [Number(o.workerId.split(":").at(-1)), o.startNs, o.endNs, o.reportedSmField])).toEqual(step.observations);
      expect(evidence.entities.runs.find(r => r.id === evidence.focus.runId)?.tapeSha256).toBe(capture.tapeSha);
      expect(evidence.relations.filter(r => r.kind === "recorded_before" && r.to === operation.id).map(r => r.from)).toEqual(step.predecessors.map(p => `operation:${capture.id}:${p}`));
      expect(evidence.boundaries.semantics).toContain("thread 0");
      expect(evidence.boundaries.missing).toContain("causal wait producer");
      expect(evidence.focus.stampState).toBe("Between entry and exit");
      expect(m.select("no-such-record", block)).toBe(false);
      expect(m.stepId).toBe(step.id);
    });
    test(`${capture.id}: relational report resolves its story and evidence back to source observations`, () => {
      const m = new DiagnosticModel(data, capture, new RecordedMotion(capture));
      m.select("l4:qtail:2", 16); m.window = "capture";
      const report = m.snapshot(capture.start);
      const ids = Object.values(report.entities).flatMap(rows => rows.map(row => row.id));
      const known = new Set(ids);
      expect(known.size).toBe(ids.length);
      for (const edge of report.relations) { expect(known.has(edge.from)).toBe(true); expect(known.has(edge.to)).toBe(true); }
      for (const chapter of report.story) for (const ref of chapter.refs) expect(known.has(ref)).toBe(true);
      for (const metric of report.entities.measurements) for (const ref of metric.inputIds) expect(known.has(ref)).toBe(true);
      expect(report.entities.runs.length).toBe(2);
      expect(report.relations.some(r => r.kind === "analogous_record")).toBe(true);
      expect(report.entities.requirements.every(r => r.availability === "missing")).toBe(true);
      expect(report.entities.hypotheses.every(h => h.status === "unproven")).toBe(true);
      expect(report.entities.workers.every(w => w.physicalSm === null)).toBe(true);
      expect(report.relations.some(r => r.kind === "causes_wait")).toBe(false);
      const segments = report.entities.segments;
      expect(segments.reduce((sum, s) => sum + s.endNs - s.startNs, 0) + m.totals(16).unassigned).toBe(capture.duration);
      for (const s of segments) expect(s.observationIds.length).toBeGreaterThan(0);
      expect(report.boundaries.comparison).toContain("no speedup");
    });
    test(`${capture.id}: analyst-selected investigation survives export, worker selection and sharing`, () => {
      const m = new DiagnosticModel(data, capture, new RecordedMotion(capture));
      m.select("l4:qtail:2", 16); m.selectArea("area:transfer"); m.select(m.stepId, 17);
      const report = m.snapshot(capture.start);
      expect(report.focus.areaId).toBe("area:transfer");
      expect(report.relations.find(r => r.kind === "investigates_area")?.basis).toBe("analyst selection; not measured placement");
      expect(report.story.find(s => s.id === "story:investigation")?.refs).toContain("experiment:area:transfer");
      const restored = inspectionFromHash(new URL(m.link(capture.start, "http://localhost:4318/")).hash, data)!;
      expect(restored.areaId).toBe("area:transfer");
      expect(m.selectArea("not-an-area")).toBe(false);
      expect(m.area.id).toBe("area:transfer");
      m.select("l0:mixer:1"); expect(m.areaId).toBeNull();
    });
    test(`${capture.id}: every worker's disjoint displayed intervals partition the window`, () => {
      const m = new DiagnosticModel(data, capture, new RecordedMotion(capture));
      for (const layer of capture.layers) {
        m.select(capture.steps.find(s => s.layer === layer.id && !s.parent && s.kind === "work")!.id);
        const b = m.bounds();
        for (let block = 0; block < capture.grid; block++) {
          const t = m.totals(block);
          expect(t.work).toBeGreaterThanOrEqual(0);
          expect(t.wait).toBeGreaterThanOrEqual(0);
          expect(t.unassigned).toBeGreaterThanOrEqual(0);
          expect(t.work + t.wait + t.mixed + t.unassigned).toBe(b.end - b.start);
        }
      }
    });
    test(`${capture.id}: link restores capture, record, worker, negative time and scope`, () => {
      const m = new DiagnosticModel(data, capture, new RecordedMotion(capture));
      m.window = "operation"; m.select(m.stepId, 3);
      const restored = inspectionFromHash(new URL(m.link(capture.start, "http://localhost:4318/")).hash, data)!;
      expect(restored.capture.id).toBe(capture.id);
      expect(restored.step.id).toBe(m.stepId);
      expect(restored.block).toBe(3);
      expect(restored.time).toBe(capture.start);
      expect(restored.window).toBe("operation");
      expect(inspectionFromHash("#diagnose=unknown", data)).toBeNull();
      expect(inspectionFromHash(`#diagnose=${capture.id}&record=${m.stepId}&block=-1&time=0&window=layer`, data)).toBeNull();
    });
    test(`${capture.id}: changing viewing duration and reverse seeking leave recorded evidence intact`, () => {
      const p = new ReplayPlayback(capture); p.use(capture, false);
      const m = new DiagnosticModel(data, capture, new RecordedMotion(capture));
      p.seek(.25); const before = m.snapshot(p.time);
      p.setDuration(600); p.seek(.75); p.seek(.25);
      expect(m.snapshot(p.time)).toEqual(before);
    });
    test(`${capture.id}: an operation-only timeline cannot select invisible concurrent work`, () => {
      const m = new DiagnosticModel(data, capture, new RecordedMotion(capture));
      const wait = capture.steps.find(s => s.id === "l0:mixer:1")!;
      m.select(wait.id, 3); m.window = "operation";
      for (let sample = 0; sample <= 10; sample++) {
        const time = wait.start + (wait.end - wait.start) * sample / 10;
        expect(m.recordAt(3, time)?.id).toBe(wait.id);
      }
    });
    test(`${capture.id}: mixed stamps retain their unresolved duration and exact equipment`, () => {
      const m = new DiagnosticModel(data, capture, new RecordedMotion(capture));
      const step = capture.steps.find(s => s.id === "l4:qtail:2")!;
      expect(step.kind).toBe("mixed");
      const block = step.observations.find(o => o[2] > o[1])![0];
      m.select(step.id, block); m.window = "operation";
      expect(m.totals(block).mixed).toBeGreaterThan(0);
      const path = equipmentPathForRole(workshopParts(capture, step.layer, mechanismForRole(step.role)), step.role);
      expect(path.at(-1)?.role).toBe(step.role);
    });
  }
  test("exit distribution reports ties and median, excluding reversed and zero intervals", () => {
    const s = { observations: [[0,0,10,""],[1,0,20,""],[2,0,30,""],[3,1,30,""],[4,5,4,""],[5,1,1,""]] } as RecordedStep;
    const d = exitDistribution(s);
    expect(d.median).toBe(25); expect(d.last).toBe(30); expect(d.tail).toBe(5);
    expect(d.lastBlocks).toEqual([2,3]); expect(d.rows.length).toBe(4);
  });
});
