import "./diagnostic.css";
import { DiagnosticModel, inspectionFromHash, observed, type TimeWindow } from "./diagnostic-model";
import { DiagnosticTimeline } from "./diagnostic-timeline";
import { EvidenceStories } from "./diagnostic-stories";
import { ProfilingPanel } from "./profiling-panel";
import { RunPanel } from "./run-panel";
import { WeightPanel } from "./weight-panel";
import { fmtNs, type ReplayData, type RecordedStep } from "./replay-model";
import type { ReplayPlayback } from "./replay-playback";
import type { MechanicalWorld } from "./replay-world";

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const text = (id: string, value: string) => { if (el(id).textContent !== value) el(id).textContent = value; };
const button = (label: string, action: () => void, detail?: string) => {
  const b = document.createElement("button"); b.type = "button"; b.textContent = label;
  if (detail) { const s = document.createElement("small"); s.textContent = detail; b.append(s); }
  b.addEventListener("click", action); return b;
};

export class DiagnosticPanel {
  active = false;
  applying = false;
  model: DiagnosticModel;
  private timeline: DiagnosticTimeline;
  private aside: HTMLElement;
  private track: HTMLElement;
  private toggle: HTMLButtonElement;
  private stories: EvidenceStories;
  private profiling: ProfilingPanel;
  private run: RunPanel;
  private weights: WeightPanel;
  private automaticProfilingCapture?: string;
  constructor(private data: ReplayData, private playback: ReplayPlayback, private world: MechanicalWorld,
    private selectMachine: (step: RecordedStep, block: number) => void,
    private show: (where: "engine" | "layer" | "internals") => void,
    private useCapture: (id: string) => void) {
    this.model = new DiagnosticModel(data, playback.clock.capture, world.recorded);
    this.toggle = button("Diagnose recording", () => this.open(!this.active));
    this.toggle.id = "diagnostic-toggle"; this.toggle.setAttribute("aria-pressed", "false");
    el("view-note").before(this.toggle);
    this.aside = document.createElement("aside"); this.aside.id = "diagnostic-inspector";
    this.aside.setAttribute("aria-label", "Recorded evidence inspector");
    this.aside.innerHTML = `<div class="diagnostic-title"><h2>Inspect the evidence</h2><button id="diagnostic-back" class="diagnostic-jump">Back to the machine ↑</button><button id="diagnostic-close" aria-label="Close diagnostics">Close</button></div>
      <div class="diagnostic-selection"><label>Layer<select id="diagnostic-layer"></select></label>
      <label>Operation<select id="diagnostic-record"></select></label>
      <label>Worker<select id="diagnostic-worker"></select></label></div>
      <div id="weight-investigation"></div><div id="run-evidence"></div><div id="profiling-lenses"></div><div id="diagnostic-stories"></div>
      <div class="diagnostic-share"><button id="diagnostic-copy">Share inspection link</button><button id="diagnostic-export">Download evidence JSON</button></div><p id="diagnostic-share-status" role="status"></p><textarea id="diagnostic-link-fallback" aria-label="Inspection link to copy" readonly hidden></textarea>`;
    this.track = document.createElement("section"); this.track.id = "diagnostic-timeline";
    this.track.setAttribute("aria-label", "Recorded worker timeline");
    this.track.innerHTML = `<div class="diagnostic-track-heading"><div><h2>Where the workers spend time</h2><p id="diagnostic-window-caption"></p></div>
      <label>Time window<select id="diagnostic-window"><option value="layer">Selected layer</option><option value="operation">Selected operation</option><option value="capture">Whole recording</option></select></label></div>
      <div class="diagnostic-track-tools"><div class="diagnostic-legend"><span><i class="work"></i>Work interval</span><span><i class="wait"></i>Wait interval</span><span><i class="mixed"></i>Mixed work / wait</span><span><i class="gap"></i>Unassigned</span></div><span id="diagnostic-window-time"></span></div>
      <input id="diagnostic-scrubber" type="range" min="0" max="1" step="0.00001" aria-label="Time in inspected window">
      <button id="diagnostic-jump" class="diagnostic-jump">Inspect selected evidence ↓</button><div id="diagnostic-chart"></div><p class="diagnostic-note">Select a span to pause there. Select a worker to keep the current time. Outlines mark the inspected operation.</p>`;
    const transport = document.querySelector(".transport")!;
    transport.before(this.aside, this.track);
    this.aside.hidden = this.track.hidden = true;
    this.timeline = new DiagnosticTimeline(el("diagnostic-chart"), (block, time, record) => {
      this.select(record ?? this.model.at(block, this.playback.time)?.step.id ?? this.model.stepId, block);
      if (time !== undefined) this.seek(time);
    });
    this.stories = new EvidenceStories(el("diagnostic-stories"), {
      select: (id, worker) => this.select(id, worker), seek: time => this.seek(time),
      equipment: () => { this.applying = true; this.show("internals"); this.applying = false; },
      compare: (capture, record, worker) => {
        if (capture === this.model.capture.id) return;
        const target = this.data.captures.find(c => c.id === capture)?.steps.find(s => s.id === record);
        const entry = target && observed(target).find(o => o[0] === worker);
        if (!entry) return;
        const window = this.model.window, areaId = this.model.areaId; this.useCapture(capture);
        this.model = new DiagnosticModel(this.data, this.playback.clock.capture, this.world.recorded);
        this.model.window = window; this.select(record, Math.min(worker, this.model.capture.grid - 1));
        if (areaId) { this.model.selectArea(areaId); this.render(); }
        this.seek(entry[1]);
      },
    });
    this.profiling = new ProfilingPanel(el("profiling-lenses"), {
      select: (record, worker) => this.select(record, worker), seek: time => this.seek(time),
      machine: () => { this.applying = true; this.show("internals"); this.applying = false; el("viewport").scrollIntoView({ block: "start" }); },
      compare: (capture, record, worker) => {
        if (capture === this.model.capture.id) return;
        this.useCapture(capture); this.model = new DiagnosticModel(this.data, this.playback.clock.capture, this.world.recorded);
        this.select(record, Math.min(worker, this.model.capture.grid - 1));
      },
    });
    this.run = new RunPanel(el("run-evidence"), (id, path, record = "l0:main:2") => {
      if (!this.data.captures.some(c => c.id === id)) return;
      this.automaticProfilingCapture = id;
      this.useCapture(id); this.model = new DiagnosticModel(this.data, this.playback.clock.capture, this.world.recorded);
      const step = this.model.capture.steps.find(s => s.id === record);
      const worker = step && observed(step).sort((a, b) => (b[2] - b[1]) - (a[2] - a[1]))[0];
      this.select(record, worker?.[0] ?? 0);
      if (worker) this.seek(worker[1]);
      this.loadRunProfiling(id, path, true);
    }, run => {
      const capture = run.captures.find(c => c.id === this.model.capture.id);
      if (capture) this.loadRunProfiling(capture.id, capture.profiling, false);
    });
    const investigate = button("Investigate weights", () => { if (!this.active) this.open(true); this.weights.focus(this.model.capture.id, this.model.step.layer, this.model.stepId); el("weight-investigation").scrollIntoView({ block: "start" }); });
    investigate.id = "weight-investigation-open"; investigate.hidden = true; this.toggle.after(investigate);
    this.weights = new WeightPanel(el("weight-investigation"), (id, record) => {
      const capture = this.data.captures.find(c => c.id === id);
      if (!capture || !capture.steps.some(s => s.id === record)) return;
      this.useCapture(id); this.model = new DiagnosticModel(this.data, this.playback.clock.capture, this.world.recorded);
      const step = capture.steps.find(s => s.id === record)!;
      const worker = observed(step).sort((a, b) => (b[2] - b[1]) - (a[2] - a[1]))[0];
      this.select(record, worker?.[0] ?? 0); if (worker) this.seek(worker[1]);
      this.applying = true; this.show("internals"); this.applying = false;
      el("viewport").scrollIntoView({ block: "start" });
    }, () => { investigate.hidden = false; });
    el("diagnostic-close").onclick = () => this.open(false);
    el("diagnostic-jump").onclick = () => this.aside.scrollIntoView({ block: "start" });
    el("diagnostic-back").onclick = () => el("viewport").scrollIntoView({ block: "start" });
    el("diagnostic-record").onchange = () => this.select(el<HTMLSelectElement>("diagnostic-record").value);
    el("diagnostic-worker").onchange = () => this.inspectWorker(Number(el<HTMLSelectElement>("diagnostic-worker").value));
    el("diagnostic-layer").onchange = () => {
      const layer = Number(el<HTMLSelectElement>("diagnostic-layer").value);
      const candidates = this.model.capture.steps.filter(s => s.layer === layer && s.kind !== "sentinel");
      this.select((candidates.find(s => s.role === this.model.step.role) ?? candidates[0]).id);
    };
    el("diagnostic-window").onchange = () => { this.model.window = el<HTMLSelectElement>("diagnostic-window").value as TimeWindow; this.render(); };
    el("diagnostic-scrubber").oninput = () => {
      const b = this.model.bounds(); this.seek(b.start + Number(el<HTMLInputElement>("diagnostic-scrubber").value) * (b.end - b.start));
    };
    el("diagnostic-copy").onclick = async () => {
      const link = this.weights.link(this.profiling.link(this.model.link(this.playback.time, location.href)));
      const field = el<HTMLTextAreaElement>("diagnostic-link-fallback"); field.hidden = false; field.value = link;
      text("diagnostic-share-status", "Link ready below, including this moment and selection.");
      try { await navigator.clipboard.writeText(link); }
      catch { field.select(); }
    };
    el("diagnostic-export").onclick = () => {
      const snapshot = this.snapshot(); if (!snapshot) return;
      const url = URL.createObjectURL(new Blob([JSON.stringify(snapshot, null, 2)], { type: "application/json" }));
      const a = document.createElement("a"); a.href = url; a.download = `decode-${this.model.capture.id}-${this.model.stepId.replaceAll(":", "-")}-worker-${this.model.block}.json`; a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000); text("diagnostic-share-status", "Evidence downloaded for this selection. No data was sent anywhere.");
    };
    Object.assign(window, { decodeInspection: { snapshot: () => this.snapshot(), importProfiling: (value: unknown) => this.profiling.import(value) } });
    window.addEventListener("hashchange", () => this.restore());
  }
  snapshot() { return this.active && !this.playback.illustrated ? { ...this.model.snapshot(this.playback.time), profiling: this.profiling.snapshot(), run: this.run.snapshot(), weightInvestigation: this.weights?.snapshot() ?? null } : null; }
  private loadRunProfiling(id: string, path: string, focus: boolean) {
    const url = new URL(path, location.href);
    if (url.origin !== location.origin) return;
    this.automaticProfilingCapture = id;
    void fetch(url).then(r => { if (!r.ok) throw Error("Profiling evidence could not load."); return r.json(); }).then(value => {
      if (this.model.capture.id !== id) return;
      this.profiling.import(value);
      if (focus) {
        this.profiling.lens = "values"; this.profiling.render(this.model);
        el("profiling-lenses").scrollIntoView({ block: "start" });
      } else if (inspectionFromHash(location.hash, this.data)?.capture.id === id) {
        this.profiling.restore(new URLSearchParams(location.hash.slice(1)));
      }
    }).catch(error => text("diagnostic-share-status", String(error)));
  }
  restore() {
    const saved = inspectionFromHash(location.hash, this.data); if (!saved) return false;
    this.useCapture(saved.capture.id); this.open(true); this.model.window = saved.window;
    this.select(saved.step.id, saved.block);
    if (saved.areaId) { this.model.selectArea(saved.areaId); this.render(); }
    this.seek(saved.time); this.profiling.restore(new URLSearchParams(location.hash.slice(1)));
    const linked = new URLSearchParams(location.hash.slice(1));
    if (linked.has("wreply")) requestAnimationFrame(() => this.weights.restoreSharedSelection());
    else if (linked.has("lens")) requestAnimationFrame(() => el("profiling-lenses").scrollIntoView({ block: "start" }));
    return true;
  }
  open(on: boolean) {
    if (on && this.playback.illustrated) this.useCapture(this.playback.clock.capture.id);
    this.active = on; this.aside.hidden = this.track.hidden = !on;
    el("replay-app").classList.toggle("diagnostic-mode", on); this.toggle.setAttribute("aria-pressed", String(on));
    this.toggle.textContent = on ? "Diagnosing recording" : "Diagnose recording";
    this.world.followTray = false;
    if (on) {
      if (this.model.capture !== this.playback.clock.capture) this.model = new DiagnosticModel(this.data, this.playback.clock.capture, this.world.recorded);
      this.show("engine");
      this.select(this.model.at(this.world.selectedBlock, this.playback.time)?.step.id ?? this.model.stepId, this.world.selectedBlock);
    } else for (const l of this.world.labels) l.element.classList.remove("diagnostic-participant", "diagnostic-selected");
    requestAnimationFrame(() => { this.world.engine.resize(); this.world.fit(); this.timeline.draw(); });
  }
  select(id: string, block = this.model.block) {
    if (!this.model.select(id, block)) return;
    this.applying = true; this.selectMachine(this.model.step, this.model.block); this.applying = false;
    this.render(true);
    window.dispatchEvent(new CustomEvent("decode-inspection-change", { detail: { capture: this.model.capture.id, record: id, block } }));
  }
  fromMachine(step: RecordedStep, block: number) {
    if (!this.active || this.applying || this.model.capture !== this.playback.clock.capture) return;
    if (this.model.select(step.id, block)) this.render(true);
  }
  inspectWorker(block: number) {
    if (this.active && !this.applying) this.select(this.model.at(block, this.playback.time)?.step.id ?? this.model.stepId, block);
  }
  private seek(time: number) {
    const c = this.model.capture; this.world.workshop.crank = null;
    this.playback.seek((time - c.start) / c.duration); this.update();
  }
  private render(reveal = false) {
    if (!this.active) return;
    const m = this.model, step = m.step;
    const options = (id: string, values: [string, string][], value: string) => {
      const s = el<HTMLSelectElement>(id); s.replaceChildren(...values.map(([v, label]) => new Option(label, v))); s.value = value;
    };
    options("diagnostic-layer", m.capture.layers.map(l => [String(l.id), `${l.id + 1} · ${l.kind === "F" ? "Full context" : "Sliding window"}`]), String(step.layer));
    options("diagnostic-record", m.capture.steps.filter(s => s.layer === step.layer && s.kind !== "sentinel").map(s => [s.id, `${s.parent ? "↳ " : ""}${s.name}`]), step.id);
    options("diagnostic-worker", Array.from({ length: m.capture.grid }, (_, b) => [String(b), `Worker ${String(b).padStart(2, "0")}`]), String(m.block));
    el<HTMLSelectElement>("diagnostic-window").value = m.window;
    this.stories.render(m, this.playback.time);
    this.profiling.render(m);
    const sidecar = this.run?.snapshot()?.captures.find(c => c.id === m.capture.id);
    if (sidecar && this.automaticProfilingCapture !== sidecar.id) this.loadRunProfiling(sidecar.id, sidecar.profiling, false);
    const b = m.bounds(); text("diagnostic-window-caption", `${m.window === "capture" ? "Whole recording" : `Layer ${step.layer + 1}${m.window === "operation" ? " · selected operation" : ""}`} · ${fmtNs(b.end - b.start)} · ${m.capture.grid} logical workers`);
    text("diagnostic-share-status", ""); el("diagnostic-link-fallback").hidden = true;
    const participants = new Set(step.observations.map(o => o[0]));
    for (const l of this.world.labels) l.element.classList.toggle("diagnostic-participant", l.block !== undefined && participants.has(l.block));
    this.timeline.render(m, reveal); this.update();
  }
  update() {
    if (!this.active) return;
    if (this.playback.illustrated) { this.open(false); return; }
    if (this.model.capture !== this.playback.clock.capture) {
      this.model = new DiagnosticModel(this.data, this.playback.clock.capture, this.world.recorded);
      this.select(this.model.stepId, this.world.selectedBlock); return;
    }
    const time = this.playback.time, m = this.model;
    this.timeline.update(time);
    const b = m.bounds(), p = (time - b.start) / (b.end - b.start);
    el<HTMLInputElement>("diagnostic-scrubber").value = String(Math.max(0, Math.min(1, p)));
    text("diagnostic-window-time", `${fmtNs(time - m.capture.start)} from capture start${p < 0 || p > 1 ? " · outside this window" : ""}`);
    const state = m.stampState(m.step, time);
    // Diagnostic inspection is pinned to a record, not a different concurrent job.
    text("phase-boundary", "Inspecting recorded evidence");
    text("story-location", `Layer ${m.step.layer + 1} · Worker ${String(m.block).padStart(2, "0")} · ${m.step.name}`);
    text("phase-name", `Worker ${String(m.block).padStart(2, "0")} · Layer ${m.step.layer + 1}`);
    text("phase-detail", `${m.step.name} · ${state}`);
    for (const l of this.world.labels) l.element.classList.toggle("diagnostic-selected",
      l.block !== undefined ? l.block === m.block : l.layer !== undefined ? l.layer === m.step.layer :
      l.part !== undefined && l.part === this.world.selectedPart);
    this.stories.update(time);
    this.profiling.update(time);
  }
}
