import { DiagnosticModel, exitDistribution, observed } from "./diagnostic-model";
import { architectureAreas } from "./diagnostic-ontology";
import { fmtNs, type RecordedStep } from "./replay-model";
import ontologyUrl from "../EVIDENCE-ONTOLOGY.md?url";
import "./diagnostic-stories.css";

const ns = "http://www.w3.org/2000/svg";
function svg(tag: string, attrs: Record<string, string | number> = {}, content?: string) {
  const node = document.createElementNS(ns, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (content) node.textContent = content;
  return node;
}
function html<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text?: string) {
  const node = document.createElement(tag); node.className = className; if (text) node.textContent = text; return node;
}
function button(label: string, act: () => void, className = "") {
  const b = html("button", className, label); b.type = "button"; b.onclick = act; return b;
}
function activate(node: Element, label: string, action: () => void) {
  node.setAttribute("role", "button"); node.setAttribute("tabindex", "0"); node.setAttribute("aria-label", label);
  node.addEventListener("click", action);
  node.addEventListener("keydown", event => { const e = event as KeyboardEvent; if (e.key === "Enter" || e.key === " ") { e.preventDefault(); action(); } });
}
function chart(label: string, height: number) {
  return svg("svg", { viewBox: `0 0 640 ${height}`, role: "group", "aria-label": label, class: "evidence-chart" });
}
function section(title: string, note?: string) {
  const node = html("section", "evidence-section"); node.append(html("h3", "", title)); if (note) node.append(html("p", "evidence-caption", note)); return node;
}
type Actions = { select: (id: string, worker?: number) => void; seek: (time: number) => void; equipment: () => void; compare: (capture: string, record: string, worker: number) => void };

export class EvidenceStories {
  private model?: DiagnosticModel;
  private clockCursor?: Element;
  private stateNodes: { node: HTMLElement; step: RecordedStep }[] = [];
  constructor(private host: HTMLElement, private actions: Actions) {}
  render(m: DiagnosticModel, time: number) {
    this.model = m; this.stateNodes = [];
    this.host.replaceChildren();
    const trail = html("ol", "evidence-trail");
    for (const name of [`Position ${m.capture.position}`, `Layer ${m.step.layer + 1}`, m.step.name, `Worker ${String(m.block).padStart(2, "0")}`]) trail.append(html("li", "", name));
    this.host.append(trail, this.workerTime(m), this.order(m), this.arrivals(m), this.comparison(m), this.opportunities(m), this.investigation(m), this.provenance(m));
    this.update(time);
  }
  private workerTime(m: DiagnosticModel) {
    const s = section("One worker through time", "Recorded time. Outline: selected operation.");
    const b = m.bounds(), width = b.end - b.start || 1, x = (time: number) => 16 + Math.max(0, Math.min(1, (time - b.start) / width)) * 608;
    const g = chart("Chronological work and wait intervals for the selected worker", 94);
    g.append(svg("rect", { x: 16, y: 20, width: 608, height: 32, class: "span-gap" }));
    for (const r of m.motion.workers[m.block] ?? []) {
      if (r.end <= b.start || r.start >= b.end) continue;
      const span = svg("rect", { x: x(r.start), y: 20, width: Math.max(.5, x(r.end) - x(r.start)), height: 32, class: `span-${r.step.kind === "wait" ? "wait" : r.step.kind === "work" ? "work" : "mixed"}` });
      span.append(svg("title", {}, `${r.step.name}: ${fmtNs(Math.min(b.end, r.end) - Math.max(b.start, r.start))}`));
      activate(span, `Go to ${r.step.name}, ${fmtNs(Math.max(b.start, r.start) - m.capture.start)} from capture start`, () => { this.actions.select(r.step.id); this.actions.seek(Math.max(b.start, r.start)); });
      g.append(span);
    }
    const o = observed(m.step).find(o => o[0] === m.block);
    if (o && o[2] > b.start && o[1] < b.end) g.append(svg("rect", { x: x(o[1]), y: 15, width: Math.max(2, x(o[2]) - x(o[1])), height: 42, class: "observation-outline" }));
    this.clockCursor = svg("path", { d: "", class: "evidence-time-cursor" }); g.append(this.clockCursor);
    g.append(svg("text", { x: 16, y: 80 }, fmtNs(b.start - m.capture.start)), svg("text", { x: 624, y: 80, "text-anchor": "end" }, fmtNs(b.end - m.capture.start)));
    s.append(g);
    const totals = m.totals(m.block), legend = html("dl", "time-breakdown");
    for (const [kind, label, value] of [["work", "Work", totals.work], ["wait", "Wait", totals.wait], ["mixed", "Mixed", totals.mixed], ["gap", "Unassigned", totals.unassigned]] as const) {
      const row = html("div"); row.append(html("dt", `key-${kind}`, label), html("dd", "", fmtNs(value))); legend.append(row);
    }
    const status = html("p", "evidence-caption"); this.stateNodes.push({ node: status, step: m.step });
    const actions = html("div", "evidence-actions");
    const entry = button("Go to entry", () => this.actions.seek(o![1])); entry.disabled = !o;
    actions.append(entry, button("Open equipment", this.actions.equipment));
    s.append(legend, status, actions); return s;
  }
  private order(m: DiagnosticModel) {
    const s = section("The recorded connections", "Program order; wait producers are not identified.");
    if (m.step.parent) {
      const parent = m.byId.get(m.step.parent);
      if (parent) s.append(button(`Inside ${parent.name}`, () => this.actions.select(parent.id), "enclosing-operation"));
    }
    const rail = html("div", "operation-rail");
    const node = (step: RecordedStep, current = false) => {
      const n = button(step.name, () => this.actions.select(step.id), `operation-node${current ? " current" : ""}`);
      const state = html("small"); n.append(state); this.stateNodes.push({ node: state, step }); return n;
    };
    for (const direction of ["before", "current", "after"] as const) {
      const col = html("div", `operation-column ${direction}`);
      col.append(html("span", "connection-label", direction === "current" ? "Inspecting" : direction === "before" ? "Before" : "After"));
      const steps = direction === "current" ? [m.step] : m.neighbors(direction).filter(s => s.kind !== "sentinel");
      for (const step of steps) col.append(node(step, direction === "current"));
      if (!steps.length) col.append(html("span", "missing-node", "No recorded link")); rail.append(col);
    }
    s.append(rail);
    if (m.step.kind === "wait" || m.step.kind === "mixed") s.append(html("div", "unknown-producer", "Producer connection missing · trace the readiness dependency"));
    return s;
  }
  private arrivals(m: DiagnosticModel) {
    const d = exitDistribution(m.step), s = section("Do the workers arrive together?", "One dot per observed exit. Time starts at the first exit.");
    if (!d.rows.length) { s.append(html("p", "missing-node", "No valid exit observations for this operation.")); return s; }
    const first = d.rows[0].end, span = Math.max(1, d.last! - first), x = (n: number) => 24 + (n - first) / span * 592;
    const laneEnds: number[] = [];
    const points = d.rows.map(row => {
      let lane = laneEnds.findIndex(end => x(row.end) - end > 14);
      if (lane < 0) lane = laneEnds.length;
      laneEnds[lane] = x(row.end); return { row, lane };
    });
    const baseline = Math.max(150, 70 + laneEnds.length * 9);
    const plot = chart("Observed exit distribution, from first to last worker", baseline + 56);
    plot.append(svg("line", { x1: 24, x2: 616, y1: baseline, y2: baseline, class: "evidence-axis" }));
    for (const { row, lane } of points) {
      const dot = svg("circle", { cx: x(row.end), cy: baseline - 11 - lane * 9, r: row.block === m.block ? 5 : 3.5,
        class: row.block === m.block ? "exit-dot selected" : "exit-dot" });
      const label = `Worker ${row.block}, exit ${fmtNs(row.end - m.capture.start)}, elapsed ${fmtNs(row.duration)}`;
      dot.append(svg("title", {}, label)); activate(dot, label, () => this.actions.select(m.stepId, row.block)); plot.append(dot);
    }
    const medianX = x(d.median!);
    plot.append(svg("line", { x1: medianX, x2: medianX, y1: 54, y2: baseline + 3, class: "median-line" }));
    plot.append(svg("text", { x: 24, y: baseline + 34 }, "First +0 ns"), svg("text", { x: 616, y: baseline + 34, "text-anchor": "end" }, `Last +${fmtNs(d.last! - first)}`));
    plot.append(svg("path", { d: `M${medianX},17 V8 H616 V17`, class: "tail-bracket" }), svg("text", { x: 320, y: 40, "text-anchor": "middle" }, `${fmtNs(d.tail!)} median to last`));
    s.append(plot, html("p", "evidence-caption", `${d.rows.length} observed / ${m.capture.grid} logical workers. Spread includes unequal work and entry timing; it is not saved time.`));
    const choice = observed(m.step).find(o => o[0] === m.block);
    if (choice) s.append(html("p", "selected-arrival", `Worker ${m.block} · entry ${fmtNs(choice[1] - m.capture.start)} · exit ${fmtNs(choice[2] - m.capture.start)} · elapsed ${fmtNs(choice[2] - choice[1])}`));
    s.append(button("Inspect the last exit", () => { this.actions.select(m.stepId, d.lastBlocks[0]); this.actions.seek(d.last!); }, "evidence-text-action")); return s;
  }
  private comparison(m: DiagnosticModel) {
    const s = section("The same operation, two recordings", "Median-to-last spread on one scale. Different positions, not a controlled A/B.");
    const rows = m.data.captures.flatMap(c => { const step = c.steps.find(s => s.id === m.stepId); if (!step) return []; const d = exitDistribution(step); return d.rows.length ? [{ c, step, d, base: c.layers[step.layer].start }] : []; });
    const max = Math.max(1, ...rows.map(r => r.d.tail!));
    for (const row of rows) {
      const current = row.c.id === m.capture.id;
      const hasEntry = observed(row.step).some(o => o[0] === m.block);
      const b = button(`${current ? "Inspecting" : "Open"} position ${row.c.position}${current ? "" : hasEntry ? " at worker entry" : " · worker entry missing"}`, () => this.actions.compare(row.c.id, row.step.id, m.block), "comparison-row");
      b.disabled = !current && !hasEntry;
      b.setAttribute("aria-pressed", String(row.c.id === m.capture.id));
      const plot = chart(`Position ${row.c.position}: ${fmtNs(row.d.tail!)} median-to-last spread`, 42);
      const end = 10 + row.d.tail! / max * 620;
      plot.append(svg("line", { x1: 10, x2: 630, y1: 21, y2: 21, class: "evidence-axis" }),
        svg("line", { x1: 10, x2: end, y1: 21, y2: 21, class: "comparison-range" }),
        svg("circle", { cx: 10, cy: 21, r: 6, class: "exit-dot" }), svg("circle", { cx: end, cy: 21, r: 6, class: "exit-dot selected" }));
      b.append(plot, html("small", "", `${fmtNs(row.d.tail!)} spread · ${row.d.rows.length} observed`)); s.append(b);
    }
    if (!rows.length) s.append(html("p", "missing-node", "This operation has no valid comparison observations.")); return s;
  }
  private opportunities(m: DiagnosticModel) {
    const detail = html("details", "evidence-section"); detail.append(html("summary", "", "Find wider exit spreads"));
    const max = Math.max(1, ...m.ranked.slice(0, 8).map(r => r.tail));
    for (const { step, tail } of m.ranked.slice(0, 8)) {
      const b = button(`Layer ${step.layer + 1} · ${step.name}`, () => this.actions.select(step.id), "spread-rank");
      const track = html("span", "spread-track"), fill = html("i"); fill.style.width = `${tail / max * 100}%`; track.append(fill);
      b.append(track, html("small", "", fmtNs(tail))); detail.append(b);
    }
    detail.append(html("p", "evidence-caption", "Root work intervals only. Ranked clues, not a critical path.")); return detail;
  }
  private investigation(m: DiagnosticModel) {
    const s = section("From observation to an experiment", "Choose an architecture area. Empty markers mean missing evidence.");
    const areas = html("div", "architecture-branches");
    for (const a of architectureAreas) {
      const b = button(a.label, () => { m.selectArea(a.id); this.render(m, this.lastTime); }, "architecture-node");
      b.setAttribute("aria-pressed", String(a.id === m.area.id)); b.append(html("small", "", a.physical)); areas.append(b);
    }
    s.append(areas);
    const a = m.area, path = html("ol", "investigation-path");
    const checkpoints = html("ol", "investigation-checkpoints");
    checkpoints.setAttribute("aria-label", "Investigation checkpoints; event timings not captured");
    for (const name of a.checkpoints) checkpoints.append(html("li", "", name));
    s.append(html("p", "evidence-caption", "What to trace · event timings not captured"), checkpoints);
    const obs = html("li", "known"), alt = html("li", "hypothesis"), need = html("li", "missing"), experiment = html("li", "conditional");
    obs.append(html("h4", "", "Observed"), html("p", "", `${observed(m.step).length} / ${m.capture.grid} workers · timing recorded`));
    const coverage = html("div", "observation-coverage");
    coverage.setAttribute("aria-label", "Worker observation coverage, not GPU utilization");
    const covered = new Set(observed(m.step).map(o => o[0]));
    for (let worker = 0; worker < m.capture.grid; worker++) { const mark = html("i", covered.has(worker) ? "recorded" : ""); mark.title = `Worker ${worker}: ${covered.has(worker) ? "observed" : "not observed"}`; coverage.append(mark); }
    obs.append(coverage);
    alt.append(html("h4", "", a.question)); const branches = html("ul", "hypothesis-branches"); for (const text of a.alternatives) branches.append(html("li", "", text)); alt.append(branches);
    need.append(html("h4", "", "Distinguish the explanations"), html("p", "", a.need), html("small", "", a.tool));
    experiment.append(html("h4", "", "Conditional experiment"), html("p", "", a.experiment));
    const details = html("details"); details.append(html("summary", "", "Prediction and refutation"), html("p", "", a.prediction), html("p", "", `Hold constant: ${a.companions.join("; ")}.`), html("p", "", `Refuted if: ${a.refutes}`)); experiment.append(details);
    path.append(obs, alt, need, experiment); s.append(path);
    const link = html("a", "evidence-text-action", "Read the investigation ontology"); link.href = ontologyUrl; link.target = "_blank"; link.rel = "noopener"; s.append(link); return s;
  }
  private provenance(m: DiagnosticModel) {
    const s = html("details", "evidence-section"); s.append(html("summary", "", "What supports this story?"));
    const chain = html("div", "provenance-chain");
    for (const [name, known] of [["Capture", true], ["Tape + tables", true], ["Exact build binding", false], ["Causal producer", false]] as const) chain.append(html("span", known ? "known" : "missing", name));
    s.append(chain, html("p", "evidence-caption", m.capture.coverage), html("p", "evidence-caption", m.capture.semantics));
    const code = html("pre", "evidence-source", `Record ${m.step.id}\n${m.step.fromTap} → ${m.step.toTap}\nTape ${m.capture.tapeSha}\nTables ${m.capture.tablesSha}\nAnatomy ${m.data.sourceRevision} (separate snapshot)`); s.append(code);
    s.append(html("p", "evidence-caption", "The export follows run → layer → operation → observations → measurements → hypotheses → required evidence → conditional experiments. Each link has a stated basis."));
    return s;
  }
  private lastTime = 0;
  update(time: number) {
    this.lastTime = time; const m = this.model; if (!m) return;
    for (const { node, step } of this.stateNodes) {
      const state = m.stampState(step, time); if (node.textContent !== state) node.textContent = state;
      node.dataset.state = state === "Exit observed" ? "observed" : state === "Between entry and exit" ? "active" : "pending";
    }
    const b = m.bounds(), p = (time - b.start) / (b.end - b.start || 1), x = 16 + p * 608;
    this.clockCursor?.setAttribute("d", p >= 0 && p <= 1 ? `M${x},8 V62` : "");
  }
}
