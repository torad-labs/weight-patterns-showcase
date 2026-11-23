import type { DiagnosticModel } from "./diagnostic-model";
import { fmtNs } from "./replay-model";
import { drawEvidence, mark } from "./profiling-drawings";
import { profilingExample } from "./profiling-example";
import { lenses, lifetime, parseProfilingBundle, profilingRelations, readiness, usefulFraction, type EvidenceMode, type LensId, type ProfilingBundle } from "./profiling-model";
import contractUrl from "../PROFILING-LENSES.md?url";
import "./profiling.css";

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className = "", text?: string) { const n = document.createElement(tag); n.className = className; if (text !== undefined) n.textContent = text; return n; }
function button(text: string, action: () => void, className = "") { const n = el("button", className, text); n.type = "button"; n.dataset.profileFocus = text; n.onclick = action; return n; }
function download(value: unknown, name: string) { const url = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" })); const a = el("a"); a.href = url; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }

export class ProfilingPanel {
  lens: LensId = "dependencies";
  mode: EvidenceMode = "recorded";
  selected: string | null = null;
  private imported = new Map<string, ProfilingBundle>();
  private linkedBundle: string | null = null;
  private model?: DiagnosticModel;
  private bundle?: ProfilingBundle;
  private progress = .52;
  private playing = false;
  private seconds = 20;
  private lastTick = 0;
  private plot = el("div", "profile-plot");
  private detail = el("div", "profile-detail");
  private navigation = el("nav", "profile-navigation");
  private title = el("h3");
  private context = el("p", "profile-context");
  private status = el("p", "profile-status");
  private toggle = el("div", "profile-mode");
  private tools = el("div", "profile-tools");
  private crank = el("input");
  private play = button("Play example", () => { if (this.progress >= 1) this.progress = 0; this.playing = !this.playing; this.play.textContent = this.playing ? "Pause example" : "Play example"; });
  private clockNodes: Element[] = [];
  private motionNodes: Element[] = [];
  private resize = new ResizeObserver(() => { const compact = this.host.clientWidth < 700; if (compact !== this.compact) { this.compact = compact; this.renderView(); } });
  private compact = false;
  constructor(readonly host: HTMLElement, private actions: { seek: (time: number) => void; select: (record: string, worker: number) => void; machine: () => void; compare: (capture: string, record: string, worker: number) => void }) {
    host.id = "profiling-lenses"; host.className = "profiling-lenses";
    host.setAttribute("aria-label", "Profiling evidence workbench");
    const heading = el("div", "profile-heading"), naming = el("div"); naming.append(el("h2", "", "Follow the reason"), this.context); heading.append(naming, this.toggle);
    this.navigation.setAttribute("aria-label", "Evidence lenses");
    for (const l of lenses) {
      const b = button(l.noun, () => { this.lens = l.id; this.selected = null; this.renderView(); }, "profile-lens"); b.dataset.lens = l.id;
      const icon = mark("svg", { viewBox: "0 0 40 28", "aria-hidden": "true" });
      const paths: Record<LensId, string> = {
        dependencies: "M3 5H15V14H27M3 23H15V14M27 7V21H37V7Z", instructions: "M3 3H28V25H3ZM9 8H13M18 8H22M9 14H13M18 14H22M9 20H13M28 14H38",
        work: "M3 5H37V23H3ZM11 5V23M20 5V23M29 5V23M3 14H37", lifetimes: "M3 4H37M3 14H37M3 24H37M8 1V7M24 11V17M17 21V27",
        bursts: "M2 24H7V11H13V4H20V24H26V8H32V18H38", values: "M3 25V20H9V25M12 25V6H18V25M21 25V11H27V25M30 25V18H36V25",
        tokens: "M3 22L11 19L19 22L27 4L37 17M27 4V25", measurement: "M3 4H15V24H3ZM9 8V13M24 4H37V24H24ZM30 8V20M15 14H24",
      };
      icon.append(mark("path", { d: paths[l.id] })); b.prepend(icon); this.navigation.append(b);
    }
    const stage = el("div", "profile-stage"); stage.append(this.title, this.plot, this.tools);
    const body = el("div", "profile-body"); body.append(this.navigation, stage, this.detail);
    this.status.setAttribute("role", "status");
    const footer = el("div", "profile-footer");
    const file = el("input"); file.type = "file"; file.accept = ".json,application/json"; file.id = "profile-file"; file.hidden = true;
    file.onchange = async () => { const chosen = file.files?.[0]; if (!chosen) return; try { if (chosen.size > 12 * 1024 * 1024) throw Error("Choose a bundle smaller than 12 MB."); this.import(JSON.parse(await chosen.text())); } catch (error) { this.status.textContent = error instanceof Error ? error.message : "Cannot read this evidence bundle."; } finally { file.value = ""; } };
    const link = el("a", "", "Capture format"); link.href = contractUrl; link.target = "_blank"; link.rel = "noopener";
    footer.append(button("Import profiling evidence", () => file.click()), button("Download example bundle", () => { if (this.model) download(profilingExample(this.model.capture, this.model.step, this.model.block), "profiling-example.json"); }), link, file);
    host.append(heading, body, footer, this.status);
    this.resize.observe(host);
    this.crank.type = "range"; this.crank.min = "0"; this.crank.max = "1"; this.crank.step = "0.001"; this.crank.value = String(this.progress); this.crank.setAttribute("aria-label", "Example position");
    this.crank.oninput = () => { this.progress = Number(this.crank.value); this.playing = false; this.play.textContent = "Play example"; this.pose(); };
  }
  import(value: unknown) {
    if (!this.model) throw Error("Open a recorded inspection first.");
    const bundle = parseProfilingBundle(value, this.model.capture); this.imported.set(`${bundle.captureId}:${bundle.origin}`, bundle);
    this.mode = bundle.origin; this.selected = null; this.linkedBundle = null; this.status.textContent = `Loaded ${bundle.label} · ${bundle.origin}.`; this.renderView();
    return { id: bundle.id, origin: bundle.origin };
  }
  render(model: DiagnosticModel) { const changed = this.model?.capture.id !== model.capture.id || this.model?.stepId !== model.stepId || this.model?.block !== model.block; this.model = model; if (changed) this.selected = null; this.renderView(); }
  restore(p: URLSearchParams) { if (lenses.some(l => l.id === p.get("lens"))) this.lens = p.get("lens") as LensId; this.mode = p.get("evidence") === "illustrated" ? "illustrated" : "recorded"; this.selected = p.get("evidenceItem"); this.linkedBundle = p.get("evidenceBundle"); const progress = Number(p.get("exampleAt")); if (p.has("exampleAt") && Number.isFinite(progress)) this.progress = Math.max(0, Math.min(1, progress)); this.renderView(); }
  link(base: string) { const url = new URL(base), p = new URLSearchParams(url.hash.slice(1)); p.set("lens", this.lens); p.set("evidence", this.mode); p.delete("evidenceItem"); p.delete("exampleAt"); p.delete("evidenceBundle"); if (this.selected) p.set("evidenceItem", this.selected); if (this.mode === "illustrated") p.set("exampleAt", String(this.progress)); const imported = this.model && this.imported.get(`${this.model.capture.id}:${this.mode}`); if (imported) p.set("evidenceBundle", imported.id); url.hash = p.toString(); return url.href; }
  snapshot() {
    const b = this.bundle;
    return { schema: "qgre.profiling-inspection.v1", focus: { lens: this.lens, mode: this.mode, itemId: this.selected, exampleProgress: this.mode === "illustrated" ? this.progress : null },
      bundle: b ?? null, relations: b ? profilingRelations(b) : [],
      joins: b ? [b.tasks, b.instructions, b.work, b.buffers, b.samples, b.values].flat().map(r => ({ from: r.id, to: `operation:${b.captureId}:${r.recordId}`, kind: "observes_operation", scope: "Referenced capture; operation may be outside the exported layer subset" })) : [],
      missing: b ? lenses.filter(l => !b[l.table].length).map(l => ({ lens: l.id, needs: l.need })) : lenses.map(l => ({ lens: l.id, needs: l.need })),
      semantics: { dependencies: "Explicit recorded prerequisites only; no complete critical-path or recoverable-time claim", buffers: "Retained endpoints are observation bounds; byte-ns is capacity-time, not latency", samples: "Each interval belongs to its own instrument/pass; no invented simultaneity", numerical: "Histograms do not authorize skipping work", imported: "Caller-supplied provenance; tape binding is checked, physical claims are not independently authenticated" } };
  }
  private scopedBundle(b: ProfilingBundle): ProfilingBundle {
    const m = this.model!; const belongs = (r: { recordId: string; worker: number | null }) => r.recordId === m.stepId && (r.worker === null || r.worker === m.block);
    const tasks = b.tasks.filter(belongs), kept = new Set(tasks.map(r => r.id));
    // Include the prerequisite closure rather than severing cross-operation producer edges.
    let grew = true; while (grew) { grew = false; for (const d of b.dependencies) if (kept.has(d.consumerId) && !kept.has(d.producerId)) { kept.add(d.producerId); grew = true; } }
    return { ...b, tasks: b.tasks.filter(t => kept.has(t.id)), dependencies: b.dependencies.filter(d => kept.has(d.producerId) && kept.has(d.consumerId)), instructions: b.instructions.filter(belongs), work: b.work.filter(belongs), buffers: b.buffers.filter(belongs), samples: b.samples.filter(belongs), values: b.values.filter(belongs) };
  }
  private renderView() {
    const m = this.model; if (!m) return;
    const active = document.activeElement, hadFocus = !!active && this.host.contains(active);
    const focusedEvidence = hadFocus ? active?.getAttribute("data-evidence-id") : null;
    const focusedControl = hadFocus ? active?.getAttribute("data-profile-focus") : null;
    const focusedPicker = hadFocus && active?.getAttribute("aria-label") === "Evidence piece";
    const lens = lenses.find(l => l.id === this.lens)!;
    const loaded = this.imported.get(`${m.capture.id}:${this.mode}`);
    const needsImport = this.linkedBundle !== null && loaded?.id !== this.linkedBundle;
    this.bundle = needsImport ? undefined : loaded ?? (this.mode === "illustrated" ? profilingExample(m.capture, m.step, m.block) : undefined);
    if (needsImport) this.status.textContent = `This shared view needs evidence bundle ${this.linkedBundle}. Import it to restore the observations; shared links do not upload files.`;
    const b = this.bundle && this.scopedBundle(this.bundle);
    const rows = b?.[lens.table] ?? [];
    if (!rows.some(r => r.id === this.selected) && !b?.tasks.some(r => r.id === this.selected)) this.selected = rows[0]?.id ?? null;
    this.context.textContent = `Layer ${m.step.layer + 1} · ${m.step.name} · worker ${String(m.block).padStart(2, "0")}`;
    this.toggle.replaceChildren();
    for (const [mode, label] of [["recorded", "Recorded evidence"], ["illustrated", "Explore an example"]] as const) {
      const btn = button(label, () => { this.mode = mode; this.selected = null; this.playing = false; this.linkedBundle = null; this.status.textContent = ""; this.renderView(); }); btn.setAttribute("aria-pressed", String(this.mode === mode)); this.toggle.append(btn);
    }
    for (const n of this.navigation.querySelectorAll<HTMLElement>("[data-lens]")) { n.setAttribute("aria-pressed", String(n.dataset.lens === this.lens)); const l = lenses.find(l => l.id === n.dataset.lens)!; n.dataset.available = String(!!b?.[l.table].length || l.id === "tokens" || l.id === "measurement"); }
    this.title.textContent = lens.label;
    this.plot.replaceChildren(); this.detail.replaceChildren(); this.tools.replaceChildren(); this.clockNodes = []; this.motionNodes = [];
    const origin = el("div", `profile-origin ${this.mode}`, this.mode === "illustrated" ? "Illustrated example · authored values, not this recording" : b ? `Imported evidence · ${b.label}` : "Recorded intervals available · deeper measurements not captured");
    this.plot.append(origin);
    if (rows.length && b) {
      const drawing = drawEvidence(this.lens, b, this.selected, id => { this.selected = id; this.renderView(); }, this.compact);
      this.plot.append(drawing.svg); this.clockNodes = [...drawing.svg.querySelectorAll(".profile-time-line")];
      this.motionNodes = [...drawing.svg.querySelectorAll("[data-route], [data-stroke], [data-card]")];
      const legend = el("div", "profile-chart-key");
      const keys = this.lens === "dependencies" ? [["amber", "Last required input"], ["ink", "Consumer starts"], ["wait", "Time between them"], ["cursor", this.mode === "illustrated" ? "Example time" : "Recorded time"]] : [["selected", "Selected evidence"]];
      for (const [kind, label] of keys) { const key = el("span"); key.append(el("i", kind), document.createTextNode(label)); legend.append(key); }
      this.plot.append(legend);
      const choices = el("div", "profile-pieces"); choices.setAttribute("aria-label", "Evidence pieces");
      // Dense sample series are selected in the chart. Other entities retain full textual access.
      if (this.lens === "bursts") {
        const groups = [...new Map(b.samples.map(r => [`${r.instrumentId}:${r.metric}:${r.unit}`, r])).values()];
        for (const row of groups) choices.append(button(`${row.metric} · ${b.instruments.find(i => i.id === row.instrumentId)?.pass}`, () => { this.selected = row.id; this.renderView(); }));
      }
      else if (rows.length > 16) {
        const label = el("label", "profile-piece-picker", `Inspect one of ${rows.length} pieces`), select = el("select");
        select.setAttribute("aria-label", "Evidence piece");
        for (const row of rows) { const name = "label" in row ? String(row.label) : "position" in row ? `Token ${row.position}` : row.id; select.add(new Option(name, row.id)); }
        select.value = this.selected ?? ""; select.onchange = () => { this.selected = select.value; this.renderView(); }; label.append(select); choices.append(label);
      }
      else for (const row of rows) {
        const name = "label" in row ? String(row.label) : "position" in row ? `Token ${row.position}` : row.id;
        const btn = button(name, () => { this.selected = row.id; this.renderView(); }); btn.setAttribute("aria-pressed", String(row.id === this.selected)); choices.append(btn);
      }
      this.plot.append(choices); this.details(b, drawing.notes);
    } else if (this.lens === "tokens") this.recordedTokens();
    else if (this.lens === "measurement") this.recordedInstruments();
    else this.missing();
    if (this.mode === "illustrated") {
      this.play.disabled = matchMedia("(prefers-reduced-motion: reduce)").matches;
      this.play.textContent = this.play.disabled ? "Scrub the example" : this.playing ? "Pause example" : "Play example";
      const label = el("label", "profile-duration", "Whole example"); const duration = el("input"); duration.type = "number"; duration.min = "2"; duration.max = "300"; duration.value = String(this.seconds); duration.setAttribute("aria-label", "Example duration in seconds"); duration.onchange = () => { this.seconds = Math.max(2, Math.min(300, Number(duration.value) || 20)); duration.value = String(this.seconds); }; label.append(duration, document.createTextNode("s"));
      this.tools.append(this.play, this.crank, label); this.pose();
    } else {
      const row = b && [b.tasks, b.instructions, b.work, b.buffers, b.samples, b.values].flat().find(r => r.id === this.selected);
      if (row) this.tools.append(button("Inspect this operation", () => this.actions.select(row.recordId, row.worker ?? m.block)));
      if (row && "startNs" in row) this.tools.append(button("Go to this interval", () => this.actions.seek(row.startNs)));
    }
    this.tools.append(button("Open selected machine", this.actions.machine));
    if (hadFocus && !this.host.contains(active)) {
      const candidates = [...this.host.querySelectorAll<HTMLElement>("[data-evidence-id], [data-profile-focus], select")];
      const replacement = candidates.find(n => focusedEvidence ? n.getAttribute("data-evidence-id") === focusedEvidence && n.getAttribute("tabindex") === "0" : focusedControl ? n.dataset.profileFocus === focusedControl : focusedPicker && n.getAttribute("aria-label") === "Evidence piece");
      replacement?.focus({ preventScroll: true });
    }
  }
  private details(b: ProfilingBundle, notes: string[]) {
    const lens = lenses.find(l => l.id === this.lens)!;
    const row = [b.tasks, b.instructions, b.work, b.buffers, b.samples, b.values, b.tokens, b.instruments].flat().find(r => r.id === this.selected);
    this.detail.append(el("h4", "", "Connected evidence"));
    const facts: [string, string][] = [];
    if (row) {
      if ("instrumentId" in row) { const instrument = b.instruments.find(i => i.id === row.instrumentId)!; facts.push(["Observed with", instrument.label], ["Pass", instrument.pass]); }
      if ("valid" in row) facts.push(["Valid / issued", `${row.valid} / ${row.issued}`], ["Useful fraction", usefulFraction(row) === null ? "Unknown" : `${(usefulFraction(row)! * 100).toFixed(1)}%`], ["Requested / moved", `${row.requestedBytes.toLocaleString()} / ${row.transferredBytes.toLocaleString()} B`]);
      if ("releaseNs" in row) { const l = lifetime(row); facts.push(["Buffer generation", row.generation], ["Waiting to be used", fmtNs(l.waitingNs)], ["Held after last use", fmtNs(l.heldNs)], ["Capacity × residency", `${(l.byteNs / 1024 / 1000).toFixed(1)} KiB·µs`], ["Endpoint", row.outcome === "retained" ? "Still retained at observation end" : row.outcome]); }
      const instruction = b.instructions.find(i => i.id === row.id);
      if (instruction) facts.push(["Compiled location", instruction.pc], ["Data ready", fmtNs(instruction.dataReadyNs - this.model!.capture.start)], ["Fetch wait", fmtNs(instruction.fetchWaitNs)], ["Bookkeeping / all instructions", `${instruction.bookkeeping} / ${instruction.instructions}`]);
      if ("counts" in row) facts.push(["Boundary", row.boundary], ["Population", `${row.counts.reduce((a, z) => a + z, 0).toLocaleString()} ${row.population ?? "values"}`], ["Zeros / saturated", `${row.zeros} / ${row.saturated ?? "Unknown"}`]);
      if ("metric" in row) facts.push(["Metric", row.metric], ["Value", `${row.value} ${row.unit}`], ["Observation width", fmtNs(row.endNs - row.startNs)]);
      if ("sequence" in row) facts.push(["Sequence", row.sequence], ["Token / context / batch", `${row.position} / ${row.context} / ${row.batch}`], ["Interval", fmtNs(row.endNs - row.startNs)], ["Core clock / power", `${row.coreMHz ?? "?"} MHz / ${row.powerW ?? "?"} W`], ["Limit", row.limit ?? "Not recorded"]);
      if ("method" in row) facts.push(["Method", row.method], ["Source", row.source], ["Cache", row.cache ?? "Unknown"], ["Clock treatment", row.clocks ?? "Unknown"], ["Sampling width", row.sampleNs === null ? "Not specified" : fmtNs(row.sampleNs)], ["Exact binary", row.binary ?? "Not bound"]);
      if (this.lens === "dependencies") {
        const consumer = b.dependencies.find(d => d.producerId === row.id)?.consumerId ?? row.id; const state = readiness(b, consumer);
        if (state) { facts.push(["After last input → start", fmtNs(state.afterReadyNs)]); const related = b.dependencies.filter(d => d.producerId === row.id || d.consumerId === row.id); for (const d of related) facts.push(["Readiness generation", d.generation]); }
      }
    }
    const dl = el("dl"); for (const [name, value] of facts) { const r = el("div"); r.append(el("dt", "", name), el("dd", "", value)); dl.append(r); } this.detail.append(dl);
    const description = el("details", "profile-reading"); description.append(el("summary", "", "How to read this relationship")); for (const n of notes) description.append(el("p", "", n)); this.detail.append(description);
    const next = el("div", "profile-next"); next.append(el("h4", "", "Next useful question"), el("p", "", lens.action));
    const related: Record<LensId, LensId> = { dependencies: "instructions", instructions: "work", work: "bursts", lifetimes: "dependencies", bursts: "lifetimes", values: "work", tokens: "measurement", measurement: "tokens" };
    const target = lenses.find(l => l.id === related[this.lens])!; next.append(button(`Follow to ${target.noun.toLowerCase()}`, () => { this.lens = target.id; this.selected = null; this.renderView(); })); this.detail.append(next);
    for (const e of b.experiments.filter(e => e.lens === this.lens)) { const details = el("details", "profile-experiment"); details.append(el("summary", "", `${e.outcome === "refuted" ? "Refuted" : e.outcome === "supported" ? "Supported" : "Proposed"} · ${e.hypothesis}`), el("p", "", e.change), el("p", "", e.prediction)); if (e.result) details.append(el("p", "", e.result)); this.detail.append(details); }
  }
  private missing() {
    const lens = lenses.find(l => l.id === this.lens)!, m = this.model!;
    const scene = el("div", "profile-missing");
    const svg = mark("svg", { viewBox: "0 0 700 240", "aria-label": "A recorded operation connected to an unmeasured prerequisite", role: "img" });
    svg.append(mark("path", { d: "M70 115H290M410 115H630", class: "connection" }), mark("rect", { x: 35, y: 74, width: 100, height: 82, rx: 5, class: "teal-part" }), mark("circle", { cx: 350, cy: 115, r: 46, class: "missing-part" }), mark("rect", { x: 570, y: 74, width: 100, height: 82, rx: 5, class: "empty-part" }), mark("text", { x: 350, y: 129, "text-anchor": "middle", class: "missing-question" }, "?"));
    const interval = m.step.observations.find(o => o[0] === m.block);
    scene.append(svg, el("p", "", interval ? `${fmtNs(interval[2] - interval[1])} recorded on this worker. The ${lens.noun.toLowerCase()} connection is still unmeasured.` : `This worker has no recorded interval for the selected operation.`));
    scene.append(button("See how this relationship works", () => { this.mode = "illustrated"; this.selected = null; this.linkedBundle = null; this.status.textContent = ""; this.renderView(); }, "profile-example-action")); this.plot.append(scene);
    this.detail.append(el("h4", "", "Connect the missing evidence"), el("p", "", lens.need));
    const chain = el("div", "profile-evidence-chain"); for (const [name, yes] of [["Operation", true], ["Worker stamps", !!interval], [lens.noun, false]] as const) chain.append(el("span", yes ? "available" : "unavailable", name));
    this.detail.append(chain, el("p", "", "Import a profiling bundle for this recording to replace this gap with its observations."));
  }
  private recordedTokens() {
    const m = this.model!; const max = Math.max(...m.data.captures.map(c => c.duration));
    const shelf = el("div", "profile-capture-comparison");
    for (const c of m.data.captures) { const row = button(`Position ${c.position}`, () => this.actions.compare(c.id, m.stepId, m.block)); const rail = el("span", "profile-capture-rail"), fill = el("i"); fill.style.width = `${c.duration / max * 100}%`; rail.append(fill); row.append(rail, el("strong", "", fmtNs(c.duration))); row.setAttribute("aria-pressed", String(c.id === m.capture.id)); shelf.append(row); }
    this.plot.append(shelf); this.detail.append(el("h4", "", "Two independent recordings"), el("p", "", "These bars share a zero and a duration scale. Their token positions are not a continuous sequence or a controlled A/B."), el("p", "", "A sequence bundle adds the surrounding tokens, context, batch, clocks and limit reasons."));
  }
  private recordedInstruments() {
    const chain = el("div", "profile-recording-chain");
    for (const [name, detail, known] of [["Device stamps", "Thread 0 entry and exit", true], ["Capture identity", "Tape and launch-table hashes", true], ["Exact binary", "Not bound to this capture", false], ["Cache & clock settings", "Not recorded", false]] as const) { const item = el("div", known ? "available" : "unavailable"); item.append(el("strong", "", name), el("span", "", detail)); chain.append(item); }
    this.plot.append(chain); this.detail.append(el("h4", "", "What this recording can establish"), el("p", "", this.model!.capture.semantics ?? "Thread-0 device stamp intervals."), el("p", "", "Source anatomy is a separate snapshot. No cache, clock, sampling-pass or production-latency claim is inferred from it."));
  }
  update(time: number) {
    const now = performance.now(), elapsed = this.lastTick ? Math.min(.1, (now - this.lastTick) / 1000) : 0; this.lastTick = now;
    if (this.mode === "illustrated") { if (this.playing && !matchMedia("(prefers-reduced-motion: reduce)").matches) { this.progress = Math.min(1, this.progress + elapsed / this.seconds); if (this.progress === 1) { this.playing = false; this.play.textContent = "Replay example"; } } this.pose(); }
    else this.pose(time);
  }
  private pose(recordedTime?: number) {
    this.crank.value = String(this.progress);
    for (const node of this.clockNodes) { const lo = Number(node.getAttribute("data-low")), hi = Number(node.getAttribute("data-high")), left = Number(node.getAttribute("data-left")), right = Number(node.getAttribute("data-right")); const p = recordedTime === undefined ? this.progress : (recordedTime - lo) / (hi - lo || 1); node.setAttribute("visibility", p < 0 || p > 1 ? "hidden" : "visible"); node.setAttribute("x1", String(left + p * (right - left))); node.setAttribute("x2", String(left + p * (right - left))); }
    if (this.mode !== "illustrated") return;
    for (const node of this.motionNodes) {
      const from = Number(node.getAttribute("data-from") ?? 0), to = Number(node.getAttribute("data-to") ?? 1), local = Math.max(0, Math.min(1, (this.progress - from) / (to - from || 1)));
      if (node.hasAttribute("data-route")) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path"); path.setAttribute("d", node.getAttribute("data-route")!);
        const p = path.getPointAtLength(path.getTotalLength() * local); node.setAttribute("transform", `translate(${p.x},${p.y})`); node.setAttribute("opacity", local === 0 || local === 1 ? ".35" : "1");
      } else if (node.hasAttribute("data-stroke")) {
        const ram = node.getAttribute("data-stroke") === "ram" ? node : node.querySelector("rect:nth-child(2)");
        ram?.setAttribute("transform", `translate(0,${Math.sin(local * Math.PI) * 18})`);
      } else node.setAttribute("opacity", Number(node.getAttribute("data-card")) / Number(node.getAttribute("data-card-total")) <= this.progress ? "1" : ".35");
    }
  }
}
