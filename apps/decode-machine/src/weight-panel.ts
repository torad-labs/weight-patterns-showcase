import { capturesInSpan, exportWeightCase, initialCase, readWeightStudy, validateCase, validateSpan, type ActivationSummary, type WeightCase, type WeightStudy } from "./weight-study";
import { appUrl } from "./replay-model";
import { mark } from "./profiling-drawings";
import "./weight-study.css";

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, text = "", cls = "") => { const n = document.createElement(tag); n.textContent = text; n.className = cls; return n; };
const button = (text: string, action: () => void, cls = "") => { const n = el("button", text, cls); n.type = "button"; n.onclick = action; return n; };
const field = (name: string, input: HTMLElement) => { const label = el("label", name); label.append(input); return label; };
const download = (name: string, value: unknown) => { const u = URL.createObjectURL(new Blob([JSON.stringify(value, null, 2)], { type: "application/json" })); const a = el("a"); a.href = u; a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(u), 1000); };
const number = (value: number, max: number, change: (value: number) => void) => { const n = el("input"); n.type = "number"; n.min = "0"; n.max = String(max); n.value = String(value); n.onchange = () => change(Number(n.value)); return n; };
const labelFor = (name: string) => name.replace(/^model\.language_model\.layers\.\d+\./, "");
const matrixLabel = (name: string) => ({ "self_attn.q_proj": "Attention query · q_proj", "self_attn.k_proj": "Attention key · k_proj", "self_attn.v_proj": "Attention value · v_proj", "self_attn.o_proj": "Attention output · o_proj", "mlp.gate_proj": "MLP gate · gate_proj", "mlp.up_proj": "MLP expansion · up_proj", "mlp.down_proj": "MLP output · down_proj" }[labelFor(name)] ?? labelFor(name));
const boundaryLabel = (name: string) => ({ n1: "Normalized layer input", n2: "Normalized MLP input", q_raw: "Query projection", k_raw: "Key projection", v_normed: "Normalized values", attn_out: "Attention result", o_proj: "Attention projection", geglu: "Gated MLP activation", mlp_down: "MLP projection output" }[name] ?? name);

export class WeightPanel {
  private study?: WeightStudy;
  private state?: WeightCase;
  private url?: URL;
  private notice = "Saved artifacts only · no new model execution";
  private readSerial = 0;
  private selectedCapture?: string;
  constructor(private host: HTMLElement, private openMachine: (capture: string, record: string) => void, private ready?: () => void) {
    host.hidden = true;
    const run = new URLSearchParams(location.search).get("run"); if (!run) return;
    const url = new URL("weight-study.json", new URL(run, location.href)); if (url.origin !== location.origin) return;
    this.url = url;
    void fetch(url).then(r => { if (!r.ok) throw Error("Weight study not prepared for this run."); return r.json(); }).then(readWeightStudy).then(s => {
      this.study = s; this.state = initialCase(s);
      try { const saved = localStorage.getItem(`weight-case:${s.fingerprint}`); if (saved) this.state = validateCase(JSON.parse(saved), s); } catch { this.notice = "A previous local selection could not be restored. Saved evidence is unchanged."; }
      this.restoreLink(); this.render(); this.ready?.();
      if (new URLSearchParams(location.hash.slice(1)).has("wreply")) this.host.scrollIntoView({ block: "start" });
    }).catch(e => { if (String(e).includes("not prepared")) return; host.hidden = false; host.textContent = String(e); });
  }
  snapshot() { return this.study && this.state ? exportWeightCase(this.study, this.state) : null; }
  restoreSharedSelection() {
    if (!this.study || !new URLSearchParams(location.hash.slice(1)).has("wreply")) return;
    this.restoreLink(); this.readSerial++; this.render(); this.host.scrollIntoView({ block: "start" });
  }
  focus(captureId: string, layer: number, recordId: string) {
    if (!this.study || !this.state) return;
    const capture = this.study.captures.find(x => x.timingId === captureId);
    const matrix = this.study.matrices.find(x => x.recordId === recordId) ?? this.study.matrices.find(x => x.layer === layer && labelFor(x.name) === labelFor(this.matrix().name));
    if (capture) { this.state.primary = { replyId: capture.replyId, start: capture.outputIndex, end: capture.outputIndex + 1 }; this.selectedCapture = capture.id; }
    if (matrix) { this.state.matrixId = matrix.id; this.state.row = this.state.block = 0; }
    this.save();
  }
  link(href: string) {
    if (!this.state) return href;
    const u = new URL(href), h = new URLSearchParams(u.hash.slice(1)), c = this.state;
    for (const [k, v] of Object.entries({ wreply: c.primary.replyId, wstart: c.primary.start, wend: c.primary.end, wmatrix: c.matrixId, wrow: c.row, wblock: c.block })) h.set(k, String(v));
    u.hash = h.toString(); return u.href;
  }
  private restoreLink() {
    const h = new URLSearchParams(location.hash.slice(1)); if (!h.has("wreply")) return;
    try { this.state = validateCase({ ...this.state!, primary: { replyId: h.get("wreply")!, start: Number(h.get("wstart")), end: Number(h.get("wend")) }, matrixId: h.get("wmatrix")!, row: Number(h.get("wrow")), block: Number(h.get("wblock")) }, this.study!); } catch { this.notice = "Shared weight selection is not valid for this study."; }
  }
  private save(render = true) {
    validateCase(this.state!, this.study!); this.readSerial++;
    try { localStorage.setItem(`weight-case:${this.study!.fingerprint}`, JSON.stringify(this.state)); this.notice = "Case saved on this browser · no model execution"; } catch { this.notice = "Browser storage unavailable. Download the case to keep it."; }
    if (render) this.render(); else {
      this.host.querySelector(".ws-status")!.textContent = this.notice;
      const drafted = !!this.state!.hypothesis && !!this.state!.prediction;
      this.host.querySelector(".ws-ribbon button:last-child small")!.textContent = this.state!.hypothesis ? "Hypothesis drafted" : "Waiting for your hypothesis";
      const readiness = this.host.querySelector<HTMLElement>(".ws-readiness li:nth-child(4)");
      if (readiness) { readiness.dataset.ready = String(drafted); readiness.querySelector("span")!.textContent = drafted ? "●" : "○"; }
    }
  }
  private matrix() { return this.study!.matrices.find(m => m.id === this.state!.matrixId)!; }
  private section(title: string, subtitle: string, cls = "") {
    const s = el("section", "", cls), h = el("div", "", "ws-section-title"); h.append(el("h3", title), el("p", subtitle)); s.append(h); return s;
  }
  private render() {
    const s = this.study!, c = this.state!; this.host.hidden = false; this.host.className = "weight-workbench"; this.host.replaceChildren();
    const heading = el("div", "", "ws-heading"), words = el("div"); words.append(el("p", "WEIGHT INVESTIGATION", "ws-eyebrow"), el("h2", "Follow a thought into the machine"), el("p", "A passage, its internal state, the weights behind it—and a test we can inspect together."));
    const tools = el("div", "", "ws-actions"); tools.append(button("Download this case", () => download("weight-investigation.json", this.snapshot()), "ws-primary"), button("Import case", () => this.importFile()));
    heading.append(words, tools); this.host.append(heading);
    const ribbon = el("nav", "", "ws-ribbon"); ribbon.setAttribute("aria-label", "Investigation relationships");
    for (const [title, detail, target] of [["1  Passage", `${c.primary.end - c.primary.start} selected tokens`, "ws-passage"], ["2  Internal state", `${capturesInSpan(s, c.primary).length} saved positions in span`, "ws-internals"], ["3  Weight block", "32 stored weights + scales", "ws-weights"], ["4  Proposed test", c.hypothesis ? "Hypothesis drafted" : "Waiting for your hypothesis", "ws-experiment"]]) {
      const b = button(title, () => this.host.querySelector(`#${target}`)?.scrollIntoView({ block: "start", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth" })); b.append(el("small", detail)); ribbon.append(b);
    }
    this.host.append(ribbon);
    const grid = el("div", "", "ws-grid"); grid.append(this.passage(), this.internals(), this.weights()); this.host.append(grid, this.experiment(), el("p", this.notice, "ws-status"));
    this.host.querySelector(".ws-status")!.setAttribute("role", "status");
  }
  private passage() {
    const s = this.study!, c = this.state!, r = s.replies.find(r => r.id === c.primary.replyId)!;
    const section = this.section("Choose a passage", "Exact output tokens · positions start at zero"); section.id = "ws-passage";
    const replies = el("select"); replies.setAttribute("aria-label", "Investigated reply");
    for (const r of s.replies) replies.add(new Option(`Reply ${r.ordinal} · ${r.tokenIds.length.toLocaleString()} tokens`, r.id)); replies.value = r.id;
    replies.onchange = () => { const r = s.replies.find(r => r.id === replies.value)!; c.primary = { replyId: r.id, start: 0, end: Math.min(32, r.tokenIds.length) }; this.selectedCapture = undefined; this.save(); }; section.append(replies);
    const rail = el("div", "", "ws-token-rail"); rail.setAttribute("aria-label", "Reply token windows");
    const width = Math.max(8, Math.ceil(r.tokenIds.length / 64));
    for (let i = 0; i < r.tokenIds.length; i += width) {
      const start = i, end = Math.min(r.tokenIds.length, i + width), b = button("", () => { c.primary = { replyId: r.id, start, end }; this.selectedCapture = undefined; this.save(); });
      b.title = `Tokens ${start}–${end - 1}`; b.setAttribute("aria-label", b.title); b.setAttribute("aria-pressed", String(start < c.primary.end && end > c.primary.start));
      if (s.captures.some(x => x.replyId === r.id && x.outputIndex >= start && x.outputIndex < end)) b.classList.add("has-capture"); rail.append(b);
    }
    section.append(rail, el("p", "Teal: selected range. Amber dot: saved internal state.", "ws-caption"));
    const range = el("div", "", "ws-fields");
    const update = (key: "start" | "end", value: number) => { const span = { ...c.primary, [key]: value }; try { validateSpan(span, s.replies); c.primary = span; this.selectedCapture = undefined; this.save(); } catch (e) { this.notice = String(e); this.render(); } };
    range.append(field("First token", number(c.primary.start, r.tokenIds.length - 1, x => update("start", x))), field("End (exclusive)", number(c.primary.end, r.tokenIds.length, x => update("end", x)))); section.append(range);
    const tokens = el("div", "", "ws-token-pieces");
    r.tokenIds.slice(c.primary.start, Math.min(c.primary.end, c.primary.start + 96)).forEach((id, i) => { const t = el("span", s.vocabulary[id] ?? `#${id}`); t.title = `Output index ${c.primary.start + i} · token ID ${id}`; tokens.append(t); }); section.append(tokens);
    section.append(el("p", `${c.primary.end - c.primary.start > 96 ? "First 96 pieces shown. " : ""}Verbatim vocabulary pieces; ▁ marks a space. These are not an inferred text alignment.`, "ws-caption"));
    const original = el("details"); original.append(el("summary", "Read the original reply"), el("p", r.text, "ws-original")); section.append(original);
    section.append(el("p", "Reply opening", "ws-caption"), el("p", r.text.slice(0, 650) + (r.text.length > 650 ? "…" : ""), "ws-excerpt"));
    if (r.promptTail) { const prompt = el("details"); prompt.append(el("summary", `Input context · ${r.promptTokens.toLocaleString()} tokens`), el("p", "Last 800 characters of the saved request. The export identifies the complete request by path and hash.", "ws-caption"), el("p", r.promptTail, "ws-original")); section.append(prompt); }
    section.append(button("Pin as comparison passage", () => { c.comparison = { ...c.primary }; this.save(); }));
    if (c.comparison) {
      const other = s.replies.find(r => r.id === c.comparison!.replyId)!;
      const comparison = el("div", "", "ws-comparison"); comparison.append(el("strong", `Comparison · Reply ${other.ordinal}`), el("p", `Tokens ${c.comparison.start}–${c.comparison.end - 1}. Analyst-selected; no success or failure label inferred.`), button("Clear comparison", () => { c.comparison = null; this.save(); })); section.append(comparison);
    }
    return section;
  }
  private internals() {
    const s = this.study!, c = this.state!, m = this.matrix();
    const section = this.section("Find the internal state", "Recorded positions remain separate from production timing"); section.id = "ws-internals";
    const layers = el("div", "", "ws-layers"); layers.setAttribute("aria-label", "Model layers");
    const layerIds = [...new Set(s.matrices.flatMap(m => m.layer === null ? [] : [m.layer]))].sort((a, b) => a - b);
    for (const layer of layerIds) { const b = button(String(layer + 1), () => { const ending = labelFor(m.name), next = s.matrices.find(x => x.layer === layer && labelFor(x.name) === ending) ?? s.matrices.find(x => x.layer === layer)!; c.matrixId = next.id; c.row = c.block = 0; this.save(); }); b.setAttribute("aria-label", `Inspect layer ${layer + 1}`); b.setAttribute("aria-pressed", String(layer === m.layer)); layers.append(b); }
    section.append(layers, el("p", m.layer === null ? "Global checkpoint matrices" : `Layer ${m.layer + 1} · source index ${m.layer}`, "ws-caption"));
    if (s.matrices.some(m => m.layer === null)) section.append(button("Global matrices", () => { c.matrixId = s.matrices.find(m => m.layer === null)!.id; c.row = c.block = 0; this.save(); }));
    const captures = capturesInSpan(s, c.primary);
    const selected = captures.find(x => x.id === this.selectedCapture) ?? captures[0];
    if (!selected) {
      const empty = el("div", "", "ws-empty"); empty.append(el("strong", "No internal capture in this passage"), el("p", "The words are recorded. Token-aligned internal states for this range still need a capture.")); section.append(empty);
    } else {
      const pick = el("select"); pick.setAttribute("aria-label", "Saved internal position"); for (const x of captures) pick.add(new Option(`Position ${x.position} · predicts output ${x.outputIndex}`, x.id)); pick.value = selected.id; pick.onchange = () => { this.selectedCapture = pick.value; this.render(); }; section.append(pick);
      section.append(el("p", selected.inputMatches && selected.outputMatches ? "Input and output token IDs match. This is a separate diagnostic execution." : "Token IDs disagree with production. Inspect as a different trajectory.", selected.inputMatches && selected.outputMatches ? "ws-caption" : "ws-alert"));
      const values = s.activations.filter(a => a.captureId === selected.id && a.layer === m.layer && [m.input, m.output].includes(a.stage));
      const inputWidth = values.find(a => a.stage === m.input)?.activeWidth;
      if (inputWidth !== undefined && inputWidth !== m.columns) section.append(el("p", `The saved input has ${inputWidth.toLocaleString()} elements; this matrix has ${m.columns.toLocaleString()} columns. Their complete coordinate relationship was not recorded.`, "ws-alert"));
      const comparisonCapture = c.comparison ? capturesInSpan(s, c.comparison).find(x => x.inputMatches && x.outputMatches) : undefined;
      for (const a of values) { const other = comparisonCapture && s.activations.find(x => x.captureId === comparisonCapture.id && x.layer === a.layer && x.stage === a.stage); section.append(this.distribution(a, other)); }
      if (values.length) { const guide = el("details"); guide.append(el("summary", "How to read the measurements"), el("p", "Bars count encoded 32-element blocks, from zero pairs through exponent bin 15. Comparison bars share a scale. These distributions do not measure neuron importance or causal influence.")); section.append(guide); }
      if (!values.length) section.append(el("p", "This matrix has no mapped recorded boundary. Its storage remains inspectable.", "ws-empty"));
      if (selected.timingId && m.recordId) section.append(button("Open this operation in the machine ↗", () => this.openMachine(selected.timingId!, m.recordId!)));
      else section.append(el("p", "Internal timing unavailable for this selection; no animation substituted.", "ws-caption"));
    }
    const available = el("details"); available.append(el("summary", `Browse all ${s.captures.length} saved positions`));
    const chips = el("div", "", "ws-actions"); for (const cap of s.captures) { const r = s.replies.find(r => r.id === cap.replyId)!; const b = button(`Reply ${r.ordinal} · ${cap.position}`, () => { c.primary = { replyId: r.id, start: cap.outputIndex, end: cap.outputIndex + 1 }; this.selectedCapture = cap.id; this.save(); }); if (cap.outputIndex < 0 || cap.outputIndex >= r.tokenIds.length) b.disabled = true; chips.append(b); } available.append(chips); section.append(available);
    return section;
  }
  private distribution(a: ActivationSummary, other?: ActivationSummary) {
    const box = el("div", "", "ws-distribution"); box.append(el("strong", `${boundaryLabel(a.stage)} · ${a.activeWidth.toLocaleString()} elements`));
    const svg = mark("svg", { viewBox: "0 0 390 114", role: "img", "aria-label": `${a.stage}: stored code exponent distribution${other ? ", with comparison passage" : ""}` });
    const populations = [a, ...(other ? [other] : [])], max = Math.max(...populations.flatMap(x => x.counts.map(n => n / (x.activeWidth / 32))), .001);
    svg.append(mark("line", {x1:40,x2:380,y1:82,y2:82,stroke:"#a5b7a5"}),mark("text",{x:2,y:14,"font-size":15,fill:"#53695e"},`${Math.round(max*100)}%`));
    for (const [x,t,anchor] of [[47,"zero","end"],[67,"0","middle"],[207,"7","middle"],[367,"15","middle"]] as const) svg.append(mark("text",{x,y:104,"font-size":15,"text-anchor":anchor,fill:"#53695e"},t));
    populations.forEach((p, j) => p.counts.forEach((n, i) => { const h = n / (p.activeWidth / 32) / max * 76; const b = mark("rect", { x: 40 + i * 20 + j * 8, y: 82 - h, width: other ? 7 : 14, height: h, fill: j ? "#ac772f" : "#397b70" }); b.append(mark("title", {}, `${i === 0 ? "Zero pair" : `Exponent bin ${i - 1}`}: ${n} blocks`)); svg.append(b); })); box.append(svg);
    box.append(el("p", `${other ? "Teal: selected · amber: comparison · " : "Stored block fractions · "}${a.nanBlocks} invalid blocks`, "ws-caption"));
    const raw = el("div", "", "ws-raw-activation"), read = button("Read first recorded block", () => { void this.readActivation(a, raw); }); box.append(read, raw); return box;
  }
  private async readActivation(a: ActivationSummary, target: HTMLElement) {
    target.textContent = "Verifying recorded planes…";
    try {
      const planes = await Promise.all([a.codePath, a.scalePath].map(async p => { const u = new URL(p, this.url!); if (u.origin !== location.origin) throw Error("Use local recorded planes."); const r = await fetch(u); if (!r.ok) throw Error("Recorded plane unavailable."); return r.arrayBuffer(); }));
      const digests = await Promise.all(planes.map(async p => [...new Uint8Array(await crypto.subtle.digest("SHA-256", p))].map(x => x.toString(16).padStart(2, "0")).join("")));
      if (digests[0] !== a.codeSha256 || digests[1] !== a.scaleSha256) throw Error("Recorded activation hash mismatch.");
      const codes = new Uint8Array(planes[0], a.codeOffset, 32), scale = new Uint8Array(planes[1])[a.scaleOffset];
      target.replaceChildren(el("p", `Stored FP8 code bytes · elements 0–31 · E8M0 scale byte ${scale}`, "ws-caption"), this.bytes([...codes], false));
    } catch (e) { target.textContent = String(e); }
  }
  private bytes(bytes: number[], weight: boolean) {
    const grid = el("div", "", `ws-bytes${weight ? " weight" : ""}`);
    bytes.forEach((b, i) => { const cell = el("span", b.toString(16).padStart(2, "0")); cell.title = `Byte ${i}: ${b}`; grid.append(cell); }); return grid;
  }
  private weights() {
    const s = this.study!, c = this.state!, m = this.matrix();
    const section = this.section("Inspect the weight block", "Checkpoint bytes · architectural relationships"); section.id = "ws-weights";
    const matrices = el("select"); matrices.setAttribute("aria-label", "Weight matrix");
    for (const x of s.matrices.filter(x => x.layer === m.layer)) matrices.add(new Option(matrixLabel(x.name), x.id)); matrices.value = m.id;
    matrices.onchange = () => { c.matrixId = matrices.value; c.row = c.block = 0; this.save(); }; section.append(matrices);
    if (m.sharedFrom) { const owner = s.matrices.find(x => x.id === m.sharedFrom)!; const shared = el("div", "", "ws-alert"); shared.append(el("p", `This schedule reuses K/V from layer ${owner.layer! + 1}. These local weights are stored, but their projection is not scheduled here.`), button("Inspect the K/V owner's weights", () => { c.matrixId = owner.id; c.row = c.block = 0; this.save(); })); section.append(shared); }
    const flow = el("div", "", "ws-matrix-flow"); flow.append(el("span", m.input ? boundaryLabel(m.input) : "Input unmapped", "ws-boundary"), el("span", "→"));
    const svg = mark("svg", { viewBox: "0 0 200 130", role: "img", "aria-label": `${m.rows} output rows by ${m.columns} input columns; selected row ${c.row}, block ${c.block}` });
    svg.append(mark("rect", { x: 12, y: 8, width: 176, height: 112, rx: 4, fill: "#e8c685", stroke: "#95703b" }));
    for (let i = 1; i < 8; i++) svg.append(mark("line", { x1: 12, x2: 188, y1: 8 + i * 14, y2: 8 + i * 14, stroke: "#c6a26c" }));
    for (let i = 1; i < 12; i++) svg.append(mark("line", { y1: 8, y2: 120, x1: 12 + i * 176 / 12, x2: 12 + i * 176 / 12, stroke: "#c6a26c" }));
    svg.append(mark("rect", { x: 12 + c.block * 32 / m.columns * 168, y: 8 + c.row / m.rows * 106, width: 8, height: 6, fill: "#263e3d" })); flow.append(svg, el("span", "→"), el("span", m.output ? boundaryLabel(m.output) : "Output unmapped", "ws-boundary")); section.append(flow);
    section.append(el("strong", `${m.rows.toLocaleString()} rows × ${m.columns.toLocaleString()} columns`, "ws-shape"), el("p", m.name, "ws-path"));
    const fields = el("div", "", "ws-fields"); fields.append(field("Output row", number(c.row, m.rows - 1, x => { const old = c.row; c.row = x; try { this.save(); } catch { c.row = old; this.render(); } })), field("32-column block", number(c.block, Math.floor(m.columns / 32) - 1, x => { const old = c.block; c.block = x; try { this.save(); } catch { c.block = old; this.render(); } }))); section.append(fields);
    section.append(el("p", `Selected columns ${c.block * 32}–${c.block * 32 + 31}. ${m.basis}`, "ws-caption"));
    const raw = el("div", "", "ws-weight-readout"); raw.append(el("p", "Read a block to inspect its actual packed bytes and scale companions.", "ws-empty"));
    const read = button("Read this weight block", () => { void this.readWeight(raw, read); }, "ws-primary"); section.append(read, raw);
    const inventory = el("details"); inventory.append(el("summary", "What is connected?"), el("p", `${s.inventory.storedTensors.toLocaleString()} storage tensors inventoried from the checkpoint. ${s.inventory.routedMatrices} matrices have an expected operation route; ${s.matrices.filter(m => m.sharedFrom).length} follow a K/V owner instead; ${s.matrices.filter(m => !m.recordId && !m.sharedFrom).length} remain unmapped. The export retains all tensors, including auxiliary and non-text tensors.`), el("p", "Matrix arrows show architecture, not measured memory reads. Normalization or a combined gate operation can lie between a projection and its recorded boundary. A raw weight block is not an importance score.")); section.append(inventory); return section;
  }
  private async readWeight(target: HTMLElement, read: HTMLButtonElement) {
    const serial = ++this.readSerial, c = this.state!; read.disabled = true; target.textContent = "Checking the checkpoint hash, then reading this block…";
    try {
      const u = new URL(appUrl("__weight-block"), location.href); for (const [k, v] of Object.entries({ study: this.url!.pathname, matrix: c.matrixId, row: c.row, block: c.block })) u.searchParams.set(k, String(v));
      const r = await fetch(u), data = await r.json(); if (!r.ok) throw Error(data.error ?? "Local weight reader unavailable. Open the Vite workbench to read checkpoint blocks."); if (serial !== this.readSerial) return;
      target.replaceChildren(); for (const chunk of data.chunks) { target.append(el("p", `${chunk.name.split(".").at(-1)} · file byte ${chunk.fileOffset.toLocaleString()}`, "ws-caption"), this.bytes(chunk.bytes, true)); }
      target.append(el("p", `Verified checkpoint ${data.checkpointSha256.slice(0, 16)}… · 16 code bytes encode 32 weights. Hexadecimal bytes; no value conversion.`, "ws-caption"));
    } catch (e) { if (serial === this.readSerial) target.textContent = String(e); } finally { read.disabled = false; }
  }
  private experiment() {
    const s = this.study!, c = this.state!, section = this.section("Prepare a test of your hypothesis", "Draft only · the workbench cannot start inference or alter a checkpoint", "ws-experiment"); section.id = "ws-experiment";
    const grid = el("div", "", "ws-proposal-grid"), form = el("div", "", "ws-proposal-form");
    for (const [key, label, placeholder] of [["hypothesis", "Your hypothesis", "What relationship do you think explains the behavior?"], ["change", "Proposed intervention", "Which weight slice or component would change, and how?"], ["prediction", "Expected observation", "What should change if the explanation is right?"], ["refutation", "What would disprove it?", "Which outcome would make us reject the explanation?"], ["strengths", "Intervention strengths and controls", "Specify the unchanged baseline, candidate strengths and control intervention."]] as const) {
      const input = key === "strengths" ? el("input") : el("textarea"); input.value = c[key]; input.placeholder = placeholder; input.oninput = () => { c[key] = input.value; this.save(false); }; form.append(field(label, input));
    }
    const comparison = el("div", "", "ws-result");
    comparison.append(el("h3", "What a capture of this proposal needs"));
    const readiness = el("ul", "", "ws-readiness"); for (const [label, available] of [["Saved words and exact token IDs", true], ["Checkpoint tensor inventory and block reader", true], ["Internal captures for the selected passage", capturesInSpan(s, c.primary).length > 0], ["Hypothesis and expected result", !!c.hypothesis && !!c.prediction]] as const) { const li = el("li", label); li.dataset.ready = String(available); li.prepend(el("span", available ? "●" : "○")); readiness.append(li); } comparison.append(readiness);
    const actions = el("div", "", "ws-actions"); actions.append(button("Download capture request", () => download("weight-capture-request.json", exportWeightCase(s, c).captureRequest))); comparison.append(actions); grid.append(form, comparison); section.append(grid); return section;
  }
  private importFile() {
    const input = el("input"); input.type = "file"; input.accept = ".json,application/json";
    input.onchange = async () => {
      try {
        const file = input.files?.[0]; if (!file) return; if (file.size > 32 * 1024 * 1024) throw Error("Choose an evidence file smaller than 32 MiB.");
        const value = JSON.parse(await file.text());
        if (value.schema !== "qgre.weight-case.v1" || value.studyFingerprint !== this.study!.fingerprint) throw Error("Case belongs to different evidence.");
        this.state = validateCase(value.case, this.study!); this.save(false);
        this.notice = "Case restored against the same evidence fingerprint."; this.render();
      } catch (e) { this.notice = String(e); this.render(); }
    }; input.click();
  }
}
