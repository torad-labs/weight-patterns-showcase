import { readRunEvidence, turnMeanMs, type RunEvidence, type RunTurn } from "./run-evidence";
import { mark } from "./profiling-drawings";
import "./run.css";

const el = (tag: string, text = "", cls = "") => { const n = document.createElement(tag); n.textContent = text; n.className = cls; return n; };
const note = (title: string, body: string) => { const n = el("details", "", "run-note"); n.append(el("summary", title), el("p", body)); return n; };
const sizeChartLabels = (svg: SVGElement) => {
  const observer = new ResizeObserver(() => {
    const width=svg.getBoundingClientRect().width;
    if(width) svg.style.setProperty("--chart-label",`${14*900/width}px`);
  });
  observer.observe(svg);
};
export class RunPanel {
  private data?: RunEvidence;
  constructor(private host: HTMLElement, private selectCapture: (id: string, profiling: string, record?: string) => void, loaded?: (run: RunEvidence) => void) {
    this.host.hidden = true;
    const source = new URLSearchParams(location.search).get("run");
    if (!source) return;
    const url = new URL(source, location.href);
    if (url.origin !== location.origin) return;
    void fetch(url).then(r => { if (!r.ok) throw Error("Run evidence could not load."); return r.json(); }).then(readRunEvidence).then(r => { this.data = r; this.render(); loaded?.(r); }).catch(e => { this.host.hidden = false; this.host.textContent = String(e); });
  }
  snapshot() { return this.data ?? null; }
  private render() {
    const r = this.data!; this.host.hidden = false; this.host.className = "run-evidence";
    const head = el("div", "", "run-heading"), name = el("div");
    name.append(el("p", "RECORDED CONVERSATION", "run-eyebrow"), el("h2", r.label), el("p", `${r.generatedTokens.toLocaleString()} / ${r.targetTokens.toLocaleString()} new tokens · ${r.turns.length} turns · earlier replies retained in context`));
    const download = el("button", "Download run evidence"); download.onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], { type: "application/json" })); const a = document.createElement("a"); a.href = url; a.download = `${r.id}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }; head.append(name, download); this.host.append(head);
    const progress = el("div", "", "run-progress"); progress.setAttribute("role", "progressbar"); progress.setAttribute("aria-label", "Generated tokens"); progress.setAttribute("aria-valuenow", String(r.generatedTokens)); progress.setAttribute("aria-valuemax", String(r.targetTokens)); const progressFill = el("i"); progressFill.style.width = `${100 * Math.min(1, r.generatedTokens / r.targetTokens)}%`; progress.append(progressFill); this.host.append(progress);
    const grid = el("div", "", "run-grid"), turns = el("section"), hardware = el("section");
    hardware.className = "run-hardware";
    turns.append(el("h3", "More history. Slower tokens?"), el("p", "Reply width: new tokens. Height: mean GPU decode time."));
    const detail = el("div", "", "run-selection");
    const svg = mark("svg", { viewBox: "0 0 900 330", role: "group", "aria-label": "Mean GPU decode time by conversation turn" });
    const max = Math.max(1, ...r.turns.map(t => turnMeanMs(t) ?? 0)), target = Math.max(1, r.generatedTokens);
    svg.append(mark("line", { x1: 58, y1: 264, x2: 874, y2: 264, class: "run-axis" }), mark("text", { x: 58, y: 30 }, `${max.toFixed(1)} ms / token`), mark("text", { x: 58, y: 302 }, "0 new tokens"), mark("text", { x: 874, y: 302, "text-anchor": "end" }, `${target.toLocaleString()} new tokens`));
    const show = (t: RunTurn) => {
      const ms = turnMeanMs(t), fraction = t.promptTokens ? t.cachedTokens / t.promptTokens : 0;
      detail.replaceChildren(el("strong", `Reply ${t.ordinal}`));
      const stats = el("div", "", "run-stats");
      for (const [value, label] of [[t.generatedTokens.toLocaleString(), "new tokens"], [ms?.toFixed(2) ?? "?", "ms / token"], [(t.hostNs / 1e9).toFixed(2), "request seconds"]]) { const stat = el("div"); stat.append(el("strong", value), el("span", label)); stats.append(stat); }
      const cache = el("div", "", "run-cache"); cache.append(el("span", `${t.cachedTokens.toLocaleString()} of ${t.promptTokens.toLocaleString()} input tokens reused`)); const rail = el("div", "", "run-rail"), fill = el("i"); fill.style.width = `${100 * fraction}%`; rail.append(fill); cache.append(rail);
      detail.append(stats, cache);
      if(t.prefillNs !== undefined && t.gpuNs !== null) {
        const times=el("div","","run-time-split"), rail=el("div","","run-rail");
        for(const [name,ns,color] of [["Prefill",t.prefillNs,"#bc9f61"],["Decode",t.gpuNs,"#536f65"],["Other",Math.max(0,t.hostNs-t.prefillNs-t.gpuNs),"#b9c4b7"]] as const) {
          const segment=el("i"); segment.style.width=`${100*ns/t.hostNs}%`; segment.style.background=color; segment.title=`${name}: ${(ns/1e9).toFixed(2)} s`; rail.append(segment);
        }
        times.append(rail,el("span",`Prefill ${(t.prefillNs/1e9).toFixed(2)} s · decode ${(t.gpuNs/1e9).toFixed(2)} s · request ${(t.hostNs/1e9).toFixed(2)} s`)); detail.append(times);
      }
      for (const bar of svg.querySelectorAll("[data-turn]")) { const selected = bar.getAttribute("data-turn") === t.id; bar.classList.toggle("selected", selected); bar.setAttribute("aria-pressed", String(selected)); }
    };
    for (const t of r.turns) {
      const value = turnMeanMs(t); if (value === null) continue;
      const x = 58 + (t.cumulativeTokens - t.generatedTokens) / target * 816, width = Math.max(2, t.generatedTokens / target * 816 - 3), h = value / max * 210;
      const bar = mark("rect", { x, y: 264 - h, width, height: h, rx: 3, class: "run-reply", tabindex: 0, role: "button", "data-turn":t.id, "aria-label": `Reply ${t.ordinal}: ${value.toFixed(2)} milliseconds per token` });
      bar.append(mark("title", {}, `Reply ${t.ordinal} · ${value.toFixed(2)} ms/token`)); bar.addEventListener("click", () => show(t)); bar.addEventListener("focus", () => show(t)); bar.addEventListener("keydown", event => { if ((event as KeyboardEvent).key === "Enter") show(t); });
      svg.append(bar);
      if (width > 22) svg.append(mark("text", { x: x + width / 2, y: 254 - h, "text-anchor": "middle", class: "run-turn-number" }, String(t.ordinal)));
    }
    if (r.turns.length) show(r.turns.at(-1)!);
    sizeChartLabels(svg);
    turns.append(svg, detail, note("Timing scope", "Reply averages from normal production launches. Individual token timestamps inside a persistent launch were not recorded. The first output token belongs to prefill and is excluded from the decode-time denominator."));
    hardware.append(el("h3", "Busy versus productive"), el("p", "Separate passes. Shared 0–100% activity scale."));
    const selected = r.metrics.filter(m => /SMs Active|SM Issue|Tensor Active|DRAM Read Bandwidth/.test(m.name));
    for (const name of new Set(selected.map(m=>m.name))) {
      const row = el("div", "", "run-meter"); row.append(el("strong",name.replace(" [Throughput %]", "")));
      for(const m of selected.filter(m=>m.name===name)) {
        const label = el("div", "", "run-meter-label"); label.append(el("span", m.sourceId==='nsys:first-reply'?'First reply':'Later diagnostic'), el("strong", `${m.mean.toFixed(1)}%`));
        const rail = el("div", "", "run-rail"), fill = el("i"); fill.style.width = `${Math.max(0, Math.min(100, m.mean))}%`; rail.append(fill); row.append(label, rail);
      }
      hardware.append(row);
    }
    hardware.append(note("Measurement scope", "Mean activity during each pass’s final decode launch. The first reply uses production execution; the later slice uses diagnostic instrumentation after cold prefill. These are separate executions, not a controlled performance comparison. Activity alone cannot distinguish waiting, instruction overhead, or useful arithmetic."));
    for (const pass of r.counterPasses ?? []) {
      const counter = (name: string) => { const m=pass.metrics.find(m=>m.name===name); return m && Number.isFinite(Number(m.value)) ? Number(m.value) : null; };
      const issue=counter("smsp__issue_active.avg.pct_of_peak_sustained_active");
      const registers=counter("launch__occupancy_limit_registers"), shared=counter("launch__occupancy_limit_shared_mem"), warps=counter("launch__occupancy_limit_warps");
      if(issue===null || registers===null || shared===null || warps===null) continue;
      const story=el("details","","run-counter-story"); story.append(el("summary","Inside the early scheduler"),el("p","Nsight Compute · separate early-token replay"));
      const issueLabel=el("div","","run-meter-label"); issueLabel.append(el("span","Instruction issue"),el("strong",`${issue.toFixed(1)}% of peak while active`));
      const issueRail=el("div","","run-rail"), issued=el("i"); issued.style.width=`${Math.max(0,Math.min(100,issue))}%`; issueRail.append(issued); story.append(issueLabel,issueRail);
      const ceiling=Math.max(1,registers,shared,warps);
      story.append(el("p","How many blocks could share one SM? Each resource sets its own limit."));
      for(const [label,value] of [["Registers",registers],["Shared memory",shared],["Warp capacity",warps]] as const) {
        const row=el("div","","run-meter"), caption=el("div","","run-meter-label"), rail=el("div","","run-rail"), fill=el("i");
        caption.append(el("span",label),el("strong",`${value} ${value===1?"block":"blocks"}`)); fill.style.width=`${100*value/ceiling}%`; rail.append(fill); row.append(caption,rail); story.append(row);
      }
      story.append(el("p",`The smallest limit wins: ${Math.min(registers,shared,warps)} block(s) per SM. Reducing only one resource may leave the other limit in place.`)); hardware.append(story);
    }
    grid.append(turns, hardware); this.host.append(grid);
    if (r.captures.length) {
      const comparison = el("section", "", "run-attention"); comparison.append(el("h3", "Which readers grow with history?"), el("p", "Full history (amber) versus sliding window (teal)."));
      const chart = mark("svg", {viewBox:"0 0 900 330",role:"group","aria-label":"Attention interval tails as context grows"});
      const ceiling = Math.max(1,...r.captures.map(c=>c.fullTailNs)), right = Math.max(...r.captures.map(c=>c.position));
      const x = (p:number)=>58+p/right*790, y=(ns:number)=>264-ns/ceiling*210;
      chart.append(mark("line",{x1:58,y1:264,x2:848,y2:264,class:"run-axis"}),mark("text",{x:58,y:30},`${(ceiling/1000).toFixed(0)} µs tail`),mark("text",{x:58,y:302},"Context position →"));
      for(const [field,color,record] of [["fullTailNs","#bc9f61","l4:mixer:2"],["slidingTailNs","#628b7a","l0:mixer:2"]] as const) {
        for(const source of new Set(r.captures.map(c=>c.sourceId))) chart.append(mark("polyline",{points:r.captures.filter(c=>c.sourceId===source).map(c=>`${x(c.position)},${y(c[field])}`).join(" "),fill:"none",stroke:color,"stroke-width":3,"stroke-dasharray":"5 4"}));
        for(const c of r.captures) { const dot=mark("circle",{cx:x(c.position),cy:y(c[field]),r:6,fill:color,tabindex:0,role:"button","aria-label":`${field==="fullTailNs"?"Full":"Sliding"} attention at position ${c.position}: ${(c[field]/1000).toFixed(1)} microseconds; inspect`}); dot.append(mark("title",{},`Position ${c.position}: ${(c[field]/1000).toFixed(1)} µs`)); const select=()=>this.selectCapture(c.id,c.profiling,record); dot.addEventListener("click",select); dot.addEventListener("keydown",e=>{if((e as KeyboardEvent).key==="Enter")select();}); chart.append(dot); }
      }
      sizeChartLabels(chart);
      chart.append(mark("text",{x:848,y:302,"text-anchor":"end"},String(right))); comparison.append(chart,el("p", "Select a dot to open its recorded attention machinery. Lines connect only captures from the same pass."),note("What this interval means","Tail = last recorded entry → last recorded exit, averaged over layers of the same kind. Dots are diagnostic captures. The later point follows a cold prefill of a saved prompt. This is a timing relationship, not a complete critical path or a recoverable-time estimate.")); grid.insertBefore(comparison,hardware);
    }
    const captures = el("div", "", "run-captures"); captures.append(el("strong", "Open a recorded diagnostic slice"));
    const longest = Math.max(1,...r.captures.map(c=>c.durationNs));
    for (const c of r.captures) { const b = el("button", `${c.sourceId==='taps:later-context'?'Later prompt · ':''}Position ${c.position} · ${(c.durationNs / 1e6).toFixed(2)} ms`); const duration = el("i"); duration.style.width = `${100 * c.durationNs / longest}%`; b.append(duration); b.onclick = () => this.selectCapture(c.id, c.profiling); captures.append(b); } this.host.append(captures);
    for (const c of r.unavailableCaptures ?? []) {
      const unavailable=note(`◌ Position ${c.position.toLocaleString()} · internal timing unavailable`,c.reason);
      unavailable.className="run-unavailable"; this.host.append(unavailable);
    }
    const limits = el("details", "", "run-limits"); limits.append(el("summary", "What this recording establishes—and what is still missing"));
    for (const o of r.observations) limits.append(el("p", o.claim), el("p", `Next test: ${o.next}`, "run-note"));
    for (const g of r.gaps) limits.append(el("p", `${g.area}: ${g.missing}`));
    for (const s of r.sources) for (const warning of s.warnings) limits.append(el("p", `${s.method}: ${warning}`, "run-note"));
    this.host.append(limits);
  }
}
