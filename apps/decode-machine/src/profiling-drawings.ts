import { fmtNs } from "./replay-model";
import { lifetime, readiness, sampleGroups, usefulFraction, type LensId, type ProfilingBundle } from "./profiling-model";

const NS = "http://www.w3.org/2000/svg";
export function mark(tag: string, attrs: Record<string, string | number> = {}, value?: string) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (value !== undefined) n.textContent = value;
  return n;
}
export type Drawing = { svg: SVGElement; notes: string[]; selected: string | null };
export function drawEvidence(lens: LensId, b: ProfilingBundle, selection: string | null, choose: (id: string) => void, compact = false): Drawing {
  const W = compact ? 400 : 920, H = compact ? 470 : 430;
  const svg = mark("svg", { viewBox: `0 0 ${W} ${H}`, class: "profile-drawing", role: "group", "aria-label": `${lens} relationships` }) as SVGElement;
  const defs = mark("defs");
  for (const [name, light, base, dark] of [["teal", "#6d9e91", "#45877b", "#2a675d"], ["amber", "#e1c181", "#d2aa5d", "#a2762c"], ["case", "#e7ece3", "#d6ded5", "#a8bcb3"]]) {
    const shade = mark("linearGradient", { id: `profile-${name}-shade`, x1: "0", y1: "0", x2: "0", y2: "1" });
    for (const [offset, color] of [["0", light], [".15", light], [".15", base], [".85", base], [".85", dark], ["1", dark]]) shade.append(mark("stop", { offset, "stop-color": color }));
    defs.append(shade);
  }
  svg.append(defs);
  const notes: string[] = [];
  const focusRows = <T extends { id: string }>(rows: T[], count: number) => {
    const picked = rows.find(r => r.id === selection);
    const result = picked ? [picked, ...rows.filter(r => r !== picked)].slice(0, count) : rows.slice(0, count);
    if (rows.length > count) notes.push(`Showing ${count} of ${rows.length} pieces. Select another piece to bring it into the drawing.`);
    return result;
  };
  const text = (x: number, y: number, value: string, cls = "", anchor = "start") => svg.append(mark("text", { x, y, class: cls, "text-anchor": anchor }, value));
  const line = (x1: number, y1: number, x2: number, y2: number, cls = "rail") => svg.append(mark("line", { x1, y1, x2, y2, class: cls }));
  const box = (x: number, y: number, w: number, h: number, cls: string, rx = 6) => mark("rect", { x, y, width: Math.max(0, w), height: h, rx, class: cls });
  const dot = (x: number, y: number, r: number, cls: string) => mark("circle", { cx: x, cy: y, r, class: cls });
  const keyboardEntities = new Set<string>();
  const interactive = (node: Element, id: string, label: string) => {
    node.setAttribute("tabindex", keyboardEntities.has(id) ? "-1" : "0"); keyboardEntities.add(id);
    node.setAttribute("role", "button"); node.setAttribute("aria-label", label); node.setAttribute("data-evidence-id", id);
    if (selection === id) node.classList.add("chosen");
    node.append(mark("title", {}, label)); node.addEventListener("click", () => choose(id));
    node.addEventListener("keydown", e => { const k = e as KeyboardEvent; if (k.key === "Enter" || k.key === " ") { k.preventDefault(); choose(id); } }); svg.append(node);
  };
  const times = [...b.tasks.flatMap(t => [t.startNs, t.endNs]), ...b.buffers.flatMap(t => [t.requestNs, t.releaseNs]), ...b.samples.flatMap(t => [t.startNs, t.endNs]), ...b.instructions.flatMap(t => [t.startNs, t.endNs])];
  const lo = times.length ? Math.min(...times) : 0, hi = times.length ? Math.max(...times) : 1;
  const x = (t: number, left = 34, right = W - 34) => left + (t - lo) / (hi - lo || 1) * (right - left);
  const axis = (y: number, left = 34) => { line(left, y, W - 34, y); text(left, y + 27, "+0", "axis"); text(W - 34, y + 27, fmtNs(hi - lo), "axis", "end"); };
  const cursor = (y1: number, y2: number, left = 34) => { const c = mark("line", { x1: left, x2: left, y1, y2, class: "profile-time-line", "data-low": lo, "data-high": hi, "data-left": left, "data-right": W - 34 }); svg.append(c); };

  if (lens === "dependencies") {
    const selectedConsumer = b.dependencies.find(d => d.producerId === selection)?.consumerId;
    const target = b.tasks.find(t => t.id === selection && b.dependencies.some(d => d.consumerId === t.id)) ?? b.tasks.find(t => t.id === selectedConsumer) ?? b.tasks.find(t => b.dependencies.some(d => d.consumerId === t.id));
    const state = target && readiness(b, target.id);
    if (target && state) {
      const endX = W - (compact ? 74 : 142), startX = compact ? 78 : 160;
      const inputs = [...state.inputs].sort((a, z) => Number(z.producerId === selection) - Number(a.producerId === selection) || Number(z.last) - Number(a.last)).slice(0, 3);
      if (state.inputs.length > 3) notes.push(`Drawing 3 of ${state.inputs.length} prerequisites, including the selected and last inputs; readiness uses all recorded inputs.`);
      inputs.forEach((dep, i) => {
        const producer = b.tasks.find(t => t.id === dep.producerId)!; const y = 65 + i * 100;
        const route = `M${startX + 43},${y} H${W / 2} V188 H${endX - 40}`;
        svg.append(mark("path", { d: route, class: dep.last ? "connection last-input" : "connection" }));
        if (b.origin === "illustrated") { const mover = dot(0, 0, 7, dep.last ? "amber-part" : "teal-part"); mover.setAttribute("data-route", route); mover.setAttribute("data-from", String((producer.startNs - lo) / (hi - lo))); mover.setAttribute("data-to", String((dep.readyNs - lo) / (hi - lo))); svg.append(mover); }
        const g = mark("g", { class: "machine-input" }); g.append(box(startX - 38, y - 23, 76, 46, dep.last ? "amber-part" : "teal-part"), dot(startX - 18, y, 6, "bolt"), dot(startX + 18, y, 6, "bolt"));
        interactive(g, producer.id, `${producer.label}: ready ${fmtNs(dep.readyNs - lo)}; ${fmtNs(dep.slackNs)} before last input`);
        text(startX, y + 43, producer.label, "small", "middle");
        text(startX, y + 63, dep.last ? "Last recorded input" : `${fmtNs(dep.slackNs)} earlier`, dep.last ? "small accent" : "small muted", "middle");
      });
      const lastRoute = svg.querySelector("path.last-input"); if (lastRoute) svg.append(lastRoute);
      svg.append(dot(W / 2, 188, 5, "ink-part"));
      const gate = mark("g"); gate.append(box(endX - 38, 139, 76, 100, "machine-case"), box(endX - 28, 155, 56, 15, "ink-part"), box(endX - 28, 211, 56, 13, "teal-part"), linePart(endX, 170, endX, 199));
      if (b.origin === "illustrated") { gate.setAttribute("data-stroke", "press"); gate.setAttribute("data-from", String((target.startNs - lo) / (hi - lo))); gate.setAttribute("data-to", String((target.endNs - lo) / (hi - lo))); }
      interactive(gate, target.id, target.label); text(endX, 265, target.label, "small", "middle");
      text(endX, 293, `${fmtNs(state.afterReadyNs)} later`, "accent", "middle");
      const y = compact ? 403 : 372; axis(y);
      svg.append(box(x(state.lastReadyNs), y - 29, x(target.startNs) - x(state.lastReadyNs), 22, "wait-part"));
      svg.append(dot(x(state.lastReadyNs), y - 18, 5, "amber-part"), dot(x(target.startNs), y - 18, 5, "ink-part"));
      cursor(y - 42, y + 3);
      notes.push("Amber is the last recorded prerequisite. Violet is the interval from that readiness to consumer start.", "Earlier inputs have slack relative to this join. This is not a complete critical path or a speedup estimate.");
    }
  } else if (lens === "instructions") {
    focusRows(b.instructions, compact ? 2 : 3).forEach((r, i) => {
      const y = 40 + i * (compact ? 160 : 118); const left = compact ? 25 : 220;
      text(compact ? 25 : 30, y + 5, r.label, "");
      const yy = y + (compact ? 35 : 23); line(left, yy + 34, W - 40, yy + 34);
      for (let j = 0; j < (compact ? 8 : 16); j++) {
        const xx = left + j * (W - left - 50) / (compact ? 8 : 16); const g = mark("g");
        g.append(box(xx, yy, compact ? 30 : 32, 43, j / (compact ? 8 : 16) < r.bookkeeping / Math.max(1, r.instructions) ? "wait-part" : "paper-part", 2));
        for (let k = 0; k < 3; k++) g.append(dot(xx + 8 + (k % 2) * 11, yy + 10 + k * 10, 2, "ink-part"));
        if (b.origin === "illustrated") { g.setAttribute("data-card", String(j)); g.setAttribute("data-card-total", compact ? "8" : "16"); }
        interactive(g, r.id, `${r.label} at ${r.pc}: ${r.instructions} instructions, ${r.bookkeeping} bookkeeping`);
      }
      const width = (W - left - 40) * r.fetchWaitNs / Math.max(1, r.endNs - r.startNs);
      svg.append(box(left, yy + 55, width, 12, "wait-part")); text(left, yy + 91, `${fmtNs(r.fetchWaitNs)} fetch wait · ${r.pc}`, "small");
    });
    notes.push("Punched cards represent executed instructions; violet cards mark bookkeeping.", "Fetch waiting is an explicit interval total. A no-instructions sample alone does not prove an instruction-cache miss.");
  } else if (lens === "work") {
    const selected = b.work.find(r => r.id === selection) ?? b.work[0];
    if (selected) {
      const cell = compact ? 30 : 37, left = compact ? 28 : 55, top = 58;
      const fraction = usefulFraction(selected);
      for (let i = 0; i < 64; i++) interactive(box(left + i % 8 * cell, top + Math.floor(i / 8) * cell, cell - 5, cell - 5, fraction !== null && i / 64 < fraction ? "teal-part" : "empty-part", 3), selected.id, `${selected.label}: ${selected.valid} valid / ${selected.issued} issued elements`);
      const xx = compact ? 292 : 430;
      svg.append(box(xx, top - 16, compact ? 74 : 390, 24, "ink-part"), box(xx, top + 36, compact ? 74 : 390, 38, "machine-case"), box(xx + 14, top + 90, compact ? 46 : 360, 17, "teal-part"));
      if (b.origin === "illustrated") { const ram = box(xx + 8, top + 12, compact ? 58 : 374, 12, "ink-part"); ram.setAttribute("data-stroke", "ram"); ram.setAttribute("data-from", "0"); ram.setAttribute("data-to", "1"); svg.append(ram); }
      text(xx + (compact ? 37 : 195), top + 153, fraction === null ? "—" : `${Math.round(fraction * 100)}%`, "large", "middle");
      text(xx + (compact ? 37 : 195), top + 180, "useful", "", "middle");
      const y = compact ? 355 : 385;
      text(left, y, `${selected.valid} valid / ${selected.issued} issued`, "");
      text(left, y + 32, `${selected.requestedBytes.toLocaleString()} B requested → ${selected.transferredBytes.toLocaleString()} B moved`, "small");
      notes.push("Solid spaces contribute valid output. Outlined spaces show the unused fraction of an issued tray.", "The 64 spaces are a proportional picture, not the actual hardware tile geometry. Requested and moved bytes are different quantities.");
    }
  } else if (lens === "lifetimes") {
    const left = compact ? 28 : 200;
    focusRows(b.buffers, compact ? 3 : 4).forEach((r, i) => {
      const y = 60 + i * (compact ? 105 : 82); text(compact ? 28 : 24, y - 17, `${r.label} · ${r.memory}`, "");
      const phases = [[r.requestNs, r.readyNs, "amber-part"], [r.readyNs, r.firstUseNs, "idle-part"], [r.firstUseNs, r.lastUseNs, "teal-part"], [r.lastUseNs, r.releaseNs, "wait-part"]] as const;
      for (const [a, z, c] of phases) interactive(box(x(a, left), y, Math.max(1, x(z, left) - x(a, left)), 28, c, 2), r.id, `${r.label}, ${r.generation}: ${fmtNs(lifetime(r).heldNs)} held after last use`);
      [r.readyNs, r.firstUseNs, r.lastUseNs, r.releaseNs].forEach(t => svg.append(dot(x(t, left), y + 14, 4, "ink-part")));
      text(left, y + 51, `${(r.bytes / 1024).toFixed(0)} KiB · ${r.outcome} · ${fmtNs(lifetime(r).heldNs)} after use`, "small");
    }); axis(H - 34, left); cursor(20, H - 34, left);
    notes.push("Amber: delivery. Pale: arrived but unused. Teal: in use. Violet: still held after use.", "Retained buffers end at the observation boundary; that endpoint does not claim a release.");
  } else if (lens === "bursts") {
    const allGroups = sampleGroups(b), selectedGroup = allGroups.find(g => g.rows.some(r => r.id === selection));
    const groups = (selectedGroup ? [selectedGroup, ...allGroups.filter(g => g !== selectedGroup)] : allGroups).slice(0, 3), height = (H - 60) / Math.max(1, groups.length);
    if (allGroups.length > 3) notes.push(`Showing 3 of ${allGroups.length} metric/pass groups. All samples remain in the export.`);
    groups.forEach((g, i) => {
      const y = 55 + i * height, base = y + height - 80, max = Math.max(1, ...g.rows.map(s => s.value));
      text(30, y - 24, g.rows[0].metric, ""); text(W - 30, y - 24, compact ? g.instrument.pass.split(" · ")[0] : g.instrument.pass, "small", "end");
      line(34, base, W - 34, base);
      for (const r of g.rows.slice(0, 2000)) {
        const h = r.value / max * Math.max(20, height - 95);
        interactive(box(x(r.startNs), base - h, Math.max(1, x(r.endNs) - x(r.startNs) - 2), Math.max(2, h), i ? "teal-part" : "amber-part", 1), r.id, `${r.metric}: ${r.value} ${r.unit}, ${fmtNs(r.endNs - r.startNs)} sample`);
      }
      text(34, base + 24, `${max} ${g.rows[0].unit} at full height`, "small");
      if (g.rows.length > 2000) notes.push(`${g.rows[0].metric}: drawing the first 2,000 of ${g.rows.length} samples. The export retains the complete series.`);
    }); axis(H - 30); cursor(30, H - 30);
    text(W / 2, H - 4, "Elapsed time", "small", "middle");
    text(W / 2, 15, b.origin === "illustrated" ? "Cursor: example time" : "Cursor: recorded time", "small", "middle");
    notes.push("Every column spans the time it sampled. The spaces between columns are unobserved time, not zero activity.", "Each row names its collection pass; different passes are not simultaneous evidence.");
  } else if (lens === "values") {
    const selected = b.values.find(r => r.id === selection) ?? b.values[0];
    const rows = selected ? [selected, ...b.values.filter(r => r !== selected && JSON.stringify(r.edges) === JSON.stringify(selected.edges))].slice(0, 2) : [];
    const max = Math.max(1, ...rows.flatMap(r => r.counts));
    const left = 34, width = W - 68, base = 295, bins = selected?.counts.length ?? 1, cell = width / bins;
    rows.forEach((r, row) => {
      r.counts.forEach((count, i) => { const h = count / max * 210; interactive(box(left + i * cell + row * cell / 2, base - h, cell / 2 - 3, h, row ? "teal-part" : "amber-part", 2), r.id, `${r.label}, [${r.edges[i]}, ${r.edges[i + 1]}): ${count} ${r.population ?? "values"}`); });
      text(34, 340 + row * 45, `${r.label} · ${r.zeros} zeros · ${r.saturated ?? "?"} saturated`, row ? "teal-text" : "amber-text");
      if (r.population) notes.push(`${r.label}: ${r.population}.`);
      if (r.saturated === null) notes.push(`${r.label}: saturation is not measurable from the stored codes; a top-bin code is not proof of clipping.`);
    });
    line(left, base, W - 34, base); text(left, base + 25, String(selected?.edges[0] ?? ""), "axis"); text(W - 34, base + 25, String(selected?.edges.at(-1) ?? ""), "axis", "end");
    text(34, 38, "Same bins. Same count scale.", "");
    notes.push("Overlaid populations share exact bin edges and one count scale. Select a boundary to compare its compatible distributions.", "Zeros or similar values do not prove that computation can be skipped. Check the output after any scaling or rotation change.");
  } else if (lens === "tokens") {
    const rows = [...b.tokens].sort((a, z) => a.position - z.position), max = Math.max(1, ...rows.map(t => t.endNs - t.startNs));
    const positions = rows.map(t => t.position), minPos = Math.min(...positions), maxPos = Math.max(...positions);
    const tx = (p: number) => 40 + (p - minPos) / (maxPos - minPos || 1) * (W - 80);
    const ty = (d: number) => 315 - d / max * 240;
    rows.forEach((r, i) => {
      const prev = rows[i - 1]; if (prev && prev.sequence === r.sequence && prev.instrumentId === r.instrumentId && prev.position + 1 === r.position) line(tx(prev.position), ty(prev.endNs - prev.startNs), tx(r.position), ty(r.endNs - r.startNs), "token-connection");
      line(tx(r.position), 315, tx(r.position), ty(r.endNs - r.startNs), "rail");
      interactive(dot(tx(r.position), ty(r.endNs - r.startNs), compact ? 7 : 10, r.limit ? "wait-part" : "teal-part"), r.id, `Token ${r.position}: ${fmtNs(r.endNs - r.startNs)}, context ${r.context}, batch ${r.batch}`);
      if (!compact || i === 0 || i === rows.length - 1) text(tx(r.position), 345, String(r.position), "small", "middle");
    });
    text(34, 34, `${fmtNs(max)} at the top`, ""); text(34, 391, "Token position →", "");
    notes.push("Only adjacent positions in the same recorded sequence and instrument pass are connected.", "Compare context, batch, clocks and collection conditions before treating a spike as a regression.");
  } else {
    focusRows(b.instruments, compact ? 2 : 4).forEach((r, i) => {
      const y = 38 + i * (compact ? 195 : 90);
      interactive(box(28, y, compact ? W - 56 : 180, 54, "machine-case"), r.id, `${r.label}: ${r.method}`);
      text(compact ? W / 2 : 118, y + 33, r.label, "", "middle");
      const labels = [r.pass, `Cache: ${r.cache ?? "unknown"}`, `Clocks: ${r.clocks ?? "unknown"}`];
      labels.forEach((label, j) => {
        const xx = compact ? 30 : 250 + j * 213, yy = compact ? y + 83 + j * 31 : y + 30;
        if (!compact) line(xx - 28, yy - 4, xx - 8, yy - 4);
        text(xx, yy, label.replace(" (illustrated)", "").replace(" · separate execution", ""), "small");
      });
    });
    notes.push("Every reading leads back to its instrument and execution pass.", "Unknown settings remain unknown. Counter passes, lifecycle traces and ordinary execution are different observations.");
  }
  return { svg, notes, selected: selection };
}

function linePart(x1: number, y1: number, x2: number, y2: number) { return mark("line", { x1, y1, x2, y2, class: "piston" }); }
