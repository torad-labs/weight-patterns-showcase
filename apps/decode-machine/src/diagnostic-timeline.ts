import { DiagnosticModel } from "./diagnostic-model";
import { fmtNs } from "./replay-model";

/** A static interval drawing with a live cursor, all on the application's clock. */
export class DiagnosticTimeline {
  private canvas: HTMLCanvasElement;
  private rows: HTMLElement;
  private cursor: HTMLElement;
  private width = 1;
  private rowHeight = 32;
  private model?: DiagnosticModel;
  constructor(private host: HTMLElement, private pick: (block: number, time?: number, record?: string) => void) {
    host.innerHTML = `<div class="diagnostic-axis"><span>Worker</span><div id="diagnostic-ticks"></div></div>
      <div class="diagnostic-scroll" aria-label="All recorded workers">
        <div class="diagnostic-plot"><div class="diagnostic-rows"></div>
          <canvas aria-label="Recorded intervals. Select a worker with the row buttons and use the window time slider to navigate."></canvas>
          <div class="diagnostic-cursor" aria-hidden="true"></div></div></div>`;
    this.canvas = host.querySelector("canvas")!;
    this.rows = host.querySelector(".diagnostic-rows")!;
    this.cursor = host.querySelector(".diagnostic-cursor")!;
    new ResizeObserver(() => this.draw()).observe(this.canvas);
    this.canvas.addEventListener("click", e => {
      if (!this.model) return;
      const rect = this.canvas.getBoundingClientRect(), bounds = this.model.bounds();
      const block = Math.min(this.model.capture.grid - 1, Math.max(0, Math.floor((e.clientY - rect.top) / this.rowHeight)));
      const time = bounds.start + (e.clientX - rect.left) / rect.width * (bounds.end - bounds.start);
      // Operation zoom retains the selected parent, rather than silently picking a child.
      const hit = this.model.recordAt(block, time);
      this.pick(block, time, hit?.id);
    });
  }
  render(model: DiagnosticModel, _reveal = false) {
    this.model = model;
    if (this.rows.children.length !== model.capture.grid) {
      this.rows.replaceChildren(...Array.from({ length: model.capture.grid }, (_, block) => {
        const b = document.createElement("button"); b.type = "button";
        b.textContent = String(block).padStart(2, "0"); b.setAttribute("aria-label", `Inspect worker ${block}`);
        b.addEventListener("click", () => this.pick(block)); return b;
      }));
    }
    Array.from(this.rows.children).forEach((b, i) => b.setAttribute("aria-pressed", String(i === model.block)));
    this.draw();
  }
  draw() {
    const m = this.model;
    if (!m || !this.host.offsetWidth) return;
    const rect = this.canvas.getBoundingClientRect();
    this.rowHeight = parseFloat(getComputedStyle(this.host).getPropertyValue("--worker-row")) || 32;
    this.width = rect.width;
    const height = m.capture.grid * this.rowHeight;
    this.canvas.style.height = `${height}px`; this.rows.style.height = `${height}px`;
    const scale = Math.min(devicePixelRatio, 2);
    this.canvas.width = Math.round(this.width * scale); this.canvas.height = Math.round(height * scale);
    const ctx = this.canvas.getContext("2d")!; ctx.scale(scale, scale);
    const bounds = m.bounds(), span = bounds.end - bounds.start || 1;
    const x = (time: number) => (time - bounds.start) / span * this.width;
    for (let b = 0; b < m.capture.grid; b++) {
      ctx.fillStyle = b === m.block ? "#d5dfd8" : b % 2 ? "#e1e4dc" : "#ecece5";
      ctx.fillRect(0, b * this.rowHeight, this.width, this.rowHeight);
      const segments = m.window === "operation"
        ? m.step.observations.filter(o => o[0] === b && o[2] > o[1]).map(o => ({ start: o[1], end: o[2], step: m.step }))
        : m.motion.workers[b];
      for (const r of segments) {
        if (r.end <= bounds.start || r.start >= bounds.end) continue;
        const left = Math.max(0, x(r.start)), right = Math.min(this.width, x(r.end));
        ctx.fillStyle = r.step.kind === "wait" ? "#8a78a6" : r.step.kind === "work" ? "#387e75" : "#91a4a7";
        ctx.fillRect(left, b * this.rowHeight + this.rowHeight * .2, Math.max(1, right - left), this.rowHeight * .6);
      }
      // Outline the selected raw observation even where a nested interval is drawn.
      const o = m.step.observations.find(o => o[0] === b && o[2] > o[1]);
      if (o && o[2] > bounds.start && o[1] < bounds.end) {
        ctx.strokeStyle = "#263e3d"; ctx.lineWidth = b === m.block ? 2 : 1;
        ctx.strokeRect(Math.max(0, x(o[1])), b * this.rowHeight + this.rowHeight * .1, Math.max(1, Math.min(this.width, x(o[2])) - Math.max(0, x(o[1]))), this.rowHeight * .8);
      }
    }
    ctx.strokeStyle = "#a5b3a766"; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      ctx.beginPath(); ctx.moveTo(this.width * i / 4, 0); ctx.lineTo(this.width * i / 4, height); ctx.stroke();
    }
    this.host.querySelector("#diagnostic-ticks")!.replaceChildren(...Array.from({ length: 5 }, (_, i) => {
      const s = document.createElement("span"); s.textContent = fmtNs(bounds.start + span * i / 4 - m.capture.start); return s;
    }));
  }
  update(time: number) {
    if (!this.model) return;
    const b = this.model.bounds(), p = (time - b.start) / (b.end - b.start);
    this.cursor.hidden = p < 0 || p > 1;
    this.cursor.style.left = `calc(var(--worker-label) + (100% - var(--worker-label)) * ${Math.max(0, Math.min(1, p))})`;
  }
}
