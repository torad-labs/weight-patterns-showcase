import type { RecordedStep } from "./replay-model";

export type Span = { start: number; end: number };
export function mergeSpans(spans: Span[]): Span[] {
  const merged: Span[] = [];
  for (const span of spans
    .filter((s) => s.end > s.start)
    .sort((a, b) => a.start - b.start)) {
    const last = merged.at(-1);
    if (last && span.start <= last.end) last.end = Math.max(last.end, span.end);
    else merged.push({ ...span });
  }
  return merged;
}
export function spanPhase(spans: Span[], time: number) {
  let elapsed = 0,
    total = 0,
    active = false;
  for (const span of spans) {
    total += span.end - span.start;
    elapsed += Math.max(0, Math.min(span.end, time) - span.start);
    active ||= span.start <= time && time < span.end;
  }
  // A gap parks the packet in place. Resuming work never sends it to the inlet.
  return {
    active,
    present:
      !!spans.length &&
      time >= spans[0].start &&
      time < spans[spans.length - 1].end,
    progress: total ? elapsed / total : 0,
  };
}
const windows = new WeakMap<RecordedStep, Span[]>();
export function workPhase(step: RecordedStep | undefined, time: number) {
  if (!step) return { active: false, present: false, progress: 0 };
  let spans = windows.get(step);
  if (!spans) {
    spans = mergeSpans(
      step.observations.map((o) => ({ start: o[1], end: o[2] })),
    );
    windows.set(step, spans);
  }
  return spanPhase(spans, time);
}

// Each copy has one inlet-to-outlet flight. It never wraps while visible.
// Staggered flights illustrate reads; they do not invent transfer timestamps.
export function packetFlight(progress: number, index = 0, count = 1) {
  const spread = count > 1 ? 0.42 : 0,
    start = count > 1 ? (spread * index) / (count - 1) : 0;
  const p = (progress - start) / (1 - spread);
  return { visible: p >= 0 && p < 1, progress: Math.max(0, Math.min(1, p)) };
}
export function transportPhase(progress: number, hasInput: boolean) {
  const travelling = hasInput && progress < 0.4;
  return {
    travelling,
    travel: Math.min(1, progress / 0.4),
    machine: hasInput ? Math.max(0, (progress - 0.4) / 0.6) : progress,
  };
}
