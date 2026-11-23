import {
  mechanisms,
  type Capture,
  type MechanismId,
  type RecordedStep,
} from "./replay-model";
import { mechanismForRole } from "./workshop-model";
import { mergeSpans, spanPhase, workPhase, type Span } from "./flow-motion";
import type { Timeline } from "./timeline";

export const weightedParts: MechanismId[] = [
  "qkv",
  "output",
  "gateup",
  "down",
  "ple",
];
export const weightRole = (part: MechanismId) =>
  (
    ({
      qkv: "mixer:0",
      output: "main:0",
      gateup: "gate-up:1",
      down: "main:10",
      ple: "main:12",
    }) as Partial<Record<MechanismId, string>>
  )[part];
export function weightConsumer(step: RecordedStep) {
  return (
    step.kind !== "wait" &&
    step.kind !== "sentinel" &&
    ["mixer:0", "q:0", "main:0", "gate-up:1", "main:10", "main:12"].includes(
      step.role,
    )
  );
}
type WorkerInterval = Span & { step: RecordedStep };
export type WorkerState = {
  active: boolean;
  wait: boolean;
  progress: number;
  step?: RecordedStep;
};

/** Shared by the worker floor and its hardware cutaway. */
export function workerActivity(
  story: Timeline | null,
  recorded: RecordedMotion,
  block: number,
  time: number,
): WorkerState {
  return story
    ? {
        active: story.layerActive && block % 8 < 6,
        wait: false,
        progress: story.layerActive ? story.layerProgress : 0,
        step: undefined,
      }
    : recorded.worker(block, time);
}

// This selects recorded work, not transfer events. Travel on the drawn pipes is
// an explanatory read copy; its phase is a pure function of the replay time.
export class RecordedMotion {
  readonly layers: RecordedStep[][];
  readonly weightSpans: Span[][];
  readonly workers: WorkerInterval[][];
  readonly workerWorkSpans: Span[][][];
  readonly workerWeightSpans: Span[][][];
  constructor(public capture: Capture) {
    this.layers = capture.layers.map((l) =>
      capture.steps.filter((s) => s.layer === l.id && s.kind !== "sentinel"),
    );
    const reads = capture.steps.filter(weightConsumer);
    this.weightSpans = this.layers.map((steps) =>
      mergeSpans(
        steps
          .filter(weightConsumer)
          .flatMap((s) =>
            s.observations.map((o) => ({ start: o[1], end: o[2] })),
          ),
      ),
    );
    this.workers = Array.from({ length: capture.grid }, () => []);
    for (const step of capture.steps)
      if (step.kind !== "sentinel")
        for (const [block, start, end] of step.observations) {
          if (end > start) this.workers[block].push({ step, start, end });
        }
    // Resolve nested observations once. Child intervals explain their parent;
    // they are not extra serial jobs and must not restart a worker's stroke.
    this.workers = this.workers.map((intervals) => {
      const times = [
        ...new Set(intervals.flatMap((o) => [o.start, o.end])),
      ].sort((a, b) => a - b);
      intervals.sort((a, b) => a.start - b.start);
      let cursor = 0,
        active: WorkerInterval[] = [];
      return times.slice(0, -1).flatMap((start, i) => {
        active = active.filter((o) => start < o.end);
        while (cursor < intervals.length && intervals[cursor].start <= start)
          active.push(intervals[cursor++]);
        active.sort(
          (a, b) =>
            Number(!!b.step.parent) - Number(!!a.step.parent) ||
            a.end - a.start - (b.end - b.start),
        );
        return active[0]
          ? [{ start, end: times[i + 1], step: active[0].step }]
          : [];
      });
    });
    this.workerWorkSpans = this.workers.map((intervals) =>
      capture.layers.map((l) =>
        mergeSpans(
          intervals.filter(
            (o) => o.step.layer === l.id && o.step.kind !== "wait",
          ),
        ),
      ),
    );
    this.workerWeightSpans = Array.from({ length: capture.grid }, (_, block) =>
      capture.layers.map((l) =>
        mergeSpans(
          reads
            .filter((s) => s.layer === l.id)
            .flatMap((s) =>
              s.observations
                .filter((o) => o[0] === block)
                .map((o) => ({ start: o[1], end: o[2] })),
            ),
        ),
      ),
    );
  }
  weights(layer: number, time: number) {
    return spanPhase(this.weightSpans[layer], time);
  }
  workerWeights(block: number, layer: number, time: number) {
    return spanPhase(this.workerWeightSpans[block][layer], time);
  }
  worker(block: number, time: number): WorkerState {
    const current = this.workers[block].find(
      (o) => o.start <= time && time < o.end,
    );
    const step =
      current?.step ??
      this.workers[block].filter((o) => o.end <= time).at(-1)?.step;
    if (!step) return { active: false, wait: false, progress: 0 };
    return {
      active: !!current,
      wait: !!current && step.kind === "wait",
      step,
      progress: spanPhase(this.workerWorkSpans[block][step.layer], time)
        .progress,
    };
  }
  layerAt(time: number) {
    return (
      this.capture.layers.filter((l) => l.journeyStart <= time).at(-1)?.id ?? 0
    );
  }
  state(layer: number, part: MechanismId, time: number, weights = false) {
    const role = weights
      ? weightRole(part)
      : mechanisms.find((m) => m.id === part)?.role;
    const step = this.layers[layer].find((s) => s.role === role);
    return { step, ...workPhase(step, time), done: !!step && time >= step.end };
  }
  focus(time: number, layer = this.layerAt(time)) {
    const steps = this.layers[layer];
    const active = steps.filter((s) =>
      s.observations.some((o) => o[1] <= time && time < o[2]),
    );
    const step =
      active.sort(
        (a, b) =>
          Number(a.kind === "wait") - Number(b.kind === "wait") ||
          b.lastStart - a.lastStart ||
          a.end - a.start - (b.end - b.start),
      )[0] ??
      steps.filter((s) => s.end <= time).sort((a, b) => b.end - a.end)[0] ??
      steps[0];
    return { layer, part: mechanismForRole(step.role), step };
  }
  reads(layer: number, time: number) {
    return weightedParts
      .map((part) => ({ part, ...this.state(layer, part, time, true) }))
      .filter((s) => s.active);
  }
}
