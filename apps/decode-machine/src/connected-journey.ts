import { Timeline, storyStages } from "./timeline";
import type { MechanismId } from "./replay-model";

// Authored explanatory timing. The capture supplies architecture, never these times or words.
export const journeyOperations: {
  id: MechanismId;
  start: number;
  end: number;
  verb: string;
}[] = [
  { id: "norm", start: 0, end: 0.08, verb: "Size the incoming information" },
  {
    id: "qkv",
    start: 0.08,
    end: 0.22,
    verb: "Read weights into the projection presses",
  },
  {
    id: "rope",
    start: 0.22,
    end: 0.28,
    verb: "Turn coordinate pairs for their position",
  },
  {
    id: "attention",
    start: 0.28,
    end: 0.5,
    verb: "Read and combine the retained context",
  },
  {
    id: "output",
    start: 0.5,
    end: 0.6,
    verb: "Project the combined information",
  },
  { id: "residual", start: 0.6, end: 0.65, verb: "Rejoin the preserved input" },
  {
    id: "gateup",
    start: 0.65,
    end: 0.8,
    verb: "Expand two branches and gate their mixture",
  },
  {
    id: "down",
    start: 0.8,
    end: 0.9,
    verb: "Compress the expanded information",
  },
  { id: "ple", start: 0.9, end: 0.97, verb: "Add the layer’s side input" },
  {
    id: "latch",
    start: 0.97,
    end: 1,
    verb: "Release the tray to the next layer",
  },
];
export const clamp = (n: number) => Math.max(0, Math.min(1, n));
export function journeyOperation(id: MechanismId) {
  return journeyOperations.find(
    (p) =>
      p.id === (id === "cache" ? "attention" : id === "weights" ? "qkv" : id),
  )!;
}
export function journeyPartState(t: Timeline, layer: number, id: MechanismId) {
  const op = journeyOperation(id),
    p = t.layerActive
      ? layer < t.layerIndex
        ? 1
        : layer > t.layerIndex
          ? 0
          : t.layerProgress
      : t.progress < 0.11
        ? 0
        : 1;
  return {
    progress: clamp((p - op.start) / (op.end - op.start)),
    active:
      t.layerActive && layer === t.layerIndex && p >= op.start && p < op.end,
    done: p >= op.end,
  };
}
export class ConnectedJourney {
  timeline: Timeline;
  scope: {
    start: number;
    end: number;
    layer: number;
    part?: MechanismId;
  } | null = null;
  wholeSeconds = 120;
  detailSeconds = 12;
  constructor(layerCount: number) {
    this.timeline = new Timeline(layerCount);
  }
  get seconds() {
    return this.scope ? this.detailSeconds : this.wholeSeconds;
  }
  get progress() {
    const s = this.scope;
    return s
      ? clamp((this.timeline.progress - s.start) / (s.end - s.start))
      : this.timeline.progress;
  }
  setDuration(seconds: number) {
    if (!Number.isFinite(seconds)) return;
    if (this.scope) this.detailSeconds = Math.max(1, Math.min(3600, seconds));
    else this.wholeSeconds = Math.max(1, Math.min(3600, seconds));
  }
  seek(p: number) {
    if (!Number.isFinite(p)) return;
    const s = this.scope;
    this.timeline.seek(s ? s.start + clamp(p) * (s.end - s.start) : p);
  }
  tick(dt: number) {
    if (!this.timeline.playing) return;
    const next = this.progress + Math.max(0, dt) / this.seconds;
    this.seek(next);
    if (next >= 1) this.timeline.playing = false;
  }
  inspect(layer: number, part?: MechanismId) {
    const t = this.timeline,
      stage = storyStages.find((s) => s.pass === t.passIndex)!,
      op = part ? journeyOperation(part) : { start: 0, end: 1 };
    const unit = (stage.end - stage.start) / t.layerCount;
    this.scope = {
      start: stage.start + unit * (layer + op.start),
      end: stage.start + unit * (layer + op.end),
      layer,
      part,
    };
    if (t.progress < this.scope.start || t.progress >= this.scope.end)
      this.seek(0);
  }
  continue() {
    this.scope = null;
  }
  restart() {
    this.seek(0);
    this.timeline.playing = true;
  }
  get operation() {
    return (
      journeyOperations.find((o) => this.timeline.layerProgress < o.end) ??
      journeyOperations.at(-1)!
    );
  }
}
