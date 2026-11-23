export type Observation = [
  block: number,
  start: number,
  end: number,
  sm: string,
];
export type RecordedStep = {
  id: string;
  layer: number;
  role: string;
  name: string;
  kind: string;
  parent: string | null;
  predecessors: string[];
  description: string;
  fromTap: string;
  toTap: string;
  start: number;
  lastStart: number;
  end: number;
  anomalies: number;
  observations: Observation[];
};
export type RecordedLayer = {
  id: number;
  kind: "S" | "F";
  owner: number;
  window: number;
  headDim: number;
  ffnWidth: number;
  start: number;
  journeyStart: number;
  end: number;
};
export type Capture = {
  id: string;
  position: number;
  grid: number;
  queryHeads: number;
  kvHeads: number;
  hidden: number;
  outputSlices: number;
  downSlices: number;
  source: string;
  tapeSha: string;
  tablesSha: string;
  start: number;
  end: number;
  duration: number;
  coverage?: string;
  semantics?: string;
  layers: RecordedLayer[];
  steps: RecordedStep[];
  promptText?: string;
  tokenRecord?: { input: number; output: number; source: string };
};
export type ReplayData = {
  schema: string;
  label: string;
  source: string;
  sourceSha: string;
  sourceRevision: string;
  captures: Capture[];
};

export function appUrl(rel: string) {
  const base = (import.meta.env?.BASE_URL ?? "/").replace(/\/?$/, "/");
  return `${base}${rel.replace(/^\//, "")}`;
}

/** Keep a recording and its source anatomy together, including in shared links. */
export function recordingUrls(href: string) {
  const page = new URL(href);
  const recording = new URL(page.searchParams.get("recording") ?? appUrl("replay/gemma4-e2b.json"), page);
  if (recording.origin !== page.origin || !recording.pathname.endsWith(".json")) throw Error("Choose a local JSON recording.");
  return { recording: recording.href, anatomy: new URL("anatomy.json", recording).href };
}
export type SourceNode = {
  id: string;
  parent?: string;
  label: string;
  level: number;
  kind: string;
  module?: string;
  source?: string;
  bindings?: { direction: string; port: string; type: string; value: string }[];
};
export type Anatomy = {
  revision: string;
  binding: string;
  entry: string;
  policy: string;
  roles: Record<string, string>;
  nodes: SourceNode[];
};
export class ReplayClock {
  progress = 0;
  seconds = 120;
  playing = false;
  constructor(public capture: Capture) {}
  get time() {
    return this.capture.start + this.progress * this.capture.duration;
  }
  setDuration(value: number) {
    if (Number.isFinite(value))
      this.seconds = Math.min(3600, Math.max(1, value));
  }
  seek(progress: number) {
    this.progress = Math.max(0, Math.min(1, progress));
    if (this.progress === 1) this.playing = false;
  }
  seekTime(time: number) {
    this.seek((time - this.capture.start) / this.capture.duration);
  }
  tick(seconds: number) {
    if (this.playing)
      this.seek(this.progress + Math.max(0, seconds) / this.seconds);
  }
  use(capture: Capture) {
    this.capture = capture;
    this.seek(0);
    this.playing = false;
  }
  get activeLayers() {
    return this.capture.layers.filter(
      (l) => l.journeyStart <= this.time && this.time < l.end,
    );
  }
  next(direction = 1) {
    const times = this.capture.steps
      .filter(
        (s) =>
          s.parent === null && s.kind !== "sentinel" && s.observations.length,
      )
      .map((s) => s.start)
      .sort((a, b) => a - b);
    this.seekTime(
      direction > 0
        ? (times.find((t) => t > this.time + 0.01) ?? this.capture.end)
        : (times.filter((t) => t < this.time - 0.01).at(-1) ??
            this.capture.start),
    );
  }
}
export function intervalProgress(start: number, end: number, time: number) {
  return end > start
    ? Math.max(0, Math.min(1, (time - start) / (end - start)))
    : 0;
}
export function observationAt(step: RecordedStep, time: number) {
  return step.observations.filter((o) => o[1] <= time && time < o[2]);
}
export function layerSteps(capture: Capture, layer: number) {
  return capture.steps.filter((s) => s.layer === layer);
}
export function stepRole(capture: Capture, layer: number, role: string) {
  return capture.steps.find((s) => s.layer === layer && s.role === role);
}
export function activeBlockSteps(
  capture: Capture,
  block: number,
  time: number,
) {
  return capture.steps.filter(
    (s) =>
      s.kind !== "sentinel" &&
      s.observations.some((o) => o[0] === block && o[1] <= time && time < o[2]),
  );
}
export function fmtNs(ns: number) {
  return ns >= 1e6
    ? `${(ns / 1e6).toFixed(3)} ms`
    : `${(ns / 1e3).toFixed(2)} µs`;
}
export function stepLabel(step: RecordedStep) {
  const pass = step.role.match(/^attention-pass(\d+):/);
  return pass ? `Pass ${Number(pass[1]) + 1} · ${step.name}` : step.name;
}
export const mechanisms = [
  {
    id: "norm",
    name: "Sizing jaws",
    role: null,
    source: "rms",
    description:
      "The jaws set a common scale: square the channels, combine their magnitude, take the inverse RMS, then apply the learned gain. No separate input-normalization interval is recorded here.",
  },
  {
    id: "qkv",
    name: "Projection presses",
    role: "mixer:0",
    source: "matvec",
    description:
      "The same hidden carrier enters differently weighted presses to form Q, K and V. Shared-cache layers form Q here and read the earlier owner’s K/V instead.",
  },
  {
    id: "rope",
    name: "Paired rotation",
    role: "qtail:3",
    source: "rope",
    description:
      "Pairs of Q coordinates turn together according to position. This changes their direction, not their pair length. The marked Q tail is the last visit when a head has two passes.",
  },
  {
    id: "attention",
    name: "Eight reading rollers",
    role: "mixer:2",
    source: "attend",
    description:
      "Eight query heads compare against one shared K/V head. Score, softmax and weighted-value work happen inside this housing. The recorded pass details mark the final tile; earlier tiles are included in its entry span.",
  },
  {
    id: "cache",
    name: "Paired K/V drum",
    role: null,
    source: "kv",
    description:
      "Two storage drums represent keys and values, with one K/V head. Sliding readers retain a 512-position window; full readers can address up to 131072. The drum has no recorded contents or occupancy.",
  },
  {
    id: "output",
    name: "Two partial-sum cups",
    role: "main:0",
    source: "matvec",
    description:
      "The attention result enters the output projection. Two split-K partitions contribute partial results for each output region; the following fold combines them.",
  },
  {
    id: "residual",
    name: "Preserved double track",
    role: "main:2",
    source: "two_term",
    description:
      "A bypass keeps the entering residual. The projected result is normalized and added back. The paired track represents deterministic hi + lo FP8 carriers, not two tokens.",
  },
  {
    id: "gateup",
    name: "Twin expansion presses",
    role: "main:8",
    source: "gate_up",
    description:
      "Two weighted projections expand the same input in parallel. A GELU-shaped gate multiplies one branch with the other. Layers 16–35 expand to 12288 channels; earlier layers use 6144.",
  },
  {
    id: "down",
    name: "Six-cup compression press",
    role: "main:10",
    source: "matvec",
    description:
      "Six split-K partitions supply partial products to the down projection. The compressor returns the expanded representation to 1536 hidden channels.",
  },
  {
    id: "ple",
    name: "Side-input cartridge",
    role: "main:12",
    source: "ple",
    description:
      "A separate 256-channel per-layer input enters through its gated projection, normalization and addition. It shares the recorded down-consumer tail with residual work and layer scaling; its own duration is not measured.",
  },
  {
    id: "latch",
    name: "Readiness latch",
    role: "main:15",
    source: "counter",
    description:
      "A release signal follows the data store. The next reader waits on its required counter. Recorded waits show time between taps, not proof of which producer caused a delay.",
  },
  {
    id: "weights",
    name: "Weight drawer & staging rack",
    role: null,
    source: "tma",
    description:
      "Stored amber balls represent reusable packed weights and their scales. Read copies can supply a shared-memory ring, then a tensor-core press. Moving copies illustrate weight use during captured work; stored balls remain. Their values, transfer timings and ring occupancy were not recorded.",
  },
] as const;
export type MechanismId = (typeof mechanisms)[number]["id"];
