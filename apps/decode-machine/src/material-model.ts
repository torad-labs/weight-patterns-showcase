import type { ReplayPlayback } from "./replay-playback";
import { Timeline, PROMPT_TOKENS } from "./timeline";
import { clamp, journeyOperations } from "./connected-journey";
import { RecordedMotion, weightedParts } from "./recorded-motion";
import type { MechanismId } from "./replay-model";

export const contextLabels = [
  ...PROMPT_TOKENS.map((s) => s.trim()),
  "Paris",
  ".",
];
export const materialTopics = [
  ["cart", "Inside the cart"],
  ["context", "Retained context"],
  ["split", "Split & collect"],
  ["supply", "Weights & staging"],
  ["worker", "Worker roles"],
  ["wait", "Readiness & waits"],
  ["precision", "Codes & scales"],
  ["choice", "Choose the next token"],
  ["requests", "Shared machinery"],
] as const;
export type MaterialTopic = (typeof materialTopics)[number][0];
export type SlotState = "empty" | "loading" | "ready" | "reading" | "released";
export function storyPhase(t: Timeline) {
  if (t.progress < 0.11) return "Intake";
  if (t.stage.id === "prefill") return "Prefill";
  if (t.stage.id === "done") return "Complete";
  if (t.stage.id.startsWith("choose")) return "Selection";
  if (t.stage.id.startsWith("return")) return "Feedback";
  return "Decode";
}

// A small explanatory calculation, not Gemma values or a second inference engine.
const examples = new Map<
  string,
  {
    input: number[];
    output: number[];
    weights: number[];
    partials: number[];
    result: number;
  }
>();
export function channelExample(
  layer: number,
  token: number,
  progress: number,
): {
  input: number[];
  current: number[];
  weights: number[];
  partials: number[];
  result: number;
} {
  const key = `${layer}:${token}`;
  let frame = examples.get(key);
  if (!frame) {
    const input =
      layer > 0
        ? channelExample(layer - 1, token, 1).current
        : Array.from({ length: 12 }, (_, i) =>
            Math.sin((token + 1) * 0.71 + i * 0.83),
          );
    const weights = Array.from(
      { length: 12 },
      (_, i) => Math.round(Math.cos((layer + 1) * 0.43 + i * 1.13) * 4) / 4,
    );
    const partials = Array.from(
      { length: 6 },
      (_, i) =>
        input[i * 2] * weights[i * 2] + input[i * 2 + 1] * weights[i * 2 + 1],
    );
    const result = partials.reduce((a, b) => a + b, 0);
    frame = {
      input,
      weights,
      partials,
      result,
      output: input.map((_, i) => Math.tanh(result / 3 + input[(i + 1) % 12])),
    };
    examples.set(key, frame);
  }
  const current = frame.input.map(
    (v, i) => v + (frame.output[i] - v) * clamp(progress),
  );
  return {
    input: frame.input,
    current,
    weights: frame.weights,
    partials: frame.partials,
    result: frame.result,
  };
}

// Eight nonnegative notches illustrate E2M1 levels. Scale and values are authored.
export function roundingExample(value: number, scale = 0.5) {
  const levels = [0, 0.5, 1, 1.5, 2, 3, 4, 6];
  const magnitude = Math.abs(value / scale);
  let code = 0;
  for (let i = 1; i < levels.length; i++) {
    const distance = Math.abs(levels[i] - magnitude),
      best = Math.abs(levels[code] - magnitude);
    if (distance < best || (distance === best && i % 2 === 0)) code = i;
  }
  const stored = Math.sign(value) * levels[code] * scale;
  return { value, scale, code, stored, difference: value - stored, levels };
}

export function taskCycle(p: number) {
  const progress = clamp(p);
  const branches = Array.from({ length: 6 }, (_, i) =>
    clamp((progress - 0.28 - i * 0.015) / 0.46),
  );
  const allReady = branches.every((x) => x === 1);
  return {
    branches,
    join: allReady ? clamp((progress - 0.815) / 0.125) : 0,
    released: allReady && progress >= 0.94,
    slots: Array.from({ length: 3 }, (_, i): SlotState => {
      const load = 0.03 + i * 0.055;
      return progress < load
        ? "empty"
        : progress < load + 0.07
          ? "loading"
          : progress < 0.28
            ? "ready"
            : progress < 0.815
              ? "reading"
              : "released";
    }),
  };
}

export class MaterialModel {
  multiple = false;
  request = 0;
  promptToken = 6;
  shortWindow = false;
  topic: MaterialTopic = "cart";
  private second: Timeline;
  private recorded: RecordedMotion;
  constructor(public playback: ReplayPlayback) {
    this.second = new Timeline(playback.clock.capture.layers.length);
    this.recorded = new RecordedMotion(playback.clock.capture);
  }
  snapshot() {
    const p = this.playback,
      capture = p.clock.capture;
    if (this.recorded.capture !== capture)
      this.recorded = new RecordedMotion(capture);
    const illustrated = p.illustrated;
    const many = illustrated && this.multiple;
    this.second.seek(clamp((p.journey.timeline.progress - 0.13) / 0.87));
    const timelines = many
      ? [p.journey.timeline, this.second]
      : [p.journey.timeline];
    const request = many ? Math.min(1, this.request) : 0;
    const t = timelines[request];
    const focus = illustrated ? null : this.recorded.focus(p.time);
    const layer = illustrated ? t.layerIndex : focus!.layer;
    const spec = capture.layers[layer];
    const operation = illustrated
      ? (journeyOperations.find((o) => t.layerProgress < o.end) ??
        journeyOperations.at(-1)!)
      : null;
    const part: MechanismId = illustrated ? operation!.id : focus!.part;
    const state = illustrated ? null : this.recorded.state(layer, part, p.time);
    const workProgress = illustrated
      ? clamp(
          (t.layerProgress - operation!.start) /
            (operation!.end - operation!.start),
        )
      : state!.progress;
    const active = illustrated ? t.layerActive : !!state?.present;
    const token =
      t.currentToken === "question"
        ? this.promptToken
        : t.currentToken === "Paris"
          ? 7
          : 8;
    const cycle = taskCycle(workProgress);
    const cacheCount = illustrated ? t.cacheCount(spec.owner) : null;
    const window = this.shortWindow && spec.kind === "S" ? 3 : spec.window;
    const entries = illustrated
      ? contextLabels.map((label, index) => ({
          label,
          index,
          stored: index < cacheCount!,
          eligible:
            index < cacheCount! &&
            index <= token &&
            (spec.kind === "F" || index > token - window),
        }))
      : null;
    const channels = illustrated
      ? channelExample(layer, token, t.layerProgress)
      : null;
    const choiceStage =
      illustrated && (t.stage.id.startsWith("choose") || t.stage.id === "done");
    const choosing = choiceStage
      ? t.stage.id === "done"
        ? 1
        : t.stageProgress
      : 0;
    const returning = illustrated && t.stage.id.startsWith("return");
    const candidates = illustrated
      ? t.candidateLabels.map((label, i) => ({
          label,
          score: [3.2, 1.1, 0.4, -0.2][i],
        }))
      : null;
    const requests = timelines.map((clock, index) => ({
      id: index === 0 ? "A" : "B",
      progress: clock.progress,
      layer: clock.layerIndex,
      layerProgress: clock.layerProgress,
      phase: storyPhase(clock),
      active: clock.layerActive,
      prefill: clock.stage.id === "prefill",
      cache: clock.cacheCount(capture.layers[clock.layerIndex].owner),
      answer: clock.answer,
    }));
    const recordedWait = !illustrated && focus!.step.kind === "wait";
    return {
      illustrated,
      request,
      requests: illustrated ? requests : [],
      layer,
      owner: spec.owner,
      kind: spec.kind,
      window: spec.window,
      teachingWindow: this.shortWindow && spec.kind === "S" && illustrated,
      token: illustrated ? token : null,
      identity: illustrated
        ? `${request ? "B" : "A"} · ${contextLabels[token]}`
        : `Position ${capture.position} · token unknown`,
      phase: illustrated ? requests[request].phase : "Recorded decode",
      part,
      operation: illustrated
        ? t.layerActive
          ? operation!.verb
          : t.stage.name
        : focus!.step.name,
      progress: workProgress,
      active,
      weighted: weightedParts.includes(part),
      branches: cycle.branches,
      joined: cycle.join,
      released: cycle.released,
      slots: illustrated ? cycle.slots : null,
      entries,
      cacheCount,
      channels,
      rounding: channels ? roundingExample(channels.current[0]) : null,
      waiting: illustrated ? active && !cycle.released : recordedWait,
      waitReason: illustrated
        ? cycle.released
          ? "All six contributions collected; the output can leave."
          : `Illustrated join: ${cycle.branches.filter((x) => x === 1).length} of 6 contributions ready.`
        : recordedWait
          ? "Recorded wait. The required condition is described by the operation; the cause and producer completion are not measured here."
          : "No active wait selected. Readiness signals were not captured individually.",
      candidates,
      choosing,
      choiceStage,
      returning,
      feedbackProgress: returning ? t.stageProgress : 0,
      answer: illustrated ? t.answer : null,
      selectedCandidate:
        (choiceStage && choosing >= 0.75) || returning
          ? candidates![0].label
          : null,
      timing: illustrated
        ? "Authored illustration · one shared clock"
        : "Captured operation interval · internal travel is symbolic",
    };
  }
}
export type MaterialSnapshot = ReturnType<MaterialModel["snapshot"]>;
