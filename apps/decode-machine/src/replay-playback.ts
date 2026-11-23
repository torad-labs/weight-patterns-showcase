import { ConnectedJourney, journeyOperations } from "./connected-journey";
import { ReplayClock, type Capture, type MechanismId } from "./replay-model";
import { storyStages } from "./timeline";

export type View =
  | "engine"
  | "stack"
  | "layer"
  | "internals"
  | "material"
  | "hardware";

/** Owns time and transport commands. Inspecting space never starts playback. */
export class ReplayPlayback {
  readonly clock: ReplayClock;
  readonly journey: ConnectedJourney;
  illustrated = true;

  constructor(capture: Capture) {
    this.clock = new ReplayClock(capture);
    this.journey = new ConnectedJourney(capture.layers.length);
  }

  get playing() {
    return this.illustrated
      ? this.journey.timeline.playing
      : this.clock.playing;
  }
  set playing(value: boolean) {
    if (this.illustrated) this.journey.timeline.playing = value;
    else this.clock.playing = value;
  }
  get progress() {
    return this.illustrated ? this.journey.progress : this.clock.progress;
  }
  get seconds() {
    return this.illustrated ? this.journey.seconds : this.clock.seconds;
  }
  get scoped() {
    return this.illustrated && this.journey.scope !== null;
  }
  get time() {
    if (!this.illustrated) return this.clock.time;
    const t = this.journey.timeline,
      layer = this.clock.capture.layers[t.layerIndex];
    return (
      layer.journeyStart + t.layerProgress * (layer.end - layer.journeyStart)
    );
  }

  use(capture: Capture, illustrated: boolean) {
    this.clock.playing = false;
    this.journey.timeline.playing = false;
    this.journey.continue();
    this.clock.use(capture);
    this.illustrated = illustrated;
  }

  tick(seconds: number) {
    if (!Number.isFinite(seconds)) return;
    if (this.illustrated) this.journey.tick(seconds);
    else this.clock.tick(seconds);
  }

  seek(progress: number) {
    if (!Number.isFinite(progress)) return;
    this.playing = false;
    if (this.illustrated) this.journey.seek(progress);
    else this.clock.seek(progress);
  }

  setDuration(seconds: number) {
    if (this.illustrated) this.journey.setDuration(seconds);
    else this.clock.setDuration(seconds);
  }

  toggle() {
    const playing = this.playing;
    if (this.progress >= 1) this.seek(0);
    this.playing = !playing;
  }

  restart(whole = false) {
    if (whole && this.illustrated) this.journey.continue();
    this.seek(0);
    this.playing = true;
  }

  inspect(layer: number, part?: MechanismId) {
    if (this.illustrated) this.journey.inspect(layer, part);
  }

  show(view: View, layer: number) {
    if (!this.illustrated) return;
    if (view === "engine" || view === "stack") this.journey.continue();
    else if (view === "layer" && this.journey.scope?.part)
      this.journey.inspect(layer);
  }

  watchLayer(layer: number) {
    this.inspect(layer);
    if (this.illustrated) this.journey.seek(0);
    else this.clock.seekTime(this.clock.capture.layers[layer].journeyStart);
    this.playing = true;
  }

  step(direction: number) {
    this.playing = false;
    if (!this.illustrated) {
      this.clock.next(direction);
      return;
    }
    const scope = this.journey.scope;
    const stages = storyStages.filter((stage) => stage.pass !== null);
    const boundaries = scope?.part
      ? Array.from(
          { length: 11 },
          (_, i) => scope.start + ((scope.end - scope.start) * i) / 10,
        )
      : [
          ...storyStages.map((stage) => stage.start),
          1,
          ...stages.flatMap((stage) =>
            Array.from(
              { length: this.journey.timeline.layerCount },
              (_, layer) =>
                journeyOperations.map(
                  (op) =>
                    stage.start +
                    ((stage.end - stage.start) * (layer + op.start)) /
                      this.journey.timeline.layerCount,
                ),
            ).flat(),
          ),
        ];
    const start = scope?.start ?? 0,
      end = scope?.end ?? 1;
    const times = [...new Set([...boundaries, start, end])]
      .filter((t) => t >= start && t <= end)
      .sort((a, b) => a - b);
    const time = this.journey.timeline.progress;
    this.journey.timeline.seek(
      direction > 0
        ? (times.find((t) => t > time + 1e-9) ?? end)
        : (times.filter((t) => t < time - 1e-9).at(-1) ?? start),
    );
  }

  layerPosition(layer: number) {
    if (!this.illustrated) {
      const c = this.clock.capture;
      return (c.layers[layer].journeyStart - c.start) / c.duration;
    }
    const stage = storyStages.find(
      (s) => s.pass === this.journey.timeline.passIndex,
    )!;
    return (
      stage.start +
      ((stage.end - stage.start) * layer) / this.journey.timeline.layerCount
    );
  }
}
