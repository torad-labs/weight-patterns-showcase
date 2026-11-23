import { describe, test, expect } from "bun:test";
import { ReplayClock, activeBlockSteps, type ReplayData } from "./replay-model";
import { normalizeInternalCapture } from "../../decode-replay/internal-model.mjs";
import { admitWorldTopology } from "../../decode-replay/world-lifecycle.mjs";
import { decodeSequence } from "../../decode-replay/decode-sequence.mjs";
import dataJson from "../public/replay/gemma4-e2b.json";
import { stackCarrierAt, stackCarrierProgress, stackPath } from "./stack-route";
import {
  workshopParts,
  equipmentEdges,
  equipmentState,
  equipmentMotionState,
  mechanismForRole,
  type Equipment,
} from "./workshop-model";
import { mechanisms } from "./replay-model";
import { RecordedMotion, weightedParts, weightRole } from "./recorded-motion";
import { ReplayPlayback } from "./replay-playback";
import { HardwareModel, warpGroups } from "./hardware-model";
import anatomyJson from "../public/replay/anatomy.json";
import type { Anatomy } from "./replay-model";
import {
  MaterialModel,
  taskCycle,
  roundingExample,
  channelExample,
} from "./material-model";

describe("SM and warp inspection", () => {
  test("the source roles open into eight separate groups of 32 logical lanes", () => {
    const groups = warpGroups(anatomyJson.roles);
    expect(groups.map((g) => g.role)).toEqual([
      "controller",
      "loader",
      "consumer",
      "consumer",
      "consumer",
      "consumer",
      "consumer",
      "consumer",
    ]);
    expect(
      new Set(groups.flatMap((g) => g.lanes.map((lane) => g.key + ":" + lane)))
        .size,
    ).toBe(256);
    for (const group of groups)
      expect(group.lanes).toEqual(Array.from({ length: 32 }, (_, i) => i));
  });
  test("hardware inspection preserves both replay clocks and does not invent placement or lane telemetry", () => {
    for (const capture of dataJson.captures)
      for (const illustrated of [false, true]) {
        const p = new ReplayPlayback(capture as ReplayData["captures"][number]);
        p.use(capture as ReplayData["captures"][number], illustrated);
        const model = new HardwareModel(p, anatomyJson as Anatomy);
        p.seek(0.31);
        const before = model.snapshot(5),
          time = p.time;
        p.setDuration(240);
        p.tick(1);
        expect(model.snapshot(5)).toEqual(before);
        model.selectedWarp = 7;
        model.selectedLane = 31;
        model.depth = "warp";
        expect(p.time).toBe(time);
        expect(p.playing).toBe(false);
        const selected = model.snapshot(5);
        expect(selected.warp.name).toBe("Consumer 6");
        expect(selected.lane).toBe(31);
        p.seek(0.8);
        p.seek(0.31);
        expect(model.snapshot(5)).toEqual(selected);
        for (const key of [
          "physicalSM",
          "otherResidentBlocks",
          "warpTiming",
          "activeLaneMask",
          "laneValues",
          "blockThreadBinding",
        ] as const)
          expect(selected[key]).toBeNull();
      }
  });
  test("waiting and idle block intervals never drive a moving warp illustration", () => {
    for (const capture of dataJson.captures) {
      const p = new ReplayPlayback(capture as ReplayData["captures"][number]);
      p.use(capture as ReplayData["captures"][number], false);
      const model = new HardwareModel(p, anatomyJson as Anatomy);
      let waits = 0,
        work = 0;
      for (let i = 0; i <= 150; i++)
        for (const block of [0, 31, 62]) {
          p.seek(i / 150);
          const s = model.snapshot(block);
          if (!s.activity.active || s.activity.wait) expect(s.phase).toBe(0);
          if (s.activity.wait) waits++;
          if (s.activity.active && !s.activity.wait) work++;
        }
      expect(waits).toBeGreaterThan(0);
      expect(work).toBeGreaterThan(0);
    }
  });
});

describe("connected material cutaway", () => {
  test("the same illustrative channels carry each layer output into the next layer", () => {
    for (let token = 0; token < 9; token++)
      for (let layer = 0; layer < 34; layer++) {
        const previous = channelExample(layer, token, 1),
          next = channelExample(layer + 1, token, 0);
        expect(next.input).toEqual(previous.current);
        expect(next.current).toEqual(next.input);
      }
  });
  test("parallel contributions must all arrive before output release", () => {
    for (let i = 0; i <= 1000; i++) {
      const state = taskCycle(i / 1000);
      if (state.join > 0 || state.released)
        expect(state.branches.every((p) => p === 1)).toBe(true);
      if (i < 280) expect(state.branches.every((p) => p === 0)).toBe(true);
    }
    expect(taskCycle(0.95).released).toBe(true);
    expect(taskCycle(0.1).slots).toContain("loading");
    expect(taskCycle(0.5).slots.every((s) => s === "reading")).toBe(true);
  });
  test("prompt history obeys causality and independent requests retain separate context", () => {
    const p = new ReplayPlayback(
      dataJson.captures[0] as ReplayData["captures"][number],
    );
    const model = new MaterialModel(p);
    p.seek(0.3);
    model.multiple = true;
    model.promptToken = 2;
    const state = model.snapshot();
    expect(state.requests).toHaveLength(2);
    expect(state.requests[0].progress).toBeGreaterThan(
      state.requests[1].progress,
    );
    expect(
      state.entries!.filter((s) => s.eligible).map((s) => s.index),
    ).toEqual([0, 1, 2]);
    p.seek(0.5);
    const later = model.snapshot();
    expect(later.requests[0].answer).toBe("Paris");
    expect(later.requests[1].answer).toBe("");
    p.seek(0.3);
    expect(model.snapshot()).toEqual(state);
    p.setDuration(600);
    expect(model.snapshot()).toEqual(state);
    p.seek(1);
    expect(model.snapshot().requests.every((r) => r.answer === "Paris.")).toBe(
      true,
    );
  });
  test("recorded cutaways never invent values, cache contents, staging occupancy or output", () => {
    for (const c of dataJson.captures) {
      const p = new ReplayPlayback(c as ReplayData["captures"][number]);
      p.use(c as ReplayData["captures"][number], false);
      const model = new MaterialModel(p);
      model.multiple = true;
      for (const position of [0.1, 0.4, 0.8]) {
        p.seek(position);
        const s = model.snapshot();
        expect(s.channels).toBeNull();
        expect(s.entries).toBeNull();
        expect(s.slots).toBeNull();
        expect(s.candidates).toBeNull();
        expect(s.answer).toBeNull();
        expect(s.requests).toEqual([]);
      }
    }
  });
  test("the rounding gauge carries scale and rounds exact ties to an even code", () => {
    expect(roundingExample(0.625, 1).stored).toBe(0.5);
    expect(roundingExample(0.75, 1).stored).toBe(1);
    expect(roundingExample(2.5, 1).stored).toBe(2);
    const s = roundingExample(-1.17, 0.5);
    expect(s.stored + s.difference).toBeCloseTo(s.value, 12);
    expect(s.stored).toBe(-1);
  });
});

describe("shared replay controls", () => {
  test("inspection preserves pause and position, with a separate duration for the same story window", () => {
    const p = new ReplayPlayback(
      dataJson.captures[0] as ReplayData["captures"][number],
    );
    p.seek(0.2);
    const position = p.journey.timeline.progress,
      layer = p.journey.timeline.layerIndex;
    p.inspect(layer);
    p.setDuration(25);
    p.show("layer", layer);
    expect(p.playing).toBe(false);
    expect(p.journey.timeline.progress).toBe(position);
    expect(p.seconds).toBe(25);
    p.inspect(layer, "down");
    expect(p.playing).toBe(false);
    const inside = p.journey.timeline.progress;
    p.show("stack", layer);
    expect(p.journey.timeline.progress).toBe(inside);
    expect(p.playing).toBe(false);
    expect(p.seconds).toBe(120);
    p.show("engine", layer);
    expect(p.playing).toBe(false);
    p.watchLayer(layer);
    expect(p.playing).toBe(true);
    expect(p.progress).toBeCloseTo(0);
    p.tick(25);
    expect(p.playing).toBe(false);
    expect(p.progress).toBeCloseTo(1);
  });
  test("story steps and layer markers follow the Gemma operations on each pass", () => {
    const p = new ReplayPlayback(
      dataJson.captures[0] as ReplayData["captures"][number],
    );
    for (const start of [0.11, 0.54, 0.81]) {
      p.seek(start + 1e-7);
      expect(p.layerPosition(0)).toBeCloseTo(start);
      p.seek(p.layerPosition(4));
      p.step(1);
      expect(p.journey.operation.id).toBe("qkv");
      expect(p.journey.timeline.layerIndex).toBe(4);
      p.step(-1);
      expect(p.journey.operation.id).toBe("norm");
      expect(p.playing).toBe(false);
    }
  });
  test("mode switching, duration edits and invalid seeks cannot advance either replay", () => {
    const p = new ReplayPlayback(
      dataJson.captures[0] as ReplayData["captures"][number],
    );
    p.seek(0.3);
    const story = p.journey.timeline.progress;
    for (const capture of dataJson.captures) {
      p.use(capture as ReplayData["captures"][number], false);
      p.seek(0.4);
      const time = p.time;
      p.setDuration(360);
      p.seek(NaN);
      p.tick(5);
      expect(p.time).toBe(time);
      expect(p.playing).toBe(false);
      p.toggle();
      p.tick(36);
      expect(p.progress).toBeCloseTo(0.5);
    }
    p.use(dataJson.captures[0] as ReplayData["captures"][number], true);
    expect(p.journey.timeline.progress).toBe(story);
    expect(p.playing).toBe(false);
  });
  test("parallel worker strokes advance through child intervals and hold throughout recorded waits", () => {
    for (const capture of dataJson.captures) {
      const motion = new RecordedMotion(
        capture as ReplayData["captures"][number],
      );
      motion.workers.forEach((intervals, block) => {
        for (const interval of intervals) {
          const a = interval.start + (interval.end - interval.start) * 0.2,
            b = interval.start + (interval.end - interval.start) * 0.8;
          const first = motion.worker(block, a),
            second = motion.worker(block, b);
          expect(first.active).toBe(true);
          expect(first.step?.id).toBe(interval.step.id);
          expect(second.progress).toBeGreaterThanOrEqual(first.progress);
          if (first.wait) expect(second.progress).toBe(first.progress);
          expect(motion.worker(block, a)).toEqual(first);
        }
        expect(motion.worker(block, capture.end).active).toBe(false);
      });
    }
  });
});
import {
  mergeSpans,
  spanPhase,
  workPhase,
  packetFlight,
  transportPhase,
} from "./flow-motion";
const data = dataJson as ReplayData;
describe("recorded mechanical replay", () => {
  test("recorded fused equipment traverses dependent stages while parallel presses stay together", () => {
    for (const c of data.captures) {
      const at = (role: string, fraction: number) => {
        const step = c.steps.find((s) => s.layer === 10 && s.role === role)!,
          spans = mergeSpans(
            step.observations.map((o) => ({ start: o[1], end: o[2] })),
          );
        let remaining =
          spans.reduce((sum, s) => sum + s.end - s.start, 0) * fraction;
        for (const span of spans) {
          if (remaining < span.end - span.start) return span.start + remaining;
          remaining -= span.end - span.start;
        }
        return step.end;
      };
      for (const [housing, role, names] of [
        [
          "gateup",
          "gate-up:2",
          ["GELU metering cam", "Branch multiplier", "Activation outlet"],
        ],
        [
          "down",
          "main:12",
          [
            "Down combining beam",
            "Post-down sizing jaws",
            "Residual joining beam",
          ],
        ],
        [
          "ple",
          "main:12",
          [
            "Side gate press",
            "Side GELU / multiply",
            "Side output press",
            "Side sizing jaws",
            "Scaled residual join",
          ],
        ],
      ] as const) {
        const parts = workshopParts(c, 10, housing);
        for (let stage = 0; stage < names.length; stage++) {
          const time = at(role, (stage + 0.5) / names.length);
          expect(
            parts
              .filter(
                (p) =>
                  p.role === role &&
                  equipmentMotionState(c, 10, parts, p, time).present,
              )
              .map((p) => p.name),
          ).toEqual([names[stage]]);
        }
      }
      const parts = workshopParts(c, 10, "gateup"),
        time = at("gate-up:1", 0.5);
      expect(
        parts
          .filter(
            (p) =>
              p.role === "gate-up:1" &&
              equipmentMotionState(c, 10, parts, p, time).active,
          )
          .map((p) => p.name),
      ).toEqual(["Gate press", "Up press"]);
    }
  });
  test("each weight copy enters once, travels forward, and exits before being reused", () => {
    for (const count of [1, 3, 4])
      for (let copy = 0; copy < count; copy++) {
        const visible = Array.from({ length: 1001 }, (_, n) =>
          packetFlight(n / 1000, copy, count),
        ).filter((p) => p.visible);
        expect(visible[0].progress).toBeLessThan(0.003);
        expect(visible.at(-1)!.progress).toBeGreaterThan(0.997);
        expect(
          visible.every((p, i) => !i || p.progress >= visible[i - 1].progress),
        ).toBe(true);
        expect(packetFlight(1, copy, count).visible).toBe(false);
      }
  });
  test("recorded travel parks during gaps and resumes forward instead of returning to the inlet", () => {
    const separated = [
      { start: 0, end: 2 },
      { start: 4, end: 6 },
    ];
    expect(spanPhase(separated, 2)).toEqual({
      active: false,
      present: true,
      progress: 0.5,
    });
    expect(spanPhase(separated, 4)).toEqual({
      active: true,
      present: true,
      progress: 0.5,
    });
    for (const c of data.captures)
      for (const step of c.steps.filter((s) => s.kind === "work")) {
        const spans = mergeSpans(
          step.observations.map((o) => ({ start: o[1], end: o[2] })),
        );
        if (!spans.length) continue;
        expect(workPhase(step, spans[0].start).progress).toBe(0);
        expect(workPhase(step, spans.at(-1)!.end)).toEqual({
          active: false,
          present: false,
          progress: 1,
        });
        for (const span of spans) {
          expect(workPhase(step, span.start).active).toBe(true);
          const end = workPhase(step, span.end);
          expect(end.progress).toBeGreaterThan(
            workPhase(step, span.start).progress,
          );
          expect(workPhase(step, span.end).active).toBe(false);
          const next = spans[spans.indexOf(span) + 1];
          if (next)
            expect(workPhase(step, next.start).progress).toBe(end.progress);
        }
      }
  });
  test("an equipment handoff completes the incoming trip before moving through the receiving machine", () => {
    const before = transportPhase(0.4 - 1e-8, true),
      after = transportPhase(0.4, true);
    expect(before.travelling).toBe(true);
    expect(before.travel).toBeGreaterThan(0.999999);
    expect(after.travelling).toBe(false);
    expect(after.machine).toBe(0);
    expect(transportPhase(1, true).machine).toBe(1);
    expect(transportPhase(0.2, false)).toMatchObject({
      travelling: false,
      machine: 0.2,
    });
  });
  test("recorded weight reads move only during consuming work and restore exactly after a reverse seek", () => {
    for (const c of data.captures) {
      const motion = new RecordedMotion(c),
        clock = new ReplayClock(c);
      for (const layer of c.layers)
        for (const part of weightedParts) {
          const step = c.steps.find(
              (s) => s.layer === layer.id && s.role === weightRole(part),
            )!,
            o = step.observations.find((o) => o[2] > o[1])!;
          const first = o[1] + (o[2] - o[1]) * 0.25,
            second = o[1] + (o[2] - o[1]) * 0.75;
          const a = motion.state(layer.id, part, first, true),
            b = motion.state(layer.id, part, second, true);
          expect(a.active).toBe(true);
          expect(b.active).toBe(true);
          expect(b.progress).toBeGreaterThan(a.progress);
          expect(motion.state(layer.id, part, step.end, true).active).toBe(
            false,
          );
          expect(
            motion.reads(layer.id, first).some((s) => s.part === part),
          ).toBe(true);
          clock.seekTime(first);
          const state = motion.state(layer.id, part, clock.time, true);
          clock.setDuration(300);
          clock.tick(5);
          expect(motion.state(layer.id, part, clock.time, true)).toEqual(state);
          clock.seek(1);
          clock.seekTime(first);
          expect(motion.state(layer.id, part, clock.time, true)).toEqual(state);
        }
      for (const layer of [...c.layers, ...c.layers.slice().reverse()]) {
        const time = layer.journeyStart + 0.01,
          focus = motion.focus(time);
        expect(focus.layer).toBe(layer.id);
        expect(focus.step.kind).not.toBe("sentinel");
      }
      expect(motion.reads(0, c.start - 1)).toEqual([]);
      expect(motion.reads(34, c.end)).toEqual([]);
    }
  });
  test("both exported captures preserve all source records and captured topology", async () => {
    const bundleFile = Bun.file(
      new URL(
        "../../../runs/legs/internal-decode-replay/lifecycle-ready/bundle.json",
        import.meta.url,
      ),
    );
    if (!(await bundleFile.exists())) return;
    const bundle = await bundleFile.json();
    expect(data.captures).toHaveLength(bundle.timelines.length);
    for (const raw of bundle.timelines) {
      const { replay } = normalizeInternalCapture(raw),
        topology = admitWorldTopology(replay, raw.captureTopology),
        c = data.captures.find(
          (c) => c.position === Number(raw.launchBinding.position),
        )!;
      expect(c.steps.map((s) => s.id)).toEqual(
        replay.steps.map((s: any) => s.id),
      );
      expect(c.layers.map((l) => String(l.owner))).toEqual(topology.owners);
      expect(c.layers.map((l) => l.journeyStart)).toEqual(
        decodeSequence(replay).layers.map((l: any) => l.start),
      );
      expect(c.layers.map((l) => l.kind)).toEqual(
        replay.layers.map((l: any) => l.kind),
      );
      expect(c.grid).toBe(Number(raw.launchBinding.grid_blocks));
      for (const s of replay.steps) {
        const exported = c.steps.find((x) => x.id === s.id)!;
        expect(exported.observations).toEqual(
          s.observations
            .filter((o: any) => o.participating)
            .map((o: any) => [Number(o.block), o.start, o.end, String(o.sm)]),
        );
      }
    }
  });
  test("duration changes never change capture time or active parallel work; scrubbing is reversible", () => {
    for (const c of data.captures) {
      const clock = new ReplayClock(c);
      let seed = 143;
      for (let i = 0; i < 80; i++) {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        const p = seed / 2 ** 32;
        clock.seek(p);
        const time = clock.time,
          active = activeBlockSteps(c, i % c.grid, time).map((s) => s.id);
        clock.setDuration(1 + i * 41);
        expect(clock.time).toBe(time);
        expect(
          activeBlockSteps(c, i % c.grid, clock.time).map((s) => s.id),
        ).toEqual(active);
        clock.seek(1);
        clock.seek(p);
        expect(clock.time).toBe(time);
      }
      clock.seek(0);
      clock.setDuration(40);
      clock.playing = true;
      clock.tick(10);
      expect(clock.progress).toBe(0.25);
      clock.tick(30);
      expect(clock.time).toBe(c.end);
      expect(clock.playing).toBe(false);
    }
  });
  test("overlapping parent and child observations stay simultaneous and retired markers never activate workers", () => {
    for (const c of data.captures) {
      const child = c.steps.find(
          (s) =>
            s.parent &&
            s.kind === "work" &&
            s.observations.some((o) => o[2] > o[1]),
        )!,
        o = child.observations.find((o) => o[2] > o[1])!,
        t = (o[1] + o[2]) / 2;
      const active = activeBlockSteps(c, o[0], t);
      expect(active.some((s) => s.id === child.id)).toBe(true);
      expect(active.some((s) => s.id === child.parent)).toBe(true);
      expect(active.some((s) => s.kind === "sentinel")).toBe(false);
      expect(
        c.layers.slice(15).every((l) => l.owner === (l.kind === "S" ? 13 : 14)),
      ).toBe(true);
    }
  });
  test("the carrier follows alternating rows and stays continuous at every layer and row boundary", () => {
    for (const capture of data.captures) {
      for (let i = 0; i < capture.layers.length; i++) {
        const layer = capture.layers[i],
          next = capture.layers[i + 1],
          end = next?.journeyStart ?? capture.end;
        const at = (p: number) =>
          stackCarrierAt(
            capture,
            layer.journeyStart + (end - layer.journeyStart) * p,
          )!;
        const direction = Math.floor(i / 5) % 2 ? -1 : 1;
        expect((at(0.4)[0] - at(0.05)[0]) * direction).toBeGreaterThan(0);
        expect(at(0)).toEqual(stackPath(i, next?.id)[0]);
        const before = stackCarrierAt(
            capture,
            end - (end - (next ? layer.end : layer.journeyStart)) * 1e-8,
          )!,
          after = at(1);
        expect(
          Math.hypot(...before.map((v, axis) => v - after[axis])),
        ).toBeLessThan(0.000001);
        if (next) {
          expect(after).toEqual(
            stackPath(next.id, capture.layers[i + 2]?.id)[0],
          );
          if (i % 5 === 4) {
            const turn = stackCarrierAt(
                capture,
                (layer.end + next.journeyStart) / 2,
              )!,
              [x, y] = stackPath(i, next.id)[0];
            expect((turn[0] - x) * direction).toBeGreaterThan(5.8);
            expect(turn[1]).toBeLessThan(y);
            expect(turn[1]).toBeGreaterThan(after[1]);
          }
        }
      }
      const final = stackCarrierAt(capture, capture.end);
      expect(final).toEqual(stackPath(34).at(-1)!);
      expect(
        stackCarrierAt(capture, capture.layers[0].journeyStart - 1),
      ).toBeNull();
      const time = capture.layers[8].journeyStart + 1,
        position = stackCarrierAt(capture, time);
      stackCarrierAt(capture, capture.end);
      expect(stackCarrierAt(capture, time)).toEqual(position);
    }
  });
  test("the cart stays inside its layer until recorded work finishes, then crosses the connector", () => {
    for (const capture of data.captures)
      for (const layer of capture.layers) {
        const next = capture.layers[layer.id + 1],
          path = stackPath(layer.id, next?.id),
          direction = Math.floor(layer.id / 5) % 2 ? -1 : 1;
        const reads = capture.steps.filter(
          (s) =>
            s.layer === layer.id &&
            weightedParts.some((part) => weightRole(part) === s.role),
        );
        for (const step of reads)
          for (const observation of step.observations) {
            const time = observation[2] - 0.001;
            if (time < layer.journeyStart) continue;
            const cart = stackCarrierAt(capture, time)!;
            expect((cart[0] - path[0][0]) * direction).toBeGreaterThanOrEqual(
              0,
            );
            expect((cart[0] - path[1][0]) * direction).toBeLessThanOrEqual(0);
            expect(cart[1]).toBe(path[0][1]);
          }
        expect(stackCarrierAt(capture, layer.end)).toEqual(path[1]);
        if (next)
          expect(stackCarrierAt(capture, next.journeyStart)).toEqual(
            stackPath(next.id)[0],
          );
        // In the authored story, the readiness latch follows the last weight-using step.
        for (let tray = 0; tray < 7; tray++) {
          const at = (progress: number) =>
              stackCarrierProgress(
                layer.id,
                next?.id,
                progress,
                0.97,
                tray * 0.035,
              ),
            beforeRelease = at(0.969999);
          expect(
            (beforeRelease[0] - path[1][0]) * direction,
          ).toBeLessThanOrEqual(0);
          expect(beforeRelease[1]).toBe(path[0][1]);
          expect(at(1)).toEqual(path.at(-1)!);
          if (next) {
            const before = at(1 - 1e-10),
              after = stackCarrierProgress(
                next.id,
                capture.layers[next.id + 1]?.id,
                0,
                0.97,
                tray * 0.035,
              );
            expect(
              Math.hypot(...before.map((v, axis) => v - after[axis])),
            ).toBeLessThan(0.000001);
          }
        }
      }
  });
  test("every housing opens into supplied equipment, shared KV omits duplicate presses, and FFN branches rejoin after GELU", () => {
    for (const c of data.captures)
      for (const layer of [0, 4, 15, 19, 34]) {
        const all: Equipment[] = [];
        const collect = (parts: Equipment[]) => {
          for (const p of parts) {
            all.push(p);
            collect(p.children);
          }
        };
        for (const m of mechanisms) {
          const parts = workshopParts(c, layer, m.id);
          expect(parts.length).toBeGreaterThan(1);
          collect(parts);
        }
        expect(all.every((p) => p.input && p.uses && p.output && p.does)).toBe(
          true,
        );
        expect(
          all.every(
            (p) =>
              !p.role ||
              c.steps.some(
                (s) =>
                  s.layer === layer &&
                  s.role === p.role &&
                  s.kind !== "sentinel",
              ),
          ),
        ).toBe(true);
        const q = workshopParts(c, layer, "qkv");
        expect(q.some((p) => p.name === "K press")).toBe(layer < 15);
        expect(q.some((p) => p.name === "V press")).toBe(layer < 15);
        const cache = workshopParts(c, layer, "cache");
        if (layer >= 15)
          expect(cache[0].input).toContain(
            `layer ${c.layers[layer].kind === "F" ? 15 : 14}`,
          );
        const ffn = workshopParts(c, layer, "gateup"),
          edges = equipmentEdges(ffn).map(([a, b]) =>
            [a.name, b.name].join(" → "),
          );
        expect(ffn[1].stage).toBe(ffn[2].stage);
        expect(ffn[1].lane).not.toBe(ffn[2].lane);
        expect(edges).toContain("Gate press → GELU metering cam");
        expect(edges).not.toContain("Up press → GELU metering cam");
        expect(edges).toContain("Up press → Branch multiplier");
        expect(edges).toContain("GELU metering cam → Branch multiplier");
        const attention = workshopParts(c, layer, "attention");
        expect(
          attention.filter((p) => p.name.startsWith("Probability / V mill")),
        ).toHaveLength(c.layers[layer].kind === "F" ? 2 : 1);
        const mill = attention.find((p) =>
          p.name.startsWith("Probability / V mill"),
        )!;
        expect(mill.children.map((p) => p.name)).toEqual([
          "Maximum selector",
          "Exponential cam",
          "Value product press",
          "Weight collecting beam",
          "Gathering cradle",
        ]);
        expect(
          c.steps
            .filter((s) => s.layer === layer && s.kind !== "sentinel")
            .every((s) =>
              mechanisms.some((m) => m.id === mechanismForRole(s.role)),
            ),
        ).toBe(true);
      }
  });
  test("equipment uses captured activity; untimed storage stays still and the hand crank leaves the recording untouched", () => {
    for (const c of data.captures) {
      const clock = new ReplayClock(c),
        weights = workshopParts(c, 0, "weights")[0],
        gate = workshopParts(c, 0, "latch").at(-1)!;
      const step = c.steps.find((s) => s.layer === 0 && s.role === gate.role)!,
        observation = step.observations.find((o) => o[2] > o[1])!;
      clock.seekTime((observation[1] + observation[2]) / 2);
      const time = clock.time;
      expect(equipmentState(c, 0, gate, time).active).toBe(true);
      expect(equipmentState(c, 0, gate, step.end).done).toBe(true);
      expect(equipmentState(c, 0, weights, time).progress).toBe(0);
      expect(equipmentState(c, 0, weights, time).active).toBe(false);
      expect(equipmentState(c, 0, weights, time, 0.5).evidence).toBe(
        "Hand-cranked illustration",
      );
      expect(equipmentState(c, 0, weights, time, 0.5).progress).toBe(0.5);
      expect(clock.time).toBe(time);
      const cam = workshopParts(c, 0, "gateup").find(
        (p) => p.name === "GELU metering cam",
      )!;
      expect(cam.children.every((p) => p.shared && p.role === cam.role)).toBe(
        true,
      );
    }
  });
});
