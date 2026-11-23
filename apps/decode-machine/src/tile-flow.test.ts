import { describe, test, expect } from "bun:test";
import {
  TileFlow,
  tileSchedule,
  deliveryPhase,
  tileSnapshotFor,
  type Pressure,
} from "./tile-flow";
import { RecordedMotion } from "./recorded-motion";
import { ReplayPlayback } from "./replay-playback";
import { HardwareModel } from "./hardware-model";
import { inspectionDirection, inspectionEase } from "./inspection-motion";
import data from "../public/replay/gemma4-e2b.json";
import anatomy from "../public/replay/anatomy.json";
import type { Anatomy, Capture } from "./replay-model";

const at = (m: TileFlow, progress: number) =>
  m.snapshot({ active: true, wait: false, progress }, true);
describe("a connected tile handoff", () => {
  test("cache misses request a refill before delivery; hits skip the reservoir", () => {
    expect(deliveryPhase(0, false)).toEqual({ stage: "request", progress: 0 });
    expect(deliveryPhase(0.2, false)).toEqual({ stage: "refill", progress: 0 });
    expect(deliveryPhase(0.5, false)).toEqual({
      stage: "deliver",
      progress: 0,
    });
    expect(deliveryPhase(1, false)).toEqual({ stage: "deliver", progress: 1 });
    for (let i = 0; i <= 100; i++)
      expect(deliveryPhase(i / 100, true)).toEqual({
        stage: "deliver",
        progress: i / 100,
      });
  });
  test("floor and SM inspection share cache choices and the same delivery at every position", () => {
    const p = new ReplayPlayback(data.captures[0] as Capture),
      flow = new TileFlow();
    const m = new HardwareModel(p, anatomy as Anatomy, flow),
      recorded = new RecordedMotion(p.clock.capture);
    p.inspect(0, "qkv");
    for (const hit of [true, false]) {
      flow.cacheHit = hit;
      for (let i = 0; i <= 100; i++) {
        p.seek(i / 100);
        expect(m.tileSnapshot(0)).toEqual(
          tileSnapshotFor(p, flow, recorded, 0),
        );
      }
    }
    const before = p.time;
    m.tile.station = "cache";
    m.depth = "tile";
    expect(flow.station).toBe("cache");
    expect(p.time).toBe(before);
    expect(p.playing).toBe(false);
  });
  test("the end of a scoped handoff retains the released result", () => {
    const p = new ReplayPlayback(data.captures[0] as Capture);
    const m = new HardwareModel(p, anatomy as Anatomy);
    p.inspect(0, "qkv");
    p.seek(1);
    const s = m.tileSnapshot(0);
    expect(s.completed).toBe(4);
    expect(s.gateOpen).toBe(true);
    expect(s.outputProgress).toBeCloseTo(1);
  });
  test("loading overlaps computing without overwriting a live shared tray", () => {
    const m = new TileFlow();
    let overlap = false;
    for (let i = 0; i <= 1000; i++) {
      const s = at(m, i / 1000);
      if (s.loading && s.using) {
        overlap = true;
        expect(s.loading.slot).not.toBe(s.using.slot);
      }
      if (s.using)
        expect(s.slots[s.using.slot]).toMatchObject({
          tile: s.using.id,
          state: "in use",
        });
      if (s.loading)
        expect(s.slots[s.loading.slot]).toMatchObject({
          tile: s.loading.id,
          state: "filling",
        });
      expect(s.storedWeights).toEqual([0, 1, 2, 3]);
      expect(s.valueOrder).toEqual([0, 1, 2, 3]);
      expect(s.gateOpen).toBe(s.contributions.every(Boolean));
      if (s.gateOpen) {
        expect(s.encoded).toBe(true);
        expect(s.completed).toBe(4);
      }
    }
    expect(overlap).toBe(true);
    expect(at(m, tileSchedule[3].load + 0.01).slots[0].tile).toBe(3);
    expect(at(m, 0.9).gateOpen).toBe(false);
    expect(at(m, 0.94).gateOpen).toBe(true);
  });
  test("cache, supply and resource examples affect mechanisms without inventing telemetry", () => {
    const m = new TileFlow();
    m.cacheHit = true;
    expect(at(m, 0.1).copySource).toBe("L2");
    for (const pressure of [
      "normal",
      "supply",
      "execution",
      "registers",
      "shared",
    ] as Pressure[]) {
      m.pressure = pressure;
      for (let i = 0; i <= 100; i++) {
        const s = at(m, i / 100);
        expect(s.resident * s.capacity.registersPerBlock).toBeLessThanOrEqual(
          8,
        );
        expect(s.resident * s.capacity.sharedPerBlock).toBeLessThanOrEqual(8);
        expect(s.transferTiming).toBeNull();
        expect(s.issuedWarp).toBeNull();
        expect(s.measuredOccupancy).toBeNull();
        if (s.held) expect(s.issued).toBeNull();
      }
      expect(at(m, 0.99).gateOpen).toBe(true);
    }
    m.pressure = "registers";
    expect(at(m, 0.4).resident).toBe(1);
    m.pressure = "normal";
    expect(at(m, 0.4).resident).toBe(2);
  });
  test("both recordings and the authored story reconstruct the cutaway through the real shared playback", () => {
    for (const capture of data.captures as Capture[])
      for (const illustrated of [true, false]) {
        const p = new ReplayPlayback(capture);
        p.use(capture, illustrated);
        const m = new HardwareModel(p, anatomy as Anatomy);
        p.seek(0.31);
        const original = m.tileSnapshot(0),
          time = p.time;
        m.depth = "tile";
        p.setDuration(300);
        expect(m.tileSnapshot(0)).toEqual(original);
        expect(p.time).toBe(time);
        expect(p.playing).toBe(false);
        p.seek(0.6);
        p.seek(0.31);
        expect(m.tileSnapshot(0)).toEqual(original);
        for (let i = 0; i < 100; i++) {
          p.seek(i / 100);
          const s = m.snapshot(0),
            tile = m.tileSnapshot(0);
          if (s.status !== "Working") {
            expect(tile.issued).toBeNull();
            expect(tile.loading).toBeUndefined();
            expect(tile.gateOpen).toBe(false);
          }
        }
      }
  });
  test("inspection depth is reversible and easing has exact endpoints", () => {
    expect(inspectionDirection("engine", "hardware")).toBe("in");
    expect(inspectionDirection("hardware", "stack")).toBe("in");
    expect(inspectionDirection("stack", "layer")).toBe("in");
    expect(inspectionDirection("layer", "internals")).toBe("in");
    expect(inspectionDirection("internals", "layer")).toBe("out");
    expect(inspectionDirection("hardware", "engine")).toBe("out");
    expect(inspectionEase(0)).toBe(0);
    expect(inspectionEase(1)).toBe(1);
  });
});
