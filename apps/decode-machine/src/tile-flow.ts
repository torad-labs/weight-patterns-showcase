import { clamp } from "./connected-journey";
import type { WorkerState } from "./recorded-motion";
import { workerActivity, type RecordedMotion } from "./recorded-motion";
import type { ReplayPlayback } from "./replay-playback";
import { journeyPartState } from "./connected-journey";

export const tileStations = [
  [
    "storage",
    "Weight reservoir",
    "The original weights stay in device memory. Only a read copy travels.",
  ],
  [
    "cache",
    "L2 cache · shared cupboard",
    "All SMs share this cache. A hit supplies a nearby copy; a miss requests one from VRAM.",
  ],
  [
    "loader",
    "TMA · automatic loading crane",
    "The TMA crane fills the next tray while a crew uses the previous one.",
  ],
  [
    "staging",
    "Reusable shared trays",
    "Fill, raise the ready peg, use, release. Only then may a tray be refilled.",
  ],
  [
    "layout",
    "Layout sorting table",
    "The same four marked values move from a row into a tile. Their identities and values stay unchanged. This rearrangement is different from RoPE.",
  ],
  [
    "registers",
    "Private operand holders",
    "Each lane has private registers. Cooperative matrix instructions distribute operand fragments across lanes; one holder is not a whole matrix.",
  ],
  [
    "scheduler",
    "Ready-crew clutch",
    "Hardware chooses a ready warp instruction for an available execution unit. A waiting crew cannot engage. This clutch is not the megakernel’s controller warp; the controller is scheduled too.",
  ],
  [
    "compute",
    "Shared execution tools",
    "A matrix press, adding lever, maximum selector and exponential cam handle different instructions. Pick one below.",
  ],
  [
    "encode",
    "Output sizing jaws",
    "The tensor press keeps its accumulating result inside the calculation. The outgoing value is encoded into narrow codes and scale tags before storage. No wide stored result is implied.",
  ],
  [
    "handoff",
    "Completion counter & latch",
    "Each result raises a completion peg. All three pegs must rise before the latch opens.",
  ],
  [
    "capacity",
    "Finite workspace",
    "An SM has finite register and shared-memory space. More space per block can leave fewer blocks resident. These drawers use teaching units, not this GPU’s capacities or measured occupancy.",
  ],
] as const;
export type TileStation = (typeof tileStations)[number][0];
export type ToolKind = "matrix" | "arithmetic" | "compare" | "exponential";
export type Pressure =
  "normal" | "supply" | "execution" | "registers" | "shared";
export type SlotState = "empty" | "filling" | "ready" | "in use" | "released";
export type TileWork = {
  id: number;
  slot: number;
  load: number;
  loaded: number;
  start: number;
  end: number;
  release: number;
};
export const tileSchedule: TileWork[] = Array.from({ length: 4 }, (_, id) => ({
  id,
  slot: id % 3,
  load: 0.02 + id * 0.14,
  loaded: 0.14 + id * 0.14,
  start: 0.22 + id * 0.14,
  end: 0.34 + id * 0.14,
  release: 0.36 + id * 0.14,
}));

/** Teaching route phases, not measured transfer latencies. A miss returns a request before the data travels. */
export function deliveryPhase(progress: number, cacheHit: boolean) {
  const p = clamp(progress);
  if (!cacheHit && p < 0.2)
    return { stage: "request" as const, progress: p / 0.2 };
  if (!cacheHit && p < 0.5)
    return { stage: "refill" as const, progress: (p - 0.2) / 0.3 };
  return {
    stage: "deliver" as const,
    progress: cacheHit ? p : (p - 0.5) / 0.5,
  };
}

export function tileSnapshotFor(
  playback: ReplayPlayback,
  flow: TileFlow,
  recorded: RecordedMotion,
  block: number,
) {
  const p = playback;
  const activity = workerActivity(
    p.illustrated ? p.journey.timeline : null,
    recorded,
    block,
    p.time,
  );
  const scope = p.journey.scope;
  const phase = p.illustrated
    ? journeyPartState(
        p.journey.timeline,
        scope?.part ? scope.layer : p.journey.timeline.layerIndex,
        scope?.part ?? p.journey.operation.id,
      )
    : activity;
  return flow.snapshot(
    { ...activity, progress: phase.progress },
    p.illustrated,
  );
}
const within = (p: number, start: number, end: number) => p >= start && p < end;

export class TileFlow {
  station: TileStation = "staging";
  cacheHit = false;
  tool: ToolKind = "matrix";
  pressure: Pressure = "normal";

  /** Pure pose derived from the shared work interval; examples never become capture telemetry. */
  snapshot(activity: WorkerState, illustrated: boolean) {
    const active = activity.active && !activity.wait;
    const p = active ? clamp(activity.progress) : 0;
    // Pause the teaching mechanism during a named illustrative bottleneck, then catch up.
    const holdAt = this.pressure === "supply" ? 0.1 : 0.23;
    const hold = this.pressure === "supply" || this.pressure === "execution";
    const held = active && hold && within(p, holdAt, holdAt + 0.13);
    const t = hold
      ? p < holdAt
        ? p
        : p < holdAt + 0.13
          ? holdAt
          : holdAt + ((p - holdAt - 0.13) * (1 - holdAt)) / (1 - holdAt - 0.13)
      : p;
    const loading = active
      ? tileSchedule.find((x) => within(t, x.load, x.loaded))
      : undefined;
    const using = active
      ? tileSchedule.find((x) => within(t, x.start, x.end))
      : undefined;
    const latest = [...tileSchedule].reverse().find((x) => t >= x.loaded);
    const sorting = active
      ? tileSchedule.find((x) => within(t, x.loaded, x.start))
      : undefined;
    const slots = Array.from({ length: 3 }, (_, slot) => {
      const work = [...tileSchedule]
        .reverse()
        .find((x) => x.slot === slot && t >= x.load);
      const state: SlotState =
        !active || !work
          ? "empty"
          : t < work.loaded
            ? "filling"
            : t < work.start
              ? "ready"
              : t < work.end
                ? "in use"
                : t < work.release
                  ? "released"
                  : "empty";
      return { slot, tile: state === "empty" ? null : work!.id, state };
    });
    const contributions = [0.84, 0.88, 0.92].map((at) => active && t >= at);
    const completed = active
      ? tileSchedule.filter((x) => t >= x.end).length
      : 0;
    const capacity = {
      total: 8,
      registersPerBlock: this.pressure === "registers" ? 5 : 3,
      sharedPerBlock: this.pressure === "shared" ? 5 : 3,
    };
    const resident = Math.min(
      Math.floor(capacity.total / capacity.registersPerBlock),
      Math.floor(capacity.total / capacity.sharedPerBlock),
    );
    const issued =
      !active || held
        ? null
        : using
          ? "consumer"
          : loading
            ? "loader"
            : t < 0.84
              ? "controller"
              : null;
    const gateOpen = contributions.every(Boolean);
    const loadProgress = loading
      ? clamp((t - loading.load) / (loading.loaded - loading.load))
      : 0;
    return {
      active,
      waiting: activity.active && activity.wait,
      progress: t,
      held,
      loading,
      using,
      sorting,
      latest,
      slots,
      completed,
      loadProgress,
      delivery: loading ? deliveryPhase(loadProgress, this.cacheHit) : null,
      computeProgress: using
        ? clamp((t - using.start) / (using.end - using.start))
        : 0,
      sortProgress: sorting
        ? clamp((t - sorting.loaded) / (sorting.start - sorting.loaded))
        : 0,
      issued,
      contributions,
      gateOpen,
      outputProgress: clamp((t - 0.92) / 0.08),
      encoded: active && t >= 0.82,
      encoding: active && within(t, 0.78, 0.82),
      capacity,
      resident,
      queued: 3 - resident,
      copySource: this.cacheHit ? "L2" : "VRAM",
      storedWeights: [0, 1, 2, 3],
      valueOrder: [0, 1, 2, 3],
      cause: !active
        ? activity.wait
          ? "Recorded dependency wait · cause unknown"
          : "No active block work"
        : held
          ? this.pressure === "supply"
            ? "Teaching example: supply road occupied"
            : "Teaching example: execution unit occupied"
          : using
            ? "A ready consumer takes a turn"
            : loading
              ? "Loading overlaps other ready work"
              : gateOpen
                ? "All results published · receiving latch open"
                : "Waiting for the next required contribution",
      transferTiming: null,
      issuedWarp: null,
      measuredOccupancy: null,
      evidence: illustrated
        ? "Authored tile choreography on the shared story clock"
        : "Illustrative substeps inside a recorded block interval · transfer and warp timing unknown",
    };
  }
}

export type TilePose = ReturnType<TileFlow["snapshot"]>;
