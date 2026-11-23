import type { Anatomy } from "./replay-model";
import type { ReplayPlayback } from "./replay-playback";
import { RecordedMotion, workerActivity } from "./recorded-motion";
import { TileFlow, tileSnapshotFor } from "./tile-flow";

export type WarpRole = "controller" | "loader" | "consumer";
export type WarpGroup = {
  key: string;
  role: WarpRole;
  name: string;
  lanes: number[];
};

/** Counts come from the separate anatomy; this display order is not a CUDA warp-ID binding. */
export function warpGroups(roles: Anatomy["roles"]): WarpGroup[] {
  return (["controller", "loader", "consumer"] as const).flatMap((role) => {
    const count = Number(roles[role]);
    if (!Number.isInteger(count) || count < 0 || count > 32)
      throw new Error(
        `The source snapshot has an unsupported ${role} warp count.`,
      );
    return Array.from({ length: count }, (_, index) => ({
      key: `${role}-${index}`,
      role,
      name: `${role[0].toUpperCase()}${role.slice(1)}${count > 1 ? " " + (index + 1) : ""}`,
      lanes: Array.from({ length: 32 }, (_, lane) => lane),
    }));
  });
}

/** Inputs: shared playback and source roles. Output: representative hardware poses, never inferred telemetry. */
export class HardwareModel {
  readonly groups: WarpGroup[];
  selectedWarp = 0;
  selectedLane = 0;
  depth: "sm" | "warp" | "tile" = "sm";
  private recorded: RecordedMotion;
  constructor(
    public playback: ReplayPlayback,
    public anatomy: Anatomy,
    readonly tile = new TileFlow(),
  ) {
    this.groups = warpGroups(anatomy.roles);
    if (!this.groups.length)
      throw new Error("The source snapshot contains no warp roles.");
    this.recorded = new RecordedMotion(playback.clock.capture);
  }
  snapshot(block: number) {
    const p = this.playback;
    if (this.recorded.capture !== p.clock.capture)
      this.recorded = new RecordedMotion(p.clock.capture);
    const selectedBlock = Math.max(
      0,
      Math.min(p.clock.capture.grid - 1, block),
    );
    const activity = workerActivity(
      p.illustrated ? p.journey.timeline : null,
      this.recorded,
      selectedBlock,
      p.time,
    );
    const warp =
      this.groups[Math.min(this.selectedWarp, this.groups.length - 1)];
    const lane = Math.max(0, Math.min(31, this.selectedLane));
    const moving = activity.active && !activity.wait;
    return {
      block: selectedBlock,
      warp,
      lane,
      activity,
      status: activity.active
        ? activity.wait
          ? "Waiting"
          : "Working"
        : "Idle",
      phase: moving ? activity.progress : 0,
      layer: p.illustrated
        ? p.journey.timeline.layerIndex
        : (activity.step?.layer ?? null),
      physicalSM: null,
      otherResidentBlocks: null,
      warpTiming: null,
      activeLaneMask: null,
      laneValues: null,
      blockThreadBinding: null,
      illustrated: p.illustrated,
    };
  }
  tileSnapshot(block: number) {
    this.snapshot(block);
    return tileSnapshotFor(this.playback, this.tile, this.recorded, block);
  }
}
