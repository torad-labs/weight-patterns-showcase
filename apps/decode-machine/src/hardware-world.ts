import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { MechanicalWorld } from "./replay-world";
import { HardwareModel } from "./hardware-model";
import { pressStroke } from "./machine-motion";
import { TileWorld } from "./tile-world";

type Point = [number, number, number];
const point = (x: number, y: number, z = 0) => new Vector3(-x, z, -y);
export const warpLanePosition = (lane: number, expanded: boolean): Point =>
  expanded
    ? [-5.25 + (lane % 8) * 1.5, 2.4 - Math.floor(lane / 8) * 1.6, 0.6]
    : [-5.7 + (lane % 16) * 0.76, lane < 16 ? 0.3 : -0.3, 0.6];

/** Representative hardware around the selected logical block. No physical placement is inferred. */
export class HardwareWorld {
  readonly groups: {
    root: TransformNode;
    bed: AbstractMesh;
    expandedBed: AbstractMesh;
    lanes: { root: TransformNode; finger: AbstractMesh }[];
    arm: TransformNode;
    instruction: TransformNode;
  }[] = [];
  readonly shell: TransformNode;
  readonly marker: AbstractMesh;
  readonly compute: AbstractMesh;
  readonly feed: TransformNode;
  readonly label: HTMLButtonElement;
  readonly tile: TileWorld;
  readonly tmaFork: TransformNode;
  readonly tmaCopy: TransformNode;
  onWarp: (warp: number) => void = () => {};
  onLane: (lane: number) => void = () => {};
  private depth = "";
  constructor(
    public host: MechanicalWorld,
    public model: HardwareModel,
  ) {
    const w = host,
      g = w.groups.hardware;
    this.tile = new TileWorld(host, model.tile);
    const root = (name: string, at: Point, parent = g) => {
      const n = new TransformNode(name, w.scene);
      n.parent = parent;
      n.position = point(...at);
      return n;
    };
    this.shell = root("representative SM housing", [0, 0, 0]);
    const cache = root(
      "same shared L2 outside the SM",
      [-20, 0, 0],
      this.shell,
    );
    cache.metadata = { supplyStation: "cache" };
    w.box("shared L2 cupboard", [0, 0, 0.3], [5, 8, 0.5], "paper", cache);
    for (let i = 0; i < 3; i++) {
      w.box("cache shelf", [0, 2 - i * 2, 0.7], [4.4, 1.5, 0.3], "ink", cache);
      for (let j = 0; j < 3; j++)
        w.ball(
          "retained cached copy",
          [-1.2 + j * 1.2, 2 - i * 2, 1.1],
          0.55,
          cache,
        );
    }
    const tma = root(
      "TMA automatic loading crane",
      [-12.4, -6.3, 0],
      this.shell,
    );
    tma.metadata = { supplyStation: "loader" };
    for (const x of [-2, 2])
      w.cyl("TMA gantry post", [x, 0, 1.6], 0.2, 3.2, "ink", tma);
    w.box("TMA gantry", [0, 0, 3.1], [5, 0.35, 0.3], "steel", tma);
    this.tmaFork = root("TMA moving fork", [0, 0, 2.2], tma);
    w.box(
      "TMA fork crossbar",
      [0, 0, 0],
      [1.5, 0.2, 0.2],
      "brass",
      this.tmaFork,
    );
    for (const x of [-0.6, 0.6])
      w.box(
        "TMA fork tine",
        [x, 0.4, -0.3],
        [0.12, 1, 0.6],
        "brass",
        this.tmaFork,
      );
    this.tmaCopy = root("TMA delivery copy", [-20, 0, 1.4], this.shell);
    for (let j = 0; j < 3; j++)
      w.ball("transferred operand", [-0.5 + j * 0.5, 0, 0], 0.4, this.tmaCopy);
    w.pipe(
      "L2 to TMA and shared trays",
      [
        [-20, 0, 1],
        [-20, -6.3, 1],
        [-12.4, -6.3, 1],
        [-12.4, -1, 1],
      ],
      "brass",
      this.shell,
      0.12,
    );
    for (const [name, station, p] of [
      ["L2 cache · shared", "cache", [-20, 5.4, 1]],
      ["TMA loading crane", "loader", [-12.4, -9, 1]],
    ] as const) {
      const label = w.label(name, [...p], "hardware", () =>
        w.onSupply(station, point(p[0], p[1], p[2])),
      );
      label.dataset.hardwareDepth = "sm";
    }
    w.box("SM foundation", [0, 0, 0], [31, 20, 0.18], "base", this.shell);
    for (const x of [-15, 15])
      w.box(
        "SM housing side rail",
        [x, 0, 0.65],
        [0.24, 20, 1.2],
        "steel",
        this.shell,
      );
    for (const y of [-9.8, 9.8])
      w.box(
        "SM housing end rail",
        [0, y, 0.65],
        [30, 0.24, 1.2],
        "steel",
        this.shell,
      );
    w.box(
      "selected block workspace",
      [0, 0, 0.22],
      [18.3, 15.6, 0.2],
      "paper",
      this.shell,
    );
    w.pipe(
      "block boundary",
      [
        [-9, -7.8, 0.4],
        [-9, 7.8, 0.4],
        [9, 7.8, 0.4],
        [9, -7.8, 0.4],
        [-9, -7.8, 0.4],
      ],
      "dark",
      this.shell,
      0.07,
    );
    w.box(
      "block shared staging rack",
      [-12.4, -1, 0.55],
      [3, 9, 0.5],
      "ink",
      this.shell,
    );
    for (let i = 0; i < 8; i++) {
      w.box(
        "shared memory shelf",
        [-12.4, 2.6 - i, 0.95],
        [2.5, 0.65, 0.18],
        "steel",
        this.shell,
      );
      if (i < 3)
        for (let j = 0; j < 3; j++)
          w.ball(
            "retained illustrative operand",
            [-13.1 + j * 0.7, 2.6 - i, 1.18],
            0.32,
            this.shell,
          );
    }
    w.cyl(
      "instruction issue cam",
      [-12.4, 6.3, 0.75],
      2.0,
      0.35,
      "brass",
      this.shell,
    );
    w.pipe(
      "shared instruction issue rail",
      [
        [-12.4, 6.3, 0.7],
        [-10.1, 6.3, 0.7],
        [-10.1, -6.3, 0.7],
      ],
      "green",
      this.shell,
      0.09,
    );
    const computeRoot = root("shared execution hardware", [12.4, 0, 0]);
    w.box(
      "shared compute bed",
      [0, 0, 0.35],
      [3.2, 8, 0.45],
      "ink",
      computeRoot,
    );
    for (const x of [-1.2, 1.2])
      for (const y of [-3, 3])
        w.cyl("compute column", [x, y, 1.3], 0.25, 2, "steel", computeRoot);
    this.compute = w.box(
      "shared tensor arithmetic press",
      [0, 0, 2.2],
      [2.8, 6.5, 0.35],
      "steel",
      computeRoot,
    );
    w.pipe(
      "operands to shared execution",
      [
        [-12.4, -6.2, 0.8],
        [-12.4, -8.7, 0.8],
        [12.4, -8.7, 0.8],
        [12.4, -4, 0.8],
      ],
      "brass",
      this.shell,
      0.1,
    );
    this.feed = root("illustrative shared operand read", [-12.4, -8.7, 1]);
    for (let j = 0; j < 3; j++)
      w.ball("operand read copy", [-0.45 + j * 0.45, 0, 0], 0.4, this.feed);
    for (const [i, warp] of model.groups.entries()) {
      const y = this.row(i),
        n = root(warp.name + " warp group", [0, y, 0.4]);
      n.metadata = { hardwareWarp: i };
      const bed = w.box(
        "warp group holder",
        [0, 0, 0],
        [14, 1.45, 0.2],
        "steel",
        n,
        true,
      );
      const expandedBed = w.box(
        "expanded lane holder",
        [0, 0, 0],
        [14, 7.5, 0.2],
        "steel",
        n,
      );
      const arm = root("one group instruction lever", [-7.5, 0, 0.5], n);
      const instruction = root("shared instruction connection", [0, 0, 0], n);
      w.pipe(
        "one warp instruction rail",
        [
          [-6.6, 2.4, 0.35],
          [-6.6, -2.4, 0.35],
        ],
        "ink",
        instruction,
        0.05,
      );
      for (let row = 0; row < 4; row++)
        w.pipe(
          "instruction to logical row",
          [
            [-6.6, 2.4 - row * 1.6, 0.35],
            [-5.7, 2.4 - row * 1.6, 0.35],
          ],
          "ink",
          instruction,
          0.05,
        );
      if (warp.role === "controller")
        w.cyl("controller cam", [0, 0, 0], 0.8, 0.25, "brass", arm);
      else if (warp.role === "loader") {
        w.box("loader crossbar", [0, 0, 0], [0.15, 1, 0.2], "brass", arm);
        for (const y of [-0.35, 0.35])
          w.box("loader tine", [0.3, y, 0], [0.7, 0.12, 0.15], "brass", arm);
      } else
        w.box("consumer lever", [0.25, 0, 0], [1, 0.16, 0.2], "brass", arm);
      const lanes = warp.lanes.map((lane) => {
        const r = root(
          `${warp.key} lane ${lane}`,
          warpLanePosition(lane, false),
          n,
        );
        r.metadata = { hardwareLane: lane, hardwareWarp: i };
        w.box(
          "lane private register holder",
          [0, -0.16, -0.2],
          [0.38, 0.28, 0.18],
          "paper",
          r,
        );
        w.cyl("lane stem", [0, 0, 0], 0.11, 0.65, "ink", r);
        const finger = w.box(
          "logical lane finger",
          [0, 0, 0.45],
          [0.36, 0.3, 0.18],
          "teal",
          r,
          true,
        );
        return { root: r, finger };
      });
      w.pipe(
        "issue path to warp " + i,
        [
          [-10.1, y, 0.7],
          [-7.5, y, 0.7],
        ],
        "green",
        this.shell,
        0.06,
      );
      w.pipe(
        "warp uses shared execution " + i,
        [
          [7.5, y, 0.7],
          [10, y, 0.7],
          [10, 0, 0.7],
          [12.4, 0, 0.7],
        ],
        "teal",
        this.shell,
        0.06,
      );
      this.groups.push({ root: n, bed, expandedBed, lanes, arm, instruction });
      const label = w.label(warp.name, [-7.6, y + 0.6, 1.35], "hardware", () =>
        this.onWarp(i),
      );
      label.dataset.hardwareDepth = "sm";
    }
    this.marker = w.box(
      "selected logical lane",
      [0, 0, 0.25],
      [0.12, 1.1, 0.12],
      "ink",
      g,
    );
    this.label = w.label("Lane 0", [0, 0, 1], "hardware", () =>
      this.onLane(model.selectedLane),
    );
    this.label.dataset.hardwareDepth = "warp";
    for (let lane = 0; lane < 32; lane++) {
      const [x, y] = warpLanePosition(lane, true);
      const number = w.label(
        String(lane),
        [x, y - 0.75, 0.65],
        "hardware",
        () => this.onLane(lane),
      );
      number.dataset.hardwareDepth = "warp";
      number.classList.add("hardware-lane-number");
    }
    for (const [name, at] of [
      ["SM housing · representative", [0, 10.6, 1]],
      ["Shared trays", [-12.4, 4.2, 1]],
      ["Issue instructions", [-12.4, 8.4, 1]],
      ["Shared execution", [12.4, 5.5, 1]],
    ] as [string, Point][]) {
      const el = w.label(name, at, "hardware", () => {});
      el.dataset.hardwareDepth = "sm";
      el.disabled = true;
    }
  }
  row(i: number) {
    return ((this.model.groups.length - 1) / 2) * 1.8 - i * 1.8;
  }
  get width() {
    return this.model.depth === "tile"
      ? 44
      : this.model.depth === "sm"
        ? 47
        : 16;
  }
  get depthSize() {
    return this.model.depth === "tile"
      ? 30
      : this.model.depth === "sm"
        ? 23
        : 9;
  }
  update() {
    const w = this.host;
    if (w.view !== "hardware") return;
    const m = this.model,
      s = m.snapshot(w.selectedBlock),
      expanded = m.depth === "warp",
      tile = m.depth === "tile",
      flow = m.tileSnapshot(w.selectedBlock),
      stroke = pressStroke(flow.computeProgress);
    this.tile.root.setEnabled(tile);
    if (tile) this.tile.update(flow);
    this.shell.setEnabled(!expanded && !tile);
    const delivery = flow.delivery;
    const transfer = delivery?.stage === "deliver";
    this.tmaFork.position.x = transfer ? 2 - delivery.progress * 4 : 0;
    this.tmaFork.position.y =
      2.2 - (transfer ? Math.sin(delivery.progress * Math.PI) * 0.5 : 0);
    this.tmaCopy.setEnabled(!expanded && !tile && transfer);
    if (transfer) {
      const p = delivery.progress;
      this.tmaCopy.position.copyFrom(
        p < 0.3
          ? point(-20, (-6.3 * p) / 0.3, 1.4)
          : p < 0.7
            ? point(-20 + (7.6 * (p - 0.3)) / 0.4, -6.3, 1.4)
            : point(-12.4, -6.3 + (5.3 * (p - 0.7)) / 0.3, 1.4),
      );
    }
    this.compute.parent!.setEnabled(!expanded && !tile);
    this.compute.position.y = 2.2 - stroke * 0.55;
    this.feed.setEnabled(!expanded && !tile && !!flow.using);
    this.feed.position.copyFrom(
      point(-12.4 + flow.computeProgress * 24.8, -8.7, 1),
    );
    this.groups.forEach((group, i) => {
      const selected = i === m.selectedWarp;
      group.root.setEnabled(!tile && (!expanded || selected));
      group.root.position.copyFrom(point(0, expanded ? 0 : this.row(i), 0.4));
      group.bed.setEnabled(!expanded);
      group.expandedBed.setEnabled(expanded);
      group.arm.setEnabled(!expanded);
      group.instruction.setEnabled(expanded);
      group.bed.material = w.materials[selected ? "dark" : "steel"];
      const issued =
        m.groups[i].role === flow.issued &&
        (m.groups[i].role !== "consumer" ||
          i ===
            m.groups.findIndex((g) => g.role === "consumer") +
              ((flow.using?.id ?? 0) %
                m.groups.filter((g) => g.role === "consumer").length));
      group.arm.rotation.y = issued ? stroke * 0.55 : 0;
      group.lanes.forEach((lane, j) => {
        lane.root.position.copyFrom(point(...warpLanePosition(j, expanded)));
        lane.root.scaling.setAll(expanded ? 1.25 : 1);
        lane.finger.material = w.materials[s.illustrated ? "teal" : "paper"];
        lane.finger.position.y = 0.45 - (issued ? stroke * 0.35 : 0);
      });
    });
    this.marker.setEnabled(expanded);
    const [x, y] = warpLanePosition(m.selectedLane, true);
    this.marker.position.copyFrom(point(x - 0.5, y, 0.65));
    this.label.textContent = `Lane ${m.selectedLane}`;
    w.labels
      .find((l) => l.element === this.label)!
      .point.copyFrom(point(7.1, y, 0.7));
    if (this.depth !== m.depth) {
      this.depth = m.depth;
    }
  }
}
