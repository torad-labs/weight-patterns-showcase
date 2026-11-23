import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { MechanicalWorld } from "./replay-world";
import { samplePath, type Timeline } from "./timeline";
import {
  journeyPartState,
  journeyOperations,
  clamp,
} from "./connected-journey";
import { stackCarrierProgress, stackCarrierAt, stackPath } from "./stack-route";
import type { MechanismId } from "./replay-model";
import { weightedParts } from "./recorded-motion";
import { packetFlight } from "./flow-motion";
type Point = [number, number, number];
const B = (p: readonly number[]) => new Vector3(-p[0], p[2], -p[1]);
type Tray = { root: TransformNode; pins: TransformNode[] };
const locations: Record<string, Point> = {
  norm: [-8, 0, 1],
  qkv: [-5, 0, 1],
  rope: [-3, 0, 1],
  attention: [0, 0, 1],
  output: [3, 0, 1],
  residual: [5.2, 0, 1],
  gateup: [5, -4, 1],
  down: [0, -4, 1],
  ple: [-5, -4, 1],
  latch: [-8, -4, 1],
};
function operationPath(part: MechanismId): Point[] {
  const index = journeyOperations.findIndex((o) => o.id === part),
    previous =
      index > 0
        ? locations[journeyOperations[index - 1].id]
        : ([-11, 0, 1] as Point);
  if (part === "output") return [locations.output, locations.output];
  if (part === "gateup")
    return [previous, [7.7, 0, 1], [7.7, -4, 1], locations.gateup];
  return [previous, locations[part]];
}
export class JourneyWorld {
  endpoints: TransformNode;
  stackTrays: Tray[] = [];
  layerTrays: Tray[] = [];
  headTrays: Tray[] = [];
  weightReads: {
    node: TransformNode;
    path: Point[];
    part: MechanismId;
    copy: number;
  }[] = [];
  operationTrays: { tray: Tray; part: MechanismId; path: Point[] }[] = [];
  cutter: TransformNode;
  slips: TransformNode[] = [];
  returnTray: Tray;
  answer: HTMLButtonElement;
  input: HTMLButtonElement;
  stackReads: {
    node: TransformNode;
    layer: number;
    part: MechanismId;
    copy: number;
    path: Point[];
  }[] = [];
  constructor(public world: MechanicalWorld) {
    const w = world,
      g = w.groups.stack;
    this.endpoints = new TransformNode(
      "question and answer machinery",
      w.scene,
    );
    this.endpoints.parent = g;
    const input = new TransformNode(
      "question cutter and carrier assembler",
      w.scene,
    );
    input.parent = this.endpoints;
    input.position = B([-16, 18, 0]);
    w.box("input table", [0, 0, 0.3], [9, 3, 0.35], "base", input);
    w.box("question paper", [-1, 0, 0.6], [6, 1.5, 0.08], "paper", input);
    for (const x of [-3, 3])
      w.cyl("paper roller", [x, 0, 0.85], 0.6, 1.3, "brass", input);
    this.cutter = w.box(
      "paper cutter",
      [0, 0, 1.7],
      [0.22, 2, 1.2],
      "steel",
      input,
    );
    for (let i = 0; i < 7; i++)
      this.slips.push(
        w.box(
          "cut question token",
          [-2.7 + i * 0.9, 0, 0.8],
          [0.6, 1, 0.1],
          "paper",
          input,
        ),
      );
    w.pipe(
      "question into layer one",
      [
        [-12, 18, 0.7],
        [-9, 18, 0.7],
        [-9, 15, 0.7],
        [-19.7, 15, 0.7],
        [-19.7, 11.9, 1.35],
      ],
      "teal",
      g,
      0.09,
    );
    this.input = w.label("Question enters here", [-16, 20.4, 1], "stack", () =>
      w.onEndpoint("input"),
    );
    const reservoir = new TransformNode("reusable weight reservoir", w.scene);
    reservoir.parent = g;
    reservoir.position = B([-26, 1, 0]);
    w.box("weight reservoir bed", [0, 0, 0.35], [4, 7, 0.5], "ink", reservoir);
    for (const x of [-1.8, 1.8])
      w.box("reservoir side", [x, 0, 1], [0.15, 7, 1.1], "steel", reservoir);
    for (let i = 0; i < 28; i++)
      w.ball(
        "stored reusable weight",
        [-1.1 + (i % 4) * 0.72, -2.6 + Math.floor(i / 4) * 0.85, 0.85],
        0.58,
        reservoir,
      );
    w.pipe(
      "weight supply spine",
      [
        [-24, 1, 0.5],
        [-23, 1, 0.5],
        [-23, 13, 0.5],
        [-23, -14, 0.5],
      ],
      "brass",
      g,
      0.1,
    );
    for (let row = 0; row < 7; row++) {
      const y = 12.6 - row * 4.2;
      w.pipe(
        "row weight supply",
        [
          [-23, y - 1.5, 0.5],
          [18, y - 1.5, 0.5],
        ],
        "brass",
        g,
        0.07,
      );
    }
    for (const layer of w.capture.layers) {
      const [x, y] = w.center(layer.id),
        path: Point[] = [
          [-24, 1, 0.8],
          [-23, 1, 0.8],
          [-23, y - 1.5, 0.8],
          [x - 1, y - 1.5, 0.8],
          [x - 1, y, 0.8],
        ];
      w.pipe(
        "layer weight inlet",
        [
          [x - 1, y - 1.5, 0.5],
          [x - 1, y, 0.5],
        ],
        "brass",
        g,
        0.07,
      );
      for (const part of weightedParts)
        for (let copy = 0; copy < 3; copy++)
          this.stackReads.push({
            node: w.ball(
              `weight read layer ${layer.id + 1} ${part} ${copy}`,
              path[0],
              0.62,
              g,
            ),
            layer: layer.id,
            part,
            copy,
            path,
          });
    }
    w.label("Stored weights · reused", [-26, -3.5, 0.8], "stack", () =>
      w.onPart("weights"),
    );
    const output = new TransformNode(
      "candidate selector and answer printer",
      w.scene,
    );
    output.parent = this.endpoints;
    output.position = B([15, -18.5, 0]);
    w.box("answer selector bed", [0, 0, 0.35], [10, 3, 0.4], "base", output);
    for (let i = 0; i < 4; i++) {
      w.box(
        "candidate slot",
        [-3 + i * 1.5, 0, 0.8],
        [0.9, 1.4, 0.4],
        "steel",
        output,
      );
      w.cyl("selector cam", [-3 + i * 1.5, 0, 1.2], 0.6, 0.2, "brass", output);
    }
    w.box("printed answer", [3, 0, 0.8], [2.8, 1.8, 0.1], "paper", output);
    this.answer = w.label("Answer printer", [15, -21, 1], "stack", () =>
      w.onEndpoint("output"),
    );
    w.pipe(
      "last layer to selector",
      [
        [19.7, -13.3, 1.35],
        [22, -13.3, 1],
        [22, -18.5, 1],
        [10, -18.5, 1],
      ],
      "teal",
      g,
      0.09,
    );
    w.pipe(
      "selected token feedback",
      [
        [15, -18.5, 1],
        [24, -18.5, 1],
        [24, 22, 1],
        [-23, 22, 1],
        [-23, 11.9, 1],
        [-19.7, 11.9, 1.35],
      ],
      "green",
      g,
      0.075,
    );
    this.returnTray = this.tray(g, "selected answer token");
    for (let i = 0; i < 7; i++)
      this.stackTrays.push(this.tray(g, `question carrier ${i + 1}`));
    this.buildLayer();
  }
  tray(parent: TransformNode, name: string): Tray {
    const w = this.world,
      root = new TransformNode(name, w.scene);
    root.parent = parent;
    root.metadata = { materialTopic: "cart" };
    w.box("information tray", [0, 0, 0], [0.95, 0.65, 0.16], "teal", root);
    const pins = [];
    for (let i = 0; i < 4; i++)
      pins.push(
        w.box(
          "illustrative information pin",
          [-0.32 + i * 0.21, 0, 0.25],
          [0.12, 0.32, 0.45],
          "paper",
          root,
        ),
      );
    return { root, pins };
  }
  place(tray: Tray, path: Point[], p: number, shape = 0) {
    tray.root.position = B(samplePath(path, clamp(p)));
    tray.pins.forEach((pin, i) => {
      const height = 0.35 + 0.15 * Math.sin(i * 1.7 + shape * Math.PI * 2);
      pin.scaling.y = height;
      pin.position.y = 0.12 + height / 2;
    });
  }
  buildLayer() {
    const w = this.world,
      g = w.detail;
    this.layerTrays = [
      this.tray(g, "same incoming token tray"),
      this.tray(g, "preserved residual tray"),
    ];
    this.headTrays = [];
    this.weightReads = [];
    this.operationTrays = journeyOperations
      .filter((o) => o.id !== "attention")
      .map((o) => ({
        part: o.id,
        path: operationPath(o.id),
        tray: this.tray(g, "recorded " + o.id + " work tray"),
      }));
    for (let i = 0; i < w.capture.queryHeads; i++)
      this.headTrays.push(this.tray(g, `query branch ${i + 1}`));
    for (const part of [
      "norm",
      "qkv",
      "output",
      "gateup",
      "down",
      "ple",
    ] as MechanismId[]) {
      const target = locations[part];
      const path: Point[] = [
        [-6.8, 4.1, 0.85],
        [target[0], 3, 0.85],
        [target[0], target[1], 0.85],
      ];
      w.pipe("reusable weights feed " + part, path, "brass", g, 0.075);
      for (let copy = 0; copy < 3; copy++)
        this.weightReads.push({
          node: w.ball("weight read copy " + part, [-6.8, 4.1, 0.85], 0.4, g),
          path,
          part,
          copy,
        });
    }
    w.pipe(
      "incoming information",
      [
        [-11, 0, 1],
        [-8, 0, 1],
      ],
      "teal",
      g,
      0.085,
    );
    w.pipe(
      "outgoing information",
      [
        [-8, -4, 1],
        [-11, -4, 1],
      ],
      "teal",
      g,
      0.085,
    );
    w.label("Information in", [-10.5, 1.3, 1], "layer", () =>
      w.onEndpoint("input"),
    );
    w.label("To the next layer", [-10.5, -5.5, 1], "layer", () =>
      w.onEndpoint("output"),
    );
  }
  update(t: Timeline | null, time: number) {
    const w = this.world;
    this.endpoints.setEnabled(true);
    this.stackTrays.forEach((r) => r.root.setEnabled(!!t));
    this.returnTray.root.setEnabled(!!t);
    this.layerTrays.forEach((r) => r.root.setEnabled(!!t));
    this.headTrays.forEach((r) => r.root.setEnabled(false));
    this.weightReads.forEach((r) => r.node.setEnabled(false));
    this.input.textContent = t
      ? "Question → token trays"
      : "Input · not recorded";
    this.answer.textContent = t
      ? t.answer
        ? `Printed: ${t.answer}`
        : "Choose & print the answer"
      : "Readout · not recorded";
    this.stackReads.forEach((r) => r.node.setEnabled(false));
    this.operationTrays.forEach((r) => r.tray.root.setEnabled(false));
    if (!t) {
      this.recorded(time);
      return;
    }
    if (w.view === "stack") {
      const stage = t.stage.id,
        p = t.stageProgress;
      this.cutter.position.y =
        1.7 - (stage === "intake" ? 0.7 * Math.sin(p * Math.PI * 7) ** 2 : 0);
      this.slips.forEach((s, i) => {
        s.position.x = 2.7 - i * 0.9 + (stage === "intake" ? 0.3 * p * i : 0);
        s.setEnabled(t.progress > 0.065 || p > i / 7);
      });
      this.stackTrays.forEach((tray, i) => {
        const many = stage === "prefill" || stage === "embed";
        tray.root.setEnabled(
          (t.layerActive || stage === "embed") && (many || i === 0),
        );
        if (t.layerActive) {
          const point = stackCarrierProgress(
            t.layerIndex,
            t.layerIndex + 1 < t.layerCount ? t.layerIndex + 1 : undefined,
            t.layerProgress,
            journeyOperations.find((o) => o.id === "latch")!.start,
            i * 0.035,
          );
          this.place(
            tray,
            [point, point],
            0,
            t.layerIndex / 35 + t.layerProgress,
          );
        } else
          this.place(
            tray,
            [
              [-16, 18, 1.2],
              [-9, 18, 1.2],
              [-9, 15, 1.2],
              [-19.7, 15, 1.2],
              [-19.7, 11.9, 1.35],
            ],
            clamp(p - i * 0.035),
          );
      });
      const returning = stage === "return-paris" || stage === "return-period",
        choosing = stage.startsWith("choose");
      this.returnTray.root.setEnabled(returning || choosing);
      this.place(
        this.returnTray,
        returning
          ? [
              [15, -18.5, 1.2],
              [24, -18.5, 1.2],
              [24, 22, 1.2],
              [-23, 22, 1.2],
              [-23, 11.9, 1.2],
              [-19.7, 11.9, 1.35],
            ]
          : [
              [19.7, -13.3, 1.35],
              [22, -13.3, 1.2],
              [22, -18.5, 1.2],
              [15, -18.5, 1.2],
            ],
        p,
      );
      for (const r of this.stackReads) {
        const state = journeyPartState(t, r.layer, r.part),
          flight = packetFlight(state.progress, r.copy, 3);
        r.node.setEnabled(state.active && flight.visible);
        r.node.position = B(samplePath(r.path, flight.progress));
      }
    }
    if (w.view === "layer") {
      const p =
        t.layerIndex === w.selectedLayer && t.layerActive
          ? t.layerProgress
          : t.layerIndex > w.selectedLayer
            ? 1
            : 0;
      const op =
          journeyOperations.find((o) => p < o.end) ?? journeyOperations.at(-1)!,
        path = operationPath(op.id);
      const phase = clamp((p - op.start) / (op.end - op.start));
      this.place(this.layerTrays[0], path, phase, p);
      this.place(
        this.layerTrays[1],
        [
          [-8, 0, 0.7],
          [-9, 0, 0.7],
          [-9, -2.5, 0.7],
          [5.2, -2.5, 0.7],
          [5.2, 0, 0.7],
        ],
        clamp(p / 0.65),
        0,
      );
      if (op.id === "attention") {
        this.layerTrays[0].root.setEnabled(false);
        this.headTrays.forEach((tray, h) => {
          tray.root.setEnabled(true);
          const y = (h - 3.5) * 0.64;
          this.place(
            tray,
            [
              [-3, 0, 1],
              [-1.6, y, 1],
              [0, y, 1],
              [1.5, y, 1],
              [3, 0, 1],
            ],
            phase,
            p + h * 0.05,
          );
        });
      }
      this.weightReads.forEach((r) => {
        const state = journeyPartState(t, w.selectedLayer, r.part),
          flight = packetFlight(state.progress, r.copy, 3);
        r.node.setEnabled(state.active && flight.visible);
        r.node.position = B(samplePath(r.path, flight.progress));
      });
    }
  }
  recorded(time: number) {
    const w = this.world,
      motion = w.recorded;
    if (w.view === "stack") {
      this.slips.forEach((s) => s.setEnabled(false));
      this.cutter.position.y = 1.7;
      const point = stackCarrierAt(w.capture, time);
      this.stackTrays[0].root.setEnabled(!!point);
      const fallback =
        time <= w.capture.layers[0].journeyStart
          ? stackPath(0)[0]
          : stackPath(w.capture.layers.length - 1).at(-1)!;
      this.place(this.stackTrays[0], [point ?? fallback, point ?? fallback], 0);
      this.stackReads.forEach((r) => {
        const state = motion.state(r.layer, r.part, time, true),
          flight = packetFlight(state.progress, r.copy, 3);
        r.node.setEnabled(state.present && flight.visible);
        r.node.position = B(samplePath(r.path, flight.progress));
      });
    }
    if (w.view === "layer") {
      const layer = w.selectedLayer;
      this.operationTrays.forEach(({ tray, part, path }) => {
        const state = motion.state(layer, part, time);
        tray.root.setEnabled(state.present);
        this.place(tray, path, state.progress);
      });
      const residual = motion.state(layer, "residual", time);
      this.layerTrays[1].root.setEnabled(residual.present);
      this.place(
        this.layerTrays[1],
        [
          [-8, 0, 0.7],
          [-9, 0, 0.7],
          [-9, -2.5, 0.7],
          [5.2, -2.5, 0.7],
          [5.2, 0, 0.7],
        ],
        residual.progress,
      );
      const attention = motion.state(layer, "attention", time);
      this.headTrays.forEach((tray, h) => {
        tray.root.setEnabled(attention.present);
        if (attention.present) {
          const y = (h - 3.5) * 0.64;
          this.place(
            tray,
            [
              [-3, 0, 1],
              [-1.6, y, 1],
              [0, y, 1],
              [1.5, y, 1],
              [3, 0, 1],
            ],
            attention.progress,
          );
        }
      });
      this.weightReads.forEach((r) => {
        const state = motion.state(layer, r.part, time, true),
          flight = packetFlight(state.progress, r.copy, 3);
        r.node.setEnabled(state.present && flight.visible);
        r.node.position = B(samplePath(r.path, flight.progress));
      });
    }
  }
  followPoint(): Vector3 {
    const w = this.world,
      t = w.story;
    if (w.view === "stack") {
      if (!t || t.layerActive || t.stage.id === "embed")
        return this.stackTrays[0].root.position;
      return t.stage.id.startsWith("return") || t.stage.id.startsWith("choose")
        ? this.returnTray.root.position
        : t.stage.id === "intake"
          ? B([-16, 18, 1])
          : B([15, -18.5, 1]);
    }
    if (w.view === "layer")
      return B(locations[w.selectedPart] ?? [-6.8, 4.1, 1]);
    if (w.view === "internals")
      return (
        w.workshop.rigs.find((r) => r.part.id === w.workshop.selected)?.root
          .position ??
        w.workshop.rigs[0]?.root.position ??
        w.camera.target
      );
    const i = w.selectedBlock;
    return B([((i % 9) - 4) * 4.8, (3 - Math.floor(i / 9)) * 4, 0.5]);
  }
}
