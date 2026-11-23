import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { pressStroke } from "./machine-motion";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { Mesh } from "@babylonjs/core/Meshes/mesh";
import type { MechanicalWorld } from "./replay-world";
import {
  equipmentState,
  equipmentMotionState,
  equipmentEdges,
  workshopParts,
  equipmentPathForRole,
  type Equipment,
} from "./workshop-model";
import { journeyPartState } from "./connected-journey";
import { transportPhase } from "./flow-motion";

type Point = [number, number, number];
const B = (x: number, y: number, z = 0) => new Vector3(-x, z, -y);
type Moving = {
  node: TransformNode;
  axis: "y" | "x" | "spin" | "gate" | "lift";
  base: Vector3;
};
export class InternalWorkshop {
  parts: Equipment[] = [];
  trail: Equipment[] = [];
  selected = "";
  crank: number | null = null;
  width = 38;
  depth = 16;
  rigs: {
    part: Equipment;
    root: TransformNode;
    lamp: Mesh;
    moving: Moving[];
    copy: TransformNode | null;
    packet: TransformNode;
  }[] = [];
  flows: { to: Equipment; points: Point[]; packet: TransformNode }[] = [];
  constructor(public world: MechanicalWorld) {}
  get current() {
    return this.parts.find((p) => p.id === this.selected) ?? this.parts[0];
  }
  open(role?: string) {
    this.trail = [];
    this.parts = workshopParts(
      this.world.capture,
      this.world.selectedLayer,
      this.world.selectedPart,
    );
    this.selected = this.parts[0].id;
    if (role) {
      const path = equipmentPathForRole(this.parts, role);
      if (path.length) {
        this.trail = path.slice(0, -1);
        this.parts = this.trail.at(-1)?.children ?? this.parts;
        this.selected = path.at(-1)!.id;
      }
    }
    this.build();
  }
  enter() {
    const part = this.current;
    if (!part?.children.length) return;
    this.world.inspectChange("in", () => {
      this.trail.push(part);
      this.parts = part.children;
      this.selected = this.parts[0].id;
      this.build();
    });
  }
  back() {
    this.world.inspectChange("out", () => {
      this.trail.pop();
      this.parts =
        this.trail.at(-1)?.children ??
        workshopParts(
          this.world.capture,
          this.world.selectedLayer,
          this.world.selectedPart,
        );
      this.selected = this.parts[0].id;
      this.build();
    });
  }
  select(id: string) {
    if (!this.parts.some((p) => p.id === id)) return;
    this.selected = id;
    if (innerWidth <= 650) this.focus();
  }
  focus() {
    const rig = this.rigs.find((r) => r.part.id === this.selected);
    if (!rig) return;
    this.world.userMovedCamera = false;
    this.world.resizeFraming = () => this.focus();
    this.world.camera.setTarget(rig.root.position);
    this.world.camera.alpha = Math.PI / 2 + 0.07;
    this.world.camera.radius = 9;
    this.world.camera.beta = 0.7;
  }
  point(part: Equipment): Point {
    const stages = Math.max(...this.parts.map((p) => p.stage)) + 1,
      columns = Math.ceil(stages / Math.ceil(stages / 5)),
      row = Math.floor(part.stage / columns),
      rows = Math.ceil(stages / columns);
    const col =
      row % 2 ? columns - 1 - (part.stage % columns) : part.stage % columns;
    return [
      (col - (columns - 1) / 2) * 7.7,
      ((rows - 1) / 2 - row) * 9 + part.lane * 4.2,
      0,
    ];
  }
  build() {
    const w = this.world,
      group = w.groups.internals;
    group.getChildMeshes().forEach((m) => m.dispose());
    group.getChildTransformNodes().forEach((n) => n.dispose());
    for (const l of w.labels.filter((l) => l.view === "internals")) {
      l.element.remove();
      l.leader.remove();
    }
    w.labels = w.labels.filter((l) => l.view !== "internals");
    this.rigs = [];
    this.flows = [];
    const points = this.parts.map((p) => this.point(p)),
      xs = points.map((p) => p[0]),
      ys = points.map((p) => p[1]);
    this.width = Math.max(...xs) - Math.min(...xs) + 8;
    this.depth = Math.max(...ys) - Math.min(...ys) + 8;
    for (const part of this.parts) {
      const point = this.point(part),
        root = new TransformNode(part.name, w.scene);
      root.parent = group;
      root.position = B(...point);
      root.metadata = { internal: part.id };
      const moving: Moving[] = [];
      const motion = (node: TransformNode, axis: Moving["axis"]) => {
        moving.push({ node, axis, base: node.position.clone() });
        return node;
      };
      w.box(
        "equipment foundation",
        [0, 0, 0.12],
        [4.7, 2.9, 0.24],
        "base",
        root,
      );
      // Input pins and output cradle stay unvalued: height is not recorded data.
      for (const x of [-2, 2]) {
        w.box("channel tray", [x, 0, 0.4], [0.45, 1.25, 0.18], "ink", root);
        for (let i = 0; i < 3; i++)
          w.box(
            "symbolic channel",
            [x, -0.4 + i * 0.4, 0.66],
            [0.22, 0.14, 0.35],
            "teal",
            root,
          );
      }
      w.pipe(
        "through-channel",
        [
          [-2, 0, 0.5],
          [2, 0, 0.5],
        ],
        "teal",
        root,
        0.055,
      );
      let copy: TransformNode | null = null;
      switch (part.kind) {
        case "press": {
          for (const x of [-1.25, 1.25])
            w.cyl("press guide", [x, 0, 1.2], 0.16, 1.85, "ink", root);
          for (let i = 0; i < 4; i++) {
            w.box(
              "operand row",
              [-0.78 + i * 0.52, 0, 0.6],
              [0.13, 1.4, 0.15],
              "teal",
              root,
            );
            w.box(
              "weight column",
              [0, -0.6 + i * 0.4, 0.8],
              [2, 0.13, 0.15],
              "brass",
              root,
            );
          }
          motion(
            w.box(
              "tensor product platen",
              [0, 0, 1.8],
              [2.5, 1.7, 0.27],
              "steel",
              root,
              true,
            ),
            "y",
          );
          w.cyl("drive cap", [0, 0, 2.12], 0.5, 0.2, "brass", root);
          break;
        }
        case "balance": {
          w.cyl("balance pedestal", [0, 0, 0.9], 0.4, 1.2, "ink", root);
          const beam = new TransformNode(
            "common magnitude adjustment",
            w.scene,
          );
          beam.parent = root;
          beam.position = B(0, 0, 1.45);
          w.box("balance beam", [0, 0, 0], [2.6, 0.2, 0.2], "brass", beam);
          for (const x of [-1, 1]) {
            w.cyl("pan stem", [x, 0, -0.25], 0.07, 0.6, "steel", beam);
            w.cyl("collecting pan", [x, 0, -0.6], 0.75, 0.12, "paper", beam);
          }
          motion(beam, "gate");
          w.box(
            "gain setting rail",
            [0, 0.95, 0.6],
            [2.2, 0.2, 0.2],
            "ink",
            root,
          );
          for (let i = 0; i < 4; i++)
            w.cyl(
              "gain setting",
              [-0.8 + i * 0.52, 0.95, 0.9],
              0.23,
              0.45,
              "brass",
              root,
            );
          break;
        }
        case "rotor": {
          for (const y of [-0.48, 0.48]) {
            const rotor = new TransformNode("coordinate pair rotor", w.scene);
            rotor.parent = root;
            rotor.position = B(0, y, 0.92);
            w.cyl("pair turntable", [0, 0, 0], 1.35, 0.18, "brass", rotor);
            w.box(
              "coordinate arm",
              [0, 0, 0.2],
              [1.6, 0.11, 0.13],
              "teal",
              rotor,
            );
            for (const x of [-0.7, 0.7])
              w.cyl("paired pin", [x, 0, 0.38], 0.18, 0.3, "paper", rotor);
            motion(rotor, "spin");
          }
          break;
        }
        case "score": {
          for (let i = 0; i < 5; i++) {
            const x = -1.2 + i * 0.6;
            w.box(
              "key comparison lane",
              [x, 0, 0.55],
              [0.28, 1.9, 0.15],
              "paper",
              root,
            );
            w.box(
              "key tooth",
              [x, 0.55, 0.87],
              [0.24, 0.2, 0.5],
              "brass",
              root,
            );
          }
          motion(
            w.box(
              "query comparison comb",
              [0, -0.65, 1.1],
              [3, 0.18, 0.4],
              "teal",
              root,
              true,
            ),
            "x",
          );
          w.box("score collector", [0, 1, 0.85], [3, 0.18, 0.3], "steel", root);
          break;
        }
        case "cam": {
          if (part.name === "Maximum selector") {
            for (let i = 0; i < 4; i++) {
              const x = -1.1 + i * 0.72;
              w.cyl(
                "score follower guide",
                [x, 0, 0.5],
                0.4,
                0.4,
                "steel",
                root,
              );
              w.cyl(
                "illustrative score follower",
                [x, 0, 0.8 + i * 0.16],
                0.2,
                0.6 + i * 0.32,
                "brass",
                root,
              );
            }
            motion(
              w.box(
                "maximum sensing bar",
                [0, 0, 1.75],
                [3.2, 0.3, 0.22],
                "teal",
                root,
                true,
              ),
              "lift",
            );
            w.box(
              "maximum output rack",
              [1.7, 0.7, 1],
              [0.18, 0.3, 1.5],
              "ink",
              root,
            );
            break;
          }
          const rotor = new TransformNode("function cam", w.scene);
          rotor.parent = root;
          rotor.position = B(0, 0, 0.9);
          w.cyl("cam wheel", [0, 0, 0], 1.8, 0.24, "brass", rotor);
          w.cyl("offset lobe", [0.45, 0, 0.18], 0.9, 0.22, "brass", rotor);
          w.box("shaft marker", [0, 0, 0.4], [1.5, 0.1, 0.12], "ink", rotor);
          motion(rotor, "spin");
          for (const x of [-1.35, 1.35]) {
            w.cyl("follower guide", [x, 0, 0.6], 0.23, 0.7, "steel", root);
            motion(
              w.box(
                "function follower",
                [x, 0, 1.3],
                [0.24, 0.55, 0.45],
                "teal",
                root,
                true,
              ),
              "y",
            );
          }
          break;
        }
        case "merge": {
          for (let i = 0; i < 3; i++) {
            const y = -0.85 + i * 0.85;
            w.cyl(
              "input cup",
              [-0.9, y, 0.7],
              0.62,
              0.35,
              i === 1 ? "green" : "paper",
              root,
            );
            w.pipe(
              "contribution channel",
              [
                [-0.9, y, 0.75],
                [0.5, 0, 0.75],
                [1.6, 0, 0.75],
              ],
              "teal",
              root,
              0.045,
            );
          }
          const beam = w.box(
            "collecting crossbeam",
            [0, 0, 1.35],
            [1.1, 2.5, 0.18],
            "brass",
            root,
            true,
          );
          motion(beam, "y");
          w.cyl("result pan", [1.3, 0, 0.6], 0.8, 0.2, "steel", root);
          break;
        }
        case "latch": {
          for (const x of [-1.1, 1.1])
            w.cyl("stop-arm post", [x, 0, 0.9], 0.18, 1.3, "ink", root);
          const gate = new TransformNode("readiness stop arm", w.scene);
          gate.parent = root;
          gate.position = B(-1.1, 0, 1.55);
          w.box("stop arm", [1.1, 0, 0], [2.2, 0.16, 0.18], "brass", gate);
          motion(gate, "gate");
          for (let i = 0; i < 4; i++)
            w.box(
              "ready counter tab",
              [-0.7 + i * 0.48, 0.85, 0.65],
              [0.3, 0.25, 0.5],
              "paper",
              root,
            );
          break;
        }
        case "weights": {
          w.box(
            "retained weight reservoir",
            [0, 0, 0.52],
            [2.9, 2, 0.3],
            "ink",
            root,
          );
          for (const x of [-1.45, 1.45])
            w.box(
              "reservoir side",
              [x, 0, 0.84],
              [0.12, 2, 0.6],
              "steel",
              root,
            );
          for (let i = 0; i < 12; i++)
            w.ball(
              "retained weight ball",
              [-1.05 + (i % 4) * 0.7, -0.65 + Math.floor(i / 4) * 0.65, 0.88],
              0.36,
              root,
            );
          copy = w.ball("illustrative read copy", [-1.4, 0, 1.3], 0.3, root);
          break;
        }
        case "rack": {
          for (const x of [-1.5, 1.5])
            w.box("rack upright", [x, 0, 1.05], [0.15, 1.8, 1.6], "ink", root);
          for (let i = 0; i < 3; i++) {
            w.box(
              "storage shelf",
              [0, 0, 0.45 + i * 0.45],
              [3, 1.7, 0.12],
              "steel",
              root,
            );
            for (let j = 0; j < 3; j++)
              w.box(
                "symbolic stored slot",
                [-0.85 + j * 0.85, 0, 0.65 + i * 0.45],
                [0.5, 1, 0.2],
                "green",
                root,
              );
          }
          break;
        }
        case "store": {
          for (let i = 0; i < 4; i++)
            w.box(
              "receiving slot",
              [-1.15 + i * 0.76, 0, 0.6],
              [0.5, 1.7, 0.25],
              "paper",
              root,
            );
          motion(
            w.box(
              "placing fork",
              [0, 0, 1.25],
              [2.9, 0.2, 0.22],
              "steel",
              root,
              true,
            ),
            "y",
          );
          w.box(
            "publication tab",
            [1.65, 0.85, 0.85],
            [0.3, 0.3, 0.8],
            "brass",
            root,
          );
          break;
        }
        case "gather": {
          for (let i = 0; i < 3; i++)
            w.cyl("feed roller", [-1 + i, 0, 0.68], 0.6, 0.3, "brass", root);
          const fork = new TransformNode("gathering fork", w.scene);
          fork.parent = root;
          fork.position = B(0, 0, 1.1);
          w.box("fork spine", [0, -0.65, 0], [2.7, 0.16, 0.18], "steel", fork);
          for (let i = 0; i < 4; i++)
            w.box(
              "fork tine",
              [-1 + i * 0.66, 0, 0],
              [0.13, 1.4, 0.14],
              "steel",
              fork,
            );
          motion(fork, "x");
          break;
        }
      }
      // A separate supply has a physical inlet and remains present while it is read.
      if (/weights|gains|gain\b/.test(part.uses) && part.kind !== "weights") {
        w.box(
          "local reusable weight drawer",
          [0, 2, 0.4],
          [2.5, 0.8, 0.2],
          "ink",
          root,
        );
        for (let i = 0; i < 5; i++)
          w.ball("stored weight", [-0.9 + i * 0.45, 2, 0.65], 0.25, root);
        w.pipe(
          "weight supply",
          [
            [0, 2, 0.6],
            [0, 1.4, 0.6],
            [0, 0.85, 0.6],
          ],
          "brass",
          root,
          0.06,
        );
        copy = w.ball("illustrative weight read", [0, 2, 0.8], 0.25, root);
      }
      if (
        /tables|cache|Cached|counter|position/i.test(part.uses) &&
        part.kind !== "rack" &&
        part.kind !== "latch"
      ) {
        w.cyl("auxiliary lookup drum", [0, 2, 0.8], 1.15, 0.9, "dark", root);
        w.cyl("lookup drum rim", [0, 2, 1.25], 1.22, 0.08, "steel", root);
        w.pipe(
          "lookup feed",
          [
            [0, 2, 0.6],
            [0, 0.95, 0.6],
          ],
          "green",
          root,
          0.055,
        );
      }
      const lamp = w.box(
        "operation state",
        [2, 1, 0.8],
        [0.23, 0.3, 0.75],
        "paper",
        root,
        true,
      ) as Mesh;
      const packet =
        w.selectedPart === "weights"
          ? w.ball(
              "weight copy passing through " + part.name,
              [0, 0, 0.9],
              0.52,
              root,
            )
          : w.journey.tray(root, "information passing through " + part.name)
              .root;
      this.rigs.push({ part, root, lamp, moving, copy, packet });
      w.label(
        part.name,
        [point[0], point[1] - 2.05, 1],
        "internals",
        () => w.onInternal(part.id),
        { internal: part.id, anchor: root.position.add(new Vector3(0, 1, 0)) },
      );
    }
    // Every stage fans into all branches of its successor. This draws real parallel
    // structure in the analogy without assigning unrecorded branch timing.
    for (const [from, to] of equipmentEdges(this.parts)) {
      const a = this.point(from),
        b = this.point(to),
        sameRow = Math.abs(a[1] - b[1]) < 6;
      const count = Math.max(...this.parts.map((p) => p.stage)) + 1,
        columns = Math.ceil(count / Math.ceil(count / 5)),
        outside = a[0] + (Math.floor(from.stage / columns) % 2 ? -3.5 : 3.5);
      const points: Point[] = sameRow
        ? [
            [a[0], a[1], 0.3],
            [(a[0] + b[0]) / 2, a[1], 0.3],
            [(a[0] + b[0]) / 2, b[1], 0.3],
            [b[0], b[1], 0.3],
          ]
        : [
            [a[0], a[1], 0.3],
            [outside, a[1], 0.3],
            [outside, b[1], 0.3],
            [b[0], b[1], 0.3],
          ];
      points[0][0] += (Math.floor(from.stage / columns) % 2 ? -1 : 1) * 1.9;
      points[points.length - 1][0] -=
        (Math.floor(to.stage / columns) % 2 ? -1 : 1) * 1.9;
      w.pipe("information connection", points, "teal", group, 0.065);
      const packet =
        w.selectedPart === "weights"
          ? w.ball(
              "weight copy travelling between equipment",
              [0, 0, 0.9],
              0.52,
              group,
            )
          : w.journey.tray(group, "information travelling between equipment")
              .root;
      this.flows.push({ to, points, packet });
    }
  }
  update(time: number) {
    const stages = Math.max(...this.parts.map((p) => p.stage)) + 1;
    const w = this.world;
    const enclosing = w.story
      ? journeyPartState(w.story, w.selectedLayer, w.selectedPart)
      : w.selectedPart === "weights"
        ? w.recorded.weights(w.selectedLayer, time)
        : null;
    const cycle = this.crank ?? enclosing?.progress ?? null;
    const stateFor = (part: Equipment) => {
      const manual =
        cycle === null
          ? null
          : Math.max(0, Math.min(1, cycle * stages - part.stage));
      const state =
        manual === null
          ? equipmentMotionState(
              w.capture,
              w.selectedLayer,
              this.parts,
              part,
              time,
            )
          : equipmentState(w.capture, w.selectedLayer, part, time, manual);
      if (this.crank === null && enclosing && !enclosing.active)
        state.active = false;
      if (
        this.crank === null &&
        enclosing &&
        !("present" in enclosing ? enclosing.present : enclosing.active)
      )
        state.present = false;
      return state;
    };
    for (const rig of this.rigs) {
      const state = stateFor(rig.part),
        p = state.progress;
      rig.lamp.material =
        this.world.materials[
          state.active
            ? rig.part.kind === "latch"
              ? "wait"
              : "teal"
            : state.done
              ? "green"
              : "paper"
        ];
      for (const moving of rig.moving) {
        moving.node.position.copyFrom(moving.base);
        if (moving.axis === "y")
          moving.node.position.y -= 0.55 * pressStroke(p);
        if (moving.axis === "x")
          moving.node.position.x += 0.5 * Math.sin(p * Math.PI * 2);
        if (moving.axis === "spin") moving.node.rotation.y = p * Math.PI * 2;
        if (moving.axis === "gate")
          moving.node.rotation.z =
            rig.part.kind === "latch"
              ? state.done
                ? -Math.PI / 2
                : 0
              : 0.12 * Math.sin(p * Math.PI * 2);
        if (moving.axis === "lift")
          moving.node.position.y = moving.base.y + 0.45 * p;
      }
      if (rig.copy) {
        rig.copy.setEnabled(state.present);
        rig.copy.position.z = -2 + p * 1.25;
      }
      const phase = transportPhase(
        p,
        this.flows.some((f) => f.to.id === rig.part.id),
      );
      rig.packet.setEnabled(state.present && !phase.travelling);
      const columns = Math.ceil(stages / Math.ceil(stages / 5)),
        direction = Math.floor(rig.part.stage / columns) % 2 ? -1 : 1;
      rig.packet.position = B(
        direction * (-1.9 + phase.machine * 3.8),
        0,
        0.95,
      );
    }
    for (const flow of this.flows) {
      const state = stateFor(flow.to),
        phase = transportPhase(state.progress, true);
      flow.packet.setEnabled(state.present && phase.travelling);
      const points = flow.points;
      const lengths = points
        .slice(1)
        .map((p, i) => Math.hypot(p[0] - points[i][0], p[1] - points[i][1]));
      let distance = phase.travel * lengths.reduce((a, b) => a + b, 0);
      for (let i = 0; i < lengths.length; i++) {
        if (distance <= lengths[i] || i === lengths.length - 1) {
          const p = lengths[i] ? Math.min(1, distance / lengths[i]) : 0,
            a = points[i],
            b = points[i + 1];
          flow.packet.position = B(
            a[0] + (b[0] - a[0]) * p,
            a[1] + (b[1] - a[1]) * p,
            0.95,
          );
          break;
        }
        distance -= lengths[i];
      }
    }
  }
}
