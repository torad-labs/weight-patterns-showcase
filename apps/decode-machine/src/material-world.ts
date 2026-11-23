import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { MechanicalWorld } from "./replay-world";
import {
  MaterialModel,
  materialTopics,
  type MaterialSnapshot,
  type MaterialTopic,
} from "./material-model";
import { clamp } from "./connected-journey";
import { samplePath } from "./timeline";
import { pressStroke } from "./machine-motion";
import { stackCarrierProgress } from "./stack-route";

type Point = [number, number, number];
const position = (p: readonly number[]) => new Vector3(-p[0], p[2], -p[1]);
const consumerY = (index: number) => (2.5 - index) * 1.2;
const anchors: Record<MaterialTopic, Point> = {
  cart: [-12, 0, 1],
  context: [-3, -5.5, 0.8],
  split: [0, 0, 1],
  supply: [-6, 5, 1],
  worker: [-3, 1.3, 1],
  wait: [6, 0, 1],
  precision: [10, 0, 1],
  choice: [10, 5, 1],
  requests: [-13, -4, 0.8],
};

/** Connected cutaway. Every pose is reconstructed from MaterialModel's shared clock. */
export class MaterialWorld {
  state: MaterialSnapshot;
  onSelect: (topic: MaterialTopic) => void = () => {};
  pins: AbstractMesh[] = [];
  beforePins: AbstractMesh[] = [];
  partials: TransformNode[] = [];
  presses: AbstractMesh[] = [];
  weights: TransformNode[] = [];
  slots: AbstractMesh[] = [];
  context: AbstractMesh[][] = [[], []];
  reads: TransformNode[] = [];
  candidates: AbstractMesh[] = [];
  returnToken: TransformNode;
  cart: TransformNode;
  gate: TransformNode;
  controller: TransformNode;
  loader: TransformNode;
  needle: TransformNode;
  secondTrays: ReturnType<MechanicalWorld["journey"]["tray"]>[] = [];
  requestGates: TransformNode[] = [];
  identityNotches: AbstractMesh[] = [];
  constructor(
    public host: MechanicalWorld,
    public model: MaterialModel,
  ) {
    this.state = model.snapshot();
    const w = host,
      g = w.groups.material;
    const root = (name: string, at: Point, topic: MaterialTopic) => {
      const n = new TransformNode(name, w.scene);
      n.parent = g;
      n.position = position(at);
      n.metadata = { materialTopic: topic };
      return n;
    };
    const pipe = (name: string, points: Point[], color = "teal") =>
      w.pipe(name, points, color, g, 0.065);
    const bed = (
      name: string,
      at: Point,
      size: Point,
      topic: MaterialTopic,
    ) => {
      const n = root(name, at, topic);
      w.box(name + " bed", [0, 0, 0], size, "base", n);
      return n;
    };
    pipe("same information enters the open tray", [
      [-16, 0, 0.55],
      [-8, 0, 0.55],
    ]);
    const trayBed = bed(
      "opened channel cartridge",
      [-12, 0, 0.4],
      [5.4, 2.7, 0.2],
      "cart",
    );
    w.box(
      "hinged cartridge lid",
      [0, 1.4, 0.8],
      [5.4, 0.16, 1.4],
      "steel",
      trayBed,
    );
    for (let i = 0; i < 12; i++) {
      const x = -2.3 + i * 0.42;
      this.beforePins.push(
        w.box(
          "entering channel " + i,
          [x, 0.55, 0.45],
          [0.19, 0.3, 0.7],
          "green",
          trayBed,
        ),
      );
      this.pins.push(
        w.box(
          "current channel " + i,
          [x, -0.45, 0.45],
          [0.22, 0.35, 0.7],
          "teal",
          trayBed,
        ),
      );
    }
    this.cart = root("same token entering and leaving", [-15, 0, 0.85], "cart");
    w.box("solid token tray", [0, 0, 0], [1.1, 0.8, 0.15], "teal", this.cart);
    for (let i = 0; i < 2; i++)
      this.identityNotches.push(
        w.box(
          "request identity tooth " + i,
          [-0.2 + i * 0.4, 0, 0.18],
          [0.12, 0.45, 0.25],
          "paper",
          this.cart,
        ),
      );
    bed("split junction", [-7.5, 0, 0.45], [1, 1.4, 0.3], "split");
    for (let i = 0; i < 6; i++) {
      const x = 0,
        y = consumerY(i);
      const press = bed(
        "consumer " + (i + 1),
        [x, y, 0.35],
        [2.5, 0.85, 0.2],
        "worker",
      );
      for (const px of [-1.05, 1.05])
        w.cyl("press column", [px, 0, 0.9], 0.12, 1.5, "ink", press);
      this.presses.push(
        w.box("consumer platen", [0, 0, 1.4], [2.2, 0.65, 0.2], "steel", press),
      );
      w.cyl("partial result cup", [1, -0.2, 0.25], 0.38, 0.3, "brass", press);
      pipe("input tile lane " + i, [
        [-7.5, 0, 0.65],
        [-6, y, 0.65],
        [x, y, 0.65],
      ]);
      pipe("partial result lane " + i, [
        [x, y, 0.65],
        [5, y, 0.65],
        [6, 0, 0.65],
      ]);
      const tile = root("partial tray " + i, [-7.5, 0, 0.8], "split");
      w.box(
        "two channel work tray",
        [0, 0, 0],
        [0.7, 0.55, 0.12],
        "teal",
        tile,
      );
      for (const px of [-0.17, 0.17])
        w.box("operand pin", [px, 0, 0.2], [0.12, 0.3, 0.32], "paper", tile);
      this.partials.push(tile);
    }
    this.controller = root("job controller", [-5, 3, 0.75], "worker");
    w.cyl("controller cam", [0, 0, 0], 0.9, 0.25, "brass", this.controller);
    w.box(
      "controller pointer",
      [0.4, 0, 0.2],
      [0.65, 0.12, 0.12],
      "ink",
      this.controller,
    );
    pipe(
      "controller readiness rail",
      [
        [-5, 3, 0.6],
        [5, 3.7, 0.6],
        [6, 0, 0.6],
      ],
      "green",
    );
    bed("collecting beam", [6, 0, 0.45], [1.4, 3.5, 0.25], "wait");
    this.gate = root("release arm", [6, -1.4, 1.3], "wait");
    w.box("stop arm", [0, 1.35, 0], [0.16, 2.7, 0.18], "brass", this.gate);
    pipe("collected result through encoding", [
      [6, 0, 0.6],
      [15, 0, 0.6],
    ]);
    const gauge = bed(
      "notched encoding gauge",
      [10, 0, 0.35],
      [3.5, 2.2, 0.2],
      "precision",
    );
    for (let i = 0; i < 8; i++)
      w.box(
        "representable notch " + i,
        [-1.4 + i * 0.4, 0.35, 0.25],
        [0.1, 0.8, 0.3],
        "ink",
        gauge,
      );
    this.needle = root("value read pointer", [10, 0, 0.9], "precision");
    w.box("read pointer", [0, 0, 0], [0.12, 1.7, 0.13], "teal", this.needle);
    w.cyl("block scale wheel", [-1, -0.6, 0.35], 0.6, 0.2, "brass", gauge);
    bed("encoded result receiver", [14, 0, 0.4], [2, 2, 0.22], "precision");

    const reservoir = bed(
      "retained weights",
      [-11, 5, 0.4],
      [5, 2.4, 0.25],
      "supply",
    );
    for (let i = 0; i < 18; i++)
      w.ball(
        "stored weight remains " + i,
        [-2 + (i % 6) * 0.8, -0.6 + Math.floor(i / 6) * 0.6, 0.55],
        0.38,
        reservoir,
      );
    this.loader = root("weight delivery fork", [-6.5, 5, 0.7], "supply");
    w.box("loader spine", [0, 0, 0], [0.15, 1.6, 0.18], "steel", this.loader);
    for (const y of [-0.65, 0.65])
      w.box(
        "loader finger",
        [0.6, y, 0],
        [1.3, 0.15, 0.18],
        "steel",
        this.loader,
      );
    pipe(
      "weights to loading dock",
      [
        [-8.5, 5, 0.65],
        [-2, 5, 0.65],
      ],
      "brass",
    );
    const staging = bed(
      "reusable staging slots",
      [0, 5, 0.35],
      [5, 2.3, 0.25],
      "supply",
    );
    for (let i = 0; i < 3; i++) {
      this.slots.push(
        w.box(
          "staging slot " + i,
          [-1.6 + i * 1.6, 0, 0.25],
          [1.25, 1.6, 0.15],
          "steel",
          staging,
          true,
        ),
      );
      const copy = root("weight tile copy " + i, [-8, 5, 0.9], "supply");
      for (let j = 0; j < 3; j++)
        w.ball("weight tile ball " + j, [-0.35 + j * 0.35, 0, 0], 0.3, copy);
      this.weights.push(copy);
      pipe(
        "staging slot to consumers " + i,
        [
          [-1.6 + i * 1.6, 5, 0.7],
          [-1.6 + i * 1.6, 3.7, 2.2],
          [0, 3.7, 2.2],
        ],
        "brass",
      );
    }

    pipe(
      "shared weight supply above independent consumers",
      [
        [0, 3.7, 2.2],
        [0, -3.1, 2.2],
      ],
      "brass",
    );
    for (let i = 0; i < 6; i++)
      pipe(
        "weight drop to consumer " + i,
        [
          [0, consumerY(i), 2.2],
          [0, consumerY(i), 1.1],
        ],
        "brass",
      );

    for (let request = 0; request < 2; request++) {
      const y = -5.4 - request * 2;
      const rack = bed(
        "request " + (request ? "B" : "A") + " private context",
        [-2, y, 0.25],
        [18, 1.45, 0.25],
        "context",
      );
      for (let i = 0; i < 9; i++) {
        const x = -7.8 + i * 1.95;
        for (const row of [-0.28, 0.28])
          w.box("K/V shelf base", [x, row, 0.2], [1.5, 0.4, 0.15], "ink", rack);
        this.context[request].push(
          w.box(
            "retained context position " + i,
            [x, 0, 0.55],
            [1.3, 0.85, 0.45],
            "paper",
            rack,
            true,
          ),
        );
      }
      const tag = root(
        "request " + (request ? "B" : "A") + " identity gate",
        [-13, y, 0.7],
        "requests",
      );
      for (let tooth = 0; tooth <= request; tooth++)
        w.box(
          "request identity notch",
          [tooth * 0.35, 0, 0],
          [0.2, 1.1, 0.4],
          request ? "green" : "teal",
          tag,
        );
      this.requestGates.push(tag);
      pipe(
        "request queue to shared intake " + request,
        [
          [-13, y, 0.5],
          [-15, y, 0.5],
          [-15, 0, 0.5],
        ],
        request ? "green" : "teal",
      );
    }
    pipe(
      "retained context read copies",
      [
        [-3, -5.4, 0.9],
        [-7, -4, 0.9],
        [-7.5, 0, 0.9],
      ],
      "green",
    );
    for (let i = 0; i < 3; i++) {
      const read = root("context read copy " + i, [-3, -5.4, 0.9], "context");
      w.box("K copy", [-0.17, 0, 0], [0.25, 0.45, 0.25], "green", read);
      w.box("V copy", [0.17, 0, 0], [0.25, 0.45, 0.25], "teal", read);
      this.reads.push(read);
    }
    const selector = bed(
      "final normalization and vocabulary",
      [10, 5, 0.35],
      [11, 2.7, 0.25],
      "choice",
    );
    for (const x of [-4.5, -3.8])
      w.cyl("final sizing roller", [x, 0, 0.65], 0.4, 1, "steel", selector);
    w.box(
      "vocabulary projection press",
      [-2.4, 0, 0.9],
      [1.5, 1.7, 0.22],
      "steel",
      selector,
    );
    for (let i = 0; i < 4; i++)
      this.candidates.push(
        w.box(
          "candidate score " + i,
          [-0.8 + i * 1.4, 0, 0.5],
          [0.8, 1.5, 0.8],
          i === 0 ? "teal" : "steel",
          selector,
        ),
      );
    pipe("final hidden state to vocabulary", [
      [14, 0, 0.6],
      [16, 0, 0.6],
      [16, 7, 0.6],
      [5, 7, 0.6],
      [5, 5, 0.6],
    ]);
    pipe(
      "selected token feedback",
      [
        [15, 5, 0.65],
        [17, 5, 0.65],
        [17, 8.3, 0.65],
        [-16, 8.3, 0.65],
        [-16, 0, 0.65],
      ],
      "green",
    );
    this.returnToken = root("selected next token", [15, 5, 0.85], "choice");
    w.box(
      "selected token returns",
      [0, 0, 0],
      [0.9, 0.65, 0.2],
      "teal",
      this.returnToken,
    );

    const labelPoints: Record<MaterialTopic, Point> = {
      cart: [-12, 2.2, 1],
      context: [-2, -4.1, 1],
      split: [-3.5, -3.2, 1],
      supply: [-4, 7.1, 1],
      worker: [2.3, 3.6, 1],
      wait: [6, -2.8, 1],
      precision: [10, -2.1, 1],
      choice: [11, 7.1, 1],
      requests: [-13, -7.7, 1],
    };
    for (const [id, title] of materialTopics) {
      const label = w.label(title, labelPoints[id], "material", () =>
        this.onSelect(id),
      );
      label.dataset.topic = id;
    }
    for (let request = 0; request < 2; request++)
      for (let i = 0; i < 9; i++) {
        const label = w.label(
          `${request ? "B" : "A"}${i + 1}`,
          [-9.8 + i * 1.95, -6.2 - request * 2, 0.5],
          "material",
          () => this.onSelect("context"),
        );
        label.dataset.topic = "context";
        label.dataset.contextSlot = String(i);
        label.dataset.request = String(request);
      }
    for (let i = 0; i < 7; i++) {
      const tray = w.journey.tray(w.groups.stack, "request B token " + i);
      tray.root.metadata = { materialTopic: "requests" };
      w.box(
        "request B double notch one",
        [-0.2, -0.4, 0.1],
        [0.12, 0.2, 0.2],
        "green",
        tray.root,
      );
      w.box(
        "request B double notch two",
        [0.2, -0.4, 0.1],
        [0.12, 0.2, 0.2],
        "green",
        tray.root,
      );
      this.secondTrays.push(tray);
    }
  }
  focus(topic = this.model.topic) {
    this.host.camera.setTarget(position(anchors[topic]));
    this.host.camera.radius =
      innerWidth <= 650
        ? topic === "context"
          ? 20
          : 12
        : topic === "context"
          ? 25
          : 18;
    this.host.camera.beta = 0.68;
    this.host.camera.alpha = Math.PI / 2 + 0.07;
  }
  update() {
    const s = (this.state = this.model.snapshot()),
      w = this.host;
    const b = s.requests[1];
    this.secondTrays.forEach((tray, i) => {
      tray.root.setEnabled(!!b?.active && (b.prefill || i === 0));
      if (b?.active) {
        const p = stackCarrierProgress(
          b.layer,
          b.layer < 34 ? b.layer + 1 : undefined,
          b.layerProgress,
          0.97,
          i * 0.035,
        );
        w.journey.place(tray, [p, p], 0, b.layer + b.layerProgress);
        tray.root.position.y += 0.2;
      }
    });
    if (w.view !== "material") return;
    const p = s.progress,
      working = s.active && (s.weighted || s.part === "attention");
    this.identityNotches[1].setEnabled(s.request === 1);
    const move = (node: TransformNode, path: Point[], progress: number) =>
      node.position.copyFrom(position(samplePath(path, clamp(progress))));
    this.pins.forEach((pin, i) => {
      const value = s.channels?.current[i];
      const height = value === undefined ? 0.35 : 0.18 + Math.abs(value) * 1.15;
      pin.scaling.y = height;
      pin.position.y = 0.2 + height / 2;
      pin.position.z = value !== undefined && value < 0 ? 0.6 : 0.2;
      const before = s.channels?.input[i];
      this.beforePins[i].scaling.y =
        before === undefined ? 0.35 : 0.18 + Math.abs(before) * 1.15;
      this.beforePins[i].position.y = 0.2 + this.beforePins[i].scaling.y / 2;
    });
    this.cart.setEnabled(s.active && (!working || p < 0.28 || s.released));
    move(
      this.cart,
      s.released
        ? [
            [6, 0, 0.9],
            [14, 0, 0.9],
          ]
        : [
            [-15, 0, 0.9],
            [-7.5, 0, 0.9],
          ],
      s.released ? (p - 0.94) / 0.06 : p / 0.28,
    );
    this.partials.forEach((tray, i) => {
      const x = 0,
        y = consumerY(i),
        q = s.branches[i];
      tray.setEnabled(working && p >= 0.28 && p < 0.94);
      move(
        tray,
        q < 0.72
          ? [
              [-7.5, 0, 0.9],
              [-6, y, 0.9],
              [x, y, 0.9],
            ]
          : [
              [x, y, 0.9],
              [5, y, 0.9],
              [6, 0, 0.9],
            ],
        q < 0.72 ? q / 0.72 : (q - 0.72) / 0.28,
      );
      this.presses[i].position.y =
        1.4 - (working ? 0.45 * pressStroke(clamp((q - 0.22) / 0.5)) : 0);
    });
    this.gate.rotation.x = s.illustrated
      ? s.released
        ? -Math.PI / 2
        : 0
      : s.waiting
        ? 0
        : -Math.PI / 2;
    this.controller.rotation.y = s.illustrated && working ? p * Math.PI * 2 : 0;
    this.loader.position.x =
      6.5 -
      (s.illustrated && working
        ? 1.5 * Math.sin(clamp(p / 0.28) * Math.PI)
        : 0);
    this.slots.forEach((slot, i) => {
      const state = s.slots?.[i];
      slot.material =
        w.materials[
          !state
            ? "paper"
            : state === "reading"
              ? "teal"
              : state === "ready"
                ? "amber"
                : "steel"
        ];
      const copy = this.weights[i];
      copy.setEnabled(
        s.illustrated &&
          s.weighted &&
          s.active &&
          p >= 0.03 + i * 0.055 &&
          p < 0.815,
      );
      const load = clamp((p - 0.03 - i * 0.055) / 0.07),
        x = -1.6 + i * 1.6;
      move(
        copy,
        p < 0.28
          ? [
              [-8, 5, 0.9],
              [x, 5, 0.9],
            ]
          : [
              [x, 5, 0.9],
              [x, 3.7, 2.4],
              [0, 3.7, 2.4],
              [0, consumerY(i * 2), 2.4],
            ],
        p < 0.28 ? load : clamp((p - 0.28) / 0.53),
      );
    });
    this.context.forEach((rack, request) => {
      const r = s.requests[request];
      rack.forEach((slot, i) => {
        const entry = request === s.request ? s.entries?.[i] : null;
        const stored = r ? i < r.cache : false;
        slot.material =
          w.materials[
            !s.illustrated
              ? "paper"
              : entry?.eligible
                ? "teal"
                : stored
                  ? "green"
                  : "paper"
          ];
        slot.scaling.y = stored ? 0.45 : 0.12;
      });
      rack[0].parent!.setEnabled(request === 0 || s.requests.length === 2);
      this.requestGates[request].setEnabled(
        request === 0 || s.requests.length === 2,
      );
    });
    this.reads.forEach((node, i) => {
      const eligible = s.entries?.filter((e) => e.eligible) ?? [];
      node.setEnabled(
        s.illustrated &&
          s.active &&
          s.part === "attention" &&
          eligible.length > 0 &&
          p > i * 0.1 &&
          p < 0.95,
      );
      const entry = eligible[Math.min(i, eligible.length - 1)]?.index ?? 0;
      move(
        node,
        [
          [-9.8 + entry * 1.95, -5.4 - s.request * 2, 0.9],
          [-7, -4, 0.9],
          [-7.5, 0, 0.9],
        ],
        (p - i * 0.1) / (0.95 - i * 0.1),
      );
    });
    this.needle.position.x = -(8.6 + (s.rounding?.code ?? 3.5) * 0.4);
    this.candidates.forEach((bar, i) => {
      const height =
        s.choiceStage || s.returning
          ? 0.15 +
            (s.candidates![i].score + 0.3) *
              (s.returning ? 1 : clamp(s.choosing / 0.65))
          : 0.15;
      bar.scaling.y = height;
      bar.position.y = 0.2 + height / 2;
    });
    this.returnToken.setEnabled(
      !!s.selectedCandidate && s.selectedCandidate !== "End",
    );
    move(
      this.returnToken,
      [
        [15, 5, 0.9],
        [17, 5, 0.9],
        [17, 8.3, 0.9],
        [-15, 8.3, 0.9],
        [-15, 0, 0.9],
      ],
      s.feedbackProgress,
    );
    for (const label of w.labels.filter((l) => l.view === "material"))
      label.element.classList.toggle(
        "selected",
        label.element.dataset.topic === this.model.topic,
      );
  }
}
