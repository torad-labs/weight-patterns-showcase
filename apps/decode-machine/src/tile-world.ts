import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { AbstractMesh } from "@babylonjs/core/Meshes/abstractMesh";
import type { MechanicalWorld } from "./replay-world";
import {
  TileFlow,
  tileStations,
  type TilePose,
  type TileStation,
  type ToolKind,
} from "./tile-flow";

type Point = [number, number, number];
const at = (x: number, y: number, z = 0) => new Vector3(-x, z, -y);
export const tileLocations: Record<TileStation, Point> = {
  storage: [-17, 8, 0],
  cache: [-9, 8, 0],
  loader: [-1, 8, 0],
  staging: [7, 8, 0],
  layout: [16, 6, 0],
  registers: [16, -1, 0],
  scheduler: [4, 1, 0],
  compute: [6, -5, 0],
  encode: [-3, -4, 0],
  handoff: [-13, -4, 0],
  capacity: [4, -11, 0],
};
const toolOffsets: Record<ToolKind, [number, number]> = {
  matrix: [1.7, 0.9],
  arithmetic: [-1.7, 0.9],
  compare: [-1.7, -2],
  exponential: [1.7, -2],
};

/** One connected cutaway of the selected block's supplies and execution. Geometry is a teaching arrangement. */
export class TileWorld {
  readonly root: TransformNode;
  readonly model: TileFlow;
  onSelect: (station: TileStation) => void = () => {};
  private rigs = new Map<TileStation, TransformNode>();
  private beds = new Map<TileStation, AbstractMesh>();
  private labels = new Map<TileStation, HTMLButtonElement>();
  private slots: {
    bed: AbstractMesh;
    contents: TransformNode;
    peg: AbstractMesh;
  }[] = [];
  private crane: TransformNode;
  private read: TransformNode;
  private request: TransformNode;
  private fragments: TransformNode;
  private information: TransformNode;
  private sorter: TransformNode[] = [];
  private tools = new Map<
    ToolKind,
    { moving: TransformNode; bed: AbstractMesh }
  >();
  private accumulation: AbstractMesh[] = [];
  private encoded: TransformNode;
  private output: TransformNode;
  private latch: TransformNode;
  private counters: AbstractMesh[] = [];
  private results: TransformNode[] = [];
  private clutch: TransformNode;
  private crews: AbstractMesh[] = [];
  private registerBudget: AbstractMesh[] = [];
  private sharedBudget: AbstractMesh[] = [];
  private residents: TransformNode[] = [];
  private congestion: AbstractMesh;
  private jaws: TransformNode;
  state?: TilePose;

  constructor(
    private host: MechanicalWorld,
    model: TileFlow,
  ) {
    const w = host;
    this.model = model;
    this.root = this.node("tile handoff cutaway", [0, 0, 0], w.groups.hardware);
    w.box(
      "connected tile workbench",
      [0, -1, -0.2],
      [42, 28, 0.25],
      "base",
      this.root,
    );
    w.pipe(
      "representative SM boundary",
      [
        [-6, 11.6, 0.1],
        [20, 11.6, 0.1],
        [20, -13.5, 0.1],
        [-6, -13.5, 0.1],
        [-6, 11.6, 0.1],
      ],
      "dark",
      this.root,
      0.08,
    );
    const boundary = w.label(
      "Device memory → representative SM",
      [5, 13, 0.5],
      "hardware",
      () => {},
    );
    boundary.dataset.hardwareDepth = "tile";
    boundary.disabled = true;
    for (const [id, name] of tileStations) {
      const n = this.node(name, tileLocations[id], this.root);
      n.metadata = { tileStation: id };
      this.rigs.set(id, n);
      this.beds.set(
        id,
        w.box(
          name + " base",
          [0, 0, 0.08],
          id === "capacity"
            ? [13, 4.5, 0.25]
            : id === "compute"
              ? [7.5, 6.5, 0.25]
              : [5.4, 4.3, 0.25],
          "paper",
          n,
          true,
        ),
      );
      const [x, y] = tileLocations[id];
      const label = w.label(
        name,
        [x, y + (id === "compute" ? 3.8 : 2.8), 1],
        "hardware",
        () => this.onSelect(id),
      );
      label.dataset.hardwareDepth = "tile";
      label.dataset.tileStation = id;
      this.labels.set(id, label);
    }
    const r = (id: TileStation) => this.rigs.get(id)!;
    // Stored balls never leave their compartments. Travelling packets below are copies.
    for (const id of ["storage", "cache"] as const) {
      for (let row = 0; row < 3; row++) {
        w.box(
          "retained weight shelf",
          [0, row - 1, 0.5],
          [4.4, 0.8, 0.2],
          "ink",
          r(id),
        );
        for (let col = 0; col < 5; col++)
          w.ball(
            "retained weight",
            [col * 0.78 - 1.56, row - 1, 0.85],
            0.45,
            r(id),
          );
      }
    }
    this.rail(
      "storage to L2",
      [
        [-17, 8, 1],
        [-9, 8, 1],
      ],
      "amber",
    );
    this.rail(
      "L2 to loading crane",
      [
        [-9, 8, 1],
        [-1, 8, 1],
        [7, 8, 1],
      ],
      "brass",
    );
    this.rail(
      "shared tray to sorter",
      [
        [7, 8, 0.7],
        [12, 8, 0.7],
        [16, 6, 0.7],
      ],
      "brass",
    );
    this.rail(
      "sorter to private registers",
      [
        [16, 6, 0.7],
        [16, -1, 0.7],
      ],
      "brass",
    );
    this.rail(
      "registers to execution tools",
      [
        [16, -1, 0.7],
        [13, -1, 0.7],
        [13, -4, 0.7],
        [8, -4, 0.7],
      ],
      "teal",
    );
    this.rail(
      "information from the layer",
      [
        [18, -11, 0.7],
        [18, -1, 0.7],
        [16, -1, 0.7],
      ],
      "teal",
    );
    const input = w.label(
      "Information from the layer",
      [17, -12, 0.8],
      "hardware",
      () => {},
    );
    input.dataset.hardwareDepth = "tile";
    input.disabled = true;
    this.rail(
      "computed result to encoder",
      [
        [6, -4, 0.7],
        [-3, -4, 0.7],
      ],
      "teal",
    );
    this.rail(
      "encoded result to collector",
      [
        [-3, -4, 0.7],
        [-13, -4, 0.7],
        [-19, -4, 0.7],
      ],
      "teal",
    );
    this.rail(
      "ready signal to scheduler",
      [
        [7, 7, 0.35],
        [7, 3.5, 0.35],
        [4, 1, 0.35],
      ],
      "green",
    );
    this.rail(
      "issued instruction to tools",
      [
        [4, 1, 0.35],
        [10.5, 1, 0.35],
        [10.5, -5, 0.35],
        [8, -5, 0.35],
      ],
      "green",
    );
    for (const [x, y, left] of [
      [-13, 8, 0],
      [-5, 8, 0],
      [11, 8, 0],
      [16, 3, 1],
      [0, -4, 1],
      [-8, -4, 1],
    ] as const) {
      const a = this.node("route direction", [x, y, 0.4], this.root);
      w.pipe(
        "arrow",
        left
          ? [
              [0.4, 0.35, 0],
              [0, 0, 0],
              [0.4, -0.35, 0],
            ]
          : [
              [-0.4, 0.35, 0],
              [0, 0, 0],
              [-0.4, -0.35, 0],
            ],
        "ink",
        a,
        0.08,
      );
      if (x === 16) a.rotation.y = Math.PI / 2;
    }
    for (const x of [-2, 2])
      w.cyl("crane support", [x, 0, 1.5], 0.2, 3, "ink", r("loader"));
    w.box("crane track", [0, 0, 3], [5, 0.4, 0.25], "steel", r("loader"));
    this.crane = this.node("loading fork", [0, 0, 1.8], r("loader"));
    w.box(
      "loading fork crossbar",
      [0, 0, 0],
      [1.4, 0.2, 0.2],
      "brass",
      this.crane,
    );
    for (const x of [-0.55, 0.55])
      w.box(
        "loading fork tine",
        [x, 0.4, -0.35],
        [0.14, 1, 0.7],
        "brass",
        this.crane,
      );
    this.read = this.packet(
      "retained weight read copy",
      [-17, 8, 1.3],
      this.root,
      "amber",
    );
    this.request = w.box(
      "L2 miss request",
      [-9, 8, 1.4],
      [0.8, 0.45, 0.15],
      "ink",
      this.root,
    );
    for (let i = 0; i < 3; i++) {
      const n = this.node(
        `staging slot ${i}`,
        [0, 1.3 - i * 1.3, 0.45],
        r("staging"),
      );
      const bed = w.box(
        "reusable tray",
        [0, 0, 0],
        [4.5, 0.95, 0.25],
        "steel",
        n,
        true,
      );
      const contents = this.packet(
        "tile in shared tray",
        [0, 0, 0.35],
        n,
        "amber",
      );
      const peg = w.shape(
        "ready peg",
        "cylinder",
        [2, 0, 0.3],
        [0.2, 0.2, 0.4],
        "ink",
        n,
        true,
      );
      this.slots.push({ bed, contents, peg });
    }
    w.cyl("layout turntable", [0, 0, 0.5], 3.5, 0.3, "steel", r("layout"));
    for (let i = 0; i < 4; i++) {
      const n = this.node(
        `unchanged value ${i + 1}`,
        [-1.5 + i, 0, 1],
        r("layout"),
      );
      w.ball(`value ${i + 1}`, [0, 0, 0], 0.5, n);
      for (let mark = 0; mark <= i; mark++)
        w.box(
          "identity notch",
          [(mark - i / 2) * 0.12, -0.36, 0],
          [0.065, 0.16, 0.12],
          "ink",
          n,
        );
      this.sorter.push(n);
      w.box(
        "private register pocket",
        [-1.5 + i, 0, 0.45],
        [0.7, 1.5, 0.6],
        "ink",
        r("registers"),
      );
    }
    this.fragments = this.packet(
      "operand fragments",
      [16, 0, 1],
      this.root,
      "amber",
    );
    this.information = this.packet(
      "same layer information",
      [18, -11, 1],
      this.root,
      "teal",
    );
    this.clutch = this.node(
      "hardware issue clutch",
      [0, 0, 0.65],
      r("scheduler"),
    );
    w.cyl("clutch axle", [0, 0, 0], 0.7, 0.4, "brass", this.clutch);
    w.box("clutch finger", [0, 0.6, 0], [0.2, 1.4, 0.2], "brass", this.clutch);
    for (let i = 0; i < 3; i++)
      this.crews.push(
        w.box(
          ["controller crew", "loader crew", "consumer crew"][i],
          [-1.6 + i * 1.6, -1, 0.5],
          [1.1, 0.8, 0.3],
          "steel",
          r("scheduler"),
          true,
        ),
      );
    for (const kind of [
      "matrix",
      "arithmetic",
      "compare",
      "exponential",
    ] as ToolKind[]) {
      const [x, y] = toolOffsets[kind],
        n = this.node(kind + " working head", [x, y, 0], r("compute"));
      const bed = w.box(
        "tool bed",
        [0, 0, 0.4],
        [2.7, 2.2, 0.4],
        "ink",
        n,
        true,
      );
      const moving = this.node(kind + " motion", [0, 0, 1.5], n);
      if (kind === "matrix") {
        for (const x of [-1, 1])
          w.cyl("press column", [x, 0, 1.3], 0.18, 2, "steel", n);
        w.box(
          "tensor product platen",
          [0, 0, 0],
          [2.5, 1.7, 0.3],
          "steel",
          moving,
        );
        for (let i = 0; i < 4; i++)
          this.accumulation.push(
            w.box(
              "transient accumulated contribution",
              [-0.8 + i * 0.5, 0, 0.65],
              [0.35, 0.9, 0.1],
              "teal",
              n,
              true,
            ),
          );
      } else if (kind === "exponential") {
        w.cyl(
          "shaped transcendental cam",
          [0, 0, 0],
          1.4,
          0.3,
          "brass",
          moving,
        );
        w.box(
          "cam follower",
          [0.65, 0, 0.35],
          [0.25, 1.2, 0.25],
          "steel",
          moving,
        );
      } else {
        w.box(
          kind === "compare" ? "maximum selector" : "addition lever",
          [0, 0, 0],
          [2, 0.35, 0.25],
          "brass",
          moving,
        );
        for (const x of [-0.8, 0.8])
          w.cyl("input cup", [x, 0, 0.8], 0.55, 0.25, "teal", n);
      }
      this.tools.set(kind, { moving, bed });
    }
    this.jaws = this.node("narrow output jaws", [0, 0, 0.7], r("encode"));
    for (const x of [-0.8, 0.8])
      w.box("encoding jaw", [x, 0, 0], [0.4, 2.3, 0.8], "steel", this.jaws);
    this.encoded = this.packet(
      "narrow encoded result",
      [-3, -4, 1],
      this.root,
      "teal",
    );
    w.box(
      "shared scale tag",
      [0, -0.6, 0.05],
      [0.6, 0.25, 0.1],
      "paper",
      this.encoded,
    );
    for (let i = 0; i < 3; i++) {
      this.counters.push(
        w.shape(
          "published completion peg",
          "cylinder",
          [-1.5 + i * 1.5, 0.8, 0.5],
          [0.3, 0.3, 0.8],
          "ink",
          r("handoff"),
          true,
        ),
      );
      this.results.push(
        this.packet(
          `producer ${i + 1} result`,
          [-1.5 + i * 1.5, -0.5, 0.7],
          r("handoff"),
          "teal",
        ),
      );
      this.rail(
        `producer ${i + 1} deposit path`,
        [
          [-13 + (-1.5 + i * 1.5), -7, 0.3],
          [-13 + (-1.5 + i * 1.5), -4.5, 0.3],
        ],
        "green",
      );
    }
    this.latch = this.node(
      "dependent consumer latch",
      [-16, -4, 0.7],
      this.root,
    );
    w.box(
      "result release gate",
      [0, 0, 0],
      [0.25, 3, 0.4],
      "brass",
      this.latch,
    );
    this.output = this.packet(
      "released result",
      [-16, -4, 1],
      this.root,
      "teal",
    );
    for (let i = 0; i < 8; i++) {
      this.registerBudget.push(
        w.box(
          "finite register unit",
          [-5 + i * 0.65, 0.7, 0.45],
          [0.5, 0.65, 0.5],
          "steel",
          r("capacity"),
          true,
        ),
      );
      this.sharedBudget.push(
        w.box(
          "finite shared-memory unit",
          [-5 + i * 0.65, -0.3, 0.45],
          [0.5, 0.65, 0.5],
          "steel",
          r("capacity"),
          true,
        ),
      );
    }
    for (let i = 0; i < 3; i++) {
      const n = this.node(
        `admission block ${i + 1}`,
        [1.6 + i * 1.6, 0.1, 0.45],
        r("capacity"),
      );
      w.box(
        "block workspace reservation",
        [0, 0, 0],
        [1.25, 1.8, 0.35],
        "teal",
        n,
      );
      this.residents.push(n);
    }
    this.congestion = w.box(
      "occupied supply road",
      [-5, 8, 1.5],
      [1, 1.5, 0.5],
      "wait",
      this.root,
    );
    this.root.setEnabled(false);
  }
  private node(name: string, p: Point, parent: TransformNode) {
    const n = new TransformNode(name, this.host.scene);
    n.parent = parent;
    n.position = at(...p);
    return n;
  }
  private packet(
    name: string,
    p: Point,
    parent: TransformNode,
    color: "teal" | "amber",
  ) {
    const n = this.node(name, p, parent);
    this.host.box("tile carrier", [0, 0, 0], [1.5, 0.9, 0.18], "ink", n);
    for (let i = 0; i < 4; i++) {
      if (color === "amber")
        this.host.ball(
          "copied weight",
          [-0.5 + (i % 2), Math.floor(i / 2) * 0.4 - 0.2, 0.3],
          0.33,
          n,
        );
      else
        this.host.box(
          "encoded information",
          [-0.5 + (i % 2), Math.floor(i / 2) * 0.4 - 0.2, 0.25],
          [0.35, 0.25, 0.3],
          "teal",
          n,
        );
    }
    return n;
  }
  private rail(name: string, points: Point[], color: string) {
    this.host.pipe(name, points, color, this.root, 0.095);
  }
  focus() {
    const p = at(...tileLocations[this.model.station]);
    this.host.camera.setTarget(p);
    this.host.camera.radius = 15;
    this.host.camera.beta = 0.6;
  }
  update(s: TilePose) {
    this.state = s;
    const w = this.host,
      m = this.model,
      t = s.progress;
    for (const [id, bed] of this.beds) {
      bed.material = w.materials[id === m.station ? "steel" : "paper"];
      this.labels.get(id)!.classList.toggle("selected", id === m.station);
    }
    const delivery = s.delivery,
      transferring = delivery?.stage === "deliver";
    const lift = transferring ? clamp01((delivery.progress - 0.375) / 0.25) : 0;
    this.crane.position.x = -(transferring ? lift * 4 - 2 : 0);
    this.crane.position.y =
      1.8 - (transferring ? Math.sin(lift * Math.PI) * 0.5 : 0);
    this.request.setEnabled(delivery?.stage === "request");
    if (delivery?.stage === "request")
      this.request.position.copyFrom(at(-9 - 8 * delivery.progress, 8, 1.4));
    this.read.setEnabled(!!delivery && delivery.stage !== "request");
    if (delivery)
      this.read.position.copyFrom(
        at(
          delivery.stage === "refill"
            ? -17 + 8 * delivery.progress
            : -9 + 16 * delivery.progress,
          8,
          1.2,
        ),
      );
    this.slots.forEach((slot, i) => {
      const v = s.slots[i];
      slot.bed.material =
        w.materials[
          v.state === "in use"
            ? "teal"
            : v.state === "ready"
              ? "green"
              : v.state === "filling"
                ? "brass"
                : "steel"
        ];
      slot.contents.setEnabled(
        v.tile !== null &&
          v.state !== "released" &&
          (v.state !== "filling" || (transferring && delivery.progress > 0.8)),
      );
      slot.contents.scaling.x =
        v.state === "filling" && transferring
          ? Math.max(0.1, (delivery.progress - 0.8) / 0.2)
          : 1;
      slot.peg.position.y = v.state === "ready" ? 0.9 : 0.3;
    });
    const sort = s.sorting ? s.sortProgress : s.latest ? 1 : 0;
    this.sorter.forEach((n, i) =>
      n.position.copyFrom(
        at(
          (-1.5 + i) * (1 - sort) + ((i % 2) * 1.4 - 0.7) * sort,
          (Math.floor(i / 2) * 1.4 - 0.7) * sort,
          1,
        ),
      ),
    );
    const [ox, oy] = toolOffsets[m.tool];
    this.fragments.setEnabled(!!s.using);
    this.fragments.position.copyFrom(
      along(
        [
          [16, -1, 1],
          [13, -1, 1],
          [13, -4, 1],
          [6 + ox, -5 + oy, 1],
        ],
        Math.min(1, s.computeProgress * 4),
      ),
    );
    this.information.setEnabled(!!s.using);
    this.information.position.copyFrom(
      along(
        [
          [18, -11, 1],
          [18, -1, 1],
          [16, -1, 1],
          [13, -1, 1],
          [13, -4, 1],
          [6 + ox, -5 + oy, 1],
        ],
        Math.min(1, s.computeProgress * 4),
      ),
    );
    const issuedIndex =
      s.issued === "controller"
        ? 0
        : s.issued === "loader"
          ? 1
          : s.issued === "consumer"
            ? 2
            : -1;
    this.crews.forEach(
      (mesh, i) =>
        (mesh.material = w.materials[i === issuedIndex ? "teal" : "steel"]),
    );
    this.clutch.rotation.y =
      issuedIndex < 0 ? Math.PI / 2 : (issuedIndex - 1) * 0.85;
    for (const [kind, tool] of this.tools) {
      const working = kind === m.tool && !!s.using && !s.held;
      tool.bed.material = w.materials[working ? "teal" : "ink"];
      tool.moving.position.y =
        1.5 - (working ? Math.sin(s.computeProgress * Math.PI) * 0.65 : 0);
      tool.moving.rotation.y =
        working && kind === "exponential" ? s.computeProgress * Math.PI * 2 : 0;
      tool.moving.rotation.z =
        working && kind !== "matrix" && kind !== "exponential"
          ? Math.sin(s.computeProgress * Math.PI) * 0.35
          : 0;
    }
    this.accumulation.forEach((mesh, i) => {
      mesh.setEnabled(m.tool === "matrix" && s.active && t < 0.82);
      mesh.scaling.y = i < s.completed ? 0.6 : 0.1;
    });
    this.jaws.scaling.x = s.encoding ? 0.65 : 1;
    this.encoded.setEnabled(s.active && t >= 0.76 && t < 0.92);
    this.encoded.position.copyFrom(
      t < 0.78
        ? along(
            [
              [6 + ox, -5 + oy, 1],
              [6, -4, 1],
              [-3, -4, 1],
            ],
            clamp01((t - 0.76) / 0.02),
          )
        : at(-3 - 10 * clamp01((t - 0.82) / 0.1), -4, 1),
    );
    this.counters.forEach((mesh, i) => {
      mesh.position.y = s.contributions[i] ? 1.1 : 0.5;
      mesh.material = w.materials[s.contributions[i] ? "teal" : "ink"];
    });
    this.results.forEach((n, i) => n.setEnabled(s.contributions[i]));
    this.latch.rotation.y = s.gateOpen ? Math.PI / 2 : 0;
    this.output.setEnabled(s.gateOpen);
    this.output.position.copyFrom(at(-16 - 4 * s.outputProgress, -4, 1));
    this.registerBudget.forEach(
      (mesh, i) =>
        (mesh.material =
          w.materials[
            i < s.resident * s.capacity.registersPerBlock ? "teal" : "steel"
          ]),
    );
    this.sharedBudget.forEach(
      (mesh, i) =>
        (mesh.material =
          w.materials[
            i < s.resident * s.capacity.sharedPerBlock ? "brass" : "steel"
          ]),
    );
    this.residents.forEach((n, i) => {
      n.position.z = i < s.resident ? -0.1 : 1.5;
      n.scaling.y = i < s.resident ? 1 : 0.5;
    });
    this.congestion.setEnabled(s.held);
    this.congestion.position.copyFrom(
      at(
        m.pressure === "supply" ? -5 : 9,
        m.pressure === "supply" ? 8 : -4,
        1.5,
      ),
    );
  }
}
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
function along(points: Point[], phase: number) {
  const lengths = points
    .slice(1)
    .map((p, i) => Vector3.Distance(at(...p), at(...points[i])));
  let remaining = clamp01(phase) * lengths.reduce((a, b) => a + b, 0);
  for (let i = 0; i < lengths.length; i++) {
    if (remaining <= lengths[i])
      return Vector3.Lerp(
        at(...points[i]),
        at(...points[i + 1]),
        lengths[i] ? remaining / lengths[i] : 0,
      );
    remaining -= lengths[i];
  }
  return at(...points[points.length - 1]);
}
