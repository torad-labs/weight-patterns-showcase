import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { Vector3, Matrix } from "@babylonjs/core/Maths/math.vector";
import { Color3, Color4 } from "@babylonjs/core/Maths/math.color";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import "@babylonjs/core/Meshes/instancedMesh";
import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { ShaderMaterial } from "@babylonjs/core/Materials/shaderMaterial";
import { PointerEventTypes } from "@babylonjs/core/Events/pointerEvents";
import "@babylonjs/core/Shaders/ShadersInclude/instancesDeclaration";
import "@babylonjs/core/Shaders/ShadersInclude/instancesVertex";
import { type Capture, type MechanismId } from "./replay-model";
import { stackCenter, stackPath } from "./stack-route";
import { InternalWorkshop } from "./workshop-world";
import { JourneyWorld } from "./journey-world";
import type { Timeline } from "./timeline";
import { RecordedMotion, type WorkerState } from "./recorded-motion";
import { updateMachinery } from "./machine-motion";
import type { View } from "./replay-playback";
import type { MaterialWorld } from "./material-world";
import type { HardwareWorld } from "./hardware-world";
import { InspectionMotion, inspectionDirection } from "./inspection-motion";
import { SupplyWorld } from "./supply-world";
import type { TileStation } from "./tile-flow";
export type { View } from "./replay-playback";

type V = [number, number, number];
type Label = {
  element: HTMLButtonElement;
  leader: SVGLineElement;
  point: Vector3;
  anchor?: Vector3;
  view: View;
  layer?: number;
  block?: number;
  part?: MechanismId;
  internal?: string;
};
const palette = {
  ink: "#344c4b",
  steel: "#91a4a7",
  base: "#b9c5bd",
  paper: "#e9e9dd",
  brass: "#c59854",
  amber: "#dc9c4a",
  teal: "#378b82",
  green: "#729e92",
  wait: "#8a78a6",
  dark: "#567875",
};
const B = (x: number, y: number, z = 0) => new Vector3(-x, z, -y);
export class MechanicalWorld {
  engine: Engine;
  scene: Scene;
  camera: ArcRotateCamera;
  inspection: InspectionMotion;
  nextInspectionDirection?: "in" | "out";
  nextInspectionOrigin?: Vector3;
  supplies: SupplyWorld;
  onSupply: (station: TileStation, origin?: Vector3) => void = () => {};
  resizeFraming?: () => void;
  userMovedCamera = false;
  view: View = "stack";
  selectedLayer = 0;
  selectedBlock = 0;
  selectedPart: MechanismId = "attention";
  focusedBlock: number | null = null;
  groups: Record<View, TransformNode>;
  labels: Label[] = [];
  materials: Record<string, ShaderMaterial> = {};
  leaders: SVGSVGElement;
  prototypes = new Map<string, Mesh>();
  macros: { piston: Mesh; lamp: Mesh; root: TransformNode }[] = [];
  workers: {
    piston: Mesh;
    latch: TransformNode;
    lamp: Mesh;
    reads: TransformNode[];
  }[] = [];
  moving: {
    part: MechanismId;
    node: TransformNode;
    motion: "press" | "rotate" | "gate";
    base: Vector3;
    role: string;
  }[] = [];
  detail: TransformNode;
  workerStates: WorkerState[] = [];
  onLayer: (n: number) => void = () => {};
  onPart: (part: MechanismId) => void = () => {};
  onBlock: (n: number) => void = () => {};
  workshop: InternalWorkshop;
  onInternal: (id: string) => void = () => {};
  story: Timeline | null = null;
  journey!: JourneyWorld;
  followTray = innerWidth <= 650;
  onEndpoint: (end: "input" | "output") => void = () => {};
  recorded: RecordedMotion;
  material?: MaterialWorld;
  onMaterial: (topic: string) => void = () => {};
  hardware?: HardwareWorld;
  onHardwareWarp: (warp: number) => void = () => {};
  onHardwareLane: (warp: number, lane: number) => void = () => {};
  constructor(
    public canvas: HTMLCanvasElement,
    public overlay: HTMLElement,
    public capture: Capture,
  ) {
    this.recorded = new RecordedMotion(capture);
    this.leaders = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "svg",
    );
    this.leaders.classList.add("label-leaders");
    this.leaders.setAttribute("aria-hidden", "true");
    overlay.append(this.leaders);
    this.engine = new Engine(canvas, true, {
      preserveDrawingBuffer: true,
      stencil: true,
    });
    this.engine.setHardwareScalingLevel(1);
    this.scene = new Scene(this.engine);
    this.scene.clearColor = Color4.FromHexString("#e8e8e1ff");
    this.camera = new ArcRotateCamera(
      "observer",
      Math.PI / 2,
      0.56,
      48,
      B(0, 0, 0),
      this.scene,
    );
    this.camera.attachControl(canvas, true);
    this.camera.lowerRadiusLimit = 5;
    this.camera.upperRadiusLimit = 110;
    this.camera.lowerBetaLimit = 0.18;
    this.camera.upperBetaLimit = 1.35;
    this.camera.wheelPrecision = 25;
    this.camera.pinchPrecision = 80;
    this.camera.panningSensibility = 90;
    this.camera.minZ = 0.1;
    this.inspection = new InspectionMotion(canvas, overlay, this.camera);
    for (const event of ["pointerdown", "wheel"])
      canvas.addEventListener(
        event,
        () => {
          this.resizeFraming = undefined;
          this.userMovedCamera = true;
        },
        { passive: true },
      );
    const vertexSource = `precision highp float;attribute vec3 position;attribute vec3 normal;uniform mat4 viewProjection;varying vec3 vNormal;
  #include<instancesDeclaration>
  void main(){
  #include<instancesVertex>
  vNormal=normalize(mat3(finalWorld)*normal);gl_Position=viewProjection*finalWorld*vec4(position,1.0);}`;
    const fragmentSource = `precision highp float;varying vec3 vNormal;uniform vec3 baseColor;void main(){float l=dot(normalize(vNormal),normalize(vec3(-.4,.85,.35)));float band=l>.55?1.0:(l>.02?.81:.62);gl_FragColor=vec4(baseColor*band,1.0);}`;
    for (const [name, color] of Object.entries(palette)) {
      const m = new ShaderMaterial(
        name,
        this.scene,
        { vertexSource, fragmentSource },
        {
          attributes: ["position", "normal"],
          uniforms: ["world", "viewProjection", "baseColor"],
        },
      );
      m.setColor3("baseColor", Color3.FromHexString(color));
      this.materials[name] = m;
    }
    this.groups = {
      stack: new TransformNode("layer installation", this.scene),
      layer: new TransformNode("open layer", this.scene),
      engine: new TransformNode("persistent worker floor", this.scene),
      internals: new TransformNode("inside the equipment", this.scene),
      material: new TransformNode("cart and connected supplies", this.scene),
      hardware: new TransformNode(
        "representative SM and warp lanes",
        this.scene,
      ),
    };
    this.workshop = new InternalWorkshop(this);
    this.detail = new TransformNode("selected layer mechanisms", this.scene);
    this.detail.parent = this.groups.layer;
    this.buildStack();
    this.buildWorkers();
    this.supplies = new SupplyWorld(this);
    this.buildLayer();
    this.journey = new JourneyWorld(this);
    this.setView("stack");
    this.scene.onPointerObservable.add((info) => {
      if (info.type !== PointerEventTypes.POINTERPICK) return;
      let node = info.pickInfo?.pickedMesh;
      while (node) {
        const meta = node.metadata;
        if (meta?.supplyStation) {
          this.onSupply(meta.supplyStation, node.getAbsolutePosition());
          return;
        }
        if (meta?.tileStation && this.hardware) {
          this.hardware.tile.onSelect(meta.tileStation);
          return;
        }
        if (meta?.hardwareLane !== undefined) {
          this.onHardwareLane(meta.hardwareWarp, meta.hardwareLane);
          return;
        }
        if (meta?.hardwareWarp !== undefined) {
          this.onHardwareWarp(meta.hardwareWarp);
          return;
        }
        if (meta?.materialTopic) {
          this.onMaterial(meta.materialTopic);
          return;
        }
        if (meta?.internal) {
          this.onInternal(meta.internal);
          return;
        }
        if (meta?.part) {
          this.onPart(meta.part);
          return;
        }
        if (meta?.layer !== undefined) {
          this.onLayer(meta.layer);
          return;
        }
        if (meta?.block !== undefined) {
          this.onBlock(meta.block);
          return;
        }
        node = node.parent as Mesh | null;
      }
    });
    let fittedWidth = 0,
      fittedHeight = 0;
    new ResizeObserver(() => {
      const { width, height } = canvas.getBoundingClientRect();
      if (
        Math.abs(width - fittedWidth) < 1 &&
        Math.abs(height - fittedHeight) < 1
      )
        return;
      fittedWidth = width;
      fittedHeight = height;
      this.engine.resize();
      if (this.followTray || this.userMovedCamera) return;
      const frame = () =>
        this.resizeFraming ? this.resizeFraming() : this.fit();
      if (this.inspection.active) this.inspection.reframe(frame);
      else frame();
    }).observe(canvas);
  }
  shape(
    name: string,
    shape: "box" | "cylinder" | "sphere",
    at: V,
    size: V,
    color: string,
    parent: TransformNode,
    mutable = false,
  ) {
    const key = shape + color;
    let proto = this.prototypes.get(key);
    if (!proto) {
      proto =
        shape === "box"
          ? MeshBuilder.CreateBox(key, { size: 1 }, this.scene)
          : shape === "cylinder"
            ? MeshBuilder.CreateCylinder(
                key,
                { diameter: 1, height: 1, tessellation: 16 },
                this.scene,
              )
            : MeshBuilder.CreateSphere(
                key,
                { diameter: 1, segments: 8 },
                this.scene,
              );
      proto.material = this.materials[color];
      proto.isVisible = false;
      proto.isPickable = false;
      this.prototypes.set(key, proto);
    }
    const mesh = mutable
      ? proto.clone(name, null, true)!
      : proto.createInstance(name);
    mesh.isVisible = true;
    mesh.isPickable = true;
    mesh.parent = parent;
    mesh.position = B(...at);
    mesh.scaling.set(size[0], size[2], size[1]);
    return mesh;
  }
  box(
    name: string,
    at: V,
    size: V,
    color: string,
    parent: TransformNode,
    mutable = false,
  ) {
    return this.shape(name, "box", at, size, color, parent, mutable);
  }
  cyl(
    name: string,
    at: V,
    diameter: number,
    height: number,
    color: string,
    parent: TransformNode,
  ) {
    return this.shape(
      name,
      "cylinder",
      at,
      [diameter, diameter, height],
      color,
      parent,
    );
  }
  ball(name: string, at: V, size: number, parent: TransformNode) {
    return this.shape(name, "sphere", at, [size, size, size], "amber", parent);
  }
  pipe(
    name: string,
    points: V[],
    color: string,
    parent: TransformNode,
    radius = 0.045,
  ) {
    const mesh = MeshBuilder.CreateTube(
      name,
      {
        path: points.map((p) => B(...p)),
        radius,
        tessellation: 6,
        cap: Mesh.CAP_ALL,
      },
      this.scene,
    );
    mesh.material = this.materials[color];
    mesh.parent = parent;
    mesh.isPickable = false;
    return mesh;
  }
  label(
    text: string,
    point: V,
    view: View,
    action: () => void,
    extra: Partial<Label> = {},
  ) {
    const element = document.createElement("button");
    element.className = "world-label";
    element.textContent = text;
    element.hidden = view !== this.view;
    element.addEventListener("click", action);
    const leader = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "line",
    );
    leader.style.display = "none";
    this.leaders.append(leader);
    this.overlay.append(element);
    this.labels.push({ element, leader, point: B(...point), view, ...extra });
    return element;
  }
  center(i: number): V {
    return stackCenter(i);
  }
  buildStack() {
    const group = this.groups.stack;
    for (const l of this.capture.layers) {
      const root = new TransformNode(`layer ${l.id + 1}`, this.scene);
      root.position = B(...this.center(l.id));
      root.parent = group;
      root.metadata = { layer: l.id };
      this.box("cast bed", [0, 0, 0.2], [6.6, 2.6, 0.25], "base", root);
      this.box(
        "read channels",
        [-1.1, 0, 0.46],
        [2.7, 2.25, 0.2],
        "paper",
        root,
      );
      for (let h = 0; h < this.capture.queryHeads; h++)
        this.cyl(
          `head ${h + 1}`,
          [-2.15 + h * 0.3, 0, 0.85],
          0.24,
          0.68,
          h % 2 ? "brass" : "steel",
          root,
        );
      this.box(
        "head crossbar",
        [-1.1, 0, 1.22],
        [2.7, 0.18, 0.14],
        "ink",
        root,
      );
      const expansion = l.ffnWidth > 6144 ? 1.8 : 1.1;
      this.box(
        "expansion chamber",
        [1.35, 0, 0.55],
        [expansion, 1.6, 0.25],
        "brass",
        root,
      );
      const piston = this.box(
        "projection press",
        [1.35, 0, 1.13],
        [expansion, 1.55, 0.18],
        "steel",
        root,
        true,
      ) as Mesh;
      for (const x of [0.45, 2.3])
        this.cyl("press post", [x, 0, 0.8], 0.13, 1, "ink", root);
      for (const x of [-0.9, -0.4])
        this.cyl(
          l.owner === l.id ? "owned KV drum" : "shared KV socket",
          [x, 1.05, 0.85],
          l.owner === l.id ? 0.42 : 0.25,
          l.kind === "F" ? 0.9 : 0.5,
          l.owner === l.id ? "dark" : "paper",
          root,
        );
      this.box(
        "weight drawer",
        [-1, -1.5, 0.4],
        [2.3, 0.55, 0.18],
        "ink",
        root,
      );
      for (let w = 0; w < 6; w++)
        this.ball("stored weight", [-1.9 + w * 0.36, -1.5, 0.62], 0.22, root);
      this.box(
        "side input",
        [2.7, -0.8, 0.6],
        [0.45, 0.65, 0.65],
        "green",
        root,
      );
      const lamp = this.box(
        "activity latch",
        [2.9, 0.7, 0.65],
        [0.16, 0.5, 0.6],
        "paper",
        root,
        true,
      ) as Mesh;
      this.macros.push({ piston, lamp, root });
      this.label(
        `${String(l.id + 1).padStart(2, "0")} · ${l.kind === "F" ? "Full" : "Sliding"}`,
        [this.center(l.id)[0], this.center(l.id)[1] - 1.95, 0.35],
        "stack",
        () => this.onLayer(l.id),
        { layer: l.id },
      );
      const next = this.capture.layers[this.capture.layers.indexOf(l) + 1];
      this.pipe(
        "residual journey",
        stackPath(l.id, next?.id).map(([x, y]) => [x, y, 0.28]),
        "green",
        group,
      );
    }
  }
  buildWorkers() {
    this.pipe(
      "one megakernel chassis",
      [
        [-22, -14, 0.1],
        [-22, 14, 0.1],
        [22, 14, 0.1],
        [22, -14, 0.1],
        [-22, -14, 0.1],
      ],
      "dark",
      this.groups.engine,
      0.12,
    );
    for (let i = 0; i < this.capture.grid; i++) {
      const x = ((i % 9) - 4) * 4.8,
        y = (3 - Math.floor(i / 9)) * 4;
      const root = new TransformNode(`worker block ${i}`, this.scene);
      root.parent = this.groups.engine;
      root.position = B(x, y);
      root.metadata = { block: i };
      this.box("worker bed", [0, 0, 0.18], [3.8, 2.7, 0.22], "base", root);
      this.box(
        "staged operand tray",
        [-1, 0, 0.5],
        [0.7, 1.8, 0.24],
        "ink",
        root,
      );
      for (let j = 0; j < 3; j++)
        this.ball(
          "symbolic staged weight",
          [-1, -0.55 + j * 0.5, 0.76],
          0.22,
          root,
        );
      for (const p of [-0.5, 1.25])
        this.cyl("press post", [p, 0.1, 0.9], 0.13, 1.4, "ink", root);
      const piston = this.box(
        "tensor product press",
        [0.4, 0.1, 1.25],
        [1.5, 1.7, 0.22],
        "steel",
        root,
        true,
      ) as Mesh;
      for (let j = 0; j < 6; j++)
        this.cyl(
          `consumer warp ${j + 1}`,
          [-0.45 + (j % 2) * 0.9, -0.5 + Math.floor(j / 2) * 0.5, -0.35],
          0.28,
          0.45,
          "brass",
          piston,
        );
      this.cyl(
        "controller escapement",
        [-1.5, 0.9, 0.6],
        0.45,
        0.2,
        "brass",
        root,
      );
      this.box(
        "loader fork spine",
        [-1, -0.9, 0.5],
        [0.7, 0.1, 0.15],
        "steel",
        root,
      );
      for (const x of [-1.3, -0.7])
        this.box(
          "loader fork finger",
          [x, -0.65, 0.5],
          [0.08, 0.5, 0.15],
          "steel",
          root,
        );
      const latch = new TransformNode("readiness latch", this.scene);
      latch.parent = root;
      latch.position = B(-0.2, -1, 0.6);
      this.box("latch arm", [0.65, 0, 0], [1.3, 0.14, 0.17], "brass", latch);
      const lamp = this.box(
        "worker state",
        [1.55, 0, 0.7],
        [0.2, 1.4, 0.7],
        "paper",
        root,
        true,
      ) as Mesh;
      const reads = Array.from({ length: 3 }, () =>
        this.ball("worker weight read copy", [-1, 0, 0.9], 0.33, root),
      );
      this.pipe(
        "staging to tensor press",
        [
          [-1, 0, 0.7],
          [0.4, 0, 0.7],
        ],
        "brass",
        root,
        0.06,
      );
      this.workers.push({ piston, latch, lamp, reads });
      this.label(
        String(i).padStart(2, "0"),
        [x, y - 1.65, 0.2],
        "engine",
        () => this.onBlock(i),
        { block: i },
      );
    }
  }
  device(id: MechanismId, x: number, y: number) {
    const root = new TransformNode(id, this.scene);
    root.parent = this.detail;
    root.position = B(x, y);
    root.metadata = { part: id };
    return root;
  }
  press(
    parent: TransformNode,
    id: MechanismId,
    x: number,
    y: number,
    width: number,
    role: string,
  ) {
    this.box("press bed", [x, y, 0.35], [width, 1.2, 0.2], "base", parent);
    for (const px of [-width / 2, width / 2])
      this.cyl("press column", [x + px, y, 0.95], 0.12, 1.4, "ink", parent);
    const node = this.box(
      "moving jaw",
      [x, y, 1.5],
      [width, 1.12, 0.18],
      "steel",
      parent,
      true,
    ) as Mesh;
    this.moving.push({
      part: id,
      node,
      motion: "press",
      base: node.position.clone(),
      role,
    });
    return node;
  }
  buildLayer() {
    this.detail.getChildMeshes().forEach((m) => m.dispose());
    this.detail.getChildTransformNodes().forEach((n) => n.dispose());
    this.moving = [];
    for (const l of this.labels.filter((l) => l.view === "layer")) {
      l.element.remove();
      l.leader.remove();
    }
    this.labels = this.labels.filter((l) => l.view !== "layer");
    const l = this.capture.layers[this.selectedLayer];
    this.box(
      "cutaway foundation",
      [0, -0.5, 0.08],
      [22, 12, 0.15],
      "base",
      this.detail,
    );
    const norm = this.device("norm", -8, 0);
    this.press(norm, "norm", 0, 0, 1, "");
    this.box(
      "normalization gauge",
      [0, 0.65, 0.8],
      [0.8, 0.08, 0.8],
      "brass",
      norm,
    );
    const qkv = this.device("qkv", -5, 0);
    this.press(qkv, "qkv", 0, 0, 1.1, "q:0");
    if (l.owner === l.id)
      for (const y of [-1.5, 1.5])
        this.press(qkv, "qkv", 0, y, 0.65, "mixer:0");
    const rope = this.device("rope", -3, 0);
    for (const y of [-0.4, 0.4]) {
      const rotor = new TransformNode("paired coordinate turn", this.scene);
      rotor.parent = rope;
      rotor.position = B(0, y, 0.7);
      this.cyl("rotor hub", [0, 0, 0], 0.65, 0.12, "brass", rotor);
      this.box("pair lever", [0, 0, 0.1], [0.8, 0.08, 0.09], "ink", rotor);
      this.moving.push({
        part: "rope",
        node: rotor,
        motion: "rotate",
        base: rotor.position.clone(),
        role: "qtail:3",
      });
    }
    const attention = this.device("attention", 0, 0);
    for (let h = 0; h < this.capture.queryHeads; h++) {
      const y = (h - 3.5) * 0.64;
      this.box(
        "reader channel",
        [0, y, 0.35],
        [2.2, 0.48, 0.12],
        "paper",
        attention,
      );
      const rotor = new TransformNode(`head ${h + 1}`, this.scene);
      rotor.parent = attention;
      rotor.position = B(0, y, 0.8);
      this.cyl("reading wheel", [0, 0, 0], 0.58, 0.24, "brass", rotor);
      this.box("wheel marker", [0, 0, 0.14], [0.57, 0.045, 0.06], "ink", rotor);
      this.moving.push({
        part: "attention",
        node: rotor,
        motion: "rotate",
        base: rotor.position.clone(),
        role: "mixer:2",
      });
      this.pipe(
        "Q fan out",
        [
          [-2.8, 0, 0.6],
          [-1.6, y, 0.6],
          [-0.6, y, 0.6],
        ],
        "teal",
        this.detail,
        0.027,
      );
      this.pipe(
        "head join",
        [
          [0.6, y, 0.6],
          [1.5, y, 0.6],
          [2.5, 0, 0.6],
        ],
        "teal",
        this.detail,
        0.027,
      );
    }
    const cache = this.device("cache", 0, 4.3);
    for (const x of [-0.65, 0.65]) {
      this.cyl(
        l.owner === l.id ? "cache drum" : "shared cache coupling",
        [x, 0, 0.8],
        0.9,
        l.kind === "F" ? 1.5 : 0.8,
        "dark",
        cache,
      );
      for (const z of [0.5, 0.85, 1.2])
        this.cyl("cache ring", [x, 0, z], 0.96, 0.08, "steel", cache);
      this.pipe(
        "cached K or V to readers",
        [
          [x, 0, 0.8],
          [x, -1.5, 0.8],
          [x, -2.7, 0.8],
        ],
        "green",
        cache,
        0.05,
      );
    }
    if (l.owner !== l.id) {
      this.pipe(
        "shared KV owner return",
        [
          [0, 4.3, 0.5],
          [4, 4.3, 0.5],
          [7.9, 3.5, 0.5],
        ],
        "green",
        this.detail,
        0.08,
      );
      this.label(`Read layer ${l.owner + 1} K/V`, [8, 3.5, 0.8], "layer", () =>
        this.onLayer(l.owner),
      );
      this.cyl("owner drum", [8, 3.5, 0.9], 1.2, 1.4, "dark", this.detail);
    }
    const out = this.device("output", 3, 0);
    this.press(out, "output", 0, 0, 1.1, "main:0");
    for (let i = 0; i < this.capture.outputSlices; i++)
      this.cyl(
        "output partial cup",
        [-0.35 + i * 0.7, -0.95, 0.5],
        0.45,
        0.3,
        "brass",
        out,
      );
    const residual = this.device("residual", 5.2, 0);
    this.cyl("residual join", [0, 0, 0.7], 1.1, 0.45, "green", residual);
    this.box("join cross", [0, 0, 0.95], [0.7, 0.12, 0.08], "ink", residual);
    this.box("join cross", [0, 0, 0.95], [0.12, 0.7, 0.08], "ink", residual);
    for (const z of [0.35, 0.55])
      this.pipe(
        "hi and lo preserved bypass",
        [
          [-8, 0, z],
          [-9, 0, z],
          [-9, -2.5, z],
          [5.2, -2.5, z],
          [5.2, 0, z],
        ],
        "green",
        this.detail,
        0.045,
      );
    const gate = this.device("gateup", 5, -4);
    const width = l.ffnWidth > 6144 ? 1.35 : 0.85;
    for (const x of [-1, 1])
      this.press(gate, "gateup", x, 0, width, "gate-up:1");
    this.box(
      "GELU metering gate",
      [0, -1, 0.7],
      [2.8, 0.18, 0.4],
      "brass",
      gate,
    );
    const down = this.device("down", 0, -4);
    this.press(down, "down", 0, 0, 1.9, "main:10");
    for (let j = 0; j < this.capture.downSlices; j++)
      this.cyl(
        "down partial cup",
        [-1.5 + j * 0.6, -0.9, 0.48],
        0.36,
        0.3,
        "brass",
        down,
      );
    const ple = this.device("ple", -5, -4);
    this.box(
      "256-channel side cartridge",
      [0, 0, 0.65],
      [1.3, 0.8, 0.8],
      "green",
      ple,
    );
    for (let i = 0; i < 4; i++)
      this.box(
        "cartridge contact",
        [-0.45 + i * 0.3, -0.46, 0.65],
        [0.1, 0.2, 0.45],
        "brass",
        ple,
      );
    this.pipe(
      "per-layer input feed",
      [
        [-5, -6, 0.3],
        [-5, -4.5, 0.3],
      ],
      "green",
      this.detail,
      0.08,
    );
    const latch = this.device("latch", -8, -4);
    for (const x of [-0.6, 0.6])
      this.cyl("gate post", [x, 0, 0.65], 0.13, 1, "ink", latch);
    const arm = new TransformNode("counter gate arm", this.scene);
    arm.parent = latch;
    arm.position = B(-0.6, 0, 1);
    this.box("gate arm", [0.6, 0, 0], [1.2, 0.13, 0.16], "brass", arm);
    this.moving.push({
      part: "latch",
      node: arm,
      motion: "gate",
      base: arm.position.clone(),
      role: "main:15",
    });
    this.pipe(
      "transform path",
      [
        [-8, 0, 0.4],
        [-3, 0, 0.4],
        [3, 0, 0.4],
        [5.2, 0, 0.4],
        [7.7, 0, 0.4],
        [7.7, -4, 0.4],
        [-8, -4, 0.4],
      ],
      "teal",
      this.detail,
      0.055,
    );
    for (const z of [0.25, 0.45])
      this.pipe(
        "feed-forward residual bypass",
        [
          [5.2, 0, z],
          [9.5, 0, z],
          [9.5, -5.6, z],
          [-8, -5.6, z],
          [-8, -4, z],
        ],
        "green",
        this.detail,
        0.04,
      );
    const weights = this.device("weights", -6.8, 4.1);
    this.box(
      "fixed weight drawer",
      [0, 0, 0.4],
      [3.5, 1.4, 0.3],
      "ink",
      weights,
    );
    for (let j = 0; j < 12; j++)
      this.ball(
        "unknown-valued stored weight",
        [-1.3 + (j % 6) * 0.52, -0.3 + Math.floor(j / 6) * 0.6, 0.75],
        0.28,
        weights,
      );
    for (let j = 0; j < 4; j++)
      this.box(
        "symbolic ring pocket",
        [2.2 + j * 0.45, 0, 0.5],
        [0.34, 0.8, 0.18],
        "steel",
        weights,
      );
    this.pipe(
      "untimed staging path",
      [
        [-4.6, 4.1, 0.45],
        [-3.1, 4.1, 0.45],
        [-3.1, 2.3, 0.45],
      ],
      "brass",
      this.detail,
      0.045,
    );
    const names: [MechanismId, string, V][] = [
      ["norm", "Sizing jaws", [-8, 1.4, 1]],
      ["qkv", l.owner === l.id ? "Q / K / V presses" : "Q press", [-5, 2.4, 1]],
      ["rope", "Pair rotation", [-3, 1.2, 1]],
      ["attention", "8 query heads", [0, -3.1, 1]],
      [
        "cache",
        l.owner === l.id ? "K / V · owned" : "K / V · shared",
        [0, 5.5, 1],
      ],
      ["output", "O · 2 partials", [3, 1.5, 1]],
      ["residual", "Residual add", [5.2, 1.2, 1]],
      ["gateup", `Gate / up · ${l.ffnWidth.toLocaleString()}`, [5, -5.6, 1]],
      ["down", "Down · 6 partials", [0, -5.6, 1]],
      ["ple", "Side input · 256", [-5, -5.6, 1]],
      ["latch", "Ready latch", [-8, -5.6, 1]],
      ["weights", "Weights → staging", [-6.8, 5.5, 1]],
    ];
    for (const [id, name, point] of names) {
      const device = this.detail
        .getChildTransformNodes()
        .find((n) => n.name === id)!;
      this.label(name, point, "layer", () => this.onPart(id), {
        part: id,
        anchor: device.getAbsolutePosition().add(new Vector3(0, 0.8, 0)),
      });
    }
    this.journey?.buildLayer();
  }
  use(capture: Capture) {
    this.capture = capture;
    this.recorded = new RecordedMotion(capture);
    this.buildLayer();
    if (this.view === "internals") this.workshop.open();
  }
  setLayer(n: number) {
    this.selectedLayer = n;
    this.buildLayer();
    this.setView("layer");
  }
  openInternals() {
    this.workshop.open();
    this.setView("internals");
  }
  setView(view: View) {
    const changed = view !== this.view;
    if (changed)
      this.beginInspection(
        this.nextInspectionDirection ?? inspectionDirection(this.view, view),
        this.nextInspectionOrigin,
      );
    this.nextInspectionDirection = undefined;
    this.nextInspectionOrigin = undefined;
    this.view = view;
    for (const [key, group] of Object.entries(this.groups))
      group.setEnabled(key === view);
    for (const l of this.labels) {
      l.element.hidden = l.view !== view;
      l.leader.style.display = "none";
    }
    this.fit();
    if (
      this.followTray &&
      view !== "material" &&
      view !== "hardware"
    )
      this.applyFollowPose(1);
    if (changed) this.inspection.finish();
  }
  beginInspection(direction: "in" | "out", target?: Vector3) {
    const selected = this.labels.find(
      (l) =>
        l.view === this.view &&
        ((this.view === "stack" && l.layer === this.selectedLayer) ||
          (this.view === "engine" && l.block === this.selectedBlock) ||
          (this.view === "layer" && l.part === this.selectedPart) ||
          (this.view === "internals" && l.internal === this.workshop.selected)),
    );
    const point =
      target ?? selected?.anchor ?? selected?.point ?? this.camera.target;
    const rect = this.canvas.getBoundingClientRect();
    const projected = Vector3.Project(
      point,
      Matrix.Identity(),
      this.scene.getTransformMatrix(),
      this.camera.viewport.toGlobal(rect.width, rect.height),
    );
    this.inspection.begin(direction, {
      x: projected.x / rect.width,
      y: projected.y / rect.height,
    });
  }
  inspectChange(direction: "in" | "out", change: () => void, target?: Vector3) {
    this.beginInspection(direction, target);
    change();
    this.fit();
    this.inspection.finish();
  }
  frameDetail(frame: () => void, begin = true) {
    if (begin) this.beginInspection("in");
    this.userMovedCamera = false;
    this.resizeFraming = frame;
    frame();
    this.inspection.finish(frame);
  }
  fit() {
    this.resizeFraming = undefined;
    this.userMovedCamera = false;
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.camera.setTarget(B(0, this.view === "layer" ? -0.4 : 0, 0.2));
    this.camera.alpha = Math.PI / 2 + 0.07;
    this.camera.beta =
      this.view === "layer"
        ? 0.57
        : this.view === "material"
          ? 0.62
          : this.view === "hardware" && this.hardware?.model.depth === "tile"
            ? 0.72
            : 0.48;
    this.camera.radius = this.view === "layer" ? 24 : 58;
    const width =
        this.view === "hardware"
          ? (this.hardware?.width ?? 33)
          : this.view === "material"
            ? 35
            : this.view === "internals"
              ? this.workshop.width
              : this.view === "layer"
                ? 26
                : this.view === "stack"
                  ? 62
                  : 72,
      depth =
        this.view === "hardware"
          ? (this.hardware?.depthSize ?? 23)
          : this.view === "material"
            ? 17
            : this.view === "internals"
              ? this.workshop.depth
              : this.view === "layer"
                ? 15
                : this.view === "stack"
                  ? 47
                  : 38;
    this.focusedBlock = null;
    const margin = innerWidth <= 650 ? 22 : rect.height < 650 ? 24 : 65;
    for (let n = 0; n < 5; n++) {
      this.camera.getViewMatrix(true);
      this.scene.updateTransformMatrix(true);
      const points = [
        [-width / 2, -depth / 2],
        [width / 2, depth / 2],
        [-width / 2, depth / 2],
        [width / 2, -depth / 2],
      ].map(([x, y]) =>
        Vector3.Project(
          B(x + (this.view === "engine" ? -11 : 0), y, 1),
          Matrix.Identity(),
          this.scene.getTransformMatrix(),
          this.camera.viewport.toGlobal(rect.width, rect.height),
        ),
      );
      const xs = points.map((p) => p.x),
        ys = points.map((p) => p.y);
      const minX = Math.min(...xs),
        maxX = Math.max(...xs),
        minY = Math.min(...ys),
        maxY = Math.max(...ys),
        perPixel =
          (2 * this.camera.radius * Math.tan(this.camera.fov / 2)) /
          rect.height;
      const { alpha, beta, radius } = this.camera;
      this.camera.setTarget(
        this.camera.target
          .add(
            this.camera
              .getDirection(Vector3.Right())
              .scale(((minX + maxX - rect.width) / 2) * perPixel),
          )
          .add(
            this.camera
              .getDirection(Vector3.Up())
              .scale(-((minY + maxY - rect.height) / 2) * perPixel),
          ),
      );
      this.camera.alpha = alpha;
      this.camera.beta = beta;
      this.camera.radius = radius;
      this.camera.radius *= Math.max(
        (maxX - minX) / (rect.width - margin * 2),
        (maxY - minY) / (rect.height - margin * 2),
      );
    }
    this.camera.inertialAlphaOffset = 0;
    this.camera.inertialBetaOffset = 0;
    this.camera.inertialRadiusOffset = 0;
    if (this.view === "internals" && innerWidth <= 650) this.workshop.focus();
    if (this.view === "material" && innerWidth <= 650) this.material?.focus();
    if (
      this.view === "hardware" &&
      this.hardware?.model.depth === "tile" &&
      innerWidth <= 650
    )
      this.hardware.tile.focus();
    this.resizeFraming = undefined;
  }
  followRadius() {
    return this.view === "stack"
      ? innerWidth <= 650
        ? 13
        : 24
      : this.view === "engine"
        ? 12
        : innerWidth <= 650
          ? 10
          : 19;
  }
  applyFollowPose(ease: number) {
    const prevAlpha = this.camera.alpha,
      prevBeta = this.camera.beta,
      prevRadius = this.camera.radius;
    const desiredAlpha = Math.PI / 2 + 0.07,
      desiredBeta = 0.65,
      desiredRadius = this.followRadius();
    const alphaDelta = Math.atan2(
      Math.sin(desiredAlpha - prevAlpha),
      Math.cos(desiredAlpha - prevAlpha),
    );
    const target = Vector3.Lerp(
      this.camera.target,
      this.journey.followPoint(),
      ease,
    );
    this.camera.setTarget(target);
    this.camera.alpha = prevAlpha + alphaDelta * ease;
    this.camera.beta = prevBeta + (desiredBeta - prevBeta) * ease;
    this.camera.radius = prevRadius + (desiredRadius - prevRadius) * ease;
    this.camera.inertialAlphaOffset = 0;
    this.camera.inertialBetaOffset = 0;
    this.camera.inertialRadiusOffset = 0;
  }
  focusWorker() {
    this.userMovedCamera = false;
    this.resizeFraming = () => this.focusWorker();
    this.focusedBlock = this.selectedBlock;
    const i = this.selectedBlock;
    this.camera.setTarget(
      B(((i % 9) - 4) * 4.8, (3 - Math.floor(i / 9)) * 4, 0.5),
    );
    this.camera.alpha = Math.PI / 2 + 0.07;
    this.camera.radius = innerWidth <= 650 ? 7 : 6;
    this.camera.beta = 0.85;
  }
  focusPart() {
    this.userMovedCamera = false;
    this.resizeFraming = () => this.focusPart();
    const node = this.detail
      .getChildTransformNodes()
      .find((n) => n.name === this.selectedPart);
    if (node) {
      this.camera.setTarget(node.getAbsolutePosition());
      this.camera.alpha = Math.PI / 2 + 0.07;
      this.camera.radius = innerWidth <= 650 ? 7 : 8;
      this.camera.beta = 0.75;
    }
  }
  update(time: number, dt = 1 / 60) {
    updateMachinery(this, time);
    this.supplies.update();
    if (this.view === "internals") this.workshop.update(time);
    this.journey.update(this.story, time);
    this.material?.update();
    this.hardware?.update();
    if (
      this.followTray &&
      !this.inspection.active &&
      this.view !== "material" &&
      this.view !== "hardware"
    ) {
      const ease = matchMedia("(prefers-reduced-motion: reduce)").matches
        ? 1
        : 1 - Math.exp(-Math.min(dt, 0.1) * 7);
      this.applyFollowPose(ease);
    }
    this.inspection.tick();
    this.scene.render();
    this.positionLabels(time);
  }
  positionLabels(time: number) {
    const rect = this.canvas.getBoundingClientRect(),
      viewport = this.camera.viewport.toGlobal(rect.width, rect.height);
    const occupied: { x: number; y: number; w: number; h: number }[] = [];
    for (const element of this.canvas.parentElement!.querySelectorAll<HTMLElement>(
      ".follow-button,.fit-button,.journey-anchors,#worker-overview",
    )) {
      if (!element.offsetWidth || element.hidden) continue;
      const box = element.getBoundingClientRect();
      occupied.push({
        x: box.x - rect.x + box.width / 2,
        y: box.y - rect.y + box.height / 2,
        w: box.width,
        h: box.height,
      });
    }
    const priority = (l: Label) =>
      l.layer === this.selectedLayer ||
      l.part === this.selectedPart ||
      l.internal === this.workshop.selected ||
      l.block === this.selectedBlock
        ? 0
        : l.layer === undefined && l.block === undefined
          ? 1
          : 2;
    for (const l of [...this.labels].sort(
      (a, b) => priority(a) - priority(b),
    )) {
      if (l.view !== this.view) continue;
      const p = Vector3.Project(
        l.point,
        Matrix.Identity(),
        this.scene.getTransformMatrix(),
        viewport,
      );
      l.element.hidden =
        p.z < 0 ||
        p.z > 1 ||
        p.x < 0 ||
        p.x > rect.width ||
        p.y < 0 ||
        p.y > rect.height ||
        (innerWidth <= 650 &&
          l.view === "layer" &&
          l.part !== this.selectedPart) ||
        (innerWidth <= 650 &&
          l.view === "internals" &&
          l.internal !== this.workshop.selected) ||
        (innerWidth <= 650 &&
          l.view === "material" &&
          l.element.dataset.topic !== this.material?.model.topic) ||
        (rect.width < 650 &&
          l.block !== undefined &&
          l.block !== this.selectedBlock) ||
        (this.focusedBlock !== null && l.block !== this.focusedBlock);
      if (
        l.view === "material" &&
        l.element.dataset.contextSlot !== undefined
      ) {
        l.element.hidden ||=
          this.material?.model.topic !== "context" ||
          (l.element.dataset.request === "1" &&
            !this.material?.state.requests[1]) ||
          (innerWidth <= 650 &&
            Number(l.element.dataset.contextSlot) !==
              this.material?.state.token);
      }
      l.leader.style.display = "none";
      if (l.view === "hardware")
        l.element.hidden ||=
          l.element.dataset.hardwareDepth !== this.hardware?.model.depth;
      if (l.element.dataset.tileStation && innerWidth <= 1600)
        l.element.hidden ||=
          l.element.dataset.tileStation !== this.hardware?.tile.model.station;
      if (
        l.view === "internals" &&
        rect.height < 650 &&
        l.internal !== this.workshop.selected
      )
        l.element.hidden = true;
      if (
        l.view === "stack" &&
        l.layer !== undefined &&
        (rect.width < 1200 || rect.height < 650)
      ) {
        const current = this.story?.layerActive
          ? this.story.layerIndex
          : this.selectedLayer;
        l.element.hidden ||=
          l.layer !== current &&
          (rect.width < 650 || (l.layer % 5 !== 0 && l.layer !== 34));
      }
      if (!l.element.hidden) {
        const w = l.element.offsetWidth,
          h = l.element.offsetHeight,
          preferredX = Math.max(
            w / 2 + 4,
            Math.min(rect.width - w / 2 - 4, p.x),
          );
        const candidates = [
          [0, 0],
          [0, h + 6],
          [0, -h - 6],
          [0, 2 * (h + 6)],
          [0, -2 * (h + 6)],
          [w / 2 + 12, 0],
          [-w / 2 - 12, 0],
        ];
        const place = candidates
          .map(([dx, dy]) => ({ x: preferredX + dx, y: p.y + dy }))
          .find(
            ({ x, y }) =>
              x - w / 2 >= 4 &&
              x + w / 2 <= rect.width - 4 &&
              y - h / 2 >= 4 &&
              y + h / 2 <= rect.height - 4 &&
              !occupied.some(
                (b) =>
                  Math.abs(b.x - x) < (b.w + w) / 2 + 4 &&
                  Math.abs(b.y - y) < (b.h + h) / 2 + 4,
              ),
          );
        if (!place) {
          l.element.hidden = true;
          continue;
        }
        const { x, y } = place;
        occupied.push({ x, y, w, h });
        l.element.style.transform = `translate(${x}px,${y}px) translate(-50%,-50%)`;
        if (Math.abs(y - p.y) > h || Math.abs(x - p.x) > 4) {
          const anchor = l.anchor
            ? Vector3.Project(
                l.anchor,
                Matrix.Identity(),
                this.scene.getTransformMatrix(),
                viewport,
              )
            : p;
          l.leader.style.display = "";
          for (const [key, value] of Object.entries({
            x1: anchor.x,
            y1: anchor.y,
            x2: x,
            y2: y + h / 2,
          }))
            l.leader.setAttribute(key, String(value));
        }
      }
      if (l.layer !== undefined) {
        const layer = this.capture.layers[l.layer];
        l.element.classList.toggle(
          "active",
          this.story
            ? this.story.layerActive && this.story.layerIndex === l.layer
            : time >= layer.journeyStart && time < layer.end,
        );
        l.element.classList.toggle("selected", this.selectedLayer === l.layer);
      }
      if (l.block !== undefined)
        l.element.classList.toggle("selected", this.selectedBlock === l.block);
      if (l.internal !== undefined)
        l.element.classList.toggle(
          "selected",
          this.workshop.selected === l.internal,
        );
    }
  }
}
