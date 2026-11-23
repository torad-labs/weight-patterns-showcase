import { TransformNode } from "@babylonjs/core/Meshes/transformNode";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { MechanicalWorld } from "./replay-world";
import { TileFlow, type TilePose } from "./tile-flow";
import { samplePath } from "./timeline";

type Point = [number, number, number];
const at = (p: Point) => new Vector3(-p[0], p[2], -p[1]);

/** Shared memory supply around logical workers, without inventing a physical SM map. */
export class SupplyWorld {
  readonly flow = new TileFlow();
  poseForBlock?: (block: number) => TilePose;
  private packets: { node: TransformNode; path: Point[] }[] = [];
  private request: TransformNode;
  private refill: TransformNode;
  constructor(private host: MechanicalWorld) {
    const w = host,
      g = w.groups.engine;
    for (const [station, x, width, depth] of [
      ["storage", -40, 7, 13],
      ["cache", -29, 6, 24],
    ] as const) {
      const root = new TransformNode(
        station === "cache" ? "shared L2 cache bank" : "VRAM retained weights",
        w.scene,
      );
      root.parent = g;
      root.position = at([x, 0, 0]);
      root.metadata = { supplyStation: station };
      w.box("supply cabinet", [0, 0, 0.2], [width, depth, 0.4], "paper", root);
      for (let row = 0; row < (station === "cache" ? 7 : 4); row++) {
        const y = station === "cache" ? 9 - row * 3 : 4.5 - row * 3;
        w.box(
          "retained compartment",
          [0, y, 0.65],
          [width - 0.6, 2.3, 0.6],
          "ink",
          root,
        );
        for (let col = 0; col < 4; col++)
          w.ball(
            "stored weight remains",
            [-1.8 + col * 1.2, y, 1.2],
            0.7,
            root,
          );
      }
      w.label(
        station === "cache"
          ? "L2 cache · shared by every SM"
          : "VRAM · stored weights",
        [x, depth / 2 + 2, 1],
        "engine",
        () => w.onSupply(station, root.getAbsolutePosition()),
      );
    }
    w.pipe(
      "VRAM to shared L2",
      [
        [-36.5, 0, 1],
        [-32, 0, 1],
      ],
      "brass",
      g,
      0.15,
    );
    w.pipe(
      "L2 distribution spine",
      [
        [-26, 0, 1],
        [-24, 0, 1],
        [-24, 13.6, 1],
        [-24, -10.4, 1],
      ],
      "brass",
      g,
      0.12,
    );
    for (let row = 0; row < Math.ceil(w.capture.grid / 9); row++) {
      const y = 13.6 - row * 4;
      w.pipe(
        "shared supply row",
        [
          [-24, y, 1],
          [18.2, y, 1],
        ],
        "brass",
        g,
        0.06,
      );
    }
    for (let block = 0; block < w.capture.grid; block++) {
      const x = ((block % 9) - 4) * 4.8 - 1,
        y = (3 - Math.floor(block / 9)) * 4;
      const path: Point[] = [
        [-26, 0, 1],
        [-24, 0, 1],
        [-24, y + 1.6, 1],
        [x, y + 1.6, 1],
        [x, y, 1],
      ];
      w.pipe(
        "worker supply inlet",
        [
          [x, y + 1.6, 1],
          [x, y, 1],
        ],
        "brass",
        g,
        0.06,
      );
      const node = w.ball(
        "L2 read copy for worker " + block,
        [-26, 0, 1.2],
        0.5,
        g,
      );
      this.packets.push({ node, path });
    }
    this.request = w.box(
      "cache miss request",
      [-32, 0, 1.4],
      [0.65, 0.45, 0.12],
      "ink",
      g,
    );
    this.refill = w.ball("VRAM refill copy", [-36.5, 0, 1.4], 0.7, g);
    w.label(
      "Delivery routes · logical workers",
      [-1, 16.7, 0.5],
      "engine",
      () => w.onSupply("loader"),
    );
  }
  update() {
    if (this.host.view !== "engine" || !this.poseForBlock) return;
    this.packets.forEach(({ node, path }, block) => {
      const s = this.poseForBlock!(block),
        d = s.delivery;
      node.setEnabled(d?.stage === "deliver");
      if (d?.stage === "deliver")
        node.position.copyFrom(at(samplePath(path, d.progress) as Point));
    });
    const s = this.poseForBlock(this.host.selectedBlock),
      d = s.delivery;
    this.request.setEnabled(d?.stage === "request");
    this.refill.setEnabled(d?.stage === "refill");
    if (d?.stage === "request")
      this.request.position.copyFrom(at([-32 - 4.5 * d.progress, 0, 1.4]));
    if (d?.stage === "refill")
      this.refill.position.copyFrom(at([-36.5 + 4.5 * d.progress, 0, 1.4]));
  }
}
