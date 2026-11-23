import type { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { Vector3 } from "@babylonjs/core/Maths/math.vector";
import type { View } from "./replay-playback";

export function inspectionDirection(from: View, to: View): "in" | "out" {
  const depth: Record<View, number> = {
    engine: 0,
    hardware: 1,
    stack: 2,
    layer: 3,
    material: 4,
    internals: 4,
  };
  return depth[to] >= depth[from] ? "in" : "out";
}
export const inspectionEase = (p: number) =>
  1 - Math.pow(1 - Math.max(0, Math.min(1, p)), 3);

/** A short spatial handoff, independent of simulation time. The outgoing frame is not retained after arrival. */
export class InspectionMotion {
  private snapshot: HTMLCanvasElement;
  private animation?: Animation;
  private pose?: {
    target: Vector3;
    alpha: number;
    beta: number;
    radius: number;
  };
  private start = 0;
  private direction: "in" | "out" = "in";
  private reduced = false;
  private frame = 0;
  private arrivalFit?: () => void;
  constructor(
    private canvas: HTMLCanvasElement,
    private labels: HTMLElement,
    private camera: ArcRotateCamera,
  ) {
    this.snapshot = document.createElement("canvas");
    this.snapshot.className = "inspection-snapshot";
    this.snapshot.setAttribute("aria-hidden", "true");
    this.snapshot.hidden = true;
    canvas.parentElement!.append(this.snapshot);
    for (const event of ["pointerdown", "wheel"])
      canvas.addEventListener(event, () => this.cancel(), { passive: true });
  }
  get active() {
    return this.pose !== undefined || this.frame !== 0;
  }
  begin(
    direction: "in" | "out",
    origin: { x: number; y: number } = { x: 0.5, y: 0.5 },
  ) {
    this.cancel();
    this.direction = direction;
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    const rect = this.canvas.getBoundingClientRect();
    this.snapshot.width = Math.max(1, Math.round(rect.width));
    this.snapshot.height = Math.max(1, Math.round(rect.height));
    this.snapshot
      .getContext("2d")!
      .drawImage(this.canvas, 0, 0, this.snapshot.width, this.snapshot.height);
    this.snapshot.style.transformOrigin = `${Math.max(0, Math.min(1, origin.x)) * 100}% ${Math.max(0, Math.min(1, origin.y)) * 100}%`;
    this.snapshot.hidden = false;
    this.labels.style.opacity = "0";
    this.canvas.parentElement!.dataset.inspection = direction;
    this.animation = this.snapshot.animate(
      [
        { transform: "scale(1)", opacity: 1 },
        {
          transform: `scale(${this.reduced ? 1 : direction === "in" ? 2.2 : 0.62})`,
          opacity: 0,
        },
      ],
      {
        duration: this.reduced ? 120 : 360,
        easing: "cubic-bezier(.2,.7,.2,1)",
        fill: "forwards",
      },
    );
  }
  finish(arrivalFit?: () => void) {
    if (this.snapshot.hidden) return;
    if (arrivalFit) this.arrivalFit = arrivalFit;
    cancelAnimationFrame(this.frame);
    // All callers finish fitting/building in this task, before we capture the destination.
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      const c = this.camera;
      this.pose = {
        target: c.target.clone(),
        alpha: c.alpha,
        beta: c.beta,
        radius: c.radius,
      };
      this.start = performance.now();
      this.tick();
    });
  }
  reframe(fit: () => void) {
    // A view change can resize the canvas after its first fit. Keep the outgoing
    // frame and retarget the arrival instead of cancelling the navigation.
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    this.pose = undefined;
    (this.arrivalFit ?? fit)();
    this.finish();
  }
  tick() {
    if (!this.pose) return;
    const elapsed = performance.now() - this.start,
      duration = this.reduced ? 120 : 520;
    const p = Math.min(1, elapsed / duration),
      ease = inspectionEase(p),
      pose = this.pose;
    this.camera.setTarget(pose.target);
    this.camera.alpha = pose.alpha;
    this.camera.beta = pose.beta;
    this.camera.radius =
      pose.radius *
      (1 +
        (this.reduced ? 0 : this.direction === "in" ? 0.6 : -0.45) *
          (1 - ease));
    this.labels.style.opacity = String(Math.max(0, (p - 0.35) / 0.65));
    if (p === 1) this.cancel();
  }
  cancel() {
    cancelAnimationFrame(this.frame);
    this.frame = 0;
    if (this.pose) {
      const p = this.pose;
      this.camera.setTarget(p.target);
      this.camera.alpha = p.alpha;
      this.camera.beta = p.beta;
      this.camera.radius = p.radius;
    }
    this.pose = undefined;
    this.arrivalFit = undefined;
    this.animation?.cancel();
    this.animation = undefined;
    this.snapshot.hidden = true;
    this.labels.style.opacity = "";
    delete this.canvas.parentElement!.dataset.inspection;
  }
}
