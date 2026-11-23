import { intervalProgress, type Capture } from "./replay-model";

export type StackPoint = [x: number, y: number, z: number];

export function stackDirection(layer: number) {
  return Math.floor(layer / 5) % 2 ? -1 : 1;
}

export function stackCenter(layer: number): StackPoint {
  const row = Math.floor(layer / 5),
    col = stackDirection(layer) < 0 ? 4 - (layer % 5) : layer % 5;
  return [(col - 2) * 8.4, (3 - row) * 4.2, 0];
}

// One route drives both the pipe and its carrier, including the turn into the next row.
export function stackPath(layer: number, next?: number): StackPoint[] {
  const [x, y] = stackCenter(layer),
    direction = stackDirection(layer);
  const points: StackPoint[] = [
    [x - direction * 2.9, y - 0.7, 1.35],
    [x + direction * 2.9, y - 0.7, 1.35],
  ];
  if (next !== undefined) {
    const [nx, ny] = stackCenter(next);
    if (Math.floor(layer / 5) !== Math.floor(next / 5)) {
      const outside = x + direction * 3.5;
      points.push([outside, y - 0.7, 1.35], [outside, ny - 0.7, 1.35]);
    }
    points.push([nx - stackDirection(next) * 2.9, ny - 0.7, 1.35]);
  }
  return points;
}

function pointAlong(points: StackPoint[], progress: number): StackPoint {
  const lengths = points
    .slice(1)
    .map((point, i) =>
      Math.hypot(...point.map((v, axis) => v - points[i][axis])),
    );
  let remaining = progress * lengths.reduce((sum, length) => sum + length, 0);
  for (let i = 0; i < lengths.length; i++) {
    if (remaining <= lengths[i]) {
      const p = lengths[i] ? remaining / lengths[i] : 0;
      return points[i].map(
        (v, axis) => v + (points[i + 1][axis] - v) * p,
      ) as StackPoint;
    }
    remaining -= lengths[i];
  }
  return points[points.length - 1];
}

export function stackCarrierAt(
  capture: Capture,
  time: number,
): StackPoint | null {
  let index = -1;
  for (let i = 0; i < capture.layers.length; i++)
    if (capture.layers[i].journeyStart <= time) index = i;
  if (index < 0 || time > capture.end) return null;
  const layer = capture.layers[index],
    next = capture.layers[index + 1];
  const arrival = next?.journeyStart ?? capture.end;
  return stackCarrierProgress(
    layer.id,
    next?.id,
    intervalProgress(layer.journeyStart, arrival, time),
    intervalProgress(layer.journeyStart, arrival, layer.end),
  );
}

// Stay inside the machine until its work is complete. Only the remaining gap
// carries the cart along the connector; distance never decides completion time.
// The route is illustrative, not a measurement of physical data transport.
export function stackCarrierProgress(
  layer: number,
  next: number | undefined,
  progress: number,
  releaseAt: number,
  delay = 0,
): StackPoint {
  // Stagger entry, then let every tray finish before the shared layer boundary.
  progress = Math.max(0, Math.min(1, (progress - delay) / (1 - delay)));
  const path = stackPath(layer, next);
  if (next === undefined) return pointAlong(path, progress);
  if (progress >= 1) return path[path.length - 1];
  return progress < releaseAt
    ? pointAlong(path.slice(0, 2), intervalProgress(0, releaseAt, progress))
    : pointAlong(path.slice(1), intervalProgress(releaseAt, 1, progress));
}
