import type { Point } from "../data";

export type Scale = (v: number) => number;

export const linearScale = (
  [d0, d1]: [number, number],
  [r0, r1]: [number, number],
): Scale => {
  return (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
};

export const logScale = (
  [d0, d1]: [number, number],
  range: [number, number],
): Scale => {
  const lin = linearScale([Math.log10(d0), Math.log10(d1)], range);
  return (v) => lin(Math.log10(Math.max(v, d0)));
};

type XY = { x: number; y: number };

// Fritsch–Carlson monotone tangents: the curve never overshoots between NEB
// images, so no spurious extrema are drawn (same idea as d3.curveMonotoneX).
const monotoneTangents = (pts: XY[]): number[] => {
  const slopes = pts
    .slice(1)
    .map((p, i) => (p.y - pts[i].y) / (p.x - pts[i].x));
  return pts.map((_, i) => {
    if (i === 0) return slopes[0];
    if (i === pts.length - 1) return slopes[slopes.length - 1];
    const [d0, d1] = [slopes[i - 1], slopes[i]];
    if (d0 * d1 <= 0) return 0;
    const h0 = pts[i].x - pts[i - 1].x;
    const h1 = pts[i + 1].x - pts[i].x;
    return (3 * (h0 + h1)) / ((2 * h1 + h0) / d0 + (h1 + 2 * h0) / d1);
  });
};

export const smoothPath = (pts: XY[]): string => {
  if (pts.length < 2) {
    return "";
  }
  const m = monotoneTangents(pts);
  const segments = pts.slice(1).map((p2, idx) => {
    const p1 = pts[idx];
    const h = (p2.x - p1.x) / 3;
    return `C ${p1.x + h} ${p1.y + m[idx] * h}, ${p2.x - h} ${p2.y - m[idx + 1] * h}, ${p2.x} ${p2.y}`;
  });
  return `M ${pts[0].x} ${pts[0].y} ${segments.join(" ")}`;
};

// Energy at reaction coordinate s on the same monotone Hermite curve.
export const energyAt = (points: Point[], s: number): number => {
  const pts = points.map((p) => ({ x: p.s, y: p.e }));
  const i = Math.min(
    Math.max(pts.findIndex((p) => p.x >= s) - 1, 0),
    pts.length - 2,
  );
  const m = monotoneTangents(pts);
  const h = pts[i + 1].x - pts[i].x;
  const t = Math.min(Math.max((s - pts[i].x) / h, 0), 1);
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    (2 * t3 - 3 * t2 + 1) * pts[i].y +
    (t3 - 2 * t2 + t) * h * m[i] +
    (-2 * t3 + 3 * t2) * pts[i + 1].y +
    (t3 - t2) * h * m[i + 1]
  );
};

export const niceTicks = (
  [d0, d1]: [number, number],
  count: number,
): number[] => {
  const rawStep = (d1 - d0) / count;
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((m) => m >= rawStep) ??
    rawStep;
  const first = Math.ceil(d0 / step) * step;
  const ticks: number[] = [];
  for (let v = first; v <= d1 + 1e-9; v += step) {
    ticks.push(Number(v.toFixed(6)));
  }
  return ticks;
};
