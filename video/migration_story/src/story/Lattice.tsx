import React, { useMemo } from "react";
import { ATOM, BOND_COLOR, STAGE } from "../theme";
import { geometry, Vec3 } from "./storyData";

// Plane heights (Å) of the canonical Cr81 cell: lower QL, vdW gap, upper QL.
export const LAYERS = {
  lowerQL: [15.0, 22.3],
  gap: [22.3, 25.1],
  upperQL: [25.1, 32.4],
} as const;
const X_RANGE: [number, number] = [-3.2, 16.2];
const Z_RANGE: [number, number] = [13.4, 34.0];
const DEPTH_RANGE: [number, number] = [0, 7.4];
// Bond cut-offs (Å): host Sb–Te / Te–Te contacts, and Cr–Te coordination.
const HOST_BOND_A = 3.35;
const CR_BOND_A = 3.05;
const DEPTH_FADE = 0.55;

type Props = {
  pathKey: string;
  /** 0..1 progress along the NEB images. */
  t: number;
  frame: number;
  width: number;
  height: number;
  slideDx?: number;
  jitter?: number;
  trail?: boolean;
  glow?: number;
  hostOpacity?: number;
  showLayers?: boolean;
};

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

const positionsAt = (images: Vec3[][], t: number): Vec3[] => {
  const f = Math.min(Math.max(t, 0), 1) * (images.length - 1);
  const i = Math.min(Math.floor(f), images.length - 2);
  const local = f - i;
  return images[i].map((p, k) => {
    const q = images[i + 1][k];
    return [
      lerp(p[0], q[0], local),
      lerp(p[1], q[1], local),
      lerp(p[2], q[2], local),
    ];
  });
};

const dist = (a: Vec3, b: Vec3) =>
  Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// Host bond topology from the first image; fixed so bonds never flicker.
const hostBonds = (pts: Vec3[], skip: number): [number, number][] => {
  const out: [number, number][] = [];
  for (let i = 0; i < pts.length; i++) {
    for (let j = i + 1; j < pts.length; j++) {
      if (i === skip || j === skip) continue;
      if (dist(pts[i], pts[j]) < HOST_BOND_A) out.push([i, j]);
    }
  }
  return out;
};

// Deterministic pseudo-thermal displacement, frame-driven so renders are stable.
const wobble = (frame: number, k: number, axis: number) =>
  Math.sin(frame * (0.37 + 0.011 * k) + k * 12.9898 + axis * 78.233) *
  Math.cos(frame * 0.23 + k * 4.1414 + axis);

const depthOf = (y: number) =>
  (y - DEPTH_RANGE[0]) / (DEPTH_RANGE[1] - DEPTH_RANGE[0]);

const LayerGuides: React.FC<{
  width: number;
  pz: (z: number) => number;
}> = ({ width, pz }) => {
  const mid = (r: readonly number[]) => (pz(r[0]) + pz(r[1])) / 2;
  return (
    <g fontSize={24} fontWeight={500} letterSpacing={2}>
      <rect
        x={-40}
        y={pz(LAYERS.gap[1])}
        width={width + 80}
        height={pz(LAYERS.gap[0]) - pz(LAYERS.gap[1])}
        fill="url(#gap-band)"
      />
      {[LAYERS.gap[0], LAYERS.gap[1]].map((z) => (
        <line
          key={z}
          x1={-40}
          x2={width + 40}
          y1={pz(z)}
          y2={pz(z)}
          stroke={ATOM.Cr.color}
          strokeOpacity={0.35}
          strokeWidth={1.5}
          strokeDasharray="3 9"
        />
      ))}
      <text x={-58} y={mid(LAYERS.upperQL)} textAnchor="end" fill={STAGE.muted}>
        QL
      </text>
      <text
        x={-58}
        y={mid(LAYERS.gap) + 8}
        textAnchor="end"
        fill={ATOM.Cr.color}
      >
        vdW GAP
      </text>
      <text x={-58} y={mid(LAYERS.lowerQL)} textAnchor="end" fill={STAGE.muted}>
        QL
      </text>
    </g>
  );
};

export const Lattice: React.FC<Props> = ({
  pathKey,
  t,
  frame,
  width,
  height,
  slideDx = 0,
  jitter = 0,
  trail = false,
  glow = 1,
  hostOpacity = 1,
  showLayers = false,
}) => {
  const path = geometry[pathKey];
  const crIndex = path.species.indexOf("Cr");
  const bonds = useMemo(
    () => hostBonds(path.images[0], crIndex),
    [path, crIndex],
  );
  const scale = Math.min(
    width / (X_RANGE[1] - X_RANGE[0]),
    height / (Z_RANGE[1] - Z_RANGE[0]),
  );
  const px = (x: number) => (x - X_RANGE[0]) * scale;
  const pz = (z: number) => height - (z - Z_RANGE[0]) * scale;
  const idPrefix = `lat-${pathKey}`;

  const atoms = positionsAt(path.images, t).map((p, k) => {
    const isUpper = p[2] > LAYERS.gap[1] - 0.2 && k !== crIndex;
    const amp = k === crIndex ? jitter * 1.6 : jitter;
    const moved: Vec3 = [
      p[0] + (isUpper ? slideDx : 0) + amp * wobble(frame, k, 0),
      p[1],
      p[2] + amp * wobble(frame, k, 2),
    ];
    return { k, sp: path.species[k] as keyof typeof ATOM, p: moved };
  });
  const cr = atoms[crIndex];
  const crBonds = atoms.filter(
    (a) => a.k !== crIndex && dist(a.p, cr.p) < CR_BOND_A,
  );
  const hosts = atoms
    .filter((a) => a.k !== crIndex)
    .sort((a, b) => b.p[1] - a.p[1]);

  const trailPts = trail
    ? path.images
        .slice(0, Math.max(1, Math.floor(t * (path.images.length - 1)) + 1))
        .map((img) => `${px(img[crIndex][0])},${pz(img[crIndex][2])}`)
        .concat(`${px(cr.p[0])},${pz(cr.p[2])}`)
    : [];
  const crX = px(cr.p[0]);
  const crY = pz(cr.p[2]);
  const crR = ATOM.Cr.radius * scale * 1.2;

  return (
    <svg width={width} height={height} style={{ overflow: "visible" }}>
      <defs>
        {Object.entries(ATOM).map(([sp, a]) => (
          <radialGradient
            key={sp}
            id={`${idPrefix}-${sp}`}
            cx="34%"
            cy="30%"
            r="75%"
          >
            <stop offset="0%" stopColor="#ffffff" stopOpacity={0.95} />
            <stop offset="22%" stopColor={a.color} stopOpacity={0.98} />
            <stop offset="78%" stopColor={a.color} stopOpacity={0.9} />
            <stop offset="100%" stopColor={STAGE.deep} stopOpacity={0.85} />
          </radialGradient>
        ))}
        <linearGradient id="gap-band" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor={ATOM.Cr.color} stopOpacity={0} />
          <stop offset="50%" stopColor={ATOM.Cr.color} stopOpacity={0.12} />
          <stop offset="100%" stopColor={ATOM.Cr.color} stopOpacity={0} />
        </linearGradient>
        <filter
          id={`${idPrefix}-glow`}
          x="-300%"
          y="-300%"
          width="700%"
          height="700%"
        >
          <feGaussianBlur stdDeviation={scale * 0.9} />
        </filter>
        <filter
          id={`${idPrefix}-soft`}
          x="-50%"
          y="-50%"
          width="200%"
          height="200%"
        >
          <feGaussianBlur stdDeviation={3} />
        </filter>
      </defs>

      {showLayers ? <LayerGuides width={width} pz={pz} /> : null}

      <g opacity={hostOpacity}>
        {bonds.map(([i, j]) => {
          const a = atoms[i].p;
          const b = atoms[j].p;
          const depth = depthOf((a[1] + b[1]) / 2);
          return (
            <line
              key={`${i}-${j}`}
              x1={px(a[0])}
              y1={pz(a[2])}
              x2={px(b[0])}
              y2={pz(b[2])}
              stroke={BOND_COLOR}
              strokeWidth={scale * 0.12 * (1 - 0.3 * depth)}
              strokeOpacity={0.5 * (1 - DEPTH_FADE * depth)}
              strokeLinecap="round"
            />
          );
        })}
        {hosts.map((a) => {
          const depth = depthOf(a.p[1]);
          return (
            <circle
              key={a.k}
              cx={px(a.p[0])}
              cy={pz(a.p[2])}
              r={ATOM[a.sp].radius * scale * (1 - 0.18 * depth)}
              fill={`url(#${idPrefix}-${a.sp})`}
              opacity={1 - DEPTH_FADE * depth}
            />
          );
        })}
      </g>

      {crBonds.map((a) => (
        <line
          key={`cr-${a.k}`}
          x1={crX}
          y1={crY}
          x2={px(a.p[0])}
          y2={pz(a.p[2])}
          stroke={ATOM.Cr.color}
          strokeWidth={scale * 0.14}
          strokeOpacity={0.75 * glow}
          strokeLinecap="round"
        />
      ))}

      {trail ? (
        <g>
          <polyline
            points={trailPts.join(" ")}
            fill="none"
            stroke={ATOM.Cr.color}
            strokeWidth={10}
            strokeOpacity={0.35}
            strokeLinecap="round"
            strokeLinejoin="round"
            filter={`url(#${idPrefix}-soft)`}
          />
          <polyline
            points={trailPts.join(" ")}
            fill="none"
            stroke="#fff3d6"
            strokeWidth={2.5}
            strokeOpacity={0.9}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>
      ) : null}

      <g opacity={glow}>
        <circle
          cx={crX}
          cy={crY}
          r={crR * 2.6}
          fill={ATOM.Cr.color}
          opacity={0.5}
          filter={`url(#${idPrefix}-glow)`}
        />
        <circle
          cx={crX}
          cy={crY}
          r={crR * 1.5}
          fill="#fff3d6"
          opacity={0.35}
          filter={`url(#${idPrefix}-soft)`}
        />
        <circle
          cx={crX}
          cy={crY}
          r={crR}
          fill={`url(#${idPrefix}-Cr)`}
          stroke="#fff8e8"
          strokeOpacity={0.8}
          strokeWidth={1.5}
        />
      </g>
    </svg>
  );
};
