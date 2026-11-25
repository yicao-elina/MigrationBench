import React from "react";
import { MarkerFill, MarkerShape } from "../theme";

type Props = {
  shape: MarkerShape;
  fill: MarkerFill;
  color: string;
  cx: number;
  cy: number;
  size: number;
  /** Fill used inside open markers (the surface colour). */
  hollow?: string;
  opacity?: number;
};

const shapePath = (shape: MarkerShape, cx: number, cy: number, r: number) => {
  switch (shape) {
    case "triangle":
      return `M${cx},${cy - r * 1.15} L${cx + r},${cy + r * 0.72} L${cx - r},${cy + r * 0.72} Z`;
    case "square":
      return `M${cx - r * 0.85},${cy - r * 0.85} h${r * 1.7} v${r * 1.7} h${-r * 1.7} Z`;
    case "diamond":
      return `M${cx},${cy - r * 1.2} L${cx + r},${cy} L${cx},${cy + r * 1.2} L${cx - r},${cy} Z`;
    default:
      return null;
  }
};

/** Data marker following the visual standard's seed / regime encoding. */
export const Marker: React.FC<Props> = ({
  shape,
  fill,
  color,
  cx,
  cy,
  size,
  hollow = "#ffffff",
  opacity = 1,
}) => {
  if (shape === "none") return null;
  const r = size / 2;
  const stroke = Math.max(2, size * 0.18);
  const body = fill === "filled" ? color : hollow;
  const d = shapePath(shape, cx, cy, r);
  return (
    <g opacity={opacity}>
      {d ? (
        <path
          d={d}
          fill={body}
          stroke={color}
          strokeWidth={stroke}
          strokeLinejoin="round"
        />
      ) : (
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill={body}
          stroke={color}
          strokeWidth={stroke}
        />
      )}
      {fill === "ring" ? (
        <circle cx={cx} cy={cy} r={r * 0.36} fill={color} />
      ) : null}
    </g>
  );
};
