import React from "react";
import { Easing, interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { MigrationPath, PATH_TITLE, Y_DOMAIN } from "../data";
import { ATOM, COLORS, SERIES_STYLE } from "../theme";
import { Marker } from "./Marker";
import { energyAt, linearScale, niceTicks, smoothPath } from "./geometry";

const MARGIN = { left: 96, right: 24, top: 70, bottom: 86 };
const DRAW_SECONDS = 1.6;
const SERIES_STAGGER_SECONDS = 0.7;
const TICK_FONT = 26;
const TICK_LEN = 9;
const AXIS_W = 2;
const MARKER_PX = 15;
const LABEL_FONT = 30;

type Props = {
  path: MigrationPath;
  letter: string;
  x: number;
  y: number;
  width: number;
  height: number;
  delaySeconds: number;
};

const clampOpts = {
  extrapolateLeft: "clamp",
  extrapolateRight: "clamp",
} as const;

export const EnergyPanel: React.FC<Props> = ({
  path,
  letter,
  x,
  y,
  width,
  height,
  delaySeconds,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const domain = Y_DOMAIN[path.key];
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;
  const sx = linearScale([0, 1], [0, plotW]);
  const sy = linearScale(domain, [plotH, 0]);
  const clipId = `clip-${path.key}`;
  const start = delaySeconds * fps;

  const progressFor = (index: number) =>
    interpolate(
      frame,
      [
        start + index * SERIES_STAGGER_SECONDS * fps,
        start + (index * SERIES_STAGGER_SECONDS + DRAW_SECONDS) * fps,
      ],
      [0, 1],
      { ...clampOpts, easing: Easing.bezier(0.33, 0, 0.2, 1) },
    );

  const reference = path.series.find((s) => s.kind === "reference");
  const riderS = progressFor(0);
  const offScale = path.series.filter((s) => s.barrier_eV > domain[1]);
  const lastSeriesDone = progressFor(path.series.length - 1);

  return (
    <g transform={`translate(${x}, ${y})`}>
      <text
        x={0}
        y={34}
        fontSize={46}
        fontWeight={700}
        fill={COLORS.ink}
        opacity={interpolate(frame, [start - 10, start + 5], [0, 1], clampOpts)}
      >
        {letter}
      </text>
      <text
        x={MARGIN.left}
        y={34}
        fontSize={34}
        fontWeight={600}
        fill={COLORS.ink}
      >
        {PATH_TITLE[path.key] ?? path.label}
      </text>

      <g transform={`translate(${MARGIN.left}, ${MARGIN.top})`}>
        <defs>
          <clipPath id={clipId}>
            <rect x={-6} y={0} width={plotW + 12} height={plotH} />
          </clipPath>
        </defs>

        {niceTicks(domain, 4).map((t) => (
          <g key={t}>
            <line
              x1={-TICK_LEN}
              x2={0}
              y1={sy(t)}
              y2={sy(t)}
              stroke={COLORS.axis}
              strokeWidth={AXIS_W}
            />
            <text
              x={-16}
              y={sy(t) + 9}
              fontSize={TICK_FONT}
              textAnchor="end"
              fill={COLORS.axis}
            >
              {Number(t.toFixed(2)).toString()}
            </text>
          </g>
        ))}
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={plotH}
              y2={plotH + TICK_LEN}
              stroke={COLORS.axis}
              strokeWidth={AXIS_W}
            />
            <text
              x={sx(t)}
              y={plotH + 42}
              fontSize={TICK_FONT}
              textAnchor="middle"
              fill={COLORS.axis}
            >
              {t}
            </text>
          </g>
        ))}
        <line
          x1={0}
          x2={0}
          y1={0}
          y2={plotH}
          stroke={COLORS.axis}
          strokeWidth={AXIS_W}
        />
        <line
          x1={0}
          x2={plotW}
          y1={plotH}
          y2={plotH}
          stroke={COLORS.axis}
          strokeWidth={AXIS_W}
        />
        <text
          x={plotW / 2}
          y={plotH + 80}
          fontSize={LABEL_FONT}
          textAnchor="middle"
          fill={COLORS.ink}
        >
          Reaction coordinate
        </text>
        <text
          transform={`translate(${-74}, ${plotH / 2}) rotate(-90)`}
          fontSize={LABEL_FONT}
          textAnchor="middle"
          fill={COLORS.ink}
        >
          ΔE (eV)
        </text>

        <g clipPath={`url(#${clipId})`}>
          {path.series.map((series, index) => {
            const style = SERIES_STYLE[series.key];
            const d = smoothPath(
              series.points.map((p) => ({ x: sx(p.s), y: sy(p.e) })),
            );
            const progress = progressFor(index);
            return (
              <path
                key={series.key}
                d={d}
                fill="none"
                stroke={style.color}
                strokeWidth={style.width}
                strokeLinecap="round"
                pathLength={1}
                strokeDasharray={style.dashed ? "0.018 0.012" : "1 1"}
                strokeDashoffset={style.dashed ? undefined : 1 - progress}
                opacity={
                  style.dashed
                    ? interpolate(progress, [0, 0.15], [0, 1], clampOpts)
                    : 1
                }
                mask={style.dashed ? `url(#reveal-${path.key})` : undefined}
              />
            );
          })}
          {path.series.map((series, index) => {
            const style = SERIES_STYLE[series.key];
            const progress = progressFor(index);
            return series.points.map((p, i) => (
              <Marker
                key={`${series.key}-${i}`}
                shape={style.marker}
                fill={style.fill}
                color={style.color}
                cx={sx(p.s)}
                cy={sy(p.e)}
                size={MARKER_PX}
                hollow={COLORS.paper}
                opacity={interpolate(
                  progress,
                  [p.s * 0.98, p.s * 0.98 + 0.04],
                  [0, 1],
                  clampOpts,
                )}
              />
            ));
          })}
          <mask id={`reveal-${path.key}`}>
            <rect
              x={-10}
              y={-10}
              width={sx(riderS) + 10}
              height={plotH + 20}
              fill="white"
            />
          </mask>
        </g>

        {reference ? (
          <circle
            cx={sx(riderS)}
            cy={sy(energyAt(reference.points, riderS))}
            r={12}
            fill={ATOM.Cr.color}
            stroke={COLORS.ink}
            strokeWidth={2.5}
            opacity={interpolate(
              riderS,
              [0, 0.05, 0.95, 1],
              [0, 1, 1, 0.35],
              clampOpts,
            )}
          />
        ) : null}

        {offScale.map((series, i) => (
          <text
            key={series.key}
            x={14}
            y={32 + i * 34}
            fontSize={26}
            fontWeight={600}
            textAnchor="start"
            fill={SERIES_STYLE[series.key].color}
            stroke={COLORS.paper}
            strokeWidth={8}
            paintOrder="stroke"
            strokeLinejoin="round"
            opacity={interpolate(lastSeriesDone, [0.6, 1], [0, 1], clampOpts)}
          >
            ▲ {series.barrier_eV.toFixed(1)} eV
          </text>
        ))}
      </g>
    </g>
  );
};
