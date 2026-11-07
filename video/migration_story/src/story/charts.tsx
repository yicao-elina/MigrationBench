import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { EnergyPanel } from "../components/EnergyPanel";
import { logScale } from "../components/geometry";
import { Marker } from "../components/Marker";
import type { MigrationPath } from "../data";
import { COLORS, EVIDENCE, SERIES_STYLE } from "../theme";
import { clamp, ramp } from "./anim";
import { story } from "./storyData";

export const CHART_W = 820;
const AXIS_W = 2;
const TICK_LEN = 9;
const TICK_FONT = 26;

const LegendSwatch: React.FC<{ styleKey: string; line?: boolean }> = ({
  styleKey,
  line = true,
}) => {
  const st = SERIES_STYLE[styleKey];
  return (
    <svg width={line ? 46 : 22} height={22}>
      {line ? (
        <line
          x1={2}
          x2={44}
          y1={11}
          y2={11}
          stroke={st.color}
          strokeWidth={st.width}
          strokeLinecap="round"
        />
      ) : null}
      <Marker
        shape={st.marker === "none" ? "none" : st.marker}
        fill={st.fill}
        color={st.color}
        cx={line ? 23 : 11}
        cy={11}
        size={15}
        hollow={COLORS.paper}
      />
    </svg>
  );
};

export const CompactLegend: React.FC<{
  path: MigrationPath;
  startSec: number;
}> = ({ path, startSec }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "10px 28px",
        fontSize: 23,
        color: COLORS.ink,
        width: CHART_W,
        paddingLeft: 96,
        marginTop: 16,
        opacity: ramp(frame, fps, startSec),
      }}
    >
      {path.series.map((s) => (
        <span
          key={s.key}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            whiteSpace: "nowrap",
            fontWeight: s.kind === "reference" ? 600 : 400,
          }}
        >
          <LegendSwatch styleKey={s.key} />
          {s.kind === "reference" ? "DFT (QE)" : s.label}
        </span>
      ))}
    </div>
  );
};

export const ProfileChart: React.FC<{
  path: MigrationPath;
  startSec: number;
  letter: string;
}> = ({ path, startSec, letter }) => (
  <div>
    <svg width={CHART_W} height={500} style={{ overflow: "visible" }}>
      <EnergyPanel
        path={path}
        letter={letter}
        x={0}
        y={0}
        width={CHART_W}
        height={500}
        delaySeconds={startSec}
      />
    </svg>
    <CompactLegend path={path} startSec={startSec} />
  </div>
);

const ChartTitle: React.FC<{ letter: string; children: React.ReactNode }> = ({
  letter,
  children,
}) => (
  <div
    style={{
      display: "flex",
      alignItems: "baseline",
      gap: 22,
      fontSize: 30,
      fontWeight: 600,
      color: COLORS.ink,
      marginBottom: 24,
    }}
  >
    <span style={{ fontSize: 46, fontWeight: 700 }}>{letter}</span>
    {children}
  </div>
);

// Static families: strategy hue, fill encodes regime (open 600 K, filled multi-T).
const STATIC_KEYS: Record<string, string> = {
  "FT-600K": "family_ft600k",
  "FT-MultiT": "family_ftmultit",
  "Scratch-600K": "family_scratch600k",
  "Scratch-MultiT": "family_scratchmultit",
};
const STATIC_MAX = 50;
const BAR_H = 40;
const ROW_H = 84;

/** Held-out energy RMSE per family (mean ± sd over 3 seeds). */
export const StaticChart: React.FC<{ startSec: number; letter: string }> = ({
  startSec,
  letter,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const labelW = 220;
  const plotW = 470;
  const sx = (v: number) => (v / STATIC_MAX) * plotW;
  const plotH = story.static.length * ROW_H;
  return (
    <div style={{ width: CHART_W }}>
      <ChartTitle letter={letter}>Energy RMSE, held-out AIMD frames</ChartTitle>
      <svg width={CHART_W} height={plotH + 100}>
        <g transform={`translate(${labelW}, 10)`}>
          {story.static.map((s, i) => {
            const grow = ramp(frame, fps, startSec + i * 0.25, 1.0);
            const st = SERIES_STYLE[STATIC_KEYS[s.family]];
            const y = i * ROW_H + (ROW_H - BAR_H) / 2;
            const w = sx(s.energy_rmse.mean) * grow;
            const lo = sx(s.energy_rmse.mean - s.energy_rmse.sd);
            const hi = sx(s.energy_rmse.mean + s.energy_rmse.sd);
            const cy = y + BAR_H / 2;
            return (
              <g key={s.family}>
                <text
                  x={-20}
                  y={cy + 9}
                  fontSize={27}
                  textAnchor="end"
                  fill={COLORS.ink}
                >
                  {s.family}
                </text>
                <rect
                  x={0}
                  y={y}
                  width={w}
                  height={BAR_H}
                  fill={st.fill === "filled" ? st.color : COLORS.paper}
                  stroke={st.color}
                  strokeWidth={3}
                />
                <g opacity={grow} stroke={COLORS.ink} strokeWidth={2}>
                  <line x1={lo} x2={hi} y1={cy} y2={cy} />
                  <line x1={lo} x2={lo} y1={cy - 8} y2={cy + 8} />
                  <line x1={hi} x2={hi} y1={cy - 8} y2={cy + 8} />
                </g>
                <text
                  x={hi + 16}
                  y={cy + 9}
                  fontSize={27}
                  fontWeight={600}
                  fill={COLORS.ink}
                  opacity={grow}
                >
                  {s.energy_rmse.mean.toFixed(1)}
                </text>
              </g>
            );
          })}
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
          {[0, 10, 20, 30, 40, 50].map((v) => (
            <g key={v}>
              <line
                x1={sx(v)}
                x2={sx(v)}
                y1={plotH}
                y2={plotH + TICK_LEN}
                stroke={COLORS.axis}
                strokeWidth={AXIS_W}
              />
              <text
                x={sx(v)}
                y={plotH + 40}
                fontSize={TICK_FONT}
                textAnchor="middle"
                fill={COLORS.axis}
              >
                {v}
              </text>
            </g>
          ))}
          <text
            x={plotW / 2}
            y={plotH + 82}
            fontSize={27}
            textAnchor="middle"
            fill={COLORS.ink}
          >
            Energy RMSE (meV atom⁻¹)
          </text>
        </g>
      </svg>
      <div style={{ fontSize: 22, color: COLORS.muted, paddingLeft: 40 }}>
        Mean ± s.d., 3 seeds. Open, 600 K training; filled, multi-T.
      </div>
    </div>
  );
};

const MD_ROWS = [
  {
    family: "FT-600K",
    window: "500_900_ps",
    key: "family_ft600k",
    failed: false,
  },
  {
    family: "FT-MultiT",
    window: "500_900_ps",
    key: "family_ftmultit",
    failed: false,
  },
  {
    family: "From-scratch",
    window: "pre_failure",
    key: "from_scratch_seed123",
    failed: true,
  },
  {
    family: "Foundation-OMAT",
    window: "pre_failure",
    key: "foundation_omat0_medium",
    failed: true,
  },
];

const superscript = (n: number) =>
  String(n)
    .replace("-", "⁻")
    .replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[Number(d)]);

/** Diffusion coefficient from 1 ns MD (log axis); failed rollouts flagged. */
export const DynamicsChart: React.FC<{ startSec: number; letter: string }> = ({
  startSec,
  letter,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const plotW = 440;
  const rowH = 92;
  const plotH = MD_ROWS.length * rowH;
  const sx = logScale([1e-7, 1e-1], [0, plotW]);
  const ticks = [1e-7, 1e-5, 1e-3, 1e-1];
  return (
    <div style={{ width: CHART_W }}>
      <ChartTitle letter={letter}>Cr-cell diffusion in 1 ns MD</ChartTitle>
      <svg width={CHART_W} height={plotH + 100}>
        <g transform="translate(270, 10)">
          <line
            x1={0}
            x2={plotW}
            y1={plotH}
            y2={plotH}
            stroke={COLORS.axis}
            strokeWidth={AXIS_W}
          />
          {ticks.map((t) => (
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
                10{superscript(Math.round(Math.log10(t)))}
              </text>
            </g>
          ))}
          <text
            x={plotW / 2}
            y={plotH + 84}
            fontSize={27}
            textAnchor="middle"
            fill={COLORS.ink}
          >
            D (cm² s⁻¹)
          </text>
          {MD_ROWS.map((r, i) => {
            const d = story.md.diffusion.find(
              (x) => x.family === r.family && x.window === r.window,
            );
            if (!d) return null;
            const o = ramp(frame, fps, startSec + i * 0.5, 0.8);
            const cx = interpolate(o, [0, 1], [sx(1e-7), sx(d.D_cm2_s)], clamp);
            const cy = rowH / 2 + i * rowH - 10;
            const st = SERIES_STYLE[r.key];
            return (
              <g key={r.family} opacity={o}>
                <text
                  x={-24}
                  y={cy + 9}
                  fontSize={27}
                  textAnchor="end"
                  fill={COLORS.ink}
                >
                  {r.family}
                </text>
                <line
                  x1={0}
                  x2={cx}
                  y1={cy}
                  y2={cy}
                  stroke={st.color}
                  strokeOpacity={0.35}
                  strokeWidth={2}
                />
                <Marker
                  shape={st.marker}
                  fill={st.fill}
                  color={st.color}
                  cx={cx}
                  cy={cy}
                  size={26}
                  hollow={COLORS.paper}
                />
                {r.failed ? (
                  <text
                    x={cx}
                    y={cy + 42}
                    textAnchor="middle"
                    fontSize={22}
                    fontWeight={600}
                    fill={EVIDENCE.failed}
                  >
                    × rollout fails
                  </text>
                ) : null}
              </g>
            );
          })}
        </g>
      </svg>
      <div style={{ fontSize: 22, color: COLORS.muted, paddingLeft: 40 }}>
        Stable families: 500–900 ps window. Failed rollouts: pre-failure window.
      </div>
    </div>
  );
};
