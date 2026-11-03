import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { Marker } from "../../components/Marker";
import {
  ATOM,
  COLORS,
  PROBE,
  PROBE_ORDER,
  ProbeKey,
  SERIES_STYLE,
  STAGE,
} from "../../theme";
import { ramp } from "../anim";
import { Lattice } from "../Lattice";
import { PaperCard, StageScene } from "../Stage";
import { Body, Headline, Kicker } from "./OpeningScenes";
import { nebPath, story } from "../storyData";

type Grade = "good" | "warn" | "bad" | "neutral" | "na";
type Cell = { text: string; grade: Grade };

// Sequential single-hue ramp (light = close to DFT, dark = far): deviation is
// a magnitude, so it is never shown with categorical or evidence colours.
const GRADE_STYLE: Record<Grade, { bg: string; fg: string; border: string }> = {
  good: { bg: "#eef0f2", fg: COLORS.ink, border: "transparent" },
  warn: { bg: "#c3cad1", fg: COLORS.ink, border: "transparent" },
  bad: { bg: "#4a555f", fg: "#ffffff", border: "transparent" },
  neutral: { bg: COLORS.paper, fg: COLORS.ink, border: "#c3cad1" },
  na: { bg: "transparent", fg: COLORS.muted, border: "transparent" },
};
// |ratio − 1| (and entry deviation / DFT span) thresholds, disclosed on screen.
const GOOD_TOL = 0.1;
const WARN_TOL = 0.3;
const SLIDING_RMSE_TOL_EV = 0.5;
const SLIDING_FAMILY: Record<string, string> = {
  foundation_omat0_medium: "Foundation (OMAT-0-M)",
  ft600k_seed123: "FT-600K",
  ftmultit_seed234: "FT-MultiT",
  from_scratch_seed123: "Scratch-600K",
};

const ROWS = [
  {
    key: "foundation_omat0_medium",
    style: "family_foundation",
    label: "Foundation",
    staticFamily: null,
    md: "fail",
  },
  {
    key: "ft600k_seed123",
    style: "family_ft600k",
    label: "FT-600K",
    staticFamily: "FT-600K",
    md: "stable",
  },
  {
    key: "ftmultit_seed234",
    style: "family_ftmultit",
    label: "FT-MultiT",
    staticFamily: "FT-MultiT",
    md: "stable",
  },
  {
    key: "from_scratch_seed123",
    style: "from_scratch_seed123",
    label: "From scratch*",
    staticFamily: "Scratch-600K",
    md: "fail",
  },
] as const;

const ratioCell = (ratio: number | undefined): Cell => {
  if (ratio === undefined || !Number.isFinite(ratio))
    return { text: "—", grade: "na" };
  const dev = Math.abs(ratio - 1);
  const grade = dev < GOOD_TOL ? "good" : dev < WARN_TOL ? "warn" : "bad";
  return {
    text: `${ratio < 10 ? ratio.toFixed(2) : ratio.toFixed(0)}×`,
    grade,
  };
};

const peakRatio = (pathKey: string, seriesKey: string): number | undefined => {
  const path = nebPath(pathKey);
  const ref = path.series.find((s) => s.kind === "reference");
  const s = path.series.find((x) => x.key === seriesKey);
  return ref && s ? s.barrier_eV / ref.barrier_eV : undefined;
};

// Entry is barrierless in DFT, so a peak ratio is undefined. Grade the largest
// profile deviation instead, relative to the DFT energy span along the path,
// with the same tolerances as the NEB peak ratios.
const entryCell = (seriesKey: string): Cell => {
  const path = nebPath("gap_to_ql");
  const ref = path.series.find((s) => s.kind === "reference");
  const s = path.series.find((x) => x.key === seriesKey);
  if (!ref || !s || s.points.length !== ref.points.length)
    return { text: "—", grade: "na" };
  const energies = ref.points.map((p) => p.e);
  const span = Math.max(...energies) - Math.min(...energies);
  const maxDev = Math.max(
    ...s.points.map((p, i) => Math.abs(p.e - ref.points[i].e)),
  );
  const rel = maxDev / span;
  const grade = rel < GOOD_TOL ? "good" : rel < WARN_TOL ? "warn" : "bad";
  return { text: `Δ ${(maxDev * 1000).toFixed(0)} meV`, grade };
};

const buildCell = (row: (typeof ROWS)[number], probe: ProbeKey): Cell => {
  switch (probe) {
    case "static": {
      const s = story.static.find((x) => x.family === row.staticFamily);
      return s
        ? { text: `${s.energy_rmse.mean.toFixed(0)} meV`, grade: "neutral" }
        : { text: "—", grade: "na" };
    }
    case "dynamics":
      return row.md === "stable"
        ? { text: "stable", grade: "good" }
        : { text: "fails", grade: "bad" };
    case "gap_to_ql":
      return entryCell(row.key);
    case "sliding": {
      const m = story.sliding.metrics.find(
        (x) => x.family === SLIDING_FAMILY[row.key],
      );
      return m
        ? {
            text: `${m.rmse.mean.toFixed(1)} eV`,
            grade: m.rmse.mean < SLIDING_RMSE_TOL_EV ? "warn" : "bad",
          }
        : { text: "—", grade: "na" };
    }
    default:
      return ratioCell(peakRatio(probe, row.key));
  }
};

const CELL_W = 214;
const CELL_H = 88;
const LABEL_W = 270;

export const ScorecardScene: React.FC<{ durationInFrames: number }> = ({
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const card = ramp(frame, fps, 0.4, 0.9);

  return (
    <StageScene durationInFrames={durationInFrames}>
      <div style={{ position: "absolute", left: 150, top: 70 }}>
        <Kicker opacity={ramp(frame, fps, 0.1)}>The verdict</Kicker>
        <Headline opacity={ramp(frame, fps, 0.2)} size={66}>
          Static accuracy does not certify migration
        </Headline>
      </div>
      <PaperCard
        style={{
          position: "absolute",
          left: 150,
          right: 150,
          top: 270,
          padding: "36px 44px 30px",
          opacity: card,
          translate: `0px ${(1 - card) * 24}px`,
        }}
      >
        <div style={{ display: "flex", marginLeft: LABEL_W }}>
          {PROBE_ORDER.map((p, j) => (
            <div
              key={p}
              style={{
                width: CELL_W,
                padding: "0 8px",
                opacity: ramp(frame, fps, 0.8 + j * 0.4),
              }}
            >
              <div style={{ height: 4, background: PROBE[p].color }} />
              <div
                style={{
                  fontSize: 26,
                  fontWeight: 600,
                  color: COLORS.ink,
                  marginTop: 12,
                }}
              >
                {PROBE[p].label}
              </div>
            </div>
          ))}
        </div>
        {ROWS.map((row) => {
          const st = SERIES_STYLE[row.style];
          return (
            <div
              key={row.key}
              style={{ display: "flex", alignItems: "center", marginTop: 12 }}
            >
              <div
                style={{
                  width: LABEL_W,
                  display: "flex",
                  alignItems: "center",
                  gap: 16,
                  fontSize: 29,
                  fontWeight: 600,
                  color: COLORS.ink,
                }}
              >
                <svg width={26} height={26}>
                  <Marker
                    shape={st.marker}
                    fill={st.fill}
                    color={st.color}
                    cx={13}
                    cy={13}
                    size={22}
                    hollow={COLORS.paper}
                  />
                </svg>
                {row.label}
              </div>
              {PROBE_ORDER.map((p, j) => {
                const cell = buildCell(row, p);
                const g = GRADE_STYLE[cell.grade];
                return (
                  <div
                    key={p}
                    style={{
                      width: CELL_W,
                      padding: "0 8px",
                      opacity: ramp(frame, fps, 1.0 + j * 0.4),
                    }}
                  >
                    <div
                      style={{
                        height: CELL_H,
                        background: g.bg,
                        border: `2px solid ${g.border}`,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        fontSize: 30,
                        fontWeight: 600,
                        fontVariantNumeric: "tabular-nums",
                        color: g.fg,
                      }}
                    >
                      {cell.text}
                    </div>
                  </div>
                );
              })}
            </div>
          );
        })}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
            marginTop: 26,
            marginLeft: LABEL_W + 8,
            fontSize: 22,
            color: COLORS.muted,
            opacity: ramp(frame, fps, 3.4),
          }}
        >
          <span>close to DFT</span>
          {(["good", "warn", "bad"] as const).map((g) => (
            <span
              key={g}
              style={{
                width: 54,
                height: 18,
                background: GRADE_STYLE[g].bg,
                border: `1px solid ${GRADE_STYLE.warn.bg}`,
              }}
            />
          ))}
          <span>far from DFT</span>
        </div>
      </PaperCard>
      <div
        style={{
          position: "absolute",
          left: 150,
          right: 150,
          bottom: 44,
          fontSize: 21,
          color: STAGE.muted,
          lineHeight: 1.45,
          opacity: ramp(frame, fps, 3.6),
        }}
      >
        NEB: predicted / DFT peak on the fixed path (lightest |ratio − 1| &lt;
        10%, middle &lt; 30%, dark otherwise). Gap-to-QL: max |ΔE − ΔE(DFT)|
        along the path, graded against the 0.096 eV DFT drop with the same
        cut-offs. Sliding: profile RMSE vs DFT. Static RMSE has no DFT-relative
        grade.
        <br />
        *From-scratch checkpoints differ between probes (S600 for
        static/sliding, seed-123 all-data for MD/NEB); lineages are not pooled.
      </div>
    </StageScene>
  );
};

export const FinaleScene: React.FC<{ durationInFrames: number }> = ({
  durationInFrames,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const statuses = Object.values(story.sources)
    .map((s) => `${s.chart_id.replace(/^chart_/, "")} · ${s.chart_status}`)
    .join("   ");

  return (
    <StageScene durationInFrames={durationInFrames}>
      <div
        style={{
          position: "absolute",
          left: 940,
          top: 140,
          opacity: ramp(frame, fps, 0.2, 1.4),
        }}
      >
        <Lattice
          pathKey="within_ql"
          t={1}
          frame={frame}
          width={840}
          height={740}
          trail
        />
      </div>
      <div style={{ position: "absolute", left: 150, top: 300, width: 760 }}>
        <Kicker opacity={ramp(frame, fps, 0.4)} color={ATOM.Cr.color}>
          MigrationBench
        </Kicker>
        <Headline opacity={ramp(frame, fps, 0.9)} size={80}>
          Test MLIPs on the motion you need.
        </Headline>
        <Body opacity={ramp(frame, fps, 1.8)}>
          A task ladder from static frames to collective sliding shows where a
          potential stops following Cr.
        </Body>
      </div>
      <div
        style={{
          position: "absolute",
          left: 150,
          right: 150,
          bottom: 40,
          fontSize: 19,
          color: STAGE.faint,
          opacity: ramp(frame, fps, 2.6),
        }}
      >
        Data: MigrationBench asset registry · {statuses}
      </div>
    </StageScene>
  );
};
