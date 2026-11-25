import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import type { Series } from "../data";
import { COLORS, SERIES_STYLE } from "../theme";

type Props = { series: Series[]; delaySeconds: number; staggerSeconds: number };

export const Legend: React.FC<Props> = ({
  series,
  delaySeconds,
  staggerSeconds,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div
      style={{
        display: "flex",
        gap: 30,
        justifyContent: "center",
        fontSize: 27,
        whiteSpace: "nowrap",
        color: COLORS.ink,
      }}
    >
      {series.map((s, i) => {
        const style = SERIES_STYLE[s.key];
        const at = (delaySeconds + i * staggerSeconds) * fps;
        return (
          <div
            key={s.key}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 14,
              opacity: interpolate(frame, [at, at + 0.4 * fps], [0, 1], {
                extrapolateLeft: "clamp",
                extrapolateRight: "clamp",
              }),
            }}
          >
            <svg width={52} height={12}>
              <line
                x1={2}
                x2={50}
                y1={6}
                y2={6}
                stroke={style.color}
                strokeWidth={5}
                strokeDasharray={style.dashed ? "9 6" : undefined}
                strokeLinecap="round"
              />
            </svg>
            <span style={{ fontWeight: s.kind === "reference" ? 600 : 400 }}>
              {s.kind === "reference" ? "DFT (QE)" : s.label}
            </span>
          </div>
        );
      })}
    </div>
  );
};
