import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { COLORS, fontFamily, STAGE } from "../theme";

const FADE_SECONDS = 0.5;
const DUST_COUNT = 46;
const BEAM_OPACITY = 0.13;

// Deterministic 0..1 hash so dust positions are identical on every render.
const hash = (i: number, salt: number) => {
  const x = Math.sin(i * 127.1 + salt * 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/**
 * Persistent cinematic backdrop: deep slate gradient, one soft light beam,
 * slow dust motes and a vignette. Rendered once behind the whole story so
 * cuts between scenes never flash.
 */
export const Stage: React.FC = () => {
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const sway = Math.sin(frame / 160) * 3;

  return (
    <AbsoluteFill style={{ backgroundColor: STAGE.deep, overflow: "hidden" }}>
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse 70% 80% at 72% 18%, ${STAGE.haze} 0%, ${STAGE.mid} 42%, ${STAGE.deep} 100%)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          left: width * 0.52,
          top: -height * 0.35,
          width: width * 0.22,
          height: height * 1.7,
          background: `linear-gradient(180deg, ${STAGE.beam} 0%, transparent 75%)`,
          opacity: BEAM_OPACITY,
          filter: "blur(60px)",
          rotate: `${24 + sway}deg`,
          transformOrigin: "50% 0%",
        }}
      />
      <svg width={width} height={height} style={{ position: "absolute" }}>
        {Array.from({ length: DUST_COUNT }, (_, i) => {
          const speed = 0.08 + hash(i, 3) * 0.25;
          const x = (hash(i, 1) * width + frame * speed * 0.6) % width;
          const y = (hash(i, 2) * height - frame * speed + height * 4) % height;
          const twinkle = 0.5 + 0.5 * Math.sin(frame / 25 + i);
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r={1 + hash(i, 4) * 2.2}
              fill={STAGE.beam}
              opacity={(0.08 + hash(i, 5) * 0.22) * twinkle}
            />
          );
        })}
      </svg>
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse 85% 85% at 50% 50%, transparent 55%, rgba(0,0,0,0.55) 100%)",
        }}
      />
    </AbsoluteFill>
  );
};

/** Scene layer on the stage: fades its own content in and out. */
export const StageScene: React.FC<{
  durationInFrames: number;
  children: React.ReactNode;
}> = ({ durationInFrames, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const fade = FADE_SECONDS * fps;
  return (
    <AbsoluteFill
      style={{
        fontFamily,
        color: STAGE.ink,
        opacity: interpolate(
          frame,
          [0, fade, durationInFrames - fade, durationInFrames],
          [0, 1, 1, 0],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" },
        ),
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

/** Light figure card: every quantitative plot sits on paper, like a journal figure. */
export const PaperCard: React.FC<{
  style?: React.CSSProperties;
  children: React.ReactNode;
}> = ({ style, children }) => (
  <div
    style={{
      background: COLORS.paper,
      color: COLORS.ink,
      borderRadius: 4,
      padding: "34px 40px 30px",
      boxShadow:
        "0 40px 90px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.06)",
      ...style,
    }}
  >
    {children}
  </div>
);
