import React from "react";
import {
  AbsoluteFill,
  interpolate,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { COLORS, fontFamily } from "../theme";

const FADE_SECONDS = 0.4;

// Full-frame scene wrapper with a short fade in/out at its own boundaries.
export const SceneFade: React.FC<{
  durationInFrames: number;
  children: React.ReactNode;
}> = ({ durationInFrames, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const fade = FADE_SECONDS * fps;

  return (
    <AbsoluteFill
      style={{
        backgroundColor: COLORS.background,
        fontFamily,
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
