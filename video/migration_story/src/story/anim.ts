import { Easing, interpolate } from "remotion";

export const clamp = {
  extrapolateLeft: "clamp",
  extrapolateRight: "clamp",
} as const;
export const EASE_OUT = Easing.bezier(0.16, 1, 0.3, 1);
export const EASE_IN_OUT = Easing.bezier(0.65, 0, 0.35, 1);

/** 0→1 ramp starting at `startSec`, lasting `durSec`. */
export const ramp = (
  frame: number,
  fps: number,
  startSec: number,
  durSec = 0.6,
) =>
  interpolate(frame, [startSec * fps, (startSec + durSec) * fps], [0, 1], {
    ...clamp,
    easing: EASE_OUT,
  });

/** Ramp with ease-in-out, for atoms travelling along a path. */
export const travel = (
  frame: number,
  fps: number,
  startSec: number,
  durSec: number,
) =>
  interpolate(frame, [startSec * fps, (startSec + durSec) * fps], [0, 1], {
    ...clamp,
    easing: EASE_IN_OUT,
  });
