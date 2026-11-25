import { loadFont } from "@remotion/google-fonts/Inter";
import { loadFont as loadSerif } from "@remotion/google-fonts/SourceSerif4";

export const { fontFamily } = loadFont("normal", {
  weights: ["400", "500", "600", "700"],
  subsets: ["latin"],
});

// Display serif for headlines, in the register of journal cover lines.
export const { fontFamily: serifFamily } = loadSerif("normal", {
  weights: ["400", "600"],
  subsets: ["latin"],
});

// All hex values below follow the MigrationBench Global Visual Standard
// (shared visual vocabulary). Change a colour here, never in a scene.

/** Light "paper" surface used for every data figure. */
export const COLORS = {
  background: "#ffffff",
  paper: "#fbfaf7",
  ink: "#303840",
  muted: "#6f7a84",
  axis: "#303840",
  grid: "#e6e8ea",
  caveat: "#9a6a00",
} as const;

/** Dark cinematic stage behind the narrative scenes. */
export const STAGE = {
  deep: "#0b1117",
  mid: "#16212b",
  haze: "#2a3a48",
  beam: "#cfe3ee",
  ink: "#eef2f4",
  muted: "#9aa7b1",
  faint: "#5d6b77",
  caveat: "#e4b44c",
} as const;

/** Evidence state: never encoded as energy. */
export const EVIDENCE = {
  reference: "#303840",
  failed: "#bd6472",
  pending: "#a6afb7",
  nonconverged: "#78838d",
} as const;

/** Model strategy hues. Thermal regime is encoded by fill, not colour. */
export const STRATEGY = {
  zeroShot: "#8fc7ab",
  fineTuning: "#29a7b6",
  scratch: "#327fb1",
} as const;

export type MarkerShape = "circle" | "triangle" | "square" | "diamond" | "none";
/** open = 600 K, filled = multi-T, ring = regime not encoded. */
export type MarkerFill = "open" | "filled" | "ring";

export type SeriesStyle = {
  color: string;
  dashed: boolean;
  marker: MarkerShape;
  fill: MarkerFill;
  width: number;
};

const model = (
  color: string,
  marker: MarkerShape,
  fill: MarkerFill,
): SeriesStyle => ({ color, dashed: false, marker, fill, width: 3.5 });

// Seed shape: circle = 123, triangle = 234, square = 345; checkpoint = diamond.
export const SERIES_STYLE: Record<string, SeriesStyle> = {
  qe_reference: {
    color: EVIDENCE.reference,
    dashed: false,
    marker: "none",
    fill: "filled",
    width: 5,
  },
  foundation_omat0_medium: model(STRATEGY.zeroShot, "diamond", "filled"),
  ft600k_seed123: model(STRATEGY.fineTuning, "circle", "open"),
  ftmultit_seed234: model(STRATEGY.fineTuning, "triangle", "filled"),
  // NEB from-scratch checkpoint is trained on all data: regime not encoded.
  from_scratch_seed123: model(STRATEGY.scratch, "circle", "ring"),
  // Family means (over seeds) carry strategy + regime only; circle is generic.
  family_foundation: model(STRATEGY.zeroShot, "diamond", "filled"),
  family_ft600k: model(STRATEGY.fineTuning, "circle", "open"),
  family_ftmultit: model(STRATEGY.fineTuning, "circle", "filled"),
  family_scratch600k: model(STRATEGY.scratch, "circle", "open"),
  family_scratchmultit: model(STRATEGY.scratch, "circle", "filled"),
};

export const FPS = 30;
export const WIDTH = 1920;
export const HEIGHT = 1080;

export const ATOM = {
  Te: { color: "#c88db7", radius: 0.6 },
  Sb: { color: "#75bec8", radius: 0.54 },
  Cr: { color: "#e4b44c", radius: 0.6 },
} as const;
export const BOND_COLOR = "#bcc3c8";

// Task identity colours.
export const PROBE = {
  static: { label: "Static", color: "#73869a", asks: "Pointwise E / F" },
  dynamics: { label: "Dynamics", color: "#5c9b7c", asks: "Rollout stability" },
  within_gap: { label: "In-gap", color: "#d28a32", asks: "Local motion" },
  gap_to_ql: {
    label: "Gap-to-QL",
    color: "#cd6d62",
    asks: "Entry environment",
  },
  within_ql: {
    label: "Within-QL",
    color: "#7d5c99",
    asks: "Intralayer motion",
  },
  sliding: { label: "Sliding", color: "#287e8a", asks: "Collective motion" },
} as const;

export type ProbeKey = keyof typeof PROBE;
export const PROBE_ORDER: ProbeKey[] = [
  "static",
  "dynamics",
  "within_gap",
  "gap_to_ql",
  "within_ql",
  "sliding",
];
