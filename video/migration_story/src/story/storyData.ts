import geometryRaw from "../../public/data/geometry.json";
import storyRaw from "../../public/data/story.json";
import { findPath, MigrationPath } from "../data";

export type Vec3 = [number, number, number];
export type GeometryPath = {
  file: string;
  sha256: string;
  lattice: number[];
  species: string[];
  images: Vec3[][];
};

type Stat = { mean: number; sd: number; n: number };

export const geometry = geometryRaw.paths as unknown as Record<
  string,
  GeometryPath
>;

export const story = storyRaw as unknown as {
  exported_on: string;
  sources: Record<
    string,
    { chart_id: string; chart_status: string; path: string; sha256: string }
  >;
  static: { family: string; energy_rmse: Stat; force_rmse: Stat }[];
  md: {
    diffusion: {
      family: string;
      window: string;
      D_cm2_s: number;
      sd: number;
      n_seeds: number;
    }[];
  };
  sliding: {
    reference: { s: number; e: number }[];
    reference_status: string;
    families: { family: string; points: ({ s: number } & Stat)[] }[];
    metrics: { family: string; rmse: Stat; peak_image: Stat; drift: Stat }[];
  };
};

// Sliding families are 3-seed means, so they get their own style keys.
const SLIDING_KEYS: Record<string, string> = {
  "Foundation (OMAT-0-M)": "family_foundation",
  "FT-600K": "family_ft600k",
  "FT-MultiT": "family_ftmultit",
  "Scratch-600K": "family_scratch600k",
};

const peak = (pts: { e: number }[]) => Math.max(...pts.map((p) => p.e));

export const slidingAsPath = (): MigrationPath => {
  const ref = story.sliding.reference;
  const refPeak = peak(ref);
  return {
    key: "sliding",
    label: "Interlayer sliding",
    series: [
      {
        key: "qe_reference",
        label: "DFT (QE)",
        kind: "reference",
        status: story.sliding.reference_status,
        points: ref,
        barrier_eV: refPeak,
        barrier_error_eV: 0,
      },
      ...story.sliding.families.map((f) => {
        const points = f.points.map((p) => ({ s: p.s, e: p.mean }));
        return {
          key: SLIDING_KEYS[f.family],
          label: f.family,
          kind: "model" as const,
          status: "FINITE_DIAGNOSTIC",
          points,
          barrier_eV: peak(points),
          barrier_error_eV: peak(points) - refPeak,
        };
      }),
    ],
  };
};

export const nebPath = findPath;
