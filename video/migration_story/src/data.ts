import raw from "../public/data/cr_neb_profiles.json";

export type Point = { s: number; e: number };

export type Series = {
  key: string;
  label: string;
  kind: "model" | "reference";
  status: string;
  points: Point[];
  barrier_eV: number;
  barrier_error_eV: number | null;
};

export type MigrationPath = { key: string; label: string; series: Series[] };

export type Provenance = {
  chart_id: string;
  chart_status: string;
  source_tsv: string;
  source_sha256: string;
  exported_on: string;
  caveat: string;
};

export const provenance = raw.provenance as Provenance;
export const paths = raw.paths as MigrationPath[];

// Per-path energy window (eV). Points outside are clipped and flagged off-scale.
export const Y_DOMAIN: Record<string, [number, number]> = {
  within_gap: [-0.2, 1.8],
  gap_to_ql: [-0.15, 0.1],
  within_ql: [-1.0, 3.5],
  sliding: [-1.2, 3.0],
};

export const PATH_TITLE: Record<string, string> = {
  within_gap: "Within vdW gap",
  gap_to_ql: "Gap → quintuple layer",
  within_ql: "Within quintuple layer",
  sliding: "Interlayer sliding (Cr in gap)",
};

export const findPath = (key: string): MigrationPath => {
  const path = paths.find((p) => p.key === key);
  if (!path) {
    throw new Error(`Migration path "${key}" missing from exported data`);
  }
  return path;
};
