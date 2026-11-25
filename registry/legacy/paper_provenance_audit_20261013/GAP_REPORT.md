# Publication provenance gap report — 2026-10-13

This report is a publication gate, not a claim that every historical artifact
has already been rebuilt.

## Verified or sufficiently identified

- The old Foundation-OMAT checkpoint has an exact path and SHA256 and is
  separated from public OMAT-0.
- The public OMAT-0 medium checkpoint has a separate path, SHA256, and public
  release URL.
- The MatPES-PBE OMAT-FT checkpoint has an exact path, SHA256, and public
  release URL.
- The original Foundation trajectory and checkpoint locations are known.
- DFT NEB is kept as a model-independent reference, separate from MACE/MLFF
  fixed-path and self-relaxed NEB results.

## Partially verified

- Current grouped FT-MultiT has script-level evidence that it starts from the
  MatPES-PBE OMAT-FT checkpoint.
- Current grouped FT-600K directories support OMAT-FT lineage, but the exact
  source-checkpoint hash is still pending.
- Historical FT-600K and FT-MultiT binaries have hashes, but their original
  launch manifests do not yet prove the exact base checkpoint.
- Some NEB and MD rows can be mapped to a model path, but the manuscript label
  is not always the model's actual lineage.

## Missing before final publication release

- Recover source-checkpoint hashes and launch manifests for historical
  FT-600K/FT-MultiT and all grouped training seeds.
- Recover the exact run manifest for the Revision 1 0.41 eV self-NEB result.
- Rebuild paper-facing tables and figures from the canonical registries after
  excluding the legacy branch and relabelling OMAT-FT rows.
- Attach a clean-checkout rebuild manifest with GitHub commit, Overleaf
  version/commit, environment, input/model hashes, and figure/table hashes.
- Compile the final manuscript and supplement from a clean checkout and
  visually inspect the resulting PDFs.

## Release rule

No unresolved row enters a headline number or figure. A historical failure may
remain in a dedicated failure-mode panel only when its model identity and
scientific failure evidence are explicit.
