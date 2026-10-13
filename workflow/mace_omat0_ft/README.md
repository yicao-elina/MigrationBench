# OMAT-0 zero-shot → FT-600K / FT-MultiT campaign

This is a new, isolated campaign. It does not rewrite the historical
`legacy_compiled_omat` branch or reuse its model identity.

## Scope

The campaign trains two model families from the public zero-shot
`mace-omat-0-medium.model`, each with training seeds `123`, `234`, and `345`:

1. FT-600K on the grouped 600 K dataset.
2. FT-MultiT on the grouped 300/600/1200 K dataset.

The MD velocity seed is recorded separately from the training seed. The default
mapping is 1100/1200/1300 for training seeds 123/234/345 and is shared across
the two families for paired comparison.

## Gates and milestones

1. **Preflight**: verify public foundation hash, split manifests, dataset file
   hashes, atom/PBC contract, environment, and no output collision.
2. **Training**: submit one Slurm array containing six independent tasks. Each
   task writes a preflight record, exact command, model hash, dataset hashes,
   artifact hash, and training seed.
3. **Model gate**: require a non-empty final artifact and finite static
   energy/forces on the fixed reference structure before any MD/NEB release.
4. **MD smoke**: run a short diagnostic only. It is never used for Dxy.
5. **MD production**: run the 1 ns protocol with immutable segments,
   `latest_restart.extxyz`, atomic `status.json`, walltime continuation, and
   independent model/velocity seed provenance.
6. **NEB**: enable only after an endpoint/path manifest is supplied. Fixed-path
   single-point scoring and self-relaxed MLFF-NEB are separate products.
7. **Analysis/release**: reuse the original publication plot grammar through
   the configured Plot Atlas recipe; produce CSV/LaTeX/PNG/SVG/PDF plus a
   claim-to-artifact matrix and clean-checkout rebuild manifest.

## Safe invocation on Rockfish

```bash
python3 workflow/mace_omat0_ft/scripts/validate_config.py \
  --config workflow/mace_omat0_ft/configs/seed_matrix.yaml

python3 workflow/mace_omat0_ft/scripts/validate_config.py \
  --config workflow/mace_omat0_ft/configs/seed_matrix.yaml \
  --check-files

python3 workflow/mace_omat0_ft/scripts/submit_training.py \
  --config workflow/mace_omat0_ft/configs/seed_matrix.yaml
```

The last command is a dry run. The six training jobs are submitted only after
the remote preflight passes and with the explicit `--submit` flag. All output
is under the configured `remote_root`; existing campaigns are not reused.

### NEB release gate

Start from `configs/neb_manifest.example.yaml`, fill in the endpoint/path
hashes and DFT provenance, then set `neb.enabled: true` in the campaign config.
The submission remains a dry run unless `--submit` is supplied:

```bash
python3 workflow/mace_omat0_ft/scripts/submit_neb.py \
  --config workflow/mace_omat0_ft/configs/seed_matrix.yaml \
  --neb-manifest /path/to/approved_neb_manifest.yaml
```

The array contains one task per family × training seed × enabled NEB mode.
Fixed-path scoring only evaluates the supplied images. Self-relaxed MLFF-NEB
uses the repository's audited runner and records its model/input hashes. Both
outputs are MLFF preconditioners/proposals; the publication barrier must still
come from the separately provenance-checked DFT NEB campaign. A non-empty
output directory, unapproved manifest, missing endpoint hash, or inconsistent
image topology blocks submission rather than triggering an implicit repair.

## Model comparison question

This campaign isolates the effect of the foundation checkpoint. The existing
OMAT-FT campaign starts from `MACE-matpes-pbe-omat-ft.model`, which already
contains a MatPES-PBE/OMAT fine-tuning lineage. The new campaign starts from
public zero-shot OMAT-0 and then applies the same downstream FT protocol.
Differences should be interpreted as a change in initial representation and
pretraining/fine-tuning lineage, not as a pure random-seed effect.
