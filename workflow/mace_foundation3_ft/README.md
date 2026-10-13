# Foundation3 seed-123 additive campaign

This workflow is isolated from the dirty working checkout and from the
protected OMAT-0 campaign. It records foundation lineage separately from the
training seed and the MD velocity seed.

The core path contains four jobs:

- MACE-MPA-0 → FT-600K / FT-MultiT, training seed 123
- MACE-MP-0b3 → FT-600K / FT-MultiT, training seed 123

The PolarMACE path is intentionally separate. It requires the scratch venv,
the official `mace_polar` loader, and an explicit `charge`/`spin`/
`external_field` structure contract. The current shared data and MD input do
not provide those fields, so the Polar branch is blocked and must not be
substituted with ordinary MACE.

## Validation and smoke

Run on Rockfish after synchronizing this directory below the new remote root:

```bash
python3 scripts/validate_config.py \
  --config configs/seed_matrix.yaml \
  --check-files \
  --families mpa0_ft600k,mpa0_multit,mp0b3_ft600k,mp0b3_multit

python3 scripts/submit_training.py \
  --config configs/seed_matrix.yaml \
  --families mpa0_ft600k,mpa0_multit,mp0b3_ft600k,mp0b3_multit \
  --runtime core --smoke --submit
```

Production training is released only after all four smoke manifests show the
required compatibility evidence. Each output directory must be absent before
the worker starts; failed or restarted work receives a new non-overwriting
directory.

Before MD or NEB, run `static_model_gate.py` for every completed training
manifest. Smoke trajectories are diagnostics only and never supply Dxy.
