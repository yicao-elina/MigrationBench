# Foundation3 preflight gap report

As of 2026-10-13, the protected OMAT-0 campaign remains pending as jobs
31917025_[0] and 31917028_[0-5]. It was not resubmitted or modified.

Verified:

- The new remote root was absent before creation and now contains isolated model/config/training/ledger namespaces.
- FT-600K, Multi-T, and the common 2050-atom MD input match the prescribed SHA256 values.
- MPA-0, MP-0b3, and Polar-1-M model files were downloaded from the official release URLs and hash-checked.
- The ordinary MACE core preflight is collision-free for the four MPA-0/MP-0b3 branches.
- The scratch Polar venv imports `mace_polar`, `mace-torch==0.3.16`, and `graph_longrange==0.4.4`; the Polar checkpoint loads.

Blocked:

- PolarMACE input contract: the common MD input has 2050 atoms, fully periodic
  PBC, finite positions/cell, but no `charge`, `spin`, or `external_field` in
  `atoms.info`. The branch is `blocked` / `EXCLUDE_FROM_PUBLICATION`; no
  Polar training, smoke, or MD is submitted.
- Downstream core training/model/MD/NEB/analysis evidence is pending the core
  smoke array `31917059_[0-3]` and its scientific gates.

Required next gates:

1. Core smoke must produce one-epoch compatibility manifests and finite loss/validation artifacts for all four branches.
2. Only smoke-passing branches may enter production training.
3. Each completed model must pass the fixed-input static model gate before MD/NEB.
4. Preserve unstable branches as `FAILURE_MODE` and report `unstable — no Dxy reported`.
