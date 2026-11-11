# MigrationBench figure evidence audit

This directory is the additive, provenance-first audit package for all logical
figure outputs found in the current `Wiley-MigrationBench` checkout, the
isolated ablation worktree at
`/Users/alina/.codex/worktrees/bf6e/Wiley-MigrationBench`, and the canonical
journal-paper checkout at `/Users/alina/Project/migrationbench-paper-repro`.

The unit of audit is a logical figure, not an export format. PNG, SVG, and PDF
files with the same stem are one figure and are listed together. The audit
records:

`raw/source data -> processing/summary table -> plotting script -> figure
manifest/hash -> rendered figure -> manuscript location -> caption -> result
description/discussion -> scientific status and gaps`

`scripts/build_inventory.py` is deterministic and writes the rendered-output
inventory. `scripts/build_candidate_inventory.py` additionally finds
data/table/script-bearing directories that have no rendered output, including
ablation studies and DFT/MD validation tables. The candidate inventory is
deliberately descriptive: it does not promote a placeholder, provisional path
table, unconverged DFT result, or failed MD run to a scientific result.

The generated `figure_candidate_register.md` is the human-readable queue for
table-backed analyses. Seven high-value table-backed packages were rendered
with the canonical Plot Atlas under
`generated/table_backed_packages/`; each contains processed tables, PNG/SVG/PDF,
an evidence manifest with hashes, and caption/results/discussion text. These
packages include the 48-run integrator/thermostat ablation, but remain labeled
as screening/gate diagnostics where the scientific acceptance gate is closed.

The final human-readable audit will group related outputs using hierarchical
IDs such as `MD-1NS.a` (parent benchmark), `MD-1NS.a.1` (temperature panel),
and parallel IDs such as `NEB-CR.61` and `NEB-CR.81` for distinct path families.
