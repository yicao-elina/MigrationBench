# Phase 2 execution log summary

- Initial attempt jobs: 31511868–31511875. They failed before integration because output directories had been pre-created, triggering the driver's no-overwrite guard. Error logs are retained under `remote_artifacts/logs/`.
- Retry1 jobs: 31511881–31511888. Early jobs exposed and retained the pair-absence NaN finite-check bug; later jobs completed but were not used as the canonical analysis because retry2 was rerun after the check was fixed.
- Canonical retry2 jobs: 31511895–31511902. All completed with `0:0`; all eight produced full 2 ps trajectories, `metrics.csv`, `handoff_input.json`, and `completion.json`.
- No old four-model continuation task was cancelled or modified. All new outputs were under `/scratch16/pclancy3/yi/phase2_handoff_md_20261002`.
