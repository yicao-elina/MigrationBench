# Phase 2 handoff audit and Phase 3 boundary

This directory is an additive audit of the canonical retry-2 handoff. Raw
trajectories remain on Rockfish under
`/scratch16/pclancy3/yi/phase2_handoff_md_20261002/runs/retry2`; the local
release retains the exact snapshots, metrics, hashes, tables, scripts, and
rendered figures needed to reproduce the conclusions.

Run the metric audit with:

```bash
python audit_phase2.py \
  --root ../remote_artifacts/runs/retry2 \
  --output results
```

Render with the documented Plot Atlas fallback environment:

```bash
MPLBACKEND=Agg \
PYTHONPATH=/Users/alina/Documents/Codex/plot_library/src \
/Users/alina/anaconda3/bin/python render_phase2_atlas.py \
  --audit results --output figures
```

The canonical venv was tested with the same source path and imported Plot
Atlas, but hung during the macOS PDF/font rendering stage. The successful
renderer is therefore explicitly recorded as an Anaconda Python fallback in
`figures/plot_env.json`; it uses the Plot Atlas source tree and its
publication style/`timecourse` primitive.

The 24-run integrator/thermostat sensitivity campaign is independent and
non-overwriting:
`/scratch16/pclancy3/yi/phase2_integrator_sensitivity_20261003` (Slurm job
31583430). It must be scientifically audited after all outputs are terminal;
submission or `COMPLETED` alone is not evidence of the decision gate.
