# Phase 2: short-range handoff MD

This directory is an isolated, non-overwriting handoff experiment. The source
snapshots include exact frames at absolute steps 460000, 461800, 462000, and
462200 from the Scratch-source continuation trajectory. The two requested
handoff branches are 461800 and 462000; the surrounding frames are retained
as provenance/anomaly context. Each requested frame is handed to the same four
MACE model families with positions, momenta, cell, PBC, species, timestep, and
weak Langevin settings retained. The experiment is a model-stability test,
not a DFT reference and not an independent replica ensemble.

The remote compute root is `/scratch16/pclancy3/yi/phase2_handoff_md_20261002`.
The analysis/plots/manifests are copied into the paper-repro repository under
the matching `analysis/phase2_handoff_md_20261002/` path.

Paper-facing model names are canonicalized as `Foundation`, `Scratch`,
`FT-600K`, and `FT-MultiT`; machine IDs and provenance paths remain unchanged.
