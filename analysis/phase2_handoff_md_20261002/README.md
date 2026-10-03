# Phase 2: short-range handoff MD

This directory is an isolated, non-overwriting handoff experiment. Exact
source snapshots at steps 460000, 461800, 462000, and 462200 were extracted
from the real Scratch-source continuation; the requested runs use 461800 and
462000. Each requested state was passed to four MACE models with positions,
momenta, cell, PBC, species, timestep, and weak Langevin settings retained.

The experiment is model-stability evidence, not a DFT reference and not an
independent replica ensemble. The remote compute root is
`/scratch16/pclancy3/yi/phase2_handoff_md_20261002`. Raw trajectories remain
there; this release contains exact input snapshots, metrics, hashes, plots,
logs, and reproducible scripts.

Paper-facing model names are canonicalized as `Foundation`, `Scratch`,
`FT-600K`, and `FT-MultiT`. Machine IDs and source/provenance paths remain
unchanged.
