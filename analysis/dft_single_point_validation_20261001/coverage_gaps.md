# Independent DFT single-point validation: coverage and blockers

The independent campaign contains 80 manifest rows: 65 MACE endpoint/NEB candidates, 10 MPA0 failure-trajectory frames (five last-physical inputs and five first-anomaly evidence frames), and five high-energy multi-temperature MD snapshots.

The initial DFT wave contains 20 inputs:

- 5 stable endpoints: `unique_044`, `unique_096`, `unique_067`, `unique_109`, `unique_097`;
- 5 NEB intermediates: one each from `cov_002`, `cov_020`, `cov_027`, `cov_031`, `cov_035`;
- 5 near-saddle structures from the same five path families;
- 5 last-physical frames from the MPA0 `unique_065`/`unique_080` failure variants.

The five MD snapshots are extracted and SHA256-identified from the real remote trajectory `3-multi_T-fine-tuning.segment.extxyz`, frames 150, 490, 149, 124, and 55. They remain `pending_md_provenance`: the available snapshot directory does not contain the original MD run manifest/checkpoint chain, and the structures contain 2050 atoms. No MD snapshot is counted as a scientifically accepted DFT result until that provenance and a resource smoke gate are complete.

The requested 15 Scratch seed234 runaway tasks were not found in the authoritative inventories searched so far. The located `scratch_grouped_seed234` directory contains training logs/checkpoints but no endpoint runaway trajectories or task-level failure exports. They are recorded as an explicit unresolved coverage gap; no structure or DFT input was fabricated for them.

The 47 existing non-magnetic endpoint relaxations remain a separate campaign. Their results may be combined only after the same energy/force/identity/method quality gate is applied to the actual QE outputs.
