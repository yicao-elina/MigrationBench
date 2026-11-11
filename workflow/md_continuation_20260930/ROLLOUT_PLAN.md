# Grouped-retrain MACE-MD diffusivity rollout

Date: 2026-09-30

## Feasibility decision

The rollout is feasible on Rockfish, with these boundaries:

- Five exact compiled models are available for the first pass: OMAT-FT grouped seeds 123/234/345, Scratch grouped seed123, and Scratch-5% grouped seed123.
- The common 2050-atom, fully periodic, momenta-bearing 600 K trajectory is usable as a continuation state. It is not a fresh Maxwell-Boltzmann initialization, so these first runs are continuation/sensitivity tests, not independent replicas.
- The existing continuation driver can preserve momenta and write checkpoints, but its current interface is fixed at friction 0.01 and is not a fresh-start driver. Do not claim replica independence or friction robustness until the driver is extended and separately validated.
- New outputs must remain under `/scratch16`; `/data` is not an output target because prior quota state is unsafe.
- A 20 ps temperature scan is feasible as a screening gate. It cannot establish Arrhenius behavior or a diffusivity convergence result.

## Acceptance gates

1. Model/input paths exist, are readable, and are registered with SHA256 when the remote checksum completes.
2. Each smoke job has finite energy/forces, stable temperature, the expected atom count/PBC, a complete checkpoint, and a non-overwriting output directory.
3. Only after smoke passes: run Tier 1 OMAT-FT grouped (3 training seeds x 3 independent velocity replicas), Tier 2 Scratch grouped seed123 (3 replicas), and Tier 3 Scratch-5% seed123 (1 screening trajectory).
4. Diffusivity analysis must unwrap periodic coordinates, fit the slab axes explicitly, report block/replica uncertainty, and count dopant hopping events. A single trajectory or a short temperature scan is never final evidence.

## TODO and status

- [x] Confirm exact five compiled model paths on Rockfish.
- [x] Confirm common input is 2050 atoms, PBC, and has momenta.
- [x] Restore explicit Rockfish Slurm command/config paths and inspect current queue.
- [x] Add a parameterized, non-overwriting 20 ps temperature smoke job.
- [~] Submit and audit the 500/600/700/800 K smoke jobs (jobs 31465647-31465650 running; first attempt 31465539-31465542 rejected because start_step was misregistered).
- [ ] Record model/input SHA256 and software/runtime versions.
- [ ] Validate smoke outputs and decide whether the temperature trend justifies long MD.
- [ ] Add/validate true fresh-start replica initialization with explicit MD seeds.
- [ ] Run Tier 1 200 ps x 3 replicas plus the 500 ps convergence branch.
- [ ] Run Tier 2 200 ps x 3 replicas.
- [ ] Run Tier 3 200 ps x 1 screening trajectory.
- [ ] Add friction sensitivity and DFT single-point validation of representative MD/saddle-region structures.
- [ ] Run rigorous post-processing: unwrap, 2D MSD, block/autocorrelation uncertainty, hop count, and convergence report.
- [ ] Update manuscript/reviewer evidence only from accepted artifacts.

## Current interpretation

The plan is computationally feasible but scientifically conditional. The 20 ps scan is a go/no-go diagnostic. It does not make 200 ps single trajectories publishable, and it does not justify room-temperature extrapolation. If hopping is absent at all scan temperatures, prioritize NEB/TST/kMC over blindly extending MD.
