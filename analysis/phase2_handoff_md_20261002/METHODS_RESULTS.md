# Phase 2 Methods and Results: short-range handoff MD

## Methods

The handoff source was the real Rockfish continuation file
`/scratch16/pclancy3/yi/mace_four_model_MD_continuation_20260930/1-from-scratch/segment_from_000200000.extxyz` (2050 atoms, fully periodic, SHA256 recorded in `phase2_manifest.json`). Exact extxyz blocks were extracted by the absolute `step` metadata, not by filename inference. The requested handoff states were steps 461800 and 462000; steps 460000 and 462200 were retained as neighboring provenance context. Every source frame contained positions, `momenta:R:3`, forces, a common cell, and `pbc="T T T"`.

Each requested snapshot was passed unchanged to Foundation, Scratch, FT-600K, and FT-MultiT. The driver did not resample velocities or alter the cell/PBC. It used a 1 fs timestep, 600 K Langevin target, friction 0.01 fs^-1, CUDA MACE evaluation, 2000 integration steps (2 ps), and saved every 10 steps. The initial positions, momenta, cell, species, and model/input hashes were recorded per run. The same-snapshot initial position and momentum hashes are identical across all four models.

At every saved frame we recorded potential/kinetic/total energy, kinetic temperature, maximum force, maximum velocity and momentum, minimum pair distance, PBC-corrected displacement, cumulative unwrapped displacement, count of atoms exceeding 5 Å unwrapped displacement, Cr–Sb/Te/Cr pair distances and coordination, and an all-pair RDF histogram distance from the initial state. A run was labeled `runaway_or_nonfinite` if a finite/core-geometry gate failed (catastrophic force >100 eV Å^-1, velocity >1000 Å ps^-1, energy span >250 eV, temperature >2000 K, minimum distance <1.5 Å, >50 Å maximum unwrapped displacement, or >100 atoms beyond 5 Å). Runs passing those catastrophic gates but showing >5 Å unwrapped displacement were labeled `bounded_with_structural_shift`; otherwise they were labeled `stable_short_handoff`.

## Results

All 8 requested runs completed with Slurm exit code 0 and 202 metric rows (initial frame plus 200 saved integration frames). The scientific outcome was 2 stable short handoffs, 2 bounded structural shifts, and 4 runaway outcomes:

| source step | Foundation | Scratch | FT-600K | FT-MultiT |
|---:|---|---|---|---|
| 461800 | runaway | runaway | stable | stable |
| 462000 | runaway | runaway | bounded shift | bounded shift |

Foundation diverged from both identical initial states, with multi-billion-to-10^11 K temperature excursions, 10^6–10^7 eV Å^-1 forces, and thousands of Å unwrapped displacement. Scratch also diverged from both states; the 461800 branch was catastrophic immediately over the short run, while the 462000 branch reached a lower but still unacceptable runaway regime (maximum temperature about 2334 K, force about 1351 eV Å^-1, and 80 atoms beyond 5 Å). These are new model-induced instabilities during handoff, not evidence that the source input itself was non-finite.

FT-600K and FT-MultiT were bounded at step 461800. From step 462000, both remained finite and avoided catastrophic force/energy growth, but each showed a short-range structural shift: maximum unwrapped displacement about 7.4–7.6 Å and 4–9 atoms beyond 5 Å. This is a snapshot-sensitive model response, not a failure to transfer the state. The distinction is important: the 462000 state is near a pre-existing source-trajectory instability window, while the four-model comparison shows that the resulting runaway is strongly model-dependent.

The figures use the registered Plot Atlas `nature_steered_dynamics` palette, `timecourse` primitive, Atlas font/tick styling, and the canonical model names. Legends use a dedicated header region and never occupy the trajectory axes; the heatmap uses the same styling and labels. Because the canonical Plot Atlas environment currently hangs during import, the script carries the same API-compatible definitions for rendering and records `self_contained_style_compatible_fallback` in the SVG provenance text.

This experiment does not establish DFT accuracy, a diffusivity, or an independent replica uncertainty. It establishes only short-range transfer integrity and model-dependent stability. Phase 3 should use the retained exact snapshots for longer, separately gated follow-up of FT-600K/FT-MultiT, isolate the first divergence step for Foundation/Scratch, and add DFT single-point/force checks before any scientific claim about the long-time trajectory.
