# Victor DFT-reference multi-arm MACE endpoint benchmark

This directory is an isolated campaign boundary. It uses the nine processed
Victor endpoints in `data/collaborator_dft_neb/` and the registered model
inventory in `analysis/mace_candidate_eval_20261001/models_rockfish.tsv`.

The source-of-truth relaxation protocol is the historical
`run_endpoint_relax.py` runner and the historical perturbations:

- `base`: `(0, 0, 0)` Å
- `perturb_a_dx020`: `(0.20, 0, 0)` Å
- `perturb_b_tilt020`: `(-0.10, 0.14, 0.10)` Å, norm `0.198997` Å

The first-stage matrix is 9 endpoints × 4 source/variant arms × 15 registered
models = 540 runs. Runs use fixed cells, CPU FIRE, 500 maximum steps, and
`fmax=0.03 eV/Å`. Every run must retain its summary, trajectory, final
structure, model/input hashes, and failure evidence. A summary is scientifically
successful only when the optimizer converges, final forces are finite and pass
the threshold, and structure identity/cell/PBC invariants remain unchanged.

The remote scratch boundary is:

`/scratch16/pclancy3/yi/victor_dft_reference_mace_20261001/`

Do not reuse existing campaign result roots. The Slurm submission is
`run_array.slurm`; scheduler access must be verified before claiming runs have
started.
