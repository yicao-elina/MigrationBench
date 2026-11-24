# Three-model MLIP evaluation of the 639 failure-attribution candidates

This is an additive, single-point campaign. The 639 saved configurations are
not modified or relaxed. Every input is evaluated with all three exact
checkpoint files registered in `models_rockfish.tsv`:

- `ablation_lmax_2`, training seed 123;
- `ablation_lmax_2`, training seed 234;
- `baseline`, training seed 345.

The canonical publication-facing registry is `MODEL_REGISTRY.tsv`; the
Rockfish launch TSV is an execution copy with the same checkpoint hashes.

The unit of comparison is the same candidate structure across models. Total
energies are retained for provenance but are not ranked across model families.
The primary comparison is force magnitude, force-vector disagreement, force
tails, geometry, and stage/category distributions. Candidate stages remain
`stable_early`, `stable_pre`, `precursor`, and `threshold`; these labels describe
the source trajectory and are not redefined by a static MLIP evaluation.

Each production array task evaluates at most 20 candidates and writes an
independent CSV plus compressed force arrays. The campaign is valid only when
all 32 chunk manifests are complete, each has 60 model-candidate rows except
the final 57-row chunk, all model hashes match this registry, and the candidate
input hash/path is preserved. Runtime failures remain in the failure ledger and
are never silently dropped.

## Rockfish execution record

- L40S smoke: Job `31923402`, completed successfully; one candidate, three model evaluations.
- Production array: Job `31923441_[0-31%2]`, L40S, two concurrent tasks.
- Production collector: the first queued Job `31923442` was canceled after the array completed; the corrected manifest-aware collector completed as Job `31924087` with `VERIFIED` coverage after source-path reconciliation. The original execution IDs and canonical `dftcand_*` IDs are retained in `results/candidate_execution_mapping.csv`.
- The earlier a100 smoke `31923045` remains queued as an independent resource cross-check; it is not a prerequisite for the production results.

## DFT-ready follow-up

The MLIP-driven selector chose 20 candidates after an all-three-model-success
gate and a source minimum-distance gate of 0.8 Angstrom. Eleven more extreme
collapsed source frames were retained but excluded from the DFT-ready batch.
The selected inputs are PBC-preserving 120-atom local-periodic-shell
reconstructions with status `PREPARED_ONLY`/`RECONSTRUCTION`; they are not
full-cell parity results. QE input smoke, pseudopotential presence, and
Rockfish `pw.x` path checks passed. The first reduced-cell DFT array is Job
`31924176_[0-19%2]` under
`/scratch16/pclancy3/yi/dft_mlip_selected_20261013`, submitted with at most two
concurrent tasks. A finite QE output and `JOB DONE` remain required before any
DFT mechanism claim.
