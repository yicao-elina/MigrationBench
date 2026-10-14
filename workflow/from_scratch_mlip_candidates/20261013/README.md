# Three-model static MLIP screen and candidate selection

This workflow records the selection layer used for the 639 prepared-only
2050-atom PBC snapshots from the from-scratch failure-attribution campaign.
It is deliberately separate from the full-cell QE workflow: the MLIP screen
is a static force/energy evaluation, while the DFT-ready subset is a later
candidate-prioritization step.

The compact, hash-pinned evidence is under
`data/raw_compact/revision1/from_scratch_failure_attribution_20261013/`.
The 639 source extxyz files, three model checkpoints, and full Rockfish
result trees remain in the approved external archive and are referenced by
immutable paths and SHA256 values in the manifests.

## Re-run order

The commands below assume a campaign directory containing the external
`results_all_replicates/` tree and the scripts in this directory. They do not
submit work unless the explicit `sbatch` command is run.

```bash
# 1. Build the canonical 639-row inventory and the original wave-01 ledger.
python categorize_dft_candidates.py

# 2. Build the execution manifest from the prepared extxyz files.
python build_manifest.py \
  --source results_all_replicates/dft_candidates_prepared_only/manifest.csv \
  --output candidate_manifest_639.csv

# 3. Run the three registered checkpoints on every candidate in chunks of 20.
#    The Slurm template uses 32 array tasks, at most two concurrent tasks.
sbatch run_mlip_candidate_chunk.slurm

# 4. Collect chunk outputs only after every chunk manifest is present.
sbatch run_collect_mlip.slurm

# 5. Reconcile execution IDs to canonical IDs by immutable prepared_extxyz path.
python reconcile_mlip_results.py \
  --results analysis \
  --canonical-manifest candidate_manifest_639.csv \
  --model-registry models_rockfish.tsv \
  --output results_reconciled

# 6. Generate static MLIP tables/figures and the analysis manifest.
python make_mlip_analysis.py \
  --results results_reconciled \
  --output figures

# 7. Select the MLIP-informed reduced-cell DFT subset without submitting DFT.
python select_dft_batch.py \
  --analysis results_reconciled \
  --candidate-manifest candidate_manifest_639.csv \
  --output dft_batch_20261013 \
  --n 20

# 8. Validate the committed compact evidence capsule from a clean checkout.
python scripts/validate_from_scratch_mlip_candidate_selection.py
```

The production array used three exact checkpoint files and produced 1,917
candidate-model rows. A result is `VERIFIED` only when there are 639 mapped
canonical candidates, 3 model rows per candidate, no duplicate candidate-model
pairs, finite successful outputs, and matching model/input hashes. The
execution-to-canonical mapping is retained because the original production
execution IDs were not the canonical `dftcand_*` IDs.

## Selection semantics

`categorize_dft_candidates.py` preserves all 639 candidates and assigns the
five stage labels. The original `wave_01` ledger is a mechanism-oriented
coverage wave: every threshold/terminal row, the latest precursor from each
run, and one early stable control per training seed.

`select_dft_batch.py` is a separate MLIP-informed DFT prioritization. It first
requires all three static MLIP evaluations and `min_distance_A >= 0.8 Å` on
the source full-cell candidate. It ranks by cross-model force spread, applies
the stage quotas, and fills any unfilled slots deterministically. The
resulting 20 structures are still `PREPARED_ONLY`/`RECONSTRUCTION` until the
reduced-cell QE gates pass.

Do not use this workflow to claim that a large force is a DFT error. It only
identifies configurations where the registered MLIP checkpoints disagree and
therefore where an independent reference calculation would be informative.
