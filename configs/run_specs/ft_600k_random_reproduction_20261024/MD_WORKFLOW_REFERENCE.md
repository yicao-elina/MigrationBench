# Reference workflow for a future FT-600K 1 ns MD run

This is the existing workflow pattern, recorded for a future repaired
checkpoint. It is not evidence that the current retraining passed.

Source files:

- `analysis/mace_foundation_matpes_pbe_omat_ft_1ns_20261013/md_submit_wrapper.slurm`
- `analysis/mace_foundation_matpes_pbe_omat_ft_1ns_20261013/continue_wrapper.slurm`

Protocol to adapt for each newly verified FT-600K checkpoint:

1. Use a new non-overwriting remote root and record the exact model SHA256 and
   the input SHA256.
2. Run the 20 ps smoke gate with the same 1 fs timestep, 600 K target, input
   structure, and MD velocity seed `1100`. The historical wrapper starts from
   absolute MD step `200000` and runs `20000` steps.
3. Continue only when `status.json` has `smoke_pass: true` and
   `reason: target_steps`.
4. Submit the continuation with an `afterok` dependency, write interval 200,
   checkpoint interval 5000, and target absolute step `1000000` (1 ns at
   1 fs). Preserve walltime checkpoint/resubmission behavior without
   overwriting segments.
5. Keep the model's training seed (`234` or `345`) separate from the MD
   velocity seed (`1100`).
6. After completion, audit finite positions/momenta/energy/forces/temperature,
   atom count/species/cell/PBC, geometry, continuity, saved-step coverage, and
   model/input provenance before any transport or figure claim.

For this campaign the model argument must point to the newly compiled
seed-specific checkpoint under the repaired training root, not to the
foundation model used by the reference wrapper. No smoke or continuation job
was submitted in this audit because the retraining gate failed.
