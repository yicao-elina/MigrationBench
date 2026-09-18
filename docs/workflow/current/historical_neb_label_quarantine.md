# Historical NEB Label Quarantine

## Decision

The nine extxyz files reachable through
`3.fine-tuning/2-layer/data/neb_extxyz_outputs` are quarantined as DFT-label
sources. Their coordinates remain useful for provenance and candidate-path
reconstruction, but their `energy` and `forces` fields must not be used for
training, validation, benchmark scores, barriers, or manuscript values.

The exact file hashes and frame counts are recorded in
`configs/historical_neb_extxyz_quarantine.json`.

## Verified Failure Mode

The historical extractor declares `.pathN` energy as Ry and multiplies it by
`13.605693122994`. Direct comparison of the same iteration against QE
`neb.out` and per-image `PW.out` shows that `.pathN` stores Hartree, requiring
`27.211386245988 eV/Hartree`.

The three vector columns after each position are potential-energy gradients in
Hartree/bohr, not atomic forces in Ry/bohr. The canonical relationship is

`F_DFT = -gradient_path * 27.211386245988 / 0.529177210903`.

For the current 1-6 and 1-7 histories, the maximum numerical residual of
`gradient_path + F_DFT` is `1.29e-7 eV/A`, and the maximum energy disagreement
between `.pathN` and `PW.out` is `6.81e-8 eV`. These checks establish both the
unit and sign semantics.

The historical files also contain 67 iteration-zero frames whose zero energy
and zero gradient are QE restart placeholders rather than DFT labels.

## Blast Radius

- 9 files and 2306 frames are affected.
- Any result that consumed only coordinates is not invalidated by this issue.
- Any energy/force RMSE, force attribution, or barrier result that consumed
  these extxyz labels must be regenerated before use.
- This is separate from training leakage. The affected files were generated in
  June 2026, while the audited fine-tuned checkpoint was trained in August
  2025; the running structure-level audit provides the definitive checkpoint
  comparison.

## Replacement

Use `parse_qe_neb_full_history.py`. It joins `.pathN` coordinates with
independent `PW.out` energies and true atomic forces, carries the scalar NEB
residual from `neb.out`, labels path gradients separately, preserves hashes,
and treats `.path0` as coordinates-only initialization.

No historical file is deleted or rewritten. Dataset builders must reject its
hashes and consume only the canonical, validated staging tables.
