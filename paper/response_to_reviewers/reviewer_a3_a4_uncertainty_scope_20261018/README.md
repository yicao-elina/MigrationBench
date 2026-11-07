# A3/A4 uncertainty and scope evidence package (2026-10-18)

This additive package supports the A3 kinetic-uncertainty response and the A4
scope-of-generalizability response. It keeps three evidence levels separate:

1. The strict grouped three-seed comparison from the A2 retraining package.
   This is the primary A3 evidence. It uses seeds 123/234/345 and reports
   absolute fixed-path barrier errors for the provisional DFT candidate
   references.
2. The broader neb_protocol_matrix_20261018 audit. It contains complete
   finite fixed-path cells and self-relaxed protocol diagnostics, but has
   separate model lineages, force-pathology flags, provisional/high-force DFT
   references, and no DFT rescoring of MLIP-relaxed images. It is therefore
   supporting descriptive evidence, not pooled kinetic uncertainty or a final
   ranking.
3. The Cr-trained broad-roster cross-element diagnostic from the
   Wiley-MigrationBench analysis/task2_cross_element_broad_roster_final_20261006
   package. It evaluates Cu/Li/Na paths in the same host across three training
   seeds. All ten archived DFT path records retain SCF-convergence warnings, so
   the package is used to document reproducibility and diagnostic seed
   stability, not transfer accuracy or cross-element barrier validation.

The canonical source data are not overwritten. Run:

    /Users/alina/Documents/Codex/plot_library/.venv/bin/python \
      paper/response_to_reviewers/reviewer_a3_a4_uncertainty_scope_20261018/plot_a3_a4_uncertainty_scope.py

The script writes derived CSV summaries, LaTeX table inserts, three-format
Plot Atlas figures, and a provenance manifest in this directory. The full
precision values remain in the source CSV files; manuscript-facing values in
the generated tables and captions are rounded to two decimal places.

## Evidence boundary

The grouped path 1--2 low-error result is seed-stable within the three tested
seeds, whereas path 1--3 and path 1--5 are seed-sensitive. This replaces the
unfalsifiable phrase “by chance” with path-resolved evidence. It does not
close the independent QE convergence gate or promote a provisional
higher-distortion path into a final benchmark reference.

The cross-element package demonstrates three reproducible broad-roster
checkpoints (seeds 123/456/789; exact state-dict match across 90 tensors) and
finite MLIP diagnostics, but all ten Cu/Li/Na DFT path records fail the
scientific convergence gate because the archived outputs retain SCF warnings.
The A4 manuscript claim is consequently “candidate framework demonstrated on
one Cr-doped Sb2Te3 system”; broader transfer remains an explicit validation
target.
