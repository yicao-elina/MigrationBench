# DFT-ready workflow

The canonical workflow is intentionally two-stage:

1. Run `select_dft_candidates_from_canonical_screen.py` only after the accepted-model 639 screen has a `VERIFIED` analysis manifest. The deterministic ranking prioritizes precursor/threshold stages, paired force disagreement, force-range tails, and source-run coverage. It writes `DFT_SELECTION_MANIFEST.json` with `qe_status=NOT_SUBMITTED`.
2. Pass that selection to the existing `dft_ready_local_tile_20261013/prepare_dft_batch.py` with `--mode local_periodic_shell --target-atoms 120`. The builder enforces a 2050-atom periodic source, 50–200 atom output, finite values, PBC flags, and minimum distance. It writes `PREPARED_ONLY` local fragments and QE inputs; it does not submit QE.

The full-cell wave campaign remains separate evidence: 2050-atom QE inputs are not silently replaced by a local reconstruction. Any QE result must pass explicit `JOB DONE`, finite energy/force, input-hash, and method/provenance gates before it can support a DFT-validated claim.
