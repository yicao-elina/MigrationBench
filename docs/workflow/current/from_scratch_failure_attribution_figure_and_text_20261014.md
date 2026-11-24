# From-scratch failure-attribution figure and manuscript integration

This workflow records how the 639-candidate MLIP screen is converted into the
Nature-style supplementary figure and the bounded main-text discussion.

## Inputs

The figure uses only the compact, provenance-linked tables under
`data/raw_compact/revision1/from_scratch_failure_attribution_20261013/`:

- `candidate_categorization.csv`: the canonical 639-row inventory and five
  source-stage labels;
- `run_summary.csv` and `initial_state_audit.csv`: the nine-run onset and
  initial-state repeatability audit used by the main-text paragraph;
- `diagnostic_manifest.json`: the operational gates, descriptor fit and
  model/MD-seed provenance for the failure audit;
- `mlip_results/mlip_candidate_results_all.csv`: the 1917 candidate--model
  static evaluations;
- `mlip_results/candidate_model_spread.csv`: paired cross-model force-spread
  metrics for the 639 candidates; and
- `mlip_results/model_stage_summary.csv`: stage-by-model summaries.

The registered model hashes and the deterministic 20-row DFT selection are
documented in the same evidence capsule. Training seeds and MD velocity seeds
are kept as separate provenance fields.

## Rendering

From the repository root, run:

```text
python paper/revision1/scripts/render_from_scratch_failure_attribution.py
```

The script writes `figS_from_scratch_failure_attribution.pdf`, `.png`, `.svg`
and a hash-bearing `.manifest.json` to `paper/revision1/figures/`. Panel (a)
is an explicitly conceptual selection schematic; panels (b--d) are generated
from the compact quantitative tables. The figure is included in
`sn-article-SI.tex` and is cross-referenced from the main-text section
“Reproducibility of the from-scratch failure”.

## Interpretation gate

The composite supports only the following bounded statements: onset timing is
protocol-sensitive; the source-stage force distributions and paired
cross-checkpoint disagreement increase near the recorded failure stage; and
the three checkpoints do not respond identically to the same saved geometry.
The force-spread score is not a calibrated uncertainty, and the static screen
does not determine which force is DFT-correct or prove that the force caused
the MD runaway. The selected reference rows remain `prepared_only` until
identity-matched, converged reference calculations are available.

## Traceability

The figure manifest records the SHA256 of every quantitative input and every
rendered output. The manuscript source ledger contains claim-level links for
the composite, the 639-row inventory, the 1917 finite evaluations, the
stage-wise force summary and the prepared-only DFT boundary.
