# Stage-wise force divergence

## Static response

All 7,668 paired evaluations are finite and successful. Across stable early and stable pre-onset frames, median maximum forces remain approximately 2.4–3.2 eV Å−1. In the precursor group, the median maximum force rises to approximately 3.2 eV Å−1 for the baseline, 8.5 eV Å−1 across the two scratch lmax-2 checkpoints, 16.2–16.4 eV Å−1 for the fine-tuned families, 17.0 eV Å−1 for legacy OMAT and 29.9–30.8 eV Å−1 for the two foundation medium models.

The 12-model maximum-force range has these stage-wise medians:

| stage | median force range (eV Å−1) |
|---|---:|
| early stable | 3.07 |
| pre-onset | 7.06 |
| precursor | 27.29 |
| threshold/terminal | 3,717.51 |

The threshold range is based on only eight threshold/terminal structures and is therefore a diagnostic extreme, not a population estimate.

## Interpretation boundary

The pattern establishes model divergence on the same structures. It does not establish which model is correct, because no reference DFT force has yet been obtained for these 2050-atom configurations. The result motivates reduced PBC-preserving fragments for targeted DFT validation.
