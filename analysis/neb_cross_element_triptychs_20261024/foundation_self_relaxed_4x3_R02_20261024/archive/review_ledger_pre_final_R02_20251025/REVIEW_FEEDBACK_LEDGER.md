# Independent review feedback ledger

This ledger records the independent review loop. The execution agent does not self-certify release.

| Review | Finding | Repair / disposition | Status |
|---|---|---|---|
| data_review_R01 | R01 pre-canonical endpoint headers were inherited from the old campaign; local/remote launch hashes diverged; smoke/formal attempts shared a row; no pending-aware collection ledger existed. | Stopped the pre-canonical array, preserved its raw attempts as diagnostic, created R02 canonical endpoint layer, bound one immutable formal-launch manifest, made collection pending-aware, and added attempt bookkeeping. | Resolved in R02 design; final evidence still required |
| data_review_R02_preexecution | R02 canonical endpoint layer and launch hash passed; release remained blocked by running array, partial collection, and smoke/formal role evidence. | Added remote bookkeeping copy with base/revision mapping; collector now classifies the smoke job from bookkeeping and retains all attempts; final collection is deferred until scheduler terminal state. | Open until final collection |
| data_review_R02_final | To be generated after all 108 target rows and final derived package exist. | Independent row-level audit and feedback loop required before release. | Pending |
| visual_review_R01 | To be generated after the final 4×3 composite and 12 standalone panels exist. | Independent full-resolution and intended-size visual inspection required before release. | Pending |

The frozen visual/data contract is [ACCEPTANCE_CRITERIA.md](ACCEPTANCE_CRITERIA.md). The R01 and R02 pre-execution reports are retained in `../foundation_self_relaxed_4x3_20261024/data_review/` and `data_review/` respectively; they are evidence of the repair loop, not release approval.

