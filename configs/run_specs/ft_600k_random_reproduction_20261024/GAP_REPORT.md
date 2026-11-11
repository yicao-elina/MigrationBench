# Gap report

| item | status | consequence |
|---|---|---|
| Rockfish terminal state for `32121744` | verified | both array elements failed; no active job remains |
| Failure cause | verified | missing `conda.sh` before `conda activate`/training |
| Seed234 compiled checkpoint | missing | no model identity, hash, or test RMSE |
| Seed345 compiled checkpoint | missing | no model identity, hash, or test RMSE |
| Seed123 anchor exact checkpoint hash | partial | historical anchor is named in the campaign manifest, but not resolved in this audit |
| 1 ns MD for the repaired models | blocked | conditional user request is not triggered |
| Correct 1 ns workflow reference | verified as a workflow recipe | existing wrapper files provide the smoke/continuation protocol; scientific acceptance still requires MD gates |

The next meaningful state change is a repaired, separately recorded training
submission and its final checkpoint audit. A scheduler `COMPLETED` state alone
will not be sufficient; the model must load and pass finite Cr/Sb/Te probes,
and its data/protocol/seed provenance must match the campaign manifest.
