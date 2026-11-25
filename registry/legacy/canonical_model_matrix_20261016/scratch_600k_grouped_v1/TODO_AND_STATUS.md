# Canonical scratch-600K grouped campaign

Status: `READY_FOR_SUBMISSION`; no model is accepted.

The grouped 600 K release is audited separately from the deduplicated Multi-T
release. The only intended within-family variation is training seed 123/234/345.
The upstream dataset manifest remains marked provisional; its exact hashes and
trajectory split are frozen in `EXPERIMENT_MANIFEST.json` and must be carried
into the acceptance record.

Before submission: run shell syntax/preflight, deploy and hash-check the script,
then record the Slurm job ID. After submission: inspect every task, preserve
failures, and require compiled/loadable finite Cr/Sb/Te models, test metrics,
and zero trajectory overlap before any 639 evaluation.
