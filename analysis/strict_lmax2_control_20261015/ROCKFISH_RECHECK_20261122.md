# Rockfish recheck — strict Lmax2 control

**Checked:** 2026-11-22 (America/Los_Angeles)
**Remote root:** `/scratch16/pclancy3/yi/strict_lmax2_control_20261015`

The original `CAMPAIGN_STATUS.md` is retained as the historical submission
record; its initial `PENDING` observation is superseded by this recheck.

## Scheduler result

A login-shell `sacct` query reports array tasks `32055099_0` and `32055099_1`
as `FAILED`, exit code `1:0`, with start/end times on 2026-10-16. Both batch
steps failed; their `extern` steps completed normally. The two `.out` logs are
empty. The `.err` logs both contain:

```text
/data/apps/go.sh: line 69: RF_CONDA_SITE: unbound variable
```

The job-level accounting and exact stderr bytes are preserved beside this note; `SHA256SUMS` records their local hashes. The current gate-by-gate disposition is in `ACCEPTANCE_MATRIX_CURRENT_20261122.tsv`.
An earlier audit mentioned a second submission ID `32069521`; a fresh query by
job ID and by job name did not return that record, and no corresponding output
or checkpoint was found. Treat that attempt as unverified, not as a successful
run.

## Scientific disposition

Neither seed reached training. No new model, compiled checkpoint, 639-point
evaluation, strict seed-dispersion estimate, or strict-trio figure exists. The
strict Lmax2 experiment remains **NOT RUN TO SCIENTIFIC OUTPUT**. Do not pool
its seeds or claim a completed control. Existing historical seed123 remains
heterogeneous and excluded from strict seed-to-seed summaries.

The failure appears in the shared Rockfish launcher before the training
command; the launch script also names a separate output root. Fixing either
condition requires a new, separately recorded submission with frozen inputs
and all acceptance gates retained.
