# R05 data tables

`cr_three_path_coverage_audit_R05.tsv` is materialized in this bundle. The
compact plotted-profile table and the two campaign processing tables are
retained as pointer-only inputs in the current macOS File Provider checkout;
their original paths and SHA256 values are recorded in
`../provenance/POINTER_ONLY_INPUTS.tsv`. The tracked copies remain available
through the original analysis-root R05 commit
`ea3a30247ed60414f0b5ac056afeeb722cf63925`.

This status is deliberate: a dataless placeholder is not treated as a local
reproducibility input. The registry records `file_exists=0` for these three
rows until they are materialized and hash-checked.
