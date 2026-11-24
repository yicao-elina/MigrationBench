# R05 source snapshot boundary

This directory contains the materialized campaign manifests and endpoint
records that were available locally during the archive build. Some source
tables and large remote artifacts are File Provider placeholders or remain on
Rockfish; they are not committed as empty stand-ins. Their original paths,
SHA256 values, and materialization status are recorded in
`../provenance/POINTER_ONLY_INPUTS.tsv` and the R05 manifest.

The figure is still traceable to the original analysis-root R05 commit
`ea3a30247ed60414f0b5ac056afeeb722cf63925`, while the external trajectory and
iteration extxyz files remain intentionally pointer-only.
