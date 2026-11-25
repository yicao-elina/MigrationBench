# QE mix external-reference check — 2026-11-23

## Result

A bounded text-file scan of
`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/1.pwrelax/3-2.2-layer_CrSb2Te3/2.Cr-in-gap`
found no explicit references to the three run directories or the exact
`sb2te3.mix1`–`sb2te3.mix4` basenames. It examined 184 readable UTF-8 text
files, with a per-file limit of 5 MiB. It skipped 9 directories (`.conda`,
`.git`, QE `out`/`out_*`/`.save` trees), 125 binary/non-text or non-allowlisted
files, and 8 files above the size limit. There were zero read errors.

A wider recursive scan of `/data/pclancy3/yi` did not complete: remote directory
traversal stalled in a filesystem D-state and was stopped after about 6 minutes
48 seconds. The remote login did not have `rg`, and the stopped scan produced no
complete count. No claim is made that other locations under `/yi` contain no
references.

Before the new lineage import, the catalog contained 985 artifacts whose paths
were under the `2.Cr-in-gap` subtree. The 12 mixing candidates had no direct
`run_artifact`, `recipe_input`, `reproduction_recipe`, or `slurm_job` links, and
workflow/recipe path strings and the three job IDs did not match. The broad
claim/derivation query exceeded its time budget and was stopped; those relation
results remain unknown.

## Cleanup interpretation

The narrow scan does not rule out references in excluded binaries, generated QE
save trees, oversized files, or other `/data/pclancy3/yi` subtrees. The QE mix
files are nevertheless internal restart state in the three corresponding
`out/` directories; they are not named in the scanned scripts/manifests in the
run subtree. The user authorized removal after cataloging and documented a
fresh-start rerun. The removal will discard the exact continuation point for
these time-limited partial calculations; it will not be represented as
bitwise reproduction or as a converged scientific result.
