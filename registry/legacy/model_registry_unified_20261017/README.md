# Unified MLIP model registry

This is an additive registry for the MLIP models used by the 639-candidate
screen, the failure-attribution work, the canonical model matrix, and the
historical comparator audits.

The registry is artifact-level and provenance-preserving. Rows are merged when
either the normalized physical model path or the non-empty SHA256 is identical.
This connected-component rule fixes both the previous hash-first/path-only
alias bug and duplicate byte-identical copies at different paths. All source registries
and source row numbers are retained in `source_registries` and `source_rows`.
The original registries and manifests are not modified.

The repaired snapshot contains 46 unique model artifacts. The previous
59-row snapshot contained both path aliases and byte-identical copies at
different paths. See `PATH_COLLISION_AUDIT.tsv` and `HASH_COLLISION_AUDIT.tsv`
for the complete collision audits, and `DUPLICATE_ARTIFACT_MAP.tsv` for the
retained source aliases.

`SOURCE_REGISTRY_AUDIT.tsv` and `SOURCE_MANIFEST_AUDIT.tsv` provide the source
hashes used to construct this snapshot. The task-3 NEB campaign manifests are
left outside this MLIP registry because they describe NEB jobs rather than
model artifacts; they remain in their original audit roots.

The historical grouped scratch controls already exist as the `scratch_grouped`
seed 123/234/345 rows with `max_L=1`, `correlation=3`. Therefore array tasks
0--2 of Slurm job 32121806 were exact duplicate training and were cancelled.
The Multi-T grouped Lmax=1 controls remain as tasks 3--5.
