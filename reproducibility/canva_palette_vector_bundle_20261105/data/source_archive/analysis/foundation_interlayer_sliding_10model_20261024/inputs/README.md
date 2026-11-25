# Input recovery boundary

The remote execution root copies the validated R13 structures from
`/scratch16/pclancy3/yi/foundation_interlayer_sliding_2x2_20261024/inputs`.
The canonical b-path2 hashes are recorded in `PATH_REGISTRY.tsv`. The local
working tree may keep only pointers because the original CloudFiles-backed
copies are not reliable local byte stores; remote manifests are the execution
source of truth and are pulled back after completion.
