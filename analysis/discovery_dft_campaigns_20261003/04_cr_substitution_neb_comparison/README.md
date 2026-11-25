# Cr substitution NEB comparison

This is an isolated, geometry-matched comparison campaign for the nine clean Cu/Li/Na paths. Every generated input keeps the source cell, atom order, coordinates, image count, and NEB parameters. Only the dopant identity and its pseudopotential are changed to Cr. Original snapshots are never overwritten.

The generated paths are controlled comparisons, not relaxed Cr-specific migration pathways. They must be labelled `same_geometry_species_substitution` until an independent Cr relaxation is performed.

Source snapshots: `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/task3_endpoint_only_neb/remote_snapshot/`
Generated inputs: `inputs/<path-id>/`
Audit manifest: `manifests/cr_substitution_manifest.tsv`

Run locally:

```bash
python3 scripts/generate_cr_paths.py
python3 scripts/validate_cr_paths.py
```

Discovery heavy-compute root (to be created only after smoke approval):
`/azwestus3/pclancy3_lab/ycao73/MigrationBench/discovery_dft_20261003/cr_substitution_neb_comparison`

Submission gate: first run only `2L-Cu__neb-QLs` as a real QE NEB smoke. Require clean `neb.x` termination, finite energies and forces for all images, no SCF convergence warning, and preserved endpoint identity/cell before releasing the other eight paths.
