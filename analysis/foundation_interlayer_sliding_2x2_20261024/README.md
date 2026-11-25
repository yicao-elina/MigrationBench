# Foundation interlayer-sliding 2×2 campaign

This is the frozen, Foundation-only campaign for the recovered QE b-path2 interlayer-sliding data.

- Rows: fixed-path single point; self-relaxed endpoint-only MLIP NEB.
- Columns: Bulk; Cr-in-gap.
- Models: current MACE-MP-0b3-medium, MACE-OMAT-0-medium, MACE-MPA-0-medium.
- Primary scope: 12 rows × 10 images.
- Remote execution root: `/scratch16/pclancy3/yi/foundation_interlayer_sliding_2x2_20261024`.
- Production arrays: fixed `32461048`; self-relaxed `32461049`; all 12 rows completed and passed the frozen scientific gates.
- Release: `manifests/RELEASE_MANIFEST_R13.json`; pre-run and post-run independent data/visual review gates are PASS.
- Figure: `results/figures/foundation_interlayer_sliding_2x2_R13/` in PDF, SVG, and PNG.

Read `ACCEPTANCE_FREEZE_R13.md` before submitting or interpreting any result. R01–R12 are retained as rejected pre-run drafts. The local `inputs/` directory is an additive copy of the recovered b-path2 evidence; it does not replace or modify the raw recovery package.


## Archived raw/output payloads from the superseded release branch

The current PR tree omits some files that remain part of the source Git history. Their exact source commit/blob, SHA256, sizes and destinations are mapped in `analysis/branch_reconciliation/PR20_SUPERSEDED_RELEASE_BRANCH_FILE_MAP_20261125.tsv`. The two recovered DFT b-path2 `neb.out` files and six MLIP attempt logs are in the Rockfish content-addressed archive at `MigrationBench_data_archive/revision1_branch_retirement_20261125/`; Slurm stdout/stderr sidecars are retained there as operational evidence. The manifest registers every old source path in the Revision 1 SQLite index. Their availability does not change the scientific acceptance/status claims above.
