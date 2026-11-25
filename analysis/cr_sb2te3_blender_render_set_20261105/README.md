# Cr migration in Sb₂Te₃ — Blender render asset set

This directory contains the small Git-tracked manifests and code for the render set. Eleven large source payloads (nine Blender scenes and two PNGs) were recovered from Git commit `15e1250112d1dda115fa2c89505b4c11007513c1` and are stored in the canonical Rockfish archive `rockfish:/scratch16/pclancy3/yi/MigrationBench_data_archive/figure1_cr_sb2te3_blender_render_set_20261105/`. Their combined logical size is 300,400,629 bytes. `RENDER_SET_ARCHIVE_MANIFEST.tsv` maps every source path to its archive path, source Git blob, size and SHA256; Rockfish copies were independently SHA256-verified on 2026-11-25.

## Contents

- `renders/` — metadata refers to eight primary PNG outputs; seven legacy payloads are in the Rockfish archive or in the current release tree, and one preview payload remains explicitly located by the archive manifest.
- `blender/` — eight `.blend` scenes are in the Rockfish archive; the checksums and Git blob IDs are in `RENDER_SET_ARCHIVE_MANIFEST.tsv`.
- `provenance/` — copied source manifests, the original sliding scene used for the no-Cr derivative, and the no-Cr preview QA note.
- `scripts/remove_cr_render_preview.py` — the small, auditable transformation used for the no-Cr scene: hide only `Cr_image_00` from render and viewport while preserving all host atoms and lights.
- `references/six_panel_contact_sheet.png` — the user-provided six-panel mapping reference (Static, MD, In-gap, Gap-to-QL, Within-QL, Sliding).
- `RENDER_SET_MANIFEST.json` — panel-level mapping, source paths, dimensions, scientific-status notes, and SHA256 values.
- `SHA256SUMS.txt` — hashes for files materialized in this repository directory. External large payloads are hashed in `RENDER_SET_ARCHIVE_MANIFEST.tsv` and the Rockfish archive `SHA256SUMS.txt`.

## Panel mapping

1. Panorama — `01_panorama_cr_migration_sb2te3_4500x3000_rgb.png`
2. Static — `02_static_L0_perfect_crystal_3600.png`
3. MD — `03_md_L1_thermal_vibration_1800.png`
4. In-gap — `04_in_gap_ideal2_3600.png`
5. Gap-to-QL — `05_gap_to_ql_ideal1_3600.png`
6. Within-QL — `06_within_ql_neb3_3600.png`
7. Sliding with Cr — `07_sliding_with_cr_1800.png`
8. Pristine sliding without Cr — `08_sliding_pristine_without_cr_1800_preview.png`

The first seven outputs are copied from the existing approved/source-bounded render packages. The eighth is a derived 1800×1800 preview. Its source `.blend` is the original sliding scene with only the `Cr_image_00` render mesh hidden; the four scene lights are preserved. It is intentionally marked `PREVIEW_ONLY` because the corresponding 3600px export has not been approved or rendered yet.

The static, MD, in-gap, gap-to-QL, within-QL, and sliding scenes are visual schematics. The MD and sliding variants are not being promoted here to AIMD, converged-barrier, or transition-state evidence.

