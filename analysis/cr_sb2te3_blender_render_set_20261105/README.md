# Cr migration in Sb₂Te₃ — Blender render asset set

This directory is the local, provenance-bounded archive of the eight requested render panels and their Blender scenes.

## Contents

- `renders/` — eight primary PNG outputs: one full-system panorama, six previously approved render states, and the current pristine sliding preview without Cr.
- `blender/` — one corresponding `.blend` scene for each primary render.
- `provenance/` — copied source manifests, the original sliding scene used for the no-Cr derivative, and the no-Cr preview QA note.
- `scripts/remove_cr_render_preview.py` — the small, auditable transformation used for the no-Cr scene: hide only `Cr_image_00` from render and viewport while preserving all host atoms and lights.
- `references/six_panel_contact_sheet.png` — the user-provided six-panel mapping reference (Static, MD, In-gap, Gap-to-QL, Within-QL, Sliding).
- `RENDER_SET_MANIFEST.json` — panel-level mapping, source paths, dimensions, scientific-status notes, and SHA256 values.
- `SHA256SUMS.txt` — hashes for every archived file.

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

