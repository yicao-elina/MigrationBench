# Cr-free interlayer-sliding preview QA

`PREVIEW_VISUAL_GATE = READY_FOR_USER_REVIEW`

## Source resolution

- User-supplied 1800px image: `/Users/alina/Downloads/interlayer_sliding__final_state__window_1800.png`
- Original Blender scene: `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/cr_sb2te3_nafsi_style_20261026/interlayer_sliding_final_preview/interlayer_sliding__final_state__window_1800.blend`
- The matching interlayer-sliding directory contains the 1800px PNG and `.blend`, but no corresponding 3600px PNG. Other nearby `*_hd_3600.png` files are different migration/QL scenes and are not substitutes.

## Revision

- Source scene loaded read-only and copied into this artifact directory.
- Only mesh object `Cr_image_00` was hidden from render and viewport.
- The `Cr_Hero_Key` light and all other lights, camera, atom materials, host atoms, slab offset, and render settings were preserved.
- No coordinates, host atoms, or layer geometry were edited.

## Output

- Preview: `preview/interlayer_sliding__final_state__no_cr__window_1800.png`
- Resolution: 1800 × 1800 px, RGB, 16-bit PNG, opaque white field.
- Preview SHA-256: `b9695a33196e4afea5d1dd3128a9f642173642fdfa7da163adfef00157994d22`
- No-Cr scene: `interlayer_sliding__final_state__no_cr__window_1800.blend`
- Manifest: `render_manifest.json`

The image is a review preview. Per the established Blender workflow, the 3600px export remains deferred until the user approves this Cr-free preview.
