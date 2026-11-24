# Blender/Python integration interface

The Python data layer does not create a synthetic top panel. The Blender
executor must provide these real-geometry PNG files in a directory of its
choice:

- within_gap.png
- gap2QL.png
- withinQL.png

The files must follow the order and source hashes in
python/generated/blender_input.json and python/generated/scene_manifest.json.
The integration script accepts either the three names above, the current
canonical names cr_canonical_in_gap.png, cr_canonical_gap_to_ql.png and
cr_canonical_within_ql.png, or one full-width triptych_top.png:

    python analysis/neb_fixed_path_trajectory_figure_20261023/integration/compose_top_bottom.py --top-dir /path/to/blender/pngs

Alternatively:

    python analysis/neb_fixed_path_trajectory_figure_20261023/integration/compose_top_bottom.py --triptych-top /path/to/triptych_top.png

Every successful composition is still marked
PROVISIONAL_UNREVIEWED_BLENDER_CANDIDATE. It is not a VIS pass.

If any file is absent, the script writes INTEGRATION_GAP.json and exits
without making a composite. It never creates placeholder atoms or a fake
trajectory.
