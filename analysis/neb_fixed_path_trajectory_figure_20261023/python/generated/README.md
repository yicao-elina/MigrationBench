# Fixed-path NEB diagnostic candidate

Run from the repository root:

    python analysis/neb_fixed_path_trajectory_figure_20261023/python/build_fixed_path_figure.py

For a rendering-only rerun from prepared data:

    python analysis/neb_fixed_path_trajectory_figure_20261023/python/render_lower_panels.py

The build writes only within this new figure directory and the separate
integration interface. Outputs include the long-form fixed-path table, DFT
profiles, independent PDF/SVG/600 dpi PNG panels, the three-column lower
panel, Blender JSON input, scene manifest, caption, manifest and QA report.

The top Blender PNGs are not manufactured by this Python layer. Once the
Blender executor has written within_gap.png, gap2QL.png and withinQL.png in a
directory of its choice, run:

    python analysis/neb_fixed_path_trajectory_figure_20261023/integration/compose_top_bottom.py --top-dir /path/to/blender/pngs

If any top PNG is absent, the integration step reports the missing interface
and does not create a synthetic replacement. The current Blender render, if
present, is always treated as an unreviewed candidate; the integration report
never writes VIS automatic approval. A future Blender rerender may instead
write triptych_top.png and be passed with --triptych-top or discovered under
--top-dir.
