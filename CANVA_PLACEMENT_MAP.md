# Canva placement map and composition handoff

The local composite board is the reviewable handoff for the ten-page Canva
design. It preserves the blueprint reading order and puts the release state on
each page-equivalent. The individual figures are available as PNG/PDF/SVG in
`figures/`.

| Canva page | Blueprint output/slot | Local asset | State at audit | Placement instruction |
| --- | --- | --- | --- | --- |
| P01 | Figure 1A–C | `figures/figure01_task_conditioned_universe.*` | PARTIAL | Full-width left/top; keep workflow schematic separate from count evidence. |
| P02 | Figure 1 workflow | included in Figure 1 composition | SCHEMATIC ONLY | Keep editable schematic boxes and arrows; no scientific count claim. |
| P03 | Figure 2A–C | `figures/figure02_near_equilibrium_parity.*` | PROVISIONAL | Use the parity panels; retain source-unit audit note. |
| P04 | Figure 2D–F | same Figure 2 asset | PROVISIONAL/PARTIAL | Keep distribution-shift panel as partial until metadata join is frozen. |
| P05 | Figure 6A–F | `figures/figure06_matched_md_diagnostics.*` | PROVISIONAL | Use observed-horizon/censored wording; no 1 ns or diffusion ranking claim. |
| P06 | Figure 5A–D | `figures/figure05_generated_neb_diagnostic.*` | BLOCKED | Keep red blocked box; do not replace with endpoint-only success/barrier plot. |
| P07 | current Canva Foundation NEB legacy page | catalogued by `CANVA_FIGURE_REGISTRY.tsv` | SCF WARNING / LEGACY | Do not substitute this legacy 3×3 page for the blocked current Figure 5. |
| P08 | current Canva Foundation NEB legacy page | catalogued by `CANVA_FIGURE_REGISTRY.tsv` | SCF WARNING / LEGACY | Same boundary as P07; retain only as lineage evidence. |
| P09 | current Canva Cr sliding page | catalogued by `CANVA_FIGURE_REGISTRY.tsv` | INTEGRITY SPLIT | Use the Downloads copy/hash as authority; do not treat the zero-filled staging PDF as evidence. |
| P10 | Figure 7A–D | `figures/figure07_interlayer_sliding_diagnostics.*` | SCF WARNING / PROXY | Keep fixed/generated separation and model-identity spread note. |
| new slot | Figure 4 transfer diagnostic | `figures/figure04_cross_environment_transfer.*` | DIAGNOSTIC | New blueprint slot; no current Canva page is claimed as an exact match. |
| new slot | Figure 8 conditional QE diagnostic | `figures/figure08_qe_diagnostic_conditional.*` | CONDITIONAL | New blueprint slot; keep B–D as placeholders until the geometry-keyed join is validated. |

## Composite board

Import the following as the current layout reference or use its SVG as a vector
placeholder board:

- `figures/atomic_migration_blueprint_canvas_board.png`
- `figures/atomic_migration_blueprint_canvas_board.pdf`
- `figures/atomic_migration_blueprint_canvas_board.svg`

The board is deliberately labeled: green/black content is data-backed or
diagnostic; red labels are release gates, warnings, or placeholders. The board
is not a scientific acceptance certificate.

## Canva connector status

The Canva short-link/design was already catalogued by PR #14, but the connected
Canva app currently returns `This app connection requires reauthentication`.
Therefore this turn does not claim an external Canva write. Once the connection
is reauthenticated, the page map above can be used to replace the existing
exports or add the SVG/PNG board with the same page ordering. No layout decision
should be changed without rerunning the visual gates.
