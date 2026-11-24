# Cr three-path fixed/self-relaxed profiles, R03

Columns are **in-gap diffusion**, **gap→QL**, and **within-QL**. The first row is the fixed-path single-point diagnostic; the second row is the endpoint-only self-relaxed MLIP-NEB proxy. Colored curves are ensemble means and translucent bands are pointwise sample standard deviations across the available model identities/accepted profiles. The black dashed curve is the DFT-NEB reference retained from `foundation_self_relaxed_4x3_R03.pdf`; its source status remains `COMPLETED_BUT_SCF_WARNING`.

This revision adds the canonical `Scratch-600K-Lmax1` campaign (`81 atoms × 10 images`, seeds 123/234/345) to all three paths. All nine fixed profiles are finite and plotted as supporting diagnostics. Seven of nine self-relaxed profiles pass the convergence gate and are plotted as primary evidence; seed 123 × within-QL and seed 234 × in-gap remain diagnostic-only and are excluded from the primary curve and band.

The historical `From-scratch` profiles are labelled `Scratch-MultiT` to match the page-13 legend because their launcher identifies `Scratch_Multi_T.model`; the underlying model IDs and source table are preserved. Self-relaxed curves remain endpoint-only MLIP-NEB proxies, not DFT minimum-energy paths or validated migration barriers.
