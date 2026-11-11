# Fixed-path NEB trajectory/profile candidate — R02

Status: **PROVISIONAL_DIAGNOSTIC_CANDIDATE_R02**. This R02 artifact is not publication final and does
not call the warning-bearing DFT/QE curves unconditional ground truth.

The three columns are strictly ordered **within gap → gap2QL → withinQL**.
The shared panel geometry contract is
`/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/neb_fixed_path_trajectory_figure_20261023/integration/panel_geometry_contract_R02.json`: the top triptych and lower axes use the same x0/x1/center
boundaries, with zero-pixel alignment tolerance. The integration script refuses
to rescale a lower raster whose width does not match the contract.

Ghosted spheres in the upper Blender candidate mean **real retained
path-image overlay**. Alpha encodes **image overlay order only, not
uncertainty/error**. No interpolated or synthetic geometry is added.

The lower curves use ΔE_i = E_i − E_0 versus normalized reaction coordinate
s and connect the ten retained images directly. DFT/QE is explicitly labeled
**SCF warning; diagnostic**. Foundation checkpoint variants and heterogeneous
From-scratch branches are discrete; FT-600K and FT-MultiT uncertainty bands
are withheld because the verified fixed-path training-seed gate is incomplete.

Current data ledger: **33 finite fixed diagnostic blocks**, **3 MODEL_LOAD
failure blocks**, and **3 × 10 = 30 DFT/QE warning rows**. The FT-600K band is
withheld because seed 123 is `MODEL_HASH_VERIFIED_BASE_UNPROVEN`; the FT-MultiT
band is withheld because seed 123 is `MODEL_LOAD` failure on all three paths.
Failures remain explicit and are not zero-filled, interpolated, or silently
removed.

For PHY-16/PUB-09, the derived
`TIER1_IMAGE_PROFILES.tsv` fallback was independently revalidated: SHA256
`431afb77043fee4104ed691c0de0a46eb8645fa3d21458f167fb4f2ed4bb5a4c`, total rows `1590`;
the selected target coverage is `36 blocks =
33 complete ten-image blocks (330 rows)
plus 3 explicit failure blocks. The mirrored remote
fixed profile paths and runtime read limitation are retained; no unavailable
raw-profile SHA256 is fabricated. This limitation remains a publication
blocker.

Excluded observables: `SELF_RELAXED_ENDPOINT_ONLY_NEB` and `MIC_LINEAR`.
