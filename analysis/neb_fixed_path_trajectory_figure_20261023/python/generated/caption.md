# Fixed-path NEB trajectory/profile candidate

Status: PROVISIONAL_DIAGNOSTIC_CANDIDATE. This is not an accepted
publication final and does not label the DFT/QE curves as unconditional
ground truth.

The columns are ordered within gap → gap2QL → withinQL. Each upper column is
intended to receive the corresponding real 81-atom, 10-image retained path
from generated/blender_input.json. The lower panel uses the same path_id and
retained image sequence.

The curves show ΔE_i = E_i − E_0 in eV at ten retained images versus the
normalized cumulative minimum-image RMS coordinate s. Points are joined
directly; no spline or extra image is generated. The black/graphite curve is
the last complete QE/NEB table and is labeled DFT/QE — SCF warning; diagnostic.
It is not unconditional DFT ground truth, and its force-error column is not an
energy uncertainty band.

Foundation checkpoint variants and heterogeneous From-scratch branches are
discrete arms, not training-seed ensembles. FT-600K and FT-MultiT mean ± 1
sample SD bands are withheld because the verified-seed gate is not complete:
FT-600K seed 123 is MODEL_HASH_VERIFIED_BASE_UNPROVEN and FT-MultiT seed 123
is a MODEL_LOAD failure on all three paths. Failed blocks remain visible as ×
status annotations. No missing value is filled, interpolated, or treated as
zero.

This candidate excludes SELF_RELAXED_ENDPOINT_ONLY_NEB and MIC_LINEAR. See
figure_data_manifest.json, data/fixed_path_long.tsv,
data/dft_profiles.tsv, and generated/scene_manifest.json for provenance.
DFT-02/DFT-04 and SEED-04 remain blocking conditions for publication sign-off.

At runtime the mirrored fixed image_profile.csv files were present but their
reads blocked, so the numeric rows were taken from the hashed derived
TIER1_IMAGE_PROFILES.tsv table. The remote path, read status, fallback source,
and unavailable raw-profile hash are retained explicitly; this is a provenance
blocker, not a silent substitution.
