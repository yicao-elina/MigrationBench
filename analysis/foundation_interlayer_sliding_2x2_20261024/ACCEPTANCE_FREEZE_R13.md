# Interlayer sliding Foundation campaign — acceptance freeze R13

Status: **FROZEN BEFORE PRODUCTION SUBMISSION**  
Campaign: `foundation_interlayer_sliding_2x2_20261024`  
Freeze date: 2026-10-24  
Revision: R13 supersedes the retained R12 (which remains immutable evidence of a rejected pre-run draft); R01–R12 remain immutable review evidence.  
Purpose: complete the Foundation-only fixed-path and self-relaxed interlayer-sliding campaign and produce the accepted 2×2 figure.

## 1. Scope frozen for this run

The campaign contains exactly 12 primary calculation rows:

| Observable row | Bulk | Cr-in-gap | Per cell |
|---|---:|---:|---:|
| Fixed path, single point | 3 models | 3 models | 10 b-path2 frames |
| Self-relaxed, endpoint-only NEB | 3 models | 3 models | 10 final images |

The three model identities are `MACE-MP-0b3-medium`, `MACE-OMAT-0-medium`, and `MACE-MPA-0-medium`. The legacy Fig. 4 Foundation checkpoint (`dc41a989…`) is excluded. Fine-tuned, scratch, multihead, 4×3, and any other Foundation checkpoints are excluded from this campaign.

The two columns are:

- `Bulk`: `1.bulk`, 80 atoms.
- `Cr-in-gap`: `2.cr-in-gap`, 81 atoms, the interstitial Cr case.

Only the recovered QE `3.neb/2.2-layer-dislocation/b-path2/{1.bulk,2.cr-in-gap}` inputs are allowed. The top-level `3.neb/2.2-layer-dislocation/{1.bulk,2.cr-in-gap}` path is explicitly excluded because it is a different, unclosed path whose endpoints do not return to zero.

## 2. Fixed-path protocol

1. Use the recovered `sb2te3.xyz` from each b-path2 directory, exactly 10 frames in the stored order.
2. Use the recovered `mace_neb_1-calc.py` algorithm: native MACE loading and native `AtomicData` inference, with one model and one system per job. The campaign copy is parameterized only through environment variables; its original recovery scripts remain beside it with recorded hashes.
3. Use the current model path from `MODEL_REGISTRY.tsv`; do not substitute the old legacy Foundation checkpoint.
4. Retain raw predicted energies for all 10 images. For profiles, subtract image 0 from every model prediction. Do not rescale, smooth, interpolate, or force the last value to zero.
5. For every plotted trace, use the frozen x coordinate `image_fraction = image_index/(N-1)`, so both observables share the same 10-point 0→1 image axis. The QE `Reaction_Coord` column is retained verbatim in the raw/reference table but is not silently mixed with the self image axis and is not used to interpolate any model trace.
6. Retain the QE `combined_sb2te3_energy.csv` profile as the reference series. QE is labeled `COMPLETED_BUT_SCF_WARNING` and is a diagnostic retained-path reference, not converged ground truth. The plotted QE y values are `raw_energy - raw_energy[0]`; raw values remain unchanged in evidence.
7. A fixed row passes only when the executable gate confirms the expected atom count and 10-frame count, the model loads, all 10 predictions are finite and ordered, a normalized `fixed_profile.csv` is written, and the input/model/registry/script/wrapper/environment hashes are recorded.

## 3. Self-relaxed protocol

1. Use only endpoint 0 and endpoint 9 from the same recovered b-path2 XYZ. No intermediate QE geometry is passed to the self branch.
2. Attach the matching QE `pw_1.in` cell and `pbc=(True, True, True)` to the two endpoints. Build 10 images with ASE IDPP interpolation.
3. Optimize with the validated endpoint-only MACE/ASE NEB engine: fixed endpoints, 10 images, `FIRE`, `fmax=0.05 eV/Å`, `steps=500`, spring constant `k=0.1`, `climb=False`, seed `42` for job metadata. The final trajectory and optimizer history are retained.
4. The executable production self role rejects `--images`, requires `--initial/--final`, requires exactly 10 images, IDPP, fixed endpoints, `climb=False`, `steps=500`, `fmax=0.05 eV/Å`, and `k=0.1`. The engine writes an explicit endpoint-drift and scientific-gate result.
5. The self branch is an MLIP-relaxed path observable, labeled `SELF_RELAXED_ENDPOINT_ONLY_NEB`; it is not a DFT minimum-energy path and is not pooled with the fixed-path observable. Its plotted y values are the engine’s final `energy_eV - energy_eV[0]` values.
6. A self row passes only when the executable first confirms both endpoint files have the declared 80/81 atom count, then produces 10 finite final images with endpoint geometry unchanged within `1e-8 Å`, all final energies and forces finite, engine convergence, and final internal NEB force and residual each `≤0.05 eV/Å`. A scheduler `COMPLETED` state alone never passes the row.

## 4. Data and provenance gates

The full campaign is accepted only if all are true:

- 12/12 primary rows have a scientifically valid status in `results/tables/RESULT_MATRIX.tsv`; failed and warning-bearing rows remain visible in the failure registry and prevent an accepted complete release.
- Every row has 10 ordered frames/images, finite numeric values, expected atom count, and matching endpoint/system identity.
- Model SHA256 observed and registry, model-registry SHA256, input SHA256, runner/wrapper/job-table SHA256, environment versions, Rockfish job ID, output SHA256, and collection timestamp are recorded for both branches.
- Every job writes an immutable `runtime_provenance.json`; its SHA256 is recorded as `environment_sha256` in the row manifest. A package-version dictionary alone is not sufficient.
- The fixed and self-relaxed observables are kept as separate tables and separate claims. No pooled barrier ranking is allowed.
- QE warning status is retained in every report and figure label. No QE series is called ground truth.
- The release manifest, result matrix, claim/evidence matrix, failure/exclusion registry, and gap report agree on row counts and statuses.

## 5. Figure contract

The accepted product is a 2×2 Plot Atlas-style figure:

| | Bulk | Cr-in-gap |
|---|---|---|
| Fixed path, single point | panel A | panel B |
| Self-relaxed, endpoint-only NEB | panel C | panel D |

Frozen presentation rules:

- Three model curves in every panel: open-circle markers with solid lines.
- Visual palette follows the Foundation-class reference grammar: `MACE-MP-0b3-medium` gold `#d5a238`, `MACE-OMAT-0-medium` red `#d85a45`, `MACE-MPA-0-medium` green `#5d8b56`. The colors are visual identities, not model-performance claims.
- Fixed panels include black dashed open diamonds for `QE b-path2 retained reference (SCF-warning)`.
- Self panels may include the same QE series only as a gray dashed open-diamond contextual trace labeled `QE retained path (context only)`; it must never be presented as the self-relaxed result or ground truth.
- Shared x label: `normalized image coordinate (0 → 1)`; y label: `ΔE from image 0 (eV)`. The x value is always `image_index/(N-1)`; no x interpolation is performed.
- Independent y limits per panel are allowed for readability, as in the supplied reference. No grid, no smoothing, no interpolation, and no uncertainty band from a single checkpoint.
- Export composite and each panel as PDF, SVG, and PNG. The composite is not accepted unless the data gate and both independent post-run reviews pass.

## 6. Review gates

Before production submission, independent data and visual reviewers must each return `PASS` against this frozen document. After all jobs finish, independent post-run data and visual reviewers must each return `PASS` against the collected artifacts. A review agent may inspect and report, but may not edit the acceptance freeze or silently repair results.

Reference-style provenance: the requested same-named R03 PDF was absent at its original path. The visual reference is formally relinked at `references/foundation_model_class_3x3_R03.pdf` to the archived same-named R03 artifact; the relink and reference SHA256 are recorded in `STYLE_REGISTRY_R13.json`. The renderer uses exact `-medium` model labels, explicit fixed/reference and self/context QE legend entries, boxed panel spines, and 10.5–11.5 pt Plot Atlas-scale typography, including the legends. The renderer's data gate independently checks exact row identity, expected 80/81 atom counts, exact `{0,…,9}` indices, per-trace metadata consistency, all required provenance hashes plus job ID, finite values, normalized x coordinates, and image-0 baselines before final rendering.

## 7. Permitted claims after acceptance

The release may state that the three named current Foundation checkpoints were evaluated on the two recovered b-path2 systems under the two explicitly separated observables, with the exact profiles shown. It may describe model-to-model profile/barrier differences within each observable. It may not call the QE warning-bearing series converged ground truth, may not equate self-relaxed MLIP paths with DFT MEPs, and may not pool fixed/self values into one barrier ranking.
