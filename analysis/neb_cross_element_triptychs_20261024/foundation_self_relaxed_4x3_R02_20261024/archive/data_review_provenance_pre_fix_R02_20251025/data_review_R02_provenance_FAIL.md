FAIL

# R02C final independent data and provenance review

## 1. Review scope and verdict

Review date: 2026-10-25 (America/Los_Angeles). This is an independent, read-only review of the local campaign directory and the Rockfish campaign root:

`/scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024_r02`

The compute campaign is terminal, not pre-execution or partial: the formal array and R02A/R02B/R02C retries have scheduler completion evidence. The scientific row-level gates for the 108 selected records pass. The package-level release gate is nevertheless **FAIL** because the final provenance package is not internally synchronized and three machine-checkable provenance contracts are broken:

1. the local `RAW_ARTIFACT_INDEX.tsv` is missing the selected R02C attempt;
2. ten selected retry rows retain `FORMAL_LAUNCH` in `ROW_RESULTS.tsv` although their selected endpoint manifests are explicitly `RETRY_ARRAY_R02A/B/C`;
3. all four scheduler task maps have tab-delimited headers but pipe-delimited data rows, so they are not valid TSV files as named.

The failures are repairable without rerunning the scientific calculations. No code, endpoint input, model file, or raw run was modified during this review.

## 2. Frozen manifest and acceptance criteria

The Rockfish copy of the immutable launch manifest was independently hashed as:

- `CAMPAIGN_MANIFEST.json`: `e226280ce571cc7df286824a627772a1b8c3ebe7808fde5ca7805e82d1f0bd4c`
- `ACCEPTANCE_CRITERIA.md`: `70e2fc23c876721d321138afd3a49ecbb79208878f0b36af848d12bffde2c5a8`

`CAMPAIGN_MANIFEST.json` records `formal_launch_manifest_immutable=true`, `prior_outputs_reused=false`, endpoint-only IDPP initialization, 10 images, seed 42, and a target of 108 rows. `EXECUTION_BOOKKEEPING_R02.json` pins the same launch-manifest hash and confirms that the frozen base criteria applies to the explicit R02 revision. Its hash is:

`0cbc3ab4eea493c3a14955b2313aa606da7a8d7621d9bb80122654370367017d`

The immutable manifest still says `status=READY_FOR_FORMAL_ARRAY` and has `formal_launch_job_id=null`. I treated that as the frozen launch document, not as final execution status; final job and collection state are taken from the additive bookkeeping and scheduler audits. The bookkeeping records formal array `32459698`, final completion, R02A `32465592`, R02B `32466571`, and R02C `32468280`.

## 3. Matrix, model, path, and endpoint audit

### Matrix and model registry

- `JOB_TABLE_IDPP.tsv`: 108 rows; each element has 27 rows, each of the 12 canonical element/path IDs has 9 model rows, and all rows use `ENDPOINTS_ONLY`, `IDPP`, 10 images, seed 42.
- `MODEL_REGISTRY.tsv`: 9 Foundation models, with 5 MACE-MP, 2 MACE-OMAT-0, and 2 MACE-MH checkpoints; heads are 7 `default` and 2 `omat_pbe`.
- All 9 resolved Rockfish model files exist and their observed SHA-256 values match the registry values.
- No MPA, fine-tuned, scratch, Blender, or legacy checkpoint entered the primary model set.

Registry hashes on the Rockfish copy are:

| Artifact | SHA-256 |
|---|---|
| `MODEL_REGISTRY.tsv` | `b50521152e5b706a960fe8266f59cc254b976cfc789d7b35235c06fa69dbdd95` |
| `PATH_REGISTRY.tsv` | `0c19a16320904a1d996c4c29c0903027a600dda2c866381a77f2b01f182357a2` |
| `JOB_TABLE_IDPP.tsv` | `53864994aa026ce41cbf53ff63d91420098b5fb165a240f49aeea3e326ff7fd8` |
| `JOB_TABLE_IDPP_RETRY_R02A.tsv` | `620fd0b8db84855451f6b5aff725c171953cac26f67d60ceecc4ed7043d5f0a4` |
| `JOB_TABLE_IDPP_RETRY_R02B.tsv` | `1117e4fcc6533ad09937caf614b2a1a5c1fe7e038941eb0ac70ae1930fef37e2` |
| `JOB_TABLE_IDPP_RETRY_R02C.tsv` | `b5c8378919d1277c4da412845cd84f2eb41feb341d158f3ae49ad0cf79c59a16` |

### Paths and canonical endpoint headers

All 12 path records and all 24 endpoint files were checked. The independent endpoint-header checks found:

- 12 initial and 12 final headers;
- all headers use campaign ID `foundation_self_relaxed_4x3_20261024_r02`, the expected canonical `path_id`, element, and initial/final role;
- all endpoint-only flags are `T`, all PBCs are `T T T`, and all files contain 81 atoms;
- all 12 registry records use `header_schema=canonical_endpoint_v2`, `endpoint_header_roles=initial,final`, and `intermediate_source_frames_used_by_runner=NO`;
- the 3 Cr paths use `source_kind=CANONICAL_ENDPOINT_INPUT`; the 9 non-Cr path records use `LEGACY_LINEAR_SOURCE_ENDPOINTS_ONLY` only as an endpoint source description;
- all cell hashes and endpoint hashes match the path/job registry contracts.

The Cr `legacy_source_stem` values, including the `2L-Cu__...` stems, are aliases only: the canonical element/path identity comes from the registry and endpoint headers. No Cr identity was inferred from the legacy stem.

## 4. Final collection and selected-row gates

`data/COLLECTION_MANIFEST.json` is finalized and has SHA-256:

`3494ae4fef036132ec637361501c7270e7e279e21caa636a2052cecc3582484d`

Its declared values are `target_rows=108`, `rows_with_manifests=108`, `converged_rows=108`, `failure_rows=0`, `profile_rows=1080`, and `attempt_rows=125`. The referenced output hashes match the actual local and Rockfish copies for the collection outputs:

| Output | SHA-256 |
|---|---|
| `ROW_RESULTS.tsv` | `3e3c22c344d71c061c66485e26fa51730906c1447930404bded4e49c15506498` |
| `IMAGE_PROFILES.tsv` | `9327bd93104703a94ca3a69a2ec5d4c208685f1bb0fee4a30256bca517de8e5f` |
| `ATTEMPT_LEDGER.tsv` | `e476d9518e947704a5d3b70dd1ed6b243b62fa1c6da77bb30ccce3bbb529f777` |
| `FAILURE_REGISTRY.tsv` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Independent row-level validation of the 108 selected attempts found:

- 108/108 `scientific_status=CONVERGED`, `publication_status=PRIMARY_EVIDENCE`, `optimizer_converged=true`, and `target_model_convergence=true`;
- 1080/1080 profile rows, exactly 10 image indices per record, all energy/relative-energy/force/profile values finite;
- 108/108 trajectories with 10 ordered images, 81 atoms per image, finite coordinates and forces, matching element/path/cell/PBC headers, and endpoint roles at image 0 and image 9;
- all selected endpoint-manifest hashes and engine-manifest hashes match `ROW_RESULTS.tsv`, with model registry/observed hashes equal;
- all selected command manifests use endpoint inputs only, IDPP interpolation, seed 42, 10 images, and an empty intermediate-input list;
- runtime provenance is present for all selected attempts;
- maximum final internal NEB force: `0.04999082252330337 eV/A`;
- maximum final NEB residual: `0.04986932797684663 eV/A`.

Therefore the scientific convergence/data gates pass for the selected 108 rows. This statement does not waive the provenance blockers below.

## 5. Retry lineage and retained failures

The retry tables are dimensionally and invariant-consistent with the base job table:

- R02A: 10 records, `steps_max=1500`, `manifest_role=RETRY_ARRAY_R02A`;
- R02B: 5 records, `steps_max=3000`, `fire_dt=0.03`, `fire_maxstep=0.05`, `spring_constant=0.1`, `manifest_role=RETRY_ARRAY_R02B`;
- R02C: 1 record, `steps_max=6000`, `fire_dt=0.02`, `fire_maxstep=0.03`, `spring_constant=0.1`, `manifest_role=RETRY_ARRAY_R02C`.

The 10 records that entered retry are:

`000020, 000024, 000046, 000052, 000054, 000078, 000080, 000100, 000101, 000106`.

Attempt accounting is internally correct on the Rockfish copy: 97 records have one attempt, 6 have two, 4 have three, and 1 has four, totaling 125 attempts. The attempt ledger contains 109 converged attempts and 16 nonconverged attempts; the raw artifact index contains the same 125 attempts, with the 16 nonconverged attempts marked `DIAGNOSTIC_ONLY`. All 16 nonconverged attempt directories retain `mlff_neb.log`; they were not silently discarded. The empty final `FAILURE_REGISTRY.tsv` therefore represents zero failed selected release rows, while historical failed attempts remain visible in the attempt ledger/raw index.

## 6. Scheduler evidence

The four scheduler audit JSON records report empty stderr and the following manually parsed task-map content:

| Audit | Job | Data rows | Parsed state/exit |
|---|---:|---:|---|
| formal | `32459698` | 108 | 108 `COMPLETED`, `0:0` |
| R02A | `32465592` | 10 | 10 `COMPLETED`, `0:0` |
| R02B | `32466571` | 5 | 5 `COMPLETED`, `0:0` |
| R02C | `32468280` | 1 | 1 `COMPLETED`, `0:0` |

The scheduler completion evidence is therefore consistent with the terminal compute outcome. The bookkeeping also records superseded R02B submission `32466555` as failed before row read due to the missing TSV suffix; that event is not present as a separate scheduler audit artifact.

## 7. Derived profiles and denominators

The local derived files are internally consistent:

- `CLASS_SUMMARY.tsv`: 36 rows, all `PASS`;
- `CLASS_PROFILES.tsv`: 360 rows = 36 class groups × 10 images, all finite and `complete_class=PASS`;
- `PROFILE_INCLUSION_AUDIT.tsv`: 108 rows, all `CONVERGED`, `PRIMARY_EVIDENCE`, `included_in_class_profile=YES`, `failure_class=NA`;
- every element/path class has 5 MACE-MP + 2 MACE-OMAT-0 + 2 MACE-MH = 9 accepted models;
- every class denominator equals its expected denominator: 5, 2, and 2 by family.

Local derived hashes are:

| Artifact | SHA-256 |
|---|---|
| `CLASS_PROFILES.tsv` | `4ef42bebe436d968e8bfd1b4e5fcdb27f89a005458e15f42bfe27efec61b076a` |
| `CLASS_SUMMARY.tsv` | `a4a69f24e6510d16898aa6396a5923c86d2937d227e361c43ab371550db4d475` |
| `PROFILE_INCLUSION_AUDIT.tsv` | `8ed7c2cea9d014848806cd19ae12e448a7dc454457353baca2b0fe16741e9acb` |
| `DERIVED_PROFILE_MANIFEST.json` | `df8fcaafc3cf9682fdf934a1a61fd71adf778ab70c13c486e5ca3771899a347f` |

The Rockfish `data/` snapshot did not contain these four derived-profile files at review time, although the local final aggregation does. In addition, `DERIVED_PROFILE_MANIFEST.json` hashes its output files but does not bind them to the exact input `ROW_RESULTS.tsv`/`IMAGE_PROFILES.tsv` hashes. This is a release-package synchronization/provenance gap, not evidence that the local class calculations are numerically incomplete.

## 8. Numbered release blockers and executable repairs

### F-01 — local raw-artifact index is stale/incomplete (blocker)

Local `data/RAW_ARTIFACT_INDEX.tsv` has 125 physical lines including its header, i.e. 124 data rows. It is missing:

`FSELF-R02-20261024-000024 / attempt_32468280_0`

The local `RAW_ARTIFACT_INDEX.json` declares `attempt_manifest_count=125` and `output_sha256=f23468d...`, which is the hash of the complete Rockfish TSV, not the local TSV. The complete Rockfish TSV has 126 lines and SHA-256 `f23468d05857f3d5587007dec65c6fefbe3244ef1d4e0479f7149058291a71fc`; the stale local TSV hash is `e9f7937c053c7e5c33ba545166c1d2cf1b81b8ec72bfce8b792d67a4beb2fb61`.

Repair: regenerate or synchronize the local raw index from the final R02C-inclusive raw run set, preserving all existing rows; verify a one-to-one key match against `ATTEMPT_LEDGER.tsv`, verify all 108 selected attempts are present, recompute `RAW_ARTIFACT_INDEX.json.output_sha256`, and update any final release manifest that binds this index. Do not hand-edit or delete raw attempts.

### F-02 — selected retry manifest role is not propagated to `ROW_RESULTS.tsv` (blocker)

For all 10 retried records, `ROW_RESULTS.tsv` says `manifest_role=FORMAL_LAUNCH`, while the selected endpoint manifest says `RETRY_ARRAY_R02A`, `RETRY_ARRAY_R02B`, or `RETRY_ARRAY_R02C`. The selected `manifest_path`, `attempt_dir`, Slurm job ID, and endpoint manifest are the retry artifacts, so the row-level provenance fields are not self-consistent.

Repair: retain the raw endpoint manifests unchanged, then regenerate the row result with either (a) the selected endpoint manifest role and an explicit `retry_round`/`selected_retry_job_id`, or (b) an explicitly named campaign-manifest role plus a separate `selected_attempt_manifest_role` field. Recheck all 108 rows so the row record, selected endpoint manifest, retry table, attempt ledger, and raw index agree. Recompute `ROW_RESULTS.tsv`, collection, class-inclusion, derived-profile, and release hashes after the repair.

### F-03 — scheduler maps are not valid TSV files (blocker)

`SCHEDULER_TASK_MAP.tsv`, `SCHEDULER_TASK_MAP_RETRY_R02A.tsv`, `SCHEDULER_TASK_MAP_RETRY_R02B.tsv`, and `SCHEDULER_TASK_MAP_RETRY_R02C.tsv` have tab-separated headers but each data row has six pipe separators and zero tabs. A tab-delimited reader therefore sees each data row as one field. Manual pipe parsing recovers 124 rows, all `COMPLETED/0:0`, but the advertised machine-readable schema is broken.

Repair: regenerate each map with a single declared delimiter. The simplest repair is to preserve the raw `sacct --parsable2` pipe output as `.psv` with a pipe-delimited header, or normalize both header and data to true tab-separated rows while recording the conversion. Recompute map hashes, link each map to its audit JSON, and rerun coverage checks against the attempt ledger and selected row jobs. Add a separate audit/map for smoke job `32459612_0` if smoke is included in the final scheduler provenance scope.

### F-04 — derived-profile package is not synchronized/bound across local and remote copies (release blocker if Rockfish is the archival source)

The local class/profile artifacts are complete and internally pass, but the reviewed Rockfish `data/` directory lacks them, and the derived manifest does not record the exact input collection hashes.

Repair: choose and document one authoritative final release location, copy the derived artifacts without changing their contents, add the derived manifest hash plus source `ROW_RESULTS.tsv` and `IMAGE_PROFILES.tsv` hashes to the final bookkeeping/release manifest, and verify the local/remote byte hashes. If derived data is intentionally local-only, state that explicitly in the release manifest instead of presenting the remote root as a complete archive.

### F-05 — empty failure registry has no schema header (secondary)

`FAILURE_REGISTRY.tsv` is zero bytes. The zero selected-row failure count is supported, and the 16 historical nonconverged attempts are retained elsewhere, so this did not hide a failed selected row. However, an empty file has no schema for downstream readers.

Repair: emit a header-only failure registry with an explicit zero-row status, or document that `ATTEMPT_LEDGER.tsv` is the historical failure ledger and add a machine-readable `selected_failure_count=0` field. Preserve the 16 excluded nonconverged attempt records and their logs.

## 9. Scientific boundary retained after repair

The accepted observable is `SELF_RELAXED_ENDPOINT_ONLY_NEB`: a Foundation-MLFF self-relaxed profile and diagnostic preconditioner. It is not DFT ground truth, not a validated DFT MEP, and not a cross-model physical barrier claim. The 10-image profiles are model-generated observables normalized to the initial image. Fixed-path single-point profiles are excluded by the campaign manifest and must remain a separate observable. The `dft_reference_status=NOT_USED_SELF_PROFILE` fields and the frozen acceptance criterion correctly preserve that boundary.

Until F-01 through F-03 are repaired and independently rechecked, the correct disposition is **FAIL / hold release**, despite the 108/108 selected scientific rows meeting the numerical convergence and profile gates.
