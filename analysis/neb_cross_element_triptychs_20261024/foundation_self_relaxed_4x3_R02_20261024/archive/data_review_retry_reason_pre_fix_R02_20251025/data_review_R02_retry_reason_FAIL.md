FAIL

# R02 final package independent data/provenance review

## 1. Scope and disposition

Review date: 2026-10-25. This is a new read-only review of the local package and the Rockfish package root:

`/scratch16/pclancy3/yi/foundation_self_relaxed_4x3_20261024_r02`

The formal array and all retry/smoke scheduler records are terminal. The previous F-01–F-05 findings were rechecked and are fixed: the local and remote final data hashes agree, the raw index is complete, selected retry roles are propagated, the failure registry has a schema header, all five scheduler maps are true TSV, and the derived package is synchronized and input-bound.

The final package remains **FAIL** for one new, specific provenance blocker: the R02A retry reason is absent from the attempt ledger and from five final selected rows. The R02A retry round and manifest role are present and correct, but the reason field is not consistent with the frozen, hashed R02A retry registry.

No code, endpoint input, model file, or raw run was modified during this review.

## 2. Hash and bookkeeping closure

The immutable launch and frozen criteria hashes remain unchanged and agree locally/remotely:

- `CAMPAIGN_MANIFEST.json`: `e226280ce571cc7df286824a627772a1b8c3ebe7808fde5ca7805e82d1f0bd4c`
- `ACCEPTANCE_CRITERIA.md`: `70e2fc23c876721d321138afd3a49ecbb79208878f0b36af848d12bffde2c5a8`

`CAMPAIGN_MANIFEST.json` still correctly acts as the immutable launch manifest. Its `READY_FOR_FORMAL_ARRAY` status and null `formal_launch_job_id` were not used as final execution status; `EXECUTION_BOOKKEEPING_R02.json` supplies the additive final state and binds the unchanged launch hash.

The local and Rockfish copies agree for the final package artifacts checked here:

| Artifact | SHA-256 |
|---|---|
| `COLLECTION_MANIFEST.json` | `d9d0e139ebbcc6a284e86be57457d85151ccbe68247e4facecf2ebc3f2678a31` |
| `RAW_ARTIFACT_INDEX.tsv` | `f23468d05857f3d5587007dec65c6fefbe3244ef1d4e0479f7149058291a71fc` |
| `RAW_ARTIFACT_INDEX.json` | `212f34687b237e7d020ea9fd1bcd7d641008edf39b4b97e50c95769cedb880fd` |
| `ROW_RESULTS.tsv` | `b4a6d567e264c8a1c248d29e48e18f28dcd772d23737ddacad1d5e798b10ab8e` |
| `IMAGE_PROFILES.tsv` | `9327bd93104703a94ca3a69a2ec5d4c208685f1bb0fee4a30256bca517de8e5f` |
| `FAILURE_REGISTRY.tsv` | `fd3ed5181ff03df402d9771b64b1b1fbcb03ec2aaaf914069e4c44632275c6fc` |
| `ATTEMPT_LEDGER.tsv` | `c49c4ab6780fba56e74bbcc40c4243509f53b07aa5b0c14706779146ccf2e245` |
| `DERIVED_PROFILE_MANIFEST.json` | `bb1fc8fda34be172c52bf6e8cf65f77c77df1553175db3909c6162bde7ddc10f` |
| `EXECUTION_BOOKKEEPING_R02.json` | `995bcb63b4741bfab8d0df2a16d9ee6f2f8b0f5554e20548f13aa0819d433ca4` |

Bookkeeping-to-file verification passed for the collection outputs, raw index, derived manifest, all scheduler map hashes, and all scheduler audit hashes. The raw-index JSON reports 125 attempts and its `output_sha256` equals the actual final TSV hash.

## 3. Matrix, registries, and endpoint contract

- `JOB_TABLE_IDPP.tsv`: 108 rows = 4 elements × 3 paths × 9 models; each element has 27 rows, each canonical element/path has 9 models, and every row uses endpoint-only IDPP with 10 images and seed 42.
- `MODEL_REGISTRY.tsv`: 9 actual Foundation checkpoints; 5 MACE-MP, 2 MACE-OMAT-0, 2 MACE-MH; 7 `default` heads and 2 `omat_pbe` heads. The resolved model files and registry hashes agree.
- `PATH_REGISTRY.tsv`: 12 canonical element/path records. `intermediate_source_frames_used_by_runner=NO` for all 12.
- All 24 endpoint headers were checked: 12 initial + 12 final, 81 atoms, matching element/path/role, `endpoint_only_input=T`, PBC `T T T`, campaign ID `foundation_self_relaxed_4x3_20261024_r02`, and registry cell/endpoint hashes.
- Cr uses `CANONICAL_ENDPOINT_INPUT` for all three paths. The `2L-Cu__...` legacy stems on Cr rows remain aliases only; identity is taken from canonical registry/header fields.

The registry SHA-256 values independently rechecked on Rockfish are:

- `MODEL_REGISTRY.tsv`: `b50521152e5b706a960fe8266f59cc254b976cfc789d7b35235c06fa69dbdd95`
- `PATH_REGISTRY.tsv`: `0c19a16320904a1d996c4c29c0903027a600dda2c866381a77f2b01f182357a2`
- `JOB_TABLE_IDPP.tsv`: `53864994aa026ce41cbf53ff63d91420098b5fb165a240f49aeea3e326ff7fd8`
- R02A/R02B/R02C retry tables: `620fd0b8db84855451f6b5aff725c171953cac26f67d60ceecc4ed7043d5f0a4`, `1117e4fcc6533ad09937caf614b2a1a5c1fe7e038941eb0ac70ae1930fef37e2`, `b5c8378919d1277c4da412845cd84f2eb41feb341d158f3ae49ad0cf79c59a16`.

## 4. Raw index, attempt ledger, and retry lineage

The repaired raw-index/ledger coverage now passes:

- `ATTEMPT_LEDGER.tsv`: 125 attempts;
- `RAW_ARTIFACT_INDEX.tsv`: 125 attempts;
- `(record_id, attempt_id)` key difference in either direction: zero;
- selected attempts missing from raw index: zero;
- endpoint-manifest and engine-manifest hash mismatches in raw index: zero;
- `RAW_ARTIFACT_INDEX.json` count/hash checks: pass;
- attempt statuses: 109 converged and 16 nonconverged historical attempts; all 16 remain retained as diagnostic-only attempts with their run logs;
- final selected rows: 108 converged rows, 0 selected failures.

Retry table dimensions and invariant fields pass:

- R02A: 10 rows, all `RETRY_ARRAY_R02A`, `retry_round=R02A`;
- R02B: 5 rows, all `RETRY_ARRAY_R02B`, `retry_round=R02B`;
- R02C: 1 row, `RETRY_ARRAY_R02C`, `retry_round=R02C`;
- base element/path/model/endpoint/cell/seed/image invariants: no mismatches.

The selected `ROW_RESULTS.tsv` manifest-role propagation now passes: 98 `FORMAL_LAUNCH`, 5 `RETRY_ARRAY_R02A`, 4 `RETRY_ARRAY_R02B`, and 1 `RETRY_ARRAY_R02C`; each agrees with `selected_attempt_manifest_role`, the selected endpoint manifest, and the retry round.

## 5. Scientific row and profile gates

Independent validation of all 108 selected rows found:

- 108/108 `CONVERGED`, `PRIMARY_EVIDENCE`, `failure_class=NA`;
- 1080 image-profile rows, exactly 10 images per record, with finite energy, relative energy, and force values;
- 108/108 trajectories with 10 ordered images, 81 atoms/image, finite coordinates and forces, matching symbols, cell/PBC, path, element, and endpoint roles;
- endpoint-only command manifests with IDPP, seed 42, 10 images, and no intermediate input list;
- model registry/observed hashes, engine-manifest hashes, runtime provenance, and row-result hashes consistent;
- maximum final internal NEB force: `0.04999082252330337 eV/A`;
- maximum final NEB residual: `0.04986932797684663 eV/A`.

The final collection manifest reports `target_rows=108`, `rows_with_manifests=108`, `converged_rows=108`, `failure_rows=0`, `profile_rows=1080`, and `attempt_rows=125`. `FAILURE_REGISTRY.tsv` now has the declared schema header and zero data rows. Historical nonconverged attempts remain in the attempt ledger/raw index and were not silently removed.

## 6. Scheduler maps and audits

All five local/remote map files are byte-identical and parse successfully with `csv.DictReader` as true TSV:

| Evidence | Job | Rows | State/exit |
|---|---:|---:|---|
| formal | `32459698` | 108 | 108 `COMPLETED/0:0` |
| R02A | `32465592` | 10 | 10 `COMPLETED/0:0` |
| R02B | `32466571` | 5 | 5 `COMPLETED/0:0` |
| R02C | `32468280` | 1 | 1 `COMPLETED/0:0` |
| smoke | `32459612_0` | 1 | 1 `COMPLETED/0:0` |

Each audit JSON points to the matching map basename and job ID, has empty stderr, and has a bookkeeping hash match. The superseded R02B submission `32466555` remains explicitly documented as a failed-before-row-read submission and is not confused with the successful R02B array.

## 7. Derived profiles and class denominators

The local and remote derived artifacts are now synchronized and the derived manifest binds exact input hashes:

- `DERIVED_PROFILE_MANIFEST.json` input hashes equal the final `ROW_RESULTS.tsv` and `IMAGE_PROFILES.tsv` hashes;
- `CLASS_SUMMARY.tsv`: 36 rows, all `PASS`;
- `CLASS_PROFILES.tsv`: 360 rows = 36 class groups × 10 images, all finite and complete;
- `PROFILE_INCLUSION_AUDIT.tsv`: 108 rows, all included, converged, primary evidence;
- every element/path class has 5 MACE-MP + 2 MACE-OMAT-0 + 2 MACE-MH = 9 models, with accepted denominators equal to expected denominators.

Derived output hashes are:

- `CLASS_PROFILES.tsv`: `4ef42bebe436d968e8bfd1b4e5fcdb27f89a005458e15f42bfe27efec61b076a`
- `CLASS_SUMMARY.tsv`: `a4a69f24e6510d16898aa6396a5923c86d2937d227e361c43ab371550db4d475`
- `PROFILE_INCLUSION_AUDIT.tsv`: `8ed7c2cea9d014848806cd19ae12e448a7dc454457353baca2b0fe16741e9acb`

## 8. Remaining blocker

### F-06 — R02A `retry_reason` is missing from the retry lineage

The frozen, hashed retry registry gives all 10 R02A rows the reason:

`target-model convergence/residual gate failed in prior formal attempt`

However, independent comparison found:

- all 10 R02A rows in `ATTEMPT_LEDGER.tsv` have `retry_round=R02A` and `manifest_role=RETRY_ARRAY_R02A`, but blank `retry_reason`;
- the 5 final rows selected from R02A also have blank `retry_reason` in `ROW_RESULTS.tsv`:
  - `FSELF-R02-20261024-000020`
  - `FSELF-R02-20261024-000078`
  - `FSELF-R02-20261024-000080`
  - `FSELF-R02-20261024-000100`
  - `FSELF-R02-20261024-000101`
- the corresponding R02A retry-table rows all contain the non-empty reason;
- R02B and R02C selected rows do contain the expected reason.

This is a provenance blocker because the final package cannot explain the cause of the R02A retry for five release rows from the selected row and ledger records themselves, even though the role and round are correct.

Executable repair: backfill the R02A `retry_reason` into the 10 attempt-ledger rows and the 5 selected row-result records from `JOB_TABLE_IDPP_RETRY_R02A.tsv`, without editing the raw endpoint manifests or raw runs. Then recompute and verify `ATTEMPT_LEDGER.tsv`, `ROW_RESULTS.tsv`, `COLLECTION_MANIFEST.json`, `EXECUTION_BOOKKEEPING_R02.json`, and the derived-manifest input hash chain. Re-run the exact table-to-ledger and selected-row-to-manifest retry comparison; no scientific rerun is required.

## 9. Scientific boundary

The accepted observable remains `SELF_RELAXED_ENDPOINT_ONLY_NEB`: a Foundation-MLFF self-relaxed model profile/diagnostic preconditioner. It is not DFT ground truth, not a validated DFT MEP, and not an unqualified physical barrier ranking. Fixed-path single-point data remain excluded and separate. The zero selected-failure result does not erase the 16 retained historical nonconverged attempts.

Until F-06 is repaired and re-reviewed, the package disposition is **FAIL / hold release**. All previously reported F-01–F-05 blockers pass in this review.
