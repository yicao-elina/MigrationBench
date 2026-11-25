# Bounded `/data/pclancy3/yi` refresh — 2026-11-23

## Scope and completion

This inventory pass is a bounded, read-only continuation. It did not repeat the broad `/data/pclancy3` walk or traverse mount-root owners. GNU `find -P -xdev` read metadata only, did not follow symlinks, and selected files above 500 MiB plus QE WFC/mixing-named files above 100 MiB. Exact hashes were read only for the 12 QE mixing candidates. Their later cleanup and its SQL updates are recorded in the execution addendum below.

- `/data/pclancy3/yi` had 20 immediate children: 17 directories and 3 regular files. Of the 17 directory roots, 16 non-Conda roots completed with exit 0 and a per-root completion marker. `.conda` was deliberately not rewalked because its package tree is expensive; prior fresh Conda cache manifests, exact cleanup receipts, and duplicate evidence were reused. The three top-level non-directory entries were explicitly skipped as non-directories.
- Root-level scan results: 16 completed directory roots, 1 Conda root reused from prior evidence, 3 non-directory children skipped, 0 incomplete roots among traversed roots. `flare-data` scan found 29 candidate records; `mylammps` found 1; `other` found 1; other completed non-Conda roots found none.
- Targeted exact `lstat` captured all 31 candidates with uid/gid, mode, logical/allocated size, mtime_ns, device, inode, and hardlink count. Errors: 0.
- The historical guarded deletion recorded no scheduler workdir overlap with the 12 selected paths and rechecked each file identity and SHA-256 immediately before unlink. Raw queue exports are excluded from this scope-limited repository package.
- No WFC candidate was found in the scanned non-Conda roots. The only QE restart candidates were 12 `sb2te3.mix1`–`mix4` mixing-history files in three completed calculations.

## Large candidate totals and immediate disposition

The 19 files above 500 MiB total **50,244,180,275 logical bytes** and **50,244,878,336 allocated path bytes**. All 12 QE mixing files total **5,352,688,896 logical bytes / 5,352,980,480 allocated path bytes**. All 31 candidates (including the mixing files) are in [large_candidate_disposition_20261123.tsv](large_candidate_disposition_20261123.tsv); each row records exact metadata, available SHA-256 provenance, lineage evidence, disposition, and next action.

Fresh SHA-256 was computed for the 12 mixing files with `O_RDONLY|O_NOFOLLOW`, chunked reads, and matching pre/post `fstat`; all were stable and their hashes are unique. The 19 >500 MiB files already had SHA-256 receipts from 2026-11-22. Their fresh exact `lstat` size, allocated bytes, owner, mode, nlink, mtime_ns, device, and inode all match those hash records, so the prior hashes still correspond to the current file identities. No matching SHA-256 duplicates were found among these 31 candidates. Five large result files have `nlink=2`; only one name per inode was found inside `/data/pclancy3/yi`, so the second hardlink path remains unmapped and removing an in-scope name would not free its inode while another link exists.

### QE mixing restart evidence

The three calculations have retained `relax.in`, `sbatch-relax.slurm`, `relax.out`, and `sb2te3.save/data-file-schema.xml`; each log reaches `JOB DONE.` and its date-bounded Slurm record says `COMPLETED 0:0`. However, all three logs also report `Maximum CPU time exceeded` first, and each Slurm stdout reports `Abort(1) ... MPI_Abort` during `pp.x`. The original scripts end with `date` and do not enable failure propagation, so a successful scheduler exit does not show that the relaxations converged or post-processing succeeded. Classify the historical PW outputs as time-limited partial/failure-mode evidence, not validated converged structures. The output trees contain no WFC files and no `charge-density.dat`; the 12 files are mixing histories. Base-spin job 7104884 and mag3 job 7105613 use the QE default fresh-start mode. Charged3-spin job 7167643 has `restart_mode='restart'`, so the reviewed clean PW recipe explicitly switches a copied input to `restart_mode='from_scratch'` and uses a new outdir. All three `pp.out` files end with `Error in routine chdens (1): wrong nx`; retain these as post-processing failure evidence and do not describe the PW+PP script as a successful end-to-end calculation. The exact QE 7.2 binary SHA-256 values, module path, and all three pseudopotential hashes are independently recorded. The fresh-start helper prepares output under `/scratch16/pclancy3/yi`; it checks source hashes and never edits originals or submits a job. The queue had no overlap at the earlier snapshot time, but a new live queue guard remains required immediately before deletion.

The focused external-consumer check scanned 184 readable text files under the
`2.Cr-in-gap` subtree (excluding `.conda`, `.git`, QE output/save directories,
binary/non-text files, and files above 5 MiB) and found no explicit run-path or
`sb2te3.mix1`–`mix4` references. A wider `/data/pclancy3/yi` scan stalled in
remote directory traversal and was stopped; therefore there is no mount-wide
negative search claim. The catalog initially had 985 artifacts under the
subtree but no direct run, recipe, job, or script link for the 12 candidates.
Their fresh-start lineage is registered in `qe_mix_reproduction_manifest_20261123.json` and the SQL catalog. Deleting the files removes the option to continue the exact time-limited historical states; the retained method starts a new trajectory.

### QE mixing cleanup execution — 2026-11-23

The SQL catalog now contains three `FAILURE_MODE` QE workflow runs, their
Slurm job IDs/accounting observations, exact input/script/PP/log/XML and
pseudopotential links, QE 7.2 binary hashes, reviewed fresh-start recipe links,
and quality gates. The new recipe was not submitted or run. It starts in a new
directory, forces `restart_mode='from_scratch'`, runs PW only, and retains the
historical 250,000-second limit; it cannot guarantee a converged replacement.
The runs are linked to campaign `MBREG-CR-IN-GAP-RELAX-202507`; no specific
publication figure or claim relation was established, so none is inferred.

Cleanup batch `RF-DATA-YI-CR-IN-GAP-MIX-CLEANUP-20261123` completed. The
remote guard recorded no workdir overlap with the 12 selected paths, then
rechecked the queue and all recorded file identity fields and SHA-256
immediately before each unlink. It deleted exactly the 12 staged paths
(5,352,688,896 logical bytes; 5,352,980,480 allocated bytes) and verified each
missing. The first remote attempt stopped before queue parsing because Rockfish
Python 3.6.8 does not accept `subprocess.run(text=...)`; its one-event receipt
and exact script are preserved separately and say no files were deleted. The
compatible rerun completed with 12 per-file deletion receipts and zero queue
overlaps. The catalog marks the cleanup batch `COMPLETED` and each removed
artifact location `REMOVED`.
The private historical queue snapshot is referenced by
`cleanup_execution_receipt.queue_snapshot_source_id`; the successful execution
receipt retains the 12 immediate pre-unlink checks. Raw queue rows are not part
of this repository package.

The historical cleanup receipt and catalog rows record the exact deleted paths and byte totals. Quota captures are excluded from this scope-limited package.

## Path-level next actions

1. **Regenerable restart files — completed:** the three time-limited PW run chains, failed PP derivatives, 12 exact mixing-file hashes, pseudopotentials, jobs, and reviewed but untested fresh-start recipe are registered. The original logs and XML remain partial/failure-mode evidence. The guarded scheduler-overlap check and immediate rehash/unlink workflow removed exactly the 12 `sb2te3.mix*` paths; no WFC cleanup was pending under the scanned roots.
2. **`flare-data/.../X-FORCE/.git/objects/pack/pack-261424fcac029ceb31a391c4bc87ac5457e35664.pack`:** keep; review refs/object reachability and backup options before any Git maintenance.
3. **`mylammps/.git/objects/pack/pack-a21eb80142455c386b435c37a9b9543ec5865d19.pack`:** keep; third-party repo and stale release branch require ownership/reproducibility review.
4. **Four band/save `atomic_proj.xml` files and `2.aimd/.../aimd.out`:** keep as result evidence; map their second hardlink names outside `/yi` before considering any path change.
5. **Collaborator dataset files `merged/all.extxyz` and `splits/train.xyz`:** preserve; establish source/split and downstream model consumers before deduplication or cleanup.
6. **`4.lmp/Rippling/.../neb_optimization.traj` and six `relax_maceMD_eval_*.xyz`:** preserve as simulation outputs until exact run scripts, Slurm lineage, and figure/model consumers are linked.
7. **`other/AI/results/checkpoint-2500/optimizer.pt`:** preserve until its training run and checkpoint consumer/replacement are identified.
8. **`.conda`:** no broad fresh root scan in this pass. Reuse the Nov 22 package-cache receipts; the 14.2 MB `libstdcxx-14.2.0-h8f9b012_2` cache subtree is already deleted and verified. The remaining 574,565,016-byte CUDA library SHA duplicate is a hardlink between the `cudatoolkit` cache and `deepmd2` environment plus a separate `mace` environment copy; deleting a cache name yields no inode space while the env link remains, so no safe duplicate cleanup is identified.

## Artifacts and limits

- Candidate roots, metadata manifests, status/exit/stderr markers: `root_*_large_candidates_20261123.jsonl.gz`, matching `.status.json` and `.stderr`, plus `candidate_scan_run_summary_20261123.json`.
- Exact large-candidate metadata: `candidate_targeted_lstat_20261123.jsonl`; exact mix hashes: `mix_group_*_sha256_20261123.normalized.jsonl` plus original raw stdout and status files.
- QE reproduction evidence: exact per-file path/hash/run/job matrix `qe_mix_file_lineage_20261123.tsv`; reviewed PW-only fresh-start recipe `qe_mix_fresh_start_recipe_proposal_20261123.md`; machine-readable chain `qe_mix_reproduction_manifest_20261123.json`; helper `prepare_qe_mix_fresh_start.py`; independent review `QE_MIX_RECIPE_INDEPENDENT_AUDIT_20261123.md`; stable retained-file hashes and exact QE binary hashes; UTC recapture `qe_candidate_sacct_utc_20261123.psv` (the earlier timezone-naive PSV remains separately identified); targeted `pp.in`/`pp.out`/input/XML metadata; `recipe_evidence_20261123.jsonl`; `qe_*_out_listing_20261123.tsv`; and `qe_*_slurm_files_20261123.tsv`.
- The latest scoped metadata refresh and the SQL snapshot record supersede this 2026-11-23 bounded inventory for current file counts. This report remains the historical QE mixing cleanup and run-lineage record; its raw queue export and mixed-root Git inventory are excluded from this package.
- The `.conda` child was not freshly enumerated, and this pass did not inspect payload contents or hash the non-QE large result files anew. Prior content hashes are reused only where fresh exact metadata matches. The inventory portion itself was read-only; the documented QE cleanup subsequently updated the shared SQLite and removed only the 12 registered mixing intermediates.
