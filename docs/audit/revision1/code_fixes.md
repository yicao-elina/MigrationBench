# LP-5 Code Fixes — Audit Report (revision1-fixes branch)

Repo: `MigrationBench` (`/Users/alina/Library/CloudStorage/OneDrive-JohnsHopkins/Research/25NeurIPS-AI4MAT/MigrationBench`)
Branch: `revision1-fixes` (5 commits, `git diff main...revision1-fixes` archived in `code_fixes.diff` alongside this report).
Date: 2026-09-06.

Reviewer items fixed: code-critical R1 (fabricated NEB forces), R3 (train/test leakage via ungrouped split), hygiene R2 (stale MD params/comment), R5 (zero force-sensitivity stub column in probe features).

---

## Fix 1 — Remove fabricated zero forces from NEB extxyz writers (R1)

### `examples/Sb2Te3_Cr_doped/03_neb/neb_data_collector.py`

**Before** (`write_enhanced_xyz`, ~lines 442-449):
```python
# Generate placeholder forces (zeros for now)
forces = [(0.0, 0.0, 0.0) for _ in range(natoms)]
...
properties = "Properties=species:S:1:pos:R:3:forces:R:3"
```
followed by atom lines with `{fx} {fy} {fz}` appended (all zeros).

**After**: the placeholder-force block is deleted; `Properties=species:S:1:pos:R:3`; atom lines contain species + positions only, with an explanatory comment:
```python
# NOTE: forces are intentionally omitted. QE NEB outputs parsed
# here do not provide per-atom forces; writing fabricated zero
# forces would silently corrupt downstream MLFF training data.
```
Energies (real, parsed from QE `neb.out`) and all provenance metadata (`energy=`, `neb_image=`, `path_name=`, activation energies, etc.) are unchanged.

### `examples/Sb2Te3_Cr_doped/03_neb/neb_to_extended_xyz.py`

**Before**: `generate_fake_forces(natoms)` returned `[(0.0, 0.0, 0.0) ...]`; `include_forces` defaulted to `True`; `--no-forces` was the opt-out.

**After**:
- `generate_fake_forces` deleted.
- CLI flipped to opt-in: `--include-forces` (replaces `--no-forces`); forces are OFF by default.
- `write_extended_xyz(..., include_forces=False)` raises immediately if forces are requested:
```python
if include_forces:
    raise NotImplementedError(
        "Force output was requested, but no real force source is available. "
        "QE NEB outputs (neb.out) do not contain per-atom forces and this "
        "converter does not read any force file. Refusing to write "
        "fabricated (zero) forces. Provide real forces or omit the "
        "--include-forces flag."
    )
```
- Bonus minimal fix enabling verification: the iteration-header regex now tolerates QE whitespace variants (`-+ iteration (\d+) -+` → `-+\s*iteration\s+(\d+)\s+-+`); the old pattern matched nothing on the shipped `neb.out` files, so the converter could not run at all on them.

### Verification
- Ran the collector on a local fixture for `2D_neb/neb_1` (temp output dir). Note: the local snapshot of `neb_1` contains only `neb.out` + `sb2te3.dat` (no `sb2te3.xyz` exists anywhere in the repo), so a 10-image × 81-atom `sb2te3.xyz` geometry fixture was synthesized in a temp dir. Energies are parsed from the real `neb.out`, not the fixture.
- Output extxyz `Properties=species:S:1:pos:R:3`; token `forces` occurs 0 times in the output.
- Energies in the generated xyz match `mlff_training_data/processing_report.json` entry for `neb_1` exactly: 10 images, `energy_range = [-177387.7877832, -177386.6029654]` (run == reference, asserted to 1e-6). `MATCH-OK`.
- `neb_to_extended_xyz.py --help` works and documents the opt-in flag.
- Default run on `neb_1` inputs: succeeds, `Properties=species:S:1:pos:R:3`, zero `forces` tokens, no zero-force triples; `neb_energy_eV` values identical to the collector's.
- Run with `--include-forces`: clear `NotImplementedError` message, and **no output file is written**.

## Fix 2 — Group-aware data splitting (R3)

`examples/Sb2Te3_Cr_doped/04_mlff_training/data/split_data.py`

**Before** (lines 52-66): flat `np.random.shuffle` of frame indices, partition by count — frames from the same NEB path/trajectory could land in both train and test (near-duplicate leakage).

**After**: new helpers `get_group_ids` (reads `path_name`, then `trajectory`, then `group` from each frame's `atoms.info`, i.e. the extxyz comment line; falls back to `file=<basename>` per-file grouping when no tag is present) and `split_groups` (seeded shuffle of groups, greedy assignment of whole groups to the partition with the largest remaining frame deficit; asserts pairwise-disjoint group sets across train/valid/test). Split sizes are then realized at group granularity. CLI interface unchanged; a warning is printed if a partition is empty because too few groups exist for the requested ratios.

### Verification (synthetic test, ase 3.25.0 env)
- 2 groups × 5 frames, tags in comment lines, 60/20/20 → train={pathB}, valid={pathA}, test=∅; **no group spans partitions** (asserted). Empty-partition warning printed (2 groups cannot fill 3 partitions).
- 10 groups × 5 frames, 60/20/20 → exactly 6/2/2 groups per partition, zero overlap (asserted).
- Untagged file → per-file fallback grouping, no crash.
- Result: `ALL-GROUP-SPLIT-TESTS-PASSED`.

## Fix 3 — MD eval parameters/comment (R2 hygiene)

`results/MD/2-naive-fine-tuning/mace_md_eval_0814.py` (was lines 359-360)

**Before**:
```python
steps = 2000               # MD steps
interval = 5            # sample every 200 steps (consistent with comment)
```
**After**:
```python
steps = 100000             # MD steps
interval = 200             # sample every 200 steps
```

## Fix 4 — Delete zero force-sensitivity stubs (R5)

Removed the placeholder `finite_diff_force_sensitivity` (always-zero column), its `np.hstack`/append concatenation, and the now-unused `FINITE_DIFF_DELTA` constant from:
- `results/t-SNE/3d/tsne.py` (was lines 208-217, usage 252-253 → `feats = np.hstack([E, F, neigh, soap_pca])`)
- `results/t-SNE/phate-tsne/tsne.py` (was lines 131-132, usage 160-161)
- `results/t-SNE/analysis_Cr_Doped_600K/mace_probe_blackbox0818.py` (was lines 171-180, usage 223-225)

Downstream feature matrices shrink by exactly one (always-zero) column; all other descriptors unchanged. Note for LP-2: previously-saved `*_probe_features.csv` files (e.g. 404×6192) contain the stub as their last column; regenerated CSVs will be one column narrower.

## Fix 5 — Port working SHAP scripts

`scripts/postprocessing/` previously contained only 0-byte stubs.
- `shap_analysis.py` ← ported from `25NeurIPS-AI4MAT/0929-MultiT/1-shap-data.py` (MACE total/binding energy predictions) with a provenance header; replaces the 0-byte stub.
- `shap_plot_jhu.py` ← ported from `25NeurIPS-AI4MAT/0929-MultiT/2-shap-plot-jhu.py` (direct-SOAP surrogate + SHAP, JHU-styled figures).
- `README.md` documents provenance and notes the hardcoded JHU cluster paths (`/data/pclancy3/yi/flare-data/1-Cr-Sb2Te3/...` in `MODEL_PATHS`, valid only on the cluster; `OUTPUT_DIR='direct_soap_shap_analysis_JHU'`).

## Global verification summary

| Check | Result |
|---|---|
| `grep -rn "0.0, 0.0, 0.0" examples/` | no matches (exit 1) — no fabricated-force generation remains |
| `grep` for `generate_fake_forces` / `finite_diff_force_sensitivity` / `FINITE_DIFF_DELTA` / `no-forces` in `examples/ results/ scripts/` | no matches |
| Collector on `neb_1` fixture | extxyz without `forces` key; energies match `processing_report.json` exactly (`MATCH-OK`) |
| `neb_to_extended_xyz.py --help` | works; `--include-forces` documented |
| Converter default run | no forces in output; energies correct |
| Converter `--include-forces` | `NotImplementedError` with clear message; no output file written |
| Grouped split synthetic tests | 3/3 pass; no group spans partitions; exact 60/20/20 at group granularity for 10-group case |
| `python3 -m py_compile` on all 9 modified/added files | clean (`PYCHECK-OK`) |

Caveat: the local `2D_neb/neb_1` snapshot lacks `sb2te3.xyz` (no `.xyz` exists in the repo), so the collector verification used a synthetic geometry fixture in a temp dir; energies (the audited quantity) come from the real `neb.out`. The pre-existing summary-report crash when `output_base_directory` does not exist was sidestepped by pre-creating the temp output dir and is unrelated to these fixes.
