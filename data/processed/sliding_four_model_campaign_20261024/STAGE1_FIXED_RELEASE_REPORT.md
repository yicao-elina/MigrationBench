# Stage 1 fixed-path release report — interlayer sliding Fig. 4b/c

Generated UTC: `2026-10-24T23:05:38.934648+00:00`
Campaign: `sliding_four_model_campaign_20261024`

## Release decision

- M0 input validation: **PASS**
- A0 legacy runner self-check: **PASS**
- Tier-1 fixed-path collection: **PASS**
- Tier-1 dense diagnostic collection: **PASS**
- Fixed-path current rows passed: **30/30**
- Dense diagnostic rows passed: **30/30**
- Fixed-path finite model profiles collected: **30**

The two panels are released as fixed-path relative-energy diagnostics. They are not self-relaxed NEB barriers and are not presented as validated migration barriers.

## Reproducibility anchors

- Input hashes: [`manifests/INPUT_HASHES.tsv`](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/manifests/INPUT_HASHES.tsv)
- Model registry used: [`manifests/MODEL_REGISTRY_USED.tsv`](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/manifests/MODEL_REGISTRY_USED.tsv)
- DFT reference: `COMPLETED_BUT_SCF_WARNING`; `Quantum ESPRESSO NEB, no-CI, 10 images, PBE/DFT-D3; FR pseudopotential filenames present, explicit SOC flags not independently confirmed`
- Slurm submissions: a0 `32455511`, a0 `32455545`, fixed `32455619`, dense `32455870`
- Runner: `scripts/eval_fixed_path.py`; no geometry optimization is performed

## Model accounting

- Current profiles in the panels: `30` rows across `2` systems.
- Foundation checkpoints are shown individually because they are checkpoint variants, not three random-seed replicates.
- FT-600K and FT-MultiT are summarized with three training seeds each.
- Scratch-600K and Scratch-MultiT remain separate max_L=2 subfamilies; they are not collapsed into one Scratch mean.
- The legacy `dc41a989...` Foundation checkpoint is retained for A0 self-check only and is excluded from the current panels and current conclusions.

## Panel artifacts

- `(b)` PDF: [fig4b_interlayer_sliding_pristine_fixed_path_20261024.pdf](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/figures/fig4b_interlayer_sliding_pristine_fixed_path_20261024.pdf) — SHA256 `4c0357f693cbd14c0064127de6479c5f648d61b4778d10441d895707edd82d21`
  PNG QA render: `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/figures/fig4b_interlayer_sliding_pristine_fixed_path_20261024.png`
- `(c)` PDF: [fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024.pdf](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/figures/fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024.pdf) — SHA256 `06005177b7519c94fdd3b0f53a8a7be8370aa0345965f9f7d9cd03e29cbca8da`
  PNG QA render: `/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/figures/fig4c_interlayer_sliding_cr_in_gap_fixed_path_20261024.png`

## Scientific boundary and open items

- DFT/QE reference records remain `COMPLETED_BUT_SCF_WARNING`; the warning-bearing reference is preserved rather than silently upgraded.
- FR pseudopotential filenames are present in the retained inputs, but explicit noncollinear/spin-orbit flags were not independently confirmed in the archived input/output; no verified SOC claim is made here.
- The fixed-path comparison tests the same retained geometries for every model. A future self-relaxed NEB campaign must be reported as a separate observable and must not be pooled with these profiles.

## Linked machine-readable evidence

- Gate: [`manifests/STAGE1_GATE.json`](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/manifests/STAGE1_GATE.json)
- Expected/actual audit: [`results/tables/EXPECTED_vs_ACTUAL.tsv`](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/tables/EXPECTED_vs_ACTUAL.tsv)
- Per-model metrics: [`results/tables/fixed_metrics_per_model.csv`](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/tables/fixed_metrics_per_model.csv)
- Family summary: [`results/tables/fixed_family_summary.csv`](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/tables/fixed_family_summary.csv)
- Discussion: [`results/FIXED_PATH_DISCUSSION.md`](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/FIXED_PATH_DISCUSSION.md)
- Caption suggestion: [`results/FIXED_PATH_CAPTION_SUGGESTION.md`](/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/sliding_four_model_campaign_20261024/results/FIXED_PATH_CAPTION_SUGGESTION.md)
