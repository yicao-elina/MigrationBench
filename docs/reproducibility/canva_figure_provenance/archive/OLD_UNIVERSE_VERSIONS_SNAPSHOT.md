# Old Universe Figure Versions — Pre-cleanup Snapshot

This manifest is committed immediately before removing superseded Universe
Figure 1 assets. The files listed below are preserved in the preceding
snapshot commit for audit/recovery; the R06 clean raw-PCA package is the
current canonical version and is not part of this archive.

## Snapshot boundary

- Repository branch: `codex/canva-figure-provenance`
- Pre-cleanup HEAD: `2714991`
- Snapshot purpose: preserve superseded Universe Figure 1 exports and
  exploratory embedding variants before cleanup.
- The tracked R03 final assets were already present at `2714991`; the
  snapshot commit records their exact paths and hashes alongside the
  untracked R03/R04/R05 assets added in the snapshot commit.

## Superseded assets and SHA256

| SHA256 | Path |
|---|---|
| `d38d676edb0d18a1ea99795e46ae9031fa66a00b0f8f2f472e0a585e97409590` | `R03_typography18pt/figures/figure01_task_conditioned_universe.svg` |
| `4853ca3beae3ec58d01bbbc9e0c02c08da618d1f6d2832fd6c0b85c6022791cc` | `R03_typography18pt/figures/figure01_task_conditioned_universe.png` |
| `3e1181e19f080f78b270315e794aa7ff4f40d3a24e6e0b674359251dd097457f` | `R03_typography18pt/figures/figure01_task_conditioned_universe.pdf` |
| `582bb8e0f78beb2acc4cbfd61c355e5d056f1c190854ffa2d798abeb8fec89ef` | `figures/figure01_task_conditioned_universe.svg` |
| `73087bfd973bc72e3312dd5c38ec8d589c74c830fb3af0d9edd824b1db588adc` | `figures/figure01_task_conditioned_universe.png` |
| `3ae9a12b90d9cb9991e4bbc8dbf414d36d6cddc050411cb077d56c7b50d1bca4` | `figures/figure01_task_conditioned_universe.pdf` |
| `61aadca34be00c0ebffb7f4610067080b76498082e254b9df04be30a697fd39b` | `R03_typography18pt_final/figures/figure01_task_conditioned_universe.svg` |
| `4853ca3beae3ec58d01bbbc9e0c02c08da618d1f6d2832fd6c0b85c6022791cc` | `R03_typography18pt_final/figures/figure01_task_conditioned_universe.png` |
| `15a00ae86d0de4dc2655f6abff242a623c6b3a4dd3e189b5534a066c37035442` | `R03_typography18pt_final/figures/figure01_task_conditioned_universe.pdf` |
| `bcc0e59a150e87d76f50b0bcfa63a22c6893aba266ae4272462b4b63ccd69c4b` | `R04_task_conditioned_ood/figures/figure01_task_conditioned_universe.svg` |
| `16f221e3170b0d50940ab318efea47992297a5990879e7f42cf517e793b34c8c` | `R04_task_conditioned_ood/figures/figure01_task_conditioned_universe.png` |
| `e82e20ab6161daab3f857153a9cf8a682956383258d85968374d9ca79cf84f26` | `R04_task_conditioned_ood/figures/figure01_task_conditioned_universe.pdf` |
| `38729f20cd2ac8dd10202c6a7f621062a35f1065449edef246ad548b94a591ac` | `R04_task_conditioned_ood/FIGURE01_A_QA.md` |
| `b13b546a134b04b149f1f831a48d4cf992437863604816aa9ecc36b683ebe3a4` | `R04_task_conditioned_ood/manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json` |
| `b86aee0d546915500529d0e755774223be3038cbc97068094ba7c87cbfca5966` | `R05_raw_pca/figures/figure01_task_conditioned_universe.svg` |
| `f0dc28b9e56b916ea311793a90cce6aa607c189de535eeec5d261c1cdfe401a1` | `R05_raw_pca/figures/figure01_task_conditioned_universe.png` |
| `74c1874c374a1effee841033af73e6d1b9f13a4de69b434c478b8423f4079471` | `R05_raw_pca/figures/figure01_task_conditioned_universe.pdf` |
| `131c13e71c6e41f77deafb5555675059121d5c6637eaa738cf547fab19754345` | `R05_raw_pca/manifests/ATOMIC_MIGRATION_FIGURE_MANIFEST.json` |
| `c0d9f70774a7e8514845fde9b74220d56fd7ec7760954b08a98b4f87c57766bb` | `R05_embedding_comparison/figures/figure01_embedding_comparison.svg` |
| `1f11e77189f44868b4f799246022f0bd0968ccf8cc2a837742a68402530cdb4f` | `R05_embedding_comparison/figures/figure01_embedding_comparison.png` |
| `b0013158741560be0ad470deedf0198d886dcd5071276aeca88a887acc4ed4ef` | `R05_embedding_comparison/figures/figure01_embedding_comparison.pdf` |
| `df4b9bd591e71d16f4daa692823a123bc60b2d564ecd6fa22c445ddceafeac30` | `R05_embedding_comparison/FIGURE01_EMBEDDING_QA.md` |
| `4bb88a0e9eae3c38638a2a0632735361ab68bb279c6f68a0b1ba81f5dc808ebe` | `R05_embedding_comparison/manifests/FIGURE01_EMBEDDING_COMPARISON.json` |
| `74118a5f5310a27253ca3490ff6635e462aa58edac6af6c8a47f01a25bd6d2de` | `scripts/build_figure01_embedding_comparison.py` |

The seven raster candidates under `R05_embedding_candidates/` and their
`candidate_manifest.json` are also included in the snapshot commit; they are
exploratory candidates and are intentionally removed from the active tree.

