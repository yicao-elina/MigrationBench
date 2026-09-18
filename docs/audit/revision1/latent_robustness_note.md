# LP-2 audit note — latent-space robustness (Cr-doped Sb2Te3, 600 K)

Script: `Revision1/scripts/latent_robustness.py` (conda env `ml`,
sklearn 1.8.0, PHATE 2.0.0).
Raw data: `MigrationBench/results/t-SNE/analysis_Cr_Doped_600K/Cr_Doped_Sb2Te3_600K_probe_features.csv`
(404 x 6192; blocks of 101 frames in order Foundation, Scratch, FT-600K,
FT-MultiT, per `mace_probe_blackbox0818.py` lines 56-62). The trailing all-zero
force-sensitivity stub column was asserted all-zero (sum=0.0) and
dropped -> 404 x 6191.

## Answer: are the models separable in the ORIGINAL descriptor space?

**Yes, but with a scale caveat.** In the ORIGINAL 6191-dimensional descriptor space, the four model clouds are moderately separable overall (euclidean silhouette = 0.333; cosine = 0.277), and the Foundation model in particular is almost perfectly separated from the rest (per-model silhouette 1.000 euclidean / 0.990 cosine). However, this raw-space separation is largely scale-driven: the descriptor vector mixes the predicted total energy (~10^3 eV offsets between models), forces, RDF histograms and PCA'd SOAP components, and a 95%-variance PCA of the raw features retains only 1 component(s) — effectively the energy axis. After removing per-descriptor mean/scale offsets (z-scoring), the overall euclidean silhouette drops to -0.019 (PCA-95% of the z-scored space, 364 components: -0.026), i.e. the residual clouds around their respective means strongly overlap. So the honest answer is: models ARE distinguishable in the original descriptor space, but the distinguishability comes predominantly from global feature offsets (energy scale and systematic shifts), not from cleanly separated local geometry — and the 2D t-SNE in old Fig 5(d) visually overstates this (overall 0.409 vs 0.333 raw).

Key numbers (see `data_processed/latent_silhouette_all_spaces.csv`):

| space | metric | overall | Foundation | Scratch | FT-600K | FT-MultiT |
|---|---|---|---|---|---|---|
| original 6191-d | euclidean | 0.333 | 1.000 | 0.354 | -0.027 | 0.003 |
| original 6191-d | cosine | 0.277 | 0.990 | 0.151 | -0.020 | -0.012 |
| original 6191-d, z-scored | euclidean | -0.019 | 0.048 | -0.042 | -0.041 | -0.042 |
| PCA 95% raw (1-d) | euclidean | 0.486 | 1.000 | 0.963 | -0.032 | 0.012 |
| PCA 95% raw (1-d) | cosine | 0.250 | 1.000 | 0.000 | 0.000 | 0.000 |
| PCA 95% z-scored (364-d) | euclidean | -0.026 | 0.070 | -0.059 | -0.058 | -0.059 |
| 2D t-SNE p=30 s=42 (old Fig 5d repro) | euclidean | 0.409 | 0.895 | 0.798 | -0.038 | -0.019 |
| 2D t-SNE, p in {5,30,50} x seeds {0,1,2} | euclidean | 0.397 ± 0.027 | — | — | — | — |
| PHATE 2D, seeds {0,1,2} | euclidean | 0.473 ± 0.000 | — | — | — | — |

## Caveats (must accompany any claim)

1. The descriptor vector includes the **predicted total energy** (column 0,
   ~thousands of eV) alongside forces, RDF histograms and PCA'd SOAP
   components. In raw euclidean distance the energy column dominates
   between-model distances (the raw 95%-variance PCA retains only
   1 component), which is why the cosine, z-scored and
   z-scored-PCA variants above are the honest scale-free checks: they show
   much weaker, partially overlapping structure.
2. The 2D t-SNE numbers in the original Fig 5(d) overstate separability
   relative to the raw descriptor space (t-SNE is designed to tear clusters
   apart) and vary with perplexity (overall silhouette spans
   0.361–0.420 across the 9 settings;
   trustworthiness 0.963–0.973).
   Note: at fixed perplexity, seeds {0,1,2} produce bitwise-identical
   silhouettes — sklearn's default PCA initialization removes the seed
   dependence, so the reported mean ± std (0.397 ± 0.027)
   reflects perplexity sensitivity only (a seed-dependence upper bound).
3. FT-600K is quarantined in the revision (2000-step MD run,
   [INVALID-pending JB-6]); silhouette values including it are shown only for
   continuity with the old figure.
4. Numerics: sklearn 1.8.0 silently downcasts silhouette distance
   aggregation to float32, injecting up to ~1e-7 absolute noise. All numbers
   reported here come from an exact float64 reference implementation; two
   independent float64 implementations (naive loop vs bincount-vectorized)
   agree to < 1e-10 in every space (asserted in the script), and
   sklearn's float32-path deviation is printed as a diagnostic.

## Reproduction check (verification gate)

Old pipeline re-run locally (perplexity 30, random_state 42) vs values logged
in `slurm.12943857`: overall 0.409 vs 0.412; Foundation
0.895 vs 0.896; Scratch
0.798 vs 0.795; FT-600K
-0.038 vs -0.064; FT-MultiT
-0.019 vs 0.021. Max |delta| =
0.040 -> gate ±0.05: **PASS**.
(Residual drift is expected from the sklearn version difference between the
cluster run and the local env; Barnes-Hut t-SNE optimization paths are not
bit-reproducible across versions.)

All seeds: t-SNE reproduction seed 42; sensitivity seeds
(0, 1, 2); PHATE seeds (0, 1, 2); trustworthiness n_neighbors =
5.

## Proposed replacement sentence for §3.4 (hedged, reviewer-I6 compliant)

> "The clustering of model representations seen in the t-SNE projection is
> consistent with genuinely distinct learned representations: the four training
> paradigms remain distinguishable in the original 6,191-dimensional descriptor
> space (overall silhouette 0.33) and across t-SNE
> perplexity/seed choices (0.40 ± 0.03), although a
> scale-controlled analysis shows the separation is driven largely by global
> feature offsets rather than disjoint local structure (z-scored silhouette
> -0.02), and we therefore interpret the embedding as supporting
> — not proving — representational differences (Fig. S-latent)."

## Outputs

- `data_processed/latent_silhouette_all_spaces.csv`
- `data_processed/latent_embedding_sensitivity.csv`
- `tables/silhouette_descriptor_space.tex`
- `figures/figS_latent_robustness.pdf`
