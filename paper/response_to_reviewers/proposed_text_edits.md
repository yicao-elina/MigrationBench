# Proposed Manuscript Text Edits — Revision 1 (DRAFT, 2026-09-06)

All OLD passages are verbatim quotes from the submitted `sn-article-final.tex` at the stated line
numbers. NEW passages are draft replacements; `[NUMBER-PENDING: JB-x]` / `[NUMBER-PENDING: LP-x]`
mark values awaiting cluster (JB) or local-analysis (LP) completion.
**Do not apply in place until pending numbers are filled (per LP-7 instruction).**
Cite keys for the three new references are defined in `Revision1/tables/new_references.bib`:
`alghamdi2025maceLiF`, `bheemaguli2026mlipBarriers`, `deng2025softening`.

---

## E1. Abstract closing sentence — rescope "generalizable standard" (reviewer I4)

**Location**: Abstract, L153.

**OLD**:
```
This work establishes migration-based non-equilibrium probes as a data-efficient, generalizable standard for MLFF evaluation and provides actionable guidance for robust MLFF development.
```

**NEW**:
```
This work demonstrates migration-based non-equilibrium probes as a data-efficient candidate framework for MLFF evaluation, developed and validated on one representative material system, and provides actionable guidance for robust MLFF development. Whether this framework generalizes across chemistries and architectures remains an open question that we identify as a priority for future cross-system validation.
```

---

## E2. Introduction research question 2 — hedge presumed generality (reviewer I4)

**Location**: Introduction, itemized questions, L195.

**OLD**:
```
\item Can non-equilibrium probes generated from methods, such as NEB, provide a generalizable and efficient benchmark for specialist versus generalist models?
```

**NEW**:
```
\item Can non-equilibrium probes generated from methods, such as NEB, provide an efficient benchmark for specialist versus generalist models, and to what extent does such a benchmark generalize beyond a single material system?
```

---

## E3. §3.2 reference-value sentence — single canonical DFT reference (reviewer C1)

**Location**: §3.2 "Local Migration Events", L272.

**OLD**:
```
Our DFT calculations serve as the ground-truth reference. This establishes a (reasonably low) migration energy barrier ($E_{\mathrm{a}}$) of 0.34~eV for this process. The results, presented in Fig.~\ref{fig:barrier}d, reveal a clear hierarchy in performance and highlight the considerable impact of training strategy on the prediction of kinetic barriers.
```

**NEW** (0.336 value verified by LP-1 against the raw `neb.out`; keep fixed):
```
Our DFT NEB calculations serve as the ground-truth reference. For the in-gap pathway (path 1-4, defined in the SI) they establish a migration energy barrier ($E_{\mathrm{a}}$) of 0.336~eV; a per-pathway convergence audit of all DFT NEB calculations, including the canonical reference value used for each pathway, is provided in Table~SX and Fig.~SX in the SI. The results, presented in Fig.~\ref{fig:barrier}d, reveal a clear hierarchy in performance and highlight the considerable impact of training strategy on the prediction of kinetic barriers.
```

---

## E4. §3.2 — side-by-side Fig 3d / Fig S3 paragraph + seed statistics (reviewer C1, C3)

**Location**: new paragraph inserted in §3.2 "NEB Stability as an Additional Robustness Criterion",
immediately after L307 paragraph.

**OLD**: (no existing text — insertion)

**NEW**:
```
\paragraph{Fixed-geometry versus self-consistent NEB evaluation}
We stress that the fixed-geometry single-point evaluation along DFT-relaxed NEB images
(Fig.~\ref{fig:barrier}d) and the self-consistent MLFF NEB optimization (Fig.~S3) measure different
things and are reported here side-by-side for the same models, the same pathway, and the same DFT
reference (0.336~eV, Table~SX). Fixed-geometry evaluation freezes the band on DFT-relaxed
geometries and isolates pure energetic accuracy; self-consistent NEB additionally relaxes the images
under the model's own forces, so it probes both energetics and force consistency. Under the unified
re-evaluation (same checkpoints, same pathway, same reference), the fixed-geometry barrier errors are
Foundation [NUMBER-PENDING: JB-5]~eV, Scratch [NUMBER-PENDING: JB-5]~eV, FT--600K
[NUMBER-PENDING: JB-5]~eV, and FT--MultiT [NUMBER-PENDING: JB-5]~eV, whereas the self-consistent NEB
barriers are [NUMBER-PENDING: JB-5]~eV, respectively. The quantitative offset between the two
protocols for the same checkpoint—most visibly the relaxation of the foundation model's barrier
toward the DFT reference once images are allowed to relax—explains the differing verdicts and yields
a single, consistent per-model ranking under each protocol. Uncertainty from training-seed variation
is reported as mean $\pm$ std over three seeds (seeds 123, 234, 345): in-gap barrier errors are
Scratch [NUMBER-PENDING: JB-2]~eV and FT--600K [NUMBER-PENDING: JB-2]~eV, and deep-penetration
barrier errors are Scratch [NUMBER-PENDING: JB-2]~eV and FT--600K [NUMBER-PENDING: JB-2]~eV
(Table~S1, extended).
```

**Also revise L307 sentence on the 0.41 eV figure** — OLD (within L307 paragraph):
```
The foundation model converges smoothly to a physically meaningful MEP with a barrier of 0.41~eV, as indicated by well-behaved energy and force evolution.
```
**NEW**:
```
The foundation model converges smoothly to a physically meaningful MEP with a barrier of 0.41~eV against the canonical DFT reference of 0.336~eV for the same pathway (path 1-4), as indicated by well-behaved energy and force evolution.
```

---

## E5. §3.4 — "direct mechanistic explanation" → "consistent with"; original-space silhouette + sensitivity (reviewer I6, repo R4/R5)

### E5a. L332 (silhouette-space caveat)

**OLD** (L332, opening sentences):
```
The $t$-SNE projection (Fig.~\ref{fig:tsne-phate}a) confirms that models trained with different strategies learn qualitatively distinct encodings. The MACE Foundation and MACE Scratch models occupy well-separated regions of the latent space, a finding quantitatively supported by their high average silhouette scores (Fig.~\ref{fig:tsne-phate}d).
```

**NEW**:
```
The $t$-SNE projection (Fig.~\ref{fig:tsne-phate}a) indicates that models trained with different strategies learn qualitatively distinct encodings. The MACE Foundation and MACE Scratch models occupy well-separated regions of the latent space. Because silhouette scores computed on low-dimensional embeddings conflate projection artifacts with true separability, we quantify separability in the original descriptor space: the average silhouette computed on the full atomic-environment feature vectors is [NUMBER-PENDING: LP-2] (per-model values in Table~SX), compared with [NUMBER-PENDING: LP-2] in a 95\%-variance PCA space; the embedding-space values reported previously are retained in the SI only as an explicitly flagged invalid baseline.
```

### E5b. L338 (mechanistic-explanation rescope + sensitivity)

**OLD** (L338, first sentence):
```
This geometric difference in the latent space provides a direct mechanistic explanation for the models' performance on the diffusion task.
```

**NEW**:
```
This geometric difference in the latent space is consistent with the models' differing performance on the diffusion task, though low-dimensional projections alone cannot establish a mechanistic link. Across t-SNE perplexities $\{5, 30, 50\}$ and three random seeds, the qualitative separation persists (silhouette [NUMBER-PENDING: LP-2], mean $\pm$ std across settings; embedding trustworthiness [NUMBER-PENDING: LP-2]; Fig.~SX), and PHATE embeddings are comparably stable across seeds ([NUMBER-PENDING: LP-2]).
```

### E5c. Fig. caption L347 (silhouette panel description)

**OLD** (L347):
```
\textbf{(d)}~Average silhouette scores (from $t$-SNE) quantify the representational dissimilarity. The results demonstrate that fine-tuning succeeds by constraining a generalist representation to the specific physical manifold of the target system, a process that regularizes the model and enables accurate prediction of complex dynamics like diffusion.}
```

**NEW**:
```
\textbf{(d)}~Average silhouette scores computed in the original descriptor space quantify the representational dissimilarity; values computed on the 2D embedding are shown in the SI only as a flagged invalid baseline. The results are consistent with fine-tuning constraining a generalist representation to the specific physical manifold of the target system, a process that regularizes the model and is associated with accurate prediction of complex dynamics like diffusion.}
```

---

## E6. §3.5 — surrogate-scoped SHAP claims, per-model R², perturbation test (reviewer I6)

### E6a. L359 (surrogate fidelity statement → explicit R²)

**OLD** (within L359):
```
This approach allows us to interpret the complex MACE potential by analyzing a simpler, yet highly faithful, model of its error behavior (see Fig. S4, for performance validation).
```

**NEW**:
```
This approach allows us to interpret the complex MACE potential by analyzing a simpler model of its error behavior, whose fidelity we quantify explicitly: the surrogate's held-out $R^2$ scores are [NUMBER-PENDING: JB-8] (Foundation), [NUMBER-PENDING: JB-8] (Scratch), [NUMBER-PENDING: JB-8] (FT--600K), and [NUMBER-PENDING: JB-8] (FT--MultiT) (see Fig. S4 and Table~SX for performance validation). All conclusions below are statements about the surrogate's model of MACE's errors, and are only as strong as these $R^2$ values permit.
```

### E6b. L365 ("internalize" → surrogate-scoped)

**OLD** (within L365):
```
The FT–Multi-T model’s ability to consistently isolate and amplify this descriptor—also identified as part of the stable consensus set—demonstrates that fine-tuning enables the model to internalize a physically coherent representation of the potential energy surface. Conversely, the scratch model’s diffuse landscape and heavy reliance on non-consensus descriptors reflect a relatively unstable and poorly structured inductive bias, explaining its inability to generalize to critical tasks such as resolving transition-state energetics or maintaining NEB stability.
```

**NEW**:
```
For the FT–Multi-T model, the surrogate's error predictions are most sensitive to this descriptor—also identified as part of the stable consensus set—which is consistent with fine-tuning producing a more physically structured error landscape. Conversely, the scratch model's surrogate distributes importance diffusely over non-consensus descriptors, a pattern consistent with a less structured error model and with the scratch model's observed failures on transition-state energetics and NEB stability. As a direct check beyond the surrogate, we perturbed MACE's own inputs along the dominant Cr--Cr structural mode (constrained Cr--Cr distance scans on held-out configurations): the ordering of MACE's own energy response [NUMBER-PENDING: JB-8] [agrees / partially agrees] with the surrogate's SHAP ranking, lending [NUMBER-PENDING: JB-8]-quantified support to the feature-level interpretation.
```

### E6c. Fig. caption L376 ("orders of magnitude" wording kept only if R² supports)

**OLD** (within L376):
```
The analysis highlights the FT--Multi-T model's reliance on the \texttt{Cr-Cr\_n13\_l3} feature, which is orders of magnitude more important than any other feature.
```

**NEW**:
```
The analysis highlights that the surrogate error model for FT--Multi-T is dominated by the \texttt{Cr-Cr\_n13\_l3} feature, whose mean absolute SHAP value is [NUMBER-PENDING: JB-8]-fold larger than the next-ranked feature (surrogate $R^2$ values reported in Table~SX).
```

---

## E7. Conclusions — hedged outlook (reviewer I4)

### E7a. L386 (opening conclusion)

**OLD**:
```
We benchmarked specialist (from-scratch) and generalist (foundation-based) MLFFs for Cr-doped \ce{Sb2Te3} using non-equilibrium configuration probes as a unified framework for evaluating both interpolation and extrapolation behavior.
```

**NEW** (insert hedged scope sentence after the unchanged first clause — minimal edit of the same sentence):
```
We benchmarked specialist (from-scratch) and generalist (foundation-based) MLFFs for Cr-doped \ce{Sb2Te3} using non-equilibrium configuration probes as a candidate framework for evaluating both interpolation and extrapolation behavior, demonstrated here on a single material system and a single architecture.
```

### E7b. L388 ("broader principles" → hypotheses)

**OLD** (L388, opening):
```
Our findings highlight several broader principles. Robust benchmarking must span both in-distribution and out-of-distribution tasks, as accuracy near equilibrium does not guarantee kinetic fidelity.
```

**NEW**:
```
Our findings support several hypotheses whose generality beyond Cr-doped \ce{Sb2Te3} remains to be established through cross-system validation. Within this case study, benchmarking must span both in-distribution and out-of-distribution tasks, as accuracy near equilibrium did not guarantee kinetic fidelity.
```

### E7c. L390 (closing outlook)

**OLD**:
```
Together, these results underscore that ``more data'' is not necessarily the solution; task-relevant, non-equilibrium sampling is what drives meaningful improvement. Migration-inspired probes, coupled with latent-space diagnostics, provide a principled route to identify precisely which additional configurations are needed and where models are likely to fail. This framework reframes the specialist–generalist debate: the goal is not to choose \textit{between} model classes, but to \textit{exploit} the complementary physics they learn in order to build more robust and data-efficient MLFFs for accelerated materials discovery.
```

**NEW**:
```
Together, these results suggest—for the system studied here—that ``more data'' is not necessarily the solution; task-relevant, non-equilibrium sampling was what drove meaningful improvement. Migration-inspired probes, coupled with latent-space diagnostics, offer a principled candidate route to identify which additional configurations are needed and where models are likely to fail. Whether this framework generalizes across materials, dopants, and architectures is an open question; validation on additional systems is the natural next step before it can be considered a general standard. If it holds, the specialist–generalist debate may be reframed: the goal would not be to choose \textit{between} model classes, but to \textit{exploit} the complementary physics they learn in order to build more robust and data-efficient MLFFs for accelerated materials discovery.
```

---

## E8. NEW delta paragraph vs arXiv:2509.00090 [46] (reviewer I7)

**Location**: Introduction, inserted after L190's closing sentence
("...diagnostic framework for designing more data-efficient learning loops.").

**OLD**: (no existing text — insertion)

**NEW**:
```
This article extends our preliminary workshop report~\citep{cao2025migration} in several
substantive ways: (i) an extended molecular-dynamics transport analysis with uncertainty
quantification (diffusivity and thermal conductivity with block-averaged error bars);
(ii) a self-consistent NEB stability criterion, in which each model must optimize the full elastic
band from the same endpoints, complementing fixed-geometry barrier evaluation (Sec. 3.2);
(iii) a feature-level interpretability framework based on SHAP surrogate models of
per-configuration error (Sec. 3.5); and
(iv) an interlayer-sliding analysis probing robustness to collective, non-local displacements
(Sec. 3.3).
```

> Formatting note: the manuscript currently has no `\label{sec:...}` anchors, so Sec. references are
> written as literal section numbers; switch to `\ref{}` when labels are added during revision.
> The three new cite keys used below (`alghamdi2025maceLiF`, `bheemaguli2026mlipBarriers`,
> `deng2025softening`) correspond to entries in `Revision1/tables/new_references.bib` (metadata
> verified against arXiv/publisher pages on 2026-09-06).

**Citation insertions (reviewer I7)**:
- "Softened PES" discussion at L282 — append `~\citep{deng2025softening}` to the sentence
  "This behavior is characteristic of a ``softened'' or ``averaged'' potential energy surface, a
  known trait of foundation models trained to generalize across vast chemical spaces." (prior art
  acknowledged, no longer implied novel).
- Migration-barrier benchmarking context — cite `~\citep{bheemaguli2026mlipBarriers}` where
  foundation-model NEB barrier evaluation is introduced (§3.2 opening, L256).
- MACE fine-tuning for ion diffusion — cite `~\citep{alghamdi2025maceLiF}` in §3.2 "The Critical
  Role of Task-Specific Fine-Tuning" (L299) alongside the discussion of targeted fine-tuning.
```
