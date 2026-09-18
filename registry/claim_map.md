# Claim Map — flagged manuscript claims → revision status → response-letter section

Line numbers refer to submitted `sn-article-final.tex`. Status codes:
**REWORDED** = replacement drafted in `Revision1/response_to_reviewers/proposed_text_edits.md`;
**PENDING-data JB-x / LP-x** = replacement drafted but contains numeric placeholders awaiting that task;
**CONTEXT-added** = claim retained, hedging/citation/scope context added around it.

| # | Claimant quote (verbatim) | Tex line | Reviewer item | Status | Edit ref | Response-letter section |
|---|---|---|---|---|---|---|
| 1 | "This work establishes migration-based non-equilibrium probes as a data-efficient, generalizable standard for MLFF evaluation and provides actionable guidance for robust MLFF development." | L153 | I4 | REWORDED | E1 | A4 |
| 2 | "Latent-space analysis reveals fundamentally distinct learned representations across training paradigms, explaining why equilibrium metrics alone are insufficient." | L151 | I6 (softening) | CONTEXT-added | (hedged via E1 wording; §3.4 softening E5) | A6 |
| 3 | "Can non-equilibrium probes generated from methods, such as NEB, provide a generalizable and efficient benchmark for specialist versus generalist models?" | L195 | I4 | REWORDED | E2 | A4 |
| 4 | "This establishes a (reasonably low) migration energy barrier ($E_{\mathrm{a}}$) of 0.34~eV for this process." | L272 | C1 (reference value) | PENDING-data (value fixed: 0.336 eV, LP-1 verified; table refs pending) | E3 | A1 |
| 5 | "overestimating the activation energy by over 4~eV, an unacceptably high amount" (Scratch in-gap) | L275 | C3 (single-run uncertainty) | PENDING-data: JB-2 | E4 (seed stats) | A3 |
| 6 | "overestimating the barrier by approximately 0.7~eV ... Both are unacceptable results." | L282 | C1 (fixed-geom vs self-NEB) | PENDING-data: JB-5 | E4 (unified re-eval) | A1 |
| 7 | "a ``softened'' or ``averaged'' potential energy surface, a known trait of foundation models" | L282 | I7 (invoked as if novel) | CONTEXT-added: cite deng2025softening | E8 citation insertions | A7 |
| 8 | "its globally inaccurate and likely unphysical PES happens, by chance, to be less pathologically incorrect" | L296 | C3 (unfalsifiable "chance") | PENDING-data: JB-2 (pre-registered decision rule; replaced by evidence either way) | E4 | A3 |
| 9 | "achieves remarkable accuracy, with a barrier error of only 0.16~eV" (FT-600K) | L299 | C2 (overlap), C3 (seeds) | PENDING-data: JB-4 (overlap) + JB-2 (seeds) | E4 | A2, A3 |
| 10 | "The foundation model converges smoothly to a physically meaningful MEP with a barrier of 0.41~eV" | L307 | C1 | REWORDED (canonical reference 0.336 eV, pathway stated) + PENDING-data: JB-5 | E4 | A1 |
| 11 | "This instability confirms that—even when barriers appear reasonable on fixed geometries—their learned PES cannot reliably guide gradient-based optimization." | L307 | C1 | PENDING-data: JB-5 (side-by-side re-eval) | E4 | A1 |
| 12 | "quantitatively supported by their high average silhouette scores (Fig.~\ref{fig:tsne-phate}d)" | L332 | I6 / repo R4 (silhouette on embedding) | PENDING-data: LP-2 (original-space silhouette) | E5a | A6, C11 |
| 13 | "This geometric difference in the latent space provides a direct mechanistic explanation for the models' performance on the diffusion task." | L338 | I6 | REWORDED ("consistent with") + PENDING-data: LP-2 (sensitivity) | E5b | A6 |
| 14 | "Fine-tuning succeeds not by creating a simple hybrid, but by aligning a general physical prior with the specific dynamical manifold of a target system, thereby enabling robust generalization" | L338 | I4/I6 | CONTEXT-added (hedged under E5b sensitivity framing) | E5b | A6 |
| 15 | "Average silhouette scores (from $t$-SNE) quantify the representational dissimilarity. The results demonstrate that fine-tuning succeeds..." (Fig. caption d) | L347 | I6 / repo R4 | PENDING-data: LP-2 | E5c | A6, C11 |
| 16 | "a simpler, yet highly faithful, model of its error behavior" (SHAP surrogate) | L359 | I6 (surrogate R²) | PENDING-data: JB-8 (per-model R²) | E6a | A6 |
| 17 | "identifying which specific atomic interactions and geometric motifs are most correlated with prediction failures or successes" | L359 | I6 | CONTEXT-added (now bounded by explicit R², E6a) | E6a | A6 |
| 18 | "demonstrates that fine-tuning enables the model to internalize a physically coherent representation of the potential energy surface" | L365 | I6 (rescope to surrogate) | REWORDED (surrogate-scoped) + PENDING-data: JB-8 (perturbation test) | E6b | A6 |
| 19 | "explaining its inability to generalize to critical tasks such as resolving transition-state energetics or maintaining NEB stability" | L365 | I6 | REWORDED ("consistent with", surrogate-scoped) | E6b | A6 |
| 20 | "\texttt{Cr-Cr\_n13\_l3} feature, which is orders of magnitude more important than any other feature" (Fig. caption a) | L376 | I6 | PENDING-data: JB-8 (exact fold + R² table) | E6c | A6 |
| 21 | "Our findings highlight several broader principles. Robust benchmarking must span both in-distribution and out-of-distribution tasks..." | L388 | I4 | REWORDED (hypotheses, case-study scope) | E7b | A4 |
| 22 | "Task-relevant, non-equilibrium sampling is what drives meaningful improvement ... a principled route to identify precisely which additional configurations are needed ... This framework reframes the specialist–generalist debate" | L390 | I4 | REWORDED (hedged outlook; generalization as open question) | E7c | A4 |
| 23 | Transport coefficients from the naive-FT condition (Foundation 1.11e-07, FT-MultiT ~1.4e-08 cm²/s etc.) | §3.1 table/figure (Fig. 2) | repo R2 (MD protocol asymmetry) | PENDING-data: LP-3 (valid conditions, block-averaged) + JB-6 (FT-600K re-run); FT-600K value quarantined [INVALID-pending JB-6] | (figure/table rebuild, LP-3/JB-6) | C9 |
| 24 | Table S1 single-seed test RMSE values (ungrouped split) | SI Table S1 | repo R3 (ungrouped split), C3 | PENDING-data: JB-2 (grouped-split retrain, 3 seeds) | (SI table replaced) | C10, A3 |
| 25 | "benchmark framework" API / SHAP reproducibility implied by Sec. 3.5 and repo | §3.5; Data Availability | repo R6 (0-byte stubs) | PENDING-data: JB-8 (R² re-run); code port done via LP-5 | E6a–c | C13 |
| 26 | (Implicit) NEB-derived training frames used with real forces in FT runs | §2.1 (data description) | repo R1 (fake zero forces) | CONTEXT-added: provenance sentence; author confirmed NOT used; evidence archived via JB-1 (verdict: CLEAN); exporters hard-disabled via LP-5 | (Methods sentence) | C8 |

## Coverage check
- Reviewer main comments C1, C2, C3, I4, I5, I6, I7: mapped to response-letter sections A1–A7; every flagged quote above has an Edit ref (E1–E8) or a PENDING-data placeholder with a JB/LP owner.
- Repo comments R1–R6: mapped to sections C8–C13 (R1→C8, R2→C9, R3→C10, R4→C11, R5→LP-2/LP-5 covered under A6/C11–C12, R6→C13).
- Scratch-5% control (I5): no existing manuscript claim; new content pending JB-3 (response section A5).
