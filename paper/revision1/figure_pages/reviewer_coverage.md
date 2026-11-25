# Reviewer-to-figure coverage

This register is generated from the page-level reviewer annotations. `DONE` means the linked figure can support the narrowed reviewer response for the declared scope; `PARTIAL` or `PENDING` means the page must retain the listed completion gate.

## REV-A1 — PARTIAL

**Reviewer question:** Reconcile Fig. 3d fixed-geometry DFT-path scoring with SI Fig. S3 self-consistent MLFF-NEB, including pathway identity and the DFT reference.

**Linked pages:** 96 (PARTIAL=96)
**Figure IDs:** FIG-025, FIG-033, FIG-034, FIG-035, FIG-036, FIG-037, FIG-038, FIG-039, FIG-040, FIG-041, FIG-042, FIG-043, FIG-044, FIG-045, FIG-046, FIG-047, FIG-048, FIG-049, FIG-050, FIG-051, FIG-052, FIG-053, FIG-054, FIG-055, FIG-056, FIG-057, FIG-058, FIG-059, FIG-060, FIG-061, FIG-062, FIG-063, FIG-082, FIG-083, FIG-084, FIG-085, FIG-086, FIG-087, FIG-088, FIG-089, FIG-090, FIG-091, FIG-092, FIG-093, FIG-094, FIG-095, FIG-096, FIG-097, FIG-105, FIG-106, FIG-107, FIG-110, FIG-112, FIG-113, FIG-114, FIG-115, FIG-116, FIG-117, FIG-118, FIG-119, FIG-120, FIG-121, FIG-122, FIG-123, FIG-124, FIG-125, FIG-126, FIG-127, FIG-128, FIG-129, FIG-130, FIG-131, FIG-132, FIG-133, FIG-134, FIG-135, FIG-136, FIG-137, FIG-138, FIG-139, FIG-140, FIG-141, FIG-142, FIG-143, FIG-144, FIG-153, FIG-155, FIG-190, FIG-192, FIG-194, FIG-195, FIG-205, FIG-206, FIG-207, FIG-208, FIG-211
**Completed evidence:** The fixed-path versus self-relaxed operator distinction and the narrower model-reported-barrier wording are integrated.
**Still required:** Close the restartable QE path-1--4 convergence gate, preserve the final path/source metadata, and complete the SI Fig. S3 endpoint/checkpoint/optimizer/source audit before calling the quantitative reference final.

## REV-A2 — DONE

**Reviewer question:** Rule out train/test overlap between in-gap NEB images and FT-600K training frames.

**Linked pages:** 1 (DONE=1)
**Figure IDs:** FIG-151
**Completed evidence:** Grouped splitting and the SOAP/RMSD nearest-neighbor audit are complete for the declared training scopes; the historical 0.16 eV point is not silently reused.
**Still required:** No additional figure evidence is required for the completed scope; retain the thresholds, source scope, and manifest in the final release.

## REV-A3 — PARTIAL

**Reviewer question:** Add seed-level uncertainty and test whether the deep-penetration result is seed-stable rather than a chance outcome.

**Linked pages:** 4 (PARTIAL=4)
**Figure IDs:** FIG-069, FIG-070, FIG-071, FIG-211
**Completed evidence:** Three-seed grouped in-gap comparisons and uncertainty are available for the declared Scratch/FT-600K/Scratch-5% scope.
**Still required:** Complete path-resolved deep-penetration classification and restarted QE references before reporting a pooled deep-penetration ranking.

## REV-A4 — DONE

**Reviewer question:** Rescope generalizability claims or add an independent material/dopant validation.

**Linked pages:** 2 (DONE=2)
**Figure IDs:** FIG-001, FIG-068
**Completed evidence:** The manuscript language is scoped to a candidate framework demonstrated on Cr-doped Sb2Te3; broader claims are framed as future validation.
**Still required:** Do not add a cross-system conclusion unless the optional second-dopant calculation is actually completed and audited.

## REV-A5 — PARTIAL

**Reviewer question:** Add a Scratch-5% random-initialization control to separate foundation pretraining from data-volume effects.

**Linked pages:** 3 (PARTIAL=3)
**Figure IDs:** FIG-069, FIG-070, FIG-071
**Completed evidence:** The three-seed Scratch-5% control and the in-gap pathway comparison are complete and can be shown with the stated small-sample caveat.
**Still required:** Do not generalize to FT-MultiT or all pathways until the corresponding grouped retraining and path comparisons exist.

## REV-A6 — PARTIAL

**Reviewer question:** Soften t-SNE/PHATE and SHAP interpretation claims and add projection/original-space and surrogate-fidelity checks.

**Linked pages:** 18 (PARTIAL=18)
**Figure IDs:** FIG-152, FIG-154, FIG-160, FIG-161, FIG-162, FIG-173, FIG-174, FIG-175, FIG-176, FIG-177, FIG-178, FIG-179, FIG-189, FIG-191, FIG-201, FIG-202, FIG-203, FIG-204
**Completed evidence:** Projection sensitivity, original-descriptor-space separation, and per-model cross-validated surrogate R2 are available; the text is scoped to descriptive/surrogate evidence.
**Still required:** A direct MACE Cr-Cr perturbation test remains pending if a causal claim about the dominant feature is desired; otherwise retain the surrogate-scoped wording.

## REV-A7 — DONE

**Reviewer question:** Position the work against nearest literature and distinguish it from the earlier workshop report.

**Linked pages:** 0 ()
**Figure IDs:** 
**Completed evidence:** The related-work comparison and the explicit delta from the earlier non-archival report are integrated in the manuscript.
**Still required:** No new figure is required; keep the figure page linked to the revised narrative rather than presenting legacy panels as new evidence.

## REV-C8 — DONE

**Reviewer question:** Address fabricated all-zero forces in NEB exporters and prove whether they entered reported training data.

**Linked pages:** 0 ()
**Figure IDs:** 
**Completed evidence:** The exporters are fail-closed and the frame-level provenance audit found no unclassified NEB-derived all-zero-force frames in the reported datasets.
**Still required:** No new figure is required; cite the release hash and machine-readable audit manifest as the direct evidence.

## REV-C9 — PENDING

**Reviewer question:** Resolve the asymmetric FT-600K MD transport protocol and its effect on D and kappa.

**Linked pages:** 22 (PENDING=22)
**Figure IDs:** FIG-002, FIG-003, FIG-004, FIG-005, FIG-006, FIG-007, FIG-008, FIG-009, FIG-010, FIG-013, FIG-014, FIG-064, FIG-065, FIG-066, FIG-067, FIG-069, FIG-071, FIG-074, FIG-075, FIG-156, FIG-193, FIG-210
**Completed evidence:** Legacy FT-600K transport values are quarantined and excluded from ranking; the figure can document that boundary.
**Still required:** Run the corrected matched-protocol FT-600K long trajectory, block-average D/kappa, and close the multi-seed uncertainty before reporting a final transport comparison.

## REV-C10 — DONE

**Reviewer question:** Replace the frame-level random split with a grouped train/test split and overlap audit.

**Linked pages:** 1 (DONE=1)
**Figure IDs:** FIG-151
**Completed evidence:** Grouped-split retraining, zero-overlap checks, and the revised RMSE/barrier statistics are available for the declared models and seeds.
**Still required:** Keep historical frame-level values visibly labeled and use grouped values for generalization claims.

## REV-C11 — DONE

**Reviewer question:** Compute separability in the original/high-dimensional descriptor space rather than only on a 2D embedding.

**Linked pages:** 3 (DONE=3)
**Figure IDs:** FIG-154, FIG-173, FIG-191
**Completed evidence:** Original-space metrics and projection-sensitivity checks are available; embedding scores are retained only as descriptive baselines.
**Still required:** No new numerical figure is required; preserve the feature-space definition and scaling sensitivity in the caption/methods.

## REV-C12 — DONE

**Reviewer question:** Remove the always-zero force-sensitivity feature stub from the latent analysis.

**Linked pages:** 3 (DONE=3)
**Figure IDs:** FIG-154, FIG-173, FIG-191
**Completed evidence:** The inactive channel was removed and no unsupported force-sensitivity claim is made.
**Still required:** No additional figure evidence is required unless a genuine finite-difference force-sensitivity analysis is later added.

## REV-C13 — PARTIAL

**Reviewer question:** Make the SHAP pipeline and advertised repository artifacts reproducible rather than leaving empty stubs.

**Linked pages:** 15 (PARTIAL=15)
**Figure IDs:** FIG-152, FIG-160, FIG-161, FIG-162, FIG-174, FIG-175, FIG-176, FIG-177, FIG-178, FIG-179, FIG-189, FIG-201, FIG-202, FIG-203, FIG-204
**Completed evidence:** The corrected SHAP inputs, scripts, lineage, and locked revised outputs are present for the declared three-model/150-frame scope.
**Still required:** Close the final release/empty-stub audit and reconcile the independent rerun with the locked historical CSV before claiming exact end-to-end regeneration.
