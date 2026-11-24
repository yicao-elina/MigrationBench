# Migration-path diagnostic framework

## Framework claim

Each retained migration path is treated as a structured examination problem rather than only as a barrier-comparison datum. The framework first qualifies the QE reference, then decomposes Foundation-model error by image, stage, path, and alignment convention, and finally tests whether geometry-only features can identify high-error frames before an expensive reference calculation.

## Acceptance boundary

The current QE campaign supplies finite path snapshots and an exploratory last-table stability screen, but it does not meet the formal reference criterion: every audited NEB output contains at least one SCF-not-converged warning, no output exposes the inner electronic SCF iteration count, and no path carries an explicit convergence marker. Accordingly, the present results are valid for diagnostic path sampling, not for certification of a converged DFT activation barrier.

## Model failure diagnosis

The same-geometry matrix makes it possible to localize disagreement by migration stage and local environment. The Cu QLs path is a high-difficulty examination case with large Foundation-model elevation; the stage-resolved outputs determine whether the error is concentrated near the operational transition-state window or already present near the endpoints. Geometry-only grouped cross-validation reports AUROC, AUPRC, false-positive rate, and false-negative rate for a pre-computation warning rule. These metrics are screening evidence, not a deployment-ready uncertainty model, until a larger set of independent paths is available.

## Training-data implication

The current embedding output is explicitly NEB-frame-only because a provenance-qualified raw training/test/AIMD structure release was not found in the active workspace. The next data-design step is therefore to add independently hashed reference distributions and rerun the same descriptor and grouped-validation pipeline; only then can embedding distance be used as a defensible training-coverage criterion.
