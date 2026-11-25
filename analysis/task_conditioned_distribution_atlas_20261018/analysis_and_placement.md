# Revised distribution atlas: analysis and placement

The release is rendered as independent Plot Atlas assets first. The grouped
reference and MD configuration-space panels use `panel_common.py`; the
original Rockfish RMSE history is parsed by
`parse_rockfish_training_rmse.py` and rendered by
`plot_original_training_rmse_four_model.py`. The composition script then
reuses only the two configuration-space recipes for the stacked overview.
Standalone panels are preferred for manuscript placement when a composite
would reduce label readability.

## Figure grammar

Panel a is the grouped reference-space diagnostic. SOAP descriptors are fit
and standardized on the grouped training frames only, then transformed for
validation, test, and the five pathway-1--4 fixed-path images. The joint plot
uses all retained frames, two-dimensional density contours, and top/right
marginals. The three split colors are not model-family colors.

The original-training RMSE figure is a separate four-column diagnostic. Its
upper row is energy RMSE and lower row force RMSE, both in meV-based units,
reconstructed from the native JSONL `mode=eval` histories. The curves are
descriptive single-run validation evidence; they are not grouped held-out
test statistics and do not provide seed uncertainty.

Panel c is the configuration-space diagnostic for the stored 600 K MLIP
continuation segments. The grouped training density is the reference envelope
and the colored clouds are model-generated configurations. These points are
not DFT errors. The current release uses the common MD velocity seed, so any
training-seed variation is a model/checkpoint comparison rather than an
independent velocity-replica uncertainty estimate. Its axes use the full
combined support of the grouped reference and MLIP-generated MD coordinates;
the broader MD range is therefore visible rather than clipped by the narrower
reference-space limits used in panel a.

The failure/CAR figure is intentionally separate. It plots the audited
candidate force trajectory, stage-wise force distributions, force-disagreement
trajectory, and distance-to-training-space relation. It is not folded into the
main distribution atlas because collapse dynamics are a distinct failure
operator.

## Quantitative reading

The grouped reference files contain 1,786 train, 311 validation, and 267 test
frames. Whole source AIMD trajectories are assigned to one partition; the split
is therefore trajectory/group level rather than a frame-level random shuffle.
The fixed-path overlay contains five current pathway-1--4 images and is shown
as a separate marker class because its 61-atom geometry is not the same as the
81--86-atom grouped AIMD cells.

The static audit gives grouped multi-seed pathway-1--4 absolute barrier errors
of 0.47 +/- 0.12 eV for FT-600K, 1.20 +/- 0.27 eV for Scratch, and 0.68 +/-
0.20 eV for Scratch-5%, with 0.24 eV for the single-seed MPA0 comparator. The
historical 0.16 eV point estimate is not reproduced. These values are not
replaced by the distribution atlas; the atlas supplies the split and
configuration-space context.

The standalone 639-candidate failure diagnostic gives median paired force
spread of 0.22 eV Angstrom^-1 in the early-stable window, 0.24 before onset,
0.44 in the precursor window, and 8.49 at threshold. This is a model
disagreement signal, not calibrated uncertainty, and the representation
distance is not a monotonic universal OOD ladder.

## Manuscript placement

Use the revised atlas as a descriptive figure in the task-conditioned model-
selection discussion or in the SI distribution-diagnostics section. The A2
response should cite the grouped split counts and the overlap audit directly;
the atlas should not be used as the sole leakage proof. The standalone failure
figure belongs in the failure/CAR subsection or SI, with its collapse and
MLIP-only scope stated in the caption.

Do not use either figure to claim universal generalization, a universal model
ranking, a calibrated uncertainty scale, or a converged self-relaxed DFT
barrier.
