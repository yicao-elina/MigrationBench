# Figure 2 distribution-first clustered diagnostics

Each train/test and model-family position shows every available finite frame-level point, a narrow violin density layer, and an empirical boxplot for the median/IQR. The three black points are the seed-level category RMSE aggregates and their whisker is the sample SD across seeds. Transparent hatched layers are train and opaque layers are held-out test. The two held-out trajectories are the fixed test identities present in the package; this is not a leave-one-trajectory-out retraining result.

Energy aggregation: RMSE of plotted per-frame absolute energy error; this equals signed-error RMSE after squaring. Force aggregation: unweighted RMS of plotted per-frame force RMSE; n_atoms are not encoded in the source SVG.
