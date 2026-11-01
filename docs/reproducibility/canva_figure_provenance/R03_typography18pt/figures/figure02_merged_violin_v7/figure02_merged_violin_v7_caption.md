# Figure 2 merged-train/test violin diagnostics

Train and held-out-test frame-level values are merged whenever their category labels are available. The temperature panel is restricted to Scratch-MultiT and FT-MultiT, because the 600K-only families do not define a 300K/600K/1200K temperature comparison. Each violin is drawn in log10-RMSE coordinates from all finite frame-level values; its width represents density and is mildly scaled by frame count. Three black points and the sample SD whisker retain the three seed-level category summaries. The held-out-trajectory panel remains test-identity-specific because training records do not carry recoverable O3/Tetra labels.

Energy aggregation: RMSE of merged plotted per-frame absolute energy error; this equals signed-error RMSE after squaring. Force aggregation: unweighted RMS of merged plotted per-frame force RMSE; n_atoms are not encoded in the source SVG.
