"""Seed sweep of the Fig. 6 SHAP surrogate (reuses scripts/shap_analysis.py unchanged)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, shap
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, str(Path(__file__).parent))
import shap_analysis as sa

N_SEEDS = 30
root = Path(__file__).resolve().parents[1]
d = root / "data/raw_compact/revision1/shap"
frame, names = sa.load_aligned(d / "sampled_trajectory_Cr2_temp.xyz",
                               d / "sampled_trajectory_Cr2_temp_predictions_with_binding1.csv")
N_JOBS = 10


def run_one(model: str, seed: int, X: pd.DataFrame, y: np.ndarray) -> dict:
    est = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=8,
                                    subsample=0.8, random_state=seed)
    cv = KFold(5, shuffle=True, random_state=seed)
    r2 = cross_val_score(est, X, y, cv=cv, scoring="r2").mean()
    mae = -cross_val_score(est, X, y, cv=cv, scoring="neg_mean_absolute_error").mean()
    est.fit(X, y)
    imp = np.abs(np.asarray(shap.TreeExplainer(est)(X).values)).mean(0)
    top = [names[i] for i in np.argsort(imp)[::-1][:5]]
    return dict(model=model, seed=seed, r2_cv=r2, mae_cv=mae, top1=top[0], top5=";".join(top))


jobs = []
for model in sa.MODELS:
    g = frame[frame.model_name == model].sort_values("image_index")
    X, y = g[names], g.target.to_numpy(float)
    jobs += [delayed(run_one)(model, seed, X, y) for seed in range(N_SEEDS)]
rows = Parallel(n_jobs=N_JOBS)(jobs)
out = root / "data/processed/shap_seed_sweep"
df = pd.DataFrame(rows); df.to_csv(out / "shap_seed_sweep_raw.csv", index=False)
locked = pd.read_csv(root / "data/processed/shap_surrogate_r2_revised.csv").set_index("model")
s = df.groupby("model").agg(r2_mean=("r2_cv","mean"), r2_std=("r2_cv","std"), r2_min=("r2_cv","min"),
    r2_max=("r2_cv","max"), mae_mean=("mae_cv","mean"), mae_std=("mae_cv","std"),
    top1_mode=("top1", lambda x: x.mode()[0]), top1_freq=("top1", lambda x: (x==x.mode()[0]).mean()))
s["locked_r2"] = locked.r2_5fold_cv; s["locked_mae"] = locked.mae_5fold_cv
s["n_seeds"] = N_SEEDS
s.to_csv(out / "shap_seed_sweep_summary.csv"); print(s.T.to_string())
