"""Row builders for SI Tables S1, S4, S8, S9, S10 and S11 (Revision 1, submitted 2026-11-10).

Each builder takes a repro_lib.Evidence, reads its inputs through ``ev.input`` (repo-relative,
sha256 recorded), records consistency checks, and returns table body rows computed from data.
Only labels, category orders, column names and display precision live here; every number in a
row comes from a data file. The per-item scripts in items/si/<item>/source/ call these and then
``ev.compare_table`` against the submitted LaTeX.

Same arithmetic as audit_current_figures.py (sample SD, ddof=1), which wrote the tables before
they were rounded by hand on Overleaf.
"""
from __future__ import annotations

import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import fmt_int, fmt_num, fmt_pm, fmt_sci, lint_script, rel, sha256  # noqa: E402

CURRENT = "paper/revision1/data_processed/current_release"
PARITY = f"{CURRENT}/parity_summary.csv"
FIG5 = f"{CURRENT}/fig5_corrected_input.tsv"
FIG5_FAMILY_STATS = f"{CURRENT}/fig5_family_statistics.csv"
MD_STATUS = f"{CURRENT}/MD15_STATUS_AUDIT_R02.tsv"
MD_FIGURE = f"{CURRENT}/MD15_FIGURE_DATA_R06G.tsv"
MD_PROTOCOL = f"{CURRENT}/md_protocol_manifest.json"
FEATURES = "release/revision1_camera_ready/reproducibility/fig6_current_input.npz"
ORIGINAL_SPACE_CSV = f"{CURRENT}/fig6_current_original_space_metrics.csv"
TSNE_CSV = f"{CURRENT}/fig6_current_tsne_sensitivity.csv"
SHAP_METRICS = f"{CURRENT}/shap_seed_metrics.csv"

TRAINED = ["Scratch-600K", "FT-600K", "Scratch-MultiT", "FT-MultiT"]            # Table S1 row order
NEB_FAMILIES = ["MP-0B3", "OMAT-0", "MPA0", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]
NEB_TASKS = ["migration_in_gap_fixed", "migration_within_ql_fixed",
             "sliding_pristine_fixed", "sliding_cr_in_gap_fixed"]                 # Table S4 columns
SHAP_FAMILIES = ["FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]
# Submitted Outcome wording per MD15_STATUS_AUDIT_R02 status code. The audit marks the three
# foundation runs TARGET_REACHED_PROVISIONAL; the submitted table prints them as "Target reached".
MD_OUTCOME = {"TARGET_REACHED": "Target reached",
              "TARGET_REACHED_PROVISIONAL": "Target reached",
              "SCIENTIFIC_FAILURE": "Threshold reached"}
FS_PER_PS = 1000.0
# Feature grammar of fig6_current_input.npz (as recompute_feature_metrics.py): the last column is an
# all-zero legacy placeholder, and columns [6151, 6181) of the remaining block are the common
# 30-bin neighbour histogram.
HIST_COLUMNS = slice(6151, 6181)
TSNE_SETTINGS = dict(n_components=2, init="pca", max_iter=1000, n_jobs=1)
TRUST_NEIGHBORS = 5
SCI_DIGITS = 3        # generated precision exceeds every submitted a x 10^b value
COMPARE_ATOL = 1e-12  # recomputed vs committed CSV metrics


def record_shared_module(ev) -> None:
    """Hash and lint this shared module in the item's evidence (repro_lib lints only the caller)."""
    here = Path(__file__).resolve()
    findings = lint_script(here)
    ev.check("shared module si_tables.py hardcode lint clean", not findings,
             {"path": rel(here), "sha256": sha256(here), "findings": findings})


# ------------------------------------------------------------------ Table S1
def pointwise_rows(ev) -> list[list[str]]:
    parity = pd.read_csv(ev.input(PARITY))
    test = parity[parity.split == "test"]
    rows, seeds = [], {}
    for model in TRAINED:
        group = test[test.model == model]
        frames = group.n_frames.unique()
        seeds[model] = sorted(group.seed.astype(int).tolist())
        ev.check(f"{model}: one test-frame count across seeds", len(frames) == 1,
                 {"n_frames": frames.tolist(), "seeds": seeds[model]})
        rows.append([model, fmt_int(frames[0]),
                     fmt_pm(group.energy_rmse_meV_per_atom.mean(), group.energy_rmse_meV_per_atom.std()),
                     fmt_pm(group.force_rmse_meV_A.mean(), group.force_rmse_meV_A.std())])
    ev.check("three training seeds per condition", all(len(s) == 3 for s in seeds.values()), seeds)
    return rows


# ------------------------------------------------------------------ Table S4
def neb_error_rows(ev) -> list[list[str]]:
    data = pd.read_csv(ev.input(FIG5), sep="\t")
    data = data.assign(family_from_row=data.row_id.str.split(" / ").str[0].str.strip())
    recomputed = np.abs(data.predicted_value_eV - data.dft_reference_eV)
    ev.check("absolute_error_eV = |predicted - DFT reference|",
             bool(np.allclose(data.absolute_error_eV, recomputed, atol=1e-10)),
             {"rows": len(data), "max_diff": float(np.max(np.abs(data.absolute_error_eV - recomputed)))})
    ev.check("row_id prefix equals family column", bool((data.family_from_row == data.family).all()),
             sorted(data.family.unique()))
    stats = data.groupby(["family", "column_id"]).absolute_error_eV.agg(["size", "mean", "std"])
    committed = pd.read_csv(ev.input(FIG5_FAMILY_STATS)).set_index(["family", "column_id"])
    joined = stats.join(committed, how="inner")
    ev.check("groupby equals committed fig5_family_statistics.csv",
             len(joined) == len(stats) == len(committed)
             and bool(np.allclose(joined["mean"], joined.error_mean, atol=COMPARE_ATOL, equal_nan=True))
             and bool(np.allclose(joined["std"], joined.error_sd, atol=COMPARE_ATOL, equal_nan=True)),
             {"groups": len(stats)})
    rows = []
    for family in NEB_FAMILIES:
        cells = []
        for task in NEB_TASKS:
            group = stats.loc[(family, task)]
            single = int(group["size"]) == 1
            cells.append(f"${group['mean']:.2f}$" if single else fmt_pm(group["mean"], group["std"]))
        rows.append([family] + cells)
    return rows


# ------------------------------------------------------------------ Table S8
def md_horizon_rows(ev) -> list[list[str]]:
    status = pd.read_csv(ev.input(MD_STATUS), sep="\t")
    figure = pd.read_csv(ev.input(MD_FIGURE), sep="\t")
    protocol = json.loads(ev.input(MD_PROTOCOL).read_text())["md_protocol"]
    start_ps = protocol["start_step"] * protocol["timestep_fs"] / FS_PER_PS
    merged = status.merge(figure[["model_id", "last_time_ps", "status"]], on="model_id",
                          suffixes=("", "_figure"), validate="one_to_one")
    ev.check("status audit and figure data agree on clock and status",
             len(merged) == len(status) == len(figure)
             and bool((merged.last_time_ps == merged.last_time_ps_figure).all())
             and bool((merged.status == merged.status_figure).all()), {"models": len(merged)})
    step_clock = merged.last_step * protocol["timestep_fs"] / FS_PER_PS
    ev.check("last_time_ps = last_step x timestep", bool(np.allclose(step_clock, merged.last_time_ps)),
             {"timestep_fs": protocol["timestep_fs"]})
    reached = merged.status.str.startswith("TARGET_REACHED")
    ev.check("target-reached runs end at target_step",
             bool((merged.last_step[reached] == merged.target_step[reached]).all()),
             merged.loc[reached, ["model_id", "status", "last_step", "target_step"]].to_dict("records"))
    unknown = sorted(set(merged.status) - set(MD_OUTCOME))
    used = merged.groupby("status").model_id.apply(list).to_dict()
    ev.check("Outcome wording mapped from MD15_STATUS_AUDIT_R02 status (provisional shown as reached)",
             not unknown, {"map": MD_OUTCOME, "models_per_status": used, "unmapped": unknown,
                           "continuation_start_ps": start_ps,
                           "continuation_start_source": f"{MD_PROTOCOL} md_protocol.start_step x timestep_fs"})
    return [[row.model_id.replace("_", "\\_"), f"{row.last_time_ps:g}", f"{row.last_time_ps - start_ps:g}",
             MD_OUTCOME.get(row.status, row.status)] for row in merged.itertuples()]


# ------------------------------------------------------------------ Tables S9 and S10
def load_features(ev):
    payload = np.load(ev.input(FEATURES), allow_pickle=False)
    raw, labels = payload["features"], payload["model_ids"].astype(str)
    placeholder_zero = np.count_nonzero(raw[:, -1]) == 0
    ev.check("feature matrix finite; last column is the all-zero placeholder (removed)",
             bool(np.isfinite(raw).all()) and placeholder_zero,
             {"shape": list(raw.shape), "models": len(set(labels))})
    features = raw[:, :-1]
    hist = features[:, HIST_COLUMNS]
    ev.check("neighbour histogram block non-negative", bool(np.all(hist >= 0)), {"shape": list(hist.shape)})
    return features, hist, labels


def _spaces(features, hist):
    from sklearn.preprocessing import StandardScaler
    scale = lambda arr: StandardScaler().fit_transform(arr)  # noqa: E731
    return {"Original output features (6191)": (features, "euclidean"),
            "Original output features, cosine": (features, "cosine"),
            "Z-scored output features": (scale(features), "euclidean"),
            "Common neighbour histogram (30)": (hist, "euclidean"),
            "Z-scored neighbour histogram": (scale(hist), "euclidean")}


def _compare_csv(ev, name, recomputed: pd.DataFrame, committed: pd.DataFrame, value_columns):
    diffs = {c: float(np.max(np.abs(recomputed[c].to_numpy() - committed[c].to_numpy()))) for c in value_columns}
    ev.check(f"recomputed values equal committed {name}",
             len(recomputed) == len(committed) and all(d <= COMPARE_ATOL for d in diffs.values()),
             {"max_abs_diff": diffs, "rows": len(recomputed)})


def original_space_rows(ev) -> list[list[str]]:
    from sklearn.metrics import silhouette_score
    features, hist, labels = load_features(ev)
    committed = pd.read_csv(ev.input(ORIGINAL_SPACE_CSV))
    spaces = _spaces(features, hist)
    ev.check("feature spaces match committed CSV (name, metric)",
             list(committed.space) == list(spaces)
             and list(committed.metric) == [metric for _, metric in spaces.values()], list(committed.space))
    recomputed = pd.DataFrame([dict(space=name, dimensions=arr.shape[1], metric=metric,
                                    silhouette=float(silhouette_score(arr, labels, metric=metric)))
                               for name, (arr, metric) in spaces.items()])
    _compare_csv(ev, ORIGINAL_SPACE_CSV, recomputed, committed, ["dimensions", "silhouette"])
    return [[r.space, f"{r.dimensions}", fmt_sci(r.silhouette, SCI_DIGITS)] for r in recomputed.itertuples()]


def tsne_rows(ev) -> list[list[str]]:
    from sklearn.manifold import TSNE, trustworthiness
    from sklearn.metrics import silhouette_score
    features, _, labels = load_features(ev)
    committed = pd.read_csv(ev.input(TSNE_CSV))  # supplies the perplexity x seed grid
    records = []
    for perplexity, seed in zip(committed.perplexity, committed.seed):
        embedding = TSNE(perplexity=float(perplexity), random_state=int(seed), **TSNE_SETTINGS).fit_transform(features)
        records.append(dict(perplexity=perplexity, seed=seed,
                            silhouette=float(silhouette_score(embedding, labels)),
                            trustworthiness=float(trustworthiness(features, embedding, n_neighbors=TRUST_NEIGHBORS))))
    recomputed = pd.DataFrame(records)
    _compare_csv(ev, TSNE_CSV, recomputed, committed, ["silhouette", "trustworthiness"])
    import sklearn
    ev.check("environment note", True, {"sklearn": sklearn.__version__, "numpy": np.__version__,
                                        "tsne": TSNE_SETTINGS, "trustworthiness_n_neighbors": TRUST_NEIGHBORS})
    return [[f"{r.perplexity}", f"{r.seed}", fmt_sci(r.silhouette, SCI_DIGITS), fmt_num(r.trustworthiness, 3)]
            for r in recomputed.itertuples()]


# ------------------------------------------------------------------ Table S11
def shap_fidelity_rows(ev) -> list[list[str]]:
    metrics = pd.read_csv(ev.input(SHAP_METRICS))
    rows, seeds, means = [], {}, {}
    for family in SHAP_FAMILIES:
        group = metrics[metrics.model_family == family]
        seeds[family] = sorted(group.seed.astype(int).tolist())
        r2, mae = group.r2_5fold_cv, group.mae_5fold_cv_meV_per_atom
        rows.append([family, f"${r2.mean():.4f}\\pm{fmt_sci(r2.std(), SCI_DIGITS).strip('$')}$",
                     fmt_pm(mae.mean(), mae.std(), 3)])
        means[family] = float(mae.mean())
    ev.check("three training seeds per condition", all(len(s) == 3 for s in seeds.values()), seeds)
    ev.check("diagnostic (informational): MAE mean rounded once vs via the 2-dp audit table", True,
             {f: {"data": m, "one_step_1dp": _half_up(m, "0.1"),
                  "two_step_2dp_then_1dp": _half_up(float(_half_up(m, "0.01")), "0.1")}
              for f, m in means.items()})
    return rows


def _half_up(value: float, quantum: str) -> str:
    return str(Decimal(repr(value)).quantize(Decimal(quantum), rounding=ROUND_HALF_UP))
