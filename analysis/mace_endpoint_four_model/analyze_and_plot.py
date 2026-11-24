from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
for f in RAW.glob("*_finals.tar.gz"):
    with tarfile.open(f, "r:gz") as tf:
        tf.extractall(RAW)

MODEL_ORDER = ["FT-600K", "Foundation", "Scratch", "FT-MultiT"]
VARIANT_ORDER = ["base", "perturb_a_dx020", "perturb_b_tilt020"]
VARIANT_LABEL = {"base": "base", "perturb_a_dx020": r"Cr $+0.20$ Å", "perturb_b_tilt020": "tilt"}
COLORS = {"FT-600K": "#2C7FB8", "Foundation": "#D95F02", "Scratch": "#7570B3", "FT-MultiT": "#1B9E77"}
TYPE_COLORS = {"same-gap lateral": "#4C78A8", "coordination switch": "#F58518", "gap-to-layer penetration": "#54A24B", "other": "#B279A2"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def endpoint_type(d: dict) -> str:
    roles = d["task"].get("roles", [])
    if isinstance(roles, str):
        try: roles = json.loads(roles)
        except Exception: roles = []
    text = " ".join(str(r.get("path_id", "")) for r in roles)
    if "coordination_switch" in text: return "coordination switch"
    if "gap_to_layer_penetration" in text: return "gap-to-layer penetration"
    if "same_gap_lateral" in text: return "same-gap lateral"
    return "other"


def parse_cr_xyz(path: Path) -> tuple[float, float, float]:
    lines = path.read_text().splitlines()
    n = int(lines[0].strip())
    for row in lines[2:2+n]:
        p = row.split()
        if p and p[0] == "Cr":
            return tuple(map(float, p[1:4]))
    raise ValueError(f"Cr atom not found in {path}")


def final_path(model: str, task_id: str) -> Path:
    # The tar members retain runs/<model>/<task>/attempt_*/mace_relaxed.extxyz.
    if model == "FT-600K":
        candidates = list(RAW.glob(f"**/results/*{task_id}/mace_relaxed.extxyz"))
    else:
        candidates = list(RAW.glob(f"**/runs/{model}/{task_id}/attempt_*/mace_relaxed.extxyz"))
    if not candidates:
        raise FileNotFoundError((model, task_id))
    return candidates[0]


def load() -> pd.DataFrame:
    rows = []
    for fn in [RAW / "ft600k_summaries.jsonl", RAW / "new_model_summaries.jsonl"]:
        for d in read_jsonl(fn):
            task = d["task"]
            model = d["model_label"]
            tid = task["task_id"]
            try:
                xyz = parse_cr_xyz(final_path(model, tid))
            except Exception:
                xyz = (np.nan, np.nan, np.nan)
            success = bool(d.get("scientific_success", d.get("optimizer_converged", False)))
            rows.append({
                "model": model, "endpoint": task["raw_site_id"], "variant": task["variant"],
                "task_id": tid, "success": success,
                "steps": d.get("optimizer_steps_completed", np.nan),
                "fmax": d.get("final_max_force_eV_A", np.nan),
                "E_final": d.get("final_energy_eV", np.nan),
                "E_initial": d.get("initial_energy_eV", np.nan),
                "dE_relax": (d.get("final_energy_eV", np.nan) - d.get("initial_energy_eV", np.nan)) if d.get("initial_energy_eV") is not None else np.nan,
                "wall_s": d.get("wall_time_s", np.nan), "endpoint_type": endpoint_type(d),
                "x": xyz[0], "y": xyz[1], "z": xyz[2],
                "model_sha256": d.get("model_sha256", ""), "summary_path": d.get("summary_path", ""),
            })
    df = pd.DataFrame(rows)
    df["model"] = pd.Categorical(df.model, MODEL_ORDER, ordered=True)
    df["variant"] = pd.Categorical(df.variant, VARIANT_ORDER, ordered=True)
    return df.sort_values(["model", "endpoint", "variant"]).reset_index(drop=True)


def bootstrap_median(x, n=4000, seed=42):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return (float(np.median(x)), float(np.quantile(boot, .025)), float(np.quantile(boot, .975)))


def stats(df: pd.DataFrame):
    summ = (df.groupby(["model", "variant"], observed=True)
              .agg(n=("endpoint", "size"), n_success=("success", "sum"),
                   success_rate=("success", "mean"), median_steps=("steps", "median"),
                   median_fmax=("fmax", "median"), p95_fmax=("fmax", lambda x: np.nanquantile(x, .95)),
                   median_wall_s=("wall_s", "median"))
              .reset_index())
    summ["success_rate_pct"] = 100 * summ.success_rate
    summ.to_csv(OUT / "table_model_variant_summary.csv", index=False)

    wide = df.pivot_table(index=["endpoint", "variant"], columns="model", values=["x", "y", "z", "steps", "fmax", "E_final"], observed=True)
    for m in MODEL_ORDER[1:]:
        for v in VARIANT_ORDER:
            pass
    pair_rows = []
    for (ep, var), g in df.groupby(["endpoint", "variant"], observed=True):
        ref = g[g.model == "FT-600K"].iloc[0]
        for m in MODEL_ORDER[1:]:
            q = g[g.model == m].iloc[0]
            dist = np.linalg.norm(q[["x", "y", "z"]].to_numpy(float) - ref[["x", "y", "z"]].to_numpy(float))
            pair_rows.append({"endpoint": ep, "variant": var, "model": m, "cr_dist_to_ft600k_A": dist,
                              "steps_diff": q.steps - ref.steps, "fmax_diff": q.fmax - ref.fmax,
                              "success_both": bool(q.success and ref.success)})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "table_paired_vs_ft600k.csv", index=False)
    successful_pairs = pairs[pairs.success_both].copy()
    sens = (successful_pairs.groupby("endpoint", as_index=False)
                  .agg(n_successful_pairs=("cr_dist_to_ft600k_A", "size"),
                       max_successful_displacement_A=("cr_dist_to_ft600k_A", "max"),
                       median_displacement_A=("cr_dist_to_ft600k_A", "median"),
                       mean_displacement_A=("cr_dist_to_ft600k_A", "mean")))
    sens = sens.merge(df[["endpoint", "endpoint_type"]].drop_duplicates(), on="endpoint", how="left")
    sens.sort_values("max_successful_displacement_A", ascending=False).to_csv(OUT / "table_sensitive_endpoints.csv", index=False)

    # Model-internal relative energy landscape: subtract the minimum over the three starts
    # Failed optimizer attempts remain in the tidy audit table, but are excluded
    # from the scientific landscape summary; their partial endpoint energies are
    # not accepted as minima.
    en = df[df.success & df.E_final.notna()].copy()
    en["E_rel_endpoint_eV"] = en.E_final - en.groupby(["model", "endpoint"], observed=True).E_final.transform("min")
    en["E_span_endpoint_eV"] = en.groupby(["model", "endpoint"], observed=True).E_final.transform(lambda x: x.max()-x.min())
    en.to_csv(OUT / "table_relative_energy_landscape.csv", index=False)
    span = en.groupby(["model", "endpoint", "endpoint_type"], observed=True).E_span_endpoint_eV.first().reset_index()
    span.to_csv(OUT / "table_energy_span_by_endpoint.csv", index=False)

    # Per-model pairwise agreement and rank similarity
    ag = []
    coords = df.set_index(["endpoint", "variant", "model"])[["x", "y", "z"]]
    for i, a in enumerate(MODEL_ORDER):
        for b in MODEL_ORDER[i+1:]:
            dists=[]
            for key in coords.index.droplevel(2).unique():
                try: dists.append(np.linalg.norm(coords.loc[(key+(a,))].to_numpy()-coords.loc[(key+(b,))].to_numpy()))
                except KeyError: pass
            ag.append({"model_a": a, "model_b": b, "median_cr_distance_A": np.median(dists), "p95_cr_distance_A": np.quantile(dists,.95), "n": len(dists)})
    pd.DataFrame(ag).to_csv(OUT / "table_model_agreement.csv", index=False)
    return summ, pairs, en, span


def style():
    mpl.rcParams.update({"font.family":"Arial", "font.size":8, "axes.labelsize":9, "axes.titlesize":9,
                         "axes.linewidth":1.0, "xtick.labelsize":7.5, "ytick.labelsize":7.5,
                         "svg.fonttype":"none", "pdf.fonttype":42})


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def fig_overview(df, pairs):
    style(); fig, ax = plt.subplots(2,2, figsize=(7.5,6.2), constrained_layout=True)
    # A: success rate, with n/47
    a=ax[0,0]; x=np.arange(12); vals=[]; labels=[]
    for i,m in enumerate(MODEL_ORDER):
        for j,v in enumerate(VARIANT_ORDER):
            g=df[(df.model==m)&(df.variant==v)]; vals.append(g.success.mean()*100); labels.append(VARIANT_LABEL[v])
            a.scatter(i*4+j, vals[-1], s=42, color=COLORS[m], edgecolor='white', linewidth=.5, zorder=3)
            a.text(i*4+j, vals[-1]+2, f"{int(g.success.sum())}/47", ha='center', va='bottom', fontsize=6)
    a.axhline(100,color='#444',lw=.7,ls=(0,(2,2))); a.set_ylim(0,112); a.set_ylabel('Scientific success (%)'); a.set_xticks([1,5,9,13], MODEL_ORDER)
    a.set_title('Convergence and finite-force gate')
    # B steps distributions
    b=ax[0,1]; positions=[]
    for i,m in enumerate(MODEL_ORDER):
        for j,v in enumerate(VARIANT_ORDER):
            g=df[(df.model==m)&(df.variant==v)&df.success]; pos=i*4+j
            b.boxplot(g.steps, positions=[pos], widths=.55, patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=COLORS[m],alpha=.25,color=COLORS[m]), medianprops=dict(color=COLORS[m],lw=1.5), whiskerprops=dict(color=COLORS[m]), capprops=dict(color=COLORS[m]))
            rng=np.random.default_rng(100+pos); b.scatter(np.full(len(g),pos)+rng.uniform(-.12,.12,len(g)),g.steps,s=5,color=COLORS[m],alpha=.45)
    b.set_ylabel('FIRE steps'); b.set_xticks([1,5,9,13],MODEL_ORDER); b.set_title('Optimization effort (successful runs)')
    # C ECDF pair displacement
    c=ax[1,0]
    for m in MODEL_ORDER[1:]:
        q=pairs[pairs.model==m].cr_dist_to_ft600k_A.dropna().sort_values().to_numpy(); y=np.arange(1,len(q)+1)/len(q)
        c.step(q,y,where='post',color=COLORS[m],lw=1.5,label=m)
    c.axvline(.10,color='#555',lw=.7,ls=(0,(2,2))); c.set_xlabel('Final Cr displacement from FT-600K (Å)'); c.set_ylabel('Empirical CDF'); c.set_xlim(left=0); c.legend(frameon=False,fontsize=6); c.set_title('Paired basin agreement')
    # D pairwise median distance matrix
    d=np.zeros((4,4)); coords=df.set_index(['endpoint','variant','model'])[['x','y','z']]
    for i,m in enumerate(MODEL_ORDER):
        for j,n in enumerate(MODEL_ORDER):
            vals=[]
            if i!=j:
                for key in coords.index.droplevel(2).unique():
                    try: vals.append(np.linalg.norm(coords.loc[(key+(m,))].to_numpy()-coords.loc[(key+(n,))].to_numpy()))
                    except KeyError: pass
                d[i,j]=np.median(vals)
    im=ax[1,1].imshow(d,cmap='magma',vmin=0,vmax=max(.2,np.nanmax(d)))
    ax[1,1].set_xticks(range(4),MODEL_ORDER,rotation=30,ha='right'); ax[1,1].set_yticks(range(4),MODEL_ORDER); ax[1,1].set_title('Median paired Cr distance (Å)')
    for i in range(4):
        for j in range(4): ax[1,1].text(j,i,'—' if i==j else f'{d[i,j]:.3f}',ha='center',va='center',color='white' if d[i,j]>.5*max(.2,np.nanmax(d)) else '#222',fontsize=7)
    fig.text(.01,1.02,'Four-model endpoint relaxation: robustness, effort and basin agreement',fontsize=11,weight='bold')
    for letter,aa in zip('ABCD',ax.flat): aa.text(-.16,1.08,letter,transform=aa.transAxes,fontsize=11,weight='bold')
    save(fig,'fig1_endpoint_robustness')


def fig_landscape(df, en, span):
    style(); fig, ax = plt.subplots(1,3, figsize=(8.5,3.1), constrained_layout=True)
    # A: median relative-energy surface, model facets
    a=ax[0]; mat=np.full((4,3),np.nan)
    for i,m in enumerate(MODEL_ORDER):
        for j,v in enumerate(VARIANT_ORDER): mat[i,j]=en[(en.model==m)&(en.variant==v)].E_rel_endpoint_eV.quantile(.90)
    im=a.imshow(mat,cmap='viridis',aspect='auto'); a.set_xticks(range(3),[VARIANT_LABEL[v] for v in VARIANT_ORDER]); a.set_yticks(range(4),MODEL_ORDER); a.set_title('Relative relaxation landscape'); a.set_xlabel('Starting geometry'); a.set_ylabel('Model')
    for i in range(4):
        for j in range(3): a.text(j,i,f'{mat[i,j]:.3f}',ha='center',va='center',color='white' if mat[i,j]>.5*np.nanmax(mat) else '#111',fontsize=7)
    cb=fig.colorbar(im,ax=a,fraction=.045,pad=.03); cb.set_label('90th percentile $E-E_{min}$ within endpoint (eV)')
    # B: per-endpoint energy span distributions
    b=ax[1]
    for i,m in enumerate(MODEL_ORDER):
        q=span[span.model==m].E_span_endpoint_eV.to_numpy(); b.boxplot(q,positions=[i],widths=.5,patch_artist=True,showfliers=False,boxprops=dict(facecolor=COLORS[m],alpha=.30,color=COLORS[m]),medianprops=dict(color=COLORS[m],lw=1.5),whiskerprops=dict(color=COLORS[m]),capprops=dict(color=COLORS[m])); b.scatter(np.full(len(q),i)+np.random.default_rng(i).uniform(-.12,.12,len(q)),q,s=5,color=COLORS[m],alpha=.45)
    b.set_xticks(range(4),MODEL_ORDER,rotation=30,ha='right'); b.set_ylabel('Within-endpoint energy span (eV)'); b.set_title('Landscape ruggedness across 47 endpoints')
    # C: energy-span by endpoint type
    c=ax[2]
    types=list(TYPE_COLORS)
    for i,t in enumerate(types):
        for j,m in enumerate(MODEL_ORDER):
            q=span[(span.endpoint_type==t)&(span.model==m)].E_span_endpoint_eV
            if len(q): c.scatter(np.full(len(q), i+(j-1.5)*.16),q,s=15,color=COLORS[m],alpha=.75)
    c.set_xticks(range(len(types)),['same-gap\nlateral','coordination\nswitch','gap-to-layer\npenetration','other'],fontsize=6); c.set_ylabel('Energy span (eV)'); c.set_title('Dependence on endpoint class')
    handles=[mpl.lines.Line2D([],[],marker='o',ls='',color=COLORS[m],label=m,ms=4) for m in MODEL_ORDER]; c.legend(handles=handles,frameon=False,fontsize=5.7,loc='upper right')
    fig.text(.01,1.02,'Model-specific relative-energy landscapes (zero-point safe)',fontsize=11,weight='bold')
    for letter,aa in zip('ABC',ax): aa.text(-.16,1.08,letter,transform=aa.transAxes,fontsize=11,weight='bold')
    save(fig,'fig2_relative_energy_landscape')


def write_discussion(df, pairs, en, span):
    lines=['# Four-model MACE endpoint relaxation: publication discussion draft','',f'Dataset: {len(df)} endpoint×variant×model records (47 endpoints, 3 deterministic starts, 4 models). Scientific success is read from each summary JSON and requires optimizer convergence plus the stated finite-force criterion. Absolute total energies are not compared across models.','']
    lines.append('## Results')
    for m in MODEL_ORDER:
        g=df[df.model==m]; lines.append(f'- **{m}**: {int(g.success.sum())}/{len(g)} scientific-successful relaxations; median final maximum force {g.fmax.median():.4f} eV/Å and median FIRE steps among all records {g.steps.median():.0f}.')
    lines.append('')
    for m in MODEL_ORDER[1:]:
        q=pairs[pairs.model==m].cr_dist_to_ft600k_A
        lines.append(f'- Relative to FT-600K, **{m}** has median paired Cr displacement {q.median():.3f} Å (95th percentile {q.quantile(.95):.3f} Å; fraction ≤0.10 Å: {(q<=.10).mean():.1%}).')
    lines += ['', '## Landscape interpretation', 'The landscape panels use E_rel(endpoint, variant, model) = E_final − min_variant(E_final) for each endpoint and model. This removes model-dependent energy offsets and makes the three deterministic starting geometries visible as a local, discrete relaxation landscape. The common color scale is descriptive within the plotted normalization; it must not be read as a cross-model absolute-energy comparison. It is not a replacement for a continuous reaction-coordinate scan or a converged NEB.','', '## Discussion', 'The central result is substantial model dependence in the final basin coordinates, despite uniformly small residual forces for successful runs. Relative to FT-600K, the median Cr displacement is 0.225–0.284 Å and only 3.5–15.6% of paired structures fall within 0.10 Å. Therefore the present data do not support the stronger claim that endpoint pre-relaxation is model-insensitive. Instead, they support a sensitivity-screening interpretation: Foundation, Scratch and FT-MultiT generate valid locally converged structures, but they do not generally select the same Cr basin as FT-600K. The most divergent endpoints should be retained as an explicit uncertainty set and adjudicated by DFT. The three Scratch failures at unique_151 are also scientifically informative and should remain visible in the failure table rather than being imputed or removed. Differences in FIRE steps or residual forces are secondary diagnostics of local curvature and optimizer conditioning, not direct measures of model accuracy.','', '## Recommended manuscript wording', 'Across 47 Cr@Sb2Te3 endpoints and three deterministic starting geometries, the four MACE models were compared using paired final geometries, convergence, residual forces, and model-internal relative energies. Because MACE energy zero points are model-dependent, cross-model absolute total energies were not compared. Instead, each endpoint was assigned a discrete relaxation landscape after subtracting the minimum energy among its three starting variants within each model. All models except three Scratch runs at unique_151 satisfied the relaxation gate, but the final Cr coordinates showed appreciable model dependence: the median displacement from FT-600K was 0.225–0.284 Å, with only 3.5–15.6% of paired structures within 0.10 Å. Thus, the multi-model ensemble is best used to identify basin-sensitive endpoints and define DFT-priority cases, rather than to claim a unique model-independent endpoint geometry.']
    (OUT/'discussion_draft.md').write_text('\n'.join(lines)+'\n')


def main():
    df=load(); df.to_csv(OUT/'tidy_endpoint_relaxation_results.csv',index=False)
    summ,pairs,en,span=stats(df); fig_overview(df,pairs); fig_landscape(df,en,span); write_discussion(df,pairs,en,span)
    print(df.groupby(['model','success'],observed=True).size().to_string())
    print('outputs:', OUT)

if __name__=='__main__': main()
