"""Analyze and plot all local MACE foundation models on fresh endpoint-only paths."""
from __future__ import annotations
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import apply_publication_style, get_palette, panel_label, save_figure, timecourse
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neb_plot_style import QE_COLOR, extract_qe_neb_uncertainty, model_colors

RESULTS = Path("/tmp/migrationbench_foundation_model_results_20261002")
INPUTS = Path("/tmp/migrationbench_foundation_base_linear_remote_20261002")
QE_IMAGE = Path("analysis/task3_endpoint_only_neb/exploratory_energy_audit/processed/mace_dft_relative_image_metrics.csv")
OUT = Path("analysis/task3_endpoint_only_neb/exploratory_energy_audit/full_foundation_models")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "mace-large-density-agnesi-stress", "mace-medium-density-agnesi-stress",
    "mace-small-density-agnesi-stress", "mace_agnesi_medium",
    "mace-mh-0", "mace-mh-1", "mace-mp-0b3-medium",
    "mace-omat-0-medium", "mace-omat-0-small",
]
LABELS = {
    "mace-large-density-agnesi-stress":"Agnesi-L-dens", "mace-medium-density-agnesi-stress":"Agnesi-M-dens",
    "mace-small-density-agnesi-stress":"Agnesi-S-dens", "mace_agnesi_medium":"Agnesi-M",
    "mace-mh-0":"MH-0", "mace-mh-1":"MH-1", "mace-mp-0b3-medium":"MP-0B3",
    "mace-omat-0-medium":"OMAT-0-M", "mace-omat-0-small":"OMAT-0-S",
}
COLORS = model_colors(MODEL_ORDER)

def load_rows():
    all_rows=[]
    for model in MODEL_ORDER:
        p=RESULTS/model/"mace_single_point_results.csv"
        d=pd.read_csv(p)
        d["model_id"]=model
        d["path_id"]=d.sample_file.str.replace("__base.extxyz", "", regex=False)
        d["image_index"]=d.frame_index+1
        d["relative_energy_eV"]=d.energy_eV-d.groupby("sample_file").energy_eV.transform("first")
        all_rows.append(d)
    models = pd.concat(all_rows, ignore_index=True)
    qe = pd.read_csv(QE_IMAGE)
    qe = qe[qe.variant.eq("base")][["path_id", "image_index", "dft_relative_energy_eV"]].drop_duplicates()
    qe = qe.rename(columns={"dft_relative_energy_eV": "qe_relative_energy_eV"})
    d = models.merge(qe, on=["path_id", "image_index"], how="left", validate="many_to_one")
    uncertainty = extract_qe_neb_uncertainty(
        d.path_id.unique(),
        remote_root=Path(__file__).resolve().parent / "remote_snapshot",
        n_images=10,
        n_tail_tables=5,
    )
    d = d.merge(
        uncertainty[["path_id", "image_index", "qe_energy_std_eV", "qe_energy_range_eV", "qe_force_error_eV_A",
                     "qe_last_iteration", "qe_n_complete_tables", "qe_uncertainty_status"]],
        on=["path_id", "image_index"], how="left", validate="many_to_one",
    )
    if d.qe_relative_energy_eV.isna().any():
        raise ValueError("QE reference trace did not align with every Foundation-model image")
    return d

def short_path(s):
    return s.replace("2L-","").replace("__neb-"," ").replace("__neb"," ").replace("gap-diffuse","gap-diff.").replace("gap2Ql","gap2QL")

def make_tables(d):
    d.to_csv(OUT/"all_foundation_model_image_profiles.csv", index=False)
    barriers=d.groupby(["model_id","path_id"]).relative_energy_eV.max().reset_index(name="barrier_like_eV")
    summary=barriers.groupby("model_id").barrier_like_eV.agg(median="median",mean="mean",minimum="min",maximum="max").reset_index()
    force=d.groupby("model_id").max_force_eV_A.agg(median="median",maximum="max").reset_index()
    pivot=d.pivot_table(index=["path_id","image_index"],columns="model_id",values="relative_energy_eV")
    ensemble=pivot.median(axis=1)
    spread=pd.DataFrame({"model_id":pivot.columns,"rmse_to_model_median":[float(np.sqrt(np.mean((pivot[c]-ensemble)**2))) for c in pivot.columns],"mae_to_model_median":[float(np.mean(np.abs(pivot[c]-ensemble))) for c in pivot.columns],"max_abs_to_model_median":[float(np.max(np.abs(pivot[c]-ensemble))) for c in pivot.columns]})
    tab=summary.merge(force,on="model_id",suffixes=("_barrier","_force")).merge(spread,on="model_id")
    tab["model_label"]=tab.model_id.map(LABELS)
    tab=tab[["model_id","model_label","median_barrier","mean","minimum","maximum_barrier","median_force","maximum_force","rmse_to_model_median","mae_to_model_median","max_abs_to_model_median"]]
    tab.columns=["model_id","model_label","median_barrier_like_eV","mean_barrier_like_eV","min_barrier_like_eV","max_barrier_like_eV","median_max_force_eV_A","max_force_eV_A","rmse_to_model_median_eV","mae_to_model_median_eV","max_abs_to_model_median_eV"]
    tab=tab.sort_values("median_barrier_like_eV")
    tab.to_csv(OUT/"all_foundation_model_summary.csv",index=False)
    (OUT/"all_foundation_model_summary.tex").write_text(tab.to_latex(index=False,index_names=False,escape=False,float_format=lambda x:f"{x:.4f}"))
    return tab

def render_profiles(d):
    apply_publication_style(8.2)
    paths=sorted(d.path_id.unique())
    fig,axes=plt.subplots(2,4,figsize=(8.3,5.0),sharey=False)
    for ax,path in zip(axes.flat,paths):
        q=d[d.path_id.eq(path)]
        xs=np.arange(1,11)
        q_qe = q.drop_duplicates("image_index").sort_values("image_index")
        qe = q_qe.qe_relative_energy_eV.to_numpy()
        ys=[qe] + [q[q.model_id.eq(m)].sort_values("image_index").relative_energy_eV.to_numpy() for m in MODEL_ORDER]
        colors=[QE_COLOR] + COLORS[:len(MODEL_ORDER)]
        labels=["QE-NEB (screened)"] + [LABELS[m] for m in MODEL_ORDER]
        timecourse(ax,xs,ys,labels,colors,markers=True,legend=False,line_width=1.05,marker_size=2.65,
                   xlabel="Image index",ylabel="ΔE (eV)")
        ax.set_title(short_path(path),fontsize=8,pad=3)
        for s in ax.spines.values(): s.set_linewidth(.8)
    handles=[plt.Line2D([],[],color=QE_COLOR,lw=1.05,ls="-",marker="o",ms=3.0,
                        mec="#202020",mfc="#202020",label="QE-NEB (screened)")]
    handles += [plt.Line2D([],[],color=COLORS[i],lw=1.05,ls="-",marker="o",ms=2.7,
                           mec=COLORS[i],mfc=COLORS[i],label=LABELS[m]) for i,m in enumerate(MODEL_ORDER)]
    axes.flat[-1].legend(handles=handles,frameon=False,fontsize=5.7,loc="upper left",bbox_to_anchor=(1.02,1.02),ncol=1)
    for i,ax in enumerate(axes.flat): panel_label(ax,chr(ord('a')+i),x=-.16,y=1.12)
    fig.subplots_adjust(left=.08,right=.84,bottom=.11,top=.94,wspace=.40,hspace=.52)
    save_figure(fig,OUT/"figure_all_foundation_model_path_profiles.pdf")
    fig.savefig(OUT/"figure_all_foundation_model_path_profiles.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

def render_spread(d,tab):
    apply_publication_style(8.2)
    fig,axs=plt.subplots(1,2,figsize=(7.0,3.2),gridspec_kw={"width_ratios":[1.15,1]})
    t=tab.sort_values("rmse_to_model_median_eV")
    y=np.arange(len(t))
    axs[0].barh(y,t.rmse_to_model_median_eV,color=[COLORS[MODEL_ORDER.index(m)] for m in t.model_id],edgecolor="none")
    axs[0].set_yticks(y,t.model_label); axs[0].set_xlabel("RMSE to model-median profile (eV)"); axs[0].invert_yaxis()
    axs[0].set_title("Model-to-ensemble spread",fontsize=9); panel_label(axs[0],"a")
    for s in axs[0].spines.values(): s.set_linewidth(.8)
    # Per-path envelope: model-wise barrier-like max, then across-model median/min/max.
    path_model=d.groupby(["path_id","model_id"]).relative_energy_eV.max().unstack()
    order=sorted(path_model.index)
    med=path_model.median(axis=1).reindex(order).to_numpy()
    lo=path_model.min(axis=1).reindex(order).to_numpy()
    hi=path_model.max(axis=1).reindex(order).to_numpy()
    xpos=np.arange(len(order)); axs[1].errorbar(xpos,med,yerr=[np.array(med)-np.array(lo),np.array(hi)-np.array(med)],fmt="o",color="#3A82B4",ecolor="#999999",capsize=2,lw=1.0)
    axs[1].set_xticks(xpos,[short_path(x) for x in order],rotation=55,ha="right"); axs[1].set_ylabel("barrier-like ΔE (eV)"); axs[1].set_title("Across-model path envelope",fontsize=9); panel_label(axs[1],"b")
    for ax in axs:
        for s in ax.spines.values(): s.set_linewidth(.8)
        for lab in (*ax.get_xticklabels(),*ax.get_yticklabels()): lab.set_fontweight("bold")
    fig.subplots_adjust(left=.22,right=.98,bottom=.30,top=.88,wspace=.48)
    save_figure(fig,OUT/"figure_all_foundation_model_spread.pdf")
    fig.savefig(OUT/"figure_all_foundation_model_spread.png",dpi=300,bbox_inches="tight")
    plt.close(fig)

def main():
    d=load_rows(); tab=make_tables(d); render_profiles(d); render_spread(d,tab)
    (OUT/"all_foundation_model_report.md").write_text("""# Complete local Foundation-model comparison (endpoint-only diagnostic)\n\nThis matrix evaluates all nine models in `/Users/alina/Downloads/mace-foundation-models` on eight fresh endpoint-only linear paths (10 frames/path; 80 frames/model). Energies are relative within each path and model. The result is a model-to-model profile comparison, not a DFT-accuracy validation and not a converged NEB barrier. MH-0 and MH-1 use the explicit `omat_pbe` head. The black QE trace uses the corrected `.path` Hartree-to-eV conversion, but its geometries are the retained QE path frames and therefore are not geometry-identical to these fresh linear inputs.\n\nThe profile figure shows one line per model for every path, with explicit markers and line segments. The model colors use related Morandi shades within each family. The NEB image-table third column is a force/path residual in eV/A and is retained as a separate audit field rather than being added to the energy as if it were a standard error. The companion residual-audit figure reports that quantity against each path's `path_thr`. The spread figure reports each model's RMSE to the across-model median profile and the path-level barrier-like envelope. For a quantitative QE comparison on identical geometries, use `../corrected_dft_foundation_models/`.\n""")
    paper_text = """# Manuscript-ready figure text\n\n## Figure caption\n\n**Figure X | Energy-profile dispersion across the complete local MACE foundation-model set.** Relative single-point energies are shown along eight endpoint-only linear paths constructed from fresh QE endpoint inputs (Cu/Li/Na in the QLs, gap-diffuse, and gap2QL/gap2Ql variants; ten images per path). Each colored line denotes one of the nine models in `/Users/alina/Downloads/mace-foundation-models`; MH-0 and MH-1 were evaluated with the explicit `omat_pbe` head. (a–h) Individual path profiles. The companion spread panel ranks models by their root-mean-square deviation from the across-model median profile and summarizes the path-wise range of the maximum relative energy across models. The black QE trace uses the corrected `.path` Hartree-to-eV conversion but is not geometry matched to these fresh linear inputs.\n\n## Results and discussion\n\nThe complete nine-model comparison shows that the apparent agreement of a two-model plot is not representative of the full foundation-model ensemble. Agreement is path dependent: the three gap-diffuse/gap2QL-type families remain relatively close, whereas the QLs paths expose very large model dependence. The path-wise median barrier-like maxima are 0.13–0.85 eV for the six gap-based paths but 16.53 eV (Li QLs) and 37.44 eV (Cu QLs) for the two direct QLs linear interpolations. These very large values are diagnostic of endpoint-only interpolation and likely short-range/overlap sensitivity, not evidence for physical migration barriers.\n\nAcross all 720 model–image evaluations, the closest profile to the ensemble median is Agnesi-M-dens (RMSE 0.151 eV). The most dispersed is Agnesi-S-dens (2.908 eV), followed by OMAT-0-M (1.122 eV), OMAT-0-S (1.095 eV), Agnesi-M (0.825 eV), and MH-0 (0.816 eV). Thus the models do not all differ equally: the spread is concentrated in particular model families and path classes. The result supports using the full model matrix as a screening/uncertainty diagnostic, but it does not identify a physically correct barrier or establish DFT accuracy. The controlled same-geometry nine-model comparison on the retained QE/NEB path frames is provided separately in `../corrected_dft_foundation_models/`.\n\n## Scope and reproducibility\n\nAll nine local model files were evaluated successfully (9 × 80 = 720 frames). Energies are normalized independently within each path to the first image. The inputs are fresh endpoint-only linear interpolations generated from remote QE endpoint files; this is intentionally a model-disagreement experiment and must not be labeled as a converged NEB calculation.\n"""
    (OUT/"all_foundation_model_paper_text.md").write_text(paper_text)
    models=[]
    for model in MODEL_ORDER:
        p=Path("/Users/alina/Downloads/mace-foundation-models")/(model+".model")
        models.append({"model_id":model,"file":str(p),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"head":"omat_pbe" if model in {"mace-mh-0","mace-mh-1"} else None})
    manifest={"analysis":"all_local_foundation_models_endpoint_only_linear_base","n_models":len(MODEL_ORDER),"n_paths":int(d.path_id.nunique()),"n_images_per_path":10,"n_frames_per_model":int(len(d)/len(MODEL_ORDER)),"models":models,"input_manifest":str(INPUTS/"manifest.json")}
    (OUT/"foundation_model_matrix_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"models":len(MODEL_ORDER),"frames":len(d),"output":str(OUT)},indent=2))

if __name__=="__main__": main()
