#!/usr/bin/env python3
"""Build manuscript tables from the selected figure releases, without fitting an MLIP.

This analysis removes the known all-zero legacy feature before every metric.
It does not relabel model output features as native MACE representations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.manifold import TSNE, trustworthiness
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/alina/Documents/ChatGPT/Wiley-MigrationBench')
BACKUP = Path('/Users/alina/Downloads/MigrationBench_ALL_CANVA_and_publication_figures_backup_20261102')
REV = HERE / 'paper/revision1'
DATA = REV / 'data_processed/current_release'
DATA.mkdir(parents=True, exist_ok=True)
TABLES = REV / 'tables'
LEDGER = []

def preserve(path, name=None):
    dst = DATA / (name or path.name)
    shutil.copy2(path, dst)
    LEDGER.append(dict(file=str(dst.relative_to(HERE)), source=str(path),
                       sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dst

def table(name, header, rows, columns=None):
    columns = columns or ('l' + 'r' * (len(header)-1))
    out = ['\\begin{tabular}{'+columns+'}', '\\toprule',
           ' & '.join(header)+r' \\', '\\midrule']
    out += [' & '.join(map(str,r))+r' \\' for r in rows]
    out += ['\\bottomrule', '\\end{tabular}']
    (TABLES/name).write_text('\n'.join(out)+'\n')

def main():
    # Selected-file byte identities; images are never regenerated here.
    for f in sorted((REV/'figures/main').rglob('*.pdf')):
        LEDGER.append(dict(file=str(f.relative_to(HERE)),source='user-selected figure',
                           sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    parity = preserve(ROOT/'analysis/rmse_train_test_20261026/parity_summary.csv')
    preserve(ROOT/'analysis/rmse_train_test_20261026/MODEL_REGISTRY.tsv','parity_model_registry.tsv')
    p = pd.read_csv(parity)
    rows = []
    for model in ['Scratch-600K','FT-600K','Scratch-MultiT','FT-MultiT']:
        r = p[(p.model==model)&(p.split=='test')]
        rows.append([model, int(r.n_frames.iloc[0]),
                     f'${r.energy_rmse_meV_per_atom.mean():.2f}\\pm{r.energy_rmse_meV_per_atom.std():.2f}$',
                     f'${r.force_rmse_meV_A.mean():.2f}\\pm{r.force_rmse_meV_A.std():.2f}$'])
    table('current_pointwise.tex',['Model','Test frames / seed','Energy','Force'],rows)

    md = preserve(ROOT/'analysis/md_1ns_updated_publication_figure_20261025/MD15_FIGURE_DATA_R06G.tsv')
    preserve(ROOT/'analysis/md_1ns_updated_publication_figure_20261025/MD15_STATUS_AUDIT_R02.tsv')
    preserve(ROOT/'analysis/md_1ns_updated_publication_figure_20261025/md15_analysis_r03/analysis_summary.tsv','md15_analysis_summary.tsv')
    m = pd.read_csv(md,sep='\t')
    rows=[]
    for _, r in m.iterrows():
        name = r.model_id.replace('_','\\_')
        outcome='Target reached' if r.last_time_ps>=1000 else 'Threshold reached'
        rows.append([name, f'{r.last_time_ps:g}', f'{r.last_time_ps-200:g}',outcome])
    table('current_md_horizons.tex',['Checkpoint / training seed','Clock (ps)','Continuation (ps)','Outcome'],rows,'lrrl')

    neb_input = DATA/'fig5_corrected_input.tsv'
    neb = preserve(neb_input,'fig5_plotted_values.tsv') if neb_input.exists() else preserve(BACKUP/'additional_publication_figures/NEB_dotmatrix_v3_latest_20261101/plotted_values.tsv','fig5_plotted_values.tsv')
    preserve(BACKUP/'additional_publication_figures/NEB_dotmatrix_v3_latest_20261101/audit.json','fig5_render_audit.json')
    d=pd.read_csv(neb,sep='\t')
    assert len(d)==120
    assert np.allclose(d.absolute_error_eV, np.abs(d.predicted_value_eV-d.dft_reference_eV),atol=1e-10)
    d['family']=d.row_id.str.split(' / ').str[0].str.strip()
    grouped=d.groupby(['family','column_id']).agg(n=('predicted_value_eV','size'),
       predicted_mean=('predicted_value_eV','mean'),predicted_sd=('predicted_value_eV','std'),
       error_mean=('absolute_error_eV','mean'),error_sd=('absolute_error_eV','std')).reset_index()
    grouped.to_csv(DATA/'fig5_family_statistics.csv',index=False)
    rows=[]
    for family in ['MP-0B3','OMAT-0','MPA0','FT-600K','FT-MultiT','Scratch-600K','Scratch-MultiT']:
        r=grouped[grouped.family==family].set_index('column_id')
        vals=[]
        for task in ['migration_in_gap_fixed','migration_within_ql_fixed','sliding_pristine_fixed','sliding_cr_in_gap_fixed']:
            v=r.loc[task]
            vals.append(f'${v.error_mean:.2f}$' if v.n==1 else f'${v.error_mean:.2f}\\pm{v.error_sd:.2f}$')
        rows.append([family]+vals)
    table('current_neb_errors.tex',['Model','In gap','Within QL','Pristine sliding','Cr sliding'],rows)

    for name in ['cr_three_path_plotted_profiles_R05.tsv','cr_three_path_coverage_audit_R05.tsv',
                 'cr_three_path_fixed_vs_self_2x3_R05.manifest.json']:
        preserve(ROOT/'analysis/cr_interlayer_sliding_grouped_20261025/results'/name)
    canonical=ROOT/'analysis/task3_endpoint_only_neb/scratch_600k_migration_20261101/processed'
    for f in canonical.glob('*.tsv'):
        preserve(f,'scratch600k_'+f.name)

    metrics=preserve(ROOT/'analysis/shap_full_campaign_20261101/results/shap_seed_metrics.csv')
    for n in ['shap_family_aggregate.csv','shap_seed_feature_importance.csv','analysis_manifest.json']:
        preserve(ROOT/'analysis/shap_full_campaign_20261101/results'/n,'shap_'+n if n=='analysis_manifest.json' else n)
    s=pd.read_csv(metrics);rows=[]
    for model in ['FT-600K','FT-MultiT','Scratch-600K','Scratch-MultiT']:
        g=s[s.model_family==model]
        rows.append([model,f'${g.r2_5fold_cv.mean():.3f}\\pm{g.r2_5fold_cv.std():.3f}$',
          f'${g.mae_5fold_cv_meV_per_atom.mean():.2f}\\pm{g.mae_5fold_cv_meV_per_atom.std():.2f}$'])
    table('current_shap_fidelity.tex',['Model','Five-fold $R^2$','MAE (meV atom$^{-1}$)'],rows)

    features=ROOT/'analysis/legacy_fig6_phate_reproduction_20261101/updated_campaign_20261102/source/fig6_full_campaign_features.npz'
    payload=np.load(features,allow_pickle=True)
    x=payload['features'];labels=payload['model_ids'].astype(str)
    assert x.shape==(927,6192) and np.isfinite(x).all()
    assert np.count_nonzero(x[:,-1])==0
    x=x[:,:-1] # The legacy placeholder never enters any distance or projection.
    z=StandardScaler().fit_transform(x)
    hist=x[:,6151:6181] # common 30-bin neighbour histogram, before per-run SOAP PCA.
    assert hist.shape==(927,30) and np.all(hist>=0)
    scores=[]
    spaces=[('Original output features (6191)',x,'euclidean'),
            ('Original output features, cosine',x,'cosine'),
            ('Z-scored output features',z,'euclidean'),
            ('Common neighbour histogram (30)',hist,'euclidean'),
            ('Z-scored neighbour histogram',StandardScaler().fit_transform(hist),'euclidean')]
    for name, arr, metric in spaces:
        v=float(silhouette_score(arr,labels,metric=metric))
        scores.append(dict(space=name,dimensions=arr.shape[1],metric=metric,silhouette=v))
    pd.DataFrame(scores).to_csv(DATA/'fig6_current_original_space_metrics.csv',index=False)
    # Direct metric on the same (constant-column-free) feature grammar as Fig. 6.
    # The sweep is visualization sensitivity, never an uncertainty estimate for MD.
    sweep=[]
    for perplexity in [5,30,50]:
        for seed in [0,42,123]:
            embedding=TSNE(n_components=2,perplexity=perplexity,random_state=seed,
                           init='pca',max_iter=1000,n_jobs=1).fit_transform(x)
            sweep.append(dict(perplexity=perplexity,seed=seed,
                silhouette=float(silhouette_score(embedding,labels)),
                trustworthiness=float(trustworthiness(x,embedding,n_neighbors=5))))
    pd.DataFrame(sweep).to_csv(DATA/'fig6_current_tsne_sensitivity.csv',index=False)
    table('current_original_space.tex',['Feature space','Dimensions','Silhouette'],
          [[r['space'],r['dimensions'],f"{r['silhouette']:.3f}"] for r in scores],'lrr')
    table('current_tsne_sensitivity.tex',['Perplexity','Seed','Silhouette','Trustworthiness'],
          [[r['perplexity'],r['seed'],f"{r['silhouette']:.3f}",f"{r['trustworthiness']:.3f}"] for r in sweep])
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','pdf.fonttype':42})
    fig, axes=plt.subplots(1,2,figsize=(9.3,3.3),gridspec_kw={'width_ratios':[1.4,1]})
    axes[0].barh(range(len(scores)),[r['silhouette'] for r in scores],color=['#8FC7AB','#8FC7AB','#29A7B6','#327FB1','#327FB1'])
    axes[0].set_yticks(range(len(scores)),['Raw outputs','Cosine outputs','Z-scored outputs','Neighbour histogram','Z-scored histogram'])
    axes[0].invert_yaxis();axes[0].axvline(0,color='#303840',lw=.8)
    axes[0].set_xlabel('Silhouette in the unprojected feature space')
    for p in [5,30,50]:
        a=[r['silhouette'] for r in sweep if r['perplexity']==p]
        axes[1].scatter([p]*len(a),a,s=35,color='#327FB1',alpha=.8)
    axes[1].set_xlabel('t-SNE perplexity');axes[1].set_ylabel('Projection silhouette')
    axes[1].set_xticks([5,30,50])
    for ax,l in zip(axes,['a','b']):
        ax.text(-.12,1.04,l,transform=ax.transAxes,fontweight='bold',fontsize=12)
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(REV/'figures/figS_current_feature_audit.pdf',bbox_inches='tight')
    fig.savefig(REV/'figures/figS_current_feature_audit.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    summary=dict(feature_source=str(features),feature_sha256=hashlib.sha256(features.read_bytes()).hexdigest(),
        feature_shape_before=[927,6192],feature_shape_after=[927,6191],removed_column=6191,
        removed_column_all_zero=True,original_space=scores,projection_sweep=sweep,
        caveat='Output features include energy offsets and forces; SOAP PCA is fitted separately per trajectory. These are not native MACE hidden states. Neighbour histograms use a common coordinate basis. No causal attribution or acquisition rule is inferred.')
    (DATA/'current_analysis_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    (DATA/'source_ledger.json').write_text(json.dumps(LEDGER,indent=2)+'\n')
    print(json.dumps({'original_space':scores,'sweep_range':[min(r['silhouette'] for r in sweep),max(r['silhouette'] for r in sweep)],'source_files':len(LEDGER)},indent=2))

if __name__=='__main__': main()
