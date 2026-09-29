#!/usr/bin/env python3
"""Publication figures for the measured 846-task multi-arm campaign.

Uses the local Plot Atlas style and primitives where their grammar matches the
analysis: joint_scatter, paired_dumbbell, and forest_effects. All source rows
are the deduplicated formal-task summaries pulled from Skipjack.
"""
from pathlib import Path
import glob, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ATLAS = Path('/Users/alina/Documents/Codex/plot_library')
import sys
sys.path.insert(0, str(ATLAS / 'src'))
from plot_atlas import apply_publication_style, get_palette, panel_label, save_figure, joint_scatter, paired_dumbbell, forest_effects

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'work/remote_results/runs'
OUT = ROOT / 'outputs/figures'
DATA = ROOT / 'outputs/figure_data'
OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)

SYSTEMS = ['Cr@Sb2Te3', 'V@Sb2Te3', 'Mn@Sb2Te3', 'Cr@Bi2Te3', 'V@Bi2Te3', 'Mn@Bi2Te3']
ARMS = ['base', 'perturb_a_dx020', 'perturb_b_tilt020']
ARM_LABELS = {'base': 'Base', 'perturb_a_dx020': 'Perturb A', 'perturb_b_tilt020': 'Perturb B'}
ARM_COLORS = {'base': '#1F4E79', 'perturb_a_dx020': '#D9772B', 'perturb_b_tilt020': '#168C8C'}

def load_rows():
    by_task = {}
    for path in glob.glob(str(RESULTS / 'task_*/mace_relax_summary.json')):
        row = json.loads(Path(path).read_text())
        by_task.setdefault(row['task']['task_id'], row)
    assert len(by_task) == 846, len(by_task)
    rows = []
    for row in by_task.values():
        t = row['task']
        site_num = int(re.search(r'(\d+)$', t['historical_site_id']).group(1))
        rows.append({
            'task_id': t['task_id'], 'system': t['system_id'], 'site': t['historical_site_id'],
            'site_num': site_num, 'arm': t['arm'], 'status': row['status'],
            'scientific_success': bool(row.get('scientific_success', False)),
            'converged': bool(row.get('optimizer_converged', False)),
            'energy': row['final_energy_eV'], 'force': row['final_max_force_eV_A'],
            'disp': row['max_displacement_A'], 'steps': row['optimizer_steps_completed'],
            'initial_force': row['initial_max_force_eV_A'], 'min_dist': row['final_min_distance_A'],
            'wall_s': row['wall_time_s'],
        })
    df = pd.DataFrame(rows).sort_values(['system','site_num','arm']).reset_index(drop=True)
    assert len(df) == 846 and df['scientific_success'].all()
    return df

def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float); rng = np.random.default_rng(seed)
    boots = np.array([np.median(rng.choice(values, len(values), replace=True)) for _ in range(2000)])
    return float(np.quantile(boots, .025)), float(np.quantile(boots, .975))

def save(fig, stem):
    save_figure(fig, OUT / f'{stem}.png')
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)

def fig1(df):
    apply_publication_style(8)
    pal = get_palette('isotope_sites')
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.4), constrained_layout=True)
    # A: scientific-health matrix
    ax = axes[0,0]
    mat = np.ones((len(SYSTEMS), len(ARMS)))
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i,s in enumerate(SYSTEMS):
        for j,a in enumerate(ARMS): ax.text(j,i,'47/47',ha='center',va='center',fontsize=8,fontweight='bold')
    ax.set(xticks=range(3), xticklabels=['Base','A','B'], yticks=range(6), yticklabels=SYSTEMS,
           xlabel='Perturbation arm', ylabel='System', title='Scientific success')
    ax.tick_params(length=0); panel_label(ax,'A',-.16,1.10)
    # B/C/E: distributions with all endpoint-level points
    for ax, col, ylabel, title, label in [(axes[0,1],'force','Final $F_{\\max}$ (eV/\\AA)','Force after relaxation','B'),
                                           (axes[0,2],'disp','Maximum displacement (\\AA)','Geometric response','C'),
                                           (axes[1,1],'steps','FIRE steps','Relaxation effort','E')]:
        positions=[]; labels=[]; vals=[]; colors=[]
        for si,s in enumerate(SYSTEMS):
            for ai,a in enumerate(ARMS):
                v=df.loc[(df.system==s)&(df.arm==a),col].to_numpy()
                positions.append(si*4+ai); labels.append((s,a)); vals.append(v); colors.append(ARM_COLORS[a])
        bp=ax.boxplot(vals, positions=positions, widths=.63, patch_artist=True, showfliers=False,
                      medianprops={'color':'black','lw':1.1}, whiskerprops={'lw':.8}, capprops={'lw':.8})
        for patch,color in zip(bp['boxes'],colors): patch.set_facecolor(color); patch.set_alpha(.70); patch.set_edgecolor('#333333')
        for p,v,c in zip(positions,vals,colors):
            rng=np.random.default_rng(p+5); ax.scatter(rng.normal(p,.055,len(v)),v,s=5,color=c,alpha=.25,edgecolors='none',rasterized=True)
        ax.set_xticks([si*4+1 for si in range(6)], SYSTEMS, rotation=42, ha='right')
        ax.set_ylabel(ylabel); ax.set_title(title); panel_label(ax,label,-.14,1.10)
        ax.grid(axis='y',color='#EAEAEA',lw=.5); ax.set_axisbelow(True)
    # D: paired energy response relative to base
    ax=axes[1,0]; wide=df.pivot_table(index=['system','site_num'],columns='arm',values='energy')
    system_palette = ['#1F4E79','#D9772B','#168C8C','#7A5195','#4C78A8','#B279A2']
    for si,s in enumerate(SYSTEMS):
        sub=wide.loc[s]; base=sub['base'].median()
        medians=[sub[a].median() for a in ARMS]
        ax.plot(range(3),medians,color=system_palette[si],lw=1.1,zorder=1)
        ax.scatter(range(3),medians,s=42,c=[ARM_COLORS[a] for a in ARMS],edgecolor='white',lw=.5,zorder=3)
    ax.axhline(0,color='#555555',ls=(0,(3,2)),lw=.8)
    # use energy shifts so each system is internally paired
    ax.clear()
    for si,s in enumerate(SYSTEMS):
        sub=wide.loc[s]; base=sub['base']; vals=[(sub[a]-base).median() for a in ARMS]
        ax.plot(range(3),vals,color=system_palette[si],lw=1.1,zorder=1,label=s); ax.scatter(range(3),vals,s=42,c=[ARM_COLORS[a] for a in ARMS],edgecolor='white',lw=.5,zorder=3)
    ax.axhline(0,color='#555555',ls=(0,(3,2)),lw=.8); ax.set_xticks(range(3),['Base','A','B']); ax.set_ylabel('$E_f-E_{f,\\mathrm{base}}$ (eV)'); ax.set_title('Paired energy response'); ax.legend(frameon=False,fontsize=6,ncol=2,loc='lower left'); panel_label(ax,'D',-.14,1.10)
    # F: initial force versus geometric displacement
    ax=axes[1,2]
    for a in ARMS:
        sub=df[df.arm==a]; ax.scatter(sub.initial_force,sub.disp,s=11,color=ARM_COLORS[a],alpha=.45,label=ARM_LABELS[a],edgecolors='white',linewidth=.2,rasterized=True)
    ax.set_xlabel('Initial $F_{\\max}$ (eV/\\AA)'); ax.set_ylabel('Maximum displacement (\\AA)'); ax.set_title('Force--geometry relation'); ax.legend(frameon=False,fontsize=7); panel_label(ax,'F',-.14,1.10)
    save(fig,'fig1_campaign_overview')

def fig2(df):
    apply_publication_style(8)
    pal = {'categorical':[ARM_COLORS['base'],ARM_COLORS['perturb_a_dx020'],ARM_COLORS['perturb_b_tilt020']], 'sequential':['#DCEAF2','#168C8C']}
    fig=plt.figure(figsize=(12.2,7.9)); gs=fig.add_gridspec(2,6,hspace=.55,wspace=.55)
    wide=df.pivot_table(index=['system','site_num'],columns='arm',values=['energy','disp','force','steps'])
    systems=np.array([s for s in SYSTEMS for _ in range(47)])
    # A/B: Atlas joint-scatter grammar
    for slot, xkey,ykey,xlab,ylab,label in [(gs[0,0:3],'energy','energy','$\\Delta E_A$ (eV)','$\\Delta E_B$ (eV)','A'),(gs[0,3:6],'disp','disp','$d_{\\max,A}$ (\\AA)','$d_{\\max,B}$ (\\AA)','B')]:
        if xkey=='energy':
            x=(wide['energy']['perturb_a_dx020']-wide['energy']['base']).to_numpy(); y=(wide['energy']['perturb_b_tilt020']-wide['energy']['base']).to_numpy()
        else:
            x=wide['disp']['perturb_a_dx020'].to_numpy(); y=wide['disp']['perturb_b_tilt020'].to_numpy()
        j=joint_scatter(fig,slot,x,y,{**pal,'categorical':['#1F4E79','#D9772B','#168C8C']},groups=systems,marginal='kde',xlabel=xlab,ylabel=ylab,point_size=13,alpha=.55,annotate=True,marginal_axes=False)
        j['main'].axline((0,0),slope=1,color='#666666',ls=(0,(3,2)),lw=.8); panel_label(j['main'],label,-.16,1.33)
    # C: forest effect with bootstrap CI for energy shifts
    ax=fig.add_subplot(gs[1,0:3]); effects=[]; lows=[]; highs=[]
    for a,seed in [('perturb_a_dx020',11),('perturb_b_tilt020',19)]:
        es=[]; lo=[]; hi=[]
        for s in SYSTEMS:
            sub=wide['energy'].loc[s]; vals=(sub[a]-sub['base']).to_numpy(); es.append(np.median(vals)); l,h=bootstrap_ci(vals,seed+SYSTEMS.index(s)); lo.append(l); hi.append(h)
        effects.append(es); lows.append(lo); highs.append(hi)
    forest_effects(ax,SYSTEMS,np.array(effects),np.array(lows),np.array(highs),['A vs Base','B vs Base'],[ARM_COLORS['perturb_a_dx020'],ARM_COLORS['perturb_b_tilt020']],xlabel='$\\Delta E_f$ (eV)',legend=True,xlim=(-.06,.06)); ax.set_title('Paired energy effects'); panel_label(ax,'C',-.16,1.10)
    # D: endpoint-level arm response heatmap
    ax=fig.add_subplot(gs[1,3:6]); mat=[]; ylabels=[]
    for s in SYSTEMS:
        sub=wide.loc[s]; mat.append((sub['disp']['perturb_a_dx020']-sub['disp']['base']).to_numpy()); ylabels.append(s+'  A-base')
        mat.append((sub['disp']['perturb_b_tilt020']-sub['disp']['base']).to_numpy()); ylabels.append(s+'  B-base')
    mat=np.array(mat); vmax=np.nanpercentile(np.abs(mat),98); im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(0,vmin=-vmax,vmax=vmax),interpolation='nearest')
    compact_labels=[s.replace('@','-').replace('2Te3','')+' '+arm for s in SYSTEMS for arm in ('A-base','B-base')]
    ax.set(yticks=np.arange(12),yticklabels=compact_labels,xticks=[0,9,19,29,39,46],xticklabels=['20','31','57','80','145','230'],xlabel='Historical endpoint site',title='Endpoint displacement response (\\AA)')
    cb=fig.colorbar(im,ax=ax,pad=.02,aspect=22); cb.set_label('$\\Delta d_{\\max}$ (\\AA)'); panel_label(ax,'D',-.16,1.10)
    save(fig,'fig2_perturbation_response')

def fig3(df):
    apply_publication_style(8)
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),constrained_layout=True)
    wide=df.pivot_table(index=['system','site_num'],columns='arm',values=['energy','force','disp','steps'])
    # A: energy response matrix
    for ax,a,title,label in [(axes[0,0],'perturb_a_dx020','A: final-energy response','A'),(axes[0,1],'perturb_b_tilt020','B: final-energy response','B')]:
        mat=[]
        for s in SYSTEMS:
            sub=wide.loc[s]; mat.append((sub['energy'][a]-sub['energy']['base']).to_numpy())
        mat=np.array(mat); lim=np.nanpercentile(np.abs(mat),98); im=ax.imshow(mat,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(0,-lim,lim),interpolation='nearest')
        ax.set(yticks=range(6),yticklabels=SYSTEMS,xticks=[0,9,19,29,39,46],xticklabels=['20','31','57','80','145','230'],xlabel='Historical endpoint site',title=title)
        cb=fig.colorbar(im,ax=ax,pad=.02,aspect=24); cb.set_label('$\\Delta E_f$ (eV)'); panel_label(ax,label,-.15,1.10)
    # C: cross-system interaction heatmap: difference in median displacement response A vs B
    ax=axes[1,0]; vals=[]
    for s in SYSTEMS:
        sub=wide.loc[s]; da=(sub['disp']['perturb_a_dx020']-sub['disp']['base']).median(); db=(sub['disp']['perturb_b_tilt020']-sub['disp']['base']).median(); vals.append([da,db,db-da])
    vals=np.array(vals); im=ax.imshow(vals,cmap='RdBu_r',norm=TwoSlopeNorm(0,np.nanmin(vals),np.nanmax(vals)),aspect='auto')
    ax.set(yticks=range(6),yticklabels=SYSTEMS,xticks=range(3),xticklabels=['A-base','B-base','B-A'],title='Host$\\times$dopant interaction');
    for i in range(6):
        for j in range(3): ax.text(j,i,f'{vals[i,j]:+.3f}',ha='center',va='center',fontsize=7,fontweight='bold')
    cb=fig.colorbar(im,ax=ax,pad=.02,aspect=24); cb.set_label('Displacement response (\\AA)'); panel_label(ax,'C',-.15,1.10)
    # D: median steps and force jointly, system colors and arm markers
    ax=axes[1,1]
    for a,marker in zip(ARMS,['o','s','^']):
        sub=df[df.arm==a].groupby('system').agg(steps=('steps','median'),force=('force','median')).reindex(SYSTEMS)
        ax.scatter(sub.steps,sub.force,s=52,marker=marker,color=ARM_COLORS[a],edgecolor='white',lw=.6,label=ARM_LABELS[a])
        for x,y,s in zip(sub.steps,sub.force,SYSTEMS): ax.annotate(s.split('@')[0]+'-'+s.split('@')[1].replace('2Te3',''),(x,y),xytext=(3,3),textcoords='offset points',fontsize=5.8)
    ax.set(xlabel='Median FIRE steps',ylabel='Median final $F_{\\max}$ (eV/\\AA)',title='Relaxation effort and force'); ax.legend(frameon=False,fontsize=7); panel_label(ax,'D',-.15,1.10)
    save(fig,'fig3_interaction_and_site_maps')

def main():
    df=load_rows()
    df.to_csv(DATA/'endpoint_level_summary.csv',index=False)
    df.to_json(DATA/'endpoint_level_summary.json',orient='records',indent=2)
    fig1(df); fig2(df); fig3(df)
    print(json.dumps({'n_rows':len(df),'figures':sorted(p.name for p in OUT.glob('fig*.pdf')),'data':str(DATA)}))

if __name__ == '__main__': main()
