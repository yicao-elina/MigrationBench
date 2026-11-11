#!/usr/bin/env python3
"""Create a minimal, hash-checked deliverable from the completed manuscript."""
from pathlib import Path
import hashlib
import json
import re
import csv
import zipfile

ROOT=Path(__file__).resolve().parent
PAPER=ROOT/'paper'
REV=PAPER/'revision1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
active=set()
def include(p):
    p=p.resolve()
    if p in active:return
    if not p.is_file():raise FileNotFoundError(p)
    active.add(p)
    if p.suffix=='.tex' and (ROOT/'source_snapshot') not in p.parents:
        text=p.read_text()
        for rel in re.findall(r'\\input\{([^}]+)\}',text):
            q=PAPER/rel
            if not q.suffix:q=q.with_suffix('.tex')
            include(q)
        for rel in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
            include(PAPER/rel)
for n in ['sn-article-final','sn-article-SI','sn-article-clean','sn-article-SI-clean','response_to_reviewers']:
    include(REV/(n+'.tex'))
    include(REV/(n+'.pdf'))
    include(REV/(n+'.bbl'))
include(REV/'references-camera-ready.bib')
include(PAPER/'sn-jnl.cls')
include(PAPER/'sn-mathphys-num.bst')
for folder in [ROOT/'reproducibility',ROOT/'source_snapshot',REV/'data_processed/current_release']:
    for p in folder.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:include(p)
for p in (REV/'appendix/data').glob('*'):
    if p.is_file():include(p)
for p in (REV/'appendix/scripts').glob('*.py'):
    include(p)
for n in ['dft_neb_barriers.csv','path_registry.csv']:
    p=REV/'data_processed'/n
    if p.is_file():include(p)
for n in ['README.md','READINESS_NOTE.md','FIGURE_PANEL_MAP.md','build.py',
          'audit_current_figures.py','write_response.py','apply_to_original.py',
          'requirements-current.txt']:
    include(ROOT/n)

aux={}
for prefix,n in [('', 'sn-article-final'),('M-','sn-article-final'),('SI-','sn-article-SI')]:
    s=(REV/(n+'.aux')).read_text()
    for key,num,page in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',s):
        aux[prefix+key]=(num,page)
def braced(s,pos):
    assert s[pos]=='{'
    depth=1;i=pos+1
    while depth:
        if s[i]=='{' and s[i-1]!='\\':depth+=1
        if s[i]=='}' and s[i-1]!='\\':depth-=1
        i+=1
    return s[pos+1:i-1]
def plain(s):
    s=re.sub(r'\\sitab\{([^}]+)\}',lambda m:'Supplementary Table '+aux.get('SI-'+m[1],('?',))[0],s)
    s=re.sub(r'\\sifig\{([^}]+)\}',lambda m:'Supplementary Fig. '+aux.get('SI-'+m[1],('?',))[0],s)
    s=re.sub(r'\\ref\{([^}]+)\}',lambda m:aux.get(m[1],aux.get('SI-'+m[1],('?',)))[0],s)
    for a,b in [(r'\ce{Sb2Te3}','Sb₂Te₃'),(r'\ce{Bi2Te3}','Bi₂Te₃'),(r'\AA','Å'),(r'\pm','±'),(r'\times','×'),('~',' '),(r'\%','%'),(r'\_','_')]:s=s.replace(a,b)
    for _ in range(4):
        s=re.sub(r'\\textbf\{([^{}]*)\}',r'**\1**',s)
        s=re.sub(r'\\(?:emph|texttt|mathrm|rm|text)\{([^{}]*)\}',r'\1',s)
    return re.sub(r'\s+',' ',s).strip()
captions=['# Figure captions','Captions are also embedded beside their corresponding figures in the manuscript and SI.','']
for file,prefix in [(REV/'sn-article-final.tex',''),(REV/'sn-article-SI.tex','SI-'),
                    (REV/'appendix/endpoint_perturbation_appendix.tex','SI-')]:
    s=file.read_text()
    for env in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}',s,re.S):
        block=env.group();c=block.find(r'\caption{')
        if c<0:continue
        label=re.search(r'\\label\{([^}]+)\}',block)
        num=aux.get(prefix+label[1],('?',))[0]
        captions += ['## Figure '+num,'',plain(braced(block,c+len(r'\caption'))),'']
(ROOT/'FIGURE_CAPTIONS.md').write_text('\n'.join(captions))
include(ROOT/'FIGURE_CAPTIONS.md')

coverage=[
 ('A1','Operators and incompatible references','sec:migration','Corrected; kinetic interpretation restricted','SCF reference convergence remains'),
 ('A2','Overlap and unreproduced point estimate','sec:models','Grouped control reported; 0.16 eV replaced','25-image audit scope is legacy'),
 ('A3','Training-seed uncertainty','sec:migration','Three-seed values reported; chance claim removed','Diagnostic reference rather than validated barrier'),
 ('A4','Generality','sec:discussion','Restricted to primary material and MACE routes','Other dopants remain diagnostics'),
 ('A5','Target-data control','sec:models','Matched full 1786-frame pair; 89-frame sensitivity separated','Architecture and optimization remain confounded'),
 ('A6','Interpretability','sec:representations','Current original-space metrics; surrogate scope and fidelity','No causal internal-mechanism claim'),
 ('A7','Related work and prior report','sec:introduction','Nearest neighbours cited; delta corrected','Earlier sliding and representation not called new'),
 ('C8','Fabricated zero forces','sec:models','Fail-closed exporters; hash-bound audit included','Audit restricted to declared files'),
 ('C9','MD duration and sampling','sec:md','Actual 15-model horizons; time reconstructed from steps','One common state; no independent thermal replicas'),
 ('C10','Grouped train/test','sec:models','Grouped parity and counts; display centering disclosed','Cross-trajectory similarity is separate'),
 ('C11','Projection-space silhouette','sec:representations','Recomputed current 927-frame original-space metrics','Output features are not native hidden states'),
 ('C12','Inactive feature','sec:diagnostics','Constant column removed in recomputed workflow','Supplied reconstruction retains disclosed legacy provenance'),
 ('C13','Reproducibility','sec:diagnostics','Actual four-table SHAP rerun matches numerically','No claim that legacy 150-frame outputs were regenerated')]
with (ROOT/'REVIEWER_COVERAGE.tsv').open('w',newline='') as stream:
    w=csv.writer(stream,delimiter='\t')
    w.writerow(['index','issue','main_section','main_page','disposition','remaining_scope'])
    for idx,issue,label,status,scope in coverage:
        num,page=aux[label]
        w.writerow([idx,issue,num,page,status,scope])
include(ROOT/'REVIEWER_COVERAGE.tsv')

base={p.name:sha(p) for p in (ROOT/'source_snapshot').glob('*.tex')}
base['figures/main/fig5_NEB_main_figure (2).pdf']=sha(ROOT/'source_snapshot/fig5_supplied_original.pdf')
updated=['sn-article-final.tex','sn-article-SI.tex','response_to_reviewers.tex',
 'sn-article-clean.tex','sn-article-SI-clean.tex','references-camera-ready.bib']
updated += [str(p.relative_to(REV)) for p in sorted(REV.glob('tables/current_*.tex'))]
updated += [str(p.relative_to(REV)) for p in sorted((REV/'data_processed/current_release').rglob('*')) if p.is_file()]
updated += ['figures/main/fig5_NEB_main_figure (2).pdf','figures/figS_current_feature_audit.pdf','figures/figS_current_feature_audit.png']
updated += [n+'.pdf' for n in ['sn-article-final','sn-article-SI','sn-article-clean','sn-article-SI-clean','response_to_reviewers']]
include(REV/'figures/figS_current_feature_audit.png')
replacement={'original_source_hashes':base,'updated_revision_files':updated}
(ROOT/'replacement_manifest.json').write_text(json.dumps(replacement,indent=2)+'\n')
include(ROOT/'replacement_manifest.json')
manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(active)]
(ROOT/'DELIVERY_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
include(ROOT/'DELIVERY_MANIFEST.json')
zip_path=ROOT.parent/'MigrationBench_revision1_revised_package.zip'
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(active):z.write(p,'MigrationBench_revision1_revised/'+str(p.relative_to(ROOT)))
with zipfile.ZipFile(zip_path) as z:
    assert z.testzip() is None
print(json.dumps({'archive':str(zip_path),'bytes':zip_path.stat().st_size,'files':len(active)},indent=2))
