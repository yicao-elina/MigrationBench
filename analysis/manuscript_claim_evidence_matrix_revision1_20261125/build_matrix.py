from __future__ import annotations
import argparse, csv, hashlib, json, re, sqlite3
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = PACKAGE_DIR.parents[1]
parser = argparse.ArgumentParser(description='Rebuild the pinned MigrationBench Revision 1 claim evidence inventory.')
parser.add_argument('--repo-root', type=Path, default=DEFAULT_ROOT)
parser.add_argument('--output', type=Path, default=PACKAGE_DIR/'MIGRATIONBENCH_R1_CLAIM_EVIDENCE_MATRIX_20261125.tsv')
args = parser.parse_args()
ROOT = args.repo_root.resolve()
OUT = args.output.resolve()
DB = ROOT/'analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite'
PINNED_INPUTS = {
    'paper/revision1/sn-article-final.tex': 'a2252edcf786a4ce575eb6afaa018b1b8aad5b8266874934b3da5fd95566416f',
    'paper/revision1/sn-article-SI.tex': '2e65eb40ca03c0b90332ae64998447f453e33acdfeabfe51be8dcc66b8226ab5',
    'paper/revision1/response_to_reviewers.tex': '02d9e8ff0b918cb4b97b86d9880d1096e0bfe6a27d2593d6d345e71abc1edc66',
    'paper/revision1/evidence_index/claim_evidence_matrix.tsv': 'e28af7c267a00046211ce1f3f25b6c969693ff3b5aa3d5cc1bc21bb87fb93691',
    'analysis/migrationbench_asset_registry_20261025/migrationbench_asset_registry.sqlite': '13948ef0963418bcbd40cbfe6cb103fd28c0a3e62744c60c2af622e02a7d20dc',
}
SOURCE_COMMIT = '412ba9c5276a07a13026184cf62e6ff36588e05f'


COLS = ['matrix_id','claim_type','source_document','source_location','source_object_id','assertion_or_panel','exact_source_quote_or_value','numeric_tokens','referenced_labels','sql_item_id','sql_panel','registry_content','artifact_pointer','artifact_sha256','generator_or_query','registry_status','reproduction_status','lineage_status','gap_reason','source_sha256']

def sha(p:Path):
    try: return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError: return ''

def verify_pinned_inputs():
    mismatches=[]
    for rel, expected in PINNED_INPUTS.items():
        path=ROOT/rel
        actual=sha(path)
        if actual != expected:
            mismatches.append(f'{rel}: expected {expected}, found {actual or "MISSING"}')
    if mismatches:
        raise SystemExit('Pinned input verification failed; create a new matrix package for changed sources:\n'+'\n'.join(mismatches))

def clean_tex_line(s):
    # strip unescaped comment suffix, retain escaped percent signs
    out=[]; i=0
    while i<len(s):
        if s[i]=='%' and (i==0 or s[i-1] != '\\'): break
        out.append(s[i]); i+=1
    return ''.join(out)

def balanced(s, pos):
    # pos at opening brace
    depth=0; esc=False
    for i in range(pos,len(s)):
        c=s[i]
        if c=='{' and not esc: depth+=1
        elif c=='}' and not esc:
            depth-=1
            if depth==0: return s[pos+1:i],i+1
        if c=='\\' and not esc: esc=True
        else: esc=False
    return s[pos+1:],len(s)

def protected_sentence_split(s):
    protected = s.replace('e.g.','e<prd>g<prd>').replace('i.e.','i<prd>e<prd>').replace('vs.','vs<prd>')
    protected = re.sub(r'(?<=\d)\.(?=\d)', '<dec>', protected)
    # Split after sentence punctuation only when followed by a fresh prose/math command token.
    pieces=[]; start=0
    for m in re.finditer(r'[.!?](?:["”’\')\]}]*)\s+(?=(?:[A-Z]|\\[A-Za-z]))', protected):
        end=m.end()-len(m.group(0).lstrip('.!?'))
        # boundary after punctuation plus any closers; include trailing whitespace on the left only as normalization.
        end=m.start()+1
        while end<len(protected) and protected[end] in '"”’\')]}': end+=1
        piece=protected[start:end].strip()
        if piece: pieces.append(piece)
        start=m.end()
    tail=protected[start:].strip()
    if tail: pieces.append(tail)
    clauses=[]
    for piece in pieces:
        clauses.extend(x.strip() for x in re.split(r'(?<!\\);\s+', piece) if x.strip())
    return [p.replace('<dec>','.').replace('<prd>','.') for p in clauses]

def numeric_tokens(s):
    return '; '.join(re.findall(r'(?<![A-Za-z])(?:[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?(?:\\times10\^\{[^}]+\})?)(?:\\pm\s*\d+(?:\.\d+)?)?', s))

def labels(s):
    return ';'.join(dict.fromkeys(re.findall(r'\\(?:ref|eqref|sifig|sitab|autoref)\s*\{([^}]+)\}',s)))

def classify_claim(s):
    if re.search(r'\d|\\pm|\\times|\\approx|\\sim',s): return 'NUMERIC_ASSERTION'
    if re.search(r'\b(method|protocol|simulation|calculation|training|evaluation|dataset|model|pathway|trajectory|figure|table|analysis)\b',s,re.I): return 'METHOD_OR_SCOPE_ASSERTION'
    return 'SCIENTIFIC_ASSERTION'

def parse_tex_blocks(path):
    lines=path.read_text(errors='replace').splitlines()
    section='preamble'; env=[]
    for i,line in enumerate(lines,1):
        stripped=clean_tex_line(line).strip()
        if not stripped: continue
        sm=re.match(r'\\(section\*?|subsection\*?|subsubsection\*?)\s*\{',stripped)
        if sm:
            title=re.sub(r'\\[A-Za-z]+\*?(?:\[[^]]*\])?', '', stripped)
            title=re.sub(r'[{}]','',title).strip()
            section=title or section
            continue
        if re.match(r'\\begin\{(figure|table|equation|align|tabular|verbatim|lstlisting)',stripped):
            env.append(re.match(r'\\begin\{([^}]+)',stripped).group(1)); continue
        if re.match(r'\\end\{',stripped):
            if env: env.pop()
            continue
        if env: continue
        if re.match(r'\\(begin|end|label|maketitle|author|affil|email|title|input|include|FloatBarrier|newcommand|renewcommand|providecommand|setlength|captionsetup|centering|small|footnotesize|toprule|midrule|bottomrule|hline|bibliographystyle|bibliography|documentclass|usepackage|graphicspath|overpic)',stripped): continue
        if stripped.startswith('%'): continue
        if stripped in {'\\maketitle'}: continue
        # keep prose paragraphs and abstract; ignore punctuation-free command-only source lines
        txt=clean_tex_line(line).strip()
        if txt and re.search(r'[A-Za-z]{3}',txt): yield i,section,txt

def extract_environments(path, doc_id):
    text=path.read_text(errors='replace')
    pattern=re.compile(r'\\begin\{(figure|table)\*?\}(.*?)\\end\{\1\*?\}',re.S)
    for idx,m in enumerate(pattern.finditer(text),1):
        kind=m.group(1); body=m.group(2); offset=m.start()
        start_line=text.count('\n',0,offset)+1
        capmatch=re.search(r'\\caption(?:\[[^]]*\])?\s*\{',body)
        caption=''; capline=start_line
        if capmatch:
            brace=body.find('{',capmatch.start()); caption,_=balanced(body,brace)
            capline=start_line+body.count('\n',0,capmatch.start())
        lab=re.search(r'\\label\s*\{([^}]+)\}',body)
        label=lab.group(1) if lab else ''
        inc=re.findall(r'\\includegraphics(?:\[[^]]*\])?\s*\{([^}]+)\}',body)
        inps=re.findall(r'\\input\s*\{([^}]+)\}',body)
        yield {'kind':kind,'ordinal':idx,'line':start_line,'caption_line':capline,'caption':caption,'label':label,'includes':inc,'inputs':inps,'body':body,'offset':offset}

# Verify the exact source set before reading the registry.
verify_pinned_inputs()

# Load publication figure/table registry.
c=sqlite3.connect(f'file:{DB}?mode=ro',uri=True)
c.row_factory=sqlite3.Row
if c.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
    raise SystemExit('Registry quick_check failed; refusing to generate matrix.')
if c.execute('PRAGMA foreign_key_check').fetchall():
    raise SystemExit('Registry foreign_key_check failed; refusing to generate matrix.')
reg=list(c.execute('select * from rev1_results_index order by section,item_id,panel'))
items_by_panel={(r['section'],r['item_id'],r['panel']):r for r in reg}
# Exact item rows indexed by display label and SQL label.
label_to_item={}
for r in reg:
    p=r['panel'] or ''
    if p.startswith(('fig:','tab:')): label_to_item[p]=(r['item_id'],r['section'])
# Known manuscript figure labels have one SQL display item; these are direct label connections, not inferred from prose.
manual_labels={'fig:overview':'Fig1','fig:parity':'Fig2','fig:md':'Fig3','fig:profiles':'Fig4','fig:taskmatrix':'Fig5','fig:representation':'Fig6','fig:shap':'Fig7',
'fig:training_design':'Fig. S1','fig:grouped_overlap':'Fig. S2','fig:other_dopants':'Fig. S3','fig:sliding_all':'Fig. S4','fig:current_feature_audit':'Fig. S5'}
for lab,item in manual_labels.items(): label_to_item.setdefault(lab,(item,'main' if not item.startswith('Fig.') else 'si'))
# Inventory table and reproducibility registries for exact evidence pointers.
repro={r['item_id']:r for r in c.execute('select * from rev1_reproducibility')}

def resolver_item(label):
    return label_to_item.get(label)

def registry_rows_for_item(item):
    return [r for r in reg if r['item_id']==item]

def evid_for_rows(rows):
    vals=[]
    for r in rows:
        for k in ['plotted_data','upstream_data','raw_location','models']:
            v=r[k]
            if v and v.strip() and v not in vals: vals.append(v.strip())
        if r['reproduce_script'] and r['reproduce_script'] not in vals: vals.append('generator:'+r['reproduce_script'])
    return ' || '.join(vals)

def row_status(r):
    status=(r['status'] or '').upper(); repro_status=(r['reproduction'] or '').upper()
    if status=='MISSING_SOURCE' or not r['final_asset'] and not r['asset_sha256'] and status not in ('FULL_SOURCE','FULL'):
        return 'MISSING','No committed final asset/hash or reproducible source pointer in registry.'
    if repro_status.startswith('PASS') and r['asset_sha256'] and r['reproduce_script'] and status not in ('MANUAL_SCHEMATIC','MISSING_SOURCE'):
        if 'MANUAL_LAYOUT' in status or 'KNOWN_ISSUE' in status or 'KNOWN_ERROR' in status or 'STALE' in status or 'MISMATCH' in status or 'WORDING' in status:
            return 'PARTIAL','Registry has an identified manual-layout/release/wording/known-issue boundary; panel result is not fully reconstructed from a single keyed row.'
        return 'TRACEABLE','Registry records a reproduction script and output status; final asset identity/panel mapping exists.'
    return 'PARTIAL','Registry points to this item, but reproduction/output identity or status is incomplete, manual, or exception-scoped.'

matrix=[]
def add(**kw):
    rec={k:'' for k in COLS}; rec.update(kw); matrix.append(rec)

# Claim-like prose sentences from authoritative submitted main and SI. Keep exact TeX line spans.
for rel,docid in [('paper/revision1/sn-article-final.tex','MAIN'),('paper/revision1/sn-article-SI.tex','SI')]:
    path=ROOT/rel; full=path.read_text(errors='replace')
    for line_no,section,rawline in parse_tex_blocks(path):
        # Preserve macros/math as TeX; split to sentence-level units.
        sentences=protected_sentence_split(rawline)
        # A line sometimes has a leading `\rev{...}` or panel/run tags; retain as exact quote.
        for n,sent in enumerate(sentences,1):
            if len(re.sub(r'\\[A-Za-z]+|[{}$~]', '', sent).strip()) < 25: continue
            refs=labels(sent); resolved=[]
            for lab in refs.split(';') if refs else []:
                m=resolver_item(lab)
                if m and m[0] not in resolved: resolved.append(m[0])
            linked=[r for item in resolved for r in registry_rows_for_item(item)]
            artifact=evid_for_rows(linked)
            if resolved:
                status='PARTIAL'; gap='Mapped via explicit manuscript figure/table reference to registry item(s), but this sentence is not itself a keyed claim/result row.'
            elif re.search(r'\\cite|\\citep|\\citet',sent):
                status='PARTIAL'; gap='Literature source is cited, but no local paper-claim row binds the citation to an independently verified source record.'
            else:
                status='MISSING'; gap='No exact claim-to-artifact row or explicit indexed figure/table reference found in the current SQL/reviewer registry.'
            # Numeric assertion with exact table label is still partial unless its exact row/cell is indexed.
            add(matrix_id=f'MBR1-{docid}-S{sum(1 for x in matrix if x["claim_type"] in ("NUMERIC_ASSERTION","SCIENTIFIC_ASSERTION","METHOD_OR_SCOPE_ASSERTION") and x["source_document"]==rel)+1:04d}',
                claim_type=classify_claim(sent),source_document=rel,source_location=f'L{line_no}',source_object_id=f'{docid}-LINE-{line_no}',
                assertion_or_panel=section,exact_source_quote_or_value=sent,numeric_tokens=numeric_tokens(sent),referenced_labels=refs,
                sql_item_id=';'.join(resolved),sql_panel='',registry_content='',artifact_pointer=artifact,
                artifact_sha256=';'.join(dict.fromkeys(r['asset_sha256'] for r in linked if r['asset_sha256'])),
                generator_or_query=';'.join(dict.fromkeys(r['reproduce_script'] for r in linked if r['reproduce_script'])),
                registry_status=';'.join(dict.fromkeys(r['status'] for r in linked)),reproduction_status=';'.join(dict.fromkeys(r['reproduction'] for r in linked)),
                lineage_status=status,gap_reason=gap,source_sha256=sha(path))

# One explicit row per SQL-indexed figure panel/table item (the source excerpt is caption/table quote where available).
for sec, docrel in [('main','paper/revision1/sn-article-final.tex'),('si','paper/revision1/sn-article-SI.tex'),('response','paper/revision1/response_to_reviewers.tex')]:
    path=ROOT/docrel; envs=list(extract_environments(path,sec))
    # label -> matching tex environment
    env_by_label={e['label']:e for e in envs if e['label']}
    for r in reg:
        if r['section']!=sec: continue
        item=r['item_id']; panel=r['panel'] or ''
        if item=='GraphicalAbstract':
            status='MISSING'; gap='Submitted graphic is present only in received PDF; registry explicitly says no source/repository asset.'
            quote=r['content'] or 'Graphical abstract; source missing.'; loc='submitted PDF p.55'
        elif item=='HiResFigureFiles':
            status='PARTIAL'; gap='Standalone upload inventory is indexed, but it is a grouped file set rather than one independently keyed scientific assertion.'
            quote=r['content'] or ''; loc='submission pp.56-62'
        else:
            env=None
            if panel in env_by_label: env=env_by_label[panel]
            # caption label may not equal index panel, so resolve exact figure/table label from manual map.
            if env is None:
                for lab,(mapped_item,mapped_sec) in label_to_item.items():
                    if mapped_item==item and mapped_sec==sec and lab in env_by_label:
                        env=env_by_label[lab]; break
            quote=(env['caption'] if env and env['caption'] else r['content'] or '')
            loc=f'{docrel}:L{env["caption_line"] if env else "?"}'
            status,gap=row_status(r)
        kind='FIGURE_PANEL' if item.startswith('Fig') or item=='GraphicalAbstract' or item=='HiResFigureFiles' else 'TABLE_OBJECT'
        pointer='; '.join(x for x in [r['final_asset'],r['plotted_data'],r['upstream_data'],r['raw_location']] if x and x.strip())
        add(matrix_id=f'MBR1-REG-{item.replace(" ","").replace(".","")}-{panel.replace(":", "_").replace("-", "_")}',
            claim_type=kind,source_document=docrel,source_location=loc,source_object_id=f'{item}:{panel}',assertion_or_panel=panel,
            exact_source_quote_or_value=quote,numeric_tokens=numeric_tokens(quote),referenced_labels=panel if panel.startswith(('fig:','tab:')) else '',
            sql_item_id=item,sql_panel=panel,registry_content=r['content'],artifact_pointer=pointer,artifact_sha256=r['asset_sha256'] or '',
            generator_or_query=r['reproduce_script'] or '',registry_status=r['status'] or '',reproduction_status=r['reproduction'] or '',
            lineage_status=status,gap_reason=gap,source_sha256=sha(path))

# Exact numeric cells from submitted main/SI/response tables and their included table source files.
def extract_cells(path):
    text=path.read_text(errors='replace'); lines=text.splitlines();
    # find table envs and extract inline tabular rows; also resolve \\input sources inside table environments.
    envs=list(extract_environments(path,path.name))
    for env in envs:
        if env['kind']!='table': continue
        label=env['label']; sql_item=label_to_item.get(label,(None,None))[0] if label else None
        start=env['line']; body=env['body']; data_sources=[]
        for inp in env['inputs']:
            candidates=[ROOT/Path(inp+'.tex'), ROOT/'paper'/Path(inp+'.tex')]
            # Most source paths are rooted from paper/ during build.
            candidates += [ROOT/'paper'/Path(inp+'.tex'), ROOT/'paper/revision1'/Path(inp+'.tex'), ROOT/'paper/revision1/tables'/Path(Path(inp).name+'.tex')]
            f=next((x for x in candidates if x.is_file()),None)
            if f: data_sources.append(f)
        if not data_sources: data_sources=[path]
        table_lines=[]
        for f in data_sources:
            ftext=f.read_text(errors='replace')
            # Inline table environment if same source, otherwise process the included file's tabular rows.
            lines2=ftext.splitlines()
            if f==path:
                envstart=max(0,start-1); envend=min(len(lines2),start-1+body.count('\n')+1)
                lines2=lines2[envstart:envend]
            line_offset = (envstart if f==path else 0)
            for ln,txt in enumerate(lines2,line_offset+1):
                if '&' in txt and re.search(r'\\\\\s*(?:%.*)?$',txt):
                    if re.search(r'\\(toprule|midrule|bottomrule|hline|cline|multicolumn)',txt): continue
                    cells=[x.strip() for x in re.sub(r'\\\\\s*(?:%.*)?$','',txt).split('&')]
                    if len(cells)<2: continue
                    table_lines.append((f,ln,cells))
        # Identify table registry rows; cell granularity is source-file exact but not yet separately keyed in SQL.
        rr=[x for x in reg if x['item_id']==sql_item] if sql_item else []
        r=rr[0] if rr else None
        for rn,(f,ln,cells) in enumerate(table_lines,1):
            # Skip obvious header/rule rows; retain any numeric or scientific cell, including identifiers.
            row_title=re.sub(r'\\[A-Za-z]+(?:\{[^}]*\})?', '', cells[0])
            for ci,cell in enumerate(cells,1):
                if not re.search(r'\d|\\pm|\\times|\\infty|\\leq|\\geq',cell): continue
                if r:
                    ptr='; '.join(x for x in [r['final_asset'],r['plotted_data'],r['upstream_data'],r['raw_location'],str(f.relative_to(ROOT))] if x and str(x).strip())
                    status='PARTIAL'; gap='Exact manuscript table cell and source-file line are captured; registry maps the table/generator/source set, but does not provide a cell-level source-row key or per-cell hash.'
                    rh=r['status']; rep=r['reproduction']; gen=r['reproduce_script']; ah=r['asset_sha256'] or ''
                else:
                    ptr=str(f.relative_to(ROOT)); status='MISSING'; gap='Numeric TeX cell found, but no matching publication-registry item/panel was resolved.'; rh=''; rep=''; gen=''; ah=''
                add(matrix_id=f'MBR1-CELL-{(sql_item or "UNK").replace(" ","").replace(".","")}-{start}-{rn:03d}-{ci:02d}',claim_type='TABLE_CELL',source_document=str(f.relative_to(ROOT)),source_location=f'L{ln}',source_object_id=f'{label or "TABLE"}-row{rn}-col{ci}',assertion_or_panel=f'{label or "table"}: row {rn} col {ci} ({row_title})',exact_source_quote_or_value=cell,numeric_tokens=numeric_tokens(cell),referenced_labels=label or '',sql_item_id=sql_item or '',sql_panel=label or '',registry_content=r['content'] if r else '',artifact_pointer=ptr,artifact_sha256=ah,generator_or_query=gen or '',registry_status=rh or '',reproduction_status=rep or '',lineage_status=status,gap_reason=gap,source_sha256=sha(f))

for p in [ROOT/'paper/revision1/sn-article-SI.tex',ROOT/'paper/revision1/response_to_reviewers.tex']:
    extract_cells(p)
# Main text tables are currently absent, but keep future-safe extraction.
extract_cells(ROOT/'paper/revision1/sn-article-final.tex')

# Retain the existing reviewer-question rows, but classify them distinctly from manuscript assertions.
review_path=ROOT/'paper/revision1/evidence_index/claim_evidence_matrix.tsv'
with review_path.open(newline='') as f:
    for i,r in enumerate(csv.DictReader(f,delimiter='\t'),1):
        pointer='; '.join(x for x in [r.get('primary_artifact',''),r.get('source_or_table',''),r.get('rebuild_script','')] if x)
        sci=r.get('scientific_status','')
        if sci.startswith(('VERIFIED','DESCRIPTIVE','FAILURE_MODE','RECONSTRUCTION')):
            status='PARTIAL' # reviewer-level mapping is not a manuscript claim row and scope may be qualified
            gap='Reviewer question is mapped separately from claims; it is not an exhaustive sentence-to-data crosswalk.'
        elif sci in ('VALIDATION_GAP','PROVENANCE_PARTIAL'):
            status='MISSING'; gap='Reviewer ledger explicitly records a validation/provenance gap; retain as open.'
        else:
            status='PARTIAL'; gap='Reviewer-question evidence only; no exact manuscript-claim row.'
        add(matrix_id=f'MBR1-REVIEW-{r.get("reviewer","X")}-{r.get("claim_id",f"ROW{i}")}-{i:02d}',claim_type='REVIEWER_QUESTION',source_document='paper/revision1/evidence_index/claim_evidence_matrix.tsv',source_location=f'row {i+1}',source_object_id=r.get('claim_id',''),assertion_or_panel=r.get('question',''),exact_source_quote_or_value=r.get('boundary',''),numeric_tokens=numeric_tokens(r.get('boundary','')),referenced_labels=r.get('manuscript_location',''),sql_item_id='',sql_panel='',registry_content=sci,artifact_pointer=pointer,artifact_sha256='',generator_or_query=r.get('rebuild_script',''),registry_status=sci,reproduction_status='',lineage_status=status,gap_reason=gap,source_sha256=sha(review_path))

# Stable order: prose, figure/table registry, cells, reviewer mapping.
ids=[r['matrix_id'] for r in matrix]
if len(ids) != len(set(ids)):
    raise SystemExit('Duplicate matrix_id detected; refusing to write output.')
if any(not r['lineage_status'] or not r['source_location'] or not r['exact_source_quote_or_value'] for r in matrix):
    raise SystemExit('Required source/status field is empty; refusing to write output.')
with OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=COLS,delimiter='\t',quoting=csv.QUOTE_MINIMAL,lineterminator='\n')
    w.writeheader(); w.writerows(matrix)

print(f'Wrote {OUT} ({len(matrix)} rows; pinned source commit {SOURCE_COMMIT})')
