#!/usr/bin/env python3
"""Read-only remote coordinate/trajectory crosswalk for exact QE restart names."""
import csv, json, pathlib, subprocess, sys, datetime
BASE=pathlib.Path('/Users/alina/.codex/worktrees/migrationbench-unified-20261121/analysis/rockfish_skipjack_migration_audit_20261121/phase2_data')
SCR=BASE/'scratch16_project_restart_names_20261123T1045Z.tsv'
DAT=BASE/'data_yi_remaining_exact_qe_mix_candidates_20261123T1047Z.tsv'
OUT=pathlib.Path(__file__).parent
cands=[]
lines=SCR.read_text().splitlines()
for line in lines[1:]:
    a=line.split('\t')
    if len(a)<9: continue
    p,size,blocks,mode,uid,gid,ino,nlink,mtime=a[:9]
    cands.append({'scope':'scratch16','path':p,'initial_size':int(size),'initial_allocated':int(blocks)*512,
      'initial_mode':mode,'initial_uid':int(uid),'initial_gid':int(gid),'initial_inode':int(ino),'initial_nlink':int(nlink),'initial_mtime_ns':int(float(mtime)*1e9)})
with DAT.open() as f:
    for r in csv.DictReader(f,delimiter='\t'):
        cands.append({'scope':'data','path':r['absolute_path'],'initial_size':int(r['logical_bytes']),'initial_allocated':int(r['allocated_bytes']),
          'initial_mode':None,'initial_uid':int(r['uid'] or 3901),'initial_gid':int(r['gid'] or 3001),'initial_inode':int(r['inode']),'initial_nlink':int(r['nlink']),
          'initial_mtime_ns':int(r['mtime_ns']),'triage_disposition':r['disposition'],'triage_reason':r['disposition_reason'],
          'triage_run_root':r['run_root'],'triage_candidate_dir':r['run_group_or_candidate_directory'],'triage_restart_mode':r['restart_mode_neb_in'],
          'triage_slurm_status':r['sacct_status'],'triage_workdir_match':r['workdir_match'],'triage_job_evidence':r['slurm_job_evidence']})
if len(cands)!=7589: raise SystemExit(f'expected 7589 exact candidate records, found {len(cands)}')
remote=r'''import os,sys,re,json,stat,hashlib,xml.etree.ElementTree as ET,datetime,collections
C=json.loads(sys.stdin.read())
ROOTS={'scratch16':'/scratch16/pclancy3/yi','data':'/data/pclancy3/yi'}
# Read only small neighboring run evidence; restart payloads are never opened or hashed.
MAX_XML=12*1024*1024
MAX_PATH=24*1024*1024
MAX_LOG=2*1024*1024

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def ident(st): return {'size_bytes':st.st_size,'allocated_bytes':getattr(st,'st_blocks',0)*512,'mode':format(stat.S_IMODE(st.st_mode),'04o'),'uid':st.st_uid,'gid':st.st_gid,'device':st.st_dev,'inode':st.st_ino,'nlink':st.st_nlink,'mtime_ns':st.st_mtime_ns,'is_regular':stat.S_ISREG(st.st_mode)}
def parent_paths(p,root):
  q=os.path.dirname(p); arr=[]
  while q.startswith(root+'/'):
    arr.append(q)
    if q==root: break
    q=os.path.dirname(q)
  return arr
def input_names(d):
  out=[]
  try: names=os.listdir(d)
  except OSError: return out
  for n in names:
    lo=n.lower()
    if lo.endswith('.in') and (lo in ('neb.in','relax.in','scf.in','pw.in') or re.match(r'pw[_-]?\d+\.in$',lo) or re.match(r'[a-z0-9_.-]+\.in$',lo)):
      try:
        st=os.stat(os.path.join(d,n),follow_symlinks=False)
        if stat.S_ISREG(st.st_mode) and st.st_size<2*1024*1024: out.append(n)
      except OSError: pass
  return out
def find_run(c):
  root=ROOTS[c['scope']]
  if c['scope']=='data' and c.get('triage_run_root'):
    return c['triage_run_root']
  for d in parent_paths(c['path'],root):
    if input_names(d): return d
  return root

def xml_info(path):
  try:
    st=os.stat(path,follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode): return None
    info={'path':path,**ident(st),'xml_parse':'SKIPPED_TOO_LARGE' if st.st_size>MAX_XML else 'ERROR','atomic_structure':False,'atomic_positions':False,'nat':None,'position_atom_count':0,'sha256':None}
    if st.st_size>MAX_XML: return info
    h=hashlib.sha256()
    with open(path,'rb') as f:
      data=f.read(MAX_XML+1)
    if len(data)<=MAX_XML: info['sha256']=hashlib.sha256(data).hexdigest()
    root=ET.fromstring(data)
    nat=None; pos=None
    for el in root.iter():
      tag=el.tag.rsplit('}',1)[-1]
      if tag=='atomic_structure': nat=el.attrib.get('nat')
      elif tag=='atomic_positions': pos=el
    atom_count=0 if pos is None else sum(1 for x in pos if x.tag.rsplit('}',1)[-1]=='atom')
    info.update({'xml_parse':'OK','atomic_structure':nat is not None,'atomic_positions':pos is not None,'nat':nat,'position_atom_count':atom_count})
    return info
  except Exception as e: return {'path':path,'xml_parse':'ERROR','error':type(e).__name__+': '+str(e)[:180]}

def parse_path(path):
  try:
    st=os.stat(path,follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode): return None
    d=ident(st); info={'path':path,**d,'trajectory_parse':'SKIPPED_TOO_LARGE' if st.st_size>MAX_PATH else 'ERROR','sha256':None,'declared_images':None,'restart_iteration':None,'nat_inferred_from_image_rows':None,'image_ids':[],'per_image_coordinate_rows':{},'complete_image_coverage':False}
    if st.st_size>MAX_PATH: return info
    raw=open(path,'rb').read(MAX_PATH+1)
    if len(raw)>MAX_PATH: return info
    info['sha256']=hashlib.sha256(raw).hexdigest()
    lines=raw.decode('utf-8',errors='replace').splitlines()
    for i,line in enumerate(lines):
      if line.strip().upper()=='NUMBER OF IMAGES':
        for nxt in lines[i+1:]:
          if nxt.strip():
            try: info['declared_images']=int(nxt.strip())
            except ValueError: pass
            break
      if line.strip().upper()=='RESTART INFORMATION':
        for nxt in lines[i+1:]:
          if nxt.strip():
            try: info['restart_iteration']=int(nxt.strip())
            except ValueError: pass
            break
    current=None; rows=collections.Counter(); images=[]
    for line in lines:
      m=re.match(r'^\s*Image:\s*(\d+)',line,re.I)
      if m:
        current=int(m.group(1)); images.append(current); continue
      if current is None: continue
      t=line.split()
      if len(t)>=6:
        try:
          [float(x.replace('D','E').replace('d','e')) for x in t[:6]]
          rows[current]+=1
        except ValueError: pass
    info['image_ids']=images; info['per_image_coordinate_rows']={str(k):v for k,v in rows.items()}
    n=info['declared_images']; image_row_counts=[rows[i] for i in range(1,n+1)] if n else []
    same_rows=bool(image_row_counts and min(image_row_counts)>0 and len(set(image_row_counts))==1)
    info['nat_inferred_from_image_rows']=image_row_counts[0] if same_rows else None
    info['trajectory_parse']='OK' if n and images and same_rows and set(images)==set(range(1,n+1)) else 'INCOMPLETE'
    info['complete_image_coverage']=bool(n and images and same_rows and set(images)==set(range(1,n+1)))
    return info
  except Exception as e: return {'path':path,'trajectory_parse':'ERROR','error':type(e).__name__+': '+str(e)[:180]}

def log_info(path):
  try:
    st=os.stat(path,follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode) or st.st_size>MAX_LOG: return {'path':path,**ident(st),'log_parse':'SKIPPED_TOO_LARGE'}
    with open(path,'rb') as f: data=f.read(MAX_LOG)
    txt=data.decode('utf-8',errors='replace')
    pats=['JOB DONE','convergence has been achieved','bfgs converged','NOT converged','convergence NOT achieved','convergence has NOT been achieved','Maximum number of steps','Maximum number of iterations','ERROR','SIGTERM','TIME LIMIT']
    marks=[x for x in pats if x.lower() in txt.lower()]
    return {'path':path,**ident(st),'log_parse':'OK','markers':marks,'tail':txt[-1500:] if any(x in path.lower() for x in ('neb.out','pw.out','relax.out','scf.out')) else None}
  except Exception as e: return {'path':path,'log_parse':'ERROR','error':str(e)[:180]}

def manifest_info(path):
  try:
    st=os.stat(path,follow_symlinks=False)
    if st.st_size>2*1024*1024: return {'path':path,**ident(st),'json_parse':'SKIPPED_TOO_LARGE'}
    obj=json.load(open(path,encoding='utf-8'))
    found=[]
    def rec(x,k=''):
      if isinstance(x,dict):
        for a,b in x.items(): rec(b,(k+'.' if k else '')+str(a))
      elif isinstance(x,list):
        for i,b in enumerate(x): rec(b,k+f'[{i}]')
      elif any(t in k.lower() for t in ('restart','wfc','mix','job_id','workdir','trajectory','coordinate','image')):
        found.append({'key':k,'value':x if isinstance(x,(str,int,float,bool)) or x is None else type(x).__name__})
    rec(obj)
    return {'path':path,**ident(st),'json_parse':'OK','selected_fields':found[:160]}
  except Exception as e: return {'path':path,'json_parse':'ERROR','error':type(e).__name__+': '+str(e)[:180]}

# Find evidence once per distinct run root, with no symlink traversal.
runroots={}
for c in C:
  rr=find_run(c); runroots.setdefault(rr,[]).append(c)
records=[]
for rr,group in runroots.items():
  files=[]; walk_errors=[]; count_seen=0
  for dp,dnames,fnames in os.walk(rr,followlinks=False):
    dnames[:]=[d for d in dnames if d not in ('.git','__pycache__','.conda','venv','env')]
    for n in fnames:
      p=os.path.join(dp,n); count_seen+=1
      low=n.lower()
      relevant=(low.endswith('.in') or low.endswith('.out') or low.endswith('.err') or 'manifest' in low or re.search(r'\.path(?:\d+)?$',low) or low in ('neb.dat','neb.xyz','neb.axsf','neb.crd') or (low.endswith('.xml') and ('/out/' in p or 'data-file-schema' in low)) or low.endswith('.slurm') or low.endswith('.sh'))
      if not relevant: continue
      try:
        st=os.stat(p,follow_symlinks=False)
        if not stat.S_ISREG(st.st_mode): continue
        if low.endswith('.xml') and st.st_size<=MAX_XML: e=xml_info(p)
        elif re.search(r'\.path(?:\d+)?$',low): e=parse_path(p)
        elif low.endswith('.out') or low.endswith('.err'): e=log_info(p)
        elif 'manifest' in low and low.endswith('.json'): e=manifest_info(p)
        else: e={'path':p,**ident(st),'evidence_kind':'input_or_script'}
        if e: files.append(e)
      except OSError as e: walk_errors.append({'path':p,'error':str(e)[:180]})
  # Select XML and path evidence by run; retain all parseable trajectory snapshots and best latest.
  xmls=[x for x in files if x.get('xml_parse') in ('OK','SKIPPED_TOO_LARGE')]
  paths=[x for x in files if x.get('trajectory_parse') in ('OK','INCOMPLETE','SKIPPED_TOO_LARGE')]
  paths.sort(key=lambda x:x.get('mtime_ns',0),reverse=True)
  run={
   'record_type':'run_evidence','run_root':rr,'candidate_count':len(group),'candidate_paths':[x['path'] for x in group],
   'walk_file_count':count_seen,'walk_errors':walk_errors[:30],
   'input_paths':[x['path'] for x in files if x.get('evidence_kind')=='input_or_script' and x['path'].lower().endswith('.in')],
   'script_paths':[x['path'] for x in files if x.get('evidence_kind')=='input_or_script' and (x['path'].lower().endswith('.slurm') or x['path'].lower().endswith('.sh'))],
   'log_evidence':[x for x in files if x.get('log_parse')],
   'manifest_evidence':[x for x in files if x.get('json_parse')],
   'xml_evidence':xmls,'path_evidence':paths,
  }
  records.append(run)
  # Candidate-level evidence mapping and current identity check.
  xml_by_path={x['path']:x for x in xmls}
  path_ok=[x for x in paths if x.get('complete_image_coverage')]
  for c in group:
    p=c['path']; name=os.path.basename(p); d=os.path.dirname(p); rr0=rr
    cur=None
    try: cur=ident(os.stat(p,follow_symlinks=False))
    except OSError as e: cur={'status':'MISSING_OR_STAT_ERROR','error':str(e)}
    prefix=None
    m=re.match(r'^(.*)\.wfc\d+$',name,re.I)
    if m: prefix=m.group(1)
    m=re.match(r'^(.*)\.mix\d+$',name,re.I)
    if m: prefix=m.group(1)
    if name.lower().startswith('wfc') and name.lower().endswith('.dat'): prefix=None
    xml_candidates=[]
    for x in xmls:
      xp=x['path'];
      if xp.startswith(d+'/') or os.path.dirname(xp)==d: xml_candidates.append(x)
      elif prefix and (os.path.basename(xp)==prefix+'.xml' or (os.path.basename(xp)=='data-file-schema.xml' and os.path.basename(os.path.dirname(xp))==prefix+'.save' and os.path.dirname(os.path.dirname(xp))==d)):
        xml_candidates.append(x)
    # Prefer same image output, parsed and matching QE prefix; then latest same-directory data-file XML.
    xml_candidates.sort(key=lambda x:(x.get('xml_parse')!='OK',('data-file-schema.xml' not in os.path.basename(x['path'])), -x.get('mtime_ns',0)))
    xml=xml_candidates[0] if xml_candidates else None
    # Path coordinate evidence must include every declared image and be current relative to restart file.
    path=path_ok[0] if path_ok else None
    pmtime=path.get('mtime_ns',0) if path else 0
    xmtime=xml.get('mtime_ns',0) if xml else 0
    cmt=int(cur.get('mtime_ns',0)) if 'mtime_ns' in cur else c['initial_mtime_ns']
    direct_xml=bool(xml and xml.get('xml_parse')=='OK' and xml.get('atomic_positions') and xml.get('atomic_structure') and (xml.get('position_atom_count',0)>=int(xml.get('nat') or 0)) and xmtime>=cmt)
    full_path=bool(path and path.get('complete_image_coverage') and pmtime>=cmt)
    # A complete path snapshot can protect NEB coordinates for all images, even if a file-level XML is absent.
    mapping={'record_type':'candidate_evidence','scope':c['scope'],'path':p,'initial':{k:v for k,v in c.items() if k not in ('path','triage_reason')},'current_identity':cur,
      'run_root':rr,'direct_coordinate_xml':xml,'complete_latest_path_evidence':path,'coordinate_evidence_current_after_restart_mtime':bool(direct_xml or full_path),
      'direct_xml_current':direct_xml,'full_path_current':full_path,
      'triage_disposition':c.get('triage_disposition'),'triage_reason':c.get('triage_reason'),'triage_run_root':c.get('triage_run_root'),
      'triage_restart_mode':c.get('triage_restart_mode'),'triage_slurm_status':c.get('triage_slurm_status'),'triage_workdir_match':c.get('triage_workdir_match'),'triage_job_evidence':c.get('triage_job_evidence')}
    records.append(mapping)
print(json.dumps({'record_type':'scan_header','captured_utc':utc(),'candidate_count':len(C),'run_root_count':len(runroots),'scope_roots':ROOTS,'read_only':True,'restart_payloads_opened_or_hashed':False,'xml_limit_bytes':MAX_XML,'path_limit_bytes':MAX_PATH},separators=(',',':')))
for x in records: print(json.dumps(x,sort_keys=True,separators=(',',':')))
'''
remote_script=remote.replace('C=json.loads(sys.stdin.read())','C=json.loads('+repr(json.dumps(cands,separators=(',',':')))+')',1)
# execute remotely; only stdout is written locally; no remote files created/changed.
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','rockfish','python3','-'],input=remote_script,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=900)
if r.returncode:
    print(r.stderr[-5000:],file=sys.stderr); raise SystemExit(r.returncode)
raw=r.stdout
# Validate records and persist unchanged JSONL output.
lines=[x for x in raw.splitlines() if x.strip()]
if not lines or json.loads(lines[0]).get('record_type')!='scan_header': raise SystemExit('missing scan header')
header=json.loads(lines[0]);
if header.get('candidate_count')!=len(cands): raise SystemExit('remote candidate count mismatch')
with (OUT/'remote_coordinate_crosswalk.jsonl').open('w') as f:
    f.write('\n'.join(lines)+'\n')
(OUT/'remote_coordinate_crosswalk.stderr.txt').write_text(r.stderr)
print(json.dumps({'header':header,'record_lines':len(lines),'stderr':r.stderr[-1000:],'output':str(OUT/'remote_coordinate_crosswalk.jsonl')},indent=2))
