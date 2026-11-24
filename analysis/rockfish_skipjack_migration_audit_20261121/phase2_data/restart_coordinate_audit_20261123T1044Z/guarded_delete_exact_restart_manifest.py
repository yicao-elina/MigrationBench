#!/usr/bin/env python3
"""Guarded exact-path QE WFC/mix deletion preflight.

Default operation is a read-only remote preflight. Remote unlink requires the
explicit --execute flag. This audit agent did not execute this script.
"""
import argparse
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "eligible_restart_deletion_manifest_20261123T1129Z.json"
REMOTE = r"""import os,sys,json,stat,re,hashlib,subprocess,datetime
P=json.loads(PAYLOAD)
ROOTS=('/scratch16/pclancy3/yi','/data/pclancy3/yi')
SQ='/cm/shared/apps/slurm/current/bin/squeue'
CONF='/cm/shared/apps/slurm/var/etc/slurm/slurm.conf'

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds').replace('+00:00','Z')
def fail(msg): raise RuntimeError(msg)
def inside(path,root):
    try: return os.path.commonpath([os.path.realpath(path),root])==root
    except Exception: return False
def kind(path):
    n=os.path.basename(path)
    if re.search(r'\.mix\d+$',n,re.I): return 'QE_MIX_N'
    if re.fullmatch(r'wfc.*\.dat',n,re.I): return 'QE_WFC_DAT'
    if re.search(r'\.wfc\d+$',n,re.I): return 'QE_WFC_N'
    return 'OTHER'
def dir_flags():
    flags=os.O_RDONLY|getattr(os,'O_DIRECTORY',0)|getattr(os,'O_NOFOLLOW',0)
    if not getattr(os,'O_DIRECTORY',0) or not getattr(os,'O_NOFOLLOW',0): fail('platform lacks O_DIRECTORY/O_NOFOLLOW')
    if os.open not in os.supports_dir_fd or os.stat not in os.supports_dir_fd or os.unlink not in os.supports_dir_fd: fail('platform lacks openat/fstatat/unlinkat dir_fd support')
    return flags
def open_parent_nofollow(path,root):
    if not path.startswith(root.rstrip('/')+'/'): fail('path is outside allowlisted root: '+path)
    rel=os.path.relpath(path,root); parts=rel.split(os.sep)
    if not parts or any(x in ('','.','..') for x in parts): fail('unsafe relative path: '+path)
    flags=dir_flags(); fd=os.open(root,flags)
    try:
        for component in parts[:-1]:
            nxt=os.open(component,flags,dir_fd=fd)
            os.close(fd); fd=nxt
        return fd,parts[-1]
    except Exception:
        os.close(fd); raise
def identity_from_stat(s):
    if not stat.S_ISREG(s.st_mode): fail('candidate is not a regular file')
    return {'size_bytes':s.st_size,'allocated_bytes':getattr(s,'st_blocks',0)*512,'mode':format(stat.S_IMODE(s.st_mode),'04o'),'uid':s.st_uid,'gid':s.st_gid,'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns}
def stat_candidate(path,root):
    fd,name=open_parent_nofollow(path,root)
    try: return identity_from_stat(os.stat(name,dir_fd=fd,follow_symlinks=False))
    finally: os.close(fd)
def open_evidence_nofollow(path,root):
    fd,name=open_parent_nofollow(path,root)
    try:
        f=os.open(name,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0),dir_fd=fd)
        st=os.fstat(f)
        if not stat.S_ISREG(st.st_mode): os.close(f); fail('coordinate evidence not regular: '+path)
        return f,st
    finally: os.close(fd)
def read_queue():
    env=os.environ.copy(); env['SLURM_CONF']=CONF
    r=subprocess.run([SQ,'-a','-h','-o','%i|%u|%T|%j|%Z'],universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,timeout=60)
    if r.returncode: fail('all-user squeue failed: '+r.stderr[-1000:])
    rows=[]
    for line in r.stdout.splitlines():
        a=line.split('|')
        if len(a)!=5: fail('malformed squeue row; refusing to continue: '+repr(line[:240]))
        if any(not x.strip() for x in a[:4]): fail('squeue row has blank ID/user/state/name: '+repr(line[:240]))
        wd=a[4].strip()
        if not wd or wd.lower() in ('unknown','n/a','none','(null)','unset') or not os.path.isabs(wd):
            fail('squeue row has blank/unknown/nonabsolute WorkDir: '+repr(line[:240]))
        rows.append({'job_id':a[0].strip(),'user':a[1].strip(),'state':a[2].strip(),'job_name':a[3].strip(),'workdir':wd})
    return utc(),rows
def overlaps(wd,rr):
    wd=wd.rstrip('/'); rr=rr.rstrip('/')
    return bool(wd and (wd==rr or wd.startswith(rr+'/') or rr.startswith(wd+'/')))
def verify_coords(entry,cache):
    for key in ('coordinate_xml','latest_path'):
        ev=entry.get(key)
        if not ev: continue
        path=ev.get('path'); expected=ev.get('sha256')
        if not path or not expected: fail('coordinate evidence missing path/hash for '+entry['path'])
        root=next((r for r in ROOTS if path.startswith(r+'/')),None)
        if not root or not inside(path,root): fail('coordinate evidence outside allowlisted roots: '+path)
        fd,st=open_evidence_nofollow(path,root)
        try:
            if st.st_size>32*1024*1024: fail('coordinate evidence exceeds 32 MiB hashing limit: '+path)
            if path not in cache:
                h=hashlib.sha256()
                with os.fdopen(fd,'rb',closefd=False) as f:
                    for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
                cache[path]=h.hexdigest()
        finally: os.close(fd)
        if cache[path]!=expected: fail('coordinate evidence hash changed: '+path)
def list_restart_candidates(root):
    found=[]
    for dp,dns,fns in os.walk(root,followlinks=False):
        dns[:]=[d for d in dns if d not in ('.git','__pycache__','.conda','venv','env')]
        for n in fns:
            path=os.path.join(dp,n)
            if kind(path)!='OTHER': found.append(path)
    return found
def active_overlaps(queue,runroot,known_ids):
    hits=[]
    for q in queue:
        jid=q['job_id']; jidbase=jid.split('_')[0].split('.')[0]
        if jid in known_ids or jidbase in known_ids:
            hits.append({'job_id':jid,'workdir':q['workdir'],'reason':'job_id_in_candidate_chain'})
        elif overlaps(q['workdir'],runroot):
            hits.append({'job_id':jid,'workdir':q['workdir'],'run_root':runroot,'reason':'workdir_overlap'})
    return hits
def queue_check_one_root(runroot,known_ids,phase):
    captured,queue=read_queue()
    hits=active_overlaps(queue,runroot,known_ids)
    return {'run_root':runroot,'captured_utc':captured,'queue_rows':len(queue),'overlap_count':len(hits),'overlaps':hits,'phase':phase}
def check_queue_per_runroot(runroots,known_ids,phase):
    checks=[]
    for root in sorted(runroots):
        check=queue_check_one_root(root,known_ids,phase)
        checks.append(check)
        print(json.dumps({'queue_check_phase':phase,'run_root':root,'captured_utc':check['captured_utc'],'queue_rows':check['queue_rows'],'overlap_count':check['overlap_count']},sort_keys=True))
        if check['overlap_count']: fail('queue overlap for run root '+root+': '+json.dumps(check['overlaps'][:20]))
    return checks
def preflight(P):
    m=P['manifest']; files=m.get('files',[])
    if m.get('status')!='DRAFT_READ_ONLY_NOT_EXECUTED': fail('manifest draft status mismatch')
    if not files or len(files)>10000: fail('empty manifest or manifest exceeds bounded 10,000-file cap')
    seen=set(); coord_cache={}; bytes_total=0; ids=set(); runroots=set()
    for e in files:
        path=e['path']; runroot=e['run_root']; root=next((r for r in ROOTS if path.startswith(r+'/') and runroot.startswith(r+'/')),None)
        if not root: fail('candidate/run root outside allowlist: '+path)
        if path in seen: fail('duplicate candidate path: '+path)
        seen.add(path); runroots.add(runroot)
        if not inside(path,root) or not inside(runroot,root): fail('real path escapes allowlisted root: '+path)
        if kind(path)!=e.get('file_class'): fail('candidate basename/class mismatch: '+path)
        if e.get('disposition')!='DELETE-ELIGIBLE': fail('noneligible disposition in manifest: '+path)
        for jid in e.get('slurm_job_ids',[]): ids.add(str(jid)); ids.add(str(jid).split('_')[0])
        verify_coords(e,coord_cache)
        actual=stat_candidate(path,root)
        if actual!=e['identity']: fail('candidate identity changed since final lstat: '+path)
        bytes_total+=actual['size_bytes']
    expected_by_root={root:set() for root in runroots}
    for e in files: expected_by_root[e['run_root']].add(e['path'])
    for root,expected in expected_by_root.items():
        actual=set(list_restart_candidates(root))
        if actual!=expected:
            extra=sorted(actual-expected)[:10]; missing=sorted(expected-actual)[:10]
            fail('run-root candidate set changed: '+root+' extra='+repr(extra)+' missing='+repr(missing))
    checks=check_queue_per_runroot(runroots,ids,'preflight')
    return {'queue_checks':len(checks),'run_root_queue_overlaps':sum(x['overlap_count'] for x in checks),'queue_capture_first_utc':checks[0]['captured_utc'] if checks else None,'queue_capture_last_utc':checks[-1]['captured_utc'] if checks else None,'candidate_count':len(files),'candidate_logical_bytes':bytes_total,'coordinate_evidence_files_hashed':len(coord_cache),'restart_payloads_opened_or_hashed':False,'candidate_roots':len(runroots),'candidate_paths_checked':len(seen),'runroots':runroots,'known_ids':ids}
def unlink_exact(entry):
    path=entry['path']; root=next(r for r in ROOTS if path.startswith(r+'/'))
    fd,name=open_parent_nofollow(path,root)
    try:
        # fstatat(AT_SYMLINK_NOFOLLOW) immediately before unlinkat(0).
        actual=identity_from_stat(os.stat(name,dir_fd=fd,follow_symlinks=False))
        if actual!=entry['identity']: fail('identity changed immediately before unlinkat: '+path)
        os.unlink(name,dir_fd=fd)
    finally: os.close(fd)
def exists_nofollow(path,root):
    try:
        fd,name=open_parent_nofollow(path,root)
        try: os.stat(name,dir_fd=fd,follow_symlinks=False); return True
        finally: os.close(fd)
    except FileNotFoundError:return False
def main():
    m=P['manifest']; execute=bool(P.get('execute'))
    check=preflight(P)
    status='PREFLIGHT_PASS_NO_DELETE' if not execute else 'PREFLIGHT_PASS_EXECUTING'
    print(json.dumps({'status':status,'execute':execute,**{k:v for k,v in check.items() if k not in ('runroots','known_ids')}},sort_keys=True))
    if not execute:return
    # Group by root; refresh that root's all-user queue immediately before unlinking its files.
    grouped={root:[] for root in check['runroots']}
    for entry in m['files']: grouped[entry['run_root']].append(entry)
    mutation_checks=[]; deleted=0; deleted_bytes=0; errors=[]
    for root in sorted(check['runroots']):
        try:
            root_check=queue_check_one_root(root,check['known_ids'],'immediate_pre_root_mutation')
            mutation_checks.append(root_check)
            print(json.dumps({'queue_check_phase':root_check['phase'],'run_root':root,'captured_utc':root_check['captured_utc'],'queue_rows':root_check['queue_rows'],'overlap_count':root_check['overlap_count']},sort_keys=True))
            if root_check['overlap_count']:
                errors.append({'run_root':root,'error':'active all-user queue overlap','overlaps':root_check['overlaps'][:20]})
                break
        except Exception as exc:
            errors.append({'run_root':root,'error':'queue recapture failed: '+str(exc)})
            break
        for entry in grouped[root]:
            try:
                unlink_exact(entry)
                if exists_nofollow(entry['path'],root): fail('path remained after unlinkat: '+entry['path'])
                deleted+=1; deleted_bytes+=entry['identity']['size_bytes']
                print(json.dumps({'event':'deleted_verified','path':entry['path'],'run_root':root,
                    'identity':entry['identity'],'queue_capture_utc':root_check['captured_utc'],
                    'verified_missing':True},sort_keys=True))
            except Exception as exc:
                errors.append({'path':entry['path'],'error':str(exc)}); break
        if errors: break
    remaining=[]
    for entry in m['files']:
        root=next(r for r in ROOTS if entry['path'].startswith(r+'/'))
        if exists_nofollow(entry['path'],root): remaining.append(entry['path'])
    after_cache={}
    try:
        for entry in m['files']: verify_coords(entry,after_cache)
    except Exception as exc: errors.append({'path':'coordinate_evidence_postcheck','error':str(exc)})
    extra=[]
    for root in check['runroots']:
        expected={e['path'] for e in m['files'] if e['run_root']==root}
        extra += sorted(set(list_restart_candidates(root))-expected)
    ok=(deleted==len(m['files']) and len(mutation_checks)==len(check['runroots']) and not remaining and not errors and not extra and all(x['overlap_count']==0 for x in mutation_checks))
    print(json.dumps({'status':'POST_DELETE_VERIFIED' if ok else 'POST_DELETE_PARTIAL_OR_FAILED','execute':True,'queue_checks_immediately_before_each_root_mutation':len(mutation_checks),'run_root_queue_overlaps':sum(x['overlap_count'] for x in mutation_checks),'root_queue_checks':mutation_checks,'deleted_count':deleted,'deleted_logical_bytes':deleted_bytes,'manifest_count':len(m['files']),'remaining_manifest_paths':len(remaining),'unlisted_restart_candidates_remaining':len(extra),'coordinate_evidence_files_rehashed_after':len(after_cache),'errors':errors[:20],'extra_candidates':extra[:20]},sort_keys=True))
    if not ok:sys.exit(2)
try:main()
except Exception as e:
    print(json.dumps({'status':'PREFLIGHT_ABORTED_NO_DELETE_OR_BATCH_STOPPED','error':str(e)},sort_keys=True));sys.exit(3)
"""

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=pathlib.Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--host',default='rockfish')
    parser.add_argument('--execute',action='store_true',help='required to permit remote unlink; without it the action is read-only preflight')
    args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    payload={'manifest':manifest,'execute':bool(args.execute)}
    payload_literal=repr(json.dumps(payload,separators=(',',':')))
    remote_code=REMOTE.replace('P=json.loads(PAYLOAD)','P=json.loads('+payload_literal+')',1)
    r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',args.host,'python3','-'],input=remote_code,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800)
    if r.stdout: print(r.stdout,end='')
    if r.stderr: print(r.stderr,file=sys.stderr,end='')
    raise SystemExit(r.returncode)

if __name__=='__main__': main()
