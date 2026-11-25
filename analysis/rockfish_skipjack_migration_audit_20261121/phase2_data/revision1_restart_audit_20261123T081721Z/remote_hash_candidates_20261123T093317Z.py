import concurrent.futures,csv,datetime,hashlib,json,os,stat,sys,threading,time
from pathlib import Path
manifest_path=Path(sys.argv[1]); partial_path=Path(sys.argv[2]); out_path=Path(sys.argv[3]); status_path=Path(sys.argv[4])
root="/scratch16/pclancy3/yi/revision1_migrationbench_runs"
expected_count=4298; expected_bytes=1036297176856
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00","Z")
def row_identity(st):
 return {"device":st.st_dev,"inode":st.st_ino,"uid":st.st_uid,"gid":st.st_gid,
  "mode":format(stat.S_IMODE(st.st_mode),"04o"),"nlink":st.st_nlink,"size_bytes":st.st_size,
  "allocated_bytes":getattr(st,"st_blocks",0)*512,"mtime_ns":st.st_mtime_ns}
with manifest_path.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f,delimiter="\t"))
if len(rows)!=expected_count: raise SystemExit("candidate count mismatch")
if sum(int(r["size_bytes"]) for r in rows)!=expected_bytes: raise SystemExit("candidate byte count mismatch")
paths=set()
for r in rows:
 p=r["absolute_path"]
 if p!=os.path.normpath(p) or not p.startswith(root+"/") or os.path.dirname(p).startswith(root+"/../"):
  raise SystemExit("path escapes exact audited root: "+p)
 if p!=root+"/"+r["relative_path"] or os.path.basename(p)!=r["basename"]: raise SystemExit("path/manifest mismatch: "+p)
 if p in paths: raise SystemExit("duplicate candidate path: "+p)
 paths.add(p)
expected_by_path={r["absolute_path"]:r for r in rows}
existing={}
if partial_path.exists():
 with partial_path.open(encoding="utf-8") as f:
  for line in f:
   if not line.strip(): continue
   x=json.loads(line); p=x.get("path")
   if p not in expected_by_path or p in existing: raise SystemExit("partial hash ledger contains unknown/duplicate path: "+str(p))
   r=expected_by_path[p]
   if int(x["size_bytes"])!=int(r["size_bytes"]) or int(x["inode"])!=int(r["inode"]) or int(x["mtime_ns"])!=int(r["mtime_ns"]):
    raise SystemExit("partial hash ledger identity no longer matches candidate manifest: "+p)
   if len(x.get("sha256", ""))!=64: raise SystemExit("partial row has no complete SHA-256: "+p)
   existing[p]=x
def hash_one(r):
 p=r["absolute_path"]; parent=os.path.dirname(p); name=os.path.basename(p)
 if os.path.realpath(parent)!=parent: raise RuntimeError("symlinked parent: "+p)
 dfd=os.open(parent,os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0))
 try:
  before_path=os.stat(name,dir_fd=dfd,follow_symlinks=False)
  if not stat.S_ISREG(before_path.st_mode): raise RuntimeError("not regular: "+p)
  fd=os.open(name,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0),dir_fd=dfd)
  try:
   before=os.fstat(fd); h=hashlib.sha256()
   while True:
    data=os.read(fd,8*1024*1024)
    if not data: break
    h.update(data)
   after=os.fstat(fd)
  finally: os.close(fd)
  after_path=os.stat(name,dir_fd=dfd,follow_symlinks=False)
 finally: os.close(dfd)
 fields=("st_dev","st_ino","st_uid","st_gid","st_mode","st_nlink","st_size","st_mtime_ns","st_blocks")
 if any(getattr(before,k,None)!=getattr(after,k,None) for k in fields) or any(getattr(before_path,k,None)!=getattr(after_path,k,None) for k in fields):
  raise RuntimeError("changed while hashing: "+p)
 actual=row_identity(after)
 expected={"device":int(r["device"]),"inode":int(r["inode"]),"uid":int(r["uid"]),"gid":int(r["gid"]),
  "mode":r["mode"],"nlink":int(r["nlink"]),"size_bytes":int(r["size_bytes"]),
  "allocated_bytes":int(r["allocated_bytes"]),"mtime_ns":int(r["mtime_ns"])}
 if actual!=expected: raise RuntimeError("audited identity changed: "+p+" actual="+json.dumps(actual,sort_keys=True))
 if actual["uid"]!=3901 or actual["gid"]!=3001 or actual["nlink"]!=1: raise RuntimeError("owner/group/link guard: "+p)
 return {"path":p,"candidate_kind":r["candidate_kind"],"run_group":r["run_group"],"run_job_id":r["run_job_id"],
  **actual,"sha256":h.hexdigest(),"coordinate_evidence_path":r["coordinate_evidence_path"],
  "coordinate_evidence_sha256":r["coordinate_evidence_sha256"],"disposition":r["disposition"]}
started=utc(); t0=time.monotonic(); errors=[]; lock=threading.Lock(); done=[len(existing)]; byte_count=[sum(int(x["size_bytes"]) for x in existing.values())]
partial_path.parent.mkdir(parents=True,exist_ok=True)
ledger_file=partial_path.open("a",encoding="utf-8")
def work(r):
 try: result=hash_one(r)
 except Exception as e:
  with lock: errors.append({"path":r["absolute_path"],"error":type(e).__name__+": "+str(e)})
  return
 with lock:
  ledger_file.write(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n")
  ledger_file.flush(); done[0]+=1; byte_count[0]+=int(result["size_bytes"])
  if done[0]%25==0: os.fsync(ledger_file.fileno())
  if done[0]%100==0: print(json.dumps({"event":"hash_progress","completed":done[0],"total":expected_count,"logical_bytes_completed":byte_count[0],"elapsed_seconds":round(time.monotonic()-t0,1)},sort_keys=True),flush=True)
to_hash=[r for r in rows if r["absolute_path"] not in existing]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 list(pool.map(work,to_hash))
ledger_file.flush(); os.fsync(ledger_file.fileno()); ledger_file.close()
if errors:
 status={"cleanup_batch_id":"RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123","state":"FAILED_HASH_PREFLIGHT",
  "started_utc":started,"finished_utc":utc(),"count_expected":expected_count,"count_hashed":done[0],"partial_ledger_path":str(partial_path),"errors":errors[:100],"deletion_performed":False}
 status_path.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
 raise SystemExit("hash preflight failed for "+str(len(errors))+" paths; no deletion occurred")
all_results=[json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(all_results)!=expected_count or {x["path"] for x in all_results}!=paths:
 status={"cleanup_batch_id":"RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123","state":"INCOMPLETE_HASH_LEDGER","started_utc":started,"finished_utc":utc(),"count_expected":expected_count,"count_hashed":len(all_results),"partial_ledger_path":str(partial_path),"deletion_performed":False}
 status_path.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8"); raise SystemExit("incomplete hash ledger; no deletion occurred")
all_results.sort(key=lambda x:x["path"])
tmp=out_path.with_suffix(out_path.suffix+".tmp")
with tmp.open("w",encoding="utf-8") as f:
 for x in all_results: f.write(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n")
 f.flush(); os.fsync(f.fileno())
os.replace(tmp,out_path)
status={"cleanup_batch_id":"RF-SCRATCH16-YI-REV1-QE-RESTART-CLEANUP-20261123","state":"HASHES_COMPLETE_NO_DELETION",
 "started_utc":started,"finished_utc":utc(),"candidate_count":len(all_results),
 "logical_bytes":sum(x["size_bytes"] for x in all_results),"allocated_bytes":sum(x["allocated_bytes"] for x in all_results),
 "hash_ledger_path":str(out_path),"hash_ledger_sha256":hashlib.sha256(out_path.read_bytes()).hexdigest(),
 "worker_count":8,"elapsed_seconds":round(time.monotonic()-t0,1),"deletion_performed":False,"partial_ledger_path":str(partial_path)}
status_path.write_text(json.dumps(status,indent=2)+"\n",encoding="utf-8")
print(json.dumps(status,sort_keys=True),flush=True)
