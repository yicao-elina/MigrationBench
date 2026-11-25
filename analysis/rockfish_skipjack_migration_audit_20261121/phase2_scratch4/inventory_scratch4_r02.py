#!/usr/bin/env python3
"""Metadata-only refresh of `/scratch4/pclancy3/yi` after WFC cleanup."""
import json
import os
import stat
import sys
import time

ROOT = "/scratch4/pclancy3/yi"
started = time.time()
print(json.dumps({"record": "scan_start", "root": ROOT, "started_epoch": started, "host": os.uname().nodename}, separators=(",", ":")))
stack = [ROOT]
counts = {"file": 0, "dir": 0, "symlink": 0, "special": 0, "error": 0}
logical = allocated = 0
while stack:
    d = stack.pop()
    try:
        entries = sorted(os.scandir(d), key=lambda entry: entry.name, reverse=True)
    except OSError as exc:
        counts["error"] += 1
        print(json.dumps({"record": "error", "path": d, "error": repr(exc)}, separators=(",", ":")))
        continue
    for ent in entries:
        path = ent.path
        try:
            st = ent.stat(follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                kind = "symlink"
            elif stat.S_ISDIR(st.st_mode):
                kind = "dir"
                stack.append(path)
            elif stat.S_ISREG(st.st_mode):
                kind = "file"
            else:
                kind = "special"
            if kind == "file":
                counts[kind] += 1
                logical += st.st_size
                allocated += st.st_blocks * 512
            elif kind in counts:
                counts[kind] += 1
            record = {
                "record": kind, "path": path, "size_bytes": st.st_size,
                "allocated_bytes": st.st_blocks * 512, "mtime_ns": st.st_mtime_ns,
                "mode_octal": format(stat.S_IMODE(st.st_mode), "04o"),
                "uid": st.st_uid, "gid": st.st_gid, "device": st.st_dev,
                "inode": st.st_ino, "nlink": st.st_nlink,
                "symlink_target": os.readlink(path) if kind == "symlink" else None,
            }
            print(json.dumps(record, separators=(",", ":")))
        except OSError as exc:
            counts["error"] += 1
            print(json.dumps({"record": "error", "path": path, "error": repr(exc)}, separators=(",", ":")))
finished = time.time()
print(json.dumps({"record": "scan_end", "root": ROOT, "finished_epoch": finished,
                  "elapsed_seconds": finished-started, "counts": counts,
                  "logical_file_bytes": logical, "allocated_file_bytes": allocated}, separators=(",", ":")))
