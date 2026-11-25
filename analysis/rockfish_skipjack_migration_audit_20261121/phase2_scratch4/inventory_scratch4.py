#!/usr/bin/env python3
"""Read-only filesystem metadata inventory; run on Rockfish and capture stdout."""
import csv
import os
import sys

root = "/scratch4/pclancy3/yi"
w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
w.writerow(["record_type", "path", "bytes", "mtime_epoch", "mode", "extension", "error"])
stack = [root]
while stack:
    d = stack.pop()
    try:
        entries = list(os.scandir(d))
    except OSError as e:
        w.writerow(["error", d, "", "", "", "", repr(e)])
        continue
    for ent in entries:
        path = ent.path
        try:
            st = ent.stat(follow_symlinks=False)
            mode = oct(st.st_mode & 0o7777)
            if ent.is_symlink():
                w.writerow(["symlink", path, st.st_size, st.st_mtime_ns, mode, "", ""])
            elif ent.is_dir(follow_symlinks=False):
                w.writerow(["dir", path, "", st.st_mtime_ns, mode, "", ""])
                stack.append(path)
            elif ent.is_file(follow_symlinks=False):
                base = ent.name
                ext = os.path.splitext(base)[1].lower()
                w.writerow(["file", path, st.st_size, st.st_mtime_ns, mode, ext, ""])
            else:
                w.writerow(["special", path, st.st_size, st.st_mtime_ns, mode, "", ""])
        except OSError as e:
            w.writerow(["error", path, "", "", "", "", repr(e)])
