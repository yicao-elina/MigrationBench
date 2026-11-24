#!/usr/bin/env python3
"""Scan both Rockfish user roots for exact QE WFC and mixing restart basenames."""

from __future__ import print_function

import datetime
import json
import os
import re
import stat
import sys


ROOTS = ("/scratch16/pclancy3/yi", "/data/pclancy3/yi")
PATTERNS = (
    ("QE_WFC_N", re.compile(r"^.*\.wfc[0-9]+$", re.I)),
    ("QE_WFC_DAT", re.compile(r"^wfc.*\.dat$", re.I)),
    ("QE_MIX_N", re.compile(r"^.*\.mix[0-9]+$", re.I)),
)


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def main():
    counts = dict((name, {"count": 0, "logical_bytes": 0}) for name, _ in PATTERNS)
    matches = []
    errors = []
    for root in ROOTS:
        for current, dirs, files in os.walk(root, topdown=True, followlinks=False,
                                            onerror=lambda exc: errors.append(str(exc))):
            dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(current, d))]
            for name in files:
                matched = None
                for kind, pattern in PATTERNS:
                    if pattern.match(name):
                        matched = kind
                        break
                if not matched:
                    continue
                path = os.path.join(current, name)
                try:
                    st = os.lstat(path)
                except OSError as exc:
                    errors.append(path + ": " + str(exc))
                    continue
                counts[matched]["count"] += 1
                counts[matched]["logical_bytes"] += int(st.st_size)
                matches.append({"class": matched, "path": path, "size_bytes": int(st.st_size),
                                "inode": int(st.st_ino), "mtime_ns": int(st.st_mtime_ns),
                                "mode": oct(stat.S_IMODE(st.st_mode)), "uid": int(st.st_uid),
                                "gid": int(st.st_gid), "is_symlink": stat.S_ISLNK(st.st_mode)})
    result = {"captured_utc": utc_now(), "scope_roots": list(ROOTS),
              "method": "Direct Rockfish Python os.walk(followlinks=false), skip symlinked directories, lstat matching basenames only; no payload opened or hashed.",
              "counts": counts, "total_files": len(matches),
              "total_logical_bytes": sum(m["size_bytes"] for m in matches),
              "scan_errors": errors, "matches": matches}
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
