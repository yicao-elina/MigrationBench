#!/usr/bin/env python3
"""Capture Rockfish personal/group quota readouts after exact duplicate linking."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent
REMOTE = r'''import datetime, subprocess, sys
parts=[]
parts.append(subprocess.run(["/usr/lpp/mmfs/bin/mmlsquota", "-u", "ycao73", "-Y", "data"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60))
parts.append(subprocess.run(["/usr/lpp/mmfs/bin/mmlsquota", "-g", "pclancy3", "-Y", "data", "scratch4"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60))
sys.stdout.write(parts[0].stdout)
sys.stdout.write(parts[1].stdout)
sys.stderr.write(parts[0].stderr + parts[1].stderr)
if any(p.returncode for p in parts): sys.exit(2)
'''


def main() -> None:
    local_captured = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    proc = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "rockfish", "python3", "-"],
        input=REMOTE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        timeout=180,
    )
    stamp = local_captured.replace(":", "").replace("-", "")
    raw_path = OUT_DIR / f"quota_after_hardlink_{stamp}.txt"
    status_path = OUT_DIR / f"quota_after_hardlink_{stamp}.status.json"
    raw_path.write_text(proc.stdout, encoding="utf-8")
    status = {
        "local_captured_utc": local_captured,
        "remote_command": "ssh rockfish python3 -: /usr/lpp/mmfs/bin/mmlsquota -u ycao73 -Y data; /usr/lpp/mmfs/bin/mmlsquota -g pclancy3 -Y data scratch4",
        "ssh_returncode": proc.returncode,
        "stderr": proc.stderr,
        "raw_output_path": str(raw_path),
        "raw_output_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
        "raw_output_bytes": len(proc.stdout.encode()),
    }
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"raw_output_path": str(raw_path), "status_path": str(status_path),
                      "ssh_returncode": proc.returncode, "stderr": proc.stderr}, sort_keys=True))
    if proc.returncode:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
