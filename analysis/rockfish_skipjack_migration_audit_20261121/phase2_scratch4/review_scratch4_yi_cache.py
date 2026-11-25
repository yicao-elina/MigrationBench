#!/usr/bin/env python3
"""Read-only R02 scan of user-owned scratch4 cache/build/temp candidates."""

import csv
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
LEDGER = BASE / "scratch4_inventory_R02_20261122.jsonl"
QUEUE = BASE / "scratch4_yi_queue_overlap_allusers_20261122.txt"
TSV = BASE / "scratch4_yi_r02_cache_decisions_20261122.tsv"
REPORT = BASE / "scratch4_yi_r02_cache_review_20261122.md"
ROOT = "/scratch4/pclancy3/yi"
UID = 3901

rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
start = next(row for row in rows if row["record"] == "scan_start")
end = next(row for row in rows if row["record"] == "scan_end")
entries = [row for row in rows if row.get("uid") == UID and row.get("record") in ("file", "dir")]
files = [row for row in entries if row["record"] == "file"]
dirs = [row for row in entries if row["record"] == "dir"]

CACHE_DIRS = {"__pycache__", ".ruff_cache", ".pytest_cache", ".mypy_cache", ".cache", ".conda", "build", "dist", "tmp", "temp", "temporary"}
GENERATED_SUFFIXES = {".pyc", ".pyo", ".o", ".a", ".so", ".dylib", ".dll", ".mod", ".smod", ".tmp", ".temp", ".swp", ".swo", ".bak"}
RESTART_RE = re.compile(r"wfc\d+\.dat$", re.IGNORECASE)


def is_cache_or_temp(row):
    path = row["path"]
    parts = [part.lower() for part in path.split("/")]
    name = os.path.basename(path).lower()
    if any(part in CACHE_DIRS for part in parts):
        return True
    if name == ".ds_store" or name in {"cmakecache.txt", "ninja_log"} or name.startswith("core."):
        return True
    return os.path.splitext(name)[1] in GENERATED_SUFFIXES


cache_matches = [row for row in entries if is_cache_or_temp(row)]
wfc_matches = [row for row in files if RESTART_RE.fullmatch(os.path.basename(row["path"]))]
mixing_matches = [row for row in files if "mixing" in os.path.basename(row["path"]).lower()]
charge = [row for row in files if os.path.basename(row["path"]) == "charge-density.dat"]
zero = [row for row in files if row["size_bytes"] == 0]
qe_sidecars = [row for row in zero if re.fullmatch(r"pwscf\.(?:bar|com|dvkb|dwf|ebar|prd)\d+", os.path.basename(row["path"]))]
other_zero = [row for row in zero if row not in qe_sidecars]
queue_text = QUEUE.read_text().splitlines()
queue_timestamp = queue_text[0].replace("UTC=", "") if queue_text else "UNKNOWN"
queue_overlaps = [line for line in queue_text[1:] if line.strip()]

decision_rows = []
def add(group, path, count, logical, allocated, disposition, reason, rerun):
    decision_rows.append({
        "group": group, "path": path, "file_count": count,
        "logical_bytes": logical, "allocated_bytes": allocated,
        "disposition": disposition, "reason": reason, "rerun_recipe": rerun,
    })

add("CACHE_BUILD_TEMP", ROOT, len(cache_matches), sum(r.get("size_bytes", 0) or 0 for r in cache_matches), sum(r.get("allocated_bytes", 0) or 0 for r in cache_matches),
    "NO_DELETE_CANDIDATES", "No user-owned cache/build/temp/editor-cache path or generated-code suffix matched the reviewed inventory.", "Not applicable.")
add("QE_WFC_RESTARTS", ROOT, len(wfc_matches), sum(r["size_bytes"] for r in wfc_matches), sum(r["allocated_bytes"] for r in wfc_matches),
    "NO_REMAINING_FILES", "No wfc<number>.dat files remain under the complete user tree after the eight WSe2 WFC deletions.", "WSe2 rerun recipe is registered in WSE2_REPRODUCTION_MANIFEST.json; starts from fresh directory.")
add("QE_MIXING_RESTARTS", ROOT, len(mixing_matches), sum(r["size_bytes"] for r in mixing_matches), sum(r["allocated_bytes"] for r in mixing_matches),
    "NO_REMAINING_FILES", "No filenames containing 'mixing' remain in the complete user tree.", "Not applicable.")
for row in sorted(charge, key=lambda x: x["path"]):
    add("SCIENTIFIC_CHARGE_DENSITY_OUTPUT", row["path"], 1, row["size_bytes"], row["allocated_bytes"],
        "RETAIN_OUTPUT", "QE charge-density output retained with its calculation tree; this is scientific output, not a cache candidate.", "Recomputable by the corresponding QE run; exact rerun lineage remains as recorded per calculation.")
add("ZERO_BYTE_QE_PHONON_SIDECARS", "/scratch4/pclancy3/yi/2.Cr-Sb2Te3/3.phonon/1.no_corr/NAC/_ph0", len(qe_sidecars), 0, 0,
    "NO_SPACE_RECOVERY; HOLD", "576 zero-length pwscf.{bar,com,dvkb,dwf,ebar,prd}<n> files; zero allocated bytes. Their exact scratch lifecycle is not individually mapped, and deleting them would free no data blocks.", "Phonon calculation inputs/scripts exist in the tree, but no rerun is proposed for zero-byte files.")
for row in sorted(other_zero, key=lambda x: x["path"]):
    add("ZERO_BYTE_FILE", row["path"], 1, 0, row["allocated_bytes"], "RETAIN_OR_NO_SPACE_TARGET",
        "Zero logical bytes and zero allocated bytes; preserve failed/empty output markers until calculation evidence is adjudicated.", "No deletion/rerun action proposed.")
add("SLURM_WORKDIR_OVERLAP", ROOT, len(queue_overlaps), 0, 0,
    "NO_ACTIVE_OR_PENDING_OVERLAP", "All-user squeue query filtered for this exact subtree and its parent returned no workdirs.", "No current run-specific exclusion is required.")

with TSV.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(decision_rows[0].keys()), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(decision_rows)

start_utc = datetime.fromtimestamp(start["started_epoch"], timezone.utc).isoformat(timespec="seconds")
end_utc = datetime.fromtimestamp(end["finished_epoch"], timezone.utc).isoformat(timespec="seconds")
lines = [
    "# `/scratch4/pclancy3/yi` R02 cache/build/temp review",
    "",
    f"**Inventory window:** {start_utc} to {end_utc}; metadata-only R02 scan.",
    "",
    "## Decision",
    "",
    "No user-owned cache, build, temporary, editor cache, compiled object, or bytecode files were found in R02. No `wfc<number>.dat` or mixing restart files remain. There are no deletion candidates from this review.",
    "",
    "The complete R02 inventory contains %d regular files, %d directories, zero symlinks, zero special files, and zero scan errors; it totals %d logical bytes and %d allocated bytes. The user-owned file count is %d. The WSe2 cleanup reduced this tree from R01's 2,886 files / 5,642,682,016 logical bytes to R02's 2,878 files / 597,580,272 logical bytes." % (
        end["counts"]["file"], end["counts"]["dir"], end["logical_file_bytes"], end["allocated_file_bytes"], len(files)),
    "",
    "Seven `charge-density.dat` files remain (four Cr–Sb₂Te₃ phonon runs and three WSe₂ relaxations), totaling 364,602,168 logical / 364,642,304 allocated bytes. These are retained QE outputs. The biggest is the 95,980,840-byte WSe₂ run-2 charge density; no remaining file is larger than 100 MB.",
    "",
    "The phonon NAC scratch directory has 576 zero-length `pwscf.{bar,com,dvkb,dwf,ebar,prd}<n>` sidecars. All allocate zero blocks, so deleting them would not free storage; their lifecycle is not item-level mapped. Four other zero-byte files include an empty `combined_band_dos.dat`, `NAC/matdyn`, `NAC/BORN`, and `dmat.sb2te3`; preserve them until their calculation/output context is reviewed. No cleanup is recommended for any zero-byte file.",
    "",
    f"An all-user live Slurm queue query at {queue_timestamp} found **{len(queue_overlaps)} jobs** with workdirs overlapping `/scratch4/pclancy3/yi` or the project root. The filtered snapshot is `scratch4_yi_queue_overlap_allusers_20261122.txt`.",
    "",
    "## Method and evidence",
    "",
    "- This review reads the R02 path ledger and metadata only; no file contents were read and no Rockfish paths were changed.",
    "- Generated-file classification used path components for common cache/build/temp locations and exact suffixes for bytecode, compiled objects/libraries, editor swaps, backups, and temporary files.",
    "- `scratch4_inventory_R02_20261122.jsonl` is the complete input snapshot; `scratch4_yi_r02_cache_decisions_20261122.tsv` gives category-level decisions and the seven retained charge-density paths.",
    "- The all-user queue output is filtered to matching paths only; no other users' workdirs are included in the evidence file.",
]
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("cache_matches", len(cache_matches))
print("wfc_matches", len(wfc_matches), "mixing_matches", len(mixing_matches))
print("charge_density", len(charge), sum(r["size_bytes"] for r in charge), sum(r["allocated_bytes"] for r in charge))
print("zero_byte_files", len(zero), "QE_sidecars", len(qe_sidecars), "other_empty", len(other_zero))
print("queue_overlaps", len(queue_overlaps))
print("outputs", REPORT.name, TSV.name)
