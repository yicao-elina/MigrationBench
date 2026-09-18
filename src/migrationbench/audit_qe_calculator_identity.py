#!/usr/bin/env python3
"""Inventory QE calculator identities and flag non-comparable energy profiles."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def value(text, key):
    match = re.search(
        rf"(?:^|,)\s*{re.escape(key)}(?:\([^)]*\))?\s*=\s*([^!,\n/]+)", text, re.I | re.M
    )
    return match.group(1).strip().strip("'\"") if match else None


def numeric(text, key):
    raw = value(text, key)
    if raw is None:
        return None
    match = re.search(FLOAT, raw)
    return float(match.group(0).replace("d", "e").replace("D", "E")) if match else None


def boolean(text, key, default=False):
    raw = value(text, key)
    if raw is None:
        return default
    return raw.lower().strip(".") in ("true", "t")


def kpoints(text):
    automatic = re.search(r"K_POINTS\s+AUTOMATIC\s*\n\s*([^\n]+)", text, re.I)
    if automatic:
        return "automatic:" + " ".join(automatic.group(1).split())
    if re.search(r"K_POINTS\s+(?:gamma|\{gamma\})", text, re.I):
        return "gamma"
    return "missing_or_other"


def pseudopotentials(text):
    values = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip().upper().startswith("ATOMIC_SPECIES"):
            for raw in lines[index + 1 :]:
                fields = raw.split()
                if len(fields) < 3 or fields[0].upper() in (
                    "K_POINTS", "CELL_PARAMETERS", "ATOMIC_POSITIONS", "BEGIN_POSITIONS"
                ):
                    break
                if fields[-1].lower().endswith((".upf", ".vdb")):
                    values.append(f"{fields[0]}:{Path(fields[-1]).name}")
            break
    return ";".join(sorted(values))


def cell_fingerprint(text):
    match = re.search(r"CELL_PARAMETERS[^\n]*\n([^\n]+)\n([^\n]+)\n([^\n]+)", text, re.I)
    if not match:
        return None
    numbers = []
    for row in match.groups():
        numbers.extend(float(number.replace("d", "e").replace("D", "E")) for number in re.findall(FLOAT, row)[:3])
    canonical = ",".join(f"{number:.5f}" for number in numbers)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def composition(text):
    block = re.search(r"ATOMIC_POSITIONS[^\n]*\n(.*?)(?=\n\s*(?:LAST_IMAGE|INTERMEDIATE_IMAGE|END_POSITIONS|CELL_PARAMETERS|K_POINTS|&|$))", text, re.I | re.S)
    if not block:
        return None
    symbols = []
    for line in block.group(1).splitlines():
        fields = line.split()
        if len(fields) < 4 or not re.fullmatch(FLOAT, fields[1]):
            break
        symbols.append(fields[0])
    counts = Counter(symbols)
    return "".join(f"{symbol}{counts[symbol]}" for symbol in sorted(counts)) if counts else None


def identity(path, root_label):
    text = path.read_text(errors="replace")
    row = {
        "root_label": root_label,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "calculation": value(text, "calculation"),
        "composition": composition(text),
        "ecutwfc_Ry": numeric(text, "ecutwfc"),
        "ecutrho_Ry": numeric(text, "ecutrho"),
        "occupations": value(text, "occupations"),
        "smearing": value(text, "smearing"),
        "degauss_Ry": numeric(text, "degauss"),
        "conv_thr_Ry": numeric(text, "conv_thr"),
        "vdw_corr": value(text, "vdw_corr"),
        "nspin": int(numeric(text, "nspin") or 1),
        "noncolin": boolean(text, "noncolin"),
        "lspinorb": boolean(text, "lspinorb"),
        "soc_enabled": boolean(text, "noncolin") and boolean(text, "lspinorb"),
        "kpoints": kpoints(text),
        "pseudopotentials": pseudopotentials(text),
        "cell_fingerprint": cell_fingerprint(text),
    }
    identity_fields = [
        "composition", "ecutwfc_Ry", "ecutrho_Ry", "occupations", "smearing",
        "degauss_Ry", "conv_thr_Ry", "vdw_corr", "nspin", "noncolin", "lspinorb",
        "kpoints", "pseudopotentials", "cell_fingerprint",
    ]
    serialized = json.dumps({key: row[key] for key in identity_fields}, sort_keys=True)
    row["calculator_identity"] = hashlib.sha256(serialized.encode()).hexdigest()[:16]
    missing = [key for key in ("composition", "ecutwfc_Ry", "ecutrho_Ry", "kpoints", "cell_fingerprint") if row[key] in (None, "missing_or_other")]
    row["identity_complete"] = not missing
    row["identity_missing_fields"] = ";".join(missing)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="LABEL=PATH; repeatable")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    roots = []
    for spec in args.root:
        label, raw = spec.split("=", 1)
        root = Path(raw).resolve()
        roots.append({"label": label, "path": str(root)})
        candidates = [root] if root.is_file() else sorted(root.rglob("*.in"))
        for path in candidates:
            try:
                rows.append(identity(path, label))
            except Exception as error:
                rows.append({"root_label": label, "path": str(path), "parse_error": repr(error)})
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (out_dir / "qe_input_identities.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        if row.get("calculator_identity"):
            groups[row["calculator_identity"]].append(row)
    profiles = []
    for identity_key, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        exemplar = members[0]
        profiles.append(
            {
                "calculator_identity": identity_key,
                "count": len(members),
                "root_labels": sorted(set(row["root_label"] for row in members)),
                "example_path": exemplar["path"],
                **{key: exemplar.get(key) for key in (
                    "composition", "ecutwfc_Ry", "ecutrho_Ry", "degauss_Ry", "vdw_corr",
                    "nspin", "noncolin", "lspinorb", "soc_enabled", "kpoints",
                    "pseudopotentials", "cell_fingerprint", "identity_complete",
                )},
            }
        )
    manifest = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": roots,
        "counts": {
            "inputs": len(rows),
            "parse_errors": sum("parse_error" in row for row in rows),
            "calculator_identities": len(groups),
            "incomplete_identities": sum(not row.get("identity_complete", False) for row in rows),
            "soc_enabled_inputs": sum(bool(row.get("soc_enabled")) for row in rows),
            "spin_polarized_inputs": sum(row.get("nspin") == 2 for row in rows),
        },
        "profiles": profiles,
        "comparability_rule": "Energies and barriers are comparable only within a complete calculator_identity unless a separately converged sensitivity study justifies a mapping.",
    }
    (out_dir / "calculator_identity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
