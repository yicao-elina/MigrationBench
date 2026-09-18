#!/usr/bin/env python3
"""Build MigrationBench NEB energy summaries and SVG barrier-profile plots.

This script uses only the Python standard library plus PyYAML when available.
It reads synced QE neb.out files, MLFF manifest files, and the canonical
pipeline config, then writes CSV/JSON/SVG/HTML artifacts.
"""

import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_processed" / "neb_energy_profiles"
PARSER_DIR = ROOT / "scripts" / "migrationbench"
sys.path.insert(0, str(PARSER_DIR))

from parse_qe_neb_output import parse_neb_out  # noqa: E402

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def safe_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def path_from_qe_file(path):
    text = str(path)
    patterns = [
        (r"mb_qe17_", "1-7"),
        (r"mb_qe16_", "1-6"),
        (r"mb_qe1_6_", "1-6"),
        (r"mb_qe81n1_", "81_neb_1"),
        (r"mb_qe81n4_", "81_neb_4"),
        (r"mb_victor01_", "victor_01_neb_v2"),
    ]
    for needle, path_id in patterns:
        if re.search(needle, text):
            return path_id
    parts = path.parts
    for part in parts:
        if part in {"neb", "neb2", "neb2_copy"} and "victor_neb_outputs" in text:
            return "victor_" + part
    return "unknown"


def branch_from_qe_file(path):
    parts = path.parts
    cluster = ROOT / "cluster"
    try:
        rel = path.relative_to(cluster)
        return rel.parts[0]
    except Exception:
        pass
    try:
        rel = path.relative_to(ROOT / "data_processed" / "victor_neb_outputs")
        return "victor_historical_" + "_".join(rel.parts[:-1])
    except Exception:
        return path.parent.name


def status_from_qe(parsed, branch):
    if parsed.get("has_overflow"):
        return "quarantine_overflow"
    if "mace_r1" in branch or "qe1_6_mace" in branch:
        return "quarantined_nonphysical" if parsed.get("activation_forward_eV") else "quarantined"
    if not parsed.get("last_images"):
        return "triage_no_complete_table"
    if parsed.get("converged_by_default_gate"):
        return "accepted_final_reference"
    if parsed.get("job_done"):
        return "stopped_unconverged"
    return "running_unconverged"


def qe_record(path):
    parsed = parse_neb_out(path)
    images = parsed.get("last_images") or []
    profile = []
    if images and images[0].get("energy_eV") is not None:
        e0 = images[0]["energy_eV"]
        for row in images:
            energy = safe_float(row.get("energy_eV"))
            profile.append(
                {
                    "image_index": row.get("image_index"),
                    "rel_energy_eV": None if energy is None else energy - e0,
                    "error_eV_A": row.get("error_eV_A"),
                    "frozen": row.get("frozen"),
                }
            )
    branch = branch_from_qe_file(path)
    return {
        "source_type": "QE",
        "path_id": path_from_qe_file(path),
        "branch_id": branch,
        "source_file": str(path),
        "status": status_from_qe(parsed, branch),
        "Em_eV": parsed.get("activation_forward_eV"),
        "reverse_Em_eV": parsed.get("activation_reverse_eV"),
        "max_image_error_eV_A": parsed.get("max_image_error_eV_A"),
        "max_image_error_all_eV_A": parsed.get("max_image_error_all_eV_A"),
        "barrier_drift_last_three_eV": parsed.get("barrier_drift_last_three_eV"),
        "last_complete_iteration": parsed.get("last_complete_iteration"),
        "job_done": parsed.get("job_done"),
        "has_overflow": parsed.get("has_overflow"),
        "n_images": len(profile) if profile else parsed.get("n_images_last_iteration"),
        "profile": profile,
    }


def mlff_path_from_manifest(path, manifest):
    inp = manifest.get("input_images", "")
    m = re.search(r"/(1-\d+)/sb2te3\.xyz", inp)
    if m:
        return m.group(1)
    if "1-6_repair" in str(path) or "1-6_mace_idpp" in inp:
        return "1-6"
    return "unknown"


def mlff_record(path):
    manifest = json.loads(path.read_text())
    profile_rows = manifest.get("profile") or []
    profile = [
        {
            "image_index": row.get("image_index", 0) + 1,
            "rel_energy_eV": row.get("rel_energy_eV"),
            "error_eV_A": row.get("fmax_eV_A"),
            "frozen": row.get("image_index") in {0, len(profile_rows) - 1},
        }
        for row in profile_rows
    ]
    branch = path.parts[-3] if path.parts[-2] == "files" else path.parts[-2]
    deltas = [
        abs(row.get("delta_energy_eV", 0.0))
        for row in manifest.get("relaxation_energy_delta", [])
        if row.get("image_index") not in {0, len(profile_rows) - 1}
    ]
    max_delta = max(deltas) if deltas else None
    status = "mlff_proxy"
    if max_delta is not None and max_delta > 5.0:
        status = "mlff_proxy_basin_collapse_or_large_relax"
    return {
        "source_type": "MACE/MLFF",
        "path_id": mlff_path_from_manifest(path, manifest),
        "branch_id": branch,
        "source_file": str(path),
        "status": status,
        "Em_eV": manifest.get("barrier_proxy_eV"),
        "reverse_Em_eV": None,
        "max_image_error_eV_A": max(
            [safe_float(row.get("fmax_eV_A")) for row in profile_rows if safe_float(row.get("fmax_eV_A")) is not None],
            default=None,
        ),
        "barrier_drift_last_three_eV": None,
        "last_complete_iteration": None,
        "job_done": True,
        "has_overflow": False,
        "n_images": len(profile),
        "max_abs_internal_relax_delta_eV": max_delta,
        "profile": profile,
    }


def historical_records():
    records = []
    cfg_path = ROOT / "configs" / "migrationbench_pipeline.yaml"
    if not yaml or not cfg_path.exists():
        return records
    cfg = yaml.safe_load(cfg_path.read_text())
    pathways = cfg.get("pathways", {})
    for path_id, row in pathways.items():
        if path_id == "81_atom_unresolved":
            for sub_id, sub in row.get("path_ids", {}).items():
                records.append(
                    {
                        "source_type": "historical_QE_candidate",
                        "path_id": "81_" + sub_id,
                        "branch_id": "historical_2D_421_" + sub_id,
                        "source_file": row.get("descriptor_table", ""),
                        "status": "historical_candidate_quarantine",
                        "Em_eV": sub.get("historical_candidate_barrier_eV"),
                        "reverse_Em_eV": None,
                        "max_image_error_eV_A": sub.get("historical_max_image_error_eV_A"),
                        "barrier_drift_last_three_eV": None,
                        "last_complete_iteration": None,
                        "job_done": None,
                        "has_overflow": False,
                        "n_images": None,
                        "profile": [],
                    }
                )
        else:
            val = row.get("current_reference_eV")
            if val is not None:
                records.append(
                    {
                        "source_type": "historical_reference_or_config",
                        "path_id": path_id,
                        "branch_id": "pipeline_config_current_reference",
                        "source_file": str(cfg_path),
                        "status": row.get("current_reference_status"),
                        "Em_eV": val,
                        "reverse_Em_eV": None,
                        "max_image_error_eV_A": None,
                        "barrier_drift_last_three_eV": None,
                        "last_complete_iteration": None,
                        "job_done": None,
                        "has_overflow": False,
                        "n_images": None,
                        "profile": [],
                    }
                )
    topology_path = ROOT / "data_processed" / "path_topology_audit" / "path_topology_manifest.json"
    if topology_path.exists():
        topology = {
            row["path_id"]: row
            for row in json.loads(topology_path.read_text()).get("results", [])
        }
        for record in records:
            row = topology.get(record["path_id"])
            if not row:
                continue
            qe = row.get("qe_summary", {})
            images = qe.get("last_images", [])
            if images:
                reference = images[0]["energy_eV"]
                record["profile"] = [
                    {
                        "image_index": image["image_index"],
                        "rel_energy_eV": image["energy_eV"] - reference,
                        "error_eV_A": image.get("error_eV_A"),
                        "frozen": image.get("frozen"),
                    }
                    for image in images
                ]
                record["Em_eV"] = qe.get("activation_forward_eV")
                record["reverse_Em_eV"] = qe.get("activation_reverse_eV")
                record["max_image_error_eV_A"] = qe.get("max_image_error_eV_A")
                record["max_image_error_all_eV_A"] = qe.get("max_image_error_all_eV_A")
                record["barrier_drift_last_three_eV"] = qe.get("barrier_drift_last_three_eV")
                record["last_complete_iteration"] = qe.get("last_complete_iteration")
                record["job_done"] = qe.get("job_done")
                record["n_images"] = len(images)
                record["source_file"] = row.get("neb_out", record["source_file"])
    return records


def write_csv(records, path):
    fields = [
        "path_id",
        "branch_id",
        "source_type",
        "status",
        "Em_eV",
        "reverse_Em_eV",
        "max_image_error_eV_A",
        "max_image_error_all_eV_A",
        "barrier_drift_last_three_eV",
        "last_complete_iteration",
        "n_images",
        "job_done",
        "has_overflow",
        "max_abs_internal_relax_delta_eV",
        "source_file",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fields})


def svg_escape(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_profile_svg(record, path):
    prof = [p for p in record.get("profile", []) if p.get("rel_energy_eV") is not None]
    if not prof:
        return
    width, height = 760, 440
    margin = {"l": 70, "r": 28, "t": 58, "b": 68}
    xs = [p["image_index"] for p in prof]
    ys = [p["rel_energy_eV"] for p in prof]
    ymin, ymax = min(ys + [0.0]), max(ys + [0.0])
    pad = max((ymax - ymin) * 0.12, 0.05)
    ymin -= pad
    ymax += pad
    if abs(ymax - ymin) < 1e-9:
        ymax += 0.1
        ymin -= 0.1

    def sx(x):
        if max(xs) == min(xs):
            return width / 2
        return margin["l"] + (x - min(xs)) / (max(xs) - min(xs)) * (width - margin["l"] - margin["r"])

    def sy(y):
        return height - margin["b"] - (y - ymin) / (ymax - ymin) * (height - margin["t"] - margin["b"])

    points = " ".join(f"{sx(p['image_index']):.1f},{sy(p['rel_energy_eV']):.1f}" for p in prof)
    zero_y = sy(0.0)
    title = f"{record['path_id']} | {record['source_type']} | Em={record.get('Em_eV')}"
    status = record.get("status", "")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin["l"]}" y="28" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#242424">{svg_escape(title)}</text>',
        f'<text x="{margin["l"]}" y="48" font-family="Arial, sans-serif" font-size="12" fill="#60656f">{svg_escape(status)}</text>',
        f'<line x1="{margin["l"]}" x2="{width-margin["r"]}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="#9ca3af" stroke-dasharray="4 4"/>',
        f'<line x1="{margin["l"]}" x2="{margin["l"]}" y1="{margin["t"]}" y2="{height-margin["b"]}" stroke="#374151"/>',
        f'<line x1="{margin["l"]}" x2="{width-margin["r"]}" y1="{height-margin["b"]}" y2="{height-margin["b"]}" stroke="#374151"/>',
        f'<polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.8" stroke-linejoin="round" stroke-linecap="round"/>',
    ]
    for p in prof:
        x, y = sx(p["image_index"]), sy(p["rel_energy_eV"])
        fill = "#ffffff" if p.get("frozen") else "#2563eb"
        stroke = "#2563eb"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-margin["b"]+24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#374151">{p["image_index"]}</text>')
        parts.append(f'<text x="{x:.1f}" y="{y-10:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#111827">{p["rel_energy_eV"]:.3f}</text>')
    for frac in [0, 0.25, 0.5, 0.75, 1]:
        yv = ymin + frac * (ymax - ymin)
        y = sy(yv)
        parts.append(f'<line x1="{margin["l"]-4}" x2="{margin["l"]}" y1="{y:.1f}" y2="{y:.1f}" stroke="#374151"/>')
        parts.append(f'<text x="{margin["l"]-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#374151">{yv:.2f}</text>')
    parts.append(f'<text x="{width/2:.1f}" y="{height-22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">NEB image index</text>')
    parts.append(f'<text transform="translate(20 {height/2:.1f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#374151">Relative energy vs image 1 (eV)</text>')
    parts.append(f'<text x="{margin["l"]}" y="{height-6}" font-family="Arial, sans-serif" font-size="10" fill="#6b7280">{svg_escape(record["branch_id"])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def make_index_html(records, profile_files, path):
    rows = []
    for rec in records:
        rows.append(
            "<tr>"
            + "".join(
                f"<td>{svg_escape(rec.get(k, ''))}</td>"
                for k in ["path_id", "branch_id", "source_type", "status", "Em_eV", "max_image_error_eV_A", "last_complete_iteration"]
            )
            + "</tr>"
        )
    links = "\n".join(f'<li><a href="{p.name}">{svg_escape(p.stem)}</a></li>' for p in profile_files)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MigrationBench NEB Energy Profiles</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 28px; color: #242424; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
td, th {{ border-bottom: 1px solid #ddd; padding: 7px 8px; text-align: left; }}
th {{ background: #f4f6f8; }}
.note {{ color: #5f6570; max-width: 980px; line-height: 1.45; }}
</style>
</head>
<body>
<h1>MigrationBench NEB Energy Profiles</h1>
<p class="note">Generated from synced QE outputs, MLFF manifests, and pipeline config. For QE, max error means the maximum over movable images; frozen-endpoint errors remain available in the all-image diagnostic column. Running, unconverged, quarantined, and historical-candidate values are not final manuscript references.</p>
<h2>Summary</h2>
<table>
<thead><tr><th>path</th><th>branch</th><th>source</th><th>status</th><th>Em eV</th><th>max movable error/fmax</th><th>iteration</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<h2>Profile SVGs</h2>
<ul>{links}</ul>
</body>
</html>
"""
    path.write_text(html)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for p in sorted((ROOT / "cluster").glob("**/neb.out")):
        records.append(qe_record(p))
    for p in sorted((ROOT / "data_processed" / "victor_neb_outputs").glob("**/neb.out")):
        records.append(qe_record(p))
    for p in sorted((ROOT / "cluster").glob("**/mlff_neb_manifest.json")):
        records.append(mlff_record(p))
    records.extend(historical_records())

    # Prefer informative records first in the human table.
    records.sort(key=lambda r: (r.get("path_id", ""), r.get("source_type", ""), r.get("branch_id", "")))
    (OUT / "neb_energy_summary.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    write_csv(records, OUT / "neb_energy_summary.csv")

    profile_files = []
    for idx, rec in enumerate(records):
        if rec.get("profile"):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{rec['path_id']}__{rec['branch_id']}")
            svg_path = OUT / f"{idx:02d}_{safe}.svg"
            make_profile_svg(rec, svg_path)
            profile_files.append(svg_path)
    make_index_html(records, profile_files, OUT / "neb_energy_profiles.html")
    print(json.dumps({"out_dir": str(OUT), "records": len(records), "profile_svgs": len(profile_files)}, indent=2))


if __name__ == "__main__":
    main()
