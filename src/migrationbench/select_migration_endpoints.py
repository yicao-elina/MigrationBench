#!/usr/bin/env python3
"""Build a representative migration endpoint/path queue from candidate sites.

The input can be a Victor/Joshitha-style site CSV with columns such as
unique_id, geometry, cr_z, mace_energy_eV, qe_input_path, or a more general CSV
with at least an id and z/energy-like columns. The output is an auditable graph:
sites, candidate endpoint pairs, and a representative path queue selected in
continuous descriptor space.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_float(row, names, default=None):
    for name in names:
        if name in row and row[name] not in ("", None):
            try:
                return float(row[name])
            except ValueError:
                pass
    return default


def raw_site_id(row, index):
    for name in ("unique_id", "site_id", "id", "name"):
        if row.get(name):
            return row[name]
    return "site_%04d" % index


def slug(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    return text.strip("-") or "unknown"


def parse_site_source(spec):
    """Parse [SYSTEM_ID::SOURCE_NAMESPACE=]PATH.

    Bare paths are deliberately isolated by source stem. Combining site lists
    requires an explicit shared SYSTEM_ID, which prevents accidental endpoint
    pairs across incompatible cells or compositions.
    """
    if "=" not in spec:
        path = Path(spec).resolve()
        namespace = slug(path.stem)
        return namespace, namespace, path
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).resolve()
    if "::" in label:
        system_id, namespace = label.split("::", 1)
    else:
        system_id = namespace = label
    return slug(system_id), slug(namespace), path


def read_sites(specs):
    sites = []
    seen_ids = set()
    sources = []
    for spec in specs:
        system_id, namespace, source = parse_site_source(spec)
        source_sha = sha256_file(source)
        sources.append(
            {
                "input_spec": spec,
                "system_id": system_id,
                "source_namespace": namespace,
                "source_csv": str(source),
                "source_sha256": source_sha,
            }
        )
        with source.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for idx, row in enumerate(reader, start=1):
                z = as_float(row, ("cr_z", "z", "migrant_z_A", "dopant_z_A"))
                energy = as_float(row, ("mace_energy_eV", "energy_eV", "relaxed_energy_eV"))
                geometry = row.get("geometry") or row.get("site_type") or "unknown"
                if z is None:
                    continue
                raw_id = raw_site_id(row, idx)
                global_id = "%s::%s::%s" % (system_id, namespace, raw_id)
                if global_id in seen_ids:
                    raise ValueError("Duplicate site key after namespacing: %s" % global_id)
                seen_ids.add(global_id)
                sites.append(
                    {
                        "site_id": global_id,
                        "raw_site_id": raw_id,
                        "system_id": system_id,
                        "source_namespace": namespace,
                        "source_csv": str(source),
                        "source_sha256": source_sha,
                        "geometry": geometry,
                        "migrant_z_A": z,
                        "screening_energy_eV": energy,
                        "qe_input_path": row.get("qe_input_path", ""),
                        "raw": row,
                    }
                )
    return sites, sources


def normalize(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return lambda _x: 0.0
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1.0e-12:
        return lambda _x: 0.5
    return lambda x: 0.0 if x is None else (float(x) - lo) / (hi - lo)


def geometry_score(label):
    label = (label or "").lower()
    if "oct" in label:
        return 0.25
    if "cn6" in label:
        return 0.45
    if "cn7" in label:
        return 0.60
    if "cn8" in label:
        return 0.72
    if "cn10" in label:
        return 0.90
    return 0.50


def annotate_sites(sites):
    znorm = normalize([s["migrant_z_A"] for s in sites])
    enorm = normalize([s["screening_energy_eV"] for s in sites])
    for site in sites:
        site["z_rank_score"] = round(znorm(site["migrant_z_A"]), 6)
        site["geometry_complexity_score"] = round(geometry_score(site["geometry"]), 6)
        e = site["screening_energy_eV"]
        site["stability_score"] = round(1.0 - enorm(e), 6) if e is not None else None
    return sites


def pair_class(delta_z, geom_delta, energy_delta, z_scale, energy_scale):
    z_norm = min(1.0, abs(delta_z) / max(z_scale, 1.0e-9))
    g_norm = min(1.0, abs(geom_delta))
    e_norm = min(1.0, abs(energy_delta) / max(energy_scale, 1.0e-9)) if energy_delta is not None else 0.5
    penetration_index = 0.50 * g_norm + 0.30 * z_norm + 0.20 * e_norm
    gap_index = 0.65 * (1.0 - g_norm) + 0.25 * z_norm + 0.10 * (1.0 - e_norm)
    if penetration_index >= 0.62:
        label = "deep_penetration_candidate"
    elif gap_index >= 0.62:
        label = "in_gap_or_open_channel_candidate"
    else:
        label = "mixed_transition_candidate"
    return label, penetration_index, gap_index


def build_pairs_one_system(sites, min_delta_z, max_delta_z, max_energy_delta):
    pairs = []
    energies = [s["screening_energy_eV"] for s in sites if s["screening_energy_eV"] is not None]
    z_values = [s["migrant_z_A"] for s in sites]
    z_scale = max(max(z_values) - min(z_values), 1.0) if z_values else 1.0
    energy_scale = max(max(energies) - min(energies), 1.0) if energies else 1.0
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            dz = b["migrant_z_A"] - a["migrant_z_A"]
            adz = abs(dz)
            if adz < min_delta_z or adz > max_delta_z:
                continue
            e_delta = None
            if a["screening_energy_eV"] is not None and b["screening_energy_eV"] is not None:
                e_delta = b["screening_energy_eV"] - a["screening_energy_eV"]
                if abs(e_delta) > max_energy_delta:
                    continue
            geom_delta = b["geometry_complexity_score"] - a["geometry_complexity_score"]
            label, pen, gap = pair_class(dz, geom_delta, e_delta, z_scale, energy_scale)
            pairs.append(
                {
                    "path_id": "%s__to__%s" % (a["site_id"], b["site_id"]),
                    "system_id": a["system_id"],
                    "start_site_id": a["site_id"],
                    "end_site_id": b["site_id"],
                    "start_geometry": a["geometry"],
                    "end_geometry": b["geometry"],
                    "delta_z_A": round(dz, 6),
                    "abs_delta_z_A": round(adz, 6),
                    "screening_energy_delta_eV": round(e_delta, 6) if e_delta is not None else None,
                    "geometry_complexity_delta": round(geom_delta, 6),
                    "transition_class": label,
                    "penetration_index": round(pen, 6),
                    "gap_index": round(gap, 6),
                    "priority_score": round(0.45 * max(pen, gap) + 0.30 * adz / z_scale + 0.25 * (1.0 - min(abs(e_delta or 0.0) / energy_scale, 1.0)), 6),
                }
            )
    return pairs


def build_pairs(sites, min_delta_z, max_delta_z, max_energy_delta):
    groups = defaultdict(list)
    for site in sites:
        groups[site["system_id"]].append(site)
    pairs = []
    for system_id in sorted(groups):
        pairs.extend(
            build_pairs_one_system(
                groups[system_id], min_delta_z, max_delta_z, max_energy_delta
            )
        )
    return pairs


def descriptor_distance(a, b):
    keys = ("abs_delta_z_A", "penetration_index", "gap_index", "geometry_complexity_delta")
    return math.sqrt(sum((float(a.get(k) or 0.0) - float(b.get(k) or 0.0)) ** 2 for k in keys))


def farthest_point_select(rows, n):
    if len(rows) <= n:
        return list(rows)
    selected = [max(rows, key=lambda r: r["priority_score"])]
    remaining = [r for r in rows if r is not selected[0]]
    while remaining and len(selected) < n:
        best = max(remaining, key=lambda r: min(descriptor_distance(r, s) for s in selected))
        selected.append(best)
        remaining.remove(best)
    return selected


def select_representatives(pairs, per_class):
    groups = defaultdict(list)
    for row in pairs:
        groups[(row["system_id"], row["transition_class"])].append(row)
    chosen = []
    for system_id, label in sorted(groups):
        rows = sorted(groups[(system_id, label)], key=lambda r: r["priority_score"], reverse=True)
        for rank, row in enumerate(farthest_point_select(rows, per_class), start=1):
            copy = dict(row)
            copy["representative_rank_within_class"] = rank
            copy["selection_reason"] = "farthest_point_sampling_in_descriptor_space"
            chosen.append(copy)
    return sorted(chosen, key=lambda r: (r["system_id"], r["transition_class"], r["representative_rank_within_class"]))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site-csv",
        action="append",
        required=True,
        help="Candidate-site CSV as PATH or SYSTEM_ID::SOURCE_NAMESPACE=PATH. Repeatable.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-delta-z-A", type=float, default=0.4)
    parser.add_argument("--max-delta-z-A", type=float, default=8.0)
    parser.add_argument("--max-energy-delta-eV", type=float, default=6.0)
    parser.add_argument("--representatives-per-class", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    sites, sources = read_sites(args.site_csv)
    sites = annotate_sites(sites)
    if len(sites) < 2:
        raise SystemExit("Need at least two candidate sites with z coordinates.")
    pairs = build_pairs(sites, args.min_delta_z_A, args.max_delta_z_A, args.max_energy_delta_eV)
    queue = select_representatives(pairs, args.representatives_per_class)

    site_fields = ["site_id", "raw_site_id", "system_id", "source_namespace", "geometry", "migrant_z_A", "screening_energy_eV", "z_rank_score", "geometry_complexity_score", "stability_score", "qe_input_path", "source_csv", "source_sha256"]
    pair_fields = ["path_id", "system_id", "start_site_id", "end_site_id", "start_geometry", "end_geometry", "delta_z_A", "abs_delta_z_A", "screening_energy_delta_eV", "geometry_complexity_delta", "transition_class", "penetration_index", "gap_index", "priority_score"]
    queue_fields = pair_fields + ["representative_rank_within_class", "selection_reason"]
    write_csv(out_dir / "candidate_sites.csv", sites, site_fields)
    write_csv(out_dir / "candidate_pairs.csv", pairs, pair_fields)
    write_csv(out_dir / "representative_path_queue.csv", queue, queue_fields)
    manifest = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "inputs": sources,
        "parameters": vars(args),
        "outputs": {"candidate_sites": str(out_dir / "candidate_sites.csv"), "candidate_pairs": str(out_dir / "candidate_pairs.csv"), "representative_path_queue": str(out_dir / "representative_path_queue.csv")},
        "counts": {"sites": len(sites), "candidate_pairs": len(pairs), "representatives": len(queue)},
        "quality_checks": {
            "site_key_unique": len({s["site_id"] for s in sites}) == len(sites),
            "path_key_unique": len({p["path_id"] for p in pairs}) == len(pairs),
            "cross_system_pairs": sum(
                1
                for p in pairs
                if p["start_site_id"].split("::", 1)[0]
                != p["end_site_id"].split("::", 1)[0]
            ),
            "missing_screening_energy_sites": sum(s["screening_energy_eV"] is None for s in sites),
        },
        "notes": ["This script proposes endpoint pairs; it does not claim a DFT barrier.", "Bare input paths are isolated into separate systems. Cross-source pairing requires an explicit shared SYSTEM_ID.", "Transition classes are bins over continuous descriptors and must be stored with the dataset.", "Representative paths should be escalated to MLFF pre-NEB first, then QE NEB if scientifically useful."],
    }
    (out_dir / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
