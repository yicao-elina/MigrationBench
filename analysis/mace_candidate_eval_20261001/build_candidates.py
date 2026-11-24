from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "mace_candidate_eval_20261001"
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = OUT / "inputs"
INPUTS.mkdir(exist_ok=True)
ENDPOINT_RAW = ROOT / "analysis" / "mace_endpoint_four_model" / "raw"
ENDPOINT_OUT = ROOT / "analysis" / "mace_endpoint_four_model" / "outputs"
NEB_ROOT = Path("/Users/alina/Documents/Codex/2026-09-21/new-chat/work/mace_path_campaign")
NEB_PLOTS = NEB_ROOT / "plots"
NEB_EXTRACTED = NEB_ROOT / "rockfish_results" / "mace36_compact_extracted"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def endpoint_source(task_id: str, model: str = "FT-600K") -> Path:
    if model == "FT-600K":
        hits = list(ENDPOINT_RAW.glob(f"**/results/*{task_id}/mace_relaxed.extxyz"))
    else:
        hits = list(ENDPOINT_RAW.glob(f"**/runs/{model}/{task_id}/attempt_*/mace_relaxed.extxyz"))
    if not hits:
        raise FileNotFoundError((model, task_id))
    return sorted(hits)[0]


def endpoint_scores() -> pd.DataFrame:
    tidy = pd.read_csv(ENDPOINT_OUT / "tidy_endpoint_relaxation_results.csv")
    tidy["success"] = tidy["success"].astype(bool)
    ref = tidy[tidy.model.eq("FT-600K")][["endpoint", "variant", "x", "y", "z", "success"]].rename(
        columns={"x": "x_ref", "y": "y_ref", "z": "z_ref", "success": "ref_success"}
    )
    x = tidy.merge(ref, on=["endpoint", "variant"], how="left")
    x["cr_dist_to_ft600k_A"] = np.sqrt((x.x - x.x_ref) ** 2 + (x.y - x.y_ref) ** 2 + (x.z - x.z_ref) ** 2)
    g = x.groupby(["endpoint", "endpoint_type"], as_index=False).agg(
        n_success=("success", "sum"), n_records=("success", "size"),
        mean_pair_disagreement_A=("cr_dist_to_ft600k_A", "mean"),
        max_pair_disagreement_A=("cr_dist_to_ft600k_A", "max"),
        n_failed=("success", lambda s: int((~s).sum())),
    )
    g["priority"] = g.endpoint.isin({"unique_044", "unique_067", "unique_096", "unique_097", "unique_109", "unique_151", "unique_065", "unique_080"})
    return g.sort_values(["priority", "mean_pair_disagreement_A"], ascending=[False, False])


def pick_endpoint_rows() -> list[dict]:
    scores = endpoint_scores()
    # 8 named priorities plus the highest-disagreement points, balanced across the 3 stated mechanisms.
    selected: list[str] = []
    for ep in scores[scores.priority].endpoint.tolist():
        if ep not in selected:
            selected.append(ep)
    for typ in ["same-gap lateral", "coordination switch", "gap-to-layer penetration"]:
        for ep in scores[scores.endpoint_type.eq(typ)].endpoint.tolist():
            if ep not in selected and len(selected) < 24:
                selected.append(ep)
    score_map = scores.set_index("endpoint").to_dict("index")
    tidy = pd.read_csv(ENDPOINT_OUT / "tidy_endpoint_relaxation_results.csv")
    rows = []
    for ep in selected:
        q = tidy[(tidy.endpoint == ep) & (tidy.model == "FT-600K") & (tidy.variant == "base")]
        if q.empty or not bool(q.iloc[0].success):
            q = tidy[(tidy.endpoint == ep) & (tidy.model == "Foundation") & (tidy.variant == "base") & tidy.success]
        if q.empty:
            q = tidy[(tidy.endpoint == ep) & tidy.success].sort_values(["model", "variant"])
        if q.empty:
            continue
        rec = q.iloc[0]
        src = endpoint_source(str(rec.task_id), str(rec.model))
        s = score_map[ep]
        rows.append({
            "category": "stable_endpoint", "mechanism_group": rec.endpoint_type,
            "endpoint_id": ep, "path_id": "", "cov_id": "", "image_index": "",
            "reaction_coordinate": "", "barrier_eV": "", "candidate_reason":
                "named_priority" if bool(s["priority"]) else "largest_endpoint_model_disagreement",
            "endpoint_type": rec.endpoint_type, "source_path": str(src), "source_hash": sha256(src),
            "source_identity": f"{rec.model}:{rec.task_id}", "input_path": "", "input_sha256": "",
            "md_status": "not_applicable", "selection_score": float(s["mean_pair_disagreement_A"]),
        })
    return rows


def neb_file_map() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in NEB_EXTRACTED.glob("**/mlff_neb_images.extxyz"):
        m = re.search(r"mlff_(cov_\d+)_", p.parent.name)
        if not m:
            continue
        cov = m.group(1)
        text = str(p).lower()
        # Provenance preference: MACE-converged extraction, then segmented/recovery MACE; never idpp_*.extxyz.
        rank = (0 if "mace" in text else 10) + (0 if "segidpp_mace" in text else 1) + (0 if "restart" not in text else 2)
        if cov not in out or rank < out[cov].stat().st_mtime:
            out[cov] = p
    # The mtime comparison above is intentionally replaced by deterministic rank selection below.
    for cov in set(re.search(r"mlff_(cov_\d+)_", p.parent.name).group(1) for p in NEB_EXTRACTED.glob("**/mlff_neb_images.extxyz") if re.search(r"mlff_(cov_\d+)_", p.parent.name)):
        hits = [p for p in NEB_EXTRACTED.glob(f"**/mlff_{cov}_*/mlff_neb_images.extxyz")]
        if not hits:
            continue
        def key(p: Path):
            t = str(p).lower()
            return (0 if "mace" in t else 10, 0 if "segidpp_mace" in t else 1, 0 if "restart" not in t else 2, str(p))
        out[cov] = sorted(hits, key=key)[0]
    return out


def barrier_rows() -> tuple[pd.DataFrame, dict[str, Path]]:
    prof = pd.read_csv(NEB_PLOTS / "mace_neb_profiles_36.csv")
    barr = pd.read_csv(NEB_PLOTS / "mace_neb_barriers_36.csv")
    return prof, barr.set_index("cov_id").to_dict("index")


def add_neb_rows() -> list[dict]:
    prof, bm = barrier_rows()
    fmap = neb_file_map()
    rows: list[dict] = []
    # One central image from every path; this preserves all 36 path identities with a compact 36-image screen.
    for cov, g in prof.groupby("cov_id", sort=True):
        mid = g.iloc[(g.image_index - 3).abs().argsort().iloc[0]]
        b = bm[cov]
        src = fmap.get(cov)
        if src is None:
            continue
        rows.append({"category": "neb_intermediate", "mechanism_group": b["mechanism_group"], "endpoint_id": "",
                     "path_id": b["path_id"], "cov_id": cov, "image_index": int(mid.image_index),
                     "reaction_coordinate": float(mid.reaction_coordinate), "barrier_eV": float(b["forward_barrier_eV"]),
                     "candidate_reason": "central_image_all_36_paths", "endpoint_type": "", "source_path": str(src),
                     "source_hash": sha256(src), "source_identity": f"{cov}:image_{int(mid.image_index)}",
                     "input_path": "", "input_sha256": "", "md_status": "not_applicable",
                     "selection_score": float(b["forward_barrier_eV"])})
    # Highest barrier plus explicit anomaly priorities, with at least two paths per mechanism.
    priority = ["cov_020", "cov_035", "cov_002", "cov_027", "cov_031"]
    chosen = []
    for cov in priority + list(prof.groupby("cov_id").size().index):
        if cov in bm and cov not in chosen:
            chosen.append(cov)
    for typ in ["same-gap lateral", "coordination switch", "gap→layer"]:
        candidates = [c for c in bm if bm[c]["mechanism_group"] == typ]
        for c in sorted(candidates, key=lambda c: bm[c]["forward_barrier_eV"], reverse=True):
            if c not in chosen:
                chosen.append(c)
                break
    chosen = chosen[:7]
    existing = {(r["cov_id"], r["image_index"]) for r in rows}
    for cov in chosen:
        g = prof[prof.cov_id.eq(cov)].sort_values("relative_energy_eV")
        imax = int(g.loc[g.relative_energy_eV.idxmax(), "image_index"])
        b = bm[cov]; src = fmap.get(cov)
        if src is None: continue
        for i in sorted({imax - 1, imax, imax + 1}):
            q = prof[(prof.cov_id == cov) & (prof.image_index == i)]
            if q.empty or (cov, i) in existing: continue
            q = q.iloc[0]
            rows.append({"category": "near_saddle", "mechanism_group": b["mechanism_group"], "endpoint_id": "",
                         "path_id": b["path_id"], "cov_id": cov, "image_index": i,
                         "reaction_coordinate": float(q.reaction_coordinate), "barrier_eV": float(b["forward_barrier_eV"]),
                         "candidate_reason": "highest_energy_and_adjacent_images", "endpoint_type": "", "source_path": str(src),
                         "source_hash": sha256(src), "source_identity": f"{cov}:image_{i}", "input_path": "",
                         "input_sha256": "", "md_status": "not_applicable", "selection_score": float(q.relative_energy_eV)})
    return rows


def materialize(rows: list[dict]) -> pd.DataFrame:
    manifest = []
    for i, r in enumerate(rows, 1):
        cid = f"cand_{i:04d}"
        src = Path(r["source_path"])
        if r["category"] == "stable_endpoint":
            atoms = read(src, index=0)
        else:
            frames = read(src, index=":")
            atoms = frames[int(r["image_index"])]
        out = INPUTS / f"{cid}.extxyz"
        write(out, atoms, format="extxyz")
        r = dict(r)
        r.update({"candidate_id": cid, "input_path": str(out), "input_sha256": sha256(out),
                  "n_atoms": len(atoms), "pbc": json.dumps(atoms.pbc.tolist()),
                  "cell_A": json.dumps(np.asarray(atoms.cell).round(8).tolist())})
        manifest.append(r)
    df = pd.DataFrame(manifest)
    cols = ["candidate_id", "category", "mechanism_group", "endpoint_id", "endpoint_type", "path_id", "cov_id", "image_index", "reaction_coordinate", "barrier_eV", "candidate_reason", "source_path", "source_hash", "source_identity", "input_path", "input_sha256", "n_atoms", "pbc", "cell_A", "md_status", "selection_score"]
    df = df[cols].sort_values("candidate_id").reset_index(drop=True)
    df.to_csv(OUT / "candidate_manifest.csv", index=False)
    (OUT / "candidate_manifest.json").write_text(json.dumps({"schema_version": "1.0", "created": "2026-10-01", "n_candidates": len(df), "md_missing": True, "rows": df.to_dict("records")}, indent=2))
    df.to_csv(OUT / "candidate_inventory.csv", index=False)
    return df


if __name__ == "__main__":
    rows = pick_endpoint_rows() + add_neb_rows()
    # MD is deliberately represented as an explicit missing category in the ledger, not as fabricated structures.
    df = materialize(rows)
    print(df.groupby(["category", "mechanism_group"], dropna=False).size().to_string())
    print(f"wrote {len(df)} candidates to {OUT}")
