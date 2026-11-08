"""Build the Cr migration-path registry: reader-facing names <-> internal audit IDs.

Outputs (all under paper/revision1):
  data_processed/path_registry.csv   machine-readable registry (also copied to data/processed)
  tables/path_registry.tex           SI table with names, classes, audit IDs, and classification basis
  tables/dft_neb_barriers.tex        SI table of QE path references keyed by registry name
"""
from pathlib import Path
import csv
import re

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parents[1]
BARRIERS_CSV = PAPER / "data_processed" / "dft_neb_barriers.csv"
REGISTRY_CSV = PAPER / "data_processed" / "path_registry.csv"
REGISTRY_COPY = REPO / "data" / "processed" / "path_registry.csv"
REGISTRY_TEX = PAPER / "tables" / "path_registry.tex"
BARRIERS_TEX = PAPER / "tables" / "dft_neb_barriers.tex"

QUEUE = "configs/experiments/migrationbench_path_queue.yaml"
EXACT = "paper/revision1/figures/si_path_audit/AUDIT_REPORT.md"
CHAIN = "data/published/path_classification/81_atom_paths_v2_pbc.csv"

GAP = "within-gap"
QL = "gap-to-QL/into-QL"
MIXED = "mixed"
WITHIN_QL = "within-QL"

# name, class, atoms, audit_id, source label, basis, evidence
REGISTRY = [
    ("G1", GAP, 61, "1-4", "in_gap", "Cr stays in the vdW gap between sites defined as in-gap in the path queue; image-set reference path", QUEUE),
    ("G2", GAP, 61, "1-7", "in_gap", "Cr stays in the vdW gap between sites defined as in-gap in the path queue", QUEUE),
    ("Q1", QL, 61, "1-2", "deep_penetration", "Path queue sites place Cr moving from the vdW gap into the quintuple layer", QUEUE),
    ("Q2", QL, 61, "1-3", "deep_penetration", "Path queue sites place Cr moving from the vdW gap into the quintuple layer", QUEUE),
    ("Q3", QL, 61, "1-5", "deep_penetration", "Path queue sites place Cr moving from the vdW gap into the quintuple layer; the queue asks for a geometry recheck", QUEUE),
    ("M1", MIXED, 61, "1-6", "mixed_transition", "Exact image coordinates give a mixed gap/QL transition", EXACT),
    ("L1", WITHIN_QL, 61, "1-8", "probable_deep_penetration", "Exact image coordinates give probable deep penetration", EXACT),
    ("C1", WITHIN_QL, 81, "neb_1", "deep_penetration", "Continuous-geometry penetration score 0.69--0.74 (gap score 0.21--0.35); five-path chain", CHAIN),
    ("C2", WITHIN_QL, 81, "neb_2", "deep_penetration", "Continuous-geometry penetration score 0.69--0.74 (gap score 0.21--0.35); five-path chain", CHAIN),
    ("C3", WITHIN_QL, 81, "neb_3", "deep_penetration", "Continuous-geometry penetration score 0.69--0.74 (gap score 0.21--0.35); five-path chain", CHAIN),
    ("C4", WITHIN_QL, 81, "neb_4", "deep_penetration", "Continuous-geometry penetration score 0.69--0.74 (gap score 0.21--0.35); five-path chain", CHAIN),
    ("C5", WITHIN_QL, 81, "neb_5", "deep_penetration", "Continuous-geometry penetration score 0.69--0.74 (gap score 0.21--0.35); five-path chain", CHAIN),
]

# QE neb.out sources (relative to the directory that contains the repository)
WORKSPACE = REPO.parent
NEB_OUT = {
    "1-4": REPO / "data/raw_compact/revision1/dft_raw/1-4/neb.out",
    **{pid: WORKSPACE / f"26MIgrationBench/data_raw/historical_neb/legacy_61/{pid}/neb.out" for pid in ("1-2", "1-3", "1-5", "1-6", "1-7", "1-8")},
    **{f"neb_{i}": REPO / f"data/raw_compact/revision1/81_atom_2D_421_Diffusion_traj/neb_{i}/neb.out" for i in range(1, 6)},
}
FORWARD = re.compile(r"activation energy \(->\)\s*=\s*([-0-9.]+)\s*eV")


def neb_record(audit_id: str) -> dict:
    """Iteration count and forward-barrier change between the last two NEB iterations."""
    source = NEB_OUT[audit_id]
    values = [float(v) for v in FORWARD.findall(source.read_text(errors="replace"))]
    last, previous = values[-1], values[-2]
    try:
        shown = str(source.relative_to(WORKSPACE))
    except ValueError:
        shown = str(source)
    return dict(neb_source=shown, neb_iterations=len(values), last_iteration_forward_eV=f"{last:.6f}",
                previous_iteration_forward_eV=f"{previous:.6f}", delta_vs_previous_eV=f"{last - previous:+.6f}")


def tex(text: str) -> str:
    return text.replace("_", r"\_").replace("--", "--")


def main() -> None:
    with BARRIERS_CSV.open() as handle:
        barriers = {row["path"]: row for row in csv.DictReader(handle)}

    fields = ["name", "class", "atoms", "audit_id", "source_geometry_label", "classification_basis",
              "evidence_file", "n_images", "qe_forward_barrier_eV", "qe_reverse_barrier_eV", "status",
              "neb_source", "neb_iterations", "last_iteration_forward_eV", "previous_iteration_forward_eV", "delta_vs_previous_eV"]
    rows = []
    for name, cls, atoms, audit_id, label, basis, evidence in REGISTRY:
        b = barriers[audit_id]
        rows.append(dict(name=name, **{"class": cls}, atoms=atoms, audit_id=audit_id,
                         source_geometry_label=label, classification_basis=basis, evidence_file=evidence,
                         n_images=b["n_images"], qe_forward_barrier_eV=b["E_act_fwd_eV"],
                         qe_reverse_barrier_eV=b["E_act_rev_eV"], status=b["status"], **neb_record(audit_id)))
    for target in (REGISTRY_CSV, REGISTRY_COPY):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    lines = [r"\small", r"\begin{tabular}{llcl p{0.40\linewidth}}", r"\toprule",
             r"Name & Class & Atoms & Audit ID & Classification basis \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['name']} & {r['class']} & {r['atoms']} & \\texttt{{{tex(r['audit_id'])}}} & {r['classification_basis']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    REGISTRY_TEX.write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{llcccl}", r"\toprule",
             r"Name & Class & $N_\mathrm{img}$ & $\mathrm{QE}_{\mathrm{ref}}^{\rightarrow}$ (eV) & $\mathrm{QE}_{\mathrm{ref}}^{\leftarrow}$ (eV) & Status \\",
             r"\midrule"]
    for r in rows:
        lines.append(f"{r['name']} & {r['class']} & {r['n_images']} & {float(r['qe_forward_barrier_eV']):.2f} & "
                     f"{float(r['qe_reverse_barrier_eV']):.2f} & {r['status']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    BARRIERS_TEX.write_text("\n".join(lines) + "\n")
    print("registry rows:", len(rows))


if __name__ == "__main__":
    main()
