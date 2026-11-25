#!/usr/bin/env python3
"""Build an Overleaf-ready figure review book from the figure audit.

The book is intentionally separate from the manuscript source.  It gives every
audited logical figure one page with a canonical export, an evidence-bounded
caption draft, results/discussion prose, article placement, and provenance
links.  It is a review artifact: it does not silently promote a candidate to a
paper result.
"""

from __future__ import annotations

import argparse
import csv
import collections
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path("/Users/alina/Documents/ChatGPT/Wiley-MigrationBench")
AUDIT = ROOT / "analysis/figure_evidence_audit_20261008"
PAPER = Path("/Users/alina/Project/migrationbench-paper-repro/paper/revision1")

SOURCE_ROOTS = {
    "wiley-migrationbench": ROOT,
    "wiley-migrationbench-ablation-worktree": Path(
        "/Users/alina/.codex/worktrees/bf6e/Wiley-MigrationBench"
    ),
    "paper-repro": Path("/Users/alina/Project/migrationbench-paper-repro"),
}


GROUP_TEXT = {
    "FIG-1-DIFFUSION": (
        "diffusion-aware benchmark framing, connecting the motivation, the evaluation protocol, "
        "and the audited dynamic evidence",
        "This composite is intended to connect the benchmark question to the observable that is actually measured. "
        "It should be read as a context-to-result figure rather than as a claim that every downstream probe is converged.",
        "Use this figure at the Introduction/Results transition or as a high-level overview. "
        "The path-level and transport-level claims remain limited by the status fields recorded in the audit matrix.",
    ),
    "MD-1NS": (
        "one-nanosecond molecular-dynamics stability and diffusion diagnostics across the audited model set",
        "The panels summarize trajectory integrity, thermal/force behavior, structural diagnostics, and diffusion observables where the required trajectory and convergence gates are available. "
        "Failed or excluded branches remain visible rather than being converted into numerical transport estimates.",
        "This figure can support dynamic-stability discussion. It must not be used to imply heat-flux-derived thermal conductivity when the same-campaign heat-flux series is absent; such quantities remain explicitly unavailable.",
    ),
    "MD-1NS-LEGACY": (
        "legacy one-nanosecond molecular-dynamics diagnostics retained for provenance comparison",
        "The figure records an earlier diagnostic view of the molecular-dynamics campaign and is useful for tracing how the analysis evolved. "
        "It is not automatically equivalent to the matched-protocol re-analysis.",
        "Use as historical or audit material only unless the local manifest confirms the current protocol, seeds, trajectory completeness, and acceptance gates.",
    ),
    "MD-ABLATION-20PS": (
        "from-scratch ablation screening under the declared short-time molecular-dynamics protocol",
        "The ablation panels distinguish complete short screening trajectories from early scientific instability and launcher/resource failures. "
        "The screening outcome is therefore a model/protocol triage signal, not a formal transport result.",
        "The 20 ps screen supports selection and failure analysis. It does not replace the pending accepted 1 ns campaign and must not be used to report Dxy, HFACF, or kappa for failed branches.",
    ),
    "MD-CONV": (
        "molecular-dynamics convergence, block statistics, or transport diagnostics",
        "The displayed curves or summaries show the convergence behavior and uncertainty treatment available in the corresponding source analysis. "
        "Only quantities backed by complete trajectories and the declared block protocol should enter manuscript ranking.",
        "Use for methods or SI validation. Short, mismatched, or provenance-only trajectories remain labeled as such and are excluded from formal cross-model conclusions.",
    ),
    "P1-FORCE": (
        "static force-disagreement and model-transfer diagnostics",
        "The figure compares model forces on a common set of configurations and exposes disagreement patterns that are not reducible to a single aggregate RMSE. "
        "The exact sampling scope and model provenance are retained in the evidence row.",
        "This supports the static-reference part of the benchmark. It does not by itself establish stability on a dynamically evolving path or agreement with converged DFT barriers.",
    ),
    "P2-HANDOFF": (
        "handoff molecular-dynamics diagnostics and first-divergence audit",
        "The panels track the temperature, force, contact, and structural observables used to locate the first anomalous behavior after protocol handoff. "
        "Run-level integrity and local-neighborhood evidence are kept alongside the visualization.",
        "This is a diagnostic figure for identifying when and how trajectories depart from the accepted regime. It is not a replacement for a completed, gate-passing production trajectory.",
    ),
    "P3-DFT": (
        "DFT validation input, force, or candidate-quality audit",
        "The figure reports the audited state of the DFT validation inputs and the available force-level checks. "
        "Input identity, atom counts, ordering, and MACE/DFT scope are separated from claims about converged Quantum ESPRESSO output.",
        "Use to document readiness and failure boundaries. A finite MACE input audit or a submitted QE job is not itself an accepted DFT reference; convergence and normal termination remain required.",
    ),
    "TASK2-ROSTER": (
        "cross-element or multi-seed benchmark roster and pathway diagnostics",
        "The figure organizes the evaluated systems, seeds, models, or pathway families so that coverage and missing cells are visible. "
        "Provisional and audited rows are not pooled when their protocols or provenance differ.",
        "This is appropriate for SI coverage and provenance discussion. It should not be read as a single uncertainty distribution across model-level and path-level variation.",
    ),
    "TASK3-NEB": (
        "NEB pathway, barrier, topology, or self-consistency diagnostics",
        "The panels distinguish fixed-path scoring, self-consistent MLFF-NEB behavior, endpoint/path topology checks, and DFT convergence status. "
        "These operators answer different questions and are kept separate in the text.",
        "A model-reported self-relaxed barrier must not be presented as the fixed-path DFT error, and a provisional or unconverged DFT path must not be promoted to a final reference barrier.",
    ),
    "ARCHIVE-ENDPOINT": (
        "archived endpoint-relaxation or endpoint-quality evidence",
        "The figure preserves endpoint candidates and their quality flags as an auditable historical record. "
        "Endpoint stability is reported separately from path and transition-state validity.",
        "Use for provenance and candidate selection. Endpoints alone cannot validate a migration barrier or a complete NEB pathway.",
    ),
    "ARCHIVE-LEAKAGE": (
        "archived training/test proximity or leakage audit",
        "The figure shows the available proximity or leakage diagnostics under the declared representation and threshold. "
        "The source scope is part of the result and must remain visible.",
        "This supports a data-separation claim only for the audited scope; it does not generalize to model families or datasets that were not included in the audit.",
    ),
    "ARCHIVE-NEB-BARRIER": (
        "archived NEB barrier inventory with convergence and provenance flags",
        "The bars or points retain candidate, unconverged, diverged, and accepted statuses rather than collapsing them into one ranking. "
        "Barrier values are interpreted only together with path identity and convergence metadata.",
        "Only rows that satisfy the final convergence criterion may serve as quantitative reference values. The remaining entries are useful for audit history and restart planning.",
    ),
    "ARCHIVE-NEB-PROFILE": (
        "archived NEB energy-profile or minimum-energy-path comparison",
        "The profiles display the parsed path-level energy variation while preserving the distinction between reference candidates and formally converged paths. "
        "The plotted path identity is part of the evidence package.",
        "The figure supports qualitative path comparison and auditability. It must not imply final barrier accuracy for profiles that remain flagged as provisional or incomplete.",
    ),
    "ARCHIVE-PRECOND": (
        "archived calculation precondition or readiness audit",
        "The figure summarizes the preconditions checked before a calculation was eligible for interpretation. "
        "Missing or failed prerequisites remain explicit.",
        "Use as a methods/audit figure rather than as a scientific result. Readiness does not substitute for a completed accepted calculation.",
    ),
    "ARCHIVE-SELFNEB": (
        "archived self-consistent MLFF-NEB evidence",
        "The figure records the available model-relaxed NEB behavior and separates convergence from instability. "
        "The reported barrier, when present, is a model-reported value on that model's own optimized band.",
        "This supports the narrower claim of self-consistent path-generation stability. It is not interchangeable with fixed-geometry DFT-path scoring.",
    ),
    "ARCHIVE-TRAJ-PORTFOLIO": (
        "archived trajectory portfolio and model/protocol comparison",
        "The figure assembles the available trajectory branches and their associated protocol or quality metadata. "
        "Different seeds, model families, and trajectory lengths remain separately identifiable.",
        "Use for coverage and provenance. Formal ranking requires the matched-protocol subset and accepted trajectory gates.",
    ),
    "COMPILED-DOCUMENT": (
        "compiled manuscript or response artifact retained as document-level evidence",
        "This entry is a compiled document rather than an independent scientific figure. It is retained because it may contain figure captions, rendered panels, or response language that must be reconciled with the source ledger.",
        "Use only as document provenance; cite the underlying figure and reproducible source chain for scientific claims.",
    ),
    "ENDPOINT-MACE": (
        "MACE endpoint energy/force or relaxation comparison",
        "The figure compares endpoint behavior across the declared MACE model set and preserves the sampled endpoint identity. "
        "It is a local structural/energetic probe, not a transition-state calculation.",
        "Use to motivate model differences at endpoints, while keeping the separate path-level and DFT convergence limitations visible.",
    ),
    "LEGACY-NEURIPS": (
        "legacy published or conference-era benchmark figure retained for historical comparison",
        "The figure is retained as an upstream reference asset. Its original protocol, data scope, and caption are not silently treated as identical to the Revision~1 audit.",
        "Use only when the manuscript explicitly labels the result as historical/upstream and states the protocol boundary.",
    ),
    "LEGACY-PUBLISHED": (
        "legacy published figure retained for source and manuscript reconciliation",
        "This page preserves the previously published visualization while linking it to the current source ledger and revision status.",
        "The figure should be replaced or relabeled if the current audit changes the underlying data scope, protocol, or interpretation.",
    ),
    "LEGACY-SHAP": (
        "legacy SHAP or feature-attribution visualization",
        "The figure displays the historical feature-attribution view and is kept separate from the corrected, frame-aligned surrogate analysis where applicable.",
        "Feature attribution is an interpretation diagnostic. It does not establish causal importance or transport/barrier accuracy without the declared surrogate-fidelity and perturbation checks.",
    ),
    "LEGACY-SI": (
        "legacy supplementary-information figure",
        "The figure is retained for SI provenance and reconciliation with the revised source ledger. The displayed result is not assumed to inherit the current audit status automatically.",
        "Use only with an explicit historical or audited label and a checked underlying source chain.",
    ),
    "PAPER-OTHER": (
        "other paper/repro figure candidate",
        "This candidate is retained because it is referenced by the paper/repro project but does not yet belong to a fully mapped primary group. Its data and manuscript linkage are listed on this page.",
        "Resolve the article label and evidence gap before promoting it into the main text or SI.",
    ),
    "REV1-APPENDIX": (
        "Revision~1 appendix figure",
        "The figure is part of the Revision~1 appendix evidence set and is presented with the current source-ledger status.",
        "The appendix placement is appropriate while the corresponding claim, data scope, and uncertainty treatment are being finalized.",
    ),
    "REV1-ARTICLE": (
        "Revision~1 article figure",
        "The figure is already associated with the Revision~1 article structure and is presented here for page-level caption and prose review.",
        "Before final submission, confirm that the caption, Results text, and source ledger agree on the current audited values and exclusions.",
    ),
    "REV1-COMPILED": (
        "compiled Revision~1 manuscript figure/document artifact",
        "This page preserves the compiled Revision~1 artifact for visual reconciliation against the editable source and source ledger.",
        "The compiled artifact is not an independent data source; use the editable figure and its reproducible chain for final claims.",
    ),
    "REV1-FOUNDATION": (
        "Revision~1 foundation-model pathway or representation diagnostic",
        "The figure compares the declared foundation-model families or representations under the audited pathway/descriptor scope. "
        "Family variation is kept distinct from training-seed uncertainty.",
        "Use for the corresponding SI or Results discussion only after the provisional reference and descriptor-space limitations are stated.",
    ),
}


# Reviewer-response annotations are deliberately evidence-bounded.  A figure
# can close only the part of a reviewer request that its source chain actually
# supports; the remaining gate is printed on the page instead of being hidden
# behind a generic ``related'' label.
REVIEWER_TEXT = {
    "REV-A1": {
        "question": "Reconcile Fig. 3d fixed-geometry DFT-path scoring with SI Fig. S3 self-consistent MLFF-NEB, including pathway identity and the DFT reference.",
        "status": "PARTIAL",
        "completed": "The fixed-path versus self-relaxed operator distinction and the narrower model-reported-barrier wording are integrated.",
        "needed": "Close the restartable QE path-1--4 convergence gate, preserve the final path/source metadata, and complete the SI Fig. S3 endpoint/checkpoint/optimizer/source audit before calling the quantitative reference final.",
    },
    "REV-A2": {
        "question": "Rule out train/test overlap between in-gap NEB images and FT-600K training frames.",
        "status": "DONE",
        "completed": "Grouped splitting and the SOAP/RMSD nearest-neighbor audit are complete for the declared training scopes; the historical 0.16 eV point is not silently reused.",
        "needed": "No additional figure evidence is required for the completed scope; retain the thresholds, source scope, and manifest in the final release.",
    },
    "REV-A3": {
        "question": "Add seed-level uncertainty and test whether the deep-penetration result is seed-stable rather than a chance outcome.",
        "status": "PARTIAL",
        "completed": "Three-seed grouped in-gap comparisons and uncertainty are available for the declared Scratch/FT-600K/Scratch-5% scope.",
        "needed": "Complete path-resolved deep-penetration classification and restarted QE references before reporting a pooled deep-penetration ranking.",
    },
    "REV-A4": {
        "question": "Rescope generalizability claims or add an independent material/dopant validation.",
        "status": "DONE",
        "completed": "The manuscript language is scoped to a candidate framework demonstrated on Cr-doped Sb2Te3; broader claims are framed as future validation.",
        "needed": "Do not add a cross-system conclusion unless the optional second-dopant calculation is actually completed and audited.",
    },
    "REV-A5": {
        "question": "Add a Scratch-5% random-initialization control to separate foundation pretraining from data-volume effects.",
        "status": "PARTIAL",
        "completed": "The three-seed Scratch-5% control and the in-gap pathway comparison are complete and can be shown with the stated small-sample caveat.",
        "needed": "Do not generalize to FT-MultiT or all pathways until the corresponding grouped retraining and path comparisons exist.",
    },
    "REV-A6": {
        "question": "Soften t-SNE/PHATE and SHAP interpretation claims and add projection/original-space and surrogate-fidelity checks.",
        "status": "PARTIAL",
        "completed": "Projection sensitivity, original-descriptor-space separation, and per-model cross-validated surrogate R2 are available; the text is scoped to descriptive/surrogate evidence.",
        "needed": "A direct MACE Cr-Cr perturbation test remains pending if a causal claim about the dominant feature is desired; otherwise retain the surrogate-scoped wording.",
    },
    "REV-A7": {
        "question": "Position the work against nearest literature and distinguish it from the earlier workshop report.",
        "status": "DONE",
        "completed": "The related-work comparison and the explicit delta from the earlier non-archival report are integrated in the manuscript.",
        "needed": "No new figure is required; keep the figure page linked to the revised narrative rather than presenting legacy panels as new evidence.",
    },
    "REV-C8": {
        "question": "Address fabricated all-zero forces in NEB exporters and prove whether they entered reported training data.",
        "status": "DONE",
        "completed": "The exporters are fail-closed and the frame-level provenance audit found no unclassified NEB-derived all-zero-force frames in the reported datasets.",
        "needed": "No new figure is required; cite the release hash and machine-readable audit manifest as the direct evidence.",
    },
    "REV-C9": {
        "question": "Resolve the asymmetric FT-600K MD transport protocol and its effect on D and kappa.",
        "status": "PENDING",
        "completed": "Legacy FT-600K transport values are quarantined and excluded from ranking; the figure can document that boundary.",
        "needed": "Run the corrected matched-protocol FT-600K long trajectory, block-average D/kappa, and close the multi-seed uncertainty before reporting a final transport comparison.",
    },
    "REV-C10": {
        "question": "Replace the frame-level random split with a grouped train/test split and overlap audit.",
        "status": "DONE",
        "completed": "Grouped-split retraining, zero-overlap checks, and the revised RMSE/barrier statistics are available for the declared models and seeds.",
        "needed": "Keep historical frame-level values visibly labeled and use grouped values for generalization claims.",
    },
    "REV-C11": {
        "question": "Compute separability in the original/high-dimensional descriptor space rather than only on a 2D embedding.",
        "status": "DONE",
        "completed": "Original-space metrics and projection-sensitivity checks are available; embedding scores are retained only as descriptive baselines.",
        "needed": "No new numerical figure is required; preserve the feature-space definition and scaling sensitivity in the caption/methods.",
    },
    "REV-C12": {
        "question": "Remove the always-zero force-sensitivity feature stub from the latent analysis.",
        "status": "DONE",
        "completed": "The inactive channel was removed and no unsupported force-sensitivity claim is made.",
        "needed": "No additional figure evidence is required unless a genuine finite-difference force-sensitivity analysis is later added.",
    },
    "REV-C13": {
        "question": "Make the SHAP pipeline and advertised repository artifacts reproducible rather than leaving empty stubs.",
        "status": "PARTIAL",
        "completed": "The corrected SHAP inputs, scripts, lineage, and locked revised outputs are present for the declared three-model/150-frame scope.",
        "needed": "Close the final release/empty-stub audit and reconcile the independent rerun with the locked historical CSV before claiming exact end-to-end regeneration.",
    },
}


def latex(text: str) -> str:
    """Escape ordinary prose for LaTeX."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "⁻": "-",
        "¹": "1",
        "²": "2",
        "³": "3",
        "±": r"\(\pm\)",
        "×": r"\(\times\)",
        "Å": r"\AA{}",
        "µ": r"\(\mu\)",
        "–": "--",
        "—": "---",
        "≤": r"\(\leq\)",
        "≥": r"\(\geq\)",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def tex_path(path: str) -> str:
    return r"\begingroup\scriptsize\url{" + path.replace("\\", "/") + r"}\endgroup"


def humanize(value: str) -> str:
    text = value.rsplit("/", 1)[-1]
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split()).strip().capitalize()


def split_sources(value: str, limit: int = 2) -> list[str]:
    if not value:
        return []
    parts = [p.strip() for p in value.split(" | ") if p.strip()]
    return parts[:limit]


def short_source(root_id: str, source: str) -> str:
    p = Path(source)
    aliases = {
        "wiley-migrationbench": "main",
        "wiley-migrationbench-ablation-worktree": "ablation",
        "paper-repro": "paper",
    }
    roots = []
    if root_id in SOURCE_ROOTS:
        roots.append((aliases.get(root_id, root_id), SOURCE_ROOTS[root_id]))
    roots.extend((aliases.get(key, key), root) for key, root in SOURCE_ROOTS.items() if key != root_id)
    for alias, root in roots:
        try:
            rel = p.relative_to(root)
            return f"{alias}/{rel.as_posix()}"
        except ValueError:
            continue
    return source


def reviewer_links(row: dict) -> list[str]:
    """Infer direct reviewer relevance from group and audited figure name."""

    group = row.get("group_id", "")
    name = row.get("logical_figure", "").lower()
    links: list[str] = []

    def add(*ids: str) -> None:
        for item in ids:
            if item not in links:
                links.append(item)

    if group == "FIG-1-DIFFUSION":
        add("REV-A4")
    if group in {"MD-1NS", "MD-1NS-LEGACY", "MD-CONV"} or (
        group not in {"FIG-1-DIFFUSION", "TASK3-NEB"}
        and any(token in name for token in ("transport", "kappa", "msd", "vacf", "diffusion"))
    ):
        add("REV-C9")
    if group == "MD-ABLATION-20PS" or any(token in name for token in ("deep", "scratch_5", "scratch5", "5pct")):
        add("REV-A3", "REV-A5")
    if group in {"TASK3-NEB", "ARCHIVE-NEB-BARRIER", "ARCHIVE-NEB-PROFILE", "ARCHIVE-SELFNEB", "REV1-FOUNDATION"} or any(
        token in name for token in ("neb", "barrier", "path_1", "path1", "foundation_profile")
    ):
        add("REV-A1")
    if group == "ARCHIVE-LEAKAGE" or any(token in name for token in ("leakage", "overlap", "proximity", "split")):
        add("REV-A2", "REV-C10")
    if group in {"LEGACY-SHAP"} or "shap" in name:
        add("REV-A6", "REV-C13")
    if any(token in name for token in ("latent", "tsne", "phate", "silhouette", "descriptor")):
        add("REV-A6", "REV-C11", "REV-C12")
    if group == "REV1-ARTICLE" and any(token in name for token in ("parity", "rmse", "performance")):
        add("REV-A2", "REV-C10")
    if group in {"P3-DFT", "ARCHIVE-NEB-BARRIER", "ARCHIVE-NEB-PROFILE"}:
        add("REV-A1")
    if group in {
        "DFT-PHASE3-INPUT-AUDIT",
        "DFT-RECOVERY-RECONSTRUCTION",
        "DFT-SINGLE-POINT-GATE",
        "FOUNDATION-PATH-GATE",
    }:
        add("REV-A1")
    if group == "NEB-TOPOLOGY-AUDIT":
        add("REV-A1", "REV-A3")
    if group == "MD-SENSITIVITY-48RUN":
        add("REV-C9")
    return links


def reviewer_summary(row: dict) -> tuple[str, list[dict]]:
    ids = reviewer_links(row)
    if not ids:
        return "NO_DIRECT_REVIEWER_FIGURE_LINK", []
    annotations = [{"id": item, **REVIEWER_TEXT[item]} for item in ids]
    statuses = {item["status"] for item in annotations}
    overall = "DONE" if statuses == {"DONE"} else "PENDING" if "PENDING" in statuses and statuses <= {"PENDING"} else "PARTIAL"
    return overall, annotations


def load_generated_packages() -> list[dict]:
    """Convert Plot Atlas evidence-package manifests into page candidates."""

    package_root = AUDIT / "generated/table_backed_packages"
    rows = []
    for manifest_path in sorted(package_root.glob("*/evidence_package_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        package_dir = manifest_path.parent
        figure_files = [Path(item["path"]) for item in manifest.get("figure_files", [])]
        figure_files = [path for path in figure_files if path.exists()]
        figure_files.sort(key=lambda path: ({".pdf": 0, ".png": 1, ".svg": 2}.get(path.suffix.lower(), 3), str(path)))
        if not figure_files:
            continue
        source_files = " | ".join(item["path"] for item in manifest.get("source_files", []))
        processed_files = " | ".join(item["path"] for item in manifest.get("processed_files", []))
        caption_source = package_dir / "caption_results_discussion.md"
        row = {
            "root_id": "wiley-migrationbench",
            "logical_figure": f"generated_table_backed/{manifest['package_id']}",
            "logical_path": f"generated_table_backed/{manifest['package_id']}",
            "group_id": manifest["package_id"],
            "role": "Plot Atlas table-backed evidence package",
            "evidence_chain_status": "manifested",
            "article_target": manifest.get("article_target", ""),
            "raw_or_processed_data_candidates": " | ".join(item for item in (source_files, processed_files) if item),
            "plot_script_candidates": manifest.get("generated_by", ""),
            "caption_discussion_source_candidates": str(caption_source),
            "status_tags": manifest.get("scientific_status", ""),
            "gaps_to_resolve": " ".join(manifest.get("caveats", [])),
            "outputs": [{"path": str(path)} for path in figure_files],
            "package_manifest": manifest,
        }
        rows.append(row)
    return rows


def preferred_output(row: dict) -> Path:
    outputs = []
    for output in row.get("outputs", []):
        p = Path(output["path"])
        if p.exists():
            outputs.append(p)
    if not outputs:
        raise FileNotFoundError(row["logical_figure"])
    outputs.sort(key=lambda p: ({".pdf": 0, ".png": 1, ".svg": 2}.get(p.suffix.lower(), 3), str(p)))
    return outputs[0]


def narrative(row: dict) -> tuple[str, str, str]:
    package = row.get("package_manifest")
    if package:
        label = humanize(row["logical_figure"])
        caption = package.get("caption", "")
        caption = f"{caption} This page shows the Plot Atlas table-backed candidate ``{label}''; its scientific status and caveats are reproduced from the package manifest."
        results = package.get("results", "")
        discussion = package.get("discussion", "")
        return caption, results, discussion
    group = row["group_id"]
    label = humanize(row["logical_figure"])
    default = (
        f"{label}; an audited figure candidate retained with its current evidence status.",
        "The available output is shown with the source-ledger and evidence-matrix context recorded on this page. "
        "No numerical claim is added beyond the available source evidence.",
        "Resolve the listed gaps before using this candidate as a final article or SI result.",
    )
    title, results, discussion = GROUP_TEXT.get(group, default)
    caption = f"{title.capitalize()}. This page shows the candidate output ``{label}'' within the audited {group} family. The panel identity, source scope, and current status are retained in the accompanying evidence matrix."
    return caption, results, discussion


def article_sentence(row: dict) -> str:
    target = row.get("article_target", "") or "Article placement pending"
    return f"Proposed article location: {target}."


def write_readme(out_dir: Path, count: int, bytes_total: int) -> None:
    text = "# Overleaf figure pages\n\n"
    text += "This is a generated review book, separate from `sn-article-final.tex` and `sn-article-SI.tex`. "
    text += "Every audited logical figure has one page with a canonical export, an evidence-bounded caption draft, Results text, Discussion/boundary text, article placement, and provenance pointers.\n\n"
    text += f"- Review pages: {count} (204 source logical figures plus table-backed audit candidates)\n- Canonical copied assets: {count}\n- Asset size: {bytes_total / 1e6:.1f} MB\n- TeX entry point: `../figure_pages.tex`\n- Source audit: `analysis/figure_evidence_audit_20261008/figure_evidence_matrix.tsv`\n- Reviewer annotation register: `figure_reviewer_annotations.tsv`\n- Reviewer coverage summary: `reviewer_coverage.md`\n\n"
    text += "The prose is deliberately conservative. Entries marked `publication gaps`, `historical`, `reconstruction`, `screening-only`, or `pending` are review candidates and must not be promoted without closing the listed gap. The page book does not change figure numbering in the manuscript. After review, copy only the accepted assets/captions into the main text or SI.\n"
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=AUDIT / "figure_inventory.json")
    parser.add_argument("--matrix", type=Path, default=AUDIT / "figure_evidence_matrix.tsv")
    parser.add_argument("--paper-root", type=Path, default=PAPER)
    parser.add_argument("--no-copy-assets", action="store_true")
    args = parser.parse_args()

    source_inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    inventory = source_inventory + load_generated_packages()
    matrix = {
        row["logical_path"]: row
        for row in csv.DictReader(args.matrix.open(encoding="utf-8"), delimiter="\t")
    }
    out_dir = args.paper_root / "figure_pages"
    assets_dir = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_copy_assets:
        assets_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict] = []
    total_bytes = 0
    manifest_rows = []
    for index, row in enumerate(inventory, start=1):
        source = preferred_output(row)
        # pdfTeX cannot include SVG directly. Convert SVG-only sources to PNG
        # while keeping the original SVG path in the audit manifest.
        asset_suffix = ".png" if source.suffix.lower() == ".svg" else source.suffix.lower()
        asset_name = f"fig_{index:03d}{asset_suffix}"
        asset = assets_dir / asset_name
        if not args.no_copy_assets:
            if source.suffix.lower() == ".svg":
                subprocess.run(["magick", str(source), str(asset)], check=True)
            else:
                shutil.copy2(source, asset)
            total_bytes += asset.stat().st_size
        matrix_row = matrix.get(row["logical_path"], {})
        merged = {**row, **matrix_row}
        caption, results, discussion = narrative(merged)
        reviewer_status, reviewer_annotations = reviewer_summary(merged)
        pages.append({
            "index": index,
            "row": merged,
            "source": source,
            "asset_name": asset_name,
            "caption": caption,
            "results": results,
            "discussion": discussion,
            "reviewer_status": reviewer_status,
            "reviewer_annotations": reviewer_annotations,
        })
        manifest_rows.append({
            "figure_id": f"FIG-{index:03d}",
            "group_id": merged.get("group_id", "unclassified"),
            "logical_figure": row["logical_figure"],
            "source_export": str(source),
            "overleaf_asset": f"figure_pages/assets/{asset_name}",
            "article_target": merged.get("article_target", ""),
            "status_tags": merged.get("status_tags", ""),
            "gaps_to_resolve": merged.get("gaps_to_resolve", ""),
            "reviewer_status": reviewer_status,
            "reviewer_ids": ",".join(item["id"] for item in reviewer_annotations),
        })

    tex_lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.72in]{geometry}",
        r"\usepackage{graphicx}",
        r"\usepackage{xcolor}",
        r"\usepackage{caption}",
        r"\usepackage{float}",
        r"\usepackage{hyperref}",
        r"\usepackage{url}",
        r"\usepackage{enumitem}",
        r"\usepackage{fancyhdr}",
        r"\usepackage{microtype}",
        r"\hypersetup{colorlinks=true,linkcolor=blue,urlcolor=blue}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{4pt}",
        r"\setlength{\emergencystretch}{3em}",
        r"\sloppy",
        r"\pagestyle{fancy}",
        r"\fancyhf{}",
        r"\lhead{MigrationBench figure review book}",
        r"\rhead{Evidence-bounded candidate pages}",
        r"\cfoot{\thepage}",
        r"\newcommand{\filepath}[1]{\par\small\texttt{\detokenize{#1}}}",
        r"\newcommand{\statusline}[1]{\colorbox{gray!12}{\parbox{0.96\linewidth}{#1}}}",
        r"\begin{document}",
        r"\begin{titlepage}",
        r"\centering",
        r"\vspace*{1.0in}",
        r"{\LARGE\bfseries MigrationBench figure review book\par}",
        r"\vspace{0.25in}",
        r"{\large Overleaf-ready candidate pages\par}",
        r"\vspace{0.5in}",
        f"{{\\large {len(pages)} audited logical figures, one page per figure\\par}}",
        r"\vfill",
        r"This document is a review artifact, not a replacement for the main manuscript or SI.\\",
        r"Every page keeps the evidence status, provenance pointers, and unresolved gaps visible.",
        r"\end{titlepage}",
        r"\tableofcontents",
        r"\clearpage",
        r"\section*{How to use this book}",
        r"Select accepted candidates from these pages, then move their final caption and Results/Discussion text into the appropriate main-text or SI location. The page labels are audit IDs, not final manuscript figure numbers. `publication gaps`, `historical`, `reconstruction`, `screening-only`, and `pending` are deliberate status boundaries.",
        r"\clearpage",
    ]

    for page in pages:
        row = page["row"]
        figure_id = f"FIG-{page['index']:03d}"
        title = humanize(row["logical_figure"])
        status = row.get("status_tags", "") or row.get("status_tags", "unclassified")
        evidence = row.get("evidence_chain_status", "") or "not explicitly classified"
        tex_lines.extend([
            r"\clearpage",
            f"\\section{{{latex(figure_id + ' — ' + title)}}}",
            f"\\label{{figpage:{figure_id.lower()}}}",
            f"\\textbf{{Group:}} {latex(row.get('group_id', ''))}\\quad "
            f"\\textbf{{Evidence chain:}} {latex(evidence)}\\quad "
            f"\\textbf{{Status:}} {latex(status)}",
            r"\begin{figure}[H]",
            r"\centering",
            f"\\includegraphics[width=\\linewidth,height=0.42\\textheight,keepaspectratio]{{figure_pages/assets/{page['asset_name']}}}",
            f"\\caption{{{latex(page['caption'])}}}",
            f"\\label{{fig:candidate_{page['index']:03d}}}",
            r"\end{figure}",
            f"\\textbf{{Results text draft.}} {latex(page['results'])}",
            f"\\textbf{{Discussion and boundary.}} {latex(page['discussion'])}",
            f"\\textbf{{Article placement.}} {latex(article_sentence(row))}",
        ])
        tex_lines.append(
            f"\\textbf{{Reviewer linkage:}} overall status {latex(page['reviewer_status'])}."
        )
        if page["reviewer_annotations"]:
            tex_lines.append(r"\begin{itemize}[leftmargin=1.4em,noitemsep]")
            for annotation in page["reviewer_annotations"]:
                tex_lines.append(
                    f"\\item \\textbf{{{latex(annotation['id'])} ({latex(annotation['status'])}):}} "
                    f"{latex(annotation['question'])} "
                    f"\\emph{{Completed:}} {latex(annotation['completed'])} "
                    f"\\emph{{Still required:}} {latex(annotation['needed'])}"
                )
            tex_lines.append(r"\end{itemize}")
        else:
            tex_lines.append(
                r"No direct reviewer-response figure linkage was identified; this page is retained for scientific/article audit rather than as a direct reviewer response."
            )
        data_sources = split_sources(row.get("raw_or_processed_data_candidates", ""))
        script_sources = split_sources(row.get("plot_script_candidates", ""))
        caption_sources = split_sources(row.get("caption_discussion_source_candidates", ""))
        tex_lines.append(r"\textbf{Evidence pointers.}")
        if data_sources or script_sources or caption_sources:
            tex_lines.append(r"\begin{itemize}[leftmargin=1.4em,noitemsep]")
            for source in data_sources:
                tex_lines.append(f"\\item Data/processed: {tex_path(short_source(row['root_id'], source))}")
            for source in script_sources:
                tex_lines.append(f"\\item Plot/processing script: {tex_path(short_source(row['root_id'], source))}")
            for source in caption_sources:
                tex_lines.append(f"\\item Caption/text source: {tex_path(short_source(row['root_id'], source))}")
            tex_lines.append(r"\end{itemize}")
        else:
            tex_lines.append(r"No data/script/text candidate was recorded in the matrix; resolve this gap before manuscript promotion.")
        gaps = row.get("gaps_to_resolve", "") or "No additional gap recorded in the matrix."
        tex_lines.append(f"\\statusline{{\\textbf{{Audit gap / next action:}} {latex(gaps)}}}")
        tex_lines.append(r"\vfill")

    tex_lines.append(r"\end{document}")
    (args.paper_root / "figure_pages.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    with (out_dir / "figure_pages_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest_rows)

    reviewer_rows = []
    for page in pages:
        if page["reviewer_annotations"]:
            for annotation in page["reviewer_annotations"]:
                reviewer_rows.append({
                    "figure_id": f"FIG-{page['index']:03d}",
                    "logical_figure": page["row"]["logical_figure"],
                    "group_id": page["row"].get("group_id", ""),
                    "overall_reviewer_status": page["reviewer_status"],
                    "reviewer_id": annotation["id"],
                    "status": annotation["status"],
                    "question": annotation["question"],
                    "completed_evidence": annotation["completed"],
                    "still_required": annotation["needed"],
                })
        else:
            reviewer_rows.append({
                "figure_id": f"FIG-{page['index']:03d}",
                "logical_figure": page["row"]["logical_figure"],
                "group_id": page["row"].get("group_id", ""),
                "overall_reviewer_status": page["reviewer_status"],
                "reviewer_id": "",
                "status": "",
                "question": "",
                "completed_evidence": "",
                "still_required": "",
            })
    reviewer_path = out_dir / "figure_reviewer_annotations.tsv"
    with reviewer_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "figure_id", "logical_figure", "group_id", "overall_reviewer_status",
            "reviewer_id", "status", "question", "completed_evidence", "still_required",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(reviewer_rows)

    coverage_lines = [
        "# Reviewer-to-figure coverage",
        "",
        "This register is generated from the page-level reviewer annotations. `DONE` means the linked figure can support the narrowed reviewer response for the declared scope; `PARTIAL` or `PENDING` means the page must retain the listed completion gate.",
        "",
    ]
    for reviewer_id, info in REVIEWER_TEXT.items():
        linked = [row for row in reviewer_rows if row["reviewer_id"] == reviewer_id]
        status_counts = collections.Counter(row["status"] for row in linked)
        figure_ids = sorted({row["figure_id"] for row in linked})
        coverage_lines.extend([
            f"## {reviewer_id} — {info['status']}",
            "",
            f"**Reviewer question:** {info['question']}",
            "",
            f"**Linked pages:** {len(figure_ids)} ({', '.join(f'{key}={value}' for key, value in sorted(status_counts.items()))})",
            f"**Figure IDs:** {', '.join(figure_ids)}",
            f"**Completed evidence:** {info['completed']}",
            f"**Still required:** {info['needed']}",
            "",
        ])
    (out_dir / "reviewer_coverage.md").write_text("\n".join(coverage_lines), encoding="utf-8")

    write_readme(out_dir, len(pages), total_bytes)
    print(f"wrote {args.paper_root / 'figure_pages.tex'} ({len(pages)} pages)")
    print(f"wrote {out_dir / 'figure_pages_manifest.tsv'}")
    print(f"wrote {reviewer_path}")
    print(f"wrote {out_dir / 'reviewer_coverage.md'}")
    if not args.no_copy_assets:
        print(f"copied {len(pages)} canonical assets ({total_bytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
