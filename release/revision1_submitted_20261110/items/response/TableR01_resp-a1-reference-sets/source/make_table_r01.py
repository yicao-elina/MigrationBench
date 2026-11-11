#!/usr/bin/env python3
"""Response Table R1 (tab:resp-a1-reference-sets): migration-path reference sets.

* Legacy G1 and the discarded 82-atom record: the A1 protocol audit CSV (atoms, images, peak,
  status); G1 is cross-checked against dft_neb_barriers.csv (path 1-4).
* Canonical in-gap: the 81-atom Rockfish QE contract (nat, num_of_images) and QE summary
  (forward activation, status); cross-checked against the Fig. 5 in-gap reference.
The "Validation status" text is a label map keyed by each record's status field.
Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_num  # noqa: E402

AUDIT = "paper/response_to_reviewers/reviewer_a1_fig3_figs3_update/fig3d_figS3_protocol_side_by_side_audited.csv"
BARRIERS = "paper/revision1/data_processed/dft_neb_barriers.csv"
CR81 = "data/raw_compact/cr81_dft_neb_reference_20261013"
CONTRACT = f"{CR81}/rockfish_cr_contract.tsv"
QE_SUMMARY = f"{CR81}/rockfish_cr_qe_summary.csv"
FIG5 = "paper/revision1/data_processed/current_release/fig5_plotted_values.tsv"

G1_RECORD, G1_AUDIT_ID = "DFT_reference", "1-4"
DISCARDED_RECORD = "Foundation_historical_selfNEB"
IN_GAP_PATH_ID, IN_GAP_FIG5_COLUMN = "2L-Cu__neb-gap-diffuse", "migration_in_gap_fixed"
DECIMALS = 2
TOLERANCE_EV = 1e-5
STATUS_TEXT = {
    "PROVISIONAL_REFERENCE": "Baseline diagnostic; restricted to legacy control",
    "COMPLETED_BUT_SCF_WARNING": "Diagnostic reference; primary canonical benchmark",
    "EXCLUDE_FROM_PUBLICATION": "Calculator mismatch; excluded from benchmark",
}


def read(path: Path, delimiter: str = ",") -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def main() -> int:
    ev = Evidence(__file__, "tab:resp-a1-reference-sets")
    audit = {r["record"]: r for r in read(ev.input(AUDIT))}
    barriers = {r["path"]: r for r in read(ev.input(BARRIERS))}
    contract = next(r for r in read(ev.input(CONTRACT), "\t") if r["path_id"] == IN_GAP_PATH_ID)
    qe = next(r for r in read(ev.input(QE_SUMMARY)) if r["path_id"] == IN_GAP_PATH_ID)
    fig5 = {float(r["dft_reference_eV"]) for r in read(ev.input(FIG5), "\t") if r["column_id"] == IN_GAP_FIG5_COLUMN}

    g1, discarded = audit[G1_RECORD], audit[DISCARDED_RECORD]
    g1_csv = barriers[G1_AUDIT_ID]
    in_gap_peak = float(qe["qe_forward_activation_eV"])
    problems = []
    if abs(float(g1["barrier_eV"]) - float(g1_csv["E_act_fwd_eV"])) > TOLERANCE_EV or \
            int(g1["images"]) != int(g1_csv["n_images"]):
        problems.append(f"G1 audit {g1['barrier_eV']}/{g1['images']} vs dft_neb_barriers {g1_csv}")
    if any(abs(v - in_gap_peak) > TOLERANCE_EV for v in fig5) or not fig5:
        problems.append(f"in-gap QE summary {in_gap_peak} vs Fig. 5 reference {sorted(fig5)}")
    ev.check("G1 equals dft_neb_barriers.csv; canonical in-gap equals the Fig. 5 reference", not problems,
             problems or "consistent")

    rows = [
        ["Legacy G1", g1["atoms"], g1["images"], fmt_num(g1["barrier_eV"], DECIMALS), STATUS_TEXT[g1["status"]]],
        ["Canonical in-gap", contract["nat"], contract["num_of_images"], fmt_num(in_gap_peak, DECIMALS),
         STATUS_TEXT[qe["status"]]],
        ["Discarded record", discarded["atoms"], discarded["images"], fmt_num(discarded["barrier_eV"], DECIMALS),
         STATUS_TEXT[discarded["status"]]],
    ]
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
