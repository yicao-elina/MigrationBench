#!/usr/bin/env python3
"""Response Table R7 (tab:resp-c9-md-endpoints): MD continuation endpoints for 15 models.

Final clock and status from the MD15 status audit; continuation = final clock minus the
shared start state (md_protocol.start_step x timestep_fs in the MD protocol manifest).
The submitted Outcome wording is mapped from the audit status code (OUTCOME_LABEL);
the mapping is recorded in the evidence. Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

TOOLS = next(p for p in Path(__file__).resolve().parents if (p / "tools/repro_lib.py").exists()) / "tools"
sys.path.insert(0, str(TOOLS))
from repro_lib import Evidence, fmt_int  # noqa: E402

RELEASE = "paper/revision1/data_processed/current_release/"
AUDIT = RELEASE + "MD15_STATUS_AUDIT_R02.tsv"
FIGURE_DATA = RELEASE + "MD15_FIGURE_DATA_R06G.tsv"
PROTOCOL = RELEASE + "md_protocol_manifest.json"
FAMILIES = ["Foundation", "FT-600K", "FT-MultiT", "Scratch-600K", "Scratch-MultiT"]  # table row order
OUTCOME_LABEL = {  # audit status code -> submitted Outcome wording
    "TARGET_REACHED": "Target reached",
    "TARGET_REACHED_PROVISIONAL": "Target reached",
    "SCIENTIFIC_FAILURE": "Scientific stop",
}
FS_PER_PS = 1000.0
CLOCK_DECIMALS = 1


def clock(value: float) -> str:
    """'1,000' for whole picoseconds, '237.2' otherwise (as typeset)."""
    rounded = round(value, CLOCK_DECIMALS)
    return fmt_int(rounded) if rounded == int(rounded) else f"{rounded:.{CLOCK_DECIMALS}f}"


def main() -> int:
    ev = Evidence(__file__, "tab:resp-c9-md-endpoints")
    audit = pd.read_csv(ev.input(AUDIT), sep="\t")
    figure = pd.read_csv(ev.input(FIGURE_DATA), sep="\t").set_index("model_id")
    protocol = json.loads(ev.input(PROTOCOL).read_text())
    md = protocol["md_protocol"]
    start_ps = md["start_step"] * md["timestep_fs"] / FS_PER_PS

    ev.check("all 15 runs continue from the same input as the protocol manifest",
             set(audit["input_sha256"]) == {protocol["input_sha256"]},
             {"start_ps": start_ps, "input_sha256": protocol["input_sha256"]})
    joined = audit.set_index("model_id")[["last_time_ps", "status"]].join(
        figure[["last_time_ps", "status"]], rsuffix="_figure")
    ev.check("status audit agrees with the MD15 figure data (final clock, status)",
             bool((joined["last_time_ps"] == joined["last_time_ps_figure"]).all()
                  and (joined["status"] == joined["status_figure"]).all()), f"{len(joined)} models compared")
    statuses = sorted(audit["status"].unique())
    ev.check("Outcome label map covers every audit status (code -> submitted wording)",
             set(statuses) <= set(OUTCOME_LABEL),
             {code: OUTCOME_LABEL.get(code) for code in statuses} | {
                 "note": "PROVISIONAL qualifier of the foundation runs and the 'scientific_threshold' reason are "
                         "shown as 'Target reached' / 'Scientific stop' in the submitted table"})

    rows = []
    for family in FAMILIES:
        group = audit[audit["family"] == family]
        if family != "Foundation":
            group = group.sort_values("training_seed_or_variant")
        for position, run in enumerate(group.itertuples()):
            seed = run.display_label if family == "Foundation" else str(run.training_seed_or_variant)
            rows.append([family if position == 0 else "", seed, clock(run.last_time_ps),
                         clock(run.last_time_ps - start_ps), OUTCOME_LABEL[run.status]])
    ev.check("every audited run appears once", len(rows) == len(audit), {"rows": len(rows), "audit": len(audit)})
    ev.compare_table(rows, "response_to_reviewers.tex")
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
