#!/usr/bin/env python3
"""Table S10 (tab:current_tsne_sensitivity): t-SNE perplexity x seed silhouette and trustworthiness.

Rows are computed from repo data by si_tables.tsne_rows (items/si/_shared_generators/si_tables.py)
and compared with the submitted LaTeX at its displayed precision. Writes rerun_output/ and evidence/.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
TOOLS = next(p for p in HERE.parents if (p / "tools/repro_lib.py").exists()) / "tools"
SHARED = next(p for p in HERE.parents if (p / "_shared_generators").is_dir()) / "_shared_generators"
sys.path[:0] = [str(TOOLS), str(SHARED)]
from repro_lib import Evidence  # noqa: E402
import si_tables  # noqa: E402

SI_TEX = "sn-article-SI.tex"


def main() -> int:
    ev = Evidence(__file__, "tab:current_tsne_sensitivity")
    si_tables.record_shared_module(ev)
    ev.compare_table(si_tables.tsne_rows(ev), SI_TEX)
    return ev.write()


if __name__ == "__main__":
    raise SystemExit(main())
