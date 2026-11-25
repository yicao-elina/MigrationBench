#!/usr/bin/env python3
"""Preflight-only geometry plot; not a DFT-vs-MACE scientific figure."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "initialization_smoke.csv")
df["label"] = df["path_id"].str.replace("2L-", "", regex=False).str.replace("__", " ", regex=False)
lin = df[df.initialization == "linear"].sort_values("min_pair_distance_A")
fig, ax = plt.subplots(figsize=(7.2, 4.5))
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
ax.barh(lin["label"], lin["min_pair_distance_A"], color="#4C78A8", alpha=0.82, label="minimum-image linear")
idpp = df[df.initialization == "idpp"]
if not idpp.empty:
    ax.scatter(idpp["min_pair_distance_A"], idpp["label"], color="#F58518", s=46, zorder=3, label="IDPP smoke (1 step)")
ax.axvline(0.70, color="#555555", linestyle="--", linewidth=1.2, label="overlap screen: 0.70 Å")
ax.set_xlabel("minimum pair distance across initialized images (Å)", fontweight="bold")
ax.set_ylabel("endpoint pair", fontweight="bold")
ax.set_title("Endpoint-only initialization geometry preflight", fontweight="bold")
ax.tick_params(axis="both", width=1.4, length=4, labelsize=9)
for s in ax.spines.values(): s.set_linewidth(1.4)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.text(0.01, -0.02, "Preflight geometry only; no DFT-converged path and no MACE energy/force result.", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "endpoint_initialization_smoke.png", dpi=220, bbox_inches="tight")
fig.savefig(ROOT / "endpoint_initialization_smoke.svg", bbox_inches="tight")
fig.savefig(ROOT / "endpoint_initialization_smoke.pdf", bbox_inches="tight")
print("wrote preflight figure")
