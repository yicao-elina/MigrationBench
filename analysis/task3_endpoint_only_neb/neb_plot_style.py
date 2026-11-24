"""Shared Plot Atlas styling and QE-NEB uncertainty extraction.

The QE ``neb.out`` image table reports energy in eV and a force/path residual
in eV/A.  The residual is not an energy standard error and is therefore kept
as a separate audit field.  Shaded energy bands are instead computed from the
last few complete NEB outer-iteration energy tables, after image-1
referencing; this gives an energy-valued iteration-dispersion envelope.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "/Users/alina/Documents/Codex/plot_library/src")
from plot_atlas import get_palette


MODEL_ORDER = [
    "mace-large-density-agnesi-stress",
    "mace-medium-density-agnesi-stress",
    "mace-small-density-agnesi-stress",
    "mace_agnesi_medium",
    "mace-mh-0",
    "mace-mh-1",
    "mace-mp-0b3-medium",
    "mace-omat-0-medium",
    "mace-omat-0-small",
]
MODEL_LABELS = {
    "mace-large-density-agnesi-stress": "Agnesi-L-dens",
    "mace-medium-density-agnesi-stress": "Agnesi-M-dens",
    "mace-small-density-agnesi-stress": "Agnesi-S-dens",
    "mace_agnesi_medium": "Agnesi-M",
    "mace-mh-0": "MH-0",
    "mace-mh-1": "MH-1",
    "mace-mp-0b3-medium": "MP-0B3",
    "mace-omat-0-medium": "OMAT-0-M",
    "mace-omat-0-small": "OMAT-0-S",
    # IDs used by the grouped Cr benchmark.
    "agnesi_stress_large": "Agnesi-L-dens",
    "agnesi_stress_medium": "Agnesi-M-dens",
    "agnesi_stress_small": "Agnesi-S-dens",
    "agnesi_medium": "Agnesi-M",
    "mh0": "MH-0",
    "mh1": "MH-1",
    "mp_0b3_medium": "MP-0B3",
    "mpa0_medium": "MPA0",
    "omat0_medium": "OMAT-0-M",
    "omat0_small": "OMAT-0-S",
}

# Muted Morandi-like shades derived from Plot Atlas' registered tissue_pastel
# palette.  Shades are grouped by model family, so model identity remains
# readable without nine unrelated saturated colours.
_BASE = get_palette("tissue_pastel")["categorical"]
_FAMILY_SEEDS = {
    "Agnesi": _BASE[1],
    "MH": _BASE[5],
    "MP": _BASE[3],
    "OMAT": _BASE[0],
}
_FAMILY = {
    "mace-large-density-agnesi-stress": "Agnesi",
    "mace-medium-density-agnesi-stress": "Agnesi",
    "mace-small-density-agnesi-stress": "Agnesi",
    "mace_agnesi_medium": "Agnesi",
    "mace-mh-0": "MH",
    "mace-mh-1": "MH",
    "mace-mp-0b3-medium": "MP",
    "mace-omat-0-medium": "OMAT",
    "mace-omat-0-small": "OMAT",
    # IDs used by the grouped Cr benchmark.
    "agnesi_medium": "Agnesi",
    "agnesi_stress_large": "Agnesi",
    "agnesi_stress_medium": "Agnesi",
    "agnesi_stress_small": "Agnesi",
    "mh0": "MH",
    "mh1": "MH",
    "mp_0b3_medium": "MP",
    "mpa0_medium": "MP",
    "omat0_medium": "OMAT",
    "omat0_small": "OMAT",
}


def _rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def _hex(rgb: np.ndarray) -> str:
    vals = np.clip(np.round(rgb * 255), 0, 255).astype(int)
    return "#" + "".join(f"{x:02X}" for x in vals)


def _shade(seed: str, fraction: float) -> str:
    # Fraction 0 is the family seed; positive values move towards a quiet
    # paper background while retaining enough contrast for thin lines.
    return _hex(_rgb(seed) * (1.0 - fraction) + np.ones(3) * fraction)


def model_colors(model_ids: Iterable[str] = MODEL_ORDER) -> list[str]:
    ids = list(model_ids)
    family_counts = {f: sum(_FAMILY.get(m, "other") == f for m in ids) for f in _FAMILY_SEEDS}
    family_seen = {f: 0 for f in _FAMILY_SEEDS}
    colors = []
    for model in ids:
        family = _FAMILY.get(model, "OMAT")
        i = family_seen[family]
        n = max(family_counts[family], 1)
        fraction = 0.04 + 0.34 * (i / max(n - 1, 1))
        colors.append(_shade(_FAMILY_SEEDS[family], fraction))
        family_seen[family] += 1
    return colors


QE_COLOR = "#303538"
QE_BAND_ALPHA = 0.20
MODEL_BAND_ALPHA = 0.10


def model_label(model_id: str) -> str:
    return MODEL_LABELS.get(model_id, model_id)


_BLOCK_RE = re.compile(r"^\s*-+ iteration\s+(\d+)\s*-+\s*$", re.MULTILINE | re.IGNORECASE)
_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?(?:[EeDd][-+]?\d+)?)\s+([TF])\s*$",
    re.MULTILINE,
)


def _float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _parse_complete_tables(path: Path, n_images: int = 10) -> list[tuple[int, np.ndarray]]:
    text = path.read_text(errors="replace")
    starts = list(_BLOCK_RE.finditer(text))
    tables: list[tuple[int, np.ndarray]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        rows = list(_ROW_RE.finditer(text[start.end():end]))
        if len(rows) != n_images:
            continue
        indices = [int(row.group(1)) for row in rows]
        if indices != list(range(1, n_images + 1)):
            continue
        arr = np.array([[_float(row.group(2)), _float(row.group(3))] for row in rows], dtype=float)
        tables.append((int(start.group(1)), arr))
    return tables


def _neb_out_for_path(remote_root: Path, path_id: str) -> Path | None:
    system, suffix = path_id.split("__", 1)
    neb_dir = remote_root / system / "NEB" / suffix
    for filename in ("neb.out", "neb1.out"):
        candidate = neb_dir / filename
        if candidate.exists():
            return candidate
    return None


def read_qe_path_threshold(remote_root: Path, path_id: str) -> float:
    """Read ``path_thr`` from the path input in eV/A."""
    system, suffix = path_id.split("__", 1)
    source = remote_root / system / "NEB" / suffix / "neb.dat"
    if not source.exists():
        return float("nan")
    text = source.read_text(errors="replace")
    match = re.search(r"\bpath_thr\s*=\s*['\"]?\s*([0-9.+-]+)(?:[dDeE][0-9.+-]+)?", text, re.IGNORECASE)
    if not match:
        return float("nan")
    return _float(match.group(1))


def extract_qe_neb_uncertainty(
    path_ids: Iterable[str],
    *,
    remote_root: Path,
    n_images: int = 10,
    n_tail_tables: int = 5,
) -> pd.DataFrame:
    """Return final QE profiles plus energy-valued iteration bands.

    ``qe_energy_std_eV`` is the across-table standard deviation over the last
    ``n_tail_tables`` complete outer iterations.  ``qe_force_error_eV_A`` is
    the final QE image-table residual and must not be interpreted as an energy
    uncertainty.
    """
    rows: list[dict] = []
    for path_id in path_ids:
        source = _neb_out_for_path(remote_root, path_id)
        if source is None:
            for image_index in range(1, n_images + 1):
                rows.append({"path_id": path_id, "image_index": image_index,
                             "qe_energy_eV": np.nan, "qe_relative_energy_eV": np.nan,
                             "qe_energy_std_eV": np.nan, "qe_energy_range_eV": np.nan,
                             "qe_force_error_eV_A": np.nan, "qe_last_iteration": np.nan,
                             "qe_n_complete_tables": 0, "qe_uncertainty_status": "missing_neb_out"})
            continue
        tables = _parse_complete_tables(source, n_images=n_images)
        if not tables:
            raise ValueError(f"No complete image energy tables parsed from {source}")
        tail = tables[-min(n_tail_tables, len(tables)):]
        tail_relative = np.array([table[:, 0] - table[0, 0] for _, table in tail])
        last_iteration, last = tables[-1]
        final_energy = last[:, 0]
        final_relative = final_energy - final_energy[0]
        std = np.std(tail_relative, axis=0, ddof=0)
        energy_range = np.max(tail_relative, axis=0) - np.min(tail_relative, axis=0)
        force_error = last[:, 1]
        for image_index in range(n_images):
            rows.append({
                "path_id": path_id,
                "image_index": image_index + 1,
                "qe_energy_eV": final_energy[image_index],
                "qe_relative_energy_eV": final_relative[image_index],
                "qe_energy_std_eV": std[image_index],
                "qe_energy_range_eV": energy_range[image_index],
                "qe_force_error_eV_A": force_error[image_index],
                "qe_last_iteration": last_iteration,
                "qe_n_complete_tables": len(tables),
                "qe_uncertainty_status": "tail_iteration_dispersion",
                "qe_out_source": str(source),
            })
    return pd.DataFrame(rows)
