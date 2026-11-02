"""Repository-relative and explicitly external paths for figure rebuilders."""

from __future__ import annotations

import os
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[4]
PACKAGE_ROOT = SCRIPT_ROOT.parent
LOCAL_DATA = PACKAGE_ROOT / "data"

_external_root = os.environ.get("MIGRATIONBENCH_EXTERNAL_ROOT")
EXTERNAL_ROOT = Path(_external_root).expanduser() if _external_root else None

_plot_atlas_root = os.environ.get("PLOT_ATLAS_ROOT")
PLOT_ATLAS_ROOT = Path(_plot_atlas_root).expanduser() if _plot_atlas_root else None


def external_path(relative: str) -> Path:
    """Resolve an external source path without embedding a user-local path.

    When no external root is configured, the repository root is used only as a
    deterministic missing-path sentinel.  Callers that need a source file then
    fail closed with the normal missing-input error instead of silently
    substituting a different dataset.
    """

    return (EXTERNAL_ROOT or REPO_ROOT) / relative
