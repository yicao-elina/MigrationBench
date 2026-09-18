from __future__ import annotations

import argparse
import filecmp
import shutil
import subprocess
from pathlib import Path

ALLOWLIST = (
    "sn-article-final.tex",
    "sn-article-SI.tex",
    "response_to_reviewers.tex",
    "bibliography.bib",
    "results_generated.tex",
)


def run(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Checked deployment of paper files to Overleaf")
    parser.add_argument("--overleaf", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="copy after all divergence checks pass")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    overleaf = args.overleaf.resolve()
    run("git", "fetch", "--all", cwd=overleaf)
    if run("git", "status", "--porcelain", cwd=overleaf):
        raise SystemExit("Overleaf clone has local changes; import or commit them before deployment")
    branch = run("git", "branch", "--show-current", cwd=overleaf)
    upstream = f"origin/{branch}"
    if run("git", "rev-list", "--count", f"HEAD..{upstream}", cwd=overleaf) != "0":
        raise SystemExit("Overleaf remote is ahead; pull and import its changes first")
    changed = []
    for name in ALLOWLIST:
        source = root / "paper" / name
        target = overleaf / name
        if source.exists() and (not target.exists() or not filecmp.cmp(source, target, shallow=False)):
            changed.append((source, target))
    for source, target in changed:
        print(f"deploy {source.relative_to(root)} -> {target.name}")
        if args.apply:
            shutil.copy2(source, target)
    if changed and not args.apply:
        print("dry run only; repeat with --apply after reviewing this list")


if __name__ == "__main__":
    main()

