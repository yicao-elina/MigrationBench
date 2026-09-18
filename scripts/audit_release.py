from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import yaml

MAX_GIT_FILE = 50 * 1024 * 1024
PROHIBITED = (
    re.compile(r"(^|/)wfc[^/]*\.dat$", re.I),
    re.compile(r"(^|/)charge-density\.dat$", re.I),
    re.compile(r"(^|/).*\.save/"),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def index_digest(root: Path, relative: str) -> str | None:
    try:
        payload = subprocess.check_output(
            ["git", "show", f":{relative}"], cwd=root, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors: list[str] = []
    excluded_roots = {".git", ".venv", "work", ".snakemake"}
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not excluded_roots.intersection(path.relative_to(root).parts)
    ]
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.stat().st_size > MAX_GIT_FILE:
            errors.append(f"oversized file: {relative}")
        if any(pattern.search(relative) for pattern in PROHIBITED):
            errors.append(f"prohibited computational state: {relative}")

    claims_path = root / "registry" / "claims.yaml"
    claims = yaml.safe_load(claims_path.read_text(encoding="utf-8"))["claims"]
    for claim in claims:
        source = root / claim["source"]
        if not source.is_file():
            errors.append(f"missing source for {claim['claim_id']}: {claim['source']}")
        for target in claim["manuscript_targets"]:
            if not (root / target).is_file():
                errors.append(f"missing manuscript target for {claim['claim_id']}: {target}")

    absolute_path = re.compile(r"/(Users|home|scratch|data)/[^\s{}]+")
    for path in (root / "paper").rglob("*.tex"):
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if absolute_path.search(line):
                errors.append(f"absolute path in {path.relative_to(root)}:{number}")

    personal_path = re.compile(r"/Users/[^/]+/")
    for base in (root / "src", root / "scripts", root / "configs", root / "workflow"):
        for path in base.rglob("*"):
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.name == "audit_release.py"
                or path.suffix.lower() in {".pdf", ".png", ".pyc"}
            ):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if personal_path.search(text):
                errors.append(f"personal absolute path in executable source: {path.relative_to(root)}")

    manifest = root / "registry" / "artifacts.sha256"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split("  ", 1)
            path = root / relative
            actual = digest(path) if path.is_file() else None
            if actual != expected and index_digest(root, relative) != expected:
                errors.append(f"hash mismatch: {relative}")
    else:
        errors.append("missing registry/artifacts.sha256")

    placeholders = []
    for path in (root / "paper").rglob("*.tex"):
        if path.name == "results_generated.tex":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if "PLACEHOLDER" in line.upper():
                placeholders.append(f"{path.relative_to(root)}:{number}")
    if placeholders and not (root / "registry" / "pending_values.csv").is_file():
        errors.append("manuscript placeholders exist but registry/pending_values.csv is missing")

    if errors:
        print("release audit failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"release audit passed: {len(files)} files, {len(claims)} registered claims")
    return 0


if __name__ == "__main__":
    sys.exit(main())
