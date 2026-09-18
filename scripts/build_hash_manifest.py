from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def canonical_bytes(path: Path, root: Path) -> bytes:
    """Return the exact bytes Git will publish when the worktree is clean.

    Git may normalize text line endings between the filesystem and its index.
    Hash the index blob for clean tracked paths so the manifest is portable
    across macOS/Linux checkouts; hash filesystem bytes for modified/new paths.
    """
    relative = path.relative_to(root).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    modified = subprocess.run(
        ["git", "diff", "--quiet", "--", relative], cwd=root, check=False
    ).returncode != 0
    if tracked and not modified:
        return subprocess.check_output(["git", "show", f":{relative}"], cwd=root)
    return path.read_bytes()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "registry" / "artifacts.sha256"
    roots = [root / "data", root / "figures" / "publication", root / "tables" / "generated"]
    files = sorted(path for base in roots for path in base.rglob("*") if path.is_file())
    output.write_text(
        "".join(
            f"{sha256_bytes(canonical_bytes(path, root))}  {path.relative_to(root)}\n"
            for path in files
        ),
        encoding="utf-8",
    )
    print(f"hashed {len(files)} artifacts")


if __name__ == "__main__":
    main()
