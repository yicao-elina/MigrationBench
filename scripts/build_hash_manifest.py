from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "registry" / "artifacts.sha256"
    roots = [root / "data", root / "figures" / "publication", root / "tables" / "generated"]
    files = sorted(path for base in roots for path in base.rglob("*") if path.is_file())
    output.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in files),
        encoding="utf-8",
    )
    print(f"hashed {len(files)} artifacts")


if __name__ == "__main__":
    main()

