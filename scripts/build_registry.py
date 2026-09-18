from pathlib import Path

from migrationbench.provenance.registry import build_registry


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    rows = build_registry(root)
    print(f"registered {len(rows)} compact runs")

