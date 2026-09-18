from pathlib import Path

from migrationbench.provenance.registry import build_registry, sha256


def test_sha256_is_stable(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("migrationbench\n", encoding="utf-8")
    assert sha256(artifact) == sha256(artifact)


def test_registry_discovers_compact_run(tmp_path: Path) -> None:
    run = tmp_path / "data" / "raw_compact" / "rockfish" / "run-1"
    run.mkdir(parents=True)
    (run / "runtime_provenance.json").write_text(
        '{"run_id": "run-1", "slurm_job_id": "123"}', encoding="utf-8"
    )
    (tmp_path / "registry").mkdir()
    rows = build_registry(tmp_path)
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["slurm_job_id"] == "123"

