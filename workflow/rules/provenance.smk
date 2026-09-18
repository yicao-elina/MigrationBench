rule run_registry:
    input:
        "data/raw_compact"
    output:
        "registry/runs.csv"
    shell:
        "PYTHONPATH=src python scripts/build_registry.py"

rule artifact_hashes:
    input:
        "registry/runs.csv",
        "paper/results_generated.tex"
    output:
        "registry/artifacts.sha256"
    shell:
        "PYTHONPATH=src python scripts/build_hash_manifest.py"

rule release_audit:
    input:
        "registry/artifacts.sha256",
        "registry/claims.yaml",
        "registry/reviewer_map.yaml"
    output:
        touch("registry/release_audit.ok")
    shell:
        "PYTHONPATH=src python scripts/audit_release.py"
