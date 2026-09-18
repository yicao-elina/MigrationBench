rule results_macros:
    input:
        "data/processed/dft_neb_barriers.csv",
        "data/processed/shap_surrogate_r2_revised.csv",
        "data/processed/md_transport_metrics.csv"
    output:
        "paper/results_generated.tex"
    shell:
        "PYTHONPATH=src python scripts/generate_results_tex.py"

