.PHONY: reproduce registry macros hashes audit test paper clean

PYTHON ?= python
SNAKEMAKE ?= snakemake

reproduce:
	$(SNAKEMAKE) --snakefile workflow/Snakefile --cores 1 all

registry:
	PYTHONPATH=src $(PYTHON) scripts/build_registry.py

macros:
	PYTHONPATH=src $(PYTHON) scripts/generate_results_tex.py

hashes:
	PYTHONPATH=src $(PYTHON) scripts/build_hash_manifest.py

audit:
	PYTHONPATH=src $(PYTHON) scripts/audit_release.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest

paper: reproduce
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error sn-article-final.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error sn-article-SI.tex
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error response_to_reviewers.tex

clean:
	cd paper && latexmk -C || true

