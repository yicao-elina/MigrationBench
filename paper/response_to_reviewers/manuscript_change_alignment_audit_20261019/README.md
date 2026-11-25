# Manuscript-change alignment audit

This additive audit checks that the wording highlighted in the response-letter
blue callouts is also present in the cited main-text or SI source. It is a
source-level alignment check only; PDF rendering, figure placement, and
scientific gates are audited separately.

Run from the repository root with:

```text
python3 paper/response_to_reviewers/manuscript_change_alignment_audit_20261019/check_alignment.py
```

The checked anchors cover A1 provenance correction, A2 grouped split, A3
three-initialization kinetic uncertainty and hard-OOD boundary, A4
candidate-framework scope and transfer-validation boundary, A5 low-data
control scope, A6 projection/SHAP scope, A7 related-work scope, and the
zero-force, matched-MD, grouped-split, descriptor, inactive-feature, and SHAP
code/provenance corrections. The historical figure-page PDF remains frozen;
this audit does not modify or regenerate it.

The expanded source audit currently passes 27 response-to-main/SI anchor
checks. It verifies both that the blue-callout wording is present in the
response letter and that the corresponding target paragraph exists in the
declared manuscript or SI source.

The companion `check_pdf_pages.py` audit was run against the dated compiled
response, main manuscript, and SI PDFs. It passed 42 page-level checks,
including the response blue-callout pages and the corresponding manuscript/SI
pages. The expected page numbers are intentionally explicit so pagination
drift causes a visible audit failure.
