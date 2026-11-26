# Scratch4 R04 importer scope — 2026-11-26

The R04 importer is restricted to `/scratch4/pclancy3/yi` and its descendants. It reads each local status sidecar only far enough to check its declared root; out-of-scope status rows are skipped before resolving or opening their ledger. A run fails closed if there is no in-scope status/ledger pair. Ledger entries must be normalized absolute paths contained by the declared personal root; `..` traversal is rejected.

This scope guard does not scan Rockfish or open any live Scratch4 directory. It only constrains a future import of already captured, local metadata evidence. Existing broad historical R04 snapshots remain historical artifacts; this importer will not re-ingest their sibling subtrees. The importer has not been run against the production catalog as part of this change.

The central catalog and old audit package may contain prior records from wider historical audits. This update makes no deletion or attribution claim about those snapshots; new scans and new import runs must remain within the personal root above.
