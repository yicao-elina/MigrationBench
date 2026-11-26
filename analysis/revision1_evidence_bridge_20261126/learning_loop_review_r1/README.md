# Failure-mode and learning-loop review index R1

This is a compact SQL index for the seven-row read-only failure-mode audit register. Its grain is **one reviewed evidence family per row**, not individual failures, attempts, or samples. It does not implement an operational failure classifier.

The table retains the audit's observed signal, available query sources, current lineage status, repeatability limits, missing next-round decision/gain evidence, and recommended closure. Every row is labeled `AUDIT_SUMMARY_NOT_FAILURE_EVENT`; every row's benefit status is `NOT_ESTABLISHED_IN_AUDIT`.

The audit identifies repeatable signals worth validating (for example, a common `instability_threshold` outcome in three audited from-scratch seeds and QE force residuals above a stated gate). Neither pattern establishes calibrated prospective detection or improved later model performance. No decision/action table is fabricated. A positive loop claim still requires a verified diagnostic → recorded decision → versioned selected/training data → new checkpoint → comparable held-out evaluation, including a baseline and uncertainty analysis.

Queries `QLR01`–`QLR03` return the review rows, open-loop evidence gates, and source pins. The SQLite snapshot is a separate read-only federated source, not merged into the central catalog.
