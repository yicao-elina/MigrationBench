# Revision1 Workflow Dashboard

This folder is an Obsidian-friendly project control plane for the WileyAIS Revision1 work.

## North Star

Make the revised manuscript defensible by turning every important claim into a traceable chain:

`raw artifact -> scripted extraction -> audited dataset/table -> figure/text edit -> reviewer response`

The main scientific pivot is:

> MigrationBench is not just an equilibrium RMSE benchmark. It is a task-relevant, non-equilibrium benchmark that separates fixed-path energy error, self-relaxed path error, path-geometry error, and data/provenance risk.

## Files

- [[project_status_overview]]: current state, gaps, and acceptance gates.
- [[workflow_nodes]]: one node per experiment/edit/response dependency.
- [[reviewer_response_matrix]]: reviewer item by item, with status and downstream manuscript outputs.
- [[manuscript_edit_plan]]: text/figure/table edits and what each one consumes.
- [[experiment_state_machine]]: how to update statuses as new jobs finish.
- `workflow_dag.html`: interactive click-to-expand DAG/status board.

## Status Semantics

- `done`: artifact exists, is reproducible, and passes the relevant acceptance gate.
- `partial`: useful progress exists, but at least one gate is missing.
- `pending`: planned but not yet executed or not yet synced.
- `blocked`: cannot progress without external access, user input, or a failed prerequisite.
- `quarantined`: data exists but must not support manuscript claims until repaired.

## Update Rule

Whenever a job, analysis, figure, or reviewer-response section changes:

1. Update the corresponding node in [[workflow_nodes]].
2. Update downstream rows in [[reviewer_response_matrix]] and [[manuscript_edit_plan]].
3. If a numeric value enters the manuscript, confirm it is derived by script from local synced raw artifacts.
4. Keep placeholders explicit as `[NUMBER-PENDING: ...]` or `[TEXT-PENDING: ...]`.
