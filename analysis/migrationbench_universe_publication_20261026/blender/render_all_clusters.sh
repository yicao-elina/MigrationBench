#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/alina/Documents/ChatGPT/Wiley-MigrationBench/analysis/migrationbench_universe_publication_20261026"
exec python "$ROOT/blender/render_all_clusters.py"
