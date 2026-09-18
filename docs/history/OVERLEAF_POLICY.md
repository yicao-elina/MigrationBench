# Overleaf mirror policy

GitHub `main` is the only source of truth. Overleaf is a checked deployment
target for the files required to compile the paper.

1. Generate and audit all paper assets in this repository.
2. Fetch the Overleaf Git remote and stop if its working tree is dirty or its
   remote is ahead.
3. Run `python scripts/sync_overleaf.py --overleaf PATH` to inspect the proposed
   deployment set.
4. After review, repeat with `--apply`, inspect the Overleaf diff, and commit it.
5. If authors edited Overleaf directly, import and commit those changes here
   before any subsequent deployment.

The sync command intentionally has no force or silent-overwrite mode.

