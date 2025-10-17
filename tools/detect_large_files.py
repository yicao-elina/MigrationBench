#!/usr/bin/env python3
import os

# Threshold in MB
THRESHOLD_MB = 20
gitignore_path = ".gitignore"
large_files = []

for root, _, files in os.walk("."):
    if root.startswith("./.git"):
        continue
    for f in files:
        path = os.path.join(root, f)
        try:
            size = os.path.getsize(path) / (1024 * 1024)
            if size > THRESHOLD_MB:
                large_files.append(path)
        except FileNotFoundError:
            pass

if large_files:
    print(f"[INFO] Detected {len(large_files)} large files (> {THRESHOLD_MB} MB):")
    for path in large_files:
        print(" -", path)
    with open(gitignore_path, "a") as f:
        for path in large_files:
            f.write(f"\n# Auto-ignored large file\n{path}\n")
    print(f"[INFO] Added these paths to {gitignore_path}")
else:
    print("[INFO] No large files found.")
