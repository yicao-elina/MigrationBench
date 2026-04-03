#!/usr/bin/env bash
set -euo pipefail

# collect_sb2te3.sh
# Usage: ./collect_sb2te3.sh /path/to/source /path/to/destination
# Example: ./collect_sb2te3.sh . /tmp/collected_files
#
# Behavior:
# - Recursively visits all directories under SOURCE (including SOURCE itself).
# - Recreates the same directory structure under DEST.
# - Copies sb2te3.dat, sb2te3.xyz and neb.out if they exist in each directory.
# - Preserves file timestamps and permissions (cp -p).
# - Works with spaces and special characters in filenames.

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 SOURCE_DIR DEST_DIR"
  exit 2
fi

SOURCE="$1"
DEST="$2"

# Remove trailing slashes for consistent behavior
SOURCE="${SOURCE%/}"
DEST="${DEST%/}"

# Verify source exists and is a directory
if [ ! -d "$SOURCE" ]; then
  echo "Error: SOURCE directory '$SOURCE' does not exist or is not a directory."
  exit 3
fi

# Create destination root if not exists
mkdir -p "$DEST"

# Files to extract (edit here if you want more)
files_to_copy=( "sb2te3.dat" "sb2te3.xyz" "neb.out" )

# Use find to enumerate directories. -print0 + read -d '' handles weird names.
find "$SOURCE" -type d -print0 |
while IFS= read -r -d '' dir; do
  # compute relative path of this directory to SOURCE
  if [ "$dir" = "$SOURCE" ]; then
    rel="."        # keep root mapped to root under DEST
  else
    rel="${dir#"$SOURCE"/}"
  fi

  dest_dir="$DEST/$rel"
  mkdir -p "$dest_dir"

  # Copy each file if it exists
  for f in "${files_to_copy[@]}"; do
    src_file="$dir/$f"
    if [ -f "$src_file" ]; then
      # copy with -p to preserve timestamps and permissions; -n to not overwrite
      cp -p -- "$src_file" "$dest_dir/" 2>/dev/null || cp -p "$src_file" "$dest_dir/" 
      echo "copied: $src_file -> $dest_dir/"
    fi
  done
done

echo "Done. Extracted files (if present) into: $DEST"
