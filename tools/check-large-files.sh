#!/bin/bash
# Check for files larger than 50MB
large_files=$(find . -type f -size +50M -not -path "./.git/*" -not -path "./.*")
if [ -n "$large_files" ]; then
    echo "❌ Large files detected (>50MB):"
    echo "$large_files"
    echo "Please add them to .gitignore or use Git LFS"
    exit 1
fi
echo "✅ No large files detected"
