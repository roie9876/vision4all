#!/bin/bash
# Ensure you have BFG installed (e.g., brew install bfg)
# Remove large files from history
bfg --delete-files "libarrow.1900.dylib"
bfg --delete-files "libtorch_cpu.dylib"

# Remove dangling commits and expire reflog
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Repository history cleaned. Now force push the changes:"
echo "git push --force"
