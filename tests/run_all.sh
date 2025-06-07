#!/usr/bin/env bash
set -euo pipefail          # exit immediately on the first failure

DIR="tests/core"           # change this if you want a different root
PY=${PYTHON:-python}       # lets you override with PYTHON=python3 ./run_tests.sh

# Find all *.py files under $DIR (recursively), sort them, and run each one
find "$DIR" -type f -name '*.py' -print0 | sort -z | while IFS= read -r -d '' file; do
  echo "⇒ $file"
  "$PY" "$file"
done
