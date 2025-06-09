#!/usr/bin/env bash
set -euo pipefail

DIR="tests/core"                    # root to search
PY=${PYTHON:-python}                # override: PYTHON=python3 ./run_tests.sh
START_INDEX=${1:-0}                 # positional arg, default = 0

# Basic numeric validation (optional but nice)
if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]]; then
  echo "Error: start index must be an integer" >&2
  exit 1
fi

i=0
find "$DIR" -type f -name '*.py' -print0 | sort -z | while IFS= read -r -d '' file; do
  if (( i < START_INDEX )); then
    ((i++))
    continue
  fi
  echo "[$i] ⇒ $file"
  "$PY" "$file"
  ((i++))
done
