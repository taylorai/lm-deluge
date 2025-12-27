#!/usr/bin/env bash
# Fail a commit if any forbidden string is present in the staged files.
# Usage: the pre-commit framework supplies "$@" with the staged paths.

set -euo pipefail

# Nothing to check.
[[ $# -eq 0 ]] && exit 0

# ---- customise this list ----------------------------------------------------
# One entry per literal string (use extended-regex -E if you need patterns).
FORBIDDEN_STRINGS=(
  "from backend.app"
  " List["
  " Tuple"
  " Union"
  "use_cursor"
  # " print("
)
# -----------------------------------------------------------------------------

status=0

for file in "$@"; do
  # Skip deleted paths and binary files.
  [[ -f "$file" ]] || continue
  grep -Iq . "$file" || continue

  for pattern in "${FORBIDDEN_STRINGS[@]}"; do
    if grep -nH -F -- "$pattern" "$file" >/dev/null; then
      printf '\n\033[31m✖ Forbidden string "%s" found in %s:\033[0m\n' "$pattern" "$file"
      # Show offending lines with line numbers and highlight.
      grep -nH -F --color=always "$pattern" "$file"
      status=1
    fi
  done
done

exit "$status"
