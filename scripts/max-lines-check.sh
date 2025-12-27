#!/usr/bin/env bash
# Fail if any listed file has more than MAX_LINES lines.

MAX_LINES=2500
status=0
RED='\033[0;31m'
NC='\033[0m'   # No Color

for file in "$@"; do
  [[ -f "$file" ]] || continue

  lines=$(wc -l < "$file" | tr -d ' ')
  if [[ "$lines" -gt "$MAX_LINES" ]]; then
    echo -e "${RED}ERROR:${NC} $file is $lines lines (limit is $MAX_LINES)."
    status=1
  fi
done

exit $status
