#!/usr/bin/env bash
# Fail a commit if any weird space characters are present in the staged files.
# These characters are often inserted by ChatGPT and other AI tools and can cause problems.
# Usage: the pre-commit framework supplies "$@" with the staged paths.

set -euo pipefail

# Nothing to check.
[[ $# -eq 0 ]] && exit 0

# ---- Define weird space characters ------------------------------------------
# Common problematic Unicode space characters that should be regular spaces
# Using printf to create the actual bytes instead of Unicode escapes
WEIRD_SPACES=(
  "$(printf '\xC2\xA0')"     # Non-breaking space (U+00A0)
  "$(printf '\xE2\x80\x80')" # En quad (U+2000)
  "$(printf '\xE2\x80\x81')" # Em quad (U+2001)
  "$(printf '\xE2\x80\x82')" # En space (U+2002)
  "$(printf '\xE2\x80\x83')" # Em space (U+2003)
  "$(printf '\xE2\x80\x84')" # Three-per-em space (U+2004)
  "$(printf '\xE2\x80\x85')" # Four-per-em space (U+2005)
  "$(printf '\xE2\x80\x86')" # Six-per-em space (U+2006)
  "$(printf '\xE2\x80\x87')" # Figure space (U+2007)
  "$(printf '\xE2\x80\x88')" # Punctuation space (U+2008)
  "$(printf '\xE2\x80\x89')" # Thin space (U+2009)
  "$(printf '\xE2\x80\x8A')" # Hair space (U+200A)
  "$(printf '\xE2\x80\x8B')" # Zero width space (U+200B)
  "$(printf '\xE2\x80\x8C')" # Zero width non-joiner (U+200C)
  "$(printf '\xE2\x80\x8D')" # Zero width joiner (U+200D)
  "$(printf '\xE2\x80\xA8')" # Line separator (U+2028)
  "$(printf '\xE2\x80\xA9')" # Paragraph separator (U+2029)
  "$(printf '\xE2\x80\xAF')" # Narrow no-break space (U+202F)
  "$(printf '\xE2\x81\x9F')" # Medium mathematical space (U+205F)
  "$(printf '\xE3\x80\x80')" # Ideographic space (U+3000)
)

# Unicode character names for better error messages
SPACE_NAMES=(
  "Non-breaking space (U+00A0)"
  "En quad (U+2000)"
  "Em quad (U+2001)"
  "En space (U+2002)"
  "Em space (U+2003)"
  "Three-per-em space (U+2004)"
  "Four-per-em space (U+2005)"
  "Six-per-em space (U+2006)"
  "Figure space (U+2007)"
  "Punctuation space (U+2008)"
  "Thin space (U+2009)"
  "Hair space (U+200A)"
  "Zero width space (U+200B)"
  "Zero width non-joiner (U+200C)"
  "Zero width joiner (U+200D)"
  "Line separator (U+2028)"
  "Paragraph separator (U+2029)"
  "Narrow no-break space (U+202F)"
  "Medium mathematical space (U+205F)"
  "Ideographic space (U+3000)"
)
# -----------------------------------------------------------------------------

status=0

for file in "$@"; do
  # Skip deleted paths and binary files.
  [[ -f "$file" ]] || continue
  grep -Iq . "$file" || continue
  
  # Skip this script itself to avoid false positives
  [[ "$(basename "$file")" == "banned-weird-spaces.sh" ]] && continue

  # Check each weird space character
  for i in "${!WEIRD_SPACES[@]}"; do
    char="${WEIRD_SPACES[$i]}"
    name="${SPACE_NAMES[$i]}"
    
    if grep -nH -F -- "$char" "$file" >/dev/null; then
      printf '\n\033[31m✖ Weird space character found in %s:\033[0m\n' "$file"
      printf '\033[31m  %s\033[0m\n' "$name"
      printf '\033[33m  Replace with regular space characters or remove if unnecessary.\033[0m\n'
      # Show offending lines with line numbers
      grep -nH -F --color=always "$char" "$file" | head -5
      if [ "$(grep -c -F "$char" "$file")" -gt 5 ]; then
        printf '\033[33m  ... and %d more occurrences\033[0m\n' "$(($(grep -c -F "$char" "$file") - 5))"
      fi
      status=1
    fi
  done
done

if [ "$status" -eq 1 ]; then
  printf '\n\033[33mTo fix these issues:\033[0m\n'
  printf '\033[33m1. Open the affected files in your editor\033[0m\n'
  printf '\033[33m2. Replace the weird space characters with regular spaces\033[0m\n'
  printf '\033[33m3. Or remove them if they are not needed\033[0m\n'
  printf '\033[33m4. Stage your changes and commit again\033[0m\n\n'
fi

exit "$status"