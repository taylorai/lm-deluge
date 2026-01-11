#!/bin/bash

# AI Code Review - runs after each commit
# Output stored in ai-reviews/ folder (committed to VCS)

# Load .env if it exists (exports SLACK_WEBHOOK for codex agent)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Bail early if SLACK_WEBHOOK is not set (prevents running on random checkouts)
if [ -z "$SLACK_WEBHOOK" ]; then
    echo "Skipping AI code review: SLACK_WEBHOOK not set"
    exit 0
fi

REVIEW_DIR="ai-reviews"
COMMIT_HASH=$(git rev-parse --short HEAD)
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$REVIEW_DIR/${TIMESTAMP}-${COMMIT_HASH}.md"

mkdir -p "$REVIEW_DIR"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROMPT_FILE="$SCRIPT_DIR/ai-code-review-prompt.md"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Prompt file not found: $PROMPT_FILE"
    exit 1
fi

PROMPT=$(cat "$PROMPT_FILE")

# Run in background, detached from terminal
nohup codex exec --sandbox workspace-write -c sandbox_workspace_write.network_access=true "$PROMPT" > "$LOG_FILE" 2>&1 &

echo "🤖 AI review started → $LOG_FILE"
