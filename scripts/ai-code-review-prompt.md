Review the last commit. Be concise. Format final message as markdown.

When you finish, send a Slack notification using:
  .venv/bin/python scripts/slack-notify.py "🚨 Code Review (<commit_hash>): <brief summary of issues>"
  
...if any issues are found. Only describe High and Medium severity issues in detail.

If no High or Medium severity issues found, instead send:
  .venv/bin/python scripts/slack-notify.py "✅ Code Review (<commit_hash>): No major problems found. <describe low severity issues if any>. Check logs for details."

Send exactly 1 Slack message. Always include the commit hash in your Slack message.
