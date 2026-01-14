Review the last commit. Be concise. Format final message as markdown.

When you finish, send a Slack notification. If any High or Medium severity issues are found:

```bash
.venv/bin/python scripts/slack-notify.py --title "Code Review" --status fail --body "<brief summary of issues>"
```

If no High or Medium severity issues found:

```bash
.venv/bin/python scripts/slack-notify.py --title "Code Review" --status pass --body "No major issues. <low severity notes if any>"
```

Send exactly 1 Slack message. The commit hash and repo name are added automatically.
