Review the last commit. Be concise. Format final message as markdown.

When you finish, send a Slack notification:

1. Write a brief summary to `/tmp/slack_msg.txt`:
   - If High or Medium issues: describe the issues
   - If no major issues: "No major issues." plus any low severity notes

2. Send the notification:
```bash
.venv/bin/python scripts/slack-notify.py --title "Code Review" --status pass --body-file /tmp/slack_msg.txt
```
Use `--status fail` if there are High or Medium issues, `--status pass` otherwise.

Send exactly 1 Slack message. The commit hash and repo name are added automatically.
