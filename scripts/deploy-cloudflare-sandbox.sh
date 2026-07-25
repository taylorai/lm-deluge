#!/usr/bin/env bash
#
# Deploy the Cloudflare Sandbox Worker for lm-deluge.
#
# Prerequisites:
#   1. Node.js >= 22
#   2. pnpm (https://pnpm.io/installation)
#   3. A Cloudflare account with Containers beta enabled
#      (request access at https://developers.cloudflare.com/containers/)
#   4. `pnpm exec wrangler login` (run once to authenticate, after installing deps)
#
# Usage:
#   ./scripts/deploy-cloudflare-sandbox.sh
#
# After deploying, you'll get a Worker URL like:
#   https://lm-deluge-sandbox.<your-subdomain>.workers.dev
#
# Then set a secret API key:
#   cd scripts/cloudflare-sandbox-worker && pnpm exec wrangler secret put SANDBOX_API_KEY
#   (paste a strong random key when prompted)
#
# Finally, use it in Python:
#   from lm_deluge.tool.prefab.sandbox import CloudflareSandbox
#   async with CloudflareSandbox(
#       worker_url="https://lm-deluge-sandbox.<your-subdomain>.workers.dev",
#       api_key="<your-key>",
#   ) as sandbox:
#       tools = sandbox.get_tools()
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKER_DIR="$SCRIPT_DIR/cloudflare-sandbox-worker"

cd "$WORKER_DIR"

echo "Installing dependencies..."
pnpm install --frozen-lockfile

echo ""
echo "Deploying worker..."
pnpm exec wrangler deploy

echo ""
echo "====================================="
echo "Worker deployed!"
echo ""
echo "Next steps:"
echo "  1. Set your API key secret:"
echo "     cd $WORKER_DIR && pnpm exec wrangler secret put SANDBOX_API_KEY"
echo ""
echo "  2. Use in Python:"
echo "     from lm_deluge.tool.prefab.sandbox import CloudflareSandbox"
echo "     sandbox = CloudflareSandbox(worker_url='<url above>', api_key='<key>')"
echo "====================================="
