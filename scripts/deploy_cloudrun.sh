#!/usr/bin/env bash
# Deploy the live demo to Google Cloud Run (builds the Dockerfile on Google's servers).
#
#   GCP_PROJECT=<your-project-id> scripts/deploy_cloudrun.sh
#
# Region defaults to Tokyo (asia-northeast1), next to the Supabase database: the agent saves
# state at almost every step, so app-to-database latency matters most.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:?Set GCP_PROJECT to your Google Cloud project id}"
REGION="${REGION:-asia-northeast1}"
SERVICE="${SERVICE:-react-support-agent}"
ENV_FILE=.cloudrun.env.yaml

trap 'rm -f "$ENV_FILE"' EXIT
uv run --no-project --with python-dotenv python scripts/cloudrun_env.py > "$ENV_FILE"

# Scale to zero when idle (free), one instance at most: sessions are cached in memory and the
# MCP tool server runs inside the same container.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION" \
  --source . \
  --env-vars-file "$ENV_FILE" \
  --memory 1Gi --cpu 1 \
  --min-instances 0 --max-instances 1 \
  --concurrency 20 --timeout 300 \
  --allow-unauthenticated \
  --quiet

echo
echo "Console: $(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')/console"
