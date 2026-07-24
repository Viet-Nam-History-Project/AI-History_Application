#!/usr/bin/env bash
set -euo pipefail

PORT="${AI_PORT:-8000}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
EXPECTED_RAG_REVISION="recoverable-boundary-evolution-planning-f9-v28"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to check the AI backend." >&2
  exit 1
fi

if RESPONSE="$(curl --fail --silent --show-error --max-time 8 "$HEALTH_URL")"; then
  if [[ "$RESPONSE" != *"\"rag_revision\":\"$EXPECTED_RAG_REVISION\""* ]]; then
    echo "AI backend is reachable but is not running the expected RAG revision." >&2
    echo "Expected: $EXPECTED_RAG_REVISION" >&2
    echo "$RESPONSE" >&2
    exit 1
  fi
  echo "AI backend is ready at $HEALTH_URL ($EXPECTED_RAG_REVISION)"
  echo "$RESPONSE"
  exit 0
fi

echo "AI backend is not reachable at $HEALTH_URL" >&2
echo "Start it with: ./start-ai.sh" >&2
exit 1
