#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_UVICORN="$PROJECT_DIR/.venv/bin/uvicorn"
ENV_FILE="$PROJECT_DIR/SourceCode/.env"
PORT="${AI_PORT:-8000}"
HOST="${AI_HOST:-0.0.0.0}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
REVISION_FILE="$PROJECT_DIR/SourceCode/src/api/rag_revision.txt"
EXPECTED_RAG_REVISION="$(tr -d '\r\n' < "$REVISION_FILE")"

if [[ ! -x "$VENV_UVICORN" ]]; then
  echo "Missing .venv. Create it and install SourceCode/requirements.txt first." >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing SourceCode/.env. Create it from SourceCode/.env.example first." >&2
  exit 1
fi

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "AI_PORT must be a valid TCP port. Received: $PORT" >&2
  exit 1
fi

if command -v curl >/dev/null 2>&1; then
  CURRENT_HEALTH="$(curl --fail --silent --max-time 2 "$HEALTH_URL" 2>/dev/null || true)"
  if [[ -n "$CURRENT_HEALTH" ]]; then
    if [[ "$CURRENT_HEALTH" == *"\"rag_revision\":\"$EXPECTED_RAG_REVISION\""* ]]; then
      echo "AI backend is already ready at $HEALTH_URL"
      echo "$CURRENT_HEALTH"
      exit 0
    fi
    echo "An older AI backend is still listening at $HEALTH_URL." >&2
    echo "Expected RAG revision: $EXPECTED_RAG_REVISION" >&2
    echo "Finish or pause any active Index job, then restart it with:" >&2
    echo "  systemctl --user restart history-chatbot-ai.service" >&2
    exit 1
  fi
fi

echo "Starting Vietnam History AI backend..."
echo "  Listen address:  http://${HOST}:${PORT}"
echo "  Web Admin:       http://127.0.0.1:${PORT}"
echo "  Android Emulator: http://10.0.2.2:${PORT}"
echo "  Health check:    $HEALTH_URL"
echo "Keep this terminal open. Press Ctrl+C to stop FastAPI."

cd "$PROJECT_DIR/SourceCode"
exec "$VENV_UVICORN" src.api.main:app --host "$HOST" --port "$PORT"
