#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_UNIT="$PROJECT_DIR/deploy/history-chatbot-ai.service"
SOURCE_UPDATE_UNIT="$PROJECT_DIR/deploy/history-chatbot-ai-update.service"
SOURCE_UPDATE_TIMER="$PROJECT_DIR/deploy/history-chatbot-ai-update.timer"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
TARGET_UNIT="$USER_UNIT_DIR/history-chatbot-ai.service"
TARGET_UPDATE_UNIT="$USER_UNIT_DIR/history-chatbot-ai-update.service"
TARGET_UPDATE_TIMER="$USER_UNIT_DIR/history-chatbot-ai-update.timer"

if [[ ! -x "$PROJECT_DIR/.venv/bin/uvicorn" ]]; then
  echo "Missing .venv/bin/uvicorn. Install SourceCode/requirements.txt first." >&2
  exit 1
fi

if [[ ! -f "$PROJECT_DIR/SourceCode/.env" ]]; then
  echo "Missing SourceCode/.env." >&2
  exit 1
fi

chmod +x "$PROJECT_DIR/auto-restart-ai.sh"

mkdir -p "$USER_UNIT_DIR"
install -m 0644 "$SOURCE_UNIT" "$TARGET_UNIT"
install -m 0644 "$SOURCE_UPDATE_UNIT" "$TARGET_UPDATE_UNIT"
install -m 0644 "$SOURCE_UPDATE_TIMER" "$TARGET_UPDATE_TIMER"
systemctl --user daemon-reload
systemctl --user enable --now history-chatbot-ai.service
systemctl --user enable --now history-chatbot-ai-update.timer

echo
echo "AI backend service has been installed and enabled."
echo "Status:  systemctl --user status history-chatbot-ai.service"
echo "Logs:    journalctl --user -u history-chatbot-ai.service -f"
echo "Restart: systemctl --user restart history-chatbot-ai.service"
echo "Stop:    systemctl --user stop history-chatbot-ai.service"
echo "Auto update timer: systemctl --user status history-chatbot-ai-update.timer"
