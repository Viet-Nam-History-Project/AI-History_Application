"""Restart the user FastAPI service after a code revision change and an idle window."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "SourceCode"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from src.api.runtime_revision import RAG_REVISION  # noqa: E402


STATE_DIR = SOURCE_ROOT / ".data"
STATE_FILE = STATE_DIR / "auto-restart-pending.json"
LOCK_FILE = STATE_DIR / "auto-restart.lock"
CHECKPOINT_DB = STATE_DIR / "index-checkpoints.sqlite3"
HEALTH_URL = os.getenv(
    "AI_REVISION_HEALTH_URL",
    "http://127.0.0.1:8000/health/revision",
)
IDLE_SECONDS = max(15, int(os.getenv("AI_AUTO_RESTART_IDLE_SECONDS", "45")))
SERVICE_NAME = os.getenv("AI_SYSTEMD_SERVICE", "history-chatbot-ai.service")


def read_health() -> dict:
    try:
        with urlopen(HEALTH_URL, timeout=8) as response:  # noqa: S310 - local URL by design
            return json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return {}


def active_index_jobs() -> int:
    if not CHECKPOINT_DB.exists():
        return 0
    with sqlite3.connect(CHECKPOINT_DB, timeout=10) as connection:
        row = connection.execute(
            """
            SELECT count(*)
            FROM index_sessions
            WHERE status IN ('queued', 'running', 'stopping')
            """
        ).fetchone()
    return int(row[0] if row else 0)


def clear_pending() -> None:
    STATE_FILE.unlink(missing_ok=True)


def mark_pending(now: float) -> None:
    STATE_FILE.write_text(
        json.dumps({"revision": RAG_REVISION, "idleSince": now}),
        encoding="utf-8",
    )


def pending_since() -> float | None:
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if state.get("revision") != RAG_REVISION:
            return None
        return float(state["idleSince"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_FILE.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0

        health = read_health()
        if not health:
            clear_pending()
            return 0
        if health.get("rag_revision") == RAG_REVISION:
            clear_pending()
            return 0

        index_count = max(
            active_index_jobs(),
            int(health.get("active_index_jobs") or 0),
        )
        chat_count = int(health.get("active_chat_requests") or 0)
        if index_count or chat_count:
            clear_pending()
            return 0

        now = time.time()
        idle_since = pending_since()
        if idle_since is None:
            mark_pending(now)
            return 0
        if now - idle_since < IDLE_SECONDS:
            return 0

        # Kiểm tra lần cuối ngay trước restart để không chen vào một action mới.
        health = read_health()
        if (
            not health
            or int(health.get("active_chat_requests") or 0) > 0
            or int(health.get("active_index_jobs") or 0) > 0
            or active_index_jobs() > 0
        ):
            clear_pending()
            return 0

        subprocess.run(
            ["systemctl", "--user", "restart", SERVICE_NAME],
            check=True,
            timeout=120,
        )
        clear_pending()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
