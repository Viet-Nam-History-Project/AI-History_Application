import hmac
import logging

import firebase_admin
from fastapi import Header, HTTPException, status
from firebase_admin import auth, credentials

from .config import get_settings


logger = logging.getLogger(__name__)


def initialize_firebase() -> None:
    settings = get_settings()
    if firebase_admin._apps:
        return
    credential = (
        credentials.Certificate(settings.firebase_service_account_path)
        if settings.firebase_service_account_path
        else None
    )
    firebase_admin.initialize_app(
        credential=credential,
        options={"projectId": settings.firebase_project_id},
    )


async def require_firebase_user(authorization: str | None = Header(default=None)) -> dict:
    settings = get_settings()
    if settings.ai_allow_unauthenticated:
        return {"uid": "local-development", "email": ""}

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Thiếu Firebase ID token.",
        )

    initialize_firebase()
    try:
        return auth.verify_id_token(
            authorization.removeprefix("Bearer ").strip(),
            clock_skew_seconds=60,
        )
    except Exception as exc:
        logger.exception("Firebase ID token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Firebase ID token không hợp lệ hoặc đã hết hạn.",
        ) from exc


async def require_admin_key(x_admin_key: str | None = Header(default=None)) -> None:
    expected = get_settings().ai_admin_api_key
    if not x_admin_key or not hmac.compare_digest(x_admin_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="AI admin key không hợp lệ.",
        )
