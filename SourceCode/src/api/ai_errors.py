from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)


def _openai_error_code(error: Exception) -> str:
    code = getattr(error, "code", None)
    if code:
        return str(code).strip().lower()

    body: Any = getattr(error, "body", None)
    if isinstance(body, dict):
        nested = body.get("error")
        if isinstance(nested, dict) and nested.get("code"):
            return str(nested["code"]).strip().lower()
        if body.get("code"):
            return str(body["code"]).strip().lower()
    return ""


def ai_service_http_exception(
    error: Exception,
    *,
    fallback_detail: str,
) -> HTTPException:
    """Convert upstream OpenAI failures into stable, user-facing API errors.

    FastAPI otherwise turns an uncaught SDK exception into an opaque HTTP 500.
    The Web Admin and mobile app need a useful message, while credentials and
    raw provider payloads must stay in server logs.
    """

    if isinstance(error, RateLimitError):
        if _openai_error_code(error) == "insufficient_quota":
            return HTTPException(
                status_code=429,
                detail=(
                    "OpenAI API đã hết quota hoặc project của API key chưa có "
                    "ngân sách khả dụng. Hãy kiểm tra Usage/Billing và API key "
                    "của OpenAI rồi thử lại."
                ),
            )
        return HTTPException(
            status_code=429,
            detail=(
                "OpenAI đang giới hạn tốc độ yêu cầu. Hãy chờ một lúc rồi thử lại."
            ),
            headers={"Retry-After": "60"},
        )

    if isinstance(error, AuthenticationError):
        return HTTPException(
            status_code=503,
            detail=(
                "OpenAI từ chối API key hiện tại. Hãy kiểm tra OPENAI_API_KEY "
                "trong cấu hình FastAPI."
            ),
        )

    if isinstance(error, PermissionDeniedError):
        return HTTPException(
            status_code=503,
            detail=(
                "API key OpenAI không có quyền dùng model đang cấu hình. "
                "Hãy kiểm tra project, model và quyền của API key."
            ),
        )

    if isinstance(error, APITimeoutError):
        return HTTPException(
            status_code=504,
            detail="OpenAI phản hồi quá chậm. Hãy thử lại sau.",
        )

    if isinstance(error, APIConnectionError):
        return HTTPException(
            status_code=503,
            detail=(
                "FastAPI không kết nối được tới OpenAI. Hãy kiểm tra mạng và "
                "trạng thái dịch vụ rồi thử lại."
            ),
        )

    return HTTPException(status_code=500, detail=fallback_detail)
