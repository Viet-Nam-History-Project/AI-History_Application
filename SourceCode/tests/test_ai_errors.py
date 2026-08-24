import unittest

import httpx
from openai import APIConnectionError, AuthenticationError, RateLimitError

from src.api.ai_errors import ai_service_http_exception


def _response(status_code: int, payload: dict) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
        json=payload,
    )


class AiErrorMappingTests(unittest.TestCase):
    def test_insufficient_quota_is_reported_as_actionable_429(self) -> None:
        body = {
            "error": {
                "message": "Quota exceeded",
                "type": "insufficient_quota",
                "code": "insufficient_quota",
            }
        }
        error = RateLimitError(
            "Quota exceeded",
            response=_response(429, body),
            body=body,
        )

        mapped = ai_service_http_exception(error, fallback_detail="fallback")

        self.assertEqual(mapped.status_code, 429)
        self.assertIn("hết quota", mapped.detail)
        self.assertIn("Usage/Billing", mapped.detail)

    def test_temporary_rate_limit_has_retry_hint(self) -> None:
        body = {
            "error": {
                "message": "Rate limit reached",
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        }
        error = RateLimitError(
            "Rate limit reached",
            response=_response(429, body),
            body=body,
        )

        mapped = ai_service_http_exception(error, fallback_detail="fallback")

        self.assertEqual(mapped.status_code, 429)
        self.assertEqual(mapped.headers, {"Retry-After": "60"})
        self.assertIn("giới hạn tốc độ", mapped.detail)

    def test_invalid_openai_key_is_not_exposed_as_internal_error(self) -> None:
        body = {
            "error": {
                "message": "Incorrect API key provided: sk-secret",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        }
        error = AuthenticationError(
            "Incorrect API key",
            response=_response(401, body),
            body=body,
        )

        mapped = ai_service_http_exception(error, fallback_detail="fallback")

        self.assertEqual(mapped.status_code, 503)
        self.assertIn("OPENAI_API_KEY", mapped.detail)
        self.assertNotIn("sk-secret", mapped.detail)

    def test_connection_failure_becomes_service_unavailable(self) -> None:
        error = APIConnectionError(
            message="network down",
            request=httpx.Request("POST", "https://api.openai.com/v1/responses"),
        )

        mapped = ai_service_http_exception(error, fallback_detail="fallback")

        self.assertEqual(mapped.status_code, 503)
        self.assertIn("không kết nối được", mapped.detail)

    def test_unknown_failure_keeps_sanitized_fallback(self) -> None:
        mapped = ai_service_http_exception(
            RuntimeError("sensitive internal detail"),
            fallback_detail="Lỗi AI nội bộ.",
        )

        self.assertEqual(mapped.status_code, 500)
        self.assertEqual(mapped.detail, "Lỗi AI nội bộ.")
