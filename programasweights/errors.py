"""Structured API failures that remain compatible with HTTPX error handling."""

from __future__ import annotations

import httpx


class APIError(httpx.HTTPStatusError):
    """An API rejection with optional server code, message, and request ID.

    The original HTTPX ``request`` and ``response`` are retained. Existing
    ``except httpx.HTTPStatusError`` handlers continue to catch this error.
    """

    def __init__(
        self,
        http_message: str,
        *,
        request: httpx.Request,
        response: httpx.Response,
        code: str | None = None,
        message: str | None = None,
        request_id: str | None = None,
    ):
        super().__init__(http_message, request=request, response=response)
        self.code = code
        self.message = message
        self.request_id = request_id


def _text_field(data: dict, name: str) -> str | None:
    value = data.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def raise_for_api_status(response: httpx.Response) -> None:
    """Raise an informative API error without retrying or dumping raw bodies."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as error:
        # Keep HTTPX's behavior for redirects and other non-error statuses.
        if not 400 <= error.response.status_code < 600:
            raise
        try:
            payload = error.response.json()
        except (ValueError, UnicodeError, httpx.ResponseNotRead):
            payload = None
        payload = payload if isinstance(payload, dict) else {}
        detail = payload.get("detail")
        fields = detail if isinstance(detail, dict) else payload
        code = _text_field(fields, "error") or _text_field(payload, "error")
        message = _text_field(fields, "message") or _text_field(payload, "message")
        if message is None and isinstance(detail, str):
            message = detail.strip() or None
        request_id = (
            _text_field(fields, "request_id")
            or _text_field(payload, "request_id")
            or _text_field(dict(error.response.headers), "x-request-id")
        )
        # FastAPI validation lists may contain the original user input. Do
        # not stringify those lists, unknown JSON fields, or HTML responses.
        details = []
        if code:
            details.append(f"code={code}")
        if message:
            details.append(message)
        if request_id:
            details.append(f"request_id={request_id}")
        http_message = str(error)
        if details:
            http_message += "\nPAW API: " + "; ".join(details)
        raise APIError(
            http_message,
            request=error.request,
            response=error.response,
            code=code,
            message=message,
            request_id=request_id,
        ) from error
