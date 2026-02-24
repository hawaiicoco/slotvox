"""Loopback HTTP client adapter with timeouts and bounded retries.

Only ``http://127.0.0.1`` / ``http://localhost`` base URLs are accepted:
this adapter exists for offline protocol testing, never for reaching the
network. Retries happen only for documented transient conditions —
connection/timeout errors and the retryable statuses (429, 500, 502,
503, 504); any other HTTP status is a hard error. Every exchange is
validated in both directions: the request before sending, the response
against the schema AND the response dataclass, plus a request-id match.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from slotvox.adapters.protocol import InferRequest, InferResponse, response_schema
from slotvox.adapters.validator import validate
from slotvox.errors import AdapterError, ValidationError

RETRYABLE_STATUSES = (429, 500, 502, 503, 504)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
INFER_PATH = "/v1/infer"


class LocalHttpAdapter:
    """Client-side half of the inference protocol over loopback HTTP."""

    kind = "http"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 2.0,
        retries: int = 2,
        retry_delay_s: float = 0.01,
        sleep: Callable[[float], Any] = time.sleep,
    ):
        if not isinstance(base_url, str):
            raise AdapterError(f"base_url must be a string, got {type(base_url).__name__}")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise AdapterError(f"base_url must be a loopback http:// URL, got {base_url!r}")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s <= 0.0
        ):
            raise AdapterError(f"timeout_s must be a positive finite number, got {timeout_s!r}")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 10:
            raise AdapterError(f"retries must be an int within [0, 10], got {retries!r}")
        if (
            isinstance(retry_delay_s, bool)
            or not isinstance(retry_delay_s, (int, float))
            or not math.isfinite(retry_delay_s)
            or retry_delay_s < 0.0
        ):
            raise AdapterError(f"retry_delay_s must be a finite number >= 0, got {retry_delay_s!r}")
        if not callable(sleep):
            raise AdapterError("sleep must be callable")
        self._url = base_url.rstrip("/") + INFER_PATH
        self._timeout_s = float(timeout_s)
        self._retries = retries
        self._retry_delay_s = float(retry_delay_s)
        self._sleep = sleep

    def infer(self, request: InferRequest) -> InferResponse:
        """POST one request; retry transient failures; validate the reply."""
        if not isinstance(request, InferRequest):
            raise AdapterError(f"request must be an InferRequest, got {type(request).__name__}")
        body = json.dumps(request.to_dict(), ensure_ascii=False).encode("utf-8")
        attempts = self._retries + 1
        for attempt in range(1, attempts + 1):
            retryable = attempt < attempts
            try:
                raw = urllib.request.Request(
                    self._url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(raw, timeout=self._timeout_s) as reply:
                    payload_bytes = reply.read(MAX_RESPONSE_BYTES + 1)
            except urllib.error.HTTPError as error:
                detail = error.read(MAX_RESPONSE_BYTES + 1)[:200].decode("utf-8", "replace")
                if error.code in RETRYABLE_STATUSES and retryable:
                    self._sleep(self._retry_delay_s)
                    continue
                raise AdapterError(
                    f"inference service returned HTTP {error.code}: {detail}"
                ) from error
            except (TimeoutError, urllib.error.URLError, ConnectionError) as error:
                if retryable:
                    self._sleep(self._retry_delay_s)
                    continue
                raise AdapterError(
                    f"inference request failed after {attempts} attempts: {error}"
                ) from error
            if len(payload_bytes) > MAX_RESPONSE_BYTES:
                raise AdapterError("inference response exceeds the size bound")
            return self._decode_reply(payload_bytes, request)
        raise AdapterError("inference request failed: retry budget exhausted")

    def _decode_reply(self, payload_bytes: bytes, request: InferRequest) -> InferResponse:
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError as error:
            raise AdapterError(f"inference response is not valid JSON: {error}") from error
        try:
            validate(payload, response_schema())
            response = InferResponse.from_dict(payload)
        except (AdapterError, ValidationError) as error:
            raise AdapterError(f"inference response violates the protocol: {error}") from error
        if response.request_id != request.request_id:
            raise AdapterError(
                f"inference response id {response.request_id!r} does not match "
                f"the request {request.request_id!r}"
            )
        return response
