"""Local loopback HTTP mock of the inference protocol (stdlib only).

Serves a :class:`ReplayAdapter` over ``127.0.0.1`` on an ephemeral port.
Test hooks script transient behavior: :meth:`fail_next` queues HTTP
error statuses (for retry tests) and :meth:`delay_next` queues response
delays (for timeout tests). Incoming payloads are validated against the
request schema — a bad request gets a 400 with an error body, never a
decision. The mock validates protocol behavior, not model quality.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from slotvox.adapters.protocol import InferRequest, request_schema
from slotvox.adapters.replay import ReplayAdapter
from slotvox.adapters.validator import validate
from slotvox.errors import AdapterError, ValidationError

MAX_BODY_BYTES = 8 * 1024 * 1024
SCRIPTED_FAILURE_BODY = {"error": "scripted failure"}


class _Handler(BaseHTTPRequestHandler):
    """POST-only handler wired to the owning :class:`MockInferServer`."""

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    def _send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib naming
        self._send_json(405, {"error": "method not allowed"})

    def do_POST(self):  # noqa: N802 - stdlib naming
        mock = self.server.mock
        mock.count_request()
        if self.path != "/v1/infer":
            self._send_json(404, {"error": f"unknown path {self.path}"})
            return
        delay = mock.take_delay()
        if delay > 0.0:
            time.sleep(delay)
        failure = mock.take_failure()
        if failure is not None:
            self._send_json(failure, SCRIPTED_FAILURE_BODY)
            return
        header = self.headers.get("Content-Length")
        try:
            length = int(header)
        except (TypeError, ValueError):
            self._send_json(411, {"error": "Content-Length is required"})
            return
        if length > MAX_BODY_BYTES:
            self._send_json(413, {"error": "request body too large"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return
        try:
            validate(payload, request_schema())
            request = InferRequest.from_dict(payload)
        except (AdapterError, ValidationError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        try:
            response = mock.adapter.infer(request)
        except AdapterError as exc:
            self._send_json(404, {"error": str(exc)})
            return
        self._send_json(200, response.to_dict())


class MockInferServer:
    """Context-managed loopback mock serving one replay adapter."""

    def __init__(self, adapter: ReplayAdapter):
        if not isinstance(adapter, ReplayAdapter):
            raise AdapterError(f"adapter must be a ReplayAdapter, got {type(adapter).__name__}")
        self._adapter = adapter
        self._failures: deque[int] = deque()
        self._delays: deque[float] = deque()
        self._requests_served = 0
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.mock = self
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> MockInferServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> bool:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
        return False

    @property
    def base_url(self) -> str:
        """Loopback base URL (ephemeral port)."""
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def adapter(self) -> ReplayAdapter:
        """The replay adapter serving decisions."""
        return self._adapter

    @property
    def requests_served(self) -> int:
        """POST requests received so far (including scripted failures)."""
        with self._lock:
            return self._requests_served

    def fail_next(self, status: int, times: int = 1) -> None:
        """Queue ``times`` responses with HTTP ``status`` (400-599)."""
        if isinstance(status, bool) or not isinstance(status, int) or not 400 <= status <= 599:
            raise AdapterError(f"status must be an int within [400, 599], got {status!r}")
        self._check_times(times)
        with self._lock:
            self._failures.extend([status] * times)

    def delay_next(self, seconds: float, times: int = 1) -> None:
        """Queue ``times`` response delays of ``seconds`` (timeout tests)."""
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0.0:
            raise AdapterError(f"seconds must be a positive number, got {seconds!r}")
        self._check_times(times)
        with self._lock:
            self._delays.extend([float(seconds)] * times)

    def _check_times(self, times: int) -> None:
        if isinstance(times, bool) or not isinstance(times, int) or not 1 <= times <= 100:
            raise AdapterError(f"times must be an int within [1, 100], got {times!r}")

    def count_request(self) -> None:
        with self._lock:
            self._requests_served += 1

    def take_failure(self) -> int | None:
        with self._lock:
            return self._failures.popleft() if self._failures else None

    def take_delay(self) -> float:
        with self._lock:
            return self._delays.popleft() if self._delays else 0.0
