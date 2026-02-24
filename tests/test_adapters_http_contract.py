"""Client-side protocol enforcement against a hostile raw server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from slotvox.adapters.http import LocalHttpAdapter
from slotvox.errors import AdapterError
from tests.test_adapters_helpers import make_request, make_response


class RawHandler(BaseHTTPRequestHandler):
    """Serves one canned raw body regardless of the request."""

    raw_body = b"{}"

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    def do_POST(self):  # noqa: N802 - stdlib naming
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.raw_body)))
        self.end_headers()
        self.wfile.write(self.raw_body)


def serve_raw(raw_body):
    handler = type("Bound", (RawHandler,), {"raw_body": raw_body})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    return httpd, thread, url


def stop(httpd, thread):
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def test_client_rejects_id_mismatch():
    body = json.dumps({**make_response().to_dict(), "request_id": "someone-else"}).encode()
    httpd, thread, url = serve_raw(body)
    try:
        with pytest.raises(AdapterError, match="does not match"):
            LocalHttpAdapter(url, retries=0).infer(make_request())
    finally:
        stop(httpd, thread)


def test_client_rejects_schema_violation():
    body = json.dumps({**make_response().to_dict(), "posterior": 1.7}).encode()
    httpd, thread, url = serve_raw(body)
    try:
        with pytest.raises(AdapterError, match="violates the protocol"):
            LocalHttpAdapter(url, retries=0).infer(make_request())
    finally:
        stop(httpd, thread)


def test_client_rejects_garbage_body():
    httpd, thread, url = serve_raw(b"this is not json at all")
    try:
        with pytest.raises(AdapterError, match="not valid JSON"):
            LocalHttpAdapter(url, retries=0).infer(make_request())
    finally:
        stop(httpd, thread)
