"""Local Dark Web fixture with cookies, forms, redirects, and failures."""

from __future__ import annotations

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
from urllib.parse import parse_qs, urlsplit


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(
        self,
        status: int,
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _form(self) -> bytes:
        return b"""
        <html><head><title>Sign in</title></head><body>
          <a href="../help">Help</a>
          <form id="login" method="post" action="/login">
            <input type="hidden" name="csrf_token" value="csrf-super-secret">
            <label for="user">User name</label>
            <input id="user" name="username">
            <label>Password <input type="password" name="password"></label>
            <button name="commit" value="yes" type="submit">Sign in</button>
          </form>
          <form id="search" method="get" action="/search">
            <label for="term">Search term</label>
            <input id="term" name="term">
          </form>
          <table><tr><th>Plan</th><th>Points</th></tr>
            <tr><td>Round 1</td><td>100</td></tr></table>
          <div class="invalid-feedback">Use assigned credentials</div>
        </body></html>
        """

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            self._send(200, self._form())
        elif parsed.path == "/cookie/set":
            value = query.get("value", ["missing"])[0]
            self._send(
                200,
                b"<html>cookie set</html>",
                headers={"Set-Cookie": f"tile={value}; Path=/; HttpOnly"},
            )
        elif parsed.path == "/cookie/check":
            cookie = self.headers.get("Cookie", "")
            self._send(200, f"<html>{cookie}</html>".encode())
        elif parsed.path == "/expired":
            self._send(401, b"<html>expired</html>")
        elif parsed.path == "/redirect/a":
            self._send(302, b"", headers={"Location": "/redirect/b"})
        elif parsed.path == "/redirect/b":
            self._send(302, b"", headers={"Location": "/redirect/a"})
        elif parsed.path == "/redirect/offsite":
            self._send(302, b"", headers={"Location": "https://example.com/nope"})
        elif parsed.path == "/binary":
            self._send(200, b"\x00\x01", "application/octet-stream")
        elif parsed.path == "/large":
            self._send(200, b"x" * 4_096)
        elif parsed.path == "/slow":
            time.sleep(0.25)
            self._send(200, b"<html>late</html>")
        elif parsed.path == "/search":
            term = query.get("term", [""])[0]
            self._send(200, f"<html>search:{term}</html>".encode())
        elif parsed.path == "/malformed":
            self._send(200, b"<html><title>Broken<body><form><input name='x'>")
        else:
            self._send(404, b"<html>not found</html>")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        if self.path == "/login":
            fields = parse_qs(body.decode())
            valid = (
                fields.get("username") == ["jun"]
                and fields.get("password") == ["round1"]
                and fields.get("csrf_token") == ["csrf-super-secret"]
            )
            if not valid:
                self._send(401, b"<html>login rejected</html>")
                return
            self._send(
                200,
                b"<html><div role='alert'>Welcome Jun</div></html>",
                headers={"Set-Cookie": "authenticated=yes; Path=/; HttpOnly"},
            )
        elif self.path == "/json":
            payload = json.loads(body or b"{}")
            self._send(200, f"<html>{payload.get('value', '')}</html>".encode())
        else:
            self._send(404, b"<html>not found</html>")


@contextmanager
def fixture_server():
    class QuietThreadingHTTPServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address) -> None:
            pass

    server = QuietThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    import threading

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
