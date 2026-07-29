from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

os.environ.setdefault("JEOPARDY_BASE_URL", "https://example.invalid")
os.environ.setdefault("TEAM_API_KEY", "team_test")
os.environ.setdefault("ANTHROPIC_API_KEY", "team_test")

from tools import MAX_OUTPUT_CHARS, TaskTools  # noqa: E402


class CookieHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/login":
            self.send_response(200)
            self.send_header("Set-Cookie", "session=ready; Path=/")
            self.end_headers()
            self.wfile.write(b"logged in")
            return
        cookie = self.headers.get("Cookie", "")
        self.send_response(200 if "session=ready" in cookie else 403)
        self.end_headers()
        self.wfile.write(b"secret-token" if "session=ready" in cookie else b"missing")

    def log_message(self, _format: str, *_args: object) -> None:
        return


class TaskToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tools = TaskTools("T1", self.root, "https://example.invalid")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_file_tools_and_path_confinement(self) -> None:
        written = self.tools.execute(
            "write_file", {"path": "nested/value.txt", "content": "hello"}
        )
        self.assertFalse(written.is_error)
        read = self.tools.execute("read_file", {"path": "nested/value.txt"})
        self.assertEqual(read.content, "hello")
        escaped = self.tools.execute("read_file", {"path": "../secret"})
        self.assertTrue(escaped.is_error)
        self.assertIn("escapes", escaped.content)

    def test_python_capture_answer_and_output_bound(self) -> None:
        result = self.tools.execute(
            "run_python",
            {
                "code": "print('working')\nprint('EXACT-123')",
                "capture_answer": True,
            },
        )
        self.assertFalse(result.is_error)
        payload = json.loads(result.content)
        ref = payload["answer_ref"]
        finalized = self.tools.execute(
            "finalize_answer",
            {
                "answer_ref": ref,
                "confidence": 0.99,
                "evidence": ["computed twice"],
                "verification_notes": "matches constraints",
            },
        )
        self.assertEqual(finalized.candidate.value, "EXACT-123")
        self.assertTrue(finalized.candidate.exact_value_from_tool)

        large = self.tools.execute(
            "run_python", {"code": "print('x' * 40000)"}
        )
        self.assertLessEqual(len(large.content), MAX_OUTPUT_CHARS)
        self.assertIn("full output saved", large.content)

    def test_python_timeout(self) -> None:
        result = self.tools.execute(
            "run_python",
            {"code": "import time; time.sleep(2)", "timeout_seconds": 1},
        )
        self.assertTrue(result.is_error)
        self.assertIn("timed out", result.content)

    def test_http_session_preserves_cookie_and_captures(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), CookieHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        tools = TaskTools("WEB1", self.root, base)
        try:
            first = tools.execute("http_request", {"url": "/login"})
            self.assertFalse(first.is_error)
            second = tools.execute(
                "http_request", {"url": "/protected", "capture_answer": True}
            )
            payload = json.loads(second.content)
            self.assertEqual(payload["status"], 200)
            self.assertIn("answer_ref", payload)
            candidate = tools.execute(
                "finalize_answer",
                {
                    "answer_ref": payload["answer_ref"],
                    "confidence": 0.9,
                    "evidence": "authenticated response",
                    "verification_notes": "HTTP status 200",
                },
            ).candidate
            self.assertEqual(candidate.value, "secret-token")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
