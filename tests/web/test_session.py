from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import unittest

from tools.web.errors import (
    AuthenticationRejectedError,
    BlockedURLError,
    NonHTMLError,
    RedirectLoopError,
    RequestTimeoutError,
    ResponseTooLargeError,
    SessionExpiredError,
)
from tools.web.session import WebClient, WebSessionManager

from .fixture_server import fixture_server


class WebClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server_context = fixture_server()
        self.origin = self.server_context.__enter__()
        self.manager = WebSessionManager()

    def tearDown(self) -> None:
        self.manager.close()
        self.server_context.__exit__(None, None, None)

    def client(self, task_id: str = "PR-W1", **kwargs) -> WebClient:
        return WebClient(task_id, self.origin, self.manager, **kwargs)

    def test_cookie_persists_and_sessions_are_stable(self) -> None:
        client = self.client()
        session = self.manager.for_task("PR-W1", self.origin)
        self.assertIs(session, self.manager.for_task("PR-W1", self.origin))
        client.request("GET", "/cookie/set", params={"value": "alpha"})
        checked = client.request("GET", "/cookie/check")
        self.assertIn("tile=alpha", checked.text)

    def test_concurrent_tiles_do_not_share_cookies(self) -> None:
        first = self.client("tile-a")
        second = self.client("tile-b")

        def round_trip(client: WebClient, value: str) -> str:
            client.request("GET", "/cookie/set", params={"value": value})
            return client.request("GET", "/cookie/check").text

        with ThreadPoolExecutor(max_workers=2) as pool:
            one = pool.submit(round_trip, first, "one")
            two = pool.submit(round_trip, second, "two")
        self.assertIn("tile=one", one.result())
        self.assertNotIn("tile=two", one.result())
        self.assertIn("tile=two", two.result())
        self.assertNotIn("tile=one", two.result())

    def test_form_preserves_hidden_token_and_resolves_label(self) -> None:
        client = self.client()
        page = client.request("GET", "/")
        self.assertNotIn("csrf-super-secret", repr(page.page.semantic))
        result = client.submit_form(
            "login", {"User name": "jun", "password": "round1"}
        )
        self.assertIn("Welcome Jun", result.text)
        self.assertEqual(client._session.cookies.get("authenticated"), "yes")

    def test_get_form_resolves_label(self) -> None:
        client = self.client()
        client.request("GET", "/")
        result = client.submit_form("search", {"Search term": "dark web"})
        self.assertIn("search:dark web", result.text)

    def test_wrong_credentials_are_typed(self) -> None:
        client = self.client()
        client.request("GET", "/")
        with self.assertRaises(AuthenticationRejectedError):
            client.submit_form("login", {"username": "jun", "password": "wrong"})

    def test_existing_cookie_makes_unauthorized_session_expired(self) -> None:
        client = self.client()
        client.request("GET", "/cookie/set", params={"value": "x"})
        with self.assertRaises(SessionExpiredError):
            client.request("GET", "/expired")

    def test_redirect_loop_and_off_origin_redirect_are_blocked(self) -> None:
        with self.assertRaises(RedirectLoopError):
            self.client(max_redirects=2).request("GET", "/redirect/a")
        with self.assertRaises(BlockedURLError):
            self.client("other").request("GET", "/redirect/offsite")

    def test_direct_off_origin_and_sensitive_headers_are_blocked(self) -> None:
        client = self.client()
        with self.assertRaises(BlockedURLError):
            client.request("GET", "https://example.com/")
        with self.assertRaises(BlockedURLError):
            client.request("GET", "/", headers={"Authorization": "Bearer secret"})

    def test_non_html_oversized_and_timeout_are_typed(self) -> None:
        with self.assertRaises(NonHTMLError):
            self.client().request("GET", "/binary")
        with self.assertRaises(ResponseTooLargeError):
            self.client("large", max_response_bytes=100).request("GET", "/large")
        with self.assertRaises(RequestTimeoutError):
            self.client("slow", timeout_seconds=0.05).request("GET", "/slow")

    def test_non_html_can_be_returned_when_requested(self) -> None:
        response = self.client().request("GET", "/binary", expect_html=False)
        self.assertEqual(response.body, b"\x00\x01")

    def test_malformed_page_does_not_crash(self) -> None:
        response = self.client().request("GET", "/malformed")
        self.assertIsNotNone(response.page)
        self.assertIsInstance(response.page.semantic["text"], str)

    def test_json_post(self) -> None:
        response = self.client().request(
            "POST", "/json", json_body={"value": "accepted"}
        )
        self.assertIn("accepted", response.text)


if __name__ == "__main__":
    unittest.main()
