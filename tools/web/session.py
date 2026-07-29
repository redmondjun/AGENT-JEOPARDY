"""Bounded stateful HTTP clients, isolated by task."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit

import requests

from .errors import (
    AmbiguousControlError,
    AuthenticationRejectedError,
    BlockedURLError,
    HTMLParseError,
    InvalidArgumentError,
    MissingControlError,
    NonHTMLError,
    RedirectLoopError,
    RequestTimeoutError,
    ResponseTooLargeError,
    SessionExpiredError,
    WebError,
)
from .html import ParsedForm, ParsedPage, parse_html

_REDIRECTS = {301, 302, 303, 307, 308}
_BLOCKED_HEADERS = {
    "authorization",
    "cookie",
    "proxy-authorization",
    "x-api-key",
    "api-key",
}


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise BlockedURLError("URL must use http or https and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise BlockedURLError("credentials in URLs are not allowed")
    port = parsed.port
    effective_port = port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme.lower(), parsed.hostname.lower(), effective_port


@dataclass(frozen=True)
class WebResponse:
    status_code: int
    url: str
    headers: Mapping[str, str]
    body: bytes
    page: ParsedPage | None = None

    @property
    def text(self) -> str:
        encoding = "utf-8"
        content_type = self.headers.get("content-type", "")
        if "charset=" in content_type:
            encoding = content_type.rsplit("charset=", 1)[-1].split(";", 1)[0].strip()
        try:
            return self.body.decode(encoding, errors="replace")
        except LookupError:
            return self.body.decode("utf-8", errors="replace")


class WebSessionManager:
    """Own one requests.Session per tile and never share cookie jars."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, tuple[tuple[str, str, int | None], requests.Session]] = {}

    def for_task(self, task_id: str, allowed_origin: str) -> requests.Session:
        if not task_id:
            raise InvalidArgumentError("task_id is required")
        normalized = _origin(allowed_origin)
        with self._lock:
            current = self._sessions.get(task_id)
            if current is not None:
                if current[0] != normalized:
                    raise BlockedURLError(
                        "a task session cannot change its allowed origin"
                    )
                return current[1]
            session = requests.Session()
            session.headers["User-Agent"] = "agent-jeopardy-web/1"
            self._sessions[task_id] = (normalized, session)
            return session

    def reset(self, task_id: str) -> None:
        with self._lock:
            current = self._sessions.pop(task_id, None)
        if current is not None:
            current[1].close()

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for _, session in sessions:
            session.close()


class WebClient:
    """A same-origin client with bounded redirects, output, and form state."""

    def __init__(
        self,
        task_id: str,
        allowed_origin: str,
        manager: WebSessionManager | None = None,
        *,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = 1_000_000,
        max_redirects: int = 5,
    ) -> None:
        if timeout_seconds <= 0 or max_response_bytes <= 0 or max_redirects < 0:
            raise InvalidArgumentError("request bounds must be positive")
        self.task_id = task_id
        self.allowed_origin = allowed_origin.rstrip("/") + "/"
        self._allowed = _origin(allowed_origin)
        self._manager = manager or WebSessionManager()
        self._session = self._manager.for_task(task_id, allowed_origin)
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects
        self._forms: tuple[ParsedForm, ...] = ()

    def _url(self, value: str) -> str:
        url = urljoin(self.allowed_origin, value)
        if _origin(url) != self._allowed:
            raise BlockedURLError("request target is outside the allowed origin")
        return url

    @staticmethod
    def _headers(headers: Mapping[str, str] | None) -> dict[str, str]:
        safe: dict[str, str] = {}
        for key, value in (headers or {}).items():
            if key.lower().strip() in _BLOCKED_HEADERS:
                raise BlockedURLError(f"sensitive header {key!r} is not allowed")
            if "\n" in key or "\r" in key or "\n" in str(value) or "\r" in str(value):
                raise InvalidArgumentError("header names and values cannot contain newlines")
            safe[str(key)] = str(value)
        return safe

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Mapping[str, Any] | None = None,
        json_body: Any | None = None,
        headers: Mapping[str, str] | None = None,
        expect_html: bool = True,
    ) -> WebResponse:
        method = method.upper()
        if method not in {"GET", "POST"}:
            raise InvalidArgumentError("only GET and POST are supported")
        if data is not None and json_body is not None:
            raise InvalidArgumentError("provide either form data or JSON, not both")
        target = self._url(url)
        safe_headers = self._headers(headers)
        redirects = 0
        current_method = method
        current_data = data
        current_json = json_body
        current_params = params

        while True:
            try:
                raw = self._session.request(
                    current_method,
                    target,
                    params=current_params,
                    data=current_data,
                    json=current_json,
                    headers=safe_headers,
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.Timeout as exc:
                raise RequestTimeoutError("web request timed out") from exc
            except requests.RequestException as exc:
                raise WebError(f"web request failed: {type(exc).__name__}") from exc

            if raw.status_code in _REDIRECTS and raw.headers.get("Location"):
                raw.close()
                if redirects >= self.max_redirects:
                    raise RedirectLoopError("redirect limit exceeded")
                target = self._url(urljoin(target, raw.headers["Location"]))
                redirects += 1
                current_params = None
                if raw.status_code == 303 or (
                    raw.status_code in {301, 302} and current_method == "POST"
                ):
                    current_method = "GET"
                    current_data = None
                    current_json = None
                continue

            content_length = raw.headers.get("Content-Length")
            if content_length and content_length.isdigit():
                if int(content_length) > self.max_response_bytes:
                    raw.close()
                    raise ResponseTooLargeError("response exceeds the byte limit")
            chunks: list[bytes] = []
            size = 0
            try:
                for chunk in raw.iter_content(chunk_size=16_384):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise ResponseTooLargeError("response exceeds the byte limit")
                    chunks.append(chunk)
            finally:
                raw.close()

            headers_out = {key.lower(): value for key, value in raw.headers.items()}
            body = b"".join(chunks)
            content_type = headers_out.get("content-type", "").lower()
            if raw.status_code in {401, 403}:
                if current_method == "GET" and self._session.cookies:
                    raise SessionExpiredError("the server rejected the existing session")
                raise AuthenticationRejectedError("authentication was rejected")
            if expect_html and "html" not in content_type:
                raise NonHTMLError(
                    f"expected HTML but received {content_type or 'unknown content'}"
                )
            response = WebResponse(
                status_code=raw.status_code,
                url=str(raw.url),
                headers=headers_out,
                body=body,
            )
            if expect_html:
                try:
                    page = parse_html(response, response.url)
                except Exception as exc:  # noqa: BLE001
                    raise HTMLParseError("the HTML response could not be parsed") from exc
                self._forms = page.forms
                response = WebResponse(
                    status_code=response.status_code,
                    url=response.url,
                    headers=response.headers,
                    body=response.body,
                    page=page,
                )
            return response

    def _find_form(self, form_ref: str) -> ParsedForm:
        matches = [
            form
            for form in self._forms
            if form_ref in {form.ref, form.form_id, form.name}
        ]
        if not matches:
            raise MissingControlError(f"form {form_ref!r} was not found")
        if len(matches) > 1:
            raise AmbiguousControlError(f"form {form_ref!r} is ambiguous")
        return matches[0]

    @staticmethod
    def _form_data(form: ParsedForm, fields: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for control in form.controls:
            if control.control_type in {"submit", "button", "reset", "file"}:
                continue
            if control.control_type in {"checkbox", "radio"} and not control.checked:
                continue
            payload[control.name] = control.value

        for field, value in fields.items():
            matches = [
                control
                for control in form.controls
                if field == control.name or field.casefold() == control.label.casefold()
            ]
            unique_names = {control.name for control in matches}
            if not matches:
                raise MissingControlError(f"form control {field!r} was not found")
            if len(unique_names) > 1:
                raise AmbiguousControlError(f"form control {field!r} is ambiguous")
            payload[matches[0].name] = value
        return payload

    def submit_form(
        self,
        form_ref: str,
        fields: Mapping[str, Any],
        *,
        expect_html: bool = True,
    ) -> WebResponse:
        form = self._find_form(form_ref)
        payload = self._form_data(form, fields)
        if form.method not in {"get", "post"}:
            raise InvalidArgumentError(f"unsupported form method {form.method!r}")
        if form.enctype.startswith("multipart/"):
            raise InvalidArgumentError("multipart forms are not supported")
        if form.method == "get":
            return self.request("GET", form.action, params=payload, expect_html=expect_html)
        if form.enctype == "application/json":
            return self.request(
                "POST", form.action, json_body=payload, expect_html=expect_html
            )
        return self.request("POST", form.action, data=payload, expect_html=expect_html)
