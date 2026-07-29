"""Frozen-contract adapter for the stateful web client."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Mapping

from contracts import TaskContext, ToolRequest, ToolResult

from .errors import InvalidArgumentError, WebError
from .session import WebClient, WebResponse, WebSessionManager

_SECRET_KEYS = re.compile(
    r"(?:pass(?:word)?|secret|token|csrf|xsrf|auth|session|cookie|api[_-]?key)",
    re.IGNORECASE,
)


def redact(value: Any) -> Any:
    """Recursively remove secrets before content reaches logs or a model."""

    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if _SECRET_KEYS.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    text = str(value)
    text = re.sub(
        r"(?i)(authorization|cookie|x-api-key|api[_-]?key|password|secret|"
        r"csrf[_-]?token|session[_-]?(?:id|token))\s*[:=]\s*\S+",
        r"\1=<redacted>",
        text,
    )
    return text


class WebTool:
    name = "web"

    def __init__(
        self,
        manager: WebSessionManager | None = None,
        *,
        max_output_chars: int = 32_000,
    ) -> None:
        self._manager = manager or WebSessionManager()
        self._max_output_chars = max_output_chars
        self._lock = threading.RLock()
        self._clients: dict[str, WebClient] = {}

    @staticmethod
    def _result(**kwargs: Any) -> ToolResult:
        return ToolResult(**kwargs)

    def _client(self, task: TaskContext) -> WebClient:
        metadata = dict(getattr(task, "metadata", {}) or {})
        allowed_origin = metadata.get("allowed_origin") or os.environ.get(
            "JEOPARDY_BASE_URL", ""
        )
        if not allowed_origin:
            raise InvalidArgumentError("task metadata must provide allowed_origin")
        with self._lock:
            client = self._clients.get(task.task_id)
            if client is None:
                client = WebClient(
                    task.task_id,
                    str(allowed_origin),
                    self._manager,
                    timeout_seconds=min(
                        30.0,
                        max(0.1, float(metadata.get("web_timeout_seconds", 10.0))),
                    ),
                    max_response_bytes=min(
                        2_000_000,
                        max(1, int(metadata.get("web_max_response_bytes", 1_000_000))),
                    ),
                )
                self._clients[task.task_id] = client
            return client

    def _output(self, response: WebResponse) -> str:
        if response.page is not None:
            body: Any = response.page.semantic
        else:
            body = {
                "url": response.url,
                "status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "body": response.text,
            }
        encoded = json.dumps(redact(body), ensure_ascii=False, separators=(",", ":"))
        if len(encoded) > self._max_output_chars:
            encoded = encoded[: self._max_output_chars] + "...<truncated>"
        return encoded

    def execute(self, request: ToolRequest, task: TaskContext) -> ToolResult:
        started = time.monotonic()
        try:
            arguments = dict(request.arguments)
            action = str(arguments.pop("action", "request"))
            if action == "reset":
                self._manager.reset(task.task_id)
                with self._lock:
                    self._clients.pop(task.task_id, None)
                output = '{"reset":true}'
            elif action == "request":
                client = self._client(task)
                response = client.request(
                    str(arguments.pop("method", "GET")),
                    str(arguments.pop("url")),
                    params=arguments.pop("params", None),
                    data=arguments.pop("data", None),
                    json_body=arguments.pop("json", None),
                    headers=arguments.pop("headers", None),
                    expect_html=bool(arguments.pop("expect_html", True)),
                )
                if arguments:
                    raise InvalidArgumentError(
                        f"unknown request arguments: {sorted(arguments)}"
                    )
                output = self._output(response)
            elif action == "submit_form":
                client = self._client(task)
                response = client.submit_form(
                    str(arguments.pop("form_ref")),
                    dict(arguments.pop("fields", {})),
                    expect_html=bool(arguments.pop("expect_html", True)),
                )
                if arguments:
                    raise InvalidArgumentError(
                        f"unknown form arguments: {sorted(arguments)}"
                    )
                output = self._output(response)
            else:
                raise InvalidArgumentError(f"unknown web action {action!r}")
            return self._result(
                ok=True,
                output=output,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        except WebError as exc:
            return self._result(
                ok=False,
                output=str(redact(str(exc))),
                error_code=exc.error_code,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        except (KeyError, TypeError, ValueError):
            return self._result(
                ok=False,
                output="invalid web tool arguments",
                error_code="INVALID_ARGUMENT",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
        except Exception:  # noqa: BLE001
            return self._result(
                ok=False,
                output="unexpected bounded web tool failure",
                error_code="WEB_INTERNAL_ERROR",
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
