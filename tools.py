"""Task-scoped tools exposed to the model.

Tools can inspect and mutate only a tile's scratch directory. They never call
the Jeopardy submission API; finalization merely returns a candidate to the
solver for policy checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urljoin, urlparse
import uuid

import requests


MAX_OUTPUT_CHARS = 32_000
MAX_WRITE_CHARS = 1_000_000
_PYTHON_SLOTS = threading.BoundedSemaphore(
    max(1, int(os.environ.get("PYTHON_SLOTS", "2")))
)
_SECRET_ENV_NAMES = {
    "TEAM_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
}


@dataclass(frozen=True)
class CandidateAnswer:
    value: str
    confidence: float
    evidence: tuple[str, ...]
    verification_notes: str
    exact_value_from_tool: bool = False
    answer_ref: str | None = None


@dataclass
class ToolExecution:
    content: str
    is_error: bool = False
    candidate: CandidateAnswer | None = None


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_files",
        "description": "List files in the current tile workspace recursively.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a text or binary file from the tile workspace. Binary data "
            "is represented with a safe escaped preview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_OUTPUT_CHARS,
                    "default": 12000,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or replace a text file inside the tile workspace. Use this "
            "for scripts, transformed data, and code fixes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean", "default": False},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run Python in the tile workspace. Print the final answer as the "
            "last non-empty stdout line and set capture_answer=true to receive "
            "an immutable answer_ref."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 90,
                    "default": 30,
                },
                "capture_answer": {"type": "boolean", "default": False},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "name": "http_request",
        "description": (
            "Make an HTTP request while preserving cookies for this tile. "
            "Relative URLs resolve against JEOPARDY_BASE_URL. The full response "
            "body is saved in the tile workspace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                    "default": "GET",
                },
                "url": {"type": "string"},
                "headers": {"type": "object"},
                "params": {"type": "object"},
                "data": {},
                "json": {},
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 120,
                    "default": 30,
                },
                "allow_redirects": {"type": "boolean", "default": True},
                "capture_answer": {"type": "boolean", "default": False},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "finalize_answer",
        "description": (
            "Propose the final answer with evidence. Prefer answer_ref when a "
            "tool captured an exact value; otherwise provide answer directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "answer_ref": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ]
                },
                "verification_notes": {"type": "string"},
            },
            "required": ["confidence", "evidence", "verification_notes"],
            "additionalProperties": False,
        },
    },
]


class TaskTools:
    """Tool dispatcher with state isolated to one tile."""

    def __init__(self, task_id: str, root: Path, base_url: str):
        self.task_id = task_id
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/") + "/"
        self.session = requests.Session()
        self.answer_refs: dict[str, str] = {}
        self._http_counter = 0

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        try:
            handler = getattr(self, f"_tool_{name}", None)
            if handler is None:
                return ToolExecution(
                    f"Unknown tool {name!r}. Use one of "
                    f"{[tool['name'] for tool in TOOL_SCHEMAS]}.",
                    is_error=True,
                )
            return handler(arguments)
        except Exception as exc:  # noqa: BLE001 - tools report structured errors
            return ToolExecution(
                f"{type(exc).__name__}: {str(exc)[:2000]}", is_error=True
            )

    def _resolve(self, raw_path: str) -> Path:
        candidate = (self.root / raw_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes the tile workspace")
        return candidate

    def _store_answer(self, value: str) -> str:
        ref = f"answer_{uuid.uuid4().hex[:12]}"
        self.answer_refs[ref] = value
        return ref

    def _bounded(self, value: str, artifact_name: str | None = None) -> str:
        if len(value) <= MAX_OUTPUT_CHARS:
            return value
        if artifact_name:
            path = self._resolve(artifact_name)
            path.write_text(value, encoding="utf-8", errors="replace")
            suffix = f"\n[full output saved to {path.name}]"
        else:
            suffix = "\n[output truncated]"
        return value[: MAX_OUTPUT_CHARS - len(suffix)] + suffix

    def _tool_list_files(self, args: dict[str, Any]) -> ToolExecution:
        base = self._resolve(str(args.get("path", ".")))
        if not base.exists():
            raise FileNotFoundError(base.name)
        if base.is_file():
            return ToolExecution(f"{base.relative_to(self.root)} ({base.stat().st_size} bytes)")
        rows = []
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(self.root)
            rows.append(f"{rel}/" if path.is_dir() else f"{rel} ({path.stat().st_size} bytes)")
        return ToolExecution(self._bounded("\n".join(rows) or "(empty)"))

    def _tool_read_file(self, args: dict[str, Any]) -> ToolExecution:
        path = self._resolve(str(args["path"]))
        if not path.is_file():
            raise FileNotFoundError(path.name)
        offset = max(0, int(args.get("offset", 0)))
        maximum = min(MAX_OUTPUT_CHARS, max(1, int(args.get("max_chars", 12000))))
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="backslashreplace")
        segment = text[offset : offset + maximum]
        trailer = ""
        if offset + len(segment) < len(text):
            trailer = f"\n[truncated; next offset={offset + len(segment)}]"
        return ToolExecution(segment + trailer)

    def _tool_write_file(self, args: dict[str, Any]) -> ToolExecution:
        content = str(args["content"])
        if len(content) > MAX_WRITE_CHARS:
            raise ValueError(f"content exceeds {MAX_WRITE_CHARS} characters")
        path = self._resolve(str(args["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if bool(args.get("append", False)) else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return ToolExecution(f"wrote {len(content)} characters to {path.relative_to(self.root)}")

    def _tool_run_python(self, args: dict[str, Any]) -> ToolExecution:
        code = str(args["code"])
        timeout = min(90.0, max(1.0, float(args.get("timeout_seconds", 30))))
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in _SECRET_ENV_NAMES
        }
        started = time.monotonic()
        acquired = _PYTHON_SLOTS.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError("timed out waiting for a Python execution slot")
        try:
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                errors="backslashreplace",
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="backslashreplace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="backslashreplace")
            return ToolExecution(
                self._bounded(
                    f"Python timed out after {timeout:.1f}s\nstdout:\n{stdout}"
                    f"\nstderr:\n{stderr}",
                    f"python_timeout_{int(started)}.txt",
                ),
                is_error=True,
            )
        finally:
            _PYTHON_SLOTS.release()

        stdout = completed.stdout
        stderr = completed.stderr
        payload: dict[str, Any] = {
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "stdout": stdout,
            "stderr": stderr,
        }
        if bool(args.get("capture_answer", False)) and completed.returncode == 0:
            lines = [line for line in stdout.splitlines() if line.strip()]
            if not lines:
                payload["capture_error"] = "stdout had no non-empty line"
            else:
                value = lines[-1].strip()
                payload["answer_ref"] = self._store_answer(value)
                payload["captured_value"] = value
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        return ToolExecution(
            self._bounded(rendered, f"python_output_{int(started)}.json"),
            is_error=completed.returncode != 0,
        )

    def _tool_http_request(self, args: dict[str, Any]) -> ToolExecution:
        method = str(args.get("method", "GET")).upper()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
            raise ValueError(f"unsupported HTTP method {method}")
        raw_url = str(args["url"])
        url = urljoin(self.base_url, raw_url)
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("HTTP URL must be absolute or event-relative")
        timeout = min(120.0, max(1.0, float(args.get("timeout_seconds", 30))))
        response = self.session.request(
            method,
            url,
            headers=args.get("headers"),
            params=args.get("params"),
            data=args.get("data"),
            json=args.get("json"),
            timeout=timeout,
            allow_redirects=bool(args.get("allow_redirects", True)),
        )
        self._http_counter += 1
        suffix = ".html" if "html" in response.headers.get("content-type", "") else ".txt"
        artifact = self._resolve(f"http_response_{self._http_counter}{suffix}")
        artifact.write_bytes(response.content)
        text = response.content.decode(response.encoding or "utf-8", errors="backslashreplace")
        payload: dict[str, Any] = {
            "status": response.status_code,
            "url": response.url,
            "headers": dict(response.headers),
            "cookies": self.session.cookies.get_dict(),
            "body_file": artifact.name,
            "body": text[:MAX_OUTPUT_CHARS],
            "body_truncated": len(text) > MAX_OUTPUT_CHARS,
        }
        if bool(args.get("capture_answer", False)):
            value = text.strip()
            if value:
                payload["answer_ref"] = self._store_answer(value)
            else:
                payload["capture_error"] = "response body was empty"
        return ToolExecution(
            self._bounded(
                json.dumps(payload, ensure_ascii=False, default=str),
                f"http_result_{self._http_counter}.json",
            )
        )

    def _tool_finalize_answer(self, args: dict[str, Any]) -> ToolExecution:
        answer_ref = args.get("answer_ref")
        answer = args.get("answer")
        if answer_ref:
            if answer_ref not in self.answer_refs:
                raise ValueError("unknown answer_ref")
            value = self.answer_refs[str(answer_ref)]
            exact = True
        elif answer is not None:
            value = str(answer)
            exact = False
            answer_ref = None
        else:
            raise ValueError("provide answer or answer_ref")
        confidence = float(args["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        raw_evidence = args["evidence"]
        if isinstance(raw_evidence, str):
            evidence = (raw_evidence.strip(),)
        elif isinstance(raw_evidence, list):
            evidence = tuple(str(item).strip() for item in raw_evidence if str(item).strip())
        else:
            raise ValueError("evidence must be a string or list of strings")
        if not evidence:
            raise ValueError("at least one evidence item is required")
        notes = str(args["verification_notes"]).strip()
        if not notes:
            raise ValueError("verification_notes must not be empty")
        candidate = CandidateAnswer(
            value=value,
            confidence=confidence,
            evidence=evidence,
            verification_notes=notes,
            exact_value_from_tool=exact,
            answer_ref=str(answer_ref) if answer_ref else None,
        )
        return ToolExecution("candidate captured for policy verification", candidate=candidate)
