"""The 8 stable error codes every runtime tool reports through
`ToolResult.error_code`. Sara's solver and Nandh's submission gate branch on
these strings, so they must never change once another team is consuming them.

Not an enum on purpose: `ToolResult.error_code` is typed `str | None` in
contracts.py, so these are plain module-level string constants — importing
code compares `result.error_code == errors.PATH_BLOCKED`, no `.value` needed.
"""
from __future__ import annotations

INVALID_ARGUMENT = "INVALID_ARGUMENT"
NOT_FOUND = "NOT_FOUND"
PATH_BLOCKED = "PATH_BLOCKED"
TIMEOUT = "TIMEOUT"
OUTPUT_TOO_LARGE = "OUTPUT_TOO_LARGE"
PROCESS_FAILED = "PROCESS_FAILED"
UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"

ALL = frozenset({
    INVALID_ARGUMENT, NOT_FOUND, PATH_BLOCKED, TIMEOUT, OUTPUT_TOO_LARGE,
    PROCESS_FAILED, UNSUPPORTED_FORMAT, DEPENDENCY_UNAVAILABLE,
})


class RuntimeToolError(Exception):
    """Raised by every tools/runtime module (paths, files, processes,
    archives) instead of a bare exception, so `tool.py` can turn it into a
    `ToolResult(ok=False, error_code=..., output=...)` without ever leaking a
    raw traceback to the model.

    `code` must be one of the constants above — enforced here rather than
    trusted at each call site, so a typo'd code fails at raise time, not
    silently at the contract boundary.
    """

    def __init__(self, code: str, message: str) -> None:
        if code not in ALL:
            raise ValueError(f"not a stable error code: {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
