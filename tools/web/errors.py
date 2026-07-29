"""Stable errors returned by the web subsystem."""

from __future__ import annotations


class WebError(RuntimeError):
    """An expected, bounded web-tool failure."""

    error_code = "WEB_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class InvalidArgumentError(WebError):
    error_code = "INVALID_ARGUMENT"


class BlockedURLError(WebError):
    error_code = "URL_BLOCKED"


class RequestTimeoutError(WebError):
    error_code = "TIMEOUT"


class ResponseTooLargeError(WebError):
    error_code = "OUTPUT_TOO_LARGE"


class RedirectLoopError(WebError):
    error_code = "REDIRECT_LOOP"


class NonHTMLError(WebError):
    error_code = "NON_HTML"


class HTMLParseError(WebError):
    error_code = "HTML_PARSE_ERROR"


class AuthenticationRejectedError(WebError):
    error_code = "AUTH_REJECTED"


class SessionExpiredError(WebError):
    error_code = "SESSION_EXPIRED"


class MissingControlError(WebError):
    error_code = "MISSING_CONTROL"


class AmbiguousControlError(WebError):
    error_code = "AMBIGUOUS_CONTROL"
