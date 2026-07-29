"""Stateful, task-isolated web tools."""

from .errors import WebError
from .html import ParsedPage, parse_html
from .session import WebClient, WebResponse, WebSessionManager
from .tool import WebTool

__all__ = [
    "ParsedPage",
    "WebClient",
    "WebError",
    "WebResponse",
    "WebSessionManager",
    "WebTool",
    "parse_html",
]
