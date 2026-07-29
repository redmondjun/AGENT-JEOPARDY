from __future__ import annotations

import json
from pathlib import Path
import time
import unittest

from contracts import TaskContext, Tool, ToolRequest, ToolResult
from tools.web.tool import WebTool, redact

from .fixture_server import fixture_server


class WebToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = fixture_server()
        self.origin = self.context.__enter__()
        self.tool = WebTool()
        self.task = TaskContext(
            task_id="PR-W1",
            category="The Dark Web",
            points=100,
            prompt="Complete the local web flow.",
            answer_format="exact",
            workdir=Path("."),
            files=(),
            deadline_monotonic=time.monotonic() + 30,
            metadata={"allowed_origin": self.origin},
        )

    def tearDown(self) -> None:
        self.tool._manager.close()
        self.context.__exit__(None, None, None)

    def execute(self, **arguments):
        return self.tool.execute(
            ToolRequest("web", arguments, timeout_seconds=10), self.task
        )

    def test_implements_frozen_tool_contract(self) -> None:
        self.assertIsInstance(self.tool, Tool)
        result = self.execute(action="reset")
        self.assertIsInstance(result, ToolResult)

    def test_request_and_form_submission_share_state(self) -> None:
        first = self.execute(action="request", method="GET", url="/")
        self.assertTrue(first.ok, first.output)
        semantic = json.loads(first.output)
        hidden = semantic["forms"][0]["controls"][0]
        self.assertEqual(hidden["value"], "<redacted>")

        second = self.execute(
            action="submit_form",
            form_ref="login",
            fields={"User name": "jun", "password": "round1"},
        )
        self.assertTrue(second.ok, second.output)
        self.assertIn("Welcome Jun", second.output)
        self.assertNotIn("round1", second.output)
        self.assertNotIn("csrf-super-secret", second.output)

    def test_structured_error_and_reset(self) -> None:
        blocked = self.execute(
            action="request", method="GET", url="https://example.com/"
        )
        self.assertFalse(blocked.ok)
        self.assertEqual(blocked.error_code, "URL_BLOCKED")
        reset = self.execute(action="reset")
        self.assertTrue(reset.ok)

    def test_redaction_is_recursive(self) -> None:
        value = redact(
            {
                "password": "secret",
                "nested": [{"csrf_token": "hidden"}],
                "message": "Authorization: bearer-value",
            }
        )
        rendered = repr(value)
        self.assertNotIn("secret", rendered)
        self.assertNotIn("hidden", rendered)
        self.assertNotIn("bearer-value", rendered)


if __name__ == "__main__":
    unittest.main()
