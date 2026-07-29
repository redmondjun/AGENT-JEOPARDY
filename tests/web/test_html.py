from __future__ import annotations

import unittest

from tools.web.html import parse_html


class FakeResponse:
    url = "https://event.example/tasks/start"
    text = """
      <html><head><title> Web Task </title><script>steal()</script></head>
      <body>
        <a href="../next"> Next step </a>
        <form name="gate" action="submit">
          <input type="hidden" name="csrf_token" value="top-secret">
          <label for="code">Access code</label><input id="code" name="code">
          <select name="choice"><option value="a">Alpha</option></select>
        </form>
        <table><tr><td>A</td><td>B</td></tr></table>
        <p role="alert">Try again</p>
      </body></html>
    """


class ParseHTMLTests(unittest.TestCase):
    def test_semantic_page_and_private_hidden_value(self) -> None:
        page = parse_html(FakeResponse())
        self.assertEqual(page.semantic["title"], "Web Task")
        self.assertEqual(
            page.semantic["links"][0]["url"], "https://event.example/next"
        )
        self.assertEqual(
            page.semantic["forms"][0]["action"],
            "https://event.example/tasks/submit",
        )
        self.assertEqual(page.semantic["forms"][0]["controls"][1]["label"], "Access code")
        self.assertEqual(page.semantic["tables"], [[["A", "B"]]])
        self.assertEqual(page.semantic["validation_messages"], ["Try again"])
        self.assertNotIn("steal()", page.semantic["text"])
        self.assertNotIn("top-secret", repr(page.semantic))
        self.assertEqual(page.forms[0].controls[0].value, "top-secret")

    def test_malformed_html_is_tolerated(self) -> None:
        class Broken:
            url = "https://event.example/"
            text = "<html><body><form><input name='value'><div"

        page = parse_html(Broken())
        self.assertEqual(page.forms[0].controls[0].name, "value")


if __name__ == "__main__":
    unittest.main()
