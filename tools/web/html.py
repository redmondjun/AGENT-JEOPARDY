"""Compact semantic HTML parsing with private form state."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

_SENSITIVE_NAME = re.compile(
    r"(?:pass(?:word)?|secret|token|csrf|xsrf|auth|session|cookie|api[_-]?key)",
    re.IGNORECASE,
)


def is_sensitive(name: str, control_type: str = "") -> bool:
    return control_type.lower() == "password" or bool(_SENSITIVE_NAME.search(name))


@dataclass(frozen=True)
class FormControl:
    name: str
    control_type: str
    value: str
    label: str
    options: tuple[tuple[str, str, bool], ...] = ()
    checked: bool = False
    sensitive: bool = False

    def semantic(self) -> dict[str, Any]:
        value = "<redacted>" if self.sensitive and self.value else self.value
        return {
            "name": self.name,
            "type": self.control_type,
            "label": self.label,
            "value": value,
            "checked": self.checked,
            "options": [
                {
                    "value": "<redacted>" if self.sensitive and option else option,
                    "label": label,
                    "selected": selected,
                }
                for option, label, selected in self.options
            ],
        }


@dataclass(frozen=True)
class ParsedForm:
    ref: str
    form_id: str
    name: str
    method: str
    action: str
    enctype: str
    controls: tuple[FormControl, ...]

    def semantic(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "id": self.form_id,
            "name": self.name,
            "method": self.method,
            "action": self.action,
            "enctype": self.enctype,
            "controls": [control.semantic() for control in self.controls],
        }


@dataclass(frozen=True)
class ParsedPage:
    """Model-safe page data plus unredacted controls kept inside the client."""

    semantic: dict[str, Any]
    forms: tuple[ParsedForm, ...]


def _clean(text: str, limit: int = 12_000) -> str:
    return " ".join(text.split())[:limit]


def _label_for(soup: BeautifulSoup, element: Any) -> str:
    element_id = element.get("id")
    label = soup.find("label", attrs={"for": element_id}) if element_id else None
    if label is None:
        label = element.find_parent("label")
    if label is not None:
        return _clean(label.get_text(" ", strip=True), 500)
    return _clean(
        element.get("aria-label")
        or element.get("placeholder")
        or element.get("name")
        or "",
        500,
    )


def _control(soup: BeautifulSoup, element: Any) -> FormControl | None:
    name = str(element.get("name") or "")
    if not name:
        return None
    tag = element.name.lower()
    control_type = (
        str(element.get("type") or "text").lower() if tag == "input" else tag
    )
    options: tuple[tuple[str, str, bool], ...] = ()
    if tag == "select":
        options = tuple(
            (
                str(option.get("value") or option.get_text(" ", strip=True)),
                _clean(option.get_text(" ", strip=True), 500),
                option.has_attr("selected"),
            )
            for option in element.find_all("option")
        )
        selected = next((value for value, _, chosen in options if chosen), "")
        value = selected or (options[0][0] if options else "")
    elif tag == "textarea":
        value = element.get_text()
    else:
        value = str(element.get("value") or "")
    return FormControl(
        name=name,
        control_type=control_type,
        value=value,
        label=_label_for(soup, element),
        options=options,
        checked=element.has_attr("checked"),
        sensitive=is_sensitive(name, control_type),
    )


def parse_html(response: Any, base_url: str | None = None) -> ParsedPage:
    """Parse a requests-like response into bounded, JSON-serializable data."""

    source_url = base_url or str(getattr(response, "url", ""))
    soup = BeautifulSoup(str(getattr(response, "text", "")), "lxml")
    for unwanted in soup(["script", "style", "noscript", "template"]):
        unwanted.decompose()

    forms: list[ParsedForm] = []
    for index, form in enumerate(soup.find_all("form")):
        form_id = str(form.get("id") or "")
        name = str(form.get("name") or "")
        ref = form_id or name or f"form-{index}"
        controls = tuple(
            control
            for element in form.find_all(["input", "select", "textarea", "button"])
            if (control := _control(soup, element)) is not None
        )
        forms.append(
            ParsedForm(
                ref=ref,
                form_id=form_id,
                name=name,
                method=str(form.get("method") or "get").lower(),
                action=urljoin(source_url, str(form.get("action") or source_url)),
                enctype=str(
                    form.get("enctype") or "application/x-www-form-urlencoded"
                ).lower(),
                controls=controls,
            )
        )

    links = [
        {
            "text": _clean(link.get_text(" ", strip=True), 500),
            "url": urljoin(source_url, str(link.get("href") or "")),
        }
        for link in soup.find_all("a", href=True)
    ][:200]
    tables = [
        [
            [_clean(cell.get_text(" ", strip=True), 1_000) for cell in row.find_all(["th", "td"])]
            for row in table.find_all("tr")[:100]
        ]
        for table in soup.find_all("table")[:20]
    ]
    validation_nodes = soup.select(
        "[role='alert'], .error, .errors, .invalid-feedback, .validation-error"
    )
    semantic = {
        "url": source_url,
        "title": _clean(soup.title.get_text(" ", strip=True), 500) if soup.title else "",
        "text": _clean(soup.get_text(" ", strip=True)),
        "links": links,
        "forms": [form.semantic() for form in forms],
        "tables": tables,
        "validation_messages": [
            message
            for node in validation_nodes
            if (message := _clean(node.get_text(" ", strip=True), 1_000))
        ][:50],
    }
    return ParsedPage(semantic=semantic, forms=tuple(forms))
