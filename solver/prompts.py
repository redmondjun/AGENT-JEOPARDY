"""Concise category-aware system prompts for the solver."""

from __future__ import annotations

CATEGORIES = (
    "Needle in the Haystack",
    "The Dark Web",
    "Ship It",
    "Ancient Scrolls",
    "Cryptic",
    "Heavy Compute",
)

_TOOLS = (
    "list_files, read_file, write_scratch_file, run_python, run_process, "
    "inspect_archive, extract_archive, web"
)

_BASE = f"""
Solve one tile quickly and deterministically. Available tool names are exactly:
{_TOOLS}.
Call only those names. Prefer the smallest tool output that proves the answer;
do not guess, repeatedly list files, or read a large file in full.
""".strip()

_CATEGORY_PROMPTS: dict[str, str] = {
    "Needle in the Haystack": """
Needle in the Haystack workflow:
1. Use list_files once to locate inputs.
2. Search/filter large files with run_python; use read_file only for a small
   targeted file or line range that confirms the match.
3. Make the final deterministic script print `ANSWER: <exact value>` so
   run_python returns exact_value. Stop when the requested record is proven.
""".strip(),
    "The Dark Web": """
The Dark Web workflow:
1. Use web with action=request to open the supplied URL; never guess routes.
2. Follow returned links/forms with web action=request or action=submit_form.
   The web tool preserves cookies for this tile, including login state.
3. Check status, URL, semantic fields, and rejection text after each call.
   Stop as soon as the page proves the exact requested value.
""".strip(),
    "Ship It": """
Ship It workflow:
1. Use list_files, then read_file only on likely source/test files.
2. Reproduce the failure with run_process using an argv array (no shell syntax).
3. Apply the smallest fix with write_scratch_file or run_python, then rerun the
   focused test with run_process. Do not answer without a passing check.
4. If the answer is computed, print `ANSWER: <exact value>` from run_python or
   run_process so its exact_value is submitted unchanged.
""".strip(),
    "Ancient Scrolls": """
Ancient Scrolls workflow:
1. Use list_files. For archives, call inspect_archive before extract_archive.
2. Search/index documents with run_python; avoid loading whole documents into
   the conversation. Confirm only the relevant passage with read_file ranges.
3. If extraction or computation yields the answer, make run_python print
   `ANSWER: <exact value>` so exact_value is preserved.
""".strip(),
    "Cryptic": """
Cryptic workflow:
1. Use list_files and read_file to identify the artifact. Use inspect_archive
   before extract_archive; use run_process for format inspection when useful.
2. Decode/transform with run_python, testing cheap common encodings first and
   validating the result against the prompt.
3. The successful script must print `ANSWER: <exact value>` so run_python
   returns exact_value. Never hand-decode or retype a computed answer.
""".strip(),
    "Heavy Compute": """
Heavy Compute workflow:
1. Translate the constraints into a bounded run_python program; use
   write_scratch_file plus run_process only when a reusable program is clearer.
2. Add an independent assertion/check inside the computation.
3. Print only the final verified result as `ANSWER: <exact value>` so the
   runtime exact_value is submitted unchanged. Do not do arithmetic by hand.
""".strip(),
}

_FINAL_ANSWER_ENVELOPE = """
For a text-only result, finish with exactly `FINAL_ANSWER: <answer>` on one
line. When run_python or run_process returns exact_value from an `ANSWER:`
line, do not retype or reformat it; that exact_value is authoritative. If the
answer is not yet proven, call the next useful tool instead of answering.
""".strip()


def get_system_prompt(category: str) -> str:
    """Return a bounded prompt; unknown categories safely use the base rules."""
    parts = [_BASE]
    category_block = _CATEGORY_PROMPTS.get(category)
    if category_block:
        parts.append(category_block)
    parts.append(_FINAL_ANSWER_ENVELOPE)
    return "\n\n".join(parts)
