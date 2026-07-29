import pytest

from solver.prompts import CATEGORIES, _CATEGORY_PROMPTS, get_system_prompt


ALL_TOOLS = {
    "list_files",
    "read_file",
    "write_scratch_file",
    "run_python",
    "run_process",
    "inspect_archive",
    "extract_archive",
    "web",
}

CATEGORY_TOOLS = {
    "Needle in the Haystack": {"list_files", "read_file", "run_python"},
    "The Dark Web": {"web"},
    "Ship It": {
        "list_files",
        "read_file",
        "write_scratch_file",
        "run_python",
        "run_process",
    },
    "Ancient Scrolls": {
        "list_files",
        "read_file",
        "run_python",
        "inspect_archive",
        "extract_archive",
    },
    "Cryptic": {
        "list_files",
        "read_file",
        "run_python",
        "run_process",
        "inspect_archive",
        "extract_archive",
    },
    "Heavy Compute": {"write_scratch_file", "run_python", "run_process"},
}


@pytest.mark.parametrize("category", CATEGORIES)
def test_category_prompt_names_registered_tools(category: str) -> None:
    category_block = _CATEGORY_PROMPTS[category]

    assert CATEGORY_TOOLS[category] <= {
        tool for tool in ALL_TOOLS if tool in category_block
    }
    assert "runtime tool" not in category_block
    assert "document tool" not in category_block


@pytest.mark.parametrize(
    "category",
    [
        "Needle in the Haystack",
        "Ship It",
        "Ancient Scrolls",
        "Cryptic",
        "Heavy Compute",
    ],
)
def test_runtime_categories_require_exact_answer_marker(category: str) -> None:
    prompt = get_system_prompt(category)

    assert "ANSWER: <exact value>" in prompt
    assert "exact_value" in prompt


def test_dark_web_prompt_requires_stateful_web_flow() -> None:
    prompt = get_system_prompt("The Dark Web")

    assert "action=request" in prompt
    assert "action=submit_form" in prompt
    assert "preserves cookies" in prompt


def test_unknown_category_is_safe_and_still_actionable() -> None:
    prompt = get_system_prompt("Unexpected Category")

    assert ALL_TOOLS <= {tool for tool in ALL_TOOLS if tool in prompt}
    assert "FINAL_ANSWER: <answer>" in prompt


@pytest.mark.parametrize("category", CATEGORIES)
def test_prompts_stay_compact(category: str) -> None:
    assert len(get_system_prompt(category)) < 2_100
