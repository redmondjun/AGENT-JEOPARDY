from solver.answer_parser import extract_final_answer, normalize_answer


def test_extract_final_answer_basic():
    text = "I worked through it.\nFINAL_ANSWER: 42\n"
    assert extract_final_answer(text) == "42"


def test_extract_final_answer_none_when_absent():
    assert extract_final_answer("still thinking, no answer yet") is None


def test_extract_final_answer_uses_last_occurrence():
    text = "FINAL_ANSWER: wrong\nwait, let me redo this.\nFINAL_ANSWER: correct"
    assert extract_final_answer(text) == "correct"


def test_normalize_numeric_strips_currency_and_commas():
    assert normalize_answer("$1,234.00", "numeric") == "1234"


def test_normalize_numeric_preserves_non_integer_value():
    assert normalize_answer("3.50", "numeric") == "3.5"


def test_normalize_numeric_handles_percent_sign():
    assert normalize_answer("42%", "numeric") == "42"


def test_normalize_literal_collapses_whitespace_and_quotes():
    assert normalize_answer('  "The   Answer"  ', "literal") == "The   Answer".replace("   ", " ")


def test_normalize_literal_preserves_internal_casing():
    assert normalize_answer("CamelCaseWord", "literal") == "CamelCaseWord"


def test_normalize_exact_only_trims_whitespace():
    assert normalize_answer("  Paris  ", "exact") == "Paris"
