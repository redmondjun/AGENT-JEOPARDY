from contracts import CandidateAnswer
from solver import specialists  # noqa: F401  (registers category checks)
from solver.verification import verify_candidate
from tests.solver.fakes import make_task


def test_numeric_recompute_matches_passes():
    task = make_task(category="Heavy Compute", answer_format="numeric", prompt="no constraints here")
    candidate = CandidateAnswer(
        value="12",
        confidence=0.8,
        evidence=("some reasoning", "4 * 3"),
        strategy="Heavy Compute",
    )
    outcome = verify_candidate(candidate, task)
    assert outcome.passed is True


def test_numeric_recompute_mismatch_fails():
    task = make_task(category="Heavy Compute", answer_format="numeric", prompt="no constraints here")
    candidate = CandidateAnswer(
        value="99",
        confidence=0.8,
        evidence=("4 * 3",),  # recomputes to 12, not 99
        strategy="Heavy Compute",
    )
    outcome = verify_candidate(candidate, task)
    assert outcome.passed is False
    assert any("recompute" in reason for reason in outcome.reasons)


def test_literal_envelope_leak_is_rejected():
    task = make_task(category="Needle in the Haystack", answer_format="literal")
    candidate = CandidateAnswer(
        value="FINAL_ANSWER: oops",
        confidence=0.9,
        evidence=("something",),
        strategy="Needle in the Haystack",
    )
    outcome = verify_candidate(candidate, task)
    assert outcome.passed is False


def test_high_confidence_with_no_evidence_is_rejected():
    task = make_task(category="Needle in the Haystack", answer_format="exact")
    candidate = CandidateAnswer(value="Paris", confidence=0.9, evidence=(), strategy="Needle in the Haystack")
    outcome = verify_candidate(candidate, task)
    assert outcome.passed is False
    assert outcome.confidence <= 0.5


def test_cryptic_rejects_value_that_still_looks_encoded():
    task = make_task(category="Cryptic", answer_format="literal")
    candidate = CandidateAnswer(
        value="aGVsbG8gd29ybGQ=",  # still base64, never decoded
        confidence=0.8,
        evidence=("tried to decode",),
        strategy="Cryptic",
    )
    outcome = verify_candidate(candidate, task)
    assert outcome.passed is False


def test_ship_it_requires_run_evidence():
    task = make_task(category="Ship It", answer_format="literal")
    candidate = CandidateAnswer(
        value="fixed the off-by-one bug",
        confidence=0.8,
        evidence=("I read the code and it looks like it should work now",),
        strategy="Ship It",
    )
    outcome = verify_candidate(candidate, task)
    assert outcome.passed is False
    assert any("no evidence of an actual test" in reason for reason in outcome.reasons)


def test_ship_it_passes_with_test_output_evidence():
    task = make_task(category="Ship It", answer_format="literal")
    candidate = CandidateAnswer(
        value="fixed the off-by-one bug",
        confidence=0.8,
        evidence=("pytest: 4 passed, 0 failed, exit code 0",),
        strategy="Ship It",
    )
    outcome = verify_candidate(candidate, task)
    assert outcome.passed is True


def test_heavy_compute_constraint_violation_is_rejected():
    task = make_task(
        category="Heavy Compute",
        answer_format="numeric",
        prompt="Find the smallest value that is at least 10 and at most 20.",
    )
    candidate = CandidateAnswer(value="5", confidence=0.8, evidence=("computed 5",), strategy="Heavy Compute")
    outcome = verify_candidate(candidate, task)
    assert outcome.passed is False
    assert any("violates" in reason for reason in outcome.reasons)
