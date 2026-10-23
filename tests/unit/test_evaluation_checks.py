import json

import pytest

from evaluation.run_evaluation import (
    DEFAULT_QUESTIONS_FILE,
    EvalCase,
    EvaluationError,
    contains_any_alternative,
    evaluate_answer,
    load_cases,
)

REFUSAL = "I don't have that information in the company documents."


def make_case(**overrides) -> EvalCase:
    values = {"case_id": "case-1", "question": "Question?"}
    values.update(overrides)
    return EvalCase(**values)


@pytest.mark.parametrize(
    ("text", "option", "expected"),
    [
        ("Carry-forward leave expires on 31 March.", "31 march|march 31", True),
        ("It expires on March 31.", "31 march|march 31", True),
        ("It expires in April.", "31 march|march 31", False),
        ("You get 12 DAYS.", "12 days", True),
    ],
)
def test_contains_any_alternative(text, option, expected):
    assert contains_any_alternative(text, option) is expected


def test_a_correct_answer_passes():
    case = make_case(must_contain=["12|twelve"])

    assert evaluate_answer(case, "You get 12 days.", answered_from_documents=True) == []


def test_missing_content_is_reported():
    case = make_case(must_contain=["HR", "department head"])

    failures = evaluate_answer(case, "Only HR approves.", answered_from_documents=True)

    assert failures == ["answer is missing 'department head'"]


def test_forbidden_content_is_reported():
    case = make_case(must_contain=["12"], must_not_contain=["stock"])

    failures = evaluate_answer(case, "12 days, plus stock options.", answered_from_documents=True)

    assert failures == ["answer must not contain 'stock'"]


def test_answering_an_out_of_scope_question_fails():
    failures = evaluate_answer(make_case(should_answer=False), "Paris.", answered_from_documents=True)

    assert failures == ["expected a refusal but got an answer"]


def test_a_correct_refusal_passes():
    assert evaluate_answer(make_case(should_answer=False), REFUSAL, answered_from_documents=False) == []


def test_refusing_an_in_scope_question_fails_without_extra_noise():
    failures = evaluate_answer(make_case(must_contain=["12"]), REFUSAL, answered_from_documents=False)

    assert failures == ["expected an answer but got a refusal"]


def test_the_shipped_questions_file_is_valid():
    cases = load_cases(DEFAULT_QUESTIONS_FILE)

    assert len(cases) >= 20
    assert all(case.question.strip() for case in cases)
    assert all(case.must_contain for case in cases if case.should_answer)


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"id": "a", "question": "Q1?"}, {"id": "a", "question": "Q2?"}]), encoding="utf-8")

    with pytest.raises(EvaluationError, match="Duplicate case id"):
        load_cases(path)


def test_invalid_session_names_are_rejected(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"id": "a", "question": "Q?", "session": "bad name!"}]), encoding="utf-8")

    with pytest.raises(EvaluationError, match="Invalid session name"):
        load_cases(path)


def test_cases_missing_required_fields_are_rejected(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"id": "a"}]), encoding="utf-8")

    with pytest.raises(EvaluationError, match="needs an 'id' and a 'question'"):
        load_cases(path)