import pytest

from app.repositories.conversation_repository import ConversationMessage
from app.services.question_rewriter import QuestionRewriter
from tests.fakes import FailingChatModel, ScriptedChatModel

HISTORY = [
    ConversationMessage(role="user", content="What is the notice period if I am on probation?"),
    ConversationMessage(role="assistant", content="Employees on probation must serve a 15-day notice period."),
]
REWRITTEN = "What is the notice period for confirmed employees?"


@pytest.mark.parametrize(
    "question",
    [
        "And for confirmed employees?",
        "Can it be shortened?",
        "What about hotels?",
        "How about managers?",
        "Hotels?",
        "Is that allowed?",
        "Same for contractors?",
    ],
)
def test_follow_up_questions_are_sent_to_the_model_for_rewriting(question):
    model = ScriptedChatModel(REWRITTEN)

    assert QuestionRewriter(model).rewrite(question, HISTORY) == REWRITTEN
    assert model.calls == 1


@pytest.mark.parametrize(
    "question",
    [
        "How many sick leave days do I get?",
        "What is the meal allowance during business travel?",
        "How long does maternity leave last?",
    ],
)
def test_complete_questions_are_never_rewritten(question):
    model = ScriptedChatModel(REWRITTEN)

    assert QuestionRewriter(model).rewrite(question, HISTORY) == question
    assert model.calls == 0


def test_first_question_of_a_conversation_is_not_rewritten():
    model = ScriptedChatModel(REWRITTEN)

    assert QuestionRewriter(model).rewrite("And for managers?", []) == "And for managers?"
    assert model.calls == 0


def test_intro_lines_and_quotes_are_removed_from_the_model_output():
    model = ScriptedChatModel(f'Here is the rewritten question:\n"{REWRITTEN}"')

    assert QuestionRewriter(model).rewrite("And for confirmed employees?", HISTORY) == REWRITTEN


def test_model_failure_falls_back_to_the_original_question():
    rewriter = QuestionRewriter(FailingChatModel())

    assert rewriter.rewrite("And for confirmed employees?", HISTORY) == "And for confirmed employees?"


@pytest.mark.parametrize("bad_output", ["", "x" * 500])
def test_unusable_model_output_falls_back_to_the_original_question(bad_output):
    rewriter = QuestionRewriter(ScriptedChatModel(bad_output))

    assert rewriter.rewrite("And it?", HISTORY) == "And it?"