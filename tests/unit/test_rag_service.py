import pytest

from app.core.exceptions import ExternalServiceError
from app.services.prompts import NO_ANSWER_MESSAGE
from app.services.rag_service import RagService
from tests.fakes import FailingChatModel, ScriptedChatModel, StubRetrievalService, make_chunk


def build_service(chunks, model, *, max_distance=0.45, margin=0.08, max_sources=3) -> RagService:
    return RagService(
        retrieval_service=StubRetrievalService(chunks),
        chat_model=model,
        max_distance=max_distance,
        source_distance_margin=margin,
        max_sources=max_sources,
    )


def test_answers_from_relevant_chunks():
    chunk = make_chunk(distance=0.2)
    service = build_service([chunk], ScriptedChatModel("12 days."))

    result = service.answer("How many sick leave days do I get?")

    assert result.answer == "12 days."
    assert result.answered_from_documents is True
    assert result.sources == [chunk]


def test_far_away_chunks_are_rejected_without_calling_the_model():
    model = ScriptedChatModel("should not be used")
    service = build_service([make_chunk(distance=0.6)], model)

    result = service.answer("What is the capital of France?")

    assert result.answer == NO_ANSWER_MESSAGE
    assert result.answered_from_documents is False
    assert result.sources == []
    assert model.calls == 0


def test_no_retrieved_chunks_means_no_answer():
    model = ScriptedChatModel("should not be used")

    result = build_service([], model).answer("Anything?")

    assert result.answered_from_documents is False
    assert model.calls == 0


def test_chunk_exactly_at_the_distance_limit_is_kept():
    service = build_service([make_chunk(distance=0.45)], ScriptedChatModel("An answer."))

    assert service.answer("Question?").answered_from_documents is True


@pytest.mark.parametrize(
    "model_output",
    [NO_ANSWER_MESSAGE, f"Sorry. {NO_ANSWER_MESSAGE.upper()}"],
)
def test_model_refusal_is_reported_as_not_answered(model_output):
    service = build_service([make_chunk()], ScriptedChatModel(model_output))

    result = service.answer("Does the company give stock options?")

    assert result.answer == NO_ANSWER_MESSAGE
    assert result.answered_from_documents is False
    assert result.sources == []


def test_sources_include_only_chunks_close_to_the_best_match():
    chunks = [
        make_chunk(chunk_index=0, distance=0.20),
        make_chunk(chunk_index=1, distance=0.25),
        make_chunk(chunk_index=2, distance=0.40),
    ]
    result = build_service(chunks, ScriptedChatModel("Answer.")).answer("Question?")

    assert [source.chunk_index for source in result.sources] == [0, 1]


def test_sources_are_capped_and_sorted_best_first():
    chunks = [make_chunk(chunk_index=1, distance=0.30), make_chunk(chunk_index=0, distance=0.20)]

    sorted_result = build_service(chunks, ScriptedChatModel("A."), margin=0.5).answer("Q?")
    capped_result = build_service(chunks, ScriptedChatModel("A."), margin=0.5, max_sources=1).answer("Q?")

    assert [source.chunk_index for source in sorted_result.sources] == [0, 1]
    assert [source.chunk_index for source in capped_result.sources] == [0]


def test_model_failure_becomes_an_external_service_error():
    service = build_service([make_chunk()], FailingChatModel())

    with pytest.raises(ExternalServiceError):
        service.answer("Question?")