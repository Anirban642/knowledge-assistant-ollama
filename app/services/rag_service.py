"""Service layer: answers a question using retrieved company documents and the LLM."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger
from app.repositories.chunk_repository import RetrievedChunk
from app.services.prompts import NO_ANSWER_MESSAGE, RAG_PROMPT
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


@dataclass(frozen=True)
class RagAnswer:
    """The assistant's answer and the document chunks that best support it."""

    answer: str
    sources: list[RetrievedChunk]
    answered_from_documents: bool


class RagService:
    """Retrieval-Augmented Generation: retrieve relevant chunks, then let the LLM answer from them."""

    def __init__(
        self,
        retrieval_service: RetrievalService,
        chat_model: BaseChatModel,
        max_distance: float,
        source_distance_margin: float,
        max_sources: int,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._chain = RAG_PROMPT | chat_model | StrOutputParser()
        self._max_distance = max_distance
        self._source_distance_margin = source_distance_margin
        self._max_sources = max_sources

    def answer(self, question: str) -> RagAnswer:
        """Answer the question from the knowledge base, or say the information is not available."""
        candidates = self._retrieval_service.search(question)
        relevant_chunks = [chunk for chunk in candidates if chunk.distance <= self._max_distance]

        logger.info(
            "Retrieved %d chunk(s), %d within max distance %.2f (distances: %s)",
            len(candidates),
            len(relevant_chunks),
            self._max_distance,
            [round(chunk.distance, 3) for chunk in candidates],
        )

        if not relevant_chunks:
            return RagAnswer(answer=NO_ANSWER_MESSAGE, sources=[], answered_from_documents=False)

        answer_text = self._generate(question, relevant_chunks)

        if NO_ANSWER_MESSAGE.lower() in answer_text.lower():
            return RagAnswer(answer=NO_ANSWER_MESSAGE, sources=[], answered_from_documents=False)

        return RagAnswer(
            answer=answer_text,
            sources=self._select_sources(relevant_chunks),
            answered_from_documents=True,
        )

    def _select_sources(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Pick the chunks to show as sources: the best match plus any nearly as close, capped.

        The LLM still received every relevant chunk. This only keeps the response readable.
        """
        ranked = sorted(chunks, key=lambda chunk: chunk.distance)
        cutoff = ranked[0].distance + self._source_distance_margin
        close_matches = [chunk for chunk in ranked if chunk.distance <= cutoff]
        return close_matches[: self._max_sources]

    def _generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Run the prompt through the LLM, converting any failure into a clear app error."""
        try:
            return self._chain.invoke({"context": self._format_context(chunks), "question": question}).strip()
        except Exception as exc:
            logger.exception("LLM call failed")
            raise ExternalServiceError(
                "The language model is unavailable. Check that Ollama is running and the model is installed."
            ) from exc

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        """Join chunks into one context block, each labelled with where it came from."""
        blocks = []
        for chunk in chunks:
            location = f"{chunk.source}, page {chunk.page}" if chunk.page else chunk.source
            blocks.append(f"[Source: {location}]\n{chunk.text}")
        return "\n\n---\n\n".join(blocks)