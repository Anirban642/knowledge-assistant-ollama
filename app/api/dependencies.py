"""FastAPI dependency providers. Routes ask for services here instead of building them."""

from functools import lru_cache

from fastapi import Depends
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings, get_settings
from app.infrastructure.embeddings import NomicOllamaEmbeddings
from app.infrastructure.file_document_loader import FileDocumentLoader
from app.infrastructure.file_storage import LocalDocumentStorage
from app.infrastructure.llm import create_chat_model
from app.infrastructure.vector_store import create_vector_store
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.conversation_repository import (
    ConversationRepository,
    InMemoryConversationRepository,
)
from app.services.chat_service import ChatService
from app.services.chunking_service import ChunkingService
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import DocumentService
from app.services.ingestion_service import IngestionService
from app.services.question_rewriter import QuestionRewriter
from app.services.rag_service import RagService
from app.services.retrieval_service import RetrievalService


@lru_cache
def get_chunk_repository() -> ChunkRepository:
    """Create the repository once and reuse it, so the vector DB is opened a single time."""
    settings = get_settings()
    embeddings = NomicOllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )
    vector_store = create_vector_store(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_dir,
        embeddings=embeddings,
    )
    return ChunkRepository(vector_store)


@lru_cache
def get_conversation_repository() -> InMemoryConversationRepository:
    """Create the history store once. It must be a single shared instance, or sessions would be forgotten."""
    settings = get_settings()
    return InMemoryConversationRepository(
        max_messages_per_session=settings.conversation_history_turns * 2,  # each turn = question + answer
        max_sessions=settings.conversation_max_sessions,
    )


@lru_cache
def get_chat_model() -> BaseChatModel:
    """Create the LLM client once and reuse it."""
    settings = get_settings()
    return create_chat_model(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
        num_gpu=settings.llm_num_gpu,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def get_document_processing_service(
    settings: Settings = Depends(get_settings),
) -> DocumentProcessingService:
    """Build the document processing service using values from settings."""
    return DocumentProcessingService(
        loader=FileDocumentLoader(),
        chunking_service=ChunkingService(settings.chunk_size, settings.chunk_overlap),
        documents_dir=settings.documents_dir,
    )


def get_ingestion_service(
    processing_service: DocumentProcessingService = Depends(get_document_processing_service),
    chunk_repository: ChunkRepository = Depends(get_chunk_repository),
) -> IngestionService:
    """Build the ingestion service."""
    return IngestionService(processing_service, chunk_repository)


def get_document_service(
    settings: Settings = Depends(get_settings),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
    chunk_repository: ChunkRepository = Depends(get_chunk_repository),
) -> DocumentService:
    """Build the document management service."""
    return DocumentService(
        storage=LocalDocumentStorage(settings.documents_dir),
        ingestion_service=ingestion_service,
        chunk_repository=chunk_repository,
        supported_extensions=FileDocumentLoader().supported_extensions,
        max_upload_bytes=settings.max_upload_bytes,
    )


def get_retrieval_service(
    settings: Settings = Depends(get_settings),
    chunk_repository: ChunkRepository = Depends(get_chunk_repository),
) -> RetrievalService:
    """Build the retrieval service."""
    return RetrievalService(chunk_repository, settings.retrieval_top_k)


def get_rag_service(
    settings: Settings = Depends(get_settings),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    chat_model: BaseChatModel = Depends(get_chat_model),
) -> RagService:
    """Build the RAG service that answers a single, standalone question."""
    return RagService(retrieval_service, chat_model, settings.retrieval_max_distance)


def get_question_rewriter(chat_model: BaseChatModel = Depends(get_chat_model)) -> QuestionRewriter:
    """Build the follow-up question rewriter."""
    return QuestionRewriter(chat_model)


def get_chat_service(
    rag_service: RagService = Depends(get_rag_service),
    question_rewriter: QuestionRewriter = Depends(get_question_rewriter),
    conversations: ConversationRepository = Depends(get_conversation_repository),
) -> ChatService:
    """Build the chat service that handles conversations."""
    return ChatService(rag_service, question_rewriter, conversations)