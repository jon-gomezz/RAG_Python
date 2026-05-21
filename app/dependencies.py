"""Construcción e inyección de la tubería RAG para la API.

La tubería se crea una sola vez (singleton) y se comparte entre peticiones para
que el almacén en memoria persista mientras la aplicación está viva. Los tests
pueden sustituir esta dependencia con ``app.dependency_overrides``.

El cliente de OpenAI se construye de forma perezosa: la aplicación arranca aunque
no haya ``LLM_API_KEY`` configurada, y el error por clave ausente solo se produce
al intentar generar una respuesta (no al subir documentos ni en /health).
"""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.answer_generator import AnswerGenerator
from app.services.embeddings_client import OpenAIEmbeddingsClient
from app.services.embeddings_retriever import EmbeddingsRetriever
from app.services.llm_client import OpenAILLMClient
from app.services.rag_pipeline import RAGPipeline
from app.services.retriever import Retriever, TfidfRetriever
from app.store.document_store import DocumentStore


class LazyOpenAILLMClient:
    """Cliente LLM que difiere la creación del cliente real hasta la primera llamada."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: OpenAILLMClient | None = None

    def complete(self, *, system: str, user: str) -> str:
        if self._client is None:
            self._client = OpenAILLMClient(
                api_key=self._settings.llm_api_key,
                model=self._settings.llm_model,
            )
        return self._client.complete(system=system, user=user)


class LazyOpenAIEmbeddingsClient:
    """Cliente de embeddings que difiere la creación del cliente real hasta usarlo."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: OpenAIEmbeddingsClient | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            self._client = OpenAIEmbeddingsClient(
                api_key=self._settings.llm_api_key,
                model=self._settings.embedding_model,
            )
        return self._client.embed(texts)


def build_retriever(settings: Settings) -> Retriever:
    """Construye el recuperador según el modo configurado (TF-IDF o embeddings)."""
    if settings.retrieval_mode == "embeddings":
        return EmbeddingsRetriever(LazyOpenAIEmbeddingsClient(settings))
    return TfidfRetriever()


@lru_cache
def get_pipeline() -> RAGPipeline:
    """Devuelve la tubería RAG compartida (creada una sola vez)."""
    settings = get_settings()
    return RAGPipeline(
        store=DocumentStore(),
        retriever=build_retriever(settings),
        answer_generator=AnswerGenerator(LazyOpenAILLMClient(settings)),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        min_relevance_score=settings.min_relevance_score,
    )
