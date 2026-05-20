"""Tests de la tubería RAG de extremo a extremo (con LLM simulado)."""

from __future__ import annotations

from app.services.answer_generator import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    AnswerGenerator,
)
from app.services.rag_pipeline import (
    NO_DOCUMENTS_MESSAGE,
    AnswerStatus,
    RAGPipeline,
)
from app.services.retriever import TfidfRetriever
from app.store.document_store import DocumentStore

import pytest


class FakeLLMClient:
    """Doble de prueba que devuelve una respuesta fija y registra las llamadas."""

    def __init__(self, response: str = "respuesta generada") -> None:
        self.response = response
        self.calls = 0

    def complete(self, *, system: str, user: str) -> str:
        self.calls += 1
        return self.response


def _make_pipeline(llm: FakeLLMClient) -> RAGPipeline:
    return RAGPipeline(
        store=DocumentStore(),
        retriever=TfidfRetriever(),
        answer_generator=AnswerGenerator(llm),
        chunk_size=50,
        chunk_overlap=10,
        top_k=3,
        min_relevance_score=0.05,
    )


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient()


# --- ingesta ---------------------------------------------------------------


def test_ingesta_crea_fragmentos(llm):
    pipeline = _make_pipeline(llm)
    resultado = pipeline.ingest_document(
        "doc.txt", b"Python es un lenguaje de programacion muy popular y versatil."
    )
    assert resultado.source == "doc.txt"
    assert resultado.chunks_created >= 1


# --- preguntar sin documentos ---------------------------------------------


def test_preguntar_sin_documentos_devuelve_estado_no_documents(llm):
    pipeline = _make_pipeline(llm)
    resultado = pipeline.ask("¿Algo?")
    assert resultado.status == AnswerStatus.NO_DOCUMENTS
    assert resultado.answer == NO_DOCUMENTS_MESSAGE
    assert resultado.sources == []
    assert llm.calls == 0  # no se llama al LLM


# --- flujo completo: pregunta respondible ----------------------------------


def test_flujo_completo_responde_y_devuelve_fuentes(llm):
    pipeline = _make_pipeline(llm)
    pipeline.ingest_document(
        "animales.txt",
        b"Los gatos son felinos. Los perros son canidos. Las aguilas son aves rapaces.",
    )
    resultado = pipeline.ask("Cuentame sobre los gatos")

    assert resultado.status == AnswerStatus.ANSWERED
    assert resultado.answer == "respuesta generada"
    assert len(resultado.sources) >= 1
    assert llm.calls == 1


# --- pregunta sin relacion -> contexto insuficiente ------------------------


def test_pregunta_irrelevante_devuelve_contexto_insuficiente(llm):
    pipeline = _make_pipeline(llm)
    pipeline.ingest_document(
        "cocina.txt",
        b"La paella es un plato tradicional valenciano elaborado con arroz.",
    )
    resultado = pipeline.ask("teoria cuantica de campos relativista")

    assert resultado.status == AnswerStatus.INSUFFICIENT_CONTEXT
    assert resultado.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert resultado.sources == []
    assert llm.calls == 0  # no se llama al LLM si no hay contexto


# --- varios documentos -----------------------------------------------------


def test_varios_documentos_se_indexan_juntos(llm):
    pipeline = _make_pipeline(llm)
    pipeline.ingest_document("a.txt", b"El sol es una estrella enana amarilla.")
    pipeline.ingest_document("b.txt", b"La luna es el satelite natural de la Tierra.")

    resultado = pipeline.ask("Hablame de la luna y su orbita")
    assert resultado.status == AnswerStatus.ANSWERED
    # la fuente recuperada debe provenir del documento sobre la luna
    assert any("luna" in r.chunk.text.lower() for r in resultado.sources)
