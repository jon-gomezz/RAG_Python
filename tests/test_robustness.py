"""Tests de robustez: manejo centralizado de errores y casos límite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import LLMGenerationError, MissingAPIKeyError
from app.dependencies import get_pipeline
from app.main import app
from app.services.answer_generator import AnswerGenerator
from app.services.rag_pipeline import RAGPipeline
from app.services.retriever import TfidfRetriever
from app.store.document_store import DocumentStore


class RaisingLLMClient:
    """Cliente LLM que lanza una excepción concreta al ser invocado."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def complete(self, *, system: str, user: str) -> str:
        raise self._exc


def _client_with_llm(llm) -> TestClient:
    pipeline = RAGPipeline(
        store=DocumentStore(),
        retriever=TfidfRetriever(),
        answer_generator=AnswerGenerator(llm),
        chunk_size=100,
        chunk_overlap=20,
        top_k=3,
        min_relevance_score=0.05,
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    app.dependency_overrides.clear()


def _subir_doc(client: TestClient) -> None:
    files = {
        "files": (
            "doc.txt",
            b"Los planetas del sistema solar giran alrededor del sol.",
            "text/plain",
        )
    }
    client.post("/documents/upload", files=files)


# --- clave de API ausente -> 503 ------------------------------------------


def test_clave_ausente_devuelve_503():
    client = _client_with_llm(RaisingLLMClient(MissingAPIKeyError()))
    _subir_doc(client)
    resp = client.post("/ask", json={"question": "Cuentame sobre los planetas y el sol"})
    assert resp.status_code == 503
    assert "LLM_API_KEY" in resp.json()["detail"]


# --- fallo del proveedor LLM -> 502 ---------------------------------------


def test_fallo_del_llm_devuelve_502():
    client = _client_with_llm(RaisingLLMClient(RuntimeError("timeout de red")))
    _subir_doc(client)
    resp = client.post("/ask", json={"question": "Cuentame sobre los planetas y el sol"})
    assert resp.status_code == 502
    assert "LLM" in resp.json()["detail"]


# --- error de generación es un RAGError (capa de servicio) -----------------


def test_generador_envuelve_errores_inesperados():
    from app.services.chunker import Chunk
    from app.services.retriever import RetrievalResult

    generator = AnswerGenerator(RaisingLLMClient(RuntimeError("boom")))
    results = [RetrievalResult(chunk=Chunk("t", 0, 0, 1, "d.txt"), score=0.5)]
    with pytest.raises(LLMGenerationError):
        generator.generate("pregunta", results)


# --- sin ficheros -> 422 (validación de FastAPI) ---------------------------


def test_subida_sin_ficheros_devuelve_422():
    client = _client_with_llm(RaisingLLMClient(RuntimeError()))
    # No se envía el campo 'files': FastAPI lo rechaza por validación.
    resp = client.post("/documents/upload")
    assert resp.status_code == 422


# --- el cuerpo de error es uniforme ---------------------------------------


def test_error_tiene_formato_detail():
    client = _client_with_llm(RaisingLLMClient(RuntimeError()))
    files = {"files": ("imagen.png", b"datos", "image/png")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 400
    assert "detail" in resp.json()
