"""Tests de la API (FastAPI TestClient) con la tubería inyectada y LLM simulado."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_pipeline
from app.main import app
from app.services.answer_generator import AnswerGenerator
from app.services.rag_pipeline import RAGPipeline
from app.services.retriever import TfidfRetriever
from app.store.document_store import DocumentStore


class FakeLLMClient:
    def __init__(self, response: str = "respuesta simulada") -> None:
        self.response = response

    def complete(self, *, system: str, user: str) -> str:
        return self.response


@pytest.fixture
def client() -> TestClient:
    """Cliente de prueba con una tubería fresca (LLM simulado) por test."""
    pipeline = RAGPipeline(
        store=DocumentStore(),
        retriever=TfidfRetriever(),
        answer_generator=AnswerGenerator(FakeLLMClient()),
        chunk_size=100,
        chunk_overlap=20,
        top_k=3,
        min_relevance_score=0.05,
    )
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- /health ---------------------------------------------------------------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- /documents/upload -----------------------------------------------------


def test_subida_txt_ok(client):
    files = {"files": ("doc.txt", b"Los delfines son mamiferos marinos muy inteligentes.", "text/plain")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["documents"][0]["filename"] == "doc.txt"
    assert body["total_chunks"] >= 1


def test_subida_extension_no_soportada_devuelve_400(client):
    files = {"files": ("imagen.png", b"datos binarios", "image/png")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 400


def test_subida_fichero_vacio_devuelve_422(client):
    files = {"files": ("vacio.txt", b"", "text/plain")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 422


# --- /ask ------------------------------------------------------------------


def test_preguntar_sin_documentos(client):
    resp = client.post("/ask", json={"question": "¿Algo?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "no_documents"
    assert body["sources"] == []


def test_flujo_subir_y_preguntar(client):
    files = {"files": ("doc.txt", b"La Torre Eiffel esta en Paris y mide 330 metros.", "text/plain")}
    client.post("/documents/upload", files=files)

    resp = client.post("/ask", json={"question": "¿Donde esta la Torre Eiffel?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["answer"] == "respuesta simulada"
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["source"] == "doc.txt"


def test_pregunta_irrelevante_devuelve_contexto_insuficiente(client):
    files = {"files": ("doc.txt", b"El cafe es una bebida popular en todo el mundo.", "text/plain")}
    client.post("/documents/upload", files=files)

    resp = client.post("/ask", json={"question": "ecuaciones diferenciales estocasticas"})
    body = resp.json()
    assert body["status"] == "insufficient_context"
    assert body["sources"] == []


def test_pregunta_vacia_devuelve_422(client):
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422  # validación Pydantic (min_length=1)
