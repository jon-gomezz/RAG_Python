"""Tests de los endpoints de gestión de documentos (listar y borrar)."""

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
    def complete(self, *, system: str, user: str) -> str:
        return "respuesta simulada"


@pytest.fixture
def client() -> TestClient:
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


def _subir(client: TestClient, nombre: str, texto: bytes) -> None:
    client.post("/documents/upload", files={"files": (nombre, texto, "text/plain")})


# --- GET /documents --------------------------------------------------------


def test_listar_sin_documentos(client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == {"documents": [], "total_documents": 0, "total_chunks": 0}


def test_listar_con_documentos(client):
    _subir(client, "a.txt", b"El agua hierve a cien grados centigrados al nivel del mar.")
    _subir(
        client,
        "b.txt",
        b"La velocidad de la luz es de unos trescientos mil kilometros por segundo.",
    )

    resp = client.get("/documents")
    body = resp.json()
    assert body["total_documents"] == 2
    nombres = {d["filename"] for d in body["documents"]}
    assert nombres == {"a.txt", "b.txt"}
    assert body["total_chunks"] == sum(d["chunks"] for d in body["documents"])


# --- DELETE /documents -----------------------------------------------------


def test_borrar_documentos(client):
    _subir(client, "a.txt", b"Contenido de prueba para borrar despues.")
    resp = client.delete("/documents")
    assert resp.status_code == 200
    assert resp.json()["documents_removed"] == 1

    # Tras borrar, el listado queda vacío.
    assert client.get("/documents").json()["total_documents"] == 0


def test_preguntar_tras_borrar_devuelve_no_documents(client):
    _subir(client, "a.txt", b"La Torre Eiffel se encuentra en Paris, Francia.")
    client.delete("/documents")

    resp = client.post("/ask", json={"question": "¿Donde esta la Torre Eiffel?"})
    assert resp.json()["status"] == "no_documents"


def test_borrar_sin_documentos_no_falla(client):
    resp = client.delete("/documents")
    assert resp.status_code == 200
    assert resp.json()["documents_removed"] == 0
