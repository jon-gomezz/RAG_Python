"""Tests del recuperador semántico con un cliente de embeddings simulado."""

from __future__ import annotations

import pytest

from app.core.exceptions import EmbeddingError, MissingAPIKeyError
from app.services.chunker import Chunk
from app.services.embeddings_client import OpenAIEmbeddingsClient
from app.services.embeddings_retriever import EmbeddingsRetriever
from app.services.retriever import RetrievalResult

# Vocabulario del embedder de prueba: cada dimensión es un "tema".
_VOCAB = ["gato", "python", "planeta"]


class FakeEmbeddingsClient:
    """Embedder determinista: vector de presencia de palabras del vocabulario.

    Añade una dimensión constante para que ningún vector sea nulo (evita NaN en
    el coseno cuando un texto no contiene ninguna palabra del vocabulario).
    """

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectores = []
        for texto in texts:
            t = texto.lower()
            vectores.append([1.0 if palabra in t else 0.0 for palabra in _VOCAB] + [0.01])
        return vectores


def _chunks() -> list[Chunk]:
    textos = [
        "Los gatos son animales domesticos.",
        "Python es un lenguaje de programacion.",
        "Marte es un planeta del sistema solar.",
    ]
    return [Chunk(text=t, index=i, start=0, end=len(t), source="d") for i, t in enumerate(textos)]


@pytest.fixture
def retriever() -> EmbeddingsRetriever:
    r = EmbeddingsRetriever(FakeEmbeddingsClient())
    r.index(_chunks())
    return r


def test_recupera_por_significado(retriever):
    resultados = retriever.retrieve("informacion sobre python", top_k=1, min_score=0.1)
    assert len(resultados) == 1
    assert "Python" in resultados[0].chunk.text


def test_devuelve_retrieval_result(retriever):
    resultados = retriever.retrieve("gato", top_k=1, min_score=0.0)
    assert isinstance(resultados[0], RetrievalResult)


def test_top_k_limita(retriever):
    resultados = retriever.retrieve("gato python planeta", top_k=2, min_score=0.0)
    assert len(resultados) <= 2


def test_indice_vacio_devuelve_vacio():
    r = EmbeddingsRetriever(FakeEmbeddingsClient())
    r.index([])
    assert r.retrieve("gato", top_k=3, min_score=0.0) == []


def test_pregunta_vacia_devuelve_vacio(retriever):
    assert retriever.retrieve("   ", top_k=3, min_score=0.0) == []


def test_top_k_no_positivo_devuelve_vacio(retriever):
    assert retriever.retrieve("gato", top_k=0, min_score=0.0) == []


# --- manejo de errores -----------------------------------------------------


class RaisingEmbeddingsClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise self._exc


def test_error_inesperado_se_envuelve_en_embedding_error():
    r = EmbeddingsRetriever(RaisingEmbeddingsClient(RuntimeError("timeout")))
    with pytest.raises(EmbeddingError):
        r.index(_chunks())


def test_error_de_dominio_se_propaga():
    r = EmbeddingsRetriever(RaisingEmbeddingsClient(MissingAPIKeyError()))
    with pytest.raises(MissingAPIKeyError):
        r.index(_chunks())


# --- cliente OpenAI --------------------------------------------------------


def test_cliente_openai_sin_clave_lanza_error():
    with pytest.raises(MissingAPIKeyError):
        OpenAIEmbeddingsClient(api_key="", model="text-embedding-3-small")
