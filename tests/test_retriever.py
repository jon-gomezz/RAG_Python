"""Tests para la recuperación TF-IDF + similitud del coseno."""

from __future__ import annotations

import pytest

from app.services.chunker import Chunk
from app.services.retriever import RetrievalResult, TfidfRetriever


def _chunk(text: str, index: int) -> Chunk:
    return Chunk(text=text, index=index, start=0, end=len(text), source="doc.txt")


@pytest.fixture
def retriever() -> TfidfRetriever:
    chunks = [
        _chunk("Los gatos son mamíferos felinos domésticos.", 0),
        _chunk("Python es un lenguaje de programación popular.", 1),
        _chunk("La fotosíntesis ocurre en las plantas verdes.", 2),
    ]
    r = TfidfRetriever()
    r.index(chunks)
    return r


# --- recuperación básica ---------------------------------------------------


def test_recupera_el_fragmento_mas_relevante(retriever):
    resultados = retriever.retrieve("¿Qué es Python?", top_k=1, min_score=0.0)
    assert len(resultados) == 1
    assert "programación" in resultados[0].chunk.text


def test_resultados_ordenados_por_puntuacion_descendente(retriever):
    resultados = retriever.retrieve("plantas y fotosíntesis", top_k=3, min_score=0.0)
    puntuaciones = [r.score for r in resultados]
    assert puntuaciones == sorted(puntuaciones, reverse=True)


def test_top_k_limita_el_numero_de_resultados(retriever):
    resultados = retriever.retrieve("gatos plantas Python", top_k=2, min_score=0.0)
    assert len(resultados) <= 2


def test_min_score_filtra_resultados_irrelevantes(retriever):
    # Una pregunta sin relación no debe devolver fragmentos por encima del umbral.
    resultados = retriever.retrieve("automóviles deportivos", top_k=3, min_score=0.1)
    assert resultados == []


def test_devuelve_tipo_retrieval_result(retriever):
    resultados = retriever.retrieve("gatos", top_k=1, min_score=0.0)
    assert isinstance(resultados[0], RetrievalResult)
    assert 0.0 <= resultados[0].score <= 1.0


# --- casos límite ----------------------------------------------------------


def test_indice_vacio_devuelve_lista_vacia():
    r = TfidfRetriever()
    r.index([])
    assert r.retrieve("cualquier cosa", top_k=3, min_score=0.0) == []


def test_sin_indexar_devuelve_lista_vacia():
    r = TfidfRetriever()
    assert r.retrieve("cualquier cosa", top_k=3, min_score=0.0) == []


def test_pregunta_vacia_devuelve_lista_vacia(retriever):
    assert retriever.retrieve("", top_k=3, min_score=0.0) == []
    assert retriever.retrieve("   ", top_k=3, min_score=0.0) == []


def test_top_k_no_positivo_devuelve_lista_vacia(retriever):
    assert retriever.retrieve("Python", top_k=0, min_score=0.0) == []
