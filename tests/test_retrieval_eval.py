"""Evaluación de recuperación: comprueba que cada pregunta recupera el fragmento esperado.

Es un test de "calidad" del recuperador sobre un pequeño corpus etiquetado: para
varias preguntas conocemos cuál es el fragmento correcto y verificamos que sale
como primer resultado. Sirve de red de seguridad ante regresiones en la
recuperación.
"""

from __future__ import annotations

from app.services.chunker import Chunk
from app.services.retriever import TfidfRetriever

# Corpus etiquetado: cada fragmento trata de un tema distinto.
_CORPUS = [
    "El sistema solar tiene ocho planetas que orbitan alrededor del Sol.",
    "Python es un lenguaje de programacion interpretado y de alto nivel.",
    "La fotosintesis es el proceso por el que las plantas producen energia.",
    "El agua hierve a cien grados centigrados al nivel del mar.",
    "La Torre Eiffel se encuentra en Paris y mide unos trescientos metros.",
]

# Pregunta -> texto del fragmento que debe recuperarse el primero.
_CASOS = [
    ("¿Cuantos planetas hay en el sistema solar?", _CORPUS[0]),
    ("¿Que es Python?", _CORPUS[1]),
    ("¿Como producen energia las plantas?", _CORPUS[2]),
    ("¿A que temperatura hierve el agua?", _CORPUS[3]),
    ("¿Donde esta la Torre Eiffel?", _CORPUS[4]),
]


def _build_retriever() -> TfidfRetriever:
    chunks = [
        Chunk(text=t, index=i, start=0, end=len(t), source="corpus") for i, t in enumerate(_CORPUS)
    ]
    retriever = TfidfRetriever()
    retriever.index(chunks)
    return retriever


def test_cada_pregunta_recupera_su_fragmento_como_primero():
    retriever = _build_retriever()
    for pregunta, esperado in _CASOS:
        resultados = retriever.retrieve(pregunta, top_k=3, min_score=0.0)
        assert resultados, f"sin resultados para: {pregunta}"
        assert resultados[0].chunk.text == esperado, f"falló para: {pregunta}"


def test_precision_global_del_top_1():
    """La precisión top-1 sobre el corpus etiquetado debe ser perfecta aquí."""
    retriever = _build_retriever()
    aciertos = 0
    for pregunta, esperado in _CASOS:
        resultados = retriever.retrieve(pregunta, top_k=1, min_score=0.0)
        if resultados and resultados[0].chunk.text == esperado:
            aciertos += 1
    assert aciertos == len(_CASOS)
