"""Recuperación semántica mediante embeddings + similitud del coseno.

A diferencia de TF-IDF (que compara palabras literales), los embeddings comparan
el *significado*: textos parecidos en sentido tienen vectores parecidos, aunque
no compartan palabras (incluso entre idiomas). Es más potente, pero requiere un
proveedor externo (con coste y latencia), por eso es un modo opcional.

El cliente de embeddings se inyecta para poder testear sin llamadas reales.
"""

from __future__ import annotations

from sklearn.metrics.pairwise import cosine_similarity

from app.core.exceptions import EmbeddingError, RAGError
from app.services.chunker import Chunk
from app.services.embeddings_client import EmbeddingsClient
from app.services.retriever import RetrievalResult


class EmbeddingsRetriever:
    """Indexa fragmentos por su embedding y recupera los más similares por coseno."""

    def __init__(self, client: EmbeddingsClient) -> None:
        self._client = client
        self._matrix: list[list[float]] | None = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Calcula y almacena el embedding de cada fragmento."""
        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = None
            return
        self._matrix = self._embed([c.text for c in self._chunks])

    def retrieve(self, query: str, *, top_k: int, min_score: float) -> list[RetrievalResult]:
        """Devuelve los fragmentos semánticamente más cercanos a la pregunta."""
        if self._matrix is None:
            return []
        if not query or not query.strip():
            return []
        if top_k <= 0:
            return []

        query_vector = self._embed([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]

        results = [
            RetrievalResult(chunk=chunk, score=float(score))
            for chunk, score in zip(self._chunks, scores, strict=True)
            if score >= min_score
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Llama al cliente envolviendo fallos inesperados en un error de dominio."""
        try:
            return self._client.embed(texts)
        except RAGError:
            # Errores de dominio (p.ej. falta de clave) se propagan tal cual.
            raise
        except Exception as exc:  # noqa: BLE001 - se reempaqueta con contexto
            raise EmbeddingError(str(exc)) from exc
