"""Recuperación de fragmentos relevantes mediante TF-IDF + similitud del coseno.

Construye una representación numérica ("huella" de palabras clave) de cada
fragmento usando TF-IDF, y ante una pregunta devuelve los fragmentos cuya huella
más se parece a la de la pregunta, medido con la similitud del coseno.

Se eligió TF-IDF por ser determinista, sin dependencias externas (no requiere
llamadas a un modelo de embeddings) y fácil de testear, suficiente para el
alcance del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.chunker import Chunk


@dataclass(frozen=True)
class RetrievalResult:
    """Un fragmento recuperado junto con su puntuación de relevancia.

    Attributes:
        chunk: el fragmento recuperado.
        score: similitud del coseno con la pregunta, en el rango [0, 1].
    """

    chunk: Chunk
    score: float


class TfidfRetriever:
    """Indexa fragmentos con TF-IDF y recupera los más relevantes por coseno."""

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None  # matriz dispersa TF-IDF de los fragmentos indexados
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        """Construye el índice TF-IDF a partir de los fragmentos dados.

        Reentrena el vocabulario desde cero con los fragmentos recibidos. Si la
        lista está vacía, el índice queda vacío y :meth:`retrieve` devolverá
        siempre una lista vacía.
        """
        self._chunks = list(chunks)
        if not self._chunks:
            self._vectorizer = None
            self._matrix = None
            return

        self._vectorizer = TfidfVectorizer()
        self._matrix = self._vectorizer.fit_transform(c.text for c in self._chunks)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        min_score: float,
    ) -> list[RetrievalResult]:
        """Devuelve los fragmentos más relevantes para una pregunta.

        Args:
            query: pregunta del usuario.
            top_k: número máximo de fragmentos a devolver.
            min_score: puntuación mínima de coseno para considerar un fragmento.

        Returns:
            Lista de :class:`RetrievalResult` ordenada de mayor a menor
            puntuación. Vacía si no hay índice, la pregunta está vacía, o ningún
            fragmento supera ``min_score``.
        """
        if self._vectorizer is None or self._matrix is None:
            return []
        if not query or not query.strip():
            return []
        if top_k <= 0:
            return []

        query_vector = self._vectorizer.transform([query])
        # Una sola fila (la pregunta) frente a todos los fragmentos.
        scores = cosine_similarity(query_vector, self._matrix)[0]

        results = [
            RetrievalResult(chunk=chunk, score=float(score))
            for chunk, score in zip(self._chunks, scores)
            if score >= min_score
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
