"""Almacén en memoria de documentos y sus fragmentos.

Guarda los fragmentos (`Chunk`) generados al subir documentos. Es deliberadamente
simple (una lista en memoria) por el alcance del proyecto: no hay persistencia en
disco ni base de datos. Sirve de única fuente de verdad sobre qué contenido se ha
cargado, y es lo que el recuperador indexa para buscar.
"""

from __future__ import annotations

from app.services.chunker import Chunk


class DocumentStore:
    """Contenedor en memoria de los fragmentos de todos los documentos cargados."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        # Nombre de documento -> número de fragmentos aportados.
        self._documents: dict[str, int] = {}

    def add_document(self, source: str, chunks: list[Chunk]) -> None:
        """Registra un documento y añade sus fragmentos al almacén.

        Args:
            source: identificador del documento (p.ej. el nombre del fichero).
            chunks: fragmentos ya generados para ese documento.
        """
        self._chunks.extend(chunks)
        self._documents[source] = self._documents.get(source, 0) + len(chunks)

    @property
    def chunks(self) -> list[Chunk]:
        """Todos los fragmentos almacenados, en orden de inserción."""
        return list(self._chunks)

    @property
    def documents(self) -> dict[str, int]:
        """Mapa de documento -> número de fragmentos almacenados."""
        return dict(self._documents)

    def is_empty(self) -> bool:
        """True si no hay ningún fragmento almacenado."""
        return not self._chunks

    def chunk_count(self) -> int:
        """Número total de fragmentos almacenados."""
        return len(self._chunks)

    def clear(self) -> None:
        """Vacía por completo el almacén."""
        self._chunks.clear()
        self._documents.clear()
