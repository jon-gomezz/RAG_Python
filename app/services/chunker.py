"""Fragmentación (chunking) de texto en trozos con solapamiento.

Convierte un texto largo en una lista de fragmentos (`Chunk`) de tamaño
acotado. Cada fragmento conserva metadatos (documento de origen, índice y
posición en el texto original) para garantizar la trazabilidad: cuando la
API devuelva una respuesta podrá señalar exactamente de dónde salió.

El troceado es por número de caracteres (no por palabras ni tokens) por ser
determinista y fácil de testear, en línea con el enfoque del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """Un fragmento de texto con sus metadatos de trazabilidad.

    Attributes:
        text: contenido textual del fragmento.
        index: posición ordinal del fragmento dentro del documento (0, 1, 2...).
        start: índice de carácter donde empieza el fragmento en el texto original.
        end: índice de carácter donde termina (exclusivo) en el texto original.
        source: identificador del documento de origen (p.ej. el nombre del fichero).
    """

    text: str
    index: int
    start: int
    end: int
    source: str | None = None


def chunk_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    source: str | None = None,
) -> list[Chunk]:
    """Divide un texto en fragmentos solapados.

    Args:
        text: texto a fragmentar.
        chunk_size: tamaño máximo de cada fragmento, en caracteres.
        chunk_overlap: número de caracteres que cada fragmento comparte con el
            anterior. Debe ser menor que ``chunk_size``.
        source: identificador del documento de origen, propagado a cada chunk.

    Returns:
        Lista de :class:`Chunk` en orden. Lista vacía si el texto está vacío
        (o solo tiene espacios).

    Raises:
        ValueError: si ``chunk_size`` no es positivo o si ``chunk_overlap`` es
            negativo o mayor/igual que ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser un entero positivo.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap no puede ser negativo.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size.")

    text = text.strip()
    if not text:
        return []

    # El avance entre el inicio de un fragmento y el siguiente: el tamaño del
    # fragmento menos el solapamiento. Garantizado > 0 por las validaciones.
    step = chunk_size - chunk_overlap

    chunks: list[Chunk] = []
    index = 0
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        fragment = text[start:end]
        chunks.append(Chunk(text=fragment, index=index, start=start, end=end, source=source))
        if end == length:
            break
        index += 1
        start += step

    return chunks


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Elimina fragmentos con texto idéntico, conservando el primero.

    Evita que contenido repetido (cabeceras, índices, pies de página que se
    repiten entre páginas) ocupe varias posiciones en la recuperación. Mantiene
    el orden original y los metadatos de los fragmentos conservados.
    """
    vistos: set[str] = set()
    unicos: list[Chunk] = []
    for chunk in chunks:
        if chunk.text in vistos:
            continue
        vistos.add(chunk.text)
        unicos.append(chunk)
    return unicos
