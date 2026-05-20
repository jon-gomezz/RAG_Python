"""Carga y extracción de texto de documentos .txt, .md y .pdf.

La función pública es :func:`load_document`, que recibe el nombre del fichero y
sus bytes, valida la extensión y devuelve el texto plano extraído. Cada formato
tiene su propio loader interno para mantener la lógica separada y testeable.
"""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader

from app.core.exceptions import (
    DocumentLoadError,
    EmptyFileError,
    UnsupportedFileTypeError,
)

# Extensiones soportadas (en minúsculas, con el punto inicial).
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".md", ".pdf"})


def get_extension(filename: str) -> str:
    """Devuelve la extensión del fichero en minúsculas (p.ej. '.pdf')."""
    return Path(filename).suffix.lower()


def _load_text(content: bytes) -> str:
    """Decodifica bytes de un fichero de texto (.txt / .md) como UTF-8.

    Usa 'replace' para no fallar ante bytes inválidos puntuales; el control de
    fichero vacío se hace después en :func:`load_document`.
    """
    return content.decode("utf-8", errors="replace")


def _load_pdf(content: bytes) -> str:
    """Extrae el texto de un PDF concatenando el texto de cada página."""
    reader = PdfReader(io.BytesIO(content))
    parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(parts)


def load_document(filename: str, content: bytes) -> str:
    """Extrae el texto plano de un documento soportado.

    Args:
        filename: nombre original del fichero (se usa para detectar la extensión).
        content: contenido del fichero en bytes.

    Returns:
        El texto extraído, sin espacios sobrantes al inicio/fin.

    Raises:
        UnsupportedFileTypeError: si la extensión no es .txt, .md o .pdf.
        EmptyFileError: si tras la extracción no queda texto útil.
        DocumentLoadError: si la extracción falla de forma inesperada.
    """
    extension = get_extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(extension)

    try:
        if extension == ".pdf":
            text = _load_pdf(content)
        else:
            text = _load_text(content)
    except (UnsupportedFileTypeError, EmptyFileError):
        raise
    except Exception as exc:  # noqa: BLE001 - se reempaqueta con contexto
        raise DocumentLoadError(filename, str(exc)) from exc

    text = text.strip()
    if not text:
        raise EmptyFileError(filename)

    return text
