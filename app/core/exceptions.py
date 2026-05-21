"""Excepciones compartidas del dominio de la aplicación.

Estas excepciones representan errores de negocio esperables (extensión no
soportada, fichero vacío, fallo al extraer texto). La capa de API las traduce
a respuestas HTTP claras; la lógica de negocio nunca debe lanzar errores
genéricos sin contexto.
"""

from __future__ import annotations


class RAGError(Exception):
    """Excepción base para todos los errores de dominio de la aplicación."""


class UnsupportedFileTypeError(RAGError):
    """La extensión del fichero no está soportada (solo .txt, .md, .pdf)."""

    def __init__(self, extension: str) -> None:
        self.extension = extension
        super().__init__(
            f"Extensión no soportada: '{extension}'. "
            "Solo se admiten ficheros .txt, .md y .pdf."
        )


class EmptyFileError(RAGError):
    """El fichero no contiene texto extraíble (está vacío o en blanco)."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        super().__init__(
            f"El fichero '{filename}' está vacío o no contiene texto extraíble."
        )


class DocumentLoadError(RAGError):
    """Fallo inesperado al extraer el texto de un documento."""

    def __init__(self, filename: str, reason: str) -> None:
        self.filename = filename
        self.reason = reason
        super().__init__(
            f"No se pudo procesar el fichero '{filename}': {reason}"
        )


class MissingAPIKeyError(RAGError):
    """No se ha configurado la clave de API del LLM (LLM_API_KEY)."""

    def __init__(self) -> None:
        super().__init__(
            "Falta la clave de API del LLM. Define la variable de entorno "
            "LLM_API_KEY para poder generar respuestas."
        )


class LLMGenerationError(RAGError):
    """Fallo al generar la respuesta con el proveedor LLM (red, API, etc.)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Error al generar la respuesta con el LLM: {reason}")
