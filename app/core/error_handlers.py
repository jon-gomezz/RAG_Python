"""Manejadores centralizados de excepciones de dominio para la API.

Traduce cada excepción de dominio (`RAGError` y subclases) a una respuesta HTTP
con el código de estado adecuado y un cuerpo JSON uniforme ``{"detail": ...}``.
Centralizar esto mantiene los endpoints delgados (no necesitan `try/except`) y
garantiza mensajes de error coherentes en toda la API.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    DocumentLoadError,
    EmptyFileError,
    LLMGenerationError,
    MissingAPIKeyError,
    RAGError,
    UnsupportedFileTypeError,
)

# Mapa excepción -> código HTTP. Las subclases no listadas usan el fallback RAGError.
_STATUS_BY_EXCEPTION: list[tuple[type[RAGError], int]] = [
    (UnsupportedFileTypeError, 400),  # petición inválida del cliente
    (EmptyFileError, 422),            # entrada no procesable
    (DocumentLoadError, 422),
    (MissingAPIKeyError, 503),        # servicio mal configurado
    (LLMGenerationError, 502),        # fallo del proveedor externo
]


def _make_handler(status_code: int):
    async def handler(_request: Request, exc: RAGError) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return handler


def register_exception_handlers(app: FastAPI) -> None:
    """Registra en la app los manejadores de las excepciones de dominio."""
    for exception_type, status_code in _STATUS_BY_EXCEPTION:
        app.add_exception_handler(exception_type, _make_handler(status_code))
    # Fallback: cualquier RAGError no contemplado arriba -> 500.
    app.add_exception_handler(RAGError, _make_handler(500))
