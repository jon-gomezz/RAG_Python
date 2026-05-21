"""Esquemas Pydantic de petición y respuesta de la API.

Definir tipos explícitos para entradas y salidas documenta el contrato de la API,
valida los datos automáticamente y mantiene los endpoints delgados (la conversión
a/desde JSON la gestiona FastAPI a partir de estos modelos).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.rag_pipeline import AnswerStatus


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud."""

    status: str = "ok"


class DocumentIngested(BaseModel):
    """Resumen de un documento ingerido."""

    filename: str
    chunks_created: int


class UploadResponse(BaseModel):
    """Respuesta de la subida de documentos."""

    documents: list[DocumentIngested]
    total_chunks: int


class DocumentInfo(BaseModel):
    """Información de un documento almacenado."""

    filename: str
    chunks: int


class DocumentsListResponse(BaseModel):
    """Listado de los documentos indexados."""

    documents: list[DocumentInfo]
    total_documents: int
    total_chunks: int


class DeleteResponse(BaseModel):
    """Confirmación del borrado de todos los documentos."""

    detail: str
    documents_removed: int


class AskRequest(BaseModel):
    """Petición para hacer una pregunta."""

    question: str = Field(..., min_length=1, description="Pregunta del usuario.")


class SourceChunk(BaseModel):
    """Fragmento fuente devuelto junto a la respuesta (trazabilidad)."""

    text: str
    source: str | None
    index: int
    score: float


class AskResponse(BaseModel):
    """Respuesta a una pregunta, con su estado y las fuentes utilizadas."""

    answer: str
    status: AnswerStatus
    sources: list[SourceChunk]
