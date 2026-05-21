"""Aplicación FastAPI: expone la subida de documentos y las preguntas.

Los endpoints son deliberadamente delgados: validan/serializan con los esquemas
Pydantic y delegan toda la lógica en la tubería RAG inyectada. El manejo de
errores se afina en una fase posterior; aquí se traducen los errores de dominio
conocidos a respuestas HTTP claras.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status

from app.core.exceptions import RAGError, UnsupportedFileTypeError
from app.dependencies import get_pipeline
from app.schemas import (
    AskRequest,
    AskResponse,
    DocumentIngested,
    HealthResponse,
    SourceChunk,
    UploadResponse,
)
from app.services.rag_pipeline import RAGPipeline

app = FastAPI(
    title="RAG — Preguntas sobre documentos",
    description="Responde preguntas usando únicamente el contenido de los documentos subidos.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Comprobación de salud del servicio."""
    return HealthResponse()


@app.post("/documents/upload", response_model=UploadResponse)
def upload_documents(
    files: list[UploadFile] = File(...),
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> UploadResponse:
    """Sube uno o varios documentos (.txt, .md, .pdf) y los indexa."""
    ingeridos: list[DocumentIngested] = []
    for upload in files:
        content = upload.file.read()
        try:
            resultado = pipeline.ingest_document(upload.filename or "", content)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except RAGError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        ingeridos.append(
            DocumentIngested(
                filename=resultado.source, chunks_created=resultado.chunks_created
            )
        )

    return UploadResponse(
        documents=ingeridos,
        total_chunks=sum(d.chunks_created for d in ingeridos),
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
) -> AskResponse:
    """Responde a una pregunta usando solo el contenido indexado."""
    resultado = pipeline.ask(request.question)
    return AskResponse(
        answer=resultado.answer,
        status=resultado.status,
        sources=[
            SourceChunk(
                text=r.chunk.text,
                source=r.chunk.source,
                index=r.chunk.index,
                score=r.score,
            )
            for r in resultado.sources
        ],
    )
