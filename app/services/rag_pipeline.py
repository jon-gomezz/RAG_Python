"""Orquestación de la tubería RAG de extremo a extremo.

Conecta las piezas ya construidas en una sola fachada:

- **Ingesta de un documento**: cargar texto -> fragmentar -> guardar -> reindexar.
- **Pregunta**: recuperar fragmentos relevantes -> generar respuesta fundamentada.

Devuelve siempre un resultado con la respuesta, un estado explícito y las fuentes
utilizadas, de modo que la capa de API (fase siguiente) solo tenga que serializarlo.
Las dependencias (almacén, recuperador, generador) se inyectan para poder testear
la orquestación con dobles de prueba.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.services.answer_generator import AnswerGenerator
from app.services.chunker import chunk_text
from app.services.loaders import load_document
from app.services.retriever import RetrievalResult, TfidfRetriever
from app.store.document_store import DocumentStore

# Mensaje cuando se pregunta sin haber subido ningún documento.
NO_DOCUMENTS_MESSAGE = (
    "Todavía no se ha subido ningún documento. Sube al menos un documento "
    "antes de hacer preguntas."
)


class AnswerStatus(str, Enum):
    """Estado del resultado de una pregunta."""

    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    NO_DOCUMENTS = "no_documents"


@dataclass(frozen=True)
class IngestResult:
    """Resultado de ingerir un documento."""

    source: str
    chunks_created: int


@dataclass(frozen=True)
class AskResult:
    """Resultado de responder a una pregunta."""

    answer: str
    status: AnswerStatus
    sources: list[RetrievalResult]


class RAGPipeline:
    """Fachada que orquesta ingesta de documentos y respuesta a preguntas."""

    def __init__(
        self,
        *,
        store: DocumentStore,
        retriever: TfidfRetriever,
        answer_generator: AnswerGenerator,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
        min_relevance_score: float,
    ) -> None:
        self._store = store
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k
        self._min_relevance_score = min_relevance_score

    def ingest_document(self, filename: str, content: bytes) -> IngestResult:
        """Carga, fragmenta y almacena un documento, y reindexa el recuperador.

        Raises:
            UnsupportedFileTypeError / EmptyFileError / DocumentLoadError:
                propagadas desde la carga del documento.
        """
        text = load_document(filename, content)
        chunks = chunk_text(
            text,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            source=filename,
        )
        self._store.add_document(filename, chunks)
        # Reconstruimos el índice con todos los fragmentos acumulados.
        self._retriever.index(self._store.chunks)
        return IngestResult(source=filename, chunks_created=len(chunks))

    def ask(self, question: str) -> AskResult:
        """Responde a una pregunta usando solo el contenido almacenado."""
        if self._store.is_empty():
            return AskResult(
                answer=NO_DOCUMENTS_MESSAGE,
                status=AnswerStatus.NO_DOCUMENTS,
                sources=[],
            )

        results = self._retriever.retrieve(
            question,
            top_k=self._top_k,
            min_score=self._min_relevance_score,
        )
        generated = self._answer_generator.generate(question, results)
        status = (
            AnswerStatus.ANSWERED
            if generated.has_sufficient_context
            else AnswerStatus.INSUFFICIENT_CONTEXT
        )
        return AskResult(
            answer=generated.answer,
            status=status,
            sources=generated.sources,
        )
