"""Generación de respuestas fundamentadas a partir del contexto recuperado.

El generador toma la pregunta y los fragmentos recuperados y produce una
respuesta basada *únicamente* en ese contexto. Si no hay contexto suficiente
(la recuperación no devolvió nada), se devuelve una respuesta clara de
"información insuficiente" **sin llegar a llamar al LLM**, ahorrando coste y
evitando respuestas inventadas.

El cliente LLM se inyecta (patrón de inversión de dependencias) para poder
testear el generador con un doble de prueba sin llamadas reales a la API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.services.retriever import RetrievalResult

# Respuesta devuelta cuando no hay contexto suficiente para responder.
INSUFFICIENT_CONTEXT_MESSAGE = (
    "No hay información suficiente en los documentos proporcionados para "
    "responder a esta pregunta."
)

# Instrucción de sistema: fuerza al modelo a ceñirse al contexto.
SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas usando ÚNICAMENTE la información "
    "del contexto proporcionado. No uses conocimiento externo. Si el contexto "
    "no contiene la respuesta, di explícitamente que no hay información "
    "suficiente. Responde en español, de forma clara y concisa."
)


class LLMClient(Protocol):
    """Interfaz mínima de un cliente LLM (permite inyectar dobles en tests)."""

    def complete(self, *, system: str, user: str) -> str:
        """Devuelve la respuesta del modelo dado un mensaje de sistema y de usuario."""
        ...


@dataclass(frozen=True)
class GeneratedAnswer:
    """Resultado de la generación.

    Attributes:
        answer: texto de la respuesta (o el mensaje de información insuficiente).
        has_sufficient_context: False si no hubo contexto y no se llamó al LLM.
        sources: fragmentos usados como contexto (vacío si no hubo).
    """

    answer: str
    has_sufficient_context: bool
    sources: list[RetrievalResult]


def build_context(results: list[RetrievalResult]) -> str:
    """Construye el bloque de contexto numerado a partir de los fragmentos.

    Cada fragmento se etiqueta con su origen e índice para favorecer la
    trazabilidad dentro del propio prompt.
    """
    bloques = []
    for i, result in enumerate(results, start=1):
        origen = result.chunk.source or "desconocido"
        bloques.append(
            f"[Fragmento {i} | fuente: {origen} | parte: {result.chunk.index}]\n"
            f"{result.chunk.text}"
        )
    return "\n\n".join(bloques)


def build_user_prompt(question: str, results: list[RetrievalResult]) -> str:
    """Compone el mensaje de usuario con el contexto y la pregunta."""
    contexto = build_context(results)
    return (
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta: {question}\n\n"
        "Responde usando solo el contexto anterior."
    )


class AnswerGenerator:
    """Genera respuestas fundamentadas usando un cliente LLM inyectado."""

    def __init__(self, client: LLMClient) -> None:
        self._client = client

    def generate(
        self, question: str, results: list[RetrievalResult]
    ) -> GeneratedAnswer:
        """Genera la respuesta a partir de la pregunta y los fragmentos.

        Si ``results`` está vacío, devuelve el mensaje de información insuficiente
        sin llamar al LLM.
        """
        if not results:
            return GeneratedAnswer(
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                has_sufficient_context=False,
                sources=[],
            )

        user_prompt = build_user_prompt(question, results)
        answer = self._client.complete(system=SYSTEM_PROMPT, user=user_prompt)
        return GeneratedAnswer(
            answer=answer.strip(),
            has_sufficient_context=True,
            sources=results,
        )
