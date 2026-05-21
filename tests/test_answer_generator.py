"""Tests para la generación de respuestas fundamentadas."""

from __future__ import annotations

import pytest

from app.services.answer_generator import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    AnswerGenerator,
    build_context,
    build_user_prompt,
)
from app.services.chunker import Chunk
from app.services.llm_client import MissingAPIKeyError, OpenAILLMClient
from app.services.retriever import RetrievalResult


class FakeLLMClient:
    """Doble de prueba: registra las llamadas y devuelve una respuesta fija."""

    def __init__(self, response: str = "respuesta del modelo") -> None:
        self.response = response
        self.calls: list[dict[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        return self.response


def _result(text: str, source: str = "doc.txt", index: int = 0, score: float = 0.9):
    chunk = Chunk(text=text, index=index, start=0, end=len(text), source=source)
    return RetrievalResult(chunk=chunk, score=score)


# --- contexto insuficiente -------------------------------------------------


def test_sin_resultados_devuelve_mensaje_insuficiente_sin_llamar_al_llm():
    client = FakeLLMClient()
    generator = AnswerGenerator(client)

    resultado = generator.generate("¿Cuál es la capital?", [])

    assert resultado.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert resultado.has_sufficient_context is False
    assert resultado.sources == []
    # Lo más importante: NO se llamó al LLM.
    assert client.calls == []


# --- generación normal -----------------------------------------------------


def test_con_resultados_llama_al_llm_y_devuelve_respuesta():
    client = FakeLLMClient(response="  París es la capital.  ")
    generator = AnswerGenerator(client)
    results = [_result("La capital de Francia es París.")]

    resultado = generator.generate("¿Capital de Francia?", results)

    assert resultado.answer == "París es la capital."  # recortado
    assert resultado.has_sufficient_context is True
    assert resultado.sources == results
    assert len(client.calls) == 1


def test_el_prompt_incluye_la_pregunta_y_el_contexto():
    client = FakeLLMClient()
    generator = AnswerGenerator(client)
    results = [_result("El cielo es azul por la dispersión de la luz.")]

    generator.generate("¿Por qué el cielo es azul?", results)

    enviado = client.calls[0]["user"]
    assert "¿Por qué el cielo es azul?" in enviado
    assert "dispersión de la luz" in enviado


# --- construcción del contexto ---------------------------------------------


def test_build_context_numera_e_incluye_fuente():
    results = [
        _result("Texto uno", source="a.txt", index=0),
        _result("Texto dos", source="b.md", index=3),
    ]
    contexto = build_context(results)
    assert "Fragmento 1" in contexto
    assert "Fragmento 2" in contexto
    assert "a.txt" in contexto
    assert "b.md" in contexto
    assert "Texto uno" in contexto


def test_build_user_prompt_contiene_contexto_y_pregunta():
    results = [_result("contenido relevante")]
    prompt = build_user_prompt("mi pregunta", results)
    assert "contenido relevante" in prompt
    assert "mi pregunta" in prompt


# --- cliente OpenAI ---------------------------------------------------------


def test_cliente_openai_sin_clave_lanza_error():
    with pytest.raises(MissingAPIKeyError):
        OpenAILLMClient(api_key="", model="gpt-4o-mini")
