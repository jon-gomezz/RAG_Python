"""Cliente LLM concreto basado en la API de OpenAI.

Implementa la interfaz :class:`~app.services.answer_generator.LLMClient`. El
import de ``openai`` se hace de forma perezosa (dentro del método) para que el
resto del código y los tests no dependan de la librería ni de tener una clave.
"""

from __future__ import annotations

from app.core.exceptions import RAGError


class MissingAPIKeyError(RAGError):
    """No se ha configurado la clave de API del LLM (LLM_API_KEY)."""

    def __init__(self) -> None:
        super().__init__(
            "Falta la clave de API del LLM. Define la variable de entorno "
            "LLM_API_KEY para poder generar respuestas."
        )


class OpenAILLMClient:
    """Cliente LLM que usa la API de chat de OpenAI."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise MissingAPIKeyError()
        self._api_key = api_key
        self._model = model

    def complete(self, *, system: str, user: str) -> str:
        """Llama al modelo de chat y devuelve el texto de la respuesta."""
        from openai import OpenAI  # import perezoso

        client = OpenAI(api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""
