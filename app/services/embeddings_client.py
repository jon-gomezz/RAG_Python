"""Cliente de embeddings basado en la API de OpenAI.

Define la interfaz mínima :class:`EmbeddingsClient` (inyectable, para poder usar
dobles de prueba) y una implementación concreta sobre OpenAI. El import de
``openai`` es perezoso para no exigir la librería ni una clave salvo al usarlo.
"""

from __future__ import annotations

from typing import Protocol

from app.core.exceptions import MissingAPIKeyError


class EmbeddingsClient(Protocol):
    """Convierte una lista de textos en sus vectores (embeddings)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Devuelve un vector por cada texto de entrada."""
        ...


class OpenAIEmbeddingsClient:
    """Cliente de embeddings que usa la API de OpenAI."""

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise MissingAPIKeyError()
        self._api_key = api_key
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI  # import perezoso

        client = OpenAI(api_key=self._api_key)
        response = client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]
