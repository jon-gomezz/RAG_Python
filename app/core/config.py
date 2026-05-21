"""Configuración de la aplicación leída desde variables de entorno.

Usa pydantic-settings para cargar y validar la configuración. Los valores se
leen de variables de entorno (o de un fichero `.env` en desarrollo). Centralizar
aquí la configuración evita esparcir `os.environ` por el código y permite validar
tipos y valores por defecto en un único sitio.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros configurables de la aplicación."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Fragmentación (chunking)
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)

    # Recuperación
    top_k: int = Field(default=4, gt=0)
    min_relevance_score: float = Field(default=0.1, ge=0.0, le=1.0)
    # Modo de recuperación: "tfidf" (léxico, por defecto) o "embeddings" (semántico).
    retrieval_mode: Literal["tfidf", "embeddings"] = "tfidf"
    embedding_model: str = Field(default="text-embedding-3-small")

    # LLM (generación de respuestas)
    llm_api_key: str = Field(default="")
    llm_model: str = Field(default="gpt-4o-mini")


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración (cacheada) de la aplicación.

    Se cachea para no releer el entorno en cada acceso; el caché puede limpiarse
    en tests con ``get_settings.cache_clear()``.
    """
    return Settings()
