"""Tests para la configuración leída del entorno."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_valores_por_defecto():
    s = Settings(_env_file=None)
    assert s.chunk_size == 800
    assert s.chunk_overlap == 150
    assert s.top_k == 4
    assert s.min_relevance_score == 0.1
    assert s.llm_model == "gpt-4o-mini"


def test_lee_de_variables_de_entorno(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("LLM_API_KEY", "clave-secreta")
    s = Settings(_env_file=None)
    assert s.chunk_size == 500
    assert s.top_k == 7
    assert s.llm_api_key == "clave-secreta"


def test_rechaza_chunk_size_no_positivo(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_rechaza_min_relevance_fuera_de_rango(monkeypatch):
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "1.5")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
