"""Tests de la factoría de recuperador según el modo configurado."""

from __future__ import annotations

from app.core.config import Settings
from app.dependencies import build_retriever
from app.services.embeddings_retriever import EmbeddingsRetriever
from app.services.retriever import TfidfRetriever


def test_modo_tfidf_construye_tfidf():
    settings = Settings(_env_file=None, retrieval_mode="tfidf")
    assert isinstance(build_retriever(settings), TfidfRetriever)


def test_modo_embeddings_construye_embeddings():
    # No se necesita clave: el cliente real solo se crea al usar embed().
    settings = Settings(_env_file=None, retrieval_mode="embeddings")
    assert isinstance(build_retriever(settings), EmbeddingsRetriever)
