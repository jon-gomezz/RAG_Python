"""Tests para el almacén en memoria de documentos."""

from __future__ import annotations

from app.services.chunker import Chunk
from app.store.document_store import DocumentStore


def _chunks(source: str, n: int) -> list[Chunk]:
    return [Chunk(text=f"texto {i}", index=i, start=i, end=i + 1, source=source) for i in range(n)]


def test_almacen_nuevo_esta_vacio():
    store = DocumentStore()
    assert store.is_empty()
    assert store.chunk_count() == 0
    assert store.chunks == []
    assert store.documents == {}


def test_add_document_guarda_fragmentos():
    store = DocumentStore()
    store.add_document("a.txt", _chunks("a.txt", 3))
    assert not store.is_empty()
    assert store.chunk_count() == 3
    assert store.documents == {"a.txt": 3}


def test_add_varios_documentos_acumula():
    store = DocumentStore()
    store.add_document("a.txt", _chunks("a.txt", 2))
    store.add_document("b.md", _chunks("b.md", 4))
    assert store.chunk_count() == 6
    assert store.documents == {"a.txt": 2, "b.md": 4}


def test_chunks_devuelve_copia_no_referencia():
    store = DocumentStore()
    store.add_document("a.txt", _chunks("a.txt", 1))
    obtenidos = store.chunks
    obtenidos.clear()
    # Modificar la copia no debe vaciar el almacén.
    assert store.chunk_count() == 1


def test_clear_vacia_el_almacen():
    store = DocumentStore()
    store.add_document("a.txt", _chunks("a.txt", 3))
    store.clear()
    assert store.is_empty()
    assert store.documents == {}
