"""Tests para la fragmentación (chunking) de texto."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.services.chunker import Chunk, chunk_text, deduplicate_chunks

# --- casos básicos ---------------------------------------------------------


def test_texto_mas_corto_que_chunk_size_devuelve_un_solo_chunk():
    chunks = chunk_text("hola mundo", chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hola mundo"
    assert chunks[0].index == 0
    assert chunks[0].start == 0
    assert chunks[0].end == len("hola mundo")


def test_texto_vacio_devuelve_lista_vacia():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_texto_solo_espacios_devuelve_lista_vacia():
    assert chunk_text("   \n\t  ", chunk_size=100, chunk_overlap=10) == []


# --- troceado y solapamiento ----------------------------------------------


def test_division_en_varios_chunks_sin_solapamiento():
    # "abcdefghij" (10 chars), tamaño 5, sin solape -> 2 chunks exactos
    chunks = chunk_text("abcdefghij", chunk_size=5, chunk_overlap=0)
    assert [c.text for c in chunks] == ["abcde", "fghij"]
    assert [c.index for c in chunks] == [0, 1]
    assert [(c.start, c.end) for c in chunks] == [(0, 5), (5, 10)]


def test_solapamiento_comparte_caracteres_entre_chunks():
    # tamaño 5, solape 2 -> avance de 3 caracteres por chunk
    chunks = chunk_text("abcdefghij", chunk_size=5, chunk_overlap=2)
    assert chunks[0].text == "abcde"
    assert chunks[1].text == "defgh"  # comparte "de" con el anterior
    assert chunks[1].start == 3
    # el final de un chunk reaparece al inicio del siguiente
    assert chunks[0].text[-2:] == chunks[1].text[:2]


def test_los_indices_son_consecutivos():
    chunks = chunk_text("a" * 100, chunk_size=10, chunk_overlap=2)
    indices = [c.index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_reconstruccion_cubre_todo_el_texto():
    texto = "El zorro marrón salta sobre el perro perezoso una y otra vez."
    chunks = chunk_text(texto, chunk_size=10, chunk_overlap=3)
    # el último chunk debe terminar exactamente al final del texto
    assert chunks[-1].end == len(texto)
    # las posiciones de cada chunk se corresponden con su contenido
    for c in chunks:
        assert texto[c.start : c.end] == c.text


# --- metadatos -------------------------------------------------------------


def test_se_propaga_el_origen_a_cada_chunk():
    chunks = chunk_text("abcdefghij", chunk_size=4, chunk_overlap=1, source="doc.txt")
    assert all(c.source == "doc.txt" for c in chunks)


def test_chunk_es_inmutable():
    chunk = Chunk(text="x", index=0, start=0, end=1)
    with pytest.raises(FrozenInstanceError):
        chunk.text = "y"  # type: ignore[misc]


# --- validación de parámetros ----------------------------------------------


@pytest.mark.parametrize("size", [0, -1, -10])
def test_chunk_size_no_positivo_lanza_error(size):
    with pytest.raises(ValueError):
        chunk_text("texto", chunk_size=size, chunk_overlap=0)


def test_overlap_negativo_lanza_error():
    with pytest.raises(ValueError):
        chunk_text("texto", chunk_size=10, chunk_overlap=-1)


@pytest.mark.parametrize("overlap", [10, 15])
def test_overlap_mayor_o_igual_que_size_lanza_error(overlap):
    with pytest.raises(ValueError):
        chunk_text("texto", chunk_size=10, chunk_overlap=overlap)


# --- deduplicado -----------------------------------------------------------


def test_deduplicate_elimina_fragmentos_identicos():
    chunks = [
        Chunk(text="repetido", index=0, start=0, end=8, source="d"),
        Chunk(text="unico", index=1, start=8, end=13, source="d"),
        Chunk(text="repetido", index=2, start=13, end=21, source="d"),
    ]
    unicos = deduplicate_chunks(chunks)
    assert [c.text for c in unicos] == ["repetido", "unico"]


def test_deduplicate_conserva_el_primero_y_su_orden():
    chunks = [
        Chunk(text="a", index=0, start=0, end=1, source="d"),
        Chunk(text="b", index=1, start=1, end=2, source="d"),
        Chunk(text="a", index=2, start=2, end=3, source="d"),
        Chunk(text="c", index=3, start=3, end=4, source="d"),
    ]
    unicos = deduplicate_chunks(chunks)
    assert [c.text for c in unicos] == ["a", "b", "c"]
    assert unicos[0].index == 0  # se conserva el primero


def test_deduplicate_sin_duplicados_no_cambia():
    chunks = [
        Chunk(text="x", index=0, start=0, end=1, source="d"),
        Chunk(text="y", index=1, start=1, end=2, source="d"),
    ]
    assert deduplicate_chunks(chunks) == chunks


def test_deduplicate_lista_vacia():
    assert deduplicate_chunks([]) == []
