"""Tests para la carga y extracción de texto de documentos."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from app.core.exceptions import (
    EmptyFileError,
    UnsupportedFileTypeError,
)
from app.services.loaders import (
    SUPPORTED_EXTENSIONS,
    get_extension,
    load_document,
)


def _make_pdf(text: str) -> bytes:
    """Crea un PDF mínimo en memoria con una página de texto."""
    writer = PdfWriter()
    # Página en blanco: no aporta texto (sirve para el caso de PDF vacío).
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# --- get_extension ---------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("doc.txt", ".txt"),
        ("notas.MD", ".md"),
        ("informe.PDF", ".pdf"),
        ("ARCHIVO.TxT", ".txt"),
    ],
)
def test_get_extension_es_insensible_a_mayusculas(filename, expected):
    assert get_extension(filename) == expected


def test_extensiones_soportadas():
    assert frozenset({".txt", ".md", ".pdf"}) == SUPPORTED_EXTENSIONS


# --- ficheros de texto (.txt / .md) ----------------------------------------


def test_carga_txt():
    content = "Hola mundo\nSegunda línea".encode()
    assert load_document("saludo.txt", content) == "Hola mundo\nSegunda línea"


def test_carga_md():
    content = "# Título\n\nContenido en markdown".encode()
    assert load_document("notas.md", content) == "# Título\n\nContenido en markdown"


def test_carga_txt_recorta_espacios():
    content = b"   \n  texto rodeado de espacios  \n  "
    assert load_document("doc.txt", content) == "texto rodeado de espacios"


# --- validación de extensión -----------------------------------------------


@pytest.mark.parametrize("filename", ["imagen.png", "hoja.csv", "sin_extension"])
def test_extension_no_soportada_lanza_error(filename):
    with pytest.raises(UnsupportedFileTypeError):
        load_document(filename, b"contenido")


# --- ficheros vacíos --------------------------------------------------------


def test_txt_vacio_lanza_error():
    with pytest.raises(EmptyFileError):
        load_document("vacio.txt", b"")


def test_txt_solo_espacios_lanza_error():
    with pytest.raises(EmptyFileError):
        load_document("blanco.txt", b"   \n\t  \n")


# --- PDF --------------------------------------------------------------------


def test_pdf_sin_texto_lanza_empty():
    """Un PDF con páginas en blanco no aporta texto -> EmptyFileError."""
    pdf_bytes = _make_pdf("")
    with pytest.raises(EmptyFileError):
        load_document("vacio.pdf", pdf_bytes)
