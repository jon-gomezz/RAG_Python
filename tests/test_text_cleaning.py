"""Tests para la normalización de texto."""

from __future__ import annotations

from app.services.text_cleaning import normalize_text


def test_unifica_finales_de_linea():
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_colapsa_espacios_repetidos():
    assert normalize_text("hola     mundo") == "hola mundo"


def test_elimina_espacios_al_final_de_linea():
    assert normalize_text("linea con espacios   \nsiguiente") == "linea con espacios\nsiguiente"


def test_reduce_saltos_de_linea_multiples():
    assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"


def test_conserva_un_salto_y_un_doble_salto():
    assert normalize_text("a\nb") == "a\nb"
    assert normalize_text("a\n\nb") == "a\n\nb"


def test_recorta_extremos():
    assert normalize_text("   \n  hola  \n  ") == "hola"


def test_texto_vacio():
    assert normalize_text("   \n\t  ") == ""
