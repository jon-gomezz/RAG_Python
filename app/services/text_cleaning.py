"""Utilidades de limpieza y normalización de texto.

La extracción de texto (sobre todo de PDFs) suele dejar espacios, tabulaciones y
saltos de línea sobrantes. Normalizarlo produce fragmentos más limpios, fuentes
más legibles y una recuperación algo más estable, sin alterar el contenido.
"""

from __future__ import annotations

import re

# Espacios/tabulaciones repetidos (2 o más) -> un solo espacio.
_SPACES = re.compile(r"[ \t]{2,}")
# Espacios o tabulaciones al final de una línea (antes de un salto).
_TRAILING = re.compile(r"[ \t]+(?=\n)")
# Tres o más saltos de línea seguidos -> dos (separación de párrafo).
_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normaliza espacios y saltos de línea sin tocar el contenido.

    - Unifica los finales de línea a ``\\n``.
    - Colapsa secuencias de espacios/tabulaciones en uno solo.
    - Elimina espacios al final de cada línea.
    - Reduce 3+ saltos de línea consecutivos a 2.
    - Recorta espacios al principio y al final del texto.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = _TRAILING.sub("", text)
    text = _NEWLINES.sub("\n\n", text)
    return text.strip()
