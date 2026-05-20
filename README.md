# RAG — Preguntas y Respuestas sobre Documentos

Un servicio FastAPI que responde preguntas usando **únicamente** el contenido de los
documentos `.txt`, `.md` y `.pdf` que se suban. La recuperación usa TF-IDF + similitud
del coseno, y las respuestas se generan solo a partir del contexto recuperado,
devolviendo siempre los fragmentos (fuentes) utilizados.

> Estado: en desarrollo. Ver las fases de implementación más abajo.

## Características

- Subida de uno o varios ficheros `.txt` / `.md` / `.pdf`.
- Fragmentación (chunking) configurable con solapamiento.
- Recuperación por TF-IDF + similitud del coseno (`top_k`, puntuación mínima de relevancia).
- Generación de respuestas fundamentadas; respuesta clara de "información insuficiente" cuando la recuperación es demasiado débil.
- Fragmentos fuente devueltos con cada respuesta (trazabilidad).

## Endpoints

- `GET /health`
- `POST /documents/upload`
- `POST /ask`

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # luego rellena LLM_API_KEY
```

## Ejecución

```bash
uvicorn app.main:app --reload
```

## Configuración

Todos los ajustes se leen desde variables de entorno (ver `.env.example`):

| Variable              | Descripción                                     |
| --------------------- | ----------------------------------------------- |
| `CHUNK_SIZE`          | Máximo de caracteres por fragmento              |
| `CHUNK_OVERLAP`       | Solapamiento (caracteres) entre fragmentos      |
| `TOP_K`               | Número de fragmentos recuperados por pregunta   |
| `MIN_RELEVANCE_SCORE` | Puntuación mínima del coseno para un fragmento  |
| `LLM_API_KEY`         | Clave de API del proveedor LLM                  |
| `LLM_MODEL`           | Modelo usado para la generación de respuestas   |

## Tests

```bash
pytest
```

## Estructura del proyecto

```
app/
  core/      configuración y excepciones compartidas
  services/  loaders, chunker, retriever, generador de respuestas, orquestación RAG
  store/     almacén en memoria de documentos/índice
tests/       tests unitarios y de API
examples/    ficheros de ejemplo para pruebas manuales
```
