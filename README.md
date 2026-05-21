# RAG — Preguntas y Respuestas sobre Documentos

Un servicio **FastAPI** que responde preguntas usando **únicamente** el contenido de los
documentos `.txt`, `.md` y `.pdf` que se suban. La recuperación usa **TF-IDF + similitud
del coseno**, y las respuestas se generan solo a partir del contexto recuperado,
devolviendo siempre los fragmentos (fuentes) utilizados.

## Características

- Subida de uno o varios ficheros `.txt` / `.md` / `.pdf`.
- Extracción de texto y fragmentación (chunking) configurable con solapamiento.
- Recuperación por TF-IDF + similitud del coseno (`top_k`, puntuación mínima de relevancia).
- Generación de respuestas fundamentadas; respuesta clara de "información insuficiente"
  cuando la recuperación es demasiado débil (sin llamar al LLM).
- Fragmentos fuente devueltos con cada respuesta (**trazabilidad**).
- Manejo de errores centralizado con códigos HTTP claros.

## Endpoints

| Método | Ruta                 | Descripción                                  |
| ------ | -------------------- | -------------------------------------------- |
| `GET`  | `/health`            | Comprobación de salud del servicio.          |
| `POST` | `/documents/upload`  | Sube uno o varios documentos y los indexa.   |
| `POST` | `/ask`               | Responde una pregunta con el contenido subido. |

Documentación interactiva (Swagger UI) disponible en `/docs` al arrancar el servicio.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env         # luego rellena LLM_API_KEY con tu clave de OpenAI
```

> La clave (`LLM_API_KEY`) es personal de quien ejecuta el servicio y **no se incluye
> en el repositorio**. Cópiala en tu `.env` local (ignorado por Git).

## Ejecución

```bash
uvicorn app.main:app --reload
```

El servicio quedará en `http://127.0.0.1:8000`. Abre `http://127.0.0.1:8000/docs`
para probarlo desde el navegador.

## Configuración

Todos los ajustes se leen desde variables de entorno (ver `.env.example`):

| Variable              | Por defecto    | Descripción                                     |
| --------------------- | -------------- | ----------------------------------------------- |
| `CHUNK_SIZE`          | `800`          | Máximo de caracteres por fragmento              |
| `CHUNK_OVERLAP`       | `150`          | Solapamiento (caracteres) entre fragmentos      |
| `TOP_K`               | `4`            | Número de fragmentos recuperados por pregunta   |
| `MIN_RELEVANCE_SCORE` | `0.1`          | Puntuación mínima del coseno para un fragmento  |
| `LLM_API_KEY`         | _(vacío)_      | Clave de API del proveedor LLM (OpenAI)         |
| `LLM_MODEL`           | `gpt-4o-mini`  | Modelo usado para la generación de respuestas   |

## Ejemplos de uso

Hay ficheros de ejemplo en `examples/`. Con el servicio en marcha:

### 1. Comprobar salud

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

### 2. Subir uno o varios documentos

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "files=@examples/sistema_solar.txt" \
  -F "files=@examples/lenguajes_programacion.md"
```

Respuesta:

```json
{
  "documents": [
    {"filename": "sistema_solar.txt", "chunks_created": 3},
    {"filename": "lenguajes_programacion.md", "chunks_created": 2}
  ],
  "total_chunks": 5
}
```

### 3. Preguntar

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuántos planetas tiene el sistema solar?"}'
```

Respuesta (con fuentes para trazabilidad):

```json
{
  "answer": "El sistema solar tiene ocho planetas.",
  "status": "answered",
  "sources": [
    {"text": "El sistema solar está formado por ocho planetas...", "source": "sistema_solar.txt", "index": 0, "score": 0.41}
  ]
}
```

### Estados posibles de la respuesta

- `answered`: se encontró contexto y se generó una respuesta.
- `insufficient_context`: hay documentos, pero ninguno es relevante para la pregunta.
- `no_documents`: aún no se ha subido ningún documento.

## Tests

```bash
pytest
```

Cubren loaders, chunker, retriever, almacén, generador de respuestas (con LLM
simulado), la tubería RAG completa, la API y los casos de error.

## Decisiones técnicas

- **FastAPI**: framework moderno, tipado y con documentación automática (`/docs`).
- **TF-IDF + similitud del coseno** para la recuperación (en lugar de embeddings):
  es **determinista**, no requiere servicios externos para buscar y es **fácil de
  testear**, suficiente para el alcance del reto.
- **Fragmentación por caracteres con solapamiento**: simple y predecible; el
  solapamiento reduce el riesgo de partir una idea entre dos fragmentos.
- **Detección de contexto insuficiente antes de llamar al LLM**: si la recuperación
  no devuelve fragmentos, se responde sin gastar una llamada al modelo, evitando
  además respuestas inventadas.
- **Trazabilidad**: cada respuesta incluye los fragmentos fuente con su puntuación.
- **Arquitectura por capas**: lógica de negocio en `app/services`, almacén en
  `app/store`, configuración y errores en `app/core`; los endpoints (`app/main.py`)
  son delgados. Las excepciones de dominio se traducen a HTTP en un único lugar.
- **Cliente LLM inyectable**: permite testear sin llamadas reales (mock) y cambiar
  de proveedor con facilidad.
- **Configuración por variables de entorno** con valores por defecto y validación.

## Limitaciones y mejoras futuras

- **TF-IDF es léxico, no semántico**: compara palabras literales, así que una
  pregunta en un idioma distinto al del documento (o con sinónimos) puede no
  recuperar nada. Mejora futura: **búsqueda semántica con embeddings**.
- **El chunking por caracteres** puede partir tablas o estructuras; un troceado
  consciente de la estructura del documento mejoraría la calidad.
- **Las páginas tipo índice** (que repiten muchas palabras clave) pueden puntuar
  alto sin aportar contenido; se podría filtrar o reordenar (re-ranking).
- **Almacenamiento en memoria**: los documentos se pierden al reiniciar. Mejora
  futura: persistencia (base de datos o índice en disco).

## Estructura del proyecto

```
app/
  core/      configuración, excepciones y manejadores de error
  services/  loaders, chunker, retriever, generador de respuestas, cliente LLM, tubería RAG
  store/     almacén en memoria de documentos/índice
  main.py    aplicación FastAPI y endpoints
  schemas.py esquemas Pydantic de petición/respuesta
tests/       tests unitarios y de API
examples/    ficheros de ejemplo para pruebas manuales
```

## Docker (opcional)

```bash
docker build -t rag-service .
docker run --rm -p 8000:8000 --env-file .env rag-service
```
