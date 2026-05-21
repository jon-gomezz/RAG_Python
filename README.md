# RAG — Preguntas y Respuestas sobre Documentos

Un servicio **FastAPI** que responde preguntas usando **únicamente** el contenido de los
documentos `.txt`, `.md` y `.pdf` que se suban. La recuperación usa **TF-IDF + similitud
del coseno** por defecto, con un **modo semántico opcional basado en embeddings**
(`RETRIEVAL_MODE=embeddings`); las respuestas se generan solo a partir del contexto
recuperado, devolviendo siempre los fragmentos (fuentes) utilizados.

## Características

- Subida de uno o varios ficheros `.txt` / `.md` / `.pdf`.
- Extracción de texto con **normalización** (espacios y saltos de línea) y
  fragmentación (chunking) configurable con solapamiento.
- **Deduplicado de fragmentos** repetidos (cabeceras, índices) antes de indexar.
- Recuperación por TF-IDF + similitud del coseno (`top_k`, puntuación mínima de
  relevancia), con **modo semántico opcional** (embeddings).
- Generación de respuestas fundamentadas; respuesta clara de "información insuficiente"
  cuando la recuperación es demasiado débil (sin llamar al LLM).
- Fragmentos fuente devueltos con cada respuesta (**trazabilidad**).
- Gestión de documentos: listar y borrar lo indexado.
- Manejo de errores centralizado con códigos HTTP claros.

## Endpoints

| Método   | Ruta                 | Descripción                                     |
| -------- | -------------------- | ----------------------------------------------- |
| `GET`    | `/health`            | Comprobación de salud del servicio.             |
| `POST`   | `/documents/upload`  | Sube uno o varios documentos y los indexa.      |
| `GET`    | `/documents`         | Lista los documentos indexados y sus fragmentos. |
| `DELETE` | `/documents`         | Borra todos los documentos indexados.           |
| `POST`   | `/ask`               | Responde una pregunta con el contenido subido.  |

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
| `RETRIEVAL_MODE`      | `tfidf`        | Modo de recuperación: `tfidf` o `embeddings`    |
| `EMBEDDING_MODEL`     | `text-embedding-3-small` | Modelo de embeddings (modo `embeddings`) |
| `LLM_API_KEY`         | _(vacío)_      | Clave de API del proveedor LLM/embeddings (OpenAI) |
| `LLM_MODEL`           | `gpt-4o-mini`  | Modelo usado para la generación de respuestas   |

### Modos de recuperación

- **`tfidf`** (por defecto): comparación léxica (palabras), determinista, gratis
  y sin dependencias externas. Ideal para tests y para el alcance del reto.
- **`embeddings`**: comparación **semántica** (significado) con embeddings de
  OpenAI. Entiende sinónimos y cruza idiomas, a cambio de coste, latencia y
  necesitar `LLM_API_KEY`. Se activa con `RETRIEVAL_MODE=embeddings`.

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

## Desarrollo y calidad de código

Para trabajar en el proyecto, instala las dependencias de desarrollo (incluye
tests y herramientas de calidad):

```bash
pip install -r requirements-dev.txt
pre-commit install        # ejecuta linter/formato antes de cada commit
```

Herramientas:

```bash
ruff check .              # linter
ruff format .             # formateo automático
mypy app                  # comprobación de tipos
```

La **Integración Continua** (GitHub Actions, `.github/workflows/ci.yml`) ejecuta
linter, formato, tipos y tests en cada push y pull request.

## Decisiones técnicas

- **FastAPI**: framework moderno, tipado y con documentación automática (`/docs`).
- **TF-IDF + similitud del coseno** como recuperación por defecto: es
  **determinista**, no requiere servicios externos para buscar y es **fácil de
  testear**. Se complementa con un **modo semántico opcional con embeddings**
  (`RETRIEVAL_MODE=embeddings`), intercambiable gracias a una interfaz común
  `Retriever`, para casos que requieren entender significados o cruzar idiomas.
- **Fragmentación por caracteres con solapamiento**: simple y predecible; el
  solapamiento reduce el riesgo de partir una idea entre dos fragmentos. El texto
  se **normaliza** (espacios y saltos sobrantes) y los fragmentos **duplicados** se
  eliminan antes de indexar, para no recuperar contenido repetido.
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
  recuperar nada. Se **mitiga** activando el modo semántico
  (`RETRIEVAL_MODE=embeddings`), disponible en el proyecto.
- **El chunking por caracteres** puede partir tablas o estructuras; un troceado
  consciente de la estructura del documento mejoraría la calidad.
- **Las páginas tipo índice** (que repiten muchas palabras clave) pueden puntuar
  alto sin aportar contenido; el deduplicado elimina fragmentos idénticos, pero un
  filtrado o reordenado (re-ranking) más fino sería una mejora futura.
- **Almacenamiento en memoria**: los documentos se pierden al reiniciar. Mejora
  futura: persistencia (base de datos o índice en disco).

## Estructura del proyecto

```
app/
  core/          configuración, excepciones y manejadores de error
  services/      loaders, limpieza de texto, chunker, recuperadores (TF-IDF y
                 embeddings), clientes LLM/embeddings, generador de respuestas,
                 tubería RAG
  store/         almacén en memoria de documentos/índice
  main.py        aplicación FastAPI y endpoints
  schemas.py     esquemas Pydantic de petición/respuesta
  dependencies.py inyección de dependencias (tubería y recuperador)
tests/           tests unitarios y de API
examples/        ficheros de ejemplo para pruebas manuales
```

## Docker (opcional)

```bash
docker build -t rag-service .
docker run --rm -p 8000:8000 --env-file .env rag-service
```
